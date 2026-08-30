#!/usr/bin/env python3
"""Manage the static TUF repository for Nitro EIF attestations.

Production root signing is an offline operator action. GitHub Actions only uses
one threshold-1 Ed25519 key shared by targets, snapshot, and timestamp.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from cryptography.hazmat.primitives import serialization
from securesystemslib.signer import CryptoSigner
from tuf.api.metadata import (
    Metadata,
    MetaFile,
    Root,
    Snapshot,
    TargetFile,
    Targets,
    Timestamp,
)

CHANNEL_SCHEMA = "https://attestations.trymaple.ai/schemas/channel/v1"
MANIFEST_SCHEMA = "https://attestations.trymaple.ai/schemas/nitro-eif-release/v1"
BUILDER_POLICY_SCHEMA = (
    "https://attestations.trymaple.ai/schemas/sigstore-builder-policy/v1"
)
COMPONENT = "opensecret-backend"
ENVIRONMENTS = {"dev", "prod"}
ONLINE_ROLES = ("targets", "snapshot", "timestamp")
SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PCR_RE = re.compile(r"^[0-9a-f]{96}$")
TARGETS_EXPIRY_DAYS = 90
SNAPSHOT_EXPIRY_DAYS = 14
TIMESTAMP_EXPIRY_HOURS = 47
ROOT_EXPIRY_DAYS = 365
MAX_LIVE_FILE_BYTES = 16 * 1024 * 1024
MAX_SAFE_INTEGER = (1 << 53) - 1
MAX_ROOT_VERSIONS = 33
MAX_ROOT_BYTES = 64 * 1024
MAX_TIMESTAMP_BYTES = 32 * 1024
MAX_SNAPSHOT_BYTES = 128 * 1024
MAX_TARGETS_METADATA_BYTES = 256 * 1024
MAX_CHANNEL_BYTES = 128 * 1024
MAX_BUILDER_POLICY_BYTES = 128 * 1024
MAX_SIGSTORE_ROOT_BYTES = 512 * 1024
MAX_MANIFEST_BYTES = 128 * 1024
MAX_BUNDLE_BYTES = 2 * 1024 * 1024
MAX_CERTIFICATE_BYTES = 64 * 1024
MAX_PUBLIC_KEY_BYTES = 16 * 1024
MAX_SIGNATURE_BYTES = 16 * 1024
MAX_RFC3161_TIMESTAMP_BYTES = 256 * 1024
MAX_REKOR_BODY_BYTES = 1024 * 1024
MAX_CHECKPOINT_CHARS = 64 * 1024
MAX_TARGETS = 256
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
WORKFLOW_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}/[A-Za-z0-9_.-]{1,128}$")
TARGET_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


class RepositoryError(ValueError):
    """Raised when repository input or state is unsafe or inconsistent."""


class LiveFileNotFound(RepositoryError):
    """Raised when a public repository path is definitely absent."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, indent=2, separators=(",", ": "), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def strict_json_bytes(data: bytes, context: str) -> Any:
    def pairs(values: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise RepositoryError(f"{context} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise RepositoryError(f"{context} contains non-finite number {value}")

    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepositoryError(f"{context} is not valid UTF-8 JSON: {error}") from error


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_file(path: Path, context: str) -> bytes:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise RepositoryError(f"cannot inspect {context} at {path}: {error}") from error
    if not stat.S_ISREG(mode):
        raise RepositoryError(
            f"{context} at {path} must be a regular file, not a symlink or device"
        )
    try:
        data = path.read_bytes()
    except OSError as error:
        raise RepositoryError(f"cannot read {context} at {path}: {error}") from error
    if not data:
        raise RepositoryError(f"{context} at {path} is empty")
    return data


def validate_safe_tree(root: Path) -> None:
    if root.is_symlink():
        raise RepositoryError(f"repository state root must not be a symlink: {root}")
    if not root.exists():
        return
    if not root.is_dir():
        raise RepositoryError(f"repository state root must be a real directory: {root}")
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise RepositoryError(
                f"repository state contains a symlink or special file: {path}"
            )


def atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_signer(path: Path) -> CryptoSigner:
    data = require_file(path, "private key")
    try:
        private_key = serialization.load_pem_private_key(data, password=None)
        signer = CryptoSigner(private_key)
    except (TypeError, ValueError) as error:
        raise RepositoryError(
            f"{path} is not an unencrypted PKCS8 Ed25519 PEM key"
        ) from error
    if signer.public_key.keytype != "ed25519" or signer.public_key.scheme != "ed25519":
        raise RepositoryError("TUF keys must be Ed25519")
    return signer


def check_key_permissions(path: Path) -> None:
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise RepositoryError(
            f"private key permissions are too broad: {path} (require chmod 600)"
        )


def load_metadata(path: Path, expected_type: type[Any]) -> Metadata[Any]:
    try:
        metadata = Metadata.from_bytes(require_file(path, "TUF metadata"))
    except Exception as error:
        raise RepositoryError(f"invalid TUF metadata {path}: {error}") from error
    if not isinstance(metadata.signed, expected_type):
        raise RepositoryError(f"{path} has the wrong signed metadata type")
    return metadata


def validate_root_client_subset(
    metadata: Metadata[Root], data: bytes, context: str
) -> None:
    require_size(data, MAX_ROOT_BYTES, context)
    root = metadata.signed
    if root.version < 1 or root.version > MAX_SAFE_INTEGER:
        raise RepositoryError(f"{context} has a browser-unsafe version")
    if set(root.roles) != {"root", "targets", "snapshot", "timestamp"}:
        raise RepositoryError(f"{context} must contain only standard top-level roles")
    if not 1 <= len(root.keys) <= 32 or not 1 <= len(metadata.signatures) <= 32:
        raise RepositoryError(f"{context} exceeds client key or signature limits")
    for keyid, key in root.keys.items():
        public = key.keyval.get("public")
        if (
            not isinstance(keyid, str)
            or not 1 <= len(keyid) <= 128
            or key.keytype != "ed25519"
            or key.scheme != "ed25519"
            or not isinstance(public, str)
            or not re.fullmatch(r"[0-9a-f]{64}", public)
        ):
            raise RepositoryError(f"{context} contains a non-Ed25519 client key")
    for role_name, role in root.roles.items():
        if (
            not 1 <= len(role.keyids) <= 16
            or len(set(role.keyids)) != len(role.keyids)
            or role.threshold < 1
            or role.threshold > len(role.keyids)
            or role.threshold > 16
            or any(keyid not in root.keys for keyid in role.keyids)
        ):
            raise RepositoryError(f"{context} has an invalid {role_name} role")
        role_material = [
            normalized_public_key_material(root.keys[keyid])
            for keyid in role.keyids
        ]
        if len(set(role_material)) != len(role_material):
            raise RepositoryError(
                f"{context} reuses key material within the {role_name} role"
            )

    root_material = {
        normalized_public_key_material(root.keys[keyid])
        for keyid in root.roles["root"].keyids
    }
    for role_name in ONLINE_ROLES:
        online_material = {
            normalized_public_key_material(root.keys[keyid])
            for keyid in root.roles[role_name].keyids
        }
        if root_material & online_material:
            raise RepositoryError(
                f"{context} shares root-role key material with online {role_name}"
            )


def normalized_public_key_material(key: Any) -> bytes:
    """Return identity-bearing key bytes independent of a caller-supplied key ID."""

    public = key.keyval.get("public")
    if not isinstance(public, str):
        raise RepositoryError("TUF key lacks normalized Ed25519 public material")
    try:
        return bytes.fromhex(public)
    except ValueError as error:
        raise RepositoryError("TUF key contains invalid Ed25519 public material") from error


def role_public_key_material(root: Root, role_name: str) -> set[bytes]:
    return {
        normalized_public_key_material(root.keys[keyid])
        for keyid in root.roles[role_name].keyids
    }


def target_size_limit(logical: str) -> int:
    if logical.startswith("channels/"):
        return MAX_CHANNEL_BYTES
    if logical == "sigstore/trusted_root.json":
        return MAX_SIGSTORE_ROOT_BYTES
    if logical.endswith("/manifest.sigstore.json"):
        return MAX_BUNDLE_BYTES
    if logical.endswith("/manifest.json"):
        return MAX_MANIFEST_BYTES
    return MAX_BUNDLE_BYTES


def validate_targets_client_subset(targets: Targets) -> None:
    if targets.version < 1 or targets.version > MAX_SAFE_INTEGER:
        raise RepositoryError("targets metadata has a browser-unsafe version")
    if targets.delegations is not None:
        raise RepositoryError(
            "delegated targets are outside the supported client subset"
        )
    if not 1 <= len(targets.targets) <= MAX_TARGETS:
        raise RepositoryError(
            f"targets metadata must contain between one and {MAX_TARGETS} files"
        )
    for logical, descriptor in targets.targets.items():
        validate_target_path(logical)
        if (
            set(descriptor.hashes) != {"sha256"}
            or descriptor.length < 0
            or descriptor.length > target_size_limit(logical)
        ):
            raise RepositoryError(
                f"target descriptor exceeds the supported client subset: {logical}"
            )


def load_root_chain(repository: Path) -> Metadata[Root]:
    validate_safe_tree(repository)
    roots: list[tuple[int, Path]] = []
    for path in (repository / "metadata").glob("*.root.json"):
        try:
            roots.append((int(path.name.split(".", 1)[0]), path))
        except ValueError:
            continue
    if not roots:
        raise RepositoryError(
            "TUF root is unconfigured; bootstrap it offline before CI promotion"
        )
    roots.sort()
    if [version for version, _ in roots] != list(range(1, roots[-1][0] + 1)):
        raise RepositoryError("root metadata versions must be contiguous starting at 1")
    if len(roots) > MAX_ROOT_VERSIONS:
        raise RepositoryError(
            f"root metadata history exceeds the {MAX_ROOT_VERSIONS}-version client limit"
        )

    previous: Metadata[Root] | None = None
    historical_root_material: set[bytes] = set()
    historical_online_material: set[bytes] = set()
    for version, path in roots:
        current = load_metadata(path, Root)
        validate_root_client_subset(
            current, require_file(path, "TUF root metadata"), str(path)
        )
        if current.signed.version != version:
            raise RepositoryError(f"{path} filename does not match its signed version")
        if not current.signed.consistent_snapshot:
            raise RepositoryError(f"{path} disables required consistent snapshots")
        if previous is not None:
            previous.signed.verify_delegate(
                "root", current.signed_bytes, current.signatures
            )
        current.signed.verify_delegate("root", current.signed_bytes, current.signatures)

        current_root_material = role_public_key_material(current.signed, "root")
        current_online_material = set().union(
            *(role_public_key_material(current.signed, role) for role in ONLINE_ROLES)
        )
        if current_root_material & historical_online_material:
            raise RepositoryError(
                f"{path} reassigns historical online key material to the root role"
            )
        if current_online_material & historical_root_material:
            raise RepositoryError(
                f"{path} reassigns historical root key material to an online role"
            )
        historical_root_material.update(current_root_material)
        historical_online_material.update(current_online_material)
        previous = current
    assert previous is not None
    if previous.signed.expires <= utcnow():
        raise RepositoryError("current root metadata is expired")
    return previous


def require_online_key(root: Root, signer: CryptoSigner) -> None:
    for role_name in ONLINE_ROLES:
        role = root.roles[role_name]
        if role.threshold != 1 or set(role.keyids) != {signer.public_key.keyid}:
            raise RepositoryError(
                f"online key is not the sole threshold-1 key for {role_name}"
            )
    if signer.public_key.keyid in root.roles["root"].keyids:
        raise RepositoryError("the online key must not also be an offline root key")


def verify_top_level(root: Root, role_name: str, metadata: Metadata[Any]) -> None:
    try:
        root.verify_delegate(role_name, metadata.signed_bytes, metadata.signatures)
    except Exception as error:
        raise RepositoryError(
            f"{role_name} metadata signature verification failed: {error}"
        ) from error


def load_current(
    repository: Path, root: Root
) -> tuple[Metadata[Targets], Metadata[Snapshot], Metadata[Timestamp]]:
    metadata_dir = repository / "metadata"
    timestamp_path = metadata_dir / "timestamp.json"
    timestamp_bytes = require_file(timestamp_path, "timestamp metadata")
    require_size(timestamp_bytes, MAX_TIMESTAMP_BYTES, "timestamp metadata")
    timestamp = load_metadata(timestamp_path, Timestamp)
    if (
        not 1 <= timestamp.signed.version <= MAX_SAFE_INTEGER
        or not 1 <= len(timestamp.signatures) <= 32
    ):
        raise RepositoryError("timestamp metadata exceeds the supported client subset")
    if (
        set(timestamp.signed.snapshot_meta.hashes) != {"sha256"}
        or timestamp.signed.snapshot_meta.length is None
        or timestamp.signed.snapshot_meta.length > MAX_SNAPSHOT_BYTES
    ):
        raise RepositoryError("timestamp snapshot descriptor exceeds client limits")
    snapshot_path = (
        metadata_dir / f"{timestamp.signed.snapshot_meta.version}.snapshot.json"
    )
    snapshot_bytes = require_file(snapshot_path, "snapshot metadata")
    require_size(snapshot_bytes, MAX_SNAPSHOT_BYTES, "snapshot metadata")
    snapshot = load_metadata(snapshot_path, Snapshot)
    if (
        not 1 <= snapshot.signed.version <= MAX_SAFE_INTEGER
        or not 1 <= len(snapshot.signatures) <= 32
    ):
        raise RepositoryError("snapshot metadata exceeds the supported client subset")
    if set(snapshot.signed.meta) != {"targets.json"}:
        raise RepositoryError("snapshot metadata must describe only top-level targets")
    targets_meta = snapshot.signed.meta.get("targets.json")
    if targets_meta is None:
        raise RepositoryError("snapshot metadata does not reference targets.json")
    if (
        set(targets_meta.hashes) != {"sha256"}
        or targets_meta.length is None
        or targets_meta.length > MAX_TARGETS_METADATA_BYTES
    ):
        raise RepositoryError("snapshot targets descriptor exceeds client limits")
    targets_path = metadata_dir / f"{targets_meta.version}.targets.json"
    targets_bytes = require_file(targets_path, "targets metadata")
    require_size(targets_bytes, MAX_TARGETS_METADATA_BYTES, "targets metadata")
    targets = load_metadata(targets_path, Targets)
    if not 1 <= len(targets.signatures) <= 32:
        raise RepositoryError("targets metadata exceeds the supported client subset")
    validate_targets_client_subset(targets.signed)
    verify_top_level(root, "targets", targets)
    verify_top_level(root, "snapshot", snapshot)
    verify_top_level(root, "timestamp", timestamp)
    if (
        MetaFile.from_data(targets.signed.version, targets.to_bytes(), ["sha256"])
        != targets_meta
    ):
        raise RepositoryError("targets bytes do not match snapshot metadata")
    if (
        MetaFile.from_data(snapshot.signed.version, snapshot.to_bytes(), ["sha256"])
        != timestamp.signed.snapshot_meta
    ):
        raise RepositoryError("snapshot bytes do not match timestamp metadata")
    return targets, snapshot, timestamp


def validate_target_path(path: str) -> None:
    candidate = Path(path)
    if (
        not path
        or len(path) > 1024
        or not TARGET_PATH_RE.fullmatch(path)
        or candidate.is_absolute()
        or path.endswith("/")
        or "//" in path
        or "%" in path
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or "\\" in path
    ):
        raise RepositoryError(f"unsafe TUF target path: {path!r}")


def validate_release_version(version: Any) -> str:
    if (
        not isinstance(version, str)
        or len(version) > 128
        or not SEMVER_RE.fullmatch(version)
    ):
        raise RepositoryError(
            "version must be MAJOR.MINOR.PATCH without v and fit one 128-byte path segment"
        )
    return version


def validate_source_path(path: Any) -> None:
    if path == ".":
        return
    if not isinstance(path, str) or len(path) > 512:
        raise RepositoryError("release manifest source path is invalid")
    validate_target_path(path)


def validate_https_url(value: Any, context: str) -> urllib.parse.SplitResult:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2048
        or value.strip() != value
    ):
        raise RepositoryError(f"{context} must be an exact HTTPS URL")
    try:
        parsed = urllib.parse.urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise RepositoryError(f"{context} must be an exact HTTPS URL") from error
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or hostname is None
        or "%" in parsed.netloc
        or not re.fullmatch(
            r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?[.])*"
            r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?[.]?",
            hostname,
        )
        or len(hostname) > 253
        or (port is not None and not 0 <= port <= 65535)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RepositoryError(f"{context} must be an exact HTTPS URL")
    return parsed


def require_exact_object(
    value: Any, required: set[str], optional: set[str], context: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or not required <= set(value) or set(value) - (
        required | optional
    ):
        raise RepositoryError(f"{context} has invalid fields")
    return value


def validate_base64(
    value: Any, maximum: int, context: str, exact: int | None = None
) -> bytes:
    if not isinstance(value, str) or not value:
        raise RepositoryError(f"{context} must be non-empty canonical base64")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RepositoryError(f"{context} is not valid base64") from error
    if not decoded or len(decoded) > maximum:
        raise RepositoryError(f"{context} exceeds its decoded byte limit")
    if exact is not None and len(decoded) != exact:
        raise RepositoryError(f"{context} must decode to exactly {exact} bytes")
    if base64.b64encode(decoded).decode("ascii") != value:
        raise RepositoryError(f"{context} is not canonically encoded base64")
    return decoded


def validate_decimal_integer(value: Any, context: str, positive: bool = False) -> int:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"0|[1-9][0-9]{0,15}", value)
        or int(value) > MAX_SAFE_INTEGER
        or (positive and value == "0")
    ):
        qualifier = "positive " if positive else ""
        raise RepositoryError(
            f"{context} must be a browser-safe {qualifier}decimal integer string"
        )
    return int(value)


def validate_datetime(value: Any, context: str) -> datetime:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 64
        or not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
            r"(?:[.][0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})",
            value,
        )
    ):
        raise RepositoryError(f"{context} must be a bounded RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RepositoryError(f"{context} is not an RFC3339 timestamp") from error
    if parsed.tzinfo is None:
        raise RepositoryError(f"{context} must include a timezone offset")
    return parsed


def javascript_milliseconds(value: datetime) -> int:
    """Return the millisecond precision used by JavaScript Date.parse()."""

    utc_value = value.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc_value - epoch
    return (
        (delta.days * 24 * 60 * 60 + delta.seconds) * 1000
        + delta.microseconds // 1000
    )


def require_size(data: bytes, maximum: int, context: str) -> None:
    if len(data) > maximum:
        raise RepositoryError(f"{context} exceeds the {maximum}-byte client limit")


def authenticated_target(
    repository: Path, targets: Targets, logical_path: str
) -> bytes:
    validate_target_path(logical_path)
    descriptor = targets.targets.get(logical_path)
    if descriptor is None:
        raise RepositoryError(
            f"required authenticated target is missing: {logical_path}"
        )
    data = require_file(repository / "targets" / logical_path, logical_path)
    if descriptor.length != len(data) or descriptor.hashes.get(
        "sha256"
    ) != sha256_bytes(data):
        raise RepositoryError(f"{logical_path} does not match signed targets metadata")
    return data


def validate_builder_policy(data: bytes) -> dict[str, Any]:
    """Validate repo-side promotion input, never a client-facing TUF target."""

    require_size(data, MAX_BUILDER_POLICY_BYTES, "builder policy")
    policy = strict_json_bytes(data, "builder policy")
    if not isinstance(policy, dict) or set(policy) != {"schema", "builders"}:
        raise RepositoryError("builder policy must contain exactly schema and builders")
    if policy["schema"] != BUILDER_POLICY_SCHEMA:
        raise RepositoryError("unsupported builder policy schema")
    builders = policy["builders"]
    if not isinstance(builders, dict) or not 1 <= len(builders) <= 32:
        raise RepositoryError("builder policy must contain between one and 32 builders")
    required = {
        "certificateIdentityRegexp",
        "certificateOidcIssuer",
        "workflowRepository",
        "workflowName",
        "workflowTrigger",
    }
    for builder_id, builder in builders.items():
        if (
            not isinstance(builder_id, str)
            or not IDENTIFIER_RE.fullmatch(builder_id)
            or not isinstance(builder, dict)
            or set(builder) != required
        ):
            raise RepositoryError(f"invalid builder policy entry {builder_id!r}")
        if any(
            not isinstance(builder[field], str) or not builder[field]
            for field in required
        ):
            raise RepositoryError(
                f"builder policy entry {builder_id!r} contains empty values"
            )
        identity = builder["certificateIdentityRegexp"]
        if (
            len(identity.encode("utf-8")) > 2048
            or not identity.startswith("^")
            or not identity.endswith("$")
        ):
            raise RepositoryError(
                f"builder {builder_id!r} identity regexp must be anchored"
            )
        validate_https_url(builder["certificateOidcIssuer"], "certificateOidcIssuer")
        if not WORKFLOW_REPOSITORY_RE.fullmatch(builder["workflowRepository"]):
            raise RepositoryError(
                f"builder {builder_id!r} workflowRepository is invalid"
            )
        if (
            len(builder["workflowName"]) > 512
            or len(builder["workflowTrigger"]) > 128
            or any(builder[field].strip() != builder[field] for field in required)
        ):
            raise RepositoryError(
                f"builder policy entry {builder_id!r} contains invalid strings"
            )
        try:
            re.compile(identity)
        except re.error as error:
            raise RepositoryError(
                f"builder {builder_id!r} has an invalid identity regexp"
            ) from error
    return policy


def validate_valid_for(value: Any, context: str) -> None:
    valid_for = require_exact_object(value, {"start"}, {"end"}, context)
    start = validate_datetime(valid_for["start"], f"{context}.start")
    if "end" in valid_for:
        end = validate_datetime(valid_for["end"], f"{context}.end")
        if javascript_milliseconds(end) <= javascript_milliseconds(start):
            raise RepositoryError(f"{context}.end must be after its start")


def validate_certificate(value: Any, context: str) -> None:
    certificate = require_exact_object(value, {"rawBytes"}, set(), context)
    validate_base64(
        certificate["rawBytes"], MAX_CERTIFICATE_BYTES, f"{context}.rawBytes"
    )


def validate_subject(value: Any, context: str) -> None:
    subject = require_exact_object(
        value, {"organization", "commonName"}, set(), context
    )
    for name in ("organization", "commonName"):
        field = subject[name]
        if not isinstance(field, str) or not field or len(field) > 512:
            raise RepositoryError(f"{context}.{name} must be a bounded string")


def validate_certificate_chain(value: Any, context: str) -> None:
    chain = require_exact_object(value, {"certificates"}, set(), context)
    certificates = chain["certificates"]
    if not isinstance(certificates, list) or not 1 <= len(certificates) <= 8:
        raise RepositoryError(f"{context}.certificates must contain between 1 and 8 entries")
    for index, certificate in enumerate(certificates):
        validate_certificate(certificate, f"{context}.certificates[{index}]")


def validate_log(value: Any, context: str) -> str:
    log = require_exact_object(
        value,
        {"baseUrl", "hashAlgorithm", "publicKey", "logId"},
        set(),
        context,
    )
    validate_https_url(log["baseUrl"], f"{context}.baseUrl")
    if log["hashAlgorithm"] != "SHA2_256":
        raise RepositoryError(f"{context}.hashAlgorithm must be SHA2_256")
    public_key = require_exact_object(
        log["publicKey"], {"rawBytes", "keyDetails", "validFor"}, set(), f"{context}.publicKey"
    )
    validate_base64(
        public_key["rawBytes"], MAX_PUBLIC_KEY_BYTES, f"{context}.publicKey.rawBytes"
    )
    if (
        not isinstance(public_key["keyDetails"], str)
        or not public_key["keyDetails"]
        or len(public_key["keyDetails"]) > 128
    ):
        raise RepositoryError(f"{context}.publicKey.keyDetails must be a bounded string")
    validate_valid_for(public_key["validFor"], f"{context}.publicKey.validFor")
    log_id = require_exact_object(log["logId"], {"keyId"}, set(), f"{context}.logId")
    validate_base64(log_id["keyId"], 32, f"{context}.logId.keyId", exact=32)
    return log_id["keyId"]


def validate_authority(value: Any, context: str) -> None:
    authority = require_exact_object(
        value, {"subject", "uri", "certChain", "validFor"}, set(), context
    )
    validate_subject(authority["subject"], f"{context}.subject")
    validate_https_url(authority["uri"], f"{context}.uri")
    validate_certificate_chain(authority["certChain"], f"{context}.certChain")
    validate_valid_for(authority["validFor"], f"{context}.validFor")


def validate_sigstore_root(data: bytes) -> dict[str, Any]:
    require_size(data, MAX_SIGSTORE_ROOT_BYTES, "Sigstore trusted root")
    root = strict_json_bytes(data, "Sigstore trusted root")
    expected = "application/vnd.dev.sigstore.trustedroot+json;version=0.1"
    root = require_exact_object(
        root,
        {
            "mediaType",
            "tlogs",
            "certificateAuthorities",
            "ctlogs",
            "timestampAuthorities",
        },
        set(),
        "Sigstore trusted root",
    )
    if root["mediaType"] != expected:
        raise RepositoryError("Sigstore trusted root has an unsupported mediaType")
    for name in ("tlogs", "certificateAuthorities", "ctlogs", "timestampAuthorities"):
        entries = root[name]
        if not isinstance(entries, list) or not 1 <= len(entries) <= 16:
            raise RepositoryError(
                f"Sigstore trusted root {name} must contain between 1 and 16 entries"
            )
    for index, log in enumerate(root["tlogs"]):
        validate_log(log, f"Sigstore trusted root.tlogs[{index}]")
    for index, authority in enumerate(root["certificateAuthorities"]):
        validate_authority(
            authority, f"Sigstore trusted root.certificateAuthorities[{index}]"
        )
    for index, log in enumerate(root["ctlogs"]):
        validate_log(log, f"Sigstore trusted root.ctlogs[{index}]")
    for index, authority in enumerate(root["timestampAuthorities"]):
        validate_authority(
            authority, f"Sigstore trusted root.timestampAuthorities[{index}]"
        )
    return root


def validate_manifest(data: bytes, environment: str, version: str) -> dict[str, Any]:
    validate_release_version(version)
    require_size(data, MAX_MANIFEST_BYTES, "release manifest")
    manifest = strict_json_bytes(data, "release manifest")
    if canonical_json(manifest) != data:
        raise RepositoryError("release manifest is not canonical JSON")
    if not isinstance(manifest, dict) or set(manifest) != {
        "artifact",
        "build",
        "component",
        "environment",
        "measurements",
        "release",
        "schema",
        "source",
    }:
        raise RepositoryError("release manifest must be an object")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("component") != COMPONENT
    ):
        raise RepositoryError("release manifest schema or component is invalid")
    if manifest.get("environment") != environment or manifest.get("release") != {
        "version": version
    }:
        raise RepositoryError(
            "release manifest environment or version does not match promotion"
        )
    validate_release_version(manifest["release"]["version"])
    source = manifest.get("source")
    if not isinstance(source, dict) or set(source) != {
        "uri",
        "path",
        "ref",
        "revision",
    }:
        raise RepositoryError("release manifest source is invalid")
    revision = source["revision"]
    if (
        not isinstance(revision, dict)
        or revision.get("algorithm") != "git-sha1"
        or not isinstance(revision.get("digest"), str)
        or not GIT_SHA1_RE.fullmatch(revision["digest"])
        or source["ref"] != f"refs/tags/v{version}"
    ):
        raise RepositoryError("release manifest source revision is invalid")
    validate_https_url(source["uri"], "release manifest source.uri")
    validate_source_path(source["path"])

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or set(artifact) != {
        "name",
        "mediaType",
        "size",
        "digests",
    }:
        raise RepositoryError("release manifest artifact is invalid")
    digests = artifact["digests"]
    if (
        artifact["mediaType"] != "application/vnd.aws.nitro.eif"
        or not isinstance(artifact["name"], str)
        or not artifact["name"]
        or len(artifact["name"]) > 512
        or artifact["name"] in {".", ".."}
        or any(character in artifact["name"] for character in ("/", "\\", "%"))
        or type(artifact["size"]) is not int
        or artifact["size"] <= 0
        or artifact["size"] > MAX_SAFE_INTEGER
        or not isinstance(digests, dict)
        or set(digests) != {"sha256"}
        or not isinstance(digests["sha256"], str)
        or not SHA256_RE.fullmatch(digests["sha256"])
    ):
        raise RepositoryError("release manifest artifact fields are invalid")

    measurements = manifest.get("measurements")
    if (
        not isinstance(measurements, dict)
        or set(measurements) != {"algorithm", "requiredPcrs", "pcrs"}
        or measurements["algorithm"] != "sha384"
        or measurements["requiredPcrs"] != [0, 1, 2]
        or any(type(value) is not int for value in measurements["requiredPcrs"])
        or not isinstance(measurements["pcrs"], dict)
        or set(measurements["pcrs"]) != {"0", "1", "2"}
        or any(
            not isinstance(value, str)
            or not PCR_RE.fullmatch(value)
            or set(value) == {"0"}
            for value in measurements["pcrs"].values()
        )
    ):
        raise RepositoryError("release manifest measurements are invalid")

    build = manifest.get("build")
    if (
        not isinstance(build, dict)
        or set(build)
        != {"system", "builderId", "derivation", "flakeLockSha256", "runUri"}
        or build["system"] != "nix"
        or build["derivation"] != f".#eif-{environment}"
        or not isinstance(build["builderId"], str)
        or not IDENTIFIER_RE.fullmatch(build["builderId"])
        or not isinstance(build["flakeLockSha256"], str)
        or not SHA256_RE.fullmatch(build["flakeLockSha256"])
    ):
        raise RepositoryError("release manifest build is invalid")
    validate_https_url(build["runUri"], "release manifest build.runUri")
    return manifest


def validate_inclusion_proof(value: Any, context: str) -> int:
    proof = require_exact_object(
        value,
        {"logIndex", "rootHash", "treeSize", "hashes", "checkpoint"},
        set(),
        context,
    )
    log_index = validate_decimal_integer(proof["logIndex"], f"{context}.logIndex")
    tree_size = validate_decimal_integer(
        proof["treeSize"], f"{context}.treeSize", positive=True
    )
    if log_index >= tree_size:
        raise RepositoryError(f"{context}.logIndex must be smaller than treeSize")
    validate_base64(proof["rootHash"], 32, f"{context}.rootHash", exact=32)
    hashes = proof["hashes"]
    if not isinstance(hashes, list) or len(hashes) > 64:
        raise RepositoryError(f"{context}.hashes must contain at most 64 entries")
    for index, digest in enumerate(hashes):
        validate_base64(digest, 32, f"{context}.hashes[{index}]", exact=32)
    checkpoint = require_exact_object(
        proof["checkpoint"], {"envelope"}, set(), f"{context}.checkpoint"
    )
    envelope = checkpoint["envelope"]
    if (
        not isinstance(envelope, str)
        or not envelope
        or len(envelope) > MAX_CHECKPOINT_CHARS
    ):
        raise RepositoryError(
            f"{context}.checkpoint.envelope must contain a bounded signed checkpoint"
        )
    return log_index


def validate_timestamp_verification_data(value: Any, context: str) -> None:
    timestamp_data = require_exact_object(
        value, {"rfc3161Timestamps"}, set(), context
    )
    timestamps = timestamp_data["rfc3161Timestamps"]
    if not isinstance(timestamps, list) or len(timestamps) != 1:
        raise RepositoryError(
            f"{context}.rfc3161Timestamps must contain exactly one entry"
        )
    for index, timestamp in enumerate(timestamps):
        timestamp = require_exact_object(
            timestamp, {"signedTimestamp"}, set(), f"{context}.rfc3161Timestamps[{index}]"
        )
        validate_base64(
            timestamp["signedTimestamp"],
            MAX_RFC3161_TIMESTAMP_BYTES,
            f"{context}.rfc3161Timestamps[{index}].signedTimestamp",
        )


def validate_bundle(data: bytes, trusted_root: dict[str, Any]) -> dict[str, Any]:
    require_size(data, MAX_BUNDLE_BYTES, "Sigstore bundle")
    bundle = strict_json_bytes(data, "Sigstore bundle")
    expected = "application/vnd.dev.sigstore.bundle.v0.3+json"
    bundle = require_exact_object(
        bundle,
        {"mediaType", "verificationMaterial", "messageSignature"},
        set(),
        "Sigstore bundle",
    )
    if bundle["mediaType"] != expected:
        raise RepositoryError("unsupported Sigstore bundle mediaType")
    message = require_exact_object(
        bundle["messageSignature"],
        {"messageDigest", "signature"},
        set(),
        "Sigstore bundle.messageSignature",
    )
    digest = require_exact_object(
        message["messageDigest"],
        {"algorithm", "digest"},
        set(),
        "Sigstore bundle.messageSignature.messageDigest",
    )
    if digest["algorithm"] != "SHA2_256":
        raise RepositoryError("Sigstore message digest algorithm must be SHA2_256")
    validate_base64(
        digest["digest"],
        32,
        "Sigstore bundle.messageSignature.messageDigest.digest",
        exact=32,
    )
    validate_base64(
        message["signature"],
        MAX_SIGNATURE_BYTES,
        "Sigstore bundle.messageSignature.signature",
    )

    material = require_exact_object(
        bundle["verificationMaterial"],
        {"certificate", "tlogEntries", "timestampVerificationData"},
        set(),
        "Sigstore bundle.verificationMaterial",
    )
    validate_certificate(
        material["certificate"], "Sigstore bundle.verificationMaterial.certificate"
    )
    validate_timestamp_verification_data(
        material["timestampVerificationData"],
        "Sigstore bundle.verificationMaterial.timestampVerificationData",
    )
    entries = material["tlogEntries"]
    if not isinstance(entries, list) or len(entries) != 1:
        raise RepositoryError(
            "Sigstore bundle must contain exactly one transparency-log entry"
        )
    entry = require_exact_object(
        entries[0],
        {
            "logIndex",
            "logId",
            "kindVersion",
            "inclusionProof",
            "canonicalizedBody",
        },
        {"integratedTime", "inclusionPromise"},
        "Sigstore bundle transparency-log entry",
    )
    log_index = validate_decimal_integer(
        entry["logIndex"], "Sigstore bundle transparency-log entry.logIndex"
    )
    log_id = require_exact_object(
        entry["logId"],
        {"keyId"},
        set(),
        "Sigstore bundle transparency-log entry.logId",
    )
    validate_base64(
        log_id["keyId"],
        32,
        "Sigstore bundle transparency-log entry.logId.keyId",
        exact=32,
    )
    kind_version = require_exact_object(
        entry["kindVersion"],
        {"kind", "version"},
        set(),
        "Sigstore bundle transparency-log entry.kindVersion",
    )
    if kind_version["kind"] != "hashedrekord" or kind_version["version"] not in {
        "0.0.1",
        "0.0.2",
    }:
        raise RepositoryError(
            "Sigstore transparency-log entry must be hashedrekord v0.0.1 or v0.0.2"
        )

    inclusion_promise = entry.get("inclusionPromise")
    if "inclusionPromise" in entry:
        inclusion_promise = require_exact_object(
            inclusion_promise,
            {"signedEntryTimestamp"},
            set(),
            "Sigstore bundle transparency-log entry.inclusionPromise",
        )
        validate_base64(
            inclusion_promise["signedEntryTimestamp"],
            MAX_SIGNATURE_BYTES,
            "Sigstore bundle transparency-log entry.inclusionPromise.signedEntryTimestamp",
        )
    if kind_version["version"] == "0.0.1":
        validate_decimal_integer(
            entry.get("integratedTime"),
            "Sigstore Rekor v1 integratedTime",
            positive=True,
        )
        if inclusion_promise is None:
            raise RepositoryError("Sigstore Rekor v1 entry requires an inclusion promise")
    elif "integratedTime" in entry:
        integrated_time = entry["integratedTime"]
        if integrated_time is not None and integrated_time != "0" and not (
            type(integrated_time) is int and integrated_time == 0
        ):
            raise RepositoryError(
                "Sigstore Rekor v2 integratedTime must be absent, null, string zero, or numeric zero"
            )

    proof_log_index = validate_inclusion_proof(
        entry["inclusionProof"],
        "Sigstore bundle transparency-log entry.inclusionProof",
    )
    if proof_log_index != log_index:
        raise RepositoryError(
            "Sigstore transparency-log entry and inclusion proof logIndex must match"
        )
    validate_base64(
        entry["canonicalizedBody"],
        MAX_REKOR_BODY_BYTES,
        "Sigstore bundle transparency-log entry.canonicalizedBody",
    )
    matching_logs = [
        log
        for log in trusted_root["tlogs"]
        if log["logId"]["keyId"] == log_id["keyId"]
    ]
    if len(matching_logs) != 1:
        raise RepositoryError(
            "Sigstore bundle must identify exactly one trusted-root transparency log"
        )
    return bundle


def validate_channel(channel: Any, environment: str) -> dict[str, Any]:
    if (
        not isinstance(channel, dict)
        or set(channel)
        != {
            "active",
            "environment",
            "schema",
            "sequence",
            "sigstoreTrustedRootTarget",
        }
        or channel["schema"] != CHANNEL_SCHEMA
        or channel["environment"] != environment
        or type(channel["sequence"]) is not int
        or channel["sequence"] < 1
    ):
        raise RepositoryError("current channel target is invalid")
    reference = channel["sigstoreTrustedRootTarget"]
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "sha256"}
        or not isinstance(reference["path"], str)
        or not isinstance(reference["sha256"], str)
        or not SHA256_RE.fullmatch(reference["sha256"])
    ):
        raise RepositoryError("current channel sigstoreTrustedRootTarget is invalid")
    validate_target_path(reference["path"])
    if channel["sigstoreTrustedRootTarget"]["path"] != "sigstore/trusted_root.json":
        raise RepositoryError("current channel Sigstore trusted-root path is invalid")
    active = channel["active"]
    if not isinstance(active, list) or len(active) > 2:
        raise RepositoryError("current channel active set is invalid")
    manifests: set[str] = set()
    bundles: set[str] = set()
    for release in active:
        manifest_match = (
            re.fullmatch(
                rf"releases/({SEMVER_RE.pattern[1:-1]})/{environment}/manifest[.]json",
                release.get("manifestTarget", "") if isinstance(release, dict) else "",
            )
            if isinstance(release, dict)
            else None
        )
        bundle_match = (
            re.fullmatch(
                rf"releases/({SEMVER_RE.pattern[1:-1]})/{environment}/manifest[.]sigstore[.]json",
                release.get("bundleTarget", "") if isinstance(release, dict) else "",
            )
            if isinstance(release, dict)
            else None
        )
        if (
            not isinstance(release, dict)
            or set(release)
            != {"manifestTarget", "manifestSha256", "bundleTarget", "bundleSha256"}
            or not all(isinstance(value, str) for value in release.values())
            or not SHA256_RE.fullmatch(release["manifestSha256"])
            or not SHA256_RE.fullmatch(release["bundleSha256"])
            or manifest_match is None
            or bundle_match is None
            or manifest_match.group(1) != bundle_match.group(1)
            or release["manifestTarget"] in manifests
            or release["bundleTarget"] in bundles
        ):
            raise RepositoryError("current channel release reference is invalid")
        validate_release_version(manifest_match.group(1))
        manifests.add(release["manifestTarget"])
        bundles.add(release["bundleTarget"])
    return channel


def target_ref(path: str, data: bytes) -> dict[str, str]:
    return {"path": path, "sha256": sha256_bytes(data)}


def archive_timestamp(repository: Path, version: int, data: bytes) -> None:
    destination = repository / "timestamp-history" / f"{version}.timestamp.json"
    if destination.exists():
        if require_file(destination, "historical timestamp") != data:
            raise RepositoryError(
                f"refusing to replace immutable timestamp version {version}"
            )
        return
    atomic_write(destination, data)


def prune_inactive_release_targets(repository: Path) -> None:
    keep: set[str] = set()
    for environment in sorted(ENVIRONMENTS):
        channel_path = repository / "targets" / "channels" / f"{environment}.json"
        if not channel_path.exists():
            continue
        channel_bytes = require_file(channel_path, f"{environment} channel")
        require_size(channel_bytes, MAX_CHANNEL_BYTES, f"{environment} channel")
        channel = validate_channel(
            strict_json_bytes(channel_bytes, f"{environment} channel"), environment
        )
        for release in channel["active"]:
            keep.add(release["manifestTarget"])
            keep.add(release["bundleTarget"])

    releases = repository / "targets" / "releases"
    if not releases.exists():
        return
    for path in sorted(releases.rglob("*"), reverse=True):
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            logical = path.relative_to(repository / "targets").as_posix()
            if logical not in keep:
                path.unlink()
        elif stat.S_ISDIR(mode):
            try:
                path.rmdir()
            except OSError:
                pass
        else:
            raise RepositoryError(
                f"release targets contain a symlink or special file: {path}"
            )


def write_top_level(
    repository: Path,
    signer: CryptoSigner,
    old_targets: Metadata[Targets],
    old_snapshot: Metadata[Snapshot],
    old_timestamp: Metadata[Timestamp],
) -> None:
    descriptors: dict[str, TargetFile] = {}
    for path in sorted((repository / "targets").rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            logical = path.relative_to(repository / "targets").as_posix()
            validate_target_path(logical)
            data = path.read_bytes()
            require_size(data, target_size_limit(logical), logical)
            descriptors[logical] = TargetFile.from_data(logical, data, ["sha256"])
        elif not stat.S_ISDIR(mode):
            raise RepositoryError(
                f"targets tree contains a symlink or special file: {path}"
            )

    targets = Metadata(
        Targets(
            version=old_targets.signed.version + 1,
            expires=utcnow() + timedelta(days=TARGETS_EXPIRY_DAYS),
            targets=descriptors,
        )
    )
    targets.sign(signer)
    targets_bytes = targets.to_bytes()
    require_size(
        targets_bytes, MAX_TARGETS_METADATA_BYTES, "generated targets metadata"
    )
    atomic_write(
        repository / "metadata" / f"{targets.signed.version}.targets.json",
        targets_bytes,
    )

    snapshot = Metadata(
        Snapshot(
            version=old_snapshot.signed.version + 1,
            expires=utcnow() + timedelta(days=SNAPSHOT_EXPIRY_DAYS),
            meta={
                "targets.json": MetaFile.from_data(
                    targets.signed.version, targets_bytes, ["sha256"]
                )
            },
        )
    )
    snapshot.sign(signer)
    snapshot_bytes = snapshot.to_bytes()
    require_size(snapshot_bytes, MAX_SNAPSHOT_BYTES, "generated snapshot metadata")
    atomic_write(
        repository / "metadata" / f"{snapshot.signed.version}.snapshot.json",
        snapshot_bytes,
    )

    timestamp = Metadata(
        Timestamp(
            version=old_timestamp.signed.version + 1,
            expires=utcnow() + timedelta(hours=TIMESTAMP_EXPIRY_HOURS),
            snapshot_meta=MetaFile.from_data(
                snapshot.signed.version, snapshot_bytes, ["sha256"]
            ),
        )
    )
    timestamp.sign(signer)
    timestamp_bytes = timestamp.to_bytes()
    require_size(timestamp_bytes, MAX_TIMESTAMP_BYTES, "generated timestamp metadata")
    archive_timestamp(repository, timestamp.signed.version, timestamp_bytes)
    atomic_write(repository / "metadata" / "timestamp.json", timestamp_bytes)


def validate_repository(repository: Path) -> None:
    validate_safe_tree(repository)
    root = load_root_chain(repository)
    targets, snapshot, current_timestamp = load_current(repository, root.signed)
    if "policy/builders.json" in targets.signed.targets:
        raise RepositoryError(
            "builder configuration must remain promotion-local, not a TUF target"
        )
    actual_targets = {
        path.relative_to(repository / "targets").as_posix()
        for path in (repository / "targets").rglob("*")
        if stat.S_ISREG(path.lstat().st_mode)
    }
    if actual_targets != set(targets.signed.targets):
        raise RepositoryError(
            "logical targets tree contains unsigned, missing, or renamed files"
        )
    for logical in targets.signed.targets:
        data = authenticated_target(repository, targets.signed, logical)
        require_size(data, target_size_limit(logical), logical)

    roots = [
        load_metadata(path, Root)
        for path in sorted(
            (repository / "metadata").glob("*.root.json"),
            key=lambda path: int(path.name.split(".", 1)[0]),
        )
    ]

    def verify_with_any_root(role_name: str, metadata: Metadata[Any]) -> None:
        for candidate in roots:
            try:
                candidate.signed.verify_delegate(
                    role_name, metadata.signed_bytes, metadata.signatures
                )
                return
            except Exception:
                continue
        raise RepositoryError(
            f"historical {role_name} metadata has no valid root-authorized signature"
        )

    archive = repository / "published-targets"
    actual_archive: set[str] = set()
    for path in archive.rglob("*"):
        if stat.S_ISDIR(path.lstat().st_mode):
            continue
        if not stat.S_ISREG(path.lstat().st_mode):
            raise RepositoryError(
                f"published target archive contains a symlink: {path}"
            )
        digest = path.name.split(".", 1)[0]
        if not SHA256_RE.fullmatch(digest) or sha256_bytes(path.read_bytes()) != digest:
            raise RepositoryError(f"published target archive is not immutable: {path}")
        actual_archive.add(path.relative_to(archive).as_posix())

    target_versions = {
        int(path.name.split(".", 1)[0]): path
        for path in (repository / "metadata").glob("*.targets.json")
    }
    snapshot_versions = {
        int(path.name.split(".", 1)[0]): path
        for path in (repository / "metadata").glob("*.snapshot.json")
    }
    expected = set(range(1, targets.signed.version + 1))
    if set(target_versions) != expected or set(snapshot_versions) != set(
        range(1, snapshot.signed.version + 1)
    ):
        raise RepositoryError(
            "numbered targets and snapshot metadata must be append-only and contiguous"
        )

    timestamp_history = repository / "timestamp-history"
    timestamp_versions: dict[int, Path] = {}
    for path in timestamp_history.glob("*"):
        match = re.fullmatch(r"([1-9][0-9]*)[.]timestamp[.]json", path.name)
        if match is None or not stat.S_ISREG(path.lstat().st_mode):
            raise RepositoryError(f"invalid timestamp history entry: {path}")
        timestamp_versions[int(match.group(1))] = path
    expected_timestamps = set(range(1, current_timestamp.signed.version + 1))
    if set(timestamp_versions) != expected_timestamps:
        raise RepositoryError(
            "timestamp history must be append-only and contiguous from version 1"
        )
    for version in sorted(expected_timestamps):
        timestamp_bytes = require_file(
            timestamp_versions[version], "historical timestamp"
        )
        require_size(
            timestamp_bytes, MAX_TIMESTAMP_BYTES, "historical timestamp metadata"
        )
        historical_timestamp = load_metadata(timestamp_versions[version], Timestamp)
        if historical_timestamp.signed.version != version:
            raise RepositoryError("historical timestamp filename/version mismatch")
        verify_with_any_root("timestamp", historical_timestamp)
        snapshot_version = historical_timestamp.signed.snapshot_meta.version
        described_snapshot = snapshot_versions.get(snapshot_version)
        if described_snapshot is None or historical_timestamp.signed.snapshot_meta != (
            MetaFile.from_data(
                snapshot_version,
                require_file(described_snapshot, "historical snapshot"),
                ["sha256"],
            )
        ):
            raise RepositoryError(
                "historical timestamp does not describe retained snapshot metadata"
            )
    if (
        require_file(
            timestamp_versions[current_timestamp.signed.version],
            "current timestamp history",
        )
        != current_timestamp.to_bytes()
    ):
        raise RepositoryError("current timestamp does not match its immutable history")

    expected_archive: set[str] = set()
    for version in sorted(expected):
        historical_targets_bytes = require_file(
            target_versions[version], "historical targets"
        )
        require_size(
            historical_targets_bytes,
            MAX_TARGETS_METADATA_BYTES,
            "historical targets metadata",
        )
        historical_targets = load_metadata(target_versions[version], Targets)
        if historical_targets.signed.version != version:
            raise RepositoryError("historical targets filename/version mismatch")
        validate_targets_client_subset(historical_targets.signed)
        verify_with_any_root("targets", historical_targets)
        for logical, descriptor in historical_targets.signed.targets.items():
            logical_path = Path(logical)
            archived_path = (
                archive
                / logical_path.parent
                / f"{descriptor.hashes['sha256']}.{logical_path.name}"
            )
            expected_archive.add(archived_path.relative_to(archive).as_posix())
            data = require_file(archived_path, f"historical target {logical}")
            if descriptor.length != len(data) or descriptor.hashes[
                "sha256"
            ] != sha256_bytes(data):
                raise RepositoryError(
                    f"historical target bytes are unavailable: {logical}"
                )

        historical_snapshot_bytes = require_file(
            snapshot_versions[version], "historical snapshot"
        )
        require_size(
            historical_snapshot_bytes,
            MAX_SNAPSHOT_BYTES,
            "historical snapshot metadata",
        )
        historical_snapshot = load_metadata(snapshot_versions[version], Snapshot)
        if historical_snapshot.signed.version != version:
            raise RepositoryError("historical snapshot filename/version mismatch")
        verify_with_any_root("snapshot", historical_snapshot)
        described_targets = historical_snapshot.signed.meta.get("targets.json")
        if described_targets != MetaFile.from_data(
            version, historical_targets.to_bytes(), ["sha256"]
        ):
            raise RepositoryError(
                "historical snapshot does not describe its targets metadata"
            )
    if actual_archive != expected_archive:
        raise RepositoryError(
            "published target archive contains unsigned, missing, or renamed files"
        )


def render_public(repository: Path, output: Path) -> None:
    root = load_root_chain(repository)
    targets, _, _ = load_current(repository, root.signed)
    archive = repository / "published-targets"
    archive.mkdir(parents=True, exist_ok=True)
    for logical, descriptor in sorted(targets.signed.targets.items()):
        data = authenticated_target(repository, targets.signed, logical)
        logical_path = Path(logical)
        destination = (
            archive
            / logical_path.parent
            / f"{descriptor.hashes['sha256']}.{logical_path.name}"
        )
        if destination.exists() and destination.read_bytes() != data:
            raise RepositoryError(f"immutable published target changed: {destination}")
        if not destination.exists():
            atomic_write(destination, data)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tuf-public-", dir=output.parent))
    try:
        metadata_output = temporary / "metadata"
        targets_output = temporary / "targets"
        metadata_output.mkdir(parents=True)
        targets_output.mkdir(parents=True)
        for path in sorted((repository / "metadata").glob("*.json")):
            if path.name == "timestamp.json":
                continue
            shutil.copyfile(path, metadata_output / path.name)
        shutil.copytree(archive, targets_output, dirs_exist_ok=True)
        shutil.copyfile(
            repository / "metadata" / "timestamp.json",
            metadata_output / "timestamp.json",
        )
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def bootstrap(
    repository: Path,
    output: Path,
    root_key: Path,
    online_key: Path,
    sigstore_root: Path,
) -> None:
    if repository.exists() and any(repository.iterdir()):
        raise RepositoryError("refusing to bootstrap a non-empty repository")
    check_key_permissions(root_key)
    check_key_permissions(online_key)
    root_signer = load_signer(root_key)
    online_signer = load_signer(online_key)
    if normalized_public_key_material(
        root_signer.public_key
    ) == normalized_public_key_material(online_signer.public_key):
        raise RepositoryError(
            "offline root and online signing key material must differ"
        )
    trusted_root = require_file(sigstore_root, "Sigstore trusted root")
    validate_sigstore_root(trusted_root)

    root = Root(
        version=1,
        expires=utcnow() + timedelta(days=ROOT_EXPIRY_DAYS),
        consistent_snapshot=True,
    )
    root.add_key(root_signer.public_key, "root")
    for role in ONLINE_ROLES:
        root.add_key(online_signer.public_key, role)
    root_metadata = Metadata(root)
    root_metadata.sign(root_signer)
    atomic_write(repository / "metadata" / "1.root.json", root_metadata.to_bytes())
    atomic_write(repository / "targets/sigstore/trusted_root.json", trusted_root)

    descriptors = {
        logical: TargetFile.from_data(
            logical, (repository / "targets" / logical).read_bytes(), ["sha256"]
        )
        for logical in ("sigstore/trusted_root.json",)
    }
    targets = Metadata(
        Targets(
            version=1,
            expires=utcnow() + timedelta(days=TARGETS_EXPIRY_DAYS),
            targets=descriptors,
        )
    )
    targets.sign(online_signer)
    targets_bytes = targets.to_bytes()
    atomic_write(repository / "metadata/1.targets.json", targets_bytes)
    snapshot = Metadata(
        Snapshot(
            version=1,
            expires=utcnow() + timedelta(days=SNAPSHOT_EXPIRY_DAYS),
            meta={"targets.json": MetaFile.from_data(1, targets_bytes, ["sha256"])},
        )
    )
    snapshot.sign(online_signer)
    snapshot_bytes = snapshot.to_bytes()
    atomic_write(repository / "metadata/1.snapshot.json", snapshot_bytes)
    timestamp = Metadata(
        Timestamp(
            version=1,
            expires=utcnow() + timedelta(hours=TIMESTAMP_EXPIRY_HOURS),
            snapshot_meta=MetaFile.from_data(1, snapshot_bytes, ["sha256"]),
        )
    )
    timestamp.sign(online_signer)
    timestamp_bytes = timestamp.to_bytes()
    archive_timestamp(repository, timestamp.signed.version, timestamp_bytes)
    atomic_write(repository / "metadata/timestamp.json", timestamp_bytes)
    render_public(repository, output)
    validate_repository(repository)


def promote(
    repository: Path,
    output: Path,
    online_key: Path,
    builder_policy: Path,
    environment: str,
    version: str,
    phase: str,
    manifest_path: Path,
    bundle_path: Path,
) -> None:
    if environment not in ENVIRONMENTS:
        raise RepositoryError("environment must be dev or prod")
    if phase not in {"rollout", "finalize"}:
        raise RepositoryError("promotion phase must be rollout or finalize")
    validate_release_version(version)
    validate_repository(repository)
    check_key_permissions(online_key)
    signer = load_signer(online_key)
    root = load_root_chain(repository)
    require_online_key(root.signed, signer)
    old_targets, old_snapshot, old_timestamp = load_current(repository, root.signed)

    trusted_root_path = "sigstore/trusted_root.json"
    trusted_root_bytes = authenticated_target(
        repository, old_targets.signed, trusted_root_path
    )
    policy_bytes = require_file(builder_policy, "promotion builder policy")
    policy = validate_builder_policy(policy_bytes)
    trusted_root = validate_sigstore_root(trusted_root_bytes)

    manifest_bytes = require_file(manifest_path, "release manifest")
    bundle_bytes = require_file(bundle_path, "Sigstore bundle")
    manifest = validate_manifest(manifest_bytes, environment, version)
    validate_bundle(bundle_bytes, trusted_root)
    if manifest["build"]["builderId"] not in policy["builders"]:
        raise RepositoryError("manifest builderId is not authorized by builder policy")
    builder = policy["builders"][manifest["build"]["builderId"]]
    source_url = validate_https_url(
        manifest["source"]["uri"], "release manifest source.uri"
    )
    if source_url.path != f"/{builder['workflowRepository']}":
        raise RepositoryError(
            "manifest source.uri does not match the authenticated builder repository"
        )

    manifest_target = f"releases/{version}/{environment}/manifest.json"
    bundle_target = f"releases/{version}/{environment}/manifest.sigstore.json"
    for logical, data in (
        (manifest_target, manifest_bytes),
        (bundle_target, bundle_bytes),
    ):
        destination = repository / "targets" / logical
        if destination.exists() and require_file(destination, logical) != data:
            raise RepositoryError(
                f"refusing to replace immutable release target: {logical}"
            )
    channel_path = f"channels/{environment}.json"
    previous_active: list[dict[str, str]] = []
    sequence = 1
    if channel_path in old_targets.signed.targets:
        previous_channel_bytes = authenticated_target(
            repository, old_targets.signed, channel_path
        )
        require_size(
            previous_channel_bytes, MAX_CHANNEL_BYTES, f"{environment} channel"
        )
        channel = strict_json_bytes(
            previous_channel_bytes,
            f"{environment} channel",
        )
        channel = validate_channel(channel, environment)
        if channel["sigstoreTrustedRootTarget"] != target_ref(
            trusted_root_path, trusted_root_bytes
        ):
            raise RepositoryError(
                "current channel Sigstore trusted-root reference does not match "
                "the authenticated target"
            )
        previous_active = channel["active"]
        sequence = channel["sequence"] + 1

    if phase == "rollout":
        candidate_pcrs = manifest["measurements"]["pcrs"]
        for active_release in previous_active:
            active_manifest_bytes = authenticated_target(
                repository,
                old_targets.signed,
                active_release["manifestTarget"],
            )
            active_bundle_bytes = authenticated_target(
                repository,
                old_targets.signed,
                active_release["bundleTarget"],
            )
            if (
                sha256_bytes(active_manifest_bytes) != active_release["manifestSha256"]
                or sha256_bytes(active_bundle_bytes) != active_release["bundleSha256"]
            ):
                raise RepositoryError(
                    "current channel release hashes do not match authenticated targets"
                )
            active_version = active_release["manifestTarget"].split("/")[1]
            active_manifest = validate_manifest(
                active_manifest_bytes, environment, active_version
            )
            if active_manifest["measurements"]["pcrs"] == candidate_pcrs:
                raise RepositoryError(
                    "rollout candidate duplicates an active PCR0/PCR1/PCR2 tuple; use finalize instead"
                )

    candidate = {
        "bundleSha256": sha256_bytes(bundle_bytes),
        "bundleTarget": bundle_target,
        "manifestSha256": sha256_bytes(manifest_bytes),
        "manifestTarget": manifest_target,
    }
    if phase == "finalize":
        active = [candidate]
    else:
        active = [
            entry
            for entry in previous_active
            if isinstance(entry, dict)
            and entry.get("manifestTarget") != manifest_target
            and entry.get("bundleTarget") != bundle_target
        ][-1:] + [candidate]
    if not 1 <= len(active) <= 2:
        raise RepositoryError("promotion must authorize one or two active releases")

    atomic_write(repository / "targets" / manifest_target, manifest_bytes)
    atomic_write(repository / "targets" / bundle_target, bundle_bytes)

    channel = {
        "active": active,
        "environment": environment,
        "schema": CHANNEL_SCHEMA,
        "sequence": sequence,
        "sigstoreTrustedRootTarget": target_ref(trusted_root_path, trusted_root_bytes),
    }
    atomic_write(repository / "targets" / channel_path, canonical_json(channel))
    prune_inactive_release_targets(repository)
    write_top_level(repository, signer, old_targets, old_snapshot, old_timestamp)
    render_public(repository, output)
    validate_repository(repository)


def revoke(repository: Path, output: Path, online_key: Path, environment: str) -> None:
    if environment not in ENVIRONMENTS:
        raise RepositoryError("environment must be dev or prod")
    validate_repository(repository)
    check_key_permissions(online_key)
    signer = load_signer(online_key)
    root = load_root_chain(repository)
    require_online_key(root.signed, signer)
    old_targets, old_snapshot, old_timestamp = load_current(repository, root.signed)

    trusted_root_path = "sigstore/trusted_root.json"
    trusted_root_bytes = authenticated_target(
        repository, old_targets.signed, trusted_root_path
    )
    validate_sigstore_root(trusted_root_bytes)

    channel_path = f"channels/{environment}.json"
    sequence = 1
    if channel_path in old_targets.signed.targets:
        previous_channel_bytes = authenticated_target(
            repository, old_targets.signed, channel_path
        )
        require_size(
            previous_channel_bytes, MAX_CHANNEL_BYTES, f"{environment} channel"
        )
        previous = validate_channel(
            strict_json_bytes(
                previous_channel_bytes,
                f"{environment} channel",
            ),
            environment,
        )
        if previous["sigstoreTrustedRootTarget"] != target_ref(
            trusted_root_path, trusted_root_bytes
        ):
            raise RepositoryError(
                "current channel Sigstore trusted-root reference does not match "
                "the authenticated target"
            )
        sequence = previous["sequence"] + 1

    channel = {
        "active": [],
        "environment": environment,
        "schema": CHANNEL_SCHEMA,
        "sequence": sequence,
        "sigstoreTrustedRootTarget": target_ref(trusted_root_path, trusted_root_bytes),
    }
    atomic_write(repository / "targets" / channel_path, canonical_json(channel))
    prune_inactive_release_targets(repository)
    write_top_level(repository, signer, old_targets, old_snapshot, old_timestamp)
    render_public(repository, output)
    validate_repository(repository)


def refresh(repository: Path, output: Path, online_key: Path) -> None:
    validate_repository(repository)
    check_key_permissions(online_key)
    signer = load_signer(online_key)
    root = load_root_chain(repository)
    require_online_key(root.signed, signer)
    targets, snapshot, timestamp = load_current(repository, root.signed)
    write_top_level(repository, signer, targets, snapshot, timestamp)
    render_public(repository, output)
    validate_repository(repository)


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        return None


def network_fetcher(base_url: str) -> Callable[[str], bytes]:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RepositoryError("live TUF base URL must be a plain HTTPS origin and path")
    normalized = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/") + "/", "", "")
    )
    opener = urllib.request.build_opener(_RejectRedirects())

    def fetch(relative_path: str) -> bytes:
        validate_target_path(relative_path)
        url = urllib.parse.urljoin(normalized, relative_path)
        if not url.startswith(normalized):
            raise RepositoryError(
                f"live TUF path escapes its base URL: {relative_path}"
            )
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json, application/octet-stream"},
            method="GET",
        )
        try:
            with opener.open(request, timeout=15) as response:
                if response.status != 200:
                    raise RepositoryError(
                        f"live TUF fetch returned HTTP {response.status} for {relative_path}"
                    )
                data = response.read(MAX_LIVE_FILE_BYTES + 1)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise LiveFileNotFound(relative_path) from error
            if 300 <= error.code < 400:
                raise RepositoryError(
                    f"live TUF fetch redirected for {relative_path}"
                ) from error
            raise RepositoryError(
                f"live TUF fetch returned HTTP {error.code} for {relative_path}"
            ) from error
        except urllib.error.URLError as error:
            raise RepositoryError(
                f"live TUF fetch failed for {relative_path}: {error.reason}"
            ) from error
        if len(data) > MAX_LIVE_FILE_BYTES:
            raise RepositoryError(f"live TUF file is too large: {relative_path}")
        return data

    return fetch


def verify_live_state(
    repository: Path,
    base_url: str,
    allow_unpublished: bool = False,
    fetcher: Callable[[str], bytes] | None = None,
) -> None:
    """Prove public state is not ahead of or forked from checked-out state."""

    validate_repository(repository)
    current_root = load_root_chain(repository)
    current_targets, current_snapshot, current_timestamp = load_current(
        repository, current_root.signed
    )
    fetch = fetcher or network_fetcher(base_url)

    try:
        live_timestamp_bytes = fetch("metadata/timestamp.json")
    except LiveFileNotFound:
        unactivated = (
            current_root.signed.version == 1
            and current_targets.signed.version == current_snapshot.signed.version
            and current_snapshot.signed.version == current_timestamp.signed.version
            and set(current_targets.signed.targets) == {"sigstore/trusted_root.json"}
        )
        if allow_unpublished and unactivated:
            return
        raise RepositoryError(
            "live timestamp is absent; only explicit publication of never-activated bootstrap state is allowed"
        ) from None

    state_roots = [
        load_metadata(path, Root)
        for path in sorted(
            (repository / "metadata").glob("*.root.json"),
            key=lambda path: int(path.name.split(".", 1)[0]),
        )
    ]

    def verify_role(role_name: str, metadata: Metadata[Any]) -> None:
        for root in state_roots:
            try:
                root.signed.verify_delegate(
                    role_name, metadata.signed_bytes, metadata.signatures
                )
                return
            except Exception:
                continue
        raise RepositoryError(
            f"live {role_name} metadata has no root-authorized signature"
        )

    try:
        live_timestamp = Metadata.from_bytes(live_timestamp_bytes)
    except Exception as error:
        raise RepositoryError(f"live timestamp metadata is invalid: {error}") from error
    if not isinstance(live_timestamp.signed, Timestamp):
        raise RepositoryError("live timestamp path does not contain timestamp metadata")
    verify_role("timestamp", live_timestamp)

    live_version = live_timestamp.signed.version
    state_version = current_timestamp.signed.version
    if live_version > state_version:
        raise RepositoryError(
            f"live timestamp version {live_version} is ahead of checked-out state {state_version}"
        )
    archived_live_timestamp = require_file(
        repository / "timestamp-history" / f"{live_version}.timestamp.json",
        "archived live timestamp",
    )
    if live_timestamp_bytes != archived_live_timestamp:
        raise RepositoryError(
            "live timestamp bytes do not match immutable checked-out timestamp history"
        )

    missing_root_seen = False
    for version, state_root in enumerate(state_roots, start=1):
        try:
            live_root = fetch(f"metadata/{version}.root.json")
        except LiveFileNotFound:
            missing_root_seen = True
            continue
        if missing_root_seen:
            raise RepositoryError("live root metadata versions are not contiguous")
        if live_root != state_root.to_bytes():
            raise RepositoryError(
                f"live root metadata forks checked-out root version {version}"
            )
    try:
        fetch(f"metadata/{len(state_roots) + 1}.root.json")
    except LiveFileNotFound:
        pass
    else:
        raise RepositoryError("live root metadata is ahead of checked-out state")
    if missing_root_seen and live_version == state_version:
        raise RepositoryError(
            "live repository is missing root metadata from current state"
        )

    snapshot_version = live_timestamp.signed.snapshot_meta.version
    if snapshot_version > current_snapshot.signed.version:
        raise RepositoryError(
            f"live snapshot version {snapshot_version} is ahead of checked-out state "
            f"{current_snapshot.signed.version}"
        )
    snapshot_relative = f"metadata/{snapshot_version}.snapshot.json"
    try:
        live_snapshot_bytes = fetch(snapshot_relative)
    except LiveFileNotFound:
        raise RepositoryError(
            f"live timestamp references missing {snapshot_relative}"
        ) from None
    try:
        live_snapshot = Metadata.from_bytes(live_snapshot_bytes)
    except Exception as error:
        raise RepositoryError(f"live snapshot metadata is invalid: {error}") from error
    if not isinstance(live_snapshot.signed, Snapshot):
        raise RepositoryError("live snapshot path does not contain snapshot metadata")
    verify_role("snapshot", live_snapshot)
    if MetaFile.from_data(snapshot_version, live_snapshot_bytes, ["sha256"]) != (
        live_timestamp.signed.snapshot_meta
    ):
        raise RepositoryError(
            "live snapshot bytes do not match live timestamp metadata"
        )
    state_snapshot_path = repository / "metadata" / f"{snapshot_version}.snapshot.json"
    if live_snapshot_bytes != require_file(state_snapshot_path, "historical snapshot"):
        raise RepositoryError(
            "live snapshot is not an exact historical subset of checked-out state"
        )

    for version in range(1, snapshot_version):
        historical_relative = f"metadata/{version}.snapshot.json"
        try:
            historical_live_bytes = fetch(historical_relative)
        except LiveFileNotFound:
            raise RepositoryError(
                f"live repository is missing historical {historical_relative}"
            ) from None
        historical_state_bytes = require_file(
            repository / "metadata" / f"{version}.snapshot.json",
            "historical snapshot",
        )
        if historical_live_bytes != historical_state_bytes:
            raise RepositoryError(
                f"live historical snapshot version {version} forks checked-out history"
            )

    targets_descriptor = live_snapshot.signed.meta.get("targets.json")
    if targets_descriptor is None:
        raise RepositoryError("live snapshot does not reference targets metadata")
    targets_version = targets_descriptor.version
    if targets_version > current_targets.signed.version:
        raise RepositoryError(
            f"live targets version {targets_version} is ahead of checked-out state "
            f"{current_targets.signed.version}"
        )
    targets_relative = f"metadata/{targets_version}.targets.json"
    try:
        live_targets_bytes = fetch(targets_relative)
    except LiveFileNotFound:
        raise RepositoryError(
            f"live snapshot references missing {targets_relative}"
        ) from None
    try:
        live_targets = Metadata.from_bytes(live_targets_bytes)
    except Exception as error:
        raise RepositoryError(f"live targets metadata is invalid: {error}") from error
    if not isinstance(live_targets.signed, Targets):
        raise RepositoryError("live targets path does not contain targets metadata")
    verify_role("targets", live_targets)
    if (
        MetaFile.from_data(targets_version, live_targets_bytes, ["sha256"])
        != targets_descriptor
    ):
        raise RepositoryError("live targets bytes do not match live snapshot metadata")
    state_targets_path = repository / "metadata" / f"{targets_version}.targets.json"
    if live_targets_bytes != require_file(state_targets_path, "historical targets"):
        raise RepositoryError(
            "live targets are not an exact historical subset of checked-out state"
        )

    historical_archived_targets: dict[str, tuple[str, TargetFile]] = {}
    for version in range(1, targets_version + 1):
        historical_state_path = (
            repository / "metadata" / f"{version}.targets.json"
        )
        historical_state_bytes = require_file(
            historical_state_path, "historical targets"
        )
        if version == targets_version:
            historical_live_bytes = live_targets_bytes
            historical_targets = live_targets
        else:
            historical_relative = f"metadata/{version}.targets.json"
            try:
                historical_live_bytes = fetch(historical_relative)
            except LiveFileNotFound:
                raise RepositoryError(
                    f"live repository is missing historical {historical_relative}"
                ) from None
            historical_targets = load_metadata(historical_state_path, Targets)
        if historical_live_bytes != historical_state_bytes:
            raise RepositoryError(
                f"live historical targets version {version} forks checked-out history"
            )
        for logical, descriptor in historical_targets.signed.targets.items():
            validate_target_path(logical)
            digest = descriptor.hashes.get("sha256")
            if digest is None:
                raise RepositoryError(
                    f"live target lacks required SHA-256 hash: {logical}"
                )
            logical_path = Path(logical)
            published_relative = (
                logical_path.parent / f"{digest}.{logical_path.name}"
            ).as_posix()
            historical_archived_targets[published_relative] = (logical, descriptor)

    archive = repository / "published-targets"
    for published_relative, (logical, descriptor) in sorted(
        historical_archived_targets.items()
    ):
        digest = descriptor.hashes["sha256"]
        live_relative = f"targets/{published_relative}"
        try:
            live_bytes = fetch(live_relative)
        except LiveFileNotFound:
            raise RepositoryError(
                f"live metadata references missing {live_relative}"
            ) from None
        if descriptor.length != len(live_bytes) or sha256_bytes(live_bytes) != digest:
            raise RepositoryError(f"live target bytes do not match metadata: {logical}")
        state_bytes = require_file(
            archive / published_relative, f"historical target {logical}"
        )
        if live_bytes != state_bytes:
            raise RepositoryError(f"live target forks checked-out history: {logical}")


def generate_key(output: Path) -> None:
    if output.exists():
        raise RepositoryError(f"refusing to overwrite {output}")
    signer = CryptoSigner.generate_ed25519()
    atomic_write(output, signer.private_bytes, mode=0o600)
    print(
        f"generated Ed25519 key {signer.public_key.keyid} at {output}", file=sys.stderr
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    key = commands.add_parser("generate-key")
    key.add_argument("--output", required=True, type=Path)
    init = commands.add_parser("bootstrap")
    for name in (
        "repository",
        "output",
        "root-key",
        "online-key",
        "sigstore-trusted-root",
    ):
        init.add_argument(f"--{name}", required=True, type=Path)
    promotion = commands.add_parser("promote")
    promotion.add_argument("--repository", required=True, type=Path)
    promotion.add_argument("--output", required=True, type=Path)
    promotion.add_argument("--online-key", required=True, type=Path)
    promotion.add_argument("--builder-policy", required=True, type=Path)
    promotion.add_argument("--environment", required=True, choices=sorted(ENVIRONMENTS))
    promotion.add_argument("--version", required=True)
    promotion.add_argument("--phase", required=True, choices=("rollout", "finalize"))
    promotion.add_argument("--manifest", required=True, type=Path)
    promotion.add_argument("--bundle", required=True, type=Path)
    renewal = commands.add_parser("refresh")
    renewal.add_argument("--repository", required=True, type=Path)
    renewal.add_argument("--output", required=True, type=Path)
    renewal.add_argument("--online-key", required=True, type=Path)
    revocation = commands.add_parser("revoke")
    revocation.add_argument("--repository", required=True, type=Path)
    revocation.add_argument("--output", required=True, type=Path)
    revocation.add_argument("--online-key", required=True, type=Path)
    revocation.add_argument(
        "--environment", required=True, choices=sorted(ENVIRONMENTS)
    )
    verify = commands.add_parser("verify")
    verify.add_argument("--repository", required=True, type=Path)
    verify_live = commands.add_parser("verify-live")
    verify_live.add_argument("--repository", required=True, type=Path)
    verify_live.add_argument("--base-url", required=True)
    verify_live.add_argument("--allow-unpublished", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str]) -> None:
    args = parse_args(argv)
    if args.command == "generate-key":
        generate_key(args.output)
    elif args.command == "bootstrap":
        bootstrap(
            args.repository,
            args.output,
            args.root_key,
            args.online_key,
            args.sigstore_trusted_root,
        )
    elif args.command == "promote":
        promote(
            args.repository,
            args.output,
            args.online_key,
            args.builder_policy,
            args.environment,
            args.version,
            args.phase,
            args.manifest,
            args.bundle,
        )
    elif args.command == "refresh":
        refresh(args.repository, args.output, args.online_key)
    elif args.command == "revoke":
        revoke(args.repository, args.output, args.online_key, args.environment)
    elif args.command == "verify-live":
        verify_live_state(
            args.repository,
            args.base_url,
            allow_unpublished=args.allow_unpublished,
        )
        print(f"verified live continuity for {args.repository}")
    else:
        validate_repository(args.repository)
        print(f"verified {args.repository}")


def main() -> int:
    try:
        run(sys.argv[1:])
        return 0
    except (RepositoryError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
