#!/usr/bin/env python3
"""Generate and verify canonical repository-neutral Nitro EIF release manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

SCHEMA = "https://attestations.trymaple.ai/schemas/nitro-eif-release/v1"
COMPONENT = "opensecret-backend"
EIF_MEDIA_TYPE = "application/vnd.aws.nitro.eif"
GIT_REVISION_ALGORITHM = "git-sha1"

SEMVER_TAG_RE = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PCR_RE = re.compile(r"^[0-9a-f]{96}$")
BUILDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
PCR_FILE_KEYS = {"HashAlgorithm", "PCR0", "PCR1", "PCR2"}
PCR_ALGORITHMS = {"Sha384", "Sha384 { ... }"}


class ManifestError(ValueError):
    """Raised when release inputs or a manifest violate the v1 contract."""


def _reject_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_number(value: str) -> None:
    raise ManifestError(f"non-finite JSON number is not allowed: {value}")


def read_strict_json(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ManifestError(f"could not read {path}: {error}") from error
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_number,
        )
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path} is not valid UTF-8 JSON: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            allow_nan=False,
            separators=(",", ": "),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ManifestError(f"required file does not exist: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ManifestError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def require_exact_keys(value: Any, expected: set[str], context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{context} must be a JSON object")
    actual = set(value)
    if actual != expected:
        details: List[str] = []
        if expected - actual:
            details.append(f"missing={sorted(expected - actual)}")
        if actual - expected:
            details.append(f"unknown={sorted(actual - expected)}")
        raise ManifestError(f"{context} has invalid keys ({', '.join(details)})")
    return value


def require_https_uri(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{context} must be an HTTPS URI")
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ManifestError(
            f"{context} must be an absolute HTTPS URI without credentials or fragment"
        )
    return value


def validate_source_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(
            "source path must be a non-empty repository-relative POSIX path"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ManifestError("source path must stay within the source repository")
    return value


def validate_release_inputs(
    environment: str,
    commit: str,
    tag: str,
    source_uri: str,
    source_path: str,
    builder_id: str,
    run_uri: str,
) -> None:
    if environment not in {"dev", "prod"}:
        raise ManifestError("environment must be exactly 'dev' or 'prod'")
    if not COMMIT_RE.fullmatch(commit):
        raise ManifestError(
            "commit must be exactly 40 lowercase hexadecimal characters"
        )
    if len(tag) > 129 or not SEMVER_TAG_RE.fullmatch(tag):
        raise ManifestError(
            "tag must match vMAJOR.MINOR.PATCH and fit a 128-byte release path segment"
        )
    require_https_uri(source_uri, "source URI")
    validate_source_path(source_path)
    if not BUILDER_ID_RE.fullmatch(builder_id):
        raise ManifestError("builder ID must be a stable lowercase identifier")
    require_https_uri(run_uri, "build run URI")


def validate_pcr_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not PCR_RE.fullmatch(value):
        raise ManifestError(
            f"{name} must be exactly 96 lowercase hexadecimal characters"
        )
    if set(value) == {"0"}:
        raise ManifestError(f"{name} must not be all zeroes")
    return value


def read_pcr_file(path: Path) -> Dict[str, str]:
    document = require_exact_keys(read_strict_json(path), PCR_FILE_KEYS, "PCR file")
    if document["HashAlgorithm"] not in PCR_ALGORITHMS:
        raise ManifestError("PCR file HashAlgorithm must identify SHA-384")
    return {
        "0": validate_pcr_value(document["PCR0"], "PCR0"),
        "1": validate_pcr_value(document["PCR1"], "PCR1"),
        "2": validate_pcr_value(document["PCR2"], "PCR2"),
    }


def version_from_tag(tag: str) -> str:
    if len(tag) > 129 or not SEMVER_TAG_RE.fullmatch(tag):
        raise ManifestError(
            "tag must match vMAJOR.MINOR.PATCH and fit a 128-byte release path segment"
        )
    return tag[1:]


def expected_artifact_name(tag: str, environment: str) -> str:
    return f"opensecret-{tag}-{environment}.eif"


def build_manifest(
    *,
    environment: str,
    commit: str,
    tag: str,
    source_uri: str,
    source_path: str,
    builder_id: str,
    run_uri: str,
    pcr_file: Path,
    eif_file: Path,
    flake_lock: Path,
) -> Dict[str, Any]:
    validate_release_inputs(
        environment, commit, tag, source_uri, source_path, builder_id, run_uri
    )
    pcrs = read_pcr_file(pcr_file)
    if not eif_file.is_file():
        raise ManifestError(f"required EIF does not exist: {eif_file}")
    eif_size = eif_file.stat().st_size
    if eif_size <= 0:
        raise ManifestError("EIF must not be empty")

    return {
        "artifact": {
            "digests": {"sha256": sha256_file(eif_file)},
            "mediaType": EIF_MEDIA_TYPE,
            "name": expected_artifact_name(tag, environment),
            "size": eif_size,
        },
        "build": {
            "builderId": builder_id,
            "derivation": f".#eif-{environment}",
            "flakeLockSha256": sha256_file(flake_lock),
            "runUri": run_uri,
            "system": "nix",
        },
        "component": COMPONENT,
        "environment": environment,
        "measurements": {
            "algorithm": "sha384",
            "pcrs": pcrs,
            "requiredPcrs": [0, 1, 2],
        },
        "release": {"version": version_from_tag(tag)},
        "schema": SCHEMA,
        "source": {
            "path": source_path,
            "ref": f"refs/tags/{tag}",
            "revision": {
                "algorithm": GIT_REVISION_ALGORITHM,
                "digest": commit,
            },
            "uri": source_uri,
        },
    }


def validate_manifest(
    manifest: Any,
    *,
    environment: str,
    commit: str,
    tag: str,
    source_uri: str,
    source_path: str,
    builder_id: str,
    run_uri: str,
    pcr_file: Path,
    eif_file: Path,
    flake_lock: Path,
) -> None:
    validate_release_inputs(
        environment, commit, tag, source_uri, source_path, builder_id, run_uri
    )
    root = require_exact_keys(
        manifest,
        {
            "artifact",
            "build",
            "component",
            "environment",
            "measurements",
            "release",
            "schema",
            "source",
        },
        "manifest",
    )
    if root["schema"] != SCHEMA:
        raise ManifestError(f"unsupported manifest schema: {root['schema']!r}")
    if root["component"] != COMPONENT:
        raise ManifestError("manifest component is invalid")
    if root["environment"] != environment:
        raise ManifestError(
            "manifest environment does not match the expected environment"
        )

    source = require_exact_keys(
        root["source"], {"uri", "path", "ref", "revision"}, "manifest.source"
    )
    revision = require_exact_keys(
        source["revision"], {"algorithm", "digest"}, "manifest.source.revision"
    )
    expected_source = {
        "uri": source_uri,
        "path": source_path,
        "ref": f"refs/tags/{tag}",
        "revision": {"algorithm": GIT_REVISION_ALGORITHM, "digest": commit},
    }
    if source != expected_source or revision["algorithm"] != GIT_REVISION_ALGORITHM:
        raise ManifestError(
            "manifest source does not match the expected release source"
        )

    release = require_exact_keys(root["release"], {"version"}, "manifest.release")
    if release["version"] != version_from_tag(tag):
        raise ManifestError("manifest release version does not match the selected tag")

    artifact = require_exact_keys(
        root["artifact"], {"digests", "mediaType", "name", "size"}, "manifest.artifact"
    )
    digests = require_exact_keys(
        artifact["digests"], {"sha256"}, "manifest.artifact.digests"
    )
    if artifact["mediaType"] != EIF_MEDIA_TYPE:
        raise ManifestError("manifest artifact media type is invalid")
    if artifact["name"] != expected_artifact_name(tag, environment):
        raise ManifestError("manifest artifact name is invalid")
    if not isinstance(digests["sha256"], str) or not SHA256_RE.fullmatch(
        digests["sha256"]
    ):
        raise ManifestError("manifest artifact SHA-256 is invalid")
    if type(artifact["size"]) is not int or artifact["size"] <= 0:
        raise ManifestError("manifest artifact size must be a positive integer")
    if digests["sha256"] != sha256_file(eif_file):
        raise ManifestError("manifest artifact SHA-256 does not match the EIF")
    if artifact["size"] != eif_file.stat().st_size:
        raise ManifestError("manifest artifact size does not match the EIF")

    measurements = require_exact_keys(
        root["measurements"],
        {"algorithm", "pcrs", "requiredPcrs"},
        "manifest.measurements",
    )
    if measurements["algorithm"] != "sha384":
        raise ManifestError("manifest measurement algorithm must be sha384")
    required = measurements["requiredPcrs"]
    if (
        not isinstance(required, list)
        or any(type(i) is not int for i in required)
        or required != [0, 1, 2]
    ):
        raise ManifestError("manifest required PCRs must be exactly [0, 1, 2]")
    pcrs = require_exact_keys(
        measurements["pcrs"], {"0", "1", "2"}, "manifest.measurements.pcrs"
    )
    for index in ("0", "1", "2"):
        validate_pcr_value(pcrs[index], f"manifest PCR{index}")
    if pcrs != read_pcr_file(pcr_file):
        raise ManifestError("manifest PCR tuple does not match the build PCR file")

    build = require_exact_keys(
        root["build"],
        {"builderId", "derivation", "flakeLockSha256", "runUri", "system"},
        "manifest.build",
    )
    if build != {
        "builderId": builder_id,
        "derivation": f".#eif-{environment}",
        "flakeLockSha256": sha256_file(flake_lock),
        "runUri": run_uri,
        "system": "nix",
    }:
        raise ManifestError("manifest build does not match the expected build")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--environment", required=True, choices=("dev", "prod"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-path", default=".")
    parser.add_argument("--builder-id", required=True)
    parser.add_argument("--run-uri", required=True)
    parser.add_argument("--pcr", required=True, type=Path)
    parser.add_argument("--eif", required=True, type=Path)
    parser.add_argument("--flake-lock", required=True, type=Path)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    add_common_arguments(generate_parser)
    generate_parser.add_argument("--output", required=True, type=Path)
    verify_parser = subparsers.add_parser("verify")
    add_common_arguments(verify_parser)
    verify_parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args(argv)


def run(argv: List[str]) -> None:
    args = parse_args(argv)
    common = {
        "environment": args.environment,
        "commit": args.commit,
        "tag": args.tag,
        "source_uri": args.source_uri,
        "source_path": args.source_path,
        "builder_id": args.builder_id,
        "run_uri": args.run_uri,
        "pcr_file": args.pcr,
        "eif_file": args.eif,
        "flake_lock": args.flake_lock,
    }
    if args.command == "generate":
        manifest = build_manifest(**common)
        validate_manifest(manifest, **common)
        output = canonical_json_bytes(manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(output)
        print(f"wrote {args.output} (sha256={hashlib.sha256(output).hexdigest()})")
        return

    raw_manifest = args.manifest.read_bytes()
    manifest = read_strict_json(args.manifest)
    if raw_manifest != canonical_json_bytes(manifest):
        raise ManifestError(
            "manifest bytes are not canonical sorted two-space JSON with one trailing LF"
        )
    validate_manifest(manifest, **common)
    print(f"verified {args.manifest}")


def main() -> int:
    try:
        run(sys.argv[1:])
        return 0
    except (ManifestError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
