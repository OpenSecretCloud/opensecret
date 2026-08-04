#!/usr/bin/env python3
"""Generate and verify canonical OpenSecret Nitro EIF release manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SCHEMA = "https://opensecret.cloud/attestations/nitro-eif-release/v1"
SOURCE_REPOSITORY = "OpenSecretCloud/opensecret"
SOURCE_REPOSITORY_ID = 921901924
SOURCE_OWNER_ID = 185423582
EIF_MEDIA_TYPE = "application/vnd.aws.nitro.eif"

SEMVER_TAG_RE = re.compile(
    r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PCR_RE = re.compile(r"^[0-9a-f]{96}$")
WORKFLOW_RUN_RE = re.compile(
    r"^https://github\.com/OpenSecretCloud/opensecret/actions/runs/"
    r"[1-9][0-9]*/attempts/[1-9][0-9]*$"
)
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
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details: List[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ManifestError(f"{context} has invalid keys ({', '.join(details)})")
    return value


def validate_release_inputs(
    environment: str,
    commit: str,
    tag: str,
    workflow_run: str,
) -> None:
    if environment not in {"dev", "prod"}:
        raise ManifestError("environment must be exactly 'dev' or 'prod'")
    if not COMMIT_RE.fullmatch(commit):
        raise ManifestError("commit must be exactly 40 lowercase hexadecimal characters")
    if not SEMVER_TAG_RE.fullmatch(tag):
        raise ManifestError("tag must match exactly vMAJOR.MINOR.PATCH without leading zeroes")
    if not WORKFLOW_RUN_RE.fullmatch(workflow_run):
        raise ManifestError(
            "workflow run must be an exact OpenSecretCloud/opensecret GitHub Actions run-attempt URL"
        )


def validate_pcr_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not PCR_RE.fullmatch(value):
        raise ManifestError(f"{name} must be exactly 96 lowercase hexadecimal characters")
    if set(value) == {"0"}:
        raise ManifestError(f"{name} must not be all zeroes")
    return value


def read_pcr_file(path: Path) -> Dict[str, str]:
    pcr_document = require_exact_keys(read_strict_json(path), PCR_FILE_KEYS, "PCR file")
    if pcr_document["HashAlgorithm"] not in PCR_ALGORITHMS:
        raise ManifestError("PCR file HashAlgorithm must identify SHA-384")

    return {
        "0": validate_pcr_value(pcr_document["PCR0"], "PCR0"),
        "1": validate_pcr_value(pcr_document["PCR1"], "PCR1"),
        "2": validate_pcr_value(pcr_document["PCR2"], "PCR2"),
    }


def expected_artifact_name(tag: str, environment: str) -> str:
    return f"opensecret-{tag}-{environment}.eif"


def build_manifest(
    *,
    environment: str,
    commit: str,
    tag: str,
    workflow_run: str,
    pcr_file: Path,
    eif_file: Path,
    flake_lock: Path,
) -> Dict[str, Any]:
    validate_release_inputs(environment, commit, tag, workflow_run)
    pcrs = read_pcr_file(pcr_file)

    if not eif_file.is_file():
        raise ManifestError(f"required EIF does not exist: {eif_file}")
    eif_size = eif_file.stat().st_size
    if eif_size <= 0:
        raise ManifestError("EIF must not be empty")

    flake_lock_digest = sha256_file(flake_lock)
    eif_digest = sha256_file(eif_file)

    return {
        "artifact": {
            "mediaType": EIF_MEDIA_TYPE,
            "name": expected_artifact_name(tag, environment),
            "sha256": eif_digest,
            "size": eif_size,
        },
        "build": {
            "derivation": f"eif-{environment}",
            "flakeLockSha256": flake_lock_digest,
            "system": "nix",
            "workflowRun": workflow_run,
        },
        "environment": environment,
        "measurements": {
            "algorithm": "sha384",
            "pcrs": pcrs,
            "requiredPcrs": [0, 1, 2],
        },
        "release": {"tag": tag},
        "schema": SCHEMA,
        "source": {
            "commit": commit,
            "ownerId": SOURCE_OWNER_ID,
            "ref": f"refs/tags/{tag}",
            "repository": SOURCE_REPOSITORY,
            "repositoryId": SOURCE_REPOSITORY_ID,
        },
    }


def validate_manifest(
    manifest: Any,
    *,
    environment: str,
    commit: str,
    tag: str,
    workflow_run: str,
    pcr_file: Path,
    eif_file: Path,
    flake_lock: Path,
) -> None:
    validate_release_inputs(environment, commit, tag, workflow_run)
    root = require_exact_keys(
        manifest,
        {
            "artifact",
            "build",
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
    if root["environment"] != environment:
        raise ManifestError("manifest environment does not match the expected environment")

    source = require_exact_keys(
        root["source"],
        {"commit", "ownerId", "ref", "repository", "repositoryId"},
        "manifest.source",
    )
    expected_source = {
        "commit": commit,
        "ownerId": SOURCE_OWNER_ID,
        "ref": f"refs/tags/{tag}",
        "repository": SOURCE_REPOSITORY,
        "repositoryId": SOURCE_REPOSITORY_ID,
    }
    if type(source["ownerId"]) is not int or type(source["repositoryId"]) is not int:
        raise ManifestError("manifest source IDs must be JSON integers")
    if source != expected_source:
        raise ManifestError("manifest source identity does not match the expected release source")

    release = require_exact_keys(root["release"], {"tag"}, "manifest.release")
    if release["tag"] != tag:
        raise ManifestError("manifest release tag does not match the selected tag")

    artifact = require_exact_keys(
        root["artifact"],
        {"mediaType", "name", "sha256", "size"},
        "manifest.artifact",
    )
    if artifact["mediaType"] != EIF_MEDIA_TYPE:
        raise ManifestError("manifest artifact media type is invalid")
    if artifact["name"] != expected_artifact_name(tag, environment):
        raise ManifestError("manifest artifact name is invalid")
    if not isinstance(artifact["sha256"], str) or not SHA256_RE.fullmatch(
        artifact["sha256"]
    ):
        raise ManifestError("manifest artifact SHA-256 is invalid")
    if type(artifact["size"]) is not int or artifact["size"] <= 0:
        raise ManifestError("manifest artifact size must be a positive integer")
    if artifact["sha256"] != sha256_file(eif_file):
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
    required_pcrs = measurements["requiredPcrs"]
    if (
        not isinstance(required_pcrs, list)
        or any(type(index) is not int for index in required_pcrs)
        or required_pcrs != [0, 1, 2]
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
        {"derivation", "flakeLockSha256", "system", "workflowRun"},
        "manifest.build",
    )
    if build["system"] != "nix":
        raise ManifestError("manifest build system must be nix")
    if build["derivation"] != f"eif-{environment}":
        raise ManifestError("manifest Nix derivation does not match the environment")
    if build["workflowRun"] != workflow_run:
        raise ManifestError("manifest workflow run does not match this release run")
    if (
        not isinstance(build["flakeLockSha256"], str)
        or not SHA256_RE.fullmatch(build["flakeLockSha256"])
        or build["flakeLockSha256"] != sha256_file(flake_lock)
    ):
        raise ManifestError("manifest flake.lock SHA-256 does not match flake.lock")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--environment", required=True, choices=("dev", "prod"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--workflow-run", required=True)
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

    if args.command == "generate":
        manifest = build_manifest(
            environment=args.environment,
            commit=args.commit,
            tag=args.tag,
            workflow_run=args.workflow_run,
            pcr_file=args.pcr,
            eif_file=args.eif,
            flake_lock=args.flake_lock,
        )
        validate_manifest(
            manifest,
            environment=args.environment,
            commit=args.commit,
            tag=args.tag,
            workflow_run=args.workflow_run,
            pcr_file=args.pcr,
            eif_file=args.eif,
            flake_lock=args.flake_lock,
        )
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
    validate_manifest(
        manifest,
        environment=args.environment,
        commit=args.commit,
        tag=args.tag,
        workflow_run=args.workflow_run,
        pcr_file=args.pcr,
        eif_file=args.eif,
        flake_lock=args.flake_lock,
    )
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
