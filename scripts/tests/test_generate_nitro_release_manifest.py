import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import generate_nitro_release_manifest as manifest_tool


class NitroReleaseManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.eif = self.root / "opensecret-v1.2.3-prod.eif"
        self.eif.write_bytes(b"deterministic-eif-test-fixture")
        self.flake_lock = self.root / "flake.lock"
        self.flake_lock.write_text(
            '{"nodes":{},"root":"root","version":7}\n', encoding="utf-8"
        )
        self.pcr = self.root / "pcr.json"
        self.pcr_values = {
            "HashAlgorithm": "Sha384 { ... }",
            "PCR0": "1" * 96,
            "PCR1": "2" * 96,
            "PCR2": "3" * 96,
        }
        self.pcr.write_text(json.dumps(self.pcr_values), encoding="utf-8")
        self.common = {
            "environment": "prod",
            "commit": "a" * 40,
            "tag": "v1.2.3",
            "source_uri": "https://github.com/OpenSecretCloud/opensecret",
            "source_path": ".",
            "builder_id": "opensecret-nitro-eif-github-actions",
            "run_uri": "https://github.com/OpenSecretCloud/opensecret/actions/runs/123/attempts/1",
            "pcr_file": self.pcr,
            "eif_file": self.eif,
            "flake_lock": self.flake_lock,
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def build(self):
        return manifest_tool.build_manifest(**self.common)

    def test_generation_is_canonical_and_deterministic(self) -> None:
        manifest = self.build()
        first = manifest_tool.canonical_json_bytes(manifest)
        self.assertEqual(first, manifest_tool.canonical_json_bytes(self.build()))
        self.assertTrue(first.endswith(b"\n"))
        self.assertFalse(first.endswith(b"\n\n"))
        self.assertEqual(json.loads(first), manifest)

    def test_manifest_is_repository_neutral_and_covers_release(self) -> None:
        manifest = self.build()
        self.assertEqual(
            manifest["schema"],
            "https://attestations.trymaple.ai/schemas/nitro-eif-release/v1",
        )
        self.assertEqual(manifest["component"], "opensecret-backend")
        self.assertEqual(manifest["release"], {"version": "1.2.3"})
        self.assertEqual(
            manifest["source"],
            {
                "uri": self.common["source_uri"],
                "path": ".",
                "ref": "refs/tags/v1.2.3",
                "revision": {"algorithm": "git-sha1", "digest": "a" * 40},
            },
        )
        self.assertEqual(
            manifest["artifact"]["digests"]["sha256"],
            manifest_tool.sha256_file(self.eif),
        )
        self.assertEqual(manifest["measurements"]["requiredPcrs"], [0, 1, 2])
        self.assertEqual(manifest["build"]["builderId"], self.common["builder_id"])
        self.assertEqual(manifest["build"]["derivation"], ".#eif-prod")

    def test_verifier_accepts_generated_manifest(self) -> None:
        manifest_tool.validate_manifest(self.build(), **self.common)

    def test_verifier_rejects_eif_tampering(self) -> None:
        manifest = self.build()
        self.eif.write_bytes(b"tampered")
        with self.assertRaisesRegex(
            manifest_tool.ManifestError, "does not match the EIF"
        ):
            manifest_tool.validate_manifest(manifest, **self.common)

    def test_verifier_rejects_unknown_fields(self) -> None:
        manifest = self.build()
        manifest["latest"] = True
        with self.assertRaisesRegex(manifest_tool.ManifestError, "unknown"):
            manifest_tool.validate_manifest(manifest, **self.common)

    def test_verifier_rejects_source_substitution(self) -> None:
        manifest = self.build()
        manifest["source"]["uri"] = "https://example.com/source"
        with self.assertRaisesRegex(manifest_tool.ManifestError, "source"):
            manifest_tool.validate_manifest(manifest, **self.common)

    def test_verifier_rejects_pcr_tuple_substitution(self) -> None:
        manifest = self.build()
        manifest["measurements"]["pcrs"]["1"] = "4" * 96
        with self.assertRaisesRegex(manifest_tool.ManifestError, "PCR tuple"):
            manifest_tool.validate_manifest(manifest, **self.common)

    def test_verifier_rejects_non_integer_contract_numbers(self) -> None:
        manifest = self.build()
        manifest["artifact"]["size"] = float(manifest["artifact"]["size"])
        with self.assertRaisesRegex(manifest_tool.ManifestError, "positive integer"):
            manifest_tool.validate_manifest(manifest, **self.common)
        manifest = self.build()
        manifest["measurements"]["requiredPcrs"] = [False, True, 2]
        with self.assertRaisesRegex(manifest_tool.ManifestError, "exactly"):
            manifest_tool.validate_manifest(manifest, **self.common)

    def test_pcr_parser_rejects_duplicate_and_nonfinite_values(self) -> None:
        self.pcr.write_text(
            '{"HashAlgorithm":"Sha384","PCR0":"'
            + ("1" * 96)
            + '","PCR0":"'
            + ("2" * 96)
            + '","PCR1":"'
            + ("2" * 96)
            + '","PCR2":"'
            + ("3" * 96)
            + '"}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(manifest_tool.ManifestError, "duplicate JSON key"):
            self.build()
        self.pcr.write_text(
            '{"HashAlgorithm":"Sha384","PCR0":NaN,"PCR1":"'
            + ("2" * 96)
            + '","PCR2":"'
            + ("3" * 96)
            + '"}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(manifest_tool.ManifestError, "non-finite"):
            self.build()

    def test_pcr_parser_rejects_zero_measurement(self) -> None:
        values = copy.deepcopy(self.pcr_values)
        values["PCR2"] = "0" * 96
        self.pcr.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(
            manifest_tool.ManifestError, "must not be all zeroes"
        ):
            self.build()

    def test_inputs_reject_unsafe_identifiers_and_uris(self) -> None:
        cases = [
            {"tag": "v01.2.3"},
            {"tag": "v" + ("1" * 129) + ".2.3"},
            {"source_uri": "git@github.com:OpenSecretCloud/opensecret"},
            {"source_path": "../opensecret"},
            {"builder_id": "Upper Case"},
            {"run_uri": "https://user:password@example.com/run"},
        ]
        for replacement in cases:
            with self.subTest(replacement=replacement):
                with self.assertRaises(manifest_tool.ManifestError):
                    manifest_tool.build_manifest(**{**self.common, **replacement})

    def test_noncanonical_manifest_bytes_are_detectable(self) -> None:
        manifest = self.build()
        compact = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
        self.assertNotEqual(compact, manifest_tool.canonical_json_bytes(manifest))


if __name__ == "__main__":
    unittest.main()
