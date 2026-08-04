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
            "workflow_run": (
                "https://github.com/OpenSecretCloud/opensecret/actions/runs/"
                "123456789/attempts/1"
            ),
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
        second = manifest_tool.canonical_json_bytes(self.build())

        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        self.assertFalse(first.endswith(b"\n\n"))
        self.assertTrue(first.startswith(b'{\n  "artifact": {'))
        self.assertEqual(
            json.loads(first),
            manifest,
        )

    def test_manifest_covers_full_release_contract(self) -> None:
        manifest = self.build()

        self.assertEqual(manifest["schema"], manifest_tool.SCHEMA)
        self.assertEqual(manifest["environment"], "prod")
        self.assertEqual(
            manifest["source"],
            {
                "commit": "a" * 40,
                "ownerId": 185423582,
                "ref": "refs/tags/v1.2.3",
                "repository": "OpenSecretCloud/opensecret",
                "repositoryId": 921901924,
            },
        )
        self.assertEqual(manifest["measurements"]["requiredPcrs"], [0, 1, 2])
        self.assertEqual(
            manifest["measurements"]["pcrs"],
            {"0": "1" * 96, "1": "2" * 96, "2": "3" * 96},
        )
        self.assertEqual(manifest["build"]["derivation"], "eif-prod")
        self.assertEqual(
            manifest["artifact"]["sha256"],
            manifest_tool.sha256_file(self.eif),
        )
        self.assertEqual(manifest["artifact"]["size"], self.eif.stat().st_size)

    def test_verifier_accepts_generated_manifest(self) -> None:
        manifest_tool.validate_manifest(self.build(), **self.common)

    def test_verifier_rejects_eif_tampering(self) -> None:
        manifest = self.build()
        self.eif.write_bytes(b"tampered")

        with self.assertRaisesRegex(manifest_tool.ManifestError, "does not match the EIF"):
            manifest_tool.validate_manifest(manifest, **self.common)

    def test_verifier_rejects_unknown_manifest_fields(self) -> None:
        manifest = self.build()
        manifest["latest"] = True

        with self.assertRaisesRegex(manifest_tool.ManifestError, "unknown"):
            manifest_tool.validate_manifest(manifest, **self.common)

    def test_verifier_rejects_environment_confusion(self) -> None:
        manifest = self.build()

        with self.assertRaisesRegex(manifest_tool.ManifestError, "environment"):
            manifest_tool.validate_manifest(
                manifest,
                **{**self.common, "environment": "dev"},
            )

    def test_verifier_rejects_pcr_tuple_substitution(self) -> None:
        manifest = self.build()
        manifest["measurements"]["pcrs"]["1"] = "4" * 96

        with self.assertRaisesRegex(manifest_tool.ManifestError, "PCR tuple"):
            manifest_tool.validate_manifest(manifest, **self.common)

    def test_verifier_rejects_non_integer_contract_numbers(self) -> None:
        manifest = self.build()
        manifest["source"]["ownerId"] = float(manifest["source"]["ownerId"])

        with self.assertRaisesRegex(manifest_tool.ManifestError, "JSON integers"):
            manifest_tool.validate_manifest(manifest, **self.common)

        manifest = self.build()
        manifest["measurements"]["requiredPcrs"] = [False, True, 2]

        with self.assertRaisesRegex(manifest_tool.ManifestError, "exactly"):
            manifest_tool.validate_manifest(manifest, **self.common)

    def test_pcr_parser_rejects_duplicate_keys(self) -> None:
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

    def test_parser_rejects_non_finite_json_numbers(self) -> None:
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

        with self.assertRaisesRegex(manifest_tool.ManifestError, "must not be all zeroes"):
            self.build()

    def test_release_inputs_reject_noncanonical_tags(self) -> None:
        for tag in ("1.2.3", "v01.2.3", "v1.2.3-rc.1", "v1.2"):
            with self.subTest(tag=tag):
                with self.assertRaisesRegex(manifest_tool.ManifestError, "tag must match"):
                    manifest_tool.validate_release_inputs(
                        "prod",
                        "a" * 40,
                        tag,
                        self.common["workflow_run"],
                    )

    def test_release_inputs_require_an_exact_positive_run_attempt(self) -> None:
        for workflow_run in (
            "https://github.com/OpenSecretCloud/opensecret/actions/runs/123456789",
            "https://github.com/OpenSecretCloud/opensecret/actions/runs/123456789/attempts/0",
            "https://github.com/OpenSecretCloud/opensecret/actions/runs/123456789/attempts/1?retry=true",
        ):
            with self.subTest(workflow_run=workflow_run):
                with self.assertRaisesRegex(manifest_tool.ManifestError, "run-attempt"):
                    manifest_tool.validate_release_inputs(
                        "prod",
                        "a" * 40,
                        "v1.2.3",
                        workflow_run,
                    )

    def test_noncanonical_manifest_bytes_are_detectable(self) -> None:
        manifest = self.build()
        compact = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")

        self.assertNotEqual(compact, manifest_tool.canonical_json_bytes(manifest))


if __name__ == "__main__":
    unittest.main()
