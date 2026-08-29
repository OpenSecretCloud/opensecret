import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

from tuf.api.exceptions import DownloadHTTPError
from tuf.ngclient import FetcherInterface, Updater

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import manage_tuf_repository as tuf_repo


class LocalRepositoryFetcher(FetcherInterface):
    def __init__(self, public_tuf: Path) -> None:
        self.public_tuf = public_tuf

    def _fetch(self, url: str):
        path = urlparse(url).path
        if not path.startswith("/tuf/"):
            raise DownloadHTTPError("outside repository", 404)
        file_path = self.public_tuf / path.removeprefix("/tuf/")
        if not file_path.is_file():
            raise DownloadHTTPError("not found", 404)
        yield file_path.read_bytes()


class TufRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "state"
        self.public = self.root / "public/tuf"
        self.root_key = self.root / "offline-root.pem"
        self.online_key = self.root / "online.pem"
        tuf_repo.generate_key(self.root_key)
        tuf_repo.generate_key(self.online_key)

        self.policy = self.root / "builders.json"
        self.policy.write_bytes(
            tuf_repo.canonical_json(
                {
                    "schema": tuf_repo.BUILDER_POLICY_SCHEMA,
                    "builders": {
                        "opensecret-nitro-eif-github-actions": {
                            "certificateIdentityRegexp": (
                                "^https://github[.]com/OpenSecretCloud/opensecret/"
                                "[.]github/workflows/release-nitro-eif[.]yml@refs/tags/"
                                "v(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)[.](0|[1-9][0-9]*)$"
                            ),
                            "certificateOidcIssuer": "https://token.actions.githubusercontent.com",
                            "workflowName": "Nitro EIF Release",
                            "workflowRepository": "OpenSecretCloud/opensecret",
                            "workflowTrigger": "workflow_dispatch",
                        }
                    },
                }
            )
        )
        self.trusted_root = self.root / "trusted_root.json"
        self.trusted_root.write_bytes(
            tuf_repo.canonical_json(
                {
                    "mediaType": "application/vnd.dev.sigstore.trustedroot+json;version=0.1",
                    "certificateAuthorities": [],
                    "tlogs": [],
                    "ctlogs": [],
                }
            )
        )
        tuf_repo.bootstrap(
            self.repository,
            self.public,
            self.root_key,
            self.online_key,
            self.policy,
            self.trusted_root,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def release_files(
        self,
        version: str,
        environment: str = "prod",
        pcrs: dict[str, str] | None = None,
    ) -> tuple[Path, Path]:
        if pcrs is None:
            pcrs = {
                str(index): hashlib.sha384(
                    f"{version}:{environment}:{index}".encode()
                ).hexdigest()
                for index in range(3)
            }
        manifest = self.root / f"{version}-{environment}.manifest.json"
        manifest.write_bytes(
            tuf_repo.canonical_json(
                {
                    "schema": tuf_repo.MANIFEST_SCHEMA,
                    "component": tuf_repo.COMPONENT,
                    "environment": environment,
                    "release": {"version": version},
                    "source": {
                        "uri": "https://github.com/OpenSecretCloud/opensecret",
                        "path": ".",
                        "ref": f"refs/tags/v{version}",
                        "revision": {"algorithm": "git-sha1", "digest": "a" * 40},
                    },
                    "artifact": {
                        "name": f"opensecret-v{version}-{environment}.eif",
                        "mediaType": "application/vnd.aws.nitro.eif",
                        "size": 10,
                        "digests": {"sha256": "b" * 64},
                    },
                    "measurements": {
                        "algorithm": "sha384",
                        "requiredPcrs": [0, 1, 2],
                        "pcrs": pcrs,
                    },
                    "build": {
                        "system": "nix",
                        "builderId": "opensecret-nitro-eif-github-actions",
                        "derivation": f".#eif-{environment}",
                        "flakeLockSha256": "c" * 64,
                        "runUri": "https://github.com/OpenSecretCloud/opensecret/actions/runs/1/attempts/1",
                    },
                }
            )
        )
        bundle = self.root / f"{version}-{environment}.sigstore.json"
        bundle.write_bytes(
            tuf_repo.canonical_json(
                {
                    "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
                    "verificationMaterial": {},
                    "messageSignature": {},
                }
            )
        )
        return manifest, bundle

    def promote(
        self,
        version: str,
        phase: str = "rollout",
        environment: str = "prod",
        pcrs: dict[str, str] | None = None,
    ) -> None:
        manifest, bundle = self.release_files(version, environment, pcrs)
        tuf_repo.promote(
            self.repository,
            self.public,
            self.online_key,
            environment,
            version,
            phase,
            manifest,
            bundle,
        )

    def current_channel(self, environment: str = "prod") -> dict:
        root = tuf_repo.load_root_chain(self.repository)
        targets, _, _ = tuf_repo.load_current(self.repository, root.signed)
        return json.loads(
            tuf_repo.authenticated_target(
                self.repository, targets.signed, f"channels/{environment}.json"
            )
        )

    @staticmethod
    def file_fetcher(public_tuf: Path):
        def fetch(relative: str) -> bytes:
            path = public_tuf / relative
            if not path.is_file():
                raise tuf_repo.LiveFileNotFound(relative)
            return path.read_bytes()

        return fetch

    def generate_signer(self, name: str):
        path = self.root / f"{name}.pem"
        tuf_repo.generate_key(path)
        return tuf_repo.load_signer(path)

    def write_root_transition(
        self,
        version: int,
        previous_root_signer,
        root_signer,
        online_signer,
        *,
        root_aliases: tuple[tuple[str, object], ...] = (),
        online_alias: tuple[str, object] | None = None,
    ):
        previous = tuf_repo.load_metadata(
            self.repository / "metadata" / f"{version - 1}.root.json",
            tuf_repo.Root,
        )
        current = tuf_repo.Metadata.from_bytes(previous.to_bytes())
        current.signed.version = version
        current.signed.keys[root_signer.public_key.keyid] = root_signer.public_key
        root_keyids = [root_signer.public_key.keyid]
        for alias_keyid, alias_key in root_aliases:
            current.signed.keys[alias_keyid] = alias_key
            root_keyids.append(alias_keyid)
        current.signed.roles["root"].keyids = root_keyids
        current.signed.roles["root"].threshold = 1

        if online_alias is None:
            online_keyid = online_signer.public_key.keyid
            current.signed.keys[online_keyid] = online_signer.public_key
        else:
            online_keyid, online_key = online_alias
            current.signed.keys[online_keyid] = online_key
        for role_name in tuf_repo.ONLINE_ROLES:
            current.signed.roles[role_name].keyids = [online_keyid]
            current.signed.roles[role_name].threshold = 1

        current.signatures.clear()
        current.sign(previous_root_signer)
        if root_signer.public_key.keyid != previous_root_signer.public_key.keyid:
            current.sign(root_signer, append=True)
        path = self.repository / "metadata" / f"{version}.root.json"
        path.write_bytes(current.to_bytes())
        return current

    def test_bootstrap_separates_root_material_and_allows_shared_online_roles(
        self,
    ) -> None:
        root = tuf_repo.load_root_chain(self.repository)
        online = tuf_repo.load_signer(self.online_key)
        self.assertTrue(root.signed.consistent_snapshot)
        self.assertNotIn(online.public_key.keyid, root.signed.roles["root"].keyids)
        for role in tuf_repo.ONLINE_ROLES:
            self.assertEqual(root.signed.roles[role].threshold, 1)
            self.assertEqual(
                set(root.signed.roles[role].keyids), {online.public_key.keyid}
            )
        self.assertFalse(
            any(path.suffix == ".pem" for path in self.repository.rglob("*"))
        )

        with self.assertRaisesRegex(
            tuf_repo.RepositoryError, "root and online signing key material must differ"
        ):
            tuf_repo.bootstrap(
                self.root / "same-key-state",
                self.root / "same-key-public",
                self.root_key,
                self.root_key,
                self.policy,
                self.trusted_root,
            )

    def test_parsed_root_chain_rejects_root_material_under_online_alias_key_ids(
        self,
    ) -> None:
        root_v1_path = self.repository / "metadata/1.root.json"
        root_signer = tuf_repo.load_signer(self.root_key)

        for role_name in tuf_repo.ONLINE_ROLES:
            with self.subTest(role=role_name):
                root_v2 = tuf_repo.Metadata.from_bytes(root_v1_path.read_bytes())
                root_v2.signed.version = 2
                root_keyid = next(iter(root_v2.signed.roles["root"].keyids))
                alias_keyid = "0" * 64
                self.assertNotIn(alias_keyid, root_v2.signed.keys)
                root_v2.signed.keys[alias_keyid] = root_v2.signed.keys[root_keyid]
                root_v2.signed.roles[role_name].keyids = [alias_keyid]
                root_v2.signatures.clear()
                root_v2.sign(root_signer)
                root_v2_path = self.repository / "metadata/2.root.json"
                root_v2_path.write_bytes(root_v2.to_bytes())
                try:
                    with self.assertRaisesRegex(
                        tuf_repo.RepositoryError,
                        f"root-role key material with online {role_name}",
                    ):
                        tuf_repo.load_root_chain(self.repository)
                finally:
                    root_v2_path.unlink()

    def test_root_chain_allows_fresh_root_and_shared_online_key_rotation(self) -> None:
        old_root = tuf_repo.load_signer(self.root_key)
        fresh_root = self.generate_signer("fresh-root")
        fresh_online = self.generate_signer("fresh-online")
        self.write_root_transition(2, old_root, fresh_root, fresh_online)

        current = tuf_repo.load_root_chain(self.repository)
        self.assertEqual(current.signed.version, 2)
        for role_name in tuf_repo.ONLINE_ROLES:
            self.assertEqual(
                current.signed.roles[role_name].keyids,
                [fresh_online.public_key.keyid],
            )

    def test_root_chain_rejects_historical_root_material_moved_online_by_alias(
        self,
    ) -> None:
        old_root = tuf_repo.load_signer(self.root_key)
        old_online = tuf_repo.load_signer(self.online_key)
        fresh_root = self.generate_signer("fresh-root")
        self.write_root_transition(2, old_root, fresh_root, old_online)
        self.write_root_transition(
            3,
            fresh_root,
            fresh_root,
            old_online,
            online_alias=("1" * 64, old_root.public_key),
        )

        with self.assertRaisesRegex(
            tuf_repo.RepositoryError,
            "historical root key material to an online role",
        ):
            tuf_repo.load_root_chain(self.repository)

    def test_root_chain_rejects_historical_online_material_moved_to_root_by_alias(
        self,
    ) -> None:
        old_root = tuf_repo.load_signer(self.root_key)
        old_online = tuf_repo.load_signer(self.online_key)
        fresh_root = self.generate_signer("fresh-root")
        fresh_online = self.generate_signer("fresh-online")
        self.write_root_transition(2, old_root, fresh_root, fresh_online)
        self.write_root_transition(
            3,
            fresh_root,
            fresh_root,
            fresh_online,
            root_aliases=(("2" * 64, old_online.public_key),),
        )

        with self.assertRaisesRegex(
            tuf_repo.RepositoryError,
            "historical online key material to the root role",
        ):
            tuf_repo.load_root_chain(self.repository)

    def test_root_history_cannot_exceed_late_client_rotation_limit(self) -> None:
        root_v1_path = self.repository / "metadata/1.root.json"
        root_signer = tuf_repo.load_signer(self.root_key)
        for version in range(2, tuf_repo.MAX_ROOT_VERSIONS + 2):
            current = tuf_repo.Metadata.from_bytes(root_v1_path.read_bytes())
            current.signed.version = version
            current.signatures.clear()
            current.sign(root_signer)
            (self.repository / "metadata" / f"{version}.root.json").write_bytes(
                current.to_bytes()
            )

        with self.assertRaisesRegex(
            tuf_repo.RepositoryError,
            f"{tuf_repo.MAX_ROOT_VERSIONS}-version client limit",
        ):
            tuf_repo.load_root_chain(self.repository)

    def test_workflow_shaped_python_cli_verify_and_refresh(self) -> None:
        script = SCRIPT_DIR / "manage_tuf_repository.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "verify",
                "--repository",
                str(self.repository),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(script),
                "refresh",
                "--repository",
                str(self.repository),
                "--output",
                str(self.public),
                "--online-key",
                str(self.online_key),
            ],
            check=True,
        )

    def test_rollout_keeps_at_most_two_active_releases_and_finalize_keeps_one(
        self,
    ) -> None:
        self.promote("1.0.0")
        self.promote("1.1.0")
        self.promote("1.2.0")
        channel = self.current_channel()
        self.assertEqual(channel["sequence"], 3)
        self.assertEqual(len(channel["active"]), 2)
        self.assertTrue(
            channel["active"][0]["manifestTarget"].startswith("releases/1.1.0/")
        )
        self.assertTrue(
            channel["active"][1]["manifestTarget"].startswith("releases/1.2.0/")
        )
        retired_digest = channel["active"][0]["manifestSha256"]
        self.promote("1.2.0", phase="finalize")
        channel = self.current_channel()
        self.assertEqual(channel["sequence"], 4)
        self.assertEqual(len(channel["active"]), 1)
        self.assertTrue(
            channel["active"][0]["manifestTarget"].startswith("releases/1.2.0/")
        )
        self.assertFalse(
            (self.repository / "targets/releases/1.1.0/prod/manifest.json").exists()
        )
        self.assertTrue(
            (
                self.repository
                / "published-targets/releases/1.1.0/prod"
                / f"{retired_digest}.manifest.json"
            ).is_file()
        )

    def test_duplicate_tuple_requires_finalize_and_revoke_remains_available(
        self,
    ) -> None:
        duplicate = {"0": "1" * 96, "1": "2" * 96, "2": "3" * 96}
        self.promote("1.3.0", pcrs=duplicate)
        with self.assertRaisesRegex(tuf_repo.RepositoryError, "use finalize"):
            self.promote("1.3.1", pcrs=duplicate)
        self.promote("1.3.1", phase="finalize", pcrs=duplicate)
        channel = self.current_channel()
        self.assertEqual(len(channel["active"]), 1)
        self.assertIn("releases/1.3.1/", channel["active"][0]["manifestTarget"])
        tuf_repo.revoke(self.repository, self.public, self.online_key, "prod")
        self.assertEqual(self.current_channel()["active"], [])

    def test_public_tree_uses_hashed_targets_and_timestamp_references_existing_bytes(
        self,
    ) -> None:
        self.promote("2.0.0")
        channel = self.current_channel()
        first_channel_bytes = (
            self.repository / "targets/channels/prod.json"
        ).read_bytes()
        first_channel_digest = tuf_repo.sha256_bytes(first_channel_bytes)
        self.promote("2.1.0")
        digest = channel["active"][0]["manifestSha256"]
        self.assertTrue(
            (
                self.public / "targets/releases/2.0.0/prod" / f"{digest}.manifest.json"
            ).is_file()
        )
        self.assertEqual(
            (
                self.public / "targets/channels" / f"{first_channel_digest}.prod.json"
            ).read_bytes(),
            first_channel_bytes,
        )
        timestamp = tuf_repo.load_metadata(
            self.public / "metadata/timestamp.json", tuf_repo.Timestamp
        )
        remaining = timestamp.signed.expires - tuf_repo.utcnow()
        self.assertGreater(remaining, timedelta(hours=46))
        self.assertLessEqual(remaining, timedelta(hours=47))
        snapshot_name = f"{timestamp.signed.snapshot_meta.version}.snapshot.json"
        self.assertTrue((self.public / "metadata" / snapshot_name).is_file())

    def test_tampering_with_authenticated_policy_fails_closed(self) -> None:
        self.policy.write_text("{}\n", encoding="utf-8")
        (self.repository / "targets/policy/builders.json").write_text(
            "{}\n", encoding="utf-8"
        )
        manifest, bundle = self.release_files("3.0.0")
        with self.assertRaisesRegex(tuf_repo.RepositoryError, "does not match signed"):
            tuf_repo.promote(
                self.repository,
                self.public,
                self.online_key,
                "prod",
                "3.0.0",
                "rollout",
                manifest,
                bundle,
            )

    def test_standard_python_tuf_client_resolves_consistent_snapshot_target(
        self,
    ) -> None:
        self.promote("3.1.4")
        updater = Updater(
            str(self.root / "client-metadata"),
            "https://attestations.trymaple.ai/tuf/metadata/",
            str(self.root / "client-targets"),
            "https://attestations.trymaple.ai/tuf/targets/",
            fetcher=LocalRepositoryFetcher(self.public),
            bootstrap=(self.public / "metadata/1.root.json").read_bytes(),
        )
        updater.refresh()
        target = updater.get_targetinfo("channels/prod.json")
        self.assertIsNotNone(target)
        downloaded = Path(updater.download_target(target))
        self.assertEqual(json.loads(downloaded.read_bytes())["sequence"], 1)

    def test_late_root_v1_client_traverses_retained_v2_and_v3(self) -> None:
        root_signer = tuf_repo.load_signer(self.root_key)
        for version in (2, 3):
            previous = tuf_repo.load_metadata(
                self.repository / "metadata" / f"{version - 1}.root.json",
                tuf_repo.Root,
            )
            current = tuf_repo.Metadata.from_bytes(previous.to_bytes())
            current.signed.version = version
            current.signatures.clear()
            current.sign(root_signer)
            (self.repository / "metadata" / f"{version}.root.json").write_bytes(
                current.to_bytes()
            )

        tuf_repo.render_public(self.repository, self.public)
        self.assertEqual(
            {
                path.name
                for path in (self.public / "metadata").glob("*.root.json")
            },
            {"1.root.json", "2.root.json", "3.root.json"},
        )

        client_metadata = self.root / "late-client-metadata"
        updater = Updater(
            str(client_metadata),
            "https://attestations.trymaple.ai/tuf/metadata/",
            str(self.root / "late-client-targets"),
            "https://attestations.trymaple.ai/tuf/targets/",
            fetcher=LocalRepositoryFetcher(self.public),
            bootstrap=(self.public / "metadata/1.root.json").read_bytes(),
        )
        updater.refresh()
        trusted_root = tuf_repo.Metadata.from_bytes(
            (client_metadata / "root.json").read_bytes()
        )
        self.assertEqual(trusted_root.signed.version, 3)

    def test_historical_metadata_and_hashed_targets_are_required(self) -> None:
        self.promote("5.0.0")
        first_channel = (self.repository / "targets/channels/prod.json").read_bytes()
        first_digest = tuf_repo.sha256_bytes(first_channel)
        self.promote("5.1.0")
        archived = (
            self.repository / "published-targets/channels" / f"{first_digest}.prod.json"
        )
        archived.unlink()
        with self.assertRaisesRegex(tuf_repo.RepositoryError, "historical target"):
            tuf_repo.validate_repository(self.repository)

    def test_timestamp_history_is_contiguous_signed_and_immutable(self) -> None:
        self.promote("5.1.1")
        history = self.repository / "timestamp-history"
        self.assertEqual(
            {path.name for path in history.iterdir()},
            {"1.timestamp.json", "2.timestamp.json"},
        )
        (history / "1.timestamp.json").unlink()
        with self.assertRaisesRegex(tuf_repo.RepositoryError, "timestamp history"):
            tuf_repo.validate_repository(self.repository)

    def test_live_timestamp_must_equal_its_archived_envelope(self) -> None:
        old_public = self.root / "old-public"
        shutil.copytree(self.public, old_public)
        self.promote("5.1.2")
        fork = tuf_repo.load_metadata(
            old_public / "metadata/timestamp.json", tuf_repo.Timestamp
        )
        fork.signed.expires += timedelta(minutes=1)
        fork.signatures.clear()
        fork.sign(tuf_repo.load_signer(self.online_key))
        (old_public / "metadata/timestamp.json").write_bytes(fork.to_bytes())
        with self.assertRaisesRegex(tuf_repo.RepositoryError, "immutable.*history"):
            tuf_repo.verify_live_state(
                self.repository,
                "https://unused.example/tuf",
                fetcher=self.file_fetcher(old_public),
            )

    def test_equal_live_timestamp_still_requires_all_referenced_target_bytes(
        self,
    ) -> None:
        regular = self.file_fetcher(self.public)

        def missing_target(relative: str) -> bytes:
            if relative.startswith("targets/"):
                raise tuf_repo.LiveFileNotFound(relative)
            return regular(relative)

        with self.assertRaisesRegex(tuf_repo.RepositoryError, "references missing"):
            tuf_repo.verify_live_state(
                self.repository,
                "https://unused.example/tuf",
                fetcher=missing_target,
            )

    def test_live_high_water_allows_equal_or_historical_subset_and_rejects_ahead(
        self,
    ) -> None:
        pristine_state = self.root / "pristine-state"
        pristine_public = self.root / "pristine-public"
        shutil.copytree(self.repository, pristine_state)
        shutil.copytree(self.public, pristine_public)

        tuf_repo.verify_live_state(
            self.repository,
            "https://unused.example/tuf",
            fetcher=self.file_fetcher(self.public),
        )
        self.promote("5.2.0")
        tuf_repo.verify_live_state(
            self.repository,
            "https://unused.example/tuf",
            fetcher=self.file_fetcher(pristine_public),
        )
        with self.assertRaisesRegex(tuf_repo.RepositoryError, "ahead"):
            tuf_repo.verify_live_state(
                pristine_state,
                "https://unused.example/tuf",
                fetcher=self.file_fetcher(self.public),
            )

    def test_missing_live_state_requires_explicit_pristine_bootstrap(self) -> None:
        def missing(relative: str) -> bytes:
            raise tuf_repo.LiveFileNotFound(relative)

        with self.assertRaisesRegex(tuf_repo.RepositoryError, "explicit publication"):
            tuf_repo.verify_live_state(
                self.repository, "https://unused.example/tuf", fetcher=missing
            )
        tuf_repo.verify_live_state(
            self.repository,
            "https://unused.example/tuf",
            allow_unpublished=True,
            fetcher=missing,
        )
        # A first deploy can fail after its refresh state was persisted. A retry
        # remains safe while no channel or release target has ever been added.
        tuf_repo.refresh(self.repository, self.public, self.online_key)
        tuf_repo.verify_live_state(
            self.repository,
            "https://unused.example/tuf",
            allow_unpublished=True,
            fetcher=missing,
        )
        self.promote("5.3.0")
        with self.assertRaisesRegex(tuf_repo.RepositoryError, "never-activated"):
            tuf_repo.verify_live_state(
                self.repository,
                "https://unused.example/tuf",
                allow_unpublished=True,
                fetcher=missing,
            )

    def test_unsigned_and_symlinked_state_is_rejected_before_signing(self) -> None:
        unsigned = self.repository / "targets/unsigned.json"
        unsigned.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(tuf_repo.RepositoryError, "unsigned"):
            tuf_repo.validate_repository(self.repository)
        unsigned.unlink()
        leak = self.repository / "targets/leak"
        leak.symlink_to(self.online_key)
        with self.assertRaisesRegex(tuf_repo.RepositoryError, "symlink"):
            tuf_repo.validate_repository(self.repository)

    def test_revoke_publishes_an_empty_fail_closed_channel(self) -> None:
        self.promote("5.4.0")
        tuf_repo.revoke(self.repository, self.public, self.online_key, "prod")
        channel = self.current_channel()
        self.assertEqual(channel["sequence"], 2)
        self.assertEqual(channel["active"], [])

    def test_release_target_paths_are_immutable(self) -> None:
        self.promote("5.5.0")
        manifest, bundle = self.release_files("5.5.0")
        value = json.loads(manifest.read_text(encoding="utf-8"))
        value["artifact"]["digests"]["sha256"] = "d" * 64
        manifest.write_bytes(tuf_repo.canonical_json(value))
        with self.assertRaisesRegex(
            tuf_repo.RepositoryError, "immutable release target"
        ):
            tuf_repo.promote(
                self.repository,
                self.public,
                self.online_key,
                "prod",
                "5.5.0",
                "rollout",
                manifest,
                bundle,
            )

    def test_wrong_online_key_and_noncanonical_manifest_fail_closed(self) -> None:
        wrong = self.root / "wrong.pem"
        tuf_repo.generate_key(wrong)
        manifest, bundle = self.release_files("4.0.0")
        with self.assertRaisesRegex(tuf_repo.RepositoryError, "sole threshold-1"):
            tuf_repo.promote(
                self.repository,
                self.public,
                wrong,
                "prod",
                "4.0.0",
                "rollout",
                manifest,
                bundle,
            )
        value = json.loads(manifest.read_text(encoding="utf-8"))
        manifest.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(tuf_repo.RepositoryError, "canonical"):
            tuf_repo.promote(
                self.repository,
                self.public,
                self.online_key,
                "prod",
                "4.0.0",
                "rollout",
                manifest,
                bundle,
            )

        long_version = ("1" * 129) + ".2.3"
        long_manifest, long_bundle = self.release_files(long_version)
        with self.assertRaisesRegex(tuf_repo.RepositoryError, "128-byte"):
            tuf_repo.promote(
                self.repository,
                self.public,
                self.online_key,
                "prod",
                long_version,
                "rollout",
                long_manifest,
                long_bundle,
            )


if __name__ == "__main__":
    unittest.main()
