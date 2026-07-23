# Nitro EIF Release Verification

OpenSecret publishes AWS Nitro EIF measurements at deliberate, manually approved
release boundaries. Each release contains deterministic manifests signed
keylessly by the tagged GitHub Actions workflow and recorded in Sigstore's
transparency infrastructure.

Sigstore is the provenance and transparency layer. It is not the artifact
transport, a software-safety oracle, a reproducibility proof, a revocation
service, or a source of truth for the newest approved release.

## Trust Layers

The complete trust decision has separate layers:

1. **AWS Nitro attestation** authenticates a fresh NSM document and binds its
   PCRs, caller nonce, and ephemeral session public key.
2. **OpenSecret release policy** decides which tagged manifest is approved for
   the expected `dev` or `prod` environment.
3. **Sigstore verification** proves the exact manifest bytes were signed by the
   authorized OpenSecret release workflow and included in the transparency log.
4. **Artifact verification** checks that the released EIF has the SHA-256 and
   size recorded in the verified manifest.
5. **Reproducibility** is established separately by rebuilding the same tagged,
   locked source and comparing the EIF digest and PCR tuple.
6. **Rollback and revocation policy** decides whether an older, correctly signed
   release remains acceptable.

All six concerns matter. A valid Sigstore bundle for an old release remains
cryptographically valid after that release is withdrawn.

## Manual Tagged Release

The release workflow is
`.github/workflows/release-nitro-eif.yml` (`Nitro EIF Release`).

Before using it, repository administrators must configure controls that cannot
be expressed in this repository:

- Protect the `production-release` GitHub Environment with required reviewers,
  prevent self-review, and restrict deployment refs to stable `v*` tags.
- Protect `v*` tags against unauthorized creation, update, and deletion.
- Require CODEOWNERS review for the release workflow and manifest generator.
- Enable immutable GitHub Releases.

To publish:

1. Merge the intended source to protected `master`.
2. Create an existing tag whose name matches exactly `vMAJOR.MINOR.PATCH`.
   Prerelease suffixes and leading zeroes are intentionally rejected.
3. Dispatch the workflow with that tag as its workflow ref:

   ```sh
   gh workflow run release-nitro-eif.yml \
     --repo OpenSecretCloud/opensecret \
     --ref vMAJOR.MINOR.PATCH
   ```

4. Approve the `production-release` deployment after reviewing the selected tag
   and commit.

Dispatching the workflow from `master` and supplying a tag as free-form text is
not supported. The selected workflow ref must itself be the tag so the Fulcio
certificate binds the signature to `refs/tags/vMAJOR.MINOR.PATCH`.

If a run must be retried, use **Re-run all jobs**. The manifests and artifact
names deliberately bind the run attempt, so **Re-run failed jobs** cannot reuse
successful outputs from an earlier attempt.

The workflow:

1. Validates the repository identity, owner identity, tag syntax, tag object,
   checked-out commit, and master ancestry.
2. Builds `eif-dev` and `eif-prod` from the exact tagged source on the ARM64 Nix
   runner.
3. Generates and independently revalidates one strict manifest per environment.
4. Uses Cosign 3.1.2 keyless signing to create a Sigstore v0.3 message-signature
   bundle over each manifest's exact bytes.
5. Creates SHA-256 checksums for the runtime assets.
6. Generates an additional GitHub SLSA/DSSE audit bundle covering both EIFs,
   both manifests, and the checksum file.
7. Transfers the signed release candidate to a separate publication job.
8. The publication job has no OIDC permission. It revalidates the manifests,
   independently verifies both Cosign bundles, checks every checksum and SLSA
   subject, attaches the explicit public assets to one draft GitHub Release,
   and only then publishes it.

OIDC signing permission exists only in the protected signing job, which has no
GitHub Release write permission. The publication job has GitHub Release write
permission but no OIDC permission. Ordinary pull-request and master builds
cannot mint release signatures.

## Release Assets

For tag `v1.2.3`, the published assets are:

```text
opensecret-v1.2.3-dev.eif
opensecret-nitro-v1.2.3-dev.manifest.json
opensecret-nitro-v1.2.3-dev.manifest.sigstore.json
opensecret-v1.2.3-prod.eif
opensecret-nitro-v1.2.3-prod.manifest.json
opensecret-nitro-v1.2.3-prod.manifest.sigstore.json
opensecret-nitro-v1.2.3.sha256
opensecret-nitro-v1.2.3.slsa.sigstore.json
```

GitHub Releases are an untrusted byte transport. Consumers authenticate a
manifest with its adjacent `manifest.sigstore.json` bundle before parsing or
using any manifest field.

The Cosign message-signature bundle is the cross-language runtime/update
contract. The SLSA/DSSE bundle is additional audit provenance and is not a
substitute for the simpler message-signature verification path.

## Manifest Contract

The schema identifier is:

```text
https://opensecret.cloud/attestations/nitro-eif-release/v1
```

The generator emits sorted, two-space-indented UTF-8 JSON followed by exactly
one line feed. It rejects duplicate keys, unknown fields, missing fields,
noncanonical tags and commits, malformed hashes, missing PCRs, and all-zero PCR
measurements. No wall-clock timestamp or mutable download URL is included.

A representative production manifest is:

```json
{
  "artifact": {
    "mediaType": "application/vnd.aws.nitro.eif",
    "name": "opensecret-v1.2.3-prod.eif",
    "sha256": "<64 lowercase hexadecimal characters>",
    "size": 123456789
  },
  "build": {
    "derivation": "eif-prod",
    "flakeLockSha256": "<64 lowercase hexadecimal characters>",
    "system": "nix",
    "workflowRun": "https://github.com/OpenSecretCloud/opensecret/actions/runs/123456789/attempts/1"
  },
  "environment": "prod",
  "measurements": {
    "algorithm": "sha384",
    "pcrs": {
      "0": "<96 lowercase hexadecimal characters>",
      "1": "<96 lowercase hexadecimal characters>",
      "2": "<96 lowercase hexadecimal characters>"
    },
    "requiredPcrs": [
      0,
      1,
      2
    ]
  },
  "release": {
    "tag": "v1.2.3"
  },
  "schema": "https://opensecret.cloud/attestations/nitro-eif-release/v1",
  "source": {
    "commit": "<40 lowercase hexadecimal characters>",
    "ownerId": 185423582,
    "ref": "refs/tags/v1.2.3",
    "repository": "OpenSecretCloud/opensecret",
    "repositoryId": 921901924
  }
}
```

PCR0 is not the raw EIF SHA-256. The manifest records both values and the full
PCR0/PCR1/PCR2 tuple.

## Consumer Verification Policy

A relying SDK or update tool must:

1. Obtain the manifest and message-signature bundle as untrusted bytes.
2. Require Sigstore bundle media type
   `application/vnd.dev.sigstore.bundle.v0.3+json`.
3. Load the Sigstore trust root independently rather than trusting roots
   supplied by the release transport.
4. Verify the Fulcio chain, transparency evidence, timestamp evidence, and the
   message signature over the exact manifest bytes.
5. Require issuer exactly `https://token.actions.githubusercontent.com`.
6. Require the signer identity to match exactly:

   ```text
   https://github.com/OpenSecretCloud/opensecret/.github/workflows/release-nitro-eif.yml@refs/tags/vMAJOR.MINOR.PATCH
   ```

7. Require the Fulcio extensions for the exact workflow name, repository,
   tag ref, source commit, `workflow_dispatch` trigger, GitHub-hosted runner,
   `production-release` environment, and run-invocation URI. The run-invocation
   URI must equal the manifest's immutable `build.workflowRun` attempt URL.
8. Strictly parse the already verified bytes and enforce repository ID
   `921901924`, owner ID `185423582`, source repository, tag/ref/commit,
   environment, schema, build derivation, and digest formats.
9. Apply a local approved-release or pinned-manifest policy. A Rekor inclusion
   proof does not mean "current" or "approved."
10. Verify a fresh AWS Nitro document and compare its full PCR0/PCR1/PCR2 tuple
   with the environment-specific manifest.
11. Only after every check succeeds, trust the attested ephemeral key and begin
    `/key_exchange`.

Production must never fall back to accepting a `dev` manifest. Verification
errors and missing evidence fail closed. A previously verified, pinned bundle
may be cached by digest so normal attestation does not require an online Rekor
lookup.

## Reproducibility

The tagged build uses the locked Nix flake, Cargo lockfile, pinned submodules,
and environment-specific EIF derivations. This is good reproducibility
groundwork, but the release signature still represents a builder claim.

Independent reproduction requires another trusted builder to check out the same
tag and locked inputs, run:

```sh
nix build '.?submodules=1#eif-dev'
nix build '.?submodules=1#eif-prod'
```

and compare the raw EIF SHA-256 plus PCR0/PCR1/PCR2 with the release manifest.
Two builds on the same runner are repeatability evidence, not independent
reproduction.

## Rollback and Revocation

Transparency logs retain old, valid records. They intentionally do not delete a
release when OpenSecret stops approving it.

Phase one consumers must ship or otherwise authenticate an explicit set of
approved manifest digests/tags. Moving to a different approved release is a new
SDK/update-policy decision. If dynamically updated authorization is needed
later, use a separate OpenSecret TUF repository; Sigstore's own TUF repository
distributes Sigstore trust roots, not OpenSecret release policy.

Never use a signed mutable `latest.json` as the sole current-release mechanism:
an attacker can replay an older correctly signed copy.

## Frozen Legacy PCR History

`pcrDevHistory.json` and `pcrProdHistory.json` are frozen, deprecated
compatibility data for already released clients. They are Git-hosted arrays
whose P-384 signatures cover only PCR0; they do not provide append-only
transparency or tagged CI provenance.

Do not append, rewrite, reorder, or remove legacy entries. The legacy `just
append-pcr-*` and `just update-pcr-*` commands now fail deliberately.
`pcr_sign.js` is deprecated. `pcr_verify.js` remains only for forensic
verification of old PCR0 signatures.

`pcrDev.json` and `pcrProd.json` remain temporary build-regression references
for the ordinary reproducible-build workflow. They are not release approval
metadata and must not be used by new clients.

## Local Generator Tests

The generator and verifier have no third-party Python dependencies:

```sh
python3 -m unittest discover -s scripts/tests -p 'test_*.py' -v
python3 -m py_compile \
  scripts/generate_nitro_release_manifest.py \
  scripts/tests/test_generate_nitro_release_manifest.py
```

Live Fulcio/Rekor publication cannot be tested without dispatching an approved
tagged release. The offline tests cover deterministic serialization, the full
contract, duplicate/unknown keys, malformed tags, zero PCRs, PCR substitution,
and EIF tampering.
