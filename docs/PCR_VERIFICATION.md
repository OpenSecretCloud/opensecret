# Nitro EIF Attestation Trust

OpenSecret now separates two questions that the old signed-PCR JSON path mixed
together:

1. **Did an authorized builder produce and publicly attest these exact release
   bytes and PCR measurements?** Sigstore answers this.
2. **Which of those valid releases may a client trust now?** TUF metadata served
   from `https://attestations.trymaple.ai` answers this.

AWS Nitro attestation then answers the live question: **is the enclave I am
connecting to currently presenting the PCR tuple and session key that I
expected?**

No release is activated by merging this code, creating a Git tag, or producing
a Sigstore bundle. Activation is a separate manual, protected TUF promotion.
The production TUF root is deliberately absent until an operator bootstraps it
offline.

## The complete trust chain

```mermaid
flowchart LR
    source["Reviewed source + flake.lock"] --> build["Manual tagged Nix build"]
    build --> eif["EIF + PCR0/PCR1/PCR2"]
    eif --> manifest["Canonical release manifest"]
    oidc["GitHub Actions OIDC"] --> sigstore["Fulcio certificate + Cosign signature"]
    manifest --> sigstore
    sigstore --> rekor["Rekor transparency evidence"]
    rekor --> bundle["Portable Sigstore bundle"]

    bundle --> promotion["Protected manual promotion"]
    manifest --> promotion
    builderPolicy["TUF-authenticated builder policy"] --> promotion
    trustedRoot["TUF-authenticated Sigstore trusted root"] --> promotion
    promotion --> tuf["New signed TUF targets/snapshot/timestamp"]
    tuf --> worker["Atomic Cloudflare Worker deployment"]

    embeddedRoot["SDK-embedded TUF root"] --> client["Rust / TypeScript client"]
    worker --> client
    client --> tuple["Authorized prod/dev PCR tuple"]
    enclave["Fresh AWS Nitro document"] --> tuple
    tuple --> decision{"Exact PCR0 + PCR1 + PCR2 match?"}
    decision -->|yes| key["Trust attested session key"]
    decision -->|no| reject["Fail closed"]
```

The append-only Rekor log makes a signing event durable and auditable. It does
not declare that release current. TUF supplies versioned, expiring, signed
authorization and rollback protection. Cloudflare is only the byte transport:
all clients authenticate the repository bytes through TUF. Rust also performs
local cryptographic verification of each portable Sigstore bundle. Browser
TypeScript authenticates the exact manifest, bundle, policy, and trusted-root
bytes through TUF; the protected promotion is the component that
cryptographically verifies Sigstore for that browser path.

## Before and after

| Concern | Previous path | New path |
| --- | --- | --- |
| Published statement | A long-lived P-384 key signed PCR0 only | A keyless Sigstore signature covers a canonical manifest containing source revision, locked Nix build data, EIF digest, and PCR0/PCR1/PCR2 |
| Discovery | Client fetched mutable JSON from `raw.githubusercontent.com` | Client starts at `attestations.trymaple.ai/tuf` and follows authenticated TUF metadata |
| Current release | Every valid history entry remained accepted | A monotonic prod/dev channel authorizes at most two active releases |
| Builder identity | Implied by repository history and the custom signing key | Exact Fulcio issuer, workflow identity, repository, trigger, and workflow name live in a TUF-authenticated builder policy |
| Root of trust | Custom PCR public key shipped by old clients | Standard TUF root metadata is pinned by new clients |
| PCR decision | PCR0-only legacy approval | Same-release, same-environment PCR0/PCR1/PCR2 tuple |
| Runtime network | GitHub was in the dependency path | GitHub, Fulcio, and Rekor are not contacted during connection. Rust verifies portable evidence locally; browser TypeScript consumes the exact TUF-authenticated evidence already verified by protected promotion |
| Rollout | Updating history immediately enlarged trust | Candidate creation and activation are separate protected actions |

Sigstore does not store values “inside Rekor” as a mutable database lookup.
The PCRs are fields in the exact manifest bytes that Cosign signs. The portable
bundle carries the certificate, signature, and transparency evidence needed to
verify those bytes later. TUF points clients to the exact manifest and bundle
that are currently authorized.

## Release manifest contract

The canonical manifest schema is repository-neutral:

```json
{
  "schema": "https://attestations.trymaple.ai/schemas/nitro-eif-release/v1",
  "component": "opensecret-backend",
  "environment": "prod",
  "release": { "version": "1.2.3" },
  "source": {
    "uri": "https://github.com/OpenSecretCloud/opensecret",
    "path": ".",
    "ref": "refs/tags/v1.2.3",
    "revision": {
      "algorithm": "git-sha1",
      "digest": "0123456789abcdef0123456789abcdef01234567"
    }
  },
  "artifact": {
    "name": "opensecret-v1.2.3-prod.eif",
    "mediaType": "application/vnd.aws.nitro.eif",
    "size": 123456,
    "digests": { "sha256": "..." }
  },
  "measurements": {
    "algorithm": "sha384",
    "requiredPcrs": [0, 1, 2],
    "pcrs": { "0": "...", "1": "...", "2": "..." }
  },
  "build": {
    "system": "nix",
    "builderId": "opensecret-nitro-eif-github-actions",
    "derivation": ".#eif-prod",
    "flakeLockSha256": "...",
    "runUri": "https://github.com/OpenSecretCloud/opensecret/actions/runs/..."
  }
}
```

The current source URI happens to be GitHub because that is where the build runs
today. It is data, not a fixed schema assumption. Moving the repository or
builder later requires publishing a newly authenticated builder-policy entry,
not changing the client protocol.

## TUF repository contract

The client base URL is `https://attestations.trymaple.ai/tuf`. The
repository uses standard top-level TUF roles and consistent snapshots:

```text
/tuf/metadata/
  1.root.json
  N.root.json
  N.targets.json
  N.snapshot.json
  timestamp.json
/tuf/targets/
  <sha256>.name                       # hash-prefixed immutable targets
  channels/<sha256>.prod.json
  channels/<sha256>.dev.json
  policy/<sha256>.builders.json
  sigstore/<sha256>.trusted_root.json
  releases/<version>/<env>/<sha256>.manifest.json
  releases/<version>/<env>/<sha256>.manifest.sigstore.json
```

Logical target names appear inside signed TUF metadata. Consistent-snapshot
clients fetch the hash-prefixed HTTP names. Numbered root, targets, and snapshot
metadata and all hash-prefixed targets remain available forever. Only
`timestamp.json` is replaced in place, and it is published last. The protected
state branch additionally retains every signed timestamp envelope as
`timestamp-history/N.timestamp.json`; those internal files are not served, but
the pre-sign continuity check requires live bytes to match the archived
version and then verifies every referenced public byte.

A channel target is deliberately small:

```json
{
  "schema": "https://attestations.trymaple.ai/schemas/channel/v1",
  "environment": "prod",
  "sequence": 12,
  "builderPolicyTarget": {
    "path": "policy/builders.json",
    "sha256": "..."
  },
  "sigstoreTrustedRootTarget": {
    "path": "sigstore/trusted_root.json",
    "sha256": "..."
  },
  "active": [
    {
      "manifestTarget": "releases/1.2.3/prod/manifest.json",
      "manifestSha256": "...",
      "bundleTarget": "releases/1.2.3/prod/manifest.sigstore.json",
      "bundleSha256": "..."
    }
  ]
}
```

`rollout` retains the previous active release and adds the new one.
`finalize` keeps only the selected release. `revoke` publishes an empty active
set so every new client fails closed if no release is safe. The manager rejects
more than two active entries. Channel JSON has no independent expiry; signed
TUF metadata owns expiration.

## What a new client does

```mermaid
sequenceDiagram
    participant SDK
    participant Site as attestations.trymaple.ai
    participant TUF as Embedded TUF root
    participant Sigstore
    participant Enclave

    SDK->>Site: GET timestamp.json
    SDK->>TUF: Verify signature, version, expiry, rollback
    SDK->>Site: GET numbered snapshot + targets
    SDK->>TUF: Verify hashes, signatures, versions, expiry
    SDK->>Site: GET environment channel
    SDK->>TUF: Verify target length + SHA-256
    SDK->>Site: GET builder policy, trusted root, active manifests + bundles
    SDK->>TUF: Verify every target
    alt Rust / native client
        SDK->>Sigstore: Verify exact manifest bytes and builder identity offline
    else Browser TypeScript
        SDK->>SDK: Validate exact TUF-authenticated evidence bytes and schema
        Note over SDK: Sigstore crypto was enforced by protected promotion
    end
    SDK->>Enclave: Send fresh nonce
    Enclave-->>SDK: AWS-signed document + key + PCR0/1/2
    SDK->>SDK: Require exact tuple from one active release
    SDK->>SDK: Accept key or fail closed
```

The SDK embeds only its initial trusted TUF root, not the latest PCR list. It can
therefore learn newly promoted releases without publishing a new SDK. TUF root
rotation is standard sequential root metadata; a client pinned to version N
downloads and verifies N+1 before trusting it.

The production SDK bootstrap remains root version 1. Every later root is a
numbered online transition (`2.root.json`, `3.root.json`, and so on), and the
attestation repository retains and serves every intermediate version. Do not
replace the SDK's embedded root with a later version merely to shorten that
chain: doing so could hide intermediate online-role authority history from a
new installation. A future embedded-root advance requires a separately
reviewed, authenticated bridge-history design that preserves that history.
The current v1 bootstrap accepts at most 32 sequential rotations (root versions
1 through 33), matching the client update bound; the repository manager refuses
a longer history until that bridge design exists.

Both clients reject redirects, expired metadata, rollback, freeze, unknown
schemas, wrong environment, incomplete tuples, unexpected builder IDs, and any
target whose length or digest differs. Rust additionally rejects any local
Sigstore signature, certificate identity, issuer, transparency proof, or
trusted-root failure. Browser TypeScript does not claim to implement that
cryptography: it relies on the separately protected promotion to admit only a
Cosign-verified bundle, while TUF prevents substituting different bytes later.

## Key model

There are two unrelated signing mechanisms:

- **Sigstore release signing is keyless.** The tagged GitHub Actions job obtains
  a short-lived OIDC identity. No Sigstore private key is stored.
- **TUF authorization uses Ed25519 keys.** The initial deployment has one
  threshold-1 offline root key and one threshold-1 online key shared by targets,
  snapshot, and timestamp.

The offline root private key is never a GitHub secret and never enters CI. Store
it outside the repository and use it only for bootstrap and root rotation. The
online private key is the single
`ATTESTATIONS_TUF_ONLINE_PRIVATE_KEY_PEM` secret in the protected
`attestations-production` GitHub Environment. Cloudflare account ID
and an API token scoped to this Worker are the only other production secrets.

This intentionally simple key model can be upgraded later. TUF supports root
updates that add separate online keys or raise role thresholds, and existing
clients verify a new root through the preceding root. The initial manager only
implements bootstrap and ordinary online publication; online-role key rotation
needs a dedicated transition command before operators attempt that upgrade.
Before adding that command or manually publishing rotated roots, the backend
must also reject same-class retired-online key reintroduction (`A -> B -> A`)
and enforce the clients' cumulative 128-fingerprint history bound per
role/class before publication; clients currently fail closed on either
violation.

## Operator setup: deliberately not performed by this change

Production bootstrap fails closed until an operator generates keys and initial
metadata. A safe outline is:

1. Generate root and online Ed25519 keys into an operator-controlled directory,
   never the repository.
2. Download and independently inspect the current official Sigstore trusted
   root.
3. Run the local `bootstrap` command with the offline root key, online
   key, reviewed `attestations/bootstrap/builders.json`, and trusted
   root.
4. Create the protected `attestations-state` branch containing only
   public TUF state. Private keys must not appear there.
5. Put only the online PEM key and scoped Cloudflare credentials in the
   protected GitHub Environment.
6. Configure required reviewers, deployment branch restrictions, tag
   protection, and master protection before the first release.

Example local commands, using paths outside the checkout:

```sh
nix develop --no-update-lock-file
python scripts/manage_tuf_repository.py generate-key --output /secure/offline-root.pem
python scripts/manage_tuf_repository.py generate-key --output /secure/online.pem
curl --fail --proto '=https' --tlsv1.2 \
  https://tuf-repo-cdn.sigstore.dev/targets/trusted_root.json \
  --output /secure/sigstore-trusted_root.json
python scripts/manage_tuf_repository.py bootstrap \
  --repository /secure/attestations-state/tuf \
  --output /secure/rendered/tuf \
  --root-key /secure/offline-root.pem \
  --online-key /secure/online.pem \
  --builder-policy attestations/bootstrap/builders.json \
  --sigstore-trusted-root /secure/sigstore-trusted_root.json
```

These commands are documentation, not authorization to run them. Review the
generated root and its key IDs before publishing it.

## Release and promotion procedure

### 1. Create an inactive candidate

Dispatch `Nitro EIF Release` on a protected
`vMAJOR.MINOR.PATCH` tag. It:

- rebuilds both dev and prod EIFs from the exact tag with locked Nix inputs;
- requires generated PCR files to match the tagged PCR references;
- generates canonical repository-neutral manifests;
- signs each manifest keylessly and creates portable Sigstore bundles;
- verifies exact workflow/ref/repository/commit/trigger claims; and
- publishes candidate assets.

Candidate publication does **not** update TUF or client trust.

### 2. Promote through the protected environment

Dispatch `Promote Nitro Attestation` from protected master. Select
environment, version, and `rollout` or `finalize`. It:

1. verifies existing TUF state before reading policy;
2. downloads the inactive candidate;
3. verifies its portable bundle with the TUF-authenticated Sigstore trusted root
   and exact TUF-authenticated builder policy;
4. proves the live repository is either byte-identical to state or an exact
   authenticated historical subset, rejecting rollback and same-version forks;
5. signs new targets, snapshot, and timestamp metadata with the online key;
6. persists append-only public state without force-pushing;
7. atomically deploys one Worker version containing code and all static bytes;
8. byte-compares every live immutable file; and
9. byte-compares `timestamp.json` last.

`revoke` increments the selected channel sequence with no active releases.
`initial_publish` is a bootstrap escape hatch, accepted only for a `refresh`
when no live timestamp exists and state has never contained a channel or
release. This also makes the first deployment retryable if state persistence
succeeded but Cloudflare deployment failed.

The state branch is persisted before deployment. If Cloudflare deployment
fails, public clients still see the old valid repository while state is one
publication ahead. Repair by rerunning the protected refresh path; do not
rewrite or force-push TUF history.

Timestamp metadata expires after 47 hours. A daily scheduled `refresh` keeps it
inside the clients' 48-hour maximum-validity policy while leaving time for one
retry. Before activation, create an `attestations-refresh` Environment limited
to protected `master`, with the same online-key and Cloudflare secrets but no
required-reviewer wait; scheduled runs select only that Environment and the
workflow itself hard-requires the `refresh` operation. Keep required reviewers
on the separate `attestations-production` Environment used by every manual TUF
rollout, finalize, revoke, and manual refresh. Alert on
workflow failure or a timestamp with less than 24 hours remaining. Targets
expire after 90 days, snapshot after 14 days, and root after one year; plan
offline root renewal well before that deadline.

## Temporary legacy bridge

Existing released clients may still fetch signed PCR history JSON from GitHub.
That bridge remains unchanged:

- `pcr_sign.js` and `pcr_verify.js` retain their original
  PCR0-only format;
- `pcrDevHistory.json` and `pcrProdHistory.json` are not
  rewritten;
- existing `just` recipes remain the manual operator path; and
- the legacy private key is not added to GitHub Actions.

For a release whose PCR tuple changes, the race-free compatibility order is:

1. Run protected TUF `rollout`, which authorizes both the old and new tuples.
2. On the legacy compatibility change, run `just append-pcr-dev` or
   `just append-pcr-prod` with `SIGNING_PRIVATE_KEY`, then run
   `just verify-pcr-history dev` or `prod` with `SIGNING_PUBLIC_KEY`.
3. Review and merge only the corresponding signed history JSON change.
4. Fetch the exact merged raw URL, compare it byte-for-byte with the merged
   file, and rerun the signature verifier against those bytes. Do not deploy on
   a GitHub UI preview or an unmerged branch.
5. Deploy the enclave and verify it reports the intended PCR tuple.
6. Run protected TUF `finalize` only after deployment verification.

For production, step 4 is concretely:

```sh
temporary_history="$(mktemp)"
trap 'rm -f "$temporary_history"' EXIT
curl --fail --silent --show-error --max-redirs 0 --proto '=https' --tlsv1.2 \
  https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrProdHistory.json \
  --output "$temporary_history"
cmp -- "$temporary_history" pcrProdHistory.json
SIGNING_PUBLIC_KEY="$SIGNING_PUBLIC_KEY" just verify-pcr-history prod
```

Use `pcrDevHistory.json` and `dev` for development. The `cmp` proves the local
bytes verified by the unchanged legacy verifier are exactly the merged raw
bytes.

If `rollout` reports that the complete PCR0/PCR1/PCR2 tuple is already active,
do not create a redundant overlap; use `finalize` as directed. New clients
never read or fall back to the legacy JSON. Once old versions no longer need
support, the bridge and GitHub raw-file dependency can be removed without
changing the TUF/Sigstore protocol.

The initial publisher deliberately has no policy/trusted-root update command.
Activating either update requires a separately reviewed transition operation
that verifies the old authenticated policy before signing the replacement.
Inactive logical release targets are pruned automatically; their
consistent-snapshot hash-prefixed bytes and historical metadata remain
append-only.

## Security boundaries

- Sigstore proves attributable, transparency-recorded provenance; it does not
  prove code safety or current approval. Rust re-verifies this locally.
  Browser TypeScript relies on the protected promotion for Sigstore
  cryptographic verification and authenticates the promoted evidence through
  TUF.
- TUF authorizes current releases and resists rollback/freeze; it does not prove
  the enclave is running.
- AWS Nitro attestation proves a fresh document and binds its key/PCRs; it does
  not decide whether that tuple is approved.
- Nix makes independent reproduction practical; one builder's manifest is not
  itself independent reproduction.
- Cloudflare availability and TLS are operational defenses. TUF authentication
  remains the byte-integrity defense for every client; local Sigstore
  verification adds another independent provenance check in Rust.
- A GitHub or builder migration is a builder-policy and source-URI update
  authorized through TUF, not a client rewrite.
