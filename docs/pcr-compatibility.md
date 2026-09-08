# Signed PCR compatibility during the Maple import

The backend import into `MaplePrivacyLabs/Maple` at `services/opensecret/` is
being prepared. Do not switch clients or publishing ownership until the import
is merged, its exact source revision is reviewed, and the new public files on
Maple's `master` branch are verified. This document prepares the legacy side;
it does not declare the cutover complete.

## Keep the installed-client URLs working

Retain this repository at `OpenSecretCloud/opensecret`, public, writable, and
unarchived, with the `master` branch and these root-level files:

| File | Existing raw URL |
| --- | --- |
| `pcrDev.json` | <https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrDev.json> |
| `pcrDevHistory.json` | <https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrDevHistory.json> |
| `pcrProd.json` | <https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrProd.json> |
| `pcrProdHistory.json` | <https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrProdHistory.json> |

Do not rename, transfer, archive, delete, or rewrite the history of this
repository as part of the import. Retaining its current source is intentional;
source retirement is a separate decision. Existing clients must continue to
receive JSON directly at these URLs rather than depending on redirects.

## Manual publication after cutover

After Maple's import is merged and verified, make new backend and signed-PCR
changes in `services/opensecret/`. Preserve the existing signing key, JSON
format, filenames, and previously published history entries. Copy the four
files from one reviewed Maple commit into this repository; do not independently
sign or regenerate a second set here.

The Maple import includes `services/opensecret/docs/pcr-compatibility.md` and
`services/opensecret/scripts/pcr_compatibility.py`. Once available on the
verified Maple revision, use that runbook and helper for the precise commands:

1. Fetch both repositories and identify the full reviewed Maple source commit
   and the current legacy `origin/master` commit. Use a clean legacy checkout
   whose branch starts at that exact legacy commit.
2. Run the helper's dry-run preparation and review its result. It checks the
   current references, complete histories, signatures against the SDK's pinned
   public key, and preservation of every existing history entry.
3. Apply the same preparation to copy only `pcrDev.json`, `pcrDevHistory.json`,
   `pcrProd.json`, and `pcrProdHistory.json`, preserving the exact source bytes.
   The helper performs no network access, signing, commit, push, or deployment.
4. Review the diff, then publish those files in a normal commit or pull request.
   Record the immutable Maple source commit in the commit message or PR body.
   Recheck the legacy tip before publishing; if it advanced, rerun validation
   against the new baseline. Never force-push through an intervening update.
5. Fetch the four public raw URLs above after merge/publication. Require a
   successful response without redirects and compare their bytes with the
   reviewed Maple source files. Do not report publication complete from a
   successful Git push alone.

The existing signature scheme authenticates **PCR0 only**. Structural checks
and matching a current reference to a history entry do not make PCR1 or PCR2
cryptographically authenticated. Do not change the scheme or claim stronger
verification as part of this copy step.

The existing `just update-pcr-dev` and `just update-pcr-prod` recipes build/copy
and append locally signed entries; they are not legacy copy or comprehensive
history-validation commands. Run them only as part of separately authorized
measurement/signing work in the owning backend checkout. Publication does not
deploy an EIF, modify KMS/IAM, or prove which measurements a live enclave uses.

Coordinate signed history publication with the backend deployment so old
clients can recognize the authorized measurement when it starts serving.
Switch new SDK defaults only after the canonical Maple raw URLs exist and are
verified. There is no automatic sunset date for this legacy publication path;
ending it needs a separate client-compatibility decision.

## CI behavior while source remains here

The Nix Reproducible Builds workflow skips automatic runs whose changes are
entirely PCR files or the documentation paths listed in
[`build.yml`](../.github/workflows/build.yml). This avoids rebuilding retained
legacy code just to publish measurements from the monorepo. A mixed change
that also edits source, dependencies, Nix, or the workflow itself still runs the
existing build workflow. Manual `workflow_dispatch` also remains available and
builds this checkout's EIFs; it is not a substitute for validating copied files.

Rust CI and scheduled supply-chain checks are unchanged. No GitHub publishing
credential, cross-organization automation, or deployment workflow is needed for
this manual compatibility path.
