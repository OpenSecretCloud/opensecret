# OpenSecret signed PCR compatibility mirror

OpenSecret, the Rust backend for confidential AI applications such as
[Maple](https://github.com/MaplePrivacyLabs/Maple), is developed in
[MaplePrivacyLabs/Maple under `services/opensecret/`](https://github.com/MaplePrivacyLabs/Maple/tree/master/services/opensecret).
This repository now serves only the signed PCR files that older installed
clients fetch from these fixed URLs:

| File | URL |
| --- | --- |
| `pcrDev.json` | <https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrDev.json> |
| `pcrDevHistory.json` | <https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrDevHistory.json> |
| `pcrProd.json` | <https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrProd.json> |
| `pcrProdHistory.json` | <https://raw.githubusercontent.com/OpenSecretCloud/opensecret/master/pcrProdHistory.json> |

Each history entry carries an ECDSA P-384 signature over the PCR0 text,
verifiable with the public key pinned in the Maple SDKs. Current SDKs read the
canonical copies under
`https://raw.githubusercontent.com/MaplePrivacyLabs/Maple/master/services/opensecret/`;
both locations publish identical bytes from one reviewed Maple commit.

## Verifying

- How clients verify enclave attestation against these files:
  [PCR verification](https://github.com/MaplePrivacyLabs/Maple/blob/master/services/opensecret/docs/PCR_VERIFICATION.md).
- Offline validation of the four files, every signature, and history
  preservation: `scripts/pcr_compatibility.py check .` from the Maple backend
  component. The workflow in this repository runs it on every change.

## Publishing

Approvals are built, signed and merged in Maple first, then copied here
byte-for-byte with the
[compatibility procedure](https://github.com/MaplePrivacyLabs/Maple/blob/master/services/opensecret/docs/pcr-compatibility.md).
Never sign or regenerate files here. Keep this repository public, writable and
unarchived; do not rename, transfer or rewrite its history. There is no sunset
date for these URLs.

The retired backend source, documentation and tooling remain in this
repository's git history before the retirement commit. Nothing here is a
development or deployment entrypoint.
