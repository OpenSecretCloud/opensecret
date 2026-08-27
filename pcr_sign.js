#!/usr/bin/env node

"use strict";

// TRANSITION-ONLY COMPATIBILITY: new clients trust tagged Sigstore manifests.
// Existing clients still fetch pcrDevHistory.json and pcrProdHistory.json, so
// releases temporarily append a PCR0 signature made by the already-pinned key.

const {
  signLegacyPcr0,
} = require("./scripts/legacy_pcr_compatibility.js");

function usage() {
  console.error("Usage: pcr_sign.js sign-pcr0 <96-character PCR0>");
}

function main() {
  const command = process.argv[2];

  if (command === "generate-keys") {
    throw new Error(
      "legacy key generation is disabled: released clients pin the existing public key",
    );
  }

  if (command !== "sign-pcr0") {
    usage();
    process.exitCode = 1;
    return;
  }

  const pcr0 = process.argv[3];
  if (process.argv.length !== 4 || pcr0 === undefined) {
    usage();
    process.exitCode = 1;
    return;
  }

  const signature = signLegacyPcr0(process.env.SIGNING_PRIVATE_KEY, pcr0);
  process.stdout.write(`${signature}\n`);
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    console.error(`Legacy PCR0 signing failed: ${error.message}`);
    process.exitCode = 1;
  }
}

module.exports = {
  main,
  signLegacyPcr0,
};
