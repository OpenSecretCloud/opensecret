"use strict";

const crypto = require("crypto");

// Existing released SDKs pin this key. It is intentionally immutable for the
// duration of the compatibility window: generating or substituting a new key
// would make every legacy client reject the new PCR entry.
const LEGACY_PCR_VERIFICATION_PUBLIC_KEY_B64 =
  "MHYwEAYHKoZIzj0CAQYFK4EEACIDYgAEHiUY9kFWK1GqBGzczohhwEwElXzgWLDZa9R6wBx3JOBocgSt9+UIzZlJbPDjYeGBfDUXh7Z62BG2vVsh2NgclLB5S7A2ucBBtb1wd8vSQHP8jpdPhZX1slauPgbnROIP";

const PCR_PATTERN = /^[0-9a-f]{96}$/;
const ZERO_PCR = "0".repeat(96);
const HISTORY_ENTRY_FIELDS = [
  "PCR0",
  "PCR1",
  "PCR2",
  "signature",
  "timestamp",
];

function assertPcr(value, field, context = "PCR record") {
  if (typeof value !== "string" || !PCR_PATTERN.test(value) || value === ZERO_PCR) {
    throw new Error(`${context} has an invalid ${field}`);
  }
}

function validateMeasurements(record, context = "PCR record") {
  if (record === null || typeof record !== "object" || Array.isArray(record)) {
    throw new Error(`${context} must be a JSON object`);
  }

  assertPcr(record.PCR0, "PCR0", context);
  assertPcr(record.PCR1, "PCR1", context);
  assertPcr(record.PCR2, "PCR2", context);

  return {
    PCR0: record.PCR0,
    PCR1: record.PCR1,
    PCR2: record.PCR2,
  };
}

function decodeCanonicalSignature(signatureBase64, context = "legacy PCR entry") {
  if (typeof signatureBase64 !== "string" || signatureBase64.length === 0) {
    throw new Error(`${context} has no signature`);
  }

  const signature = Buffer.from(signatureBase64, "base64");
  if (
    signature.length !== 96 ||
    signature.toString("base64") !== signatureBase64
  ) {
    throw new Error(`${context} has a malformed P-384 signature`);
  }
  return signature;
}

function legacyPublicKey() {
  return crypto.createPublicKey({
    key: Buffer.from(LEGACY_PCR_VERIFICATION_PUBLIC_KEY_B64, "base64"),
    format: "der",
    type: "spki",
  });
}

function verifyLegacyPcr0Signature(pcr0, signatureBase64) {
  assertPcr(pcr0, "PCR0");
  const signature = decodeCanonicalSignature(signatureBase64);
  const verifier = crypto.createVerify("SHA384");
  verifier.update(pcr0);
  verifier.end();
  return verifier.verify(
    {
      key: legacyPublicKey(),
      dsaEncoding: "ieee-p1363",
    },
    signature,
  );
}

function assertLegacySigningKey(privateKeyBase64) {
  if (typeof privateKeyBase64 !== "string" || privateKeyBase64.length === 0) {
    throw new Error("SIGNING_PRIVATE_KEY is not set");
  }

  let privateKey;
  try {
    privateKey = crypto.createPrivateKey({
      key: Buffer.from(privateKeyBase64, "base64"),
      format: "der",
      type: "pkcs8",
    });
  } catch (error) {
    throw new Error(`SIGNING_PRIVATE_KEY is not valid PKCS#8 DER: ${error.message}`);
  }

  const derivedPublicKey = crypto
    .createPublicKey(privateKey)
    .export({ format: "der", type: "spki" })
    .toString("base64");
  if (derivedPublicKey !== LEGACY_PCR_VERIFICATION_PUBLIC_KEY_B64) {
    throw new Error(
      "SIGNING_PRIVATE_KEY does not match the public key pinned by legacy clients",
    );
  }

  return privateKey;
}

function signLegacyPcr0(privateKeyBase64, pcr0) {
  assertPcr(pcr0, "PCR0");
  const privateKey = assertLegacySigningKey(privateKeyBase64);
  const signer = crypto.createSign("SHA384");
  signer.update(pcr0);
  signer.end();
  const signature = signer.sign({
    key: privateKey,
    dsaEncoding: "ieee-p1363",
  });

  if (signature.length !== 96) {
    throw new Error(
      `generated an unexpected ${signature.length}-byte P-384 signature`,
    );
  }

  const signatureBase64 = signature.toString("base64");
  if (!verifyLegacyPcr0Signature(pcr0, signatureBase64)) {
    throw new Error("generated signature did not verify with the pinned legacy key");
  }
  return signatureBase64;
}

function validateLegacyHistory(history, requiredMeasurements = undefined) {
  if (!Array.isArray(history) || history.length === 0) {
    throw new Error("legacy PCR history must be a non-empty JSON array");
  }

  const seenPcr0 = new Set();
  for (const [index, entry] of history.entries()) {
    const context = `legacy PCR entry ${index}`;
    const fields = Object.keys(entry).sort();
    if (
      fields.length !== HISTORY_ENTRY_FIELDS.length ||
      fields.some((field, fieldIndex) => field !== HISTORY_ENTRY_FIELDS[fieldIndex])
    ) {
      throw new Error(`${context} has missing or unknown fields`);
    }
    validateMeasurements(entry, context);
    if (!Number.isSafeInteger(entry.timestamp) || entry.timestamp <= 0) {
      throw new Error(`${context} has an invalid timestamp`);
    }
    if (seenPcr0.has(entry.PCR0)) {
      throw new Error(`${context} duplicates PCR0 ${entry.PCR0}`);
    }
    seenPcr0.add(entry.PCR0);

    if (!verifyLegacyPcr0Signature(entry.PCR0, entry.signature)) {
      throw new Error(`${context} has an invalid PCR0 signature`);
    }
  }

  if (requiredMeasurements !== undefined) {
    const required = validateMeasurements(
      requiredMeasurements,
      "required PCR measurements",
    );
    const pcr0Entry = history.find((entry) => entry.PCR0 === required.PCR0);
    if (pcr0Entry === undefined) {
      throw new Error(
        `legacy PCR history does not contain required PCR0 ${required.PCR0}`,
      );
    }
    if (
      pcr0Entry.PCR1 !== required.PCR1 ||
      pcr0Entry.PCR2 !== required.PCR2
    ) {
      throw new Error(
        "legacy PCR history contains the required PCR0 with a different PCR1/PCR2 tuple",
      );
    }
  }

  return { entries: history.length };
}

function validateAppendOnlyTransition(baseHistory, candidateHistory) {
  validateLegacyHistory(baseHistory);
  validateLegacyHistory(candidateHistory);

  if (candidateHistory.length < baseHistory.length) {
    throw new Error("legacy PCR history was truncated");
  }

  for (let index = 0; index < baseHistory.length; index += 1) {
    const baseEntry = baseHistory[index];
    const candidateEntry = candidateHistory[index];
    for (const field of HISTORY_ENTRY_FIELDS) {
      if (candidateEntry[field] !== baseEntry[field]) {
        throw new Error(
          `legacy PCR history changed existing entry ${index} field ${field}`,
        );
      }
    }
  }

  return {
    addedEntries: candidateHistory.length - baseHistory.length,
    baseEntries: baseHistory.length,
    candidateEntries: candidateHistory.length,
  };
}

function validateLegacyHistories(devHistory, prodHistory) {
  const devResult = validateLegacyHistory(devHistory);
  const prodResult = validateLegacyHistory(prodHistory);
  const devPcr0s = new Set(devHistory.map((entry) => entry.PCR0));

  for (const [index, entry] of prodHistory.entries()) {
    if (devPcr0s.has(entry.PCR0)) {
      throw new Error(
        `legacy prod entry ${index} reuses dev PCR0 ${entry.PCR0}`,
      );
    }
  }

  return {
    devEntries: devResult.entries,
    prodEntries: prodResult.entries,
  };
}

module.exports = {
  LEGACY_PCR_VERIFICATION_PUBLIC_KEY_B64,
  assertLegacySigningKey,
  assertPcr,
  decodeCanonicalSignature,
  signLegacyPcr0,
  validateAppendOnlyTransition,
  validateLegacyHistory,
  validateLegacyHistories,
  validateMeasurements,
  verifyLegacyPcr0Signature,
};
