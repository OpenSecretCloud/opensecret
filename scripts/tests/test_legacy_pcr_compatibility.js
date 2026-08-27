"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const {
  LEGACY_PCR_VERIFICATION_PUBLIC_KEY_B64,
  assertLegacySigningKey,
  validateAppendOnlyTransition,
  validateLegacyHistory,
  validateLegacyHistories,
  verifyLegacyPcr0Signature,
} = require("../legacy_pcr_compatibility.js");
const { parseCanonicalHistory } = require("../../pcr_verify.js");

const repositoryRoot = path.resolve(__dirname, "..", "..");

function readJson(file) {
  return JSON.parse(fs.readFileSync(path.join(repositoryRoot, file), "utf8"));
}

test("the legacy client public key remains byte-for-byte pinned", () => {
  const digest = crypto
    .createHash("sha256")
    .update(Buffer.from(LEGACY_PCR_VERIFICATION_PUBLIC_KEY_B64, "base64"))
    .digest("hex");
  assert.equal(
    digest,
    "f170f4916d312a80960150a097248fb3a70ca2004c05ba834392d979c3b56f15",
  );
});

for (const [environment, historyFile] of [
  ["dev", "pcrDevHistory.json"],
  ["prod", "pcrProdHistory.json"],
]) {
  test(`${environment} history verifies independently of the next release reference`, () => {
    const history = readJson(historyFile);
    const result = validateLegacyHistory(history);
    assert.equal(result.entries, history.length);

    const existingMeasurements = {
      PCR0: history.at(-1).PCR0,
      PCR1: history.at(-1).PCR1,
      PCR2: history.at(-1).PCR2,
    };
    assert.equal(
      validateLegacyHistory(history, existingMeasurements).entries,
      history.length,
    );
  });
}

test("tampering with a signed PCR0 is rejected", () => {
  const history = readJson("pcrDevHistory.json");
  const tampered = structuredClone(history);
  tampered[0].PCR0 = `a${tampered[0].PCR0.slice(1)}`;
  if (tampered[0].PCR0 === history[0].PCR0) {
    tampered[0].PCR0 = `b${tampered[0].PCR0.slice(1)}`;
  }

  assert.equal(
    verifyLegacyPcr0Signature(history[0].PCR0, history[0].signature),
    true,
  );
  assert.throws(
    () => validateLegacyHistory(tampered),
    /invalid PCR0 signature/,
  );
});

test("a required PCR0 cannot be paired with PCR1 and PCR2 from another release", () => {
  const history = readJson("pcrProdHistory.json");
  const mixedTuple = {
    PCR0: history.at(-1).PCR0,
    PCR1: history.at(-2).PCR1,
    PCR2: history.at(-2).PCR2,
  };
  if (
    mixedTuple.PCR1 === history.at(-1).PCR1 &&
    mixedTuple.PCR2 === history.at(-1).PCR2
  ) {
    mixedTuple.PCR2 = `a${mixedTuple.PCR2.slice(1)}`;
    if (mixedTuple.PCR2 === history.at(-1).PCR2) {
      mixedTuple.PCR2 = `b${mixedTuple.PCR2.slice(1)}`;
    }
  }

  assert.throws(
    () => validateLegacyHistory(history, mixedTuple),
    /different PCR1\/PCR2 tuple/,
  );
});

test("a newly generated private key cannot sign transition entries", () => {
  const { privateKey } = crypto.generateKeyPairSync("ec", {
    namedCurve: "secp384r1",
  });
  const unrelatedPrivateKey = privateKey
    .export({ format: "der", type: "pkcs8" })
    .toString("base64");

  assert.throws(
    () => assertLegacySigningKey(unrelatedPrivateKey),
    /does not match the public key pinned by legacy clients/,
  );
});

test("only exact suffix appends are accepted", () => {
  const history = readJson("pcrDevHistory.json");
  const base = history.slice(0, -1);
  assert.deepEqual(validateAppendOnlyTransition(base, history), {
    addedEntries: 1,
    baseEntries: base.length,
    candidateEntries: history.length,
  });

  assert.throws(
    () => validateAppendOnlyTransition(history, base),
    /history was truncated/,
  );

  const reordered = structuredClone(history);
  [reordered[0], reordered[1]] = [reordered[1], reordered[0]];
  assert.throws(
    () => validateAppendOnlyTransition(history, reordered),
    /changed existing entry 0/,
  );

  const unsignedFieldMutation = structuredClone(history);
  unsignedFieldMutation[0].PCR1 =
    `${history[0].PCR1[0] === "a" ? "b" : "a"}${history[0].PCR1.slice(1)}`;
  assert.throws(
    () => validateAppendOnlyTransition(history, unsignedFieldMutation),
    /changed existing entry 0 field PCR1/,
  );
});

test("unknown fields and duplicate PCR0 entries are rejected", () => {
  const history = readJson("pcrProdHistory.json");

  const unknownField = structuredClone(history);
  unknownField[0].note = "not covered by the legacy signature";
  assert.throws(
    () => validateLegacyHistory(unknownField),
    /missing or unknown fields/,
  );

  const duplicate = [...history, structuredClone(history.at(-1))];
  assert.throws(() => validateLegacyHistory(duplicate), /duplicates PCR0/);
});

test("history bytes must be canonical and cannot contain shadow JSON keys", () => {
  const history = readJson("pcrDevHistory.json");
  const canonical = `${JSON.stringify(history, null, 2)}\n`;
  assert.deepEqual(
    parseCanonicalHistory(canonical, "test history"),
    history,
  );
  assert.throws(
    () => parseCanonicalHistory(canonical.trimEnd(), "test history"),
    /not canonical JSON/,
  );

  const duplicateShadow = canonical.replace(
    `"PCR0": "${history[0].PCR0}"`,
    `"PCR0": "${"a".repeat(96)}",\n    "PCR0": "${history[0].PCR0}"`,
  );
  assert.throws(
    () => parseCanonicalHistory(duplicateShadow, "test history"),
    /duplicate shadow keys are forbidden/,
  );
});

test("dev and prod histories cannot reuse a signed PCR0", () => {
  const devHistory = readJson("pcrDevHistory.json");
  const prodHistory = readJson("pcrProdHistory.json");

  assert.deepEqual(validateLegacyHistories(devHistory, prodHistory), {
    devEntries: devHistory.length,
    prodEntries: prodHistory.length,
  });

  const confusedProdHistory = [
    ...prodHistory,
    structuredClone(devHistory.at(-1)),
  ];
  assert.throws(
    () => validateLegacyHistories(devHistory, confusedProdHistory),
    /reuses dev PCR0/,
  );
});
