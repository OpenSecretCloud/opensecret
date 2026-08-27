#!/usr/bin/env node

"use strict";

// TRANSITION-ONLY COMPATIBILITY: verifies the PCR0 signatures consumed by
// already-released clients. Sigstore manifest verification is a separate path.

const fs = require("fs");
const {
  LEGACY_PCR_VERIFICATION_PUBLIC_KEY_B64,
  validateAppendOnlyTransition,
  validateLegacyHistory,
  validateLegacyHistories,
  verifyLegacyPcr0Signature,
} = require("./scripts/legacy_pcr_compatibility.js");

function usage() {
  console.error(
    "Usage: pcr_verify.js <dev|prod> [--history-file <path>] [--other-history-file <path>] [--base-history-file <path>] [--require-pcr-file <path>]",
  );
}

function parseArguments(argv) {
  const environment = argv[0];
  if (environment !== "dev" && environment !== "prod") {
    throw new Error("environment must be exactly dev or prod");
  }

  let historyFile =
    environment === "dev" ? "./pcrDevHistory.json" : "./pcrProdHistory.json";
  let otherHistoryFile =
    environment === "dev" ? "./pcrProdHistory.json" : "./pcrDevHistory.json";
  let baseHistoryFile;
  let requiredPcrFile;

  for (let index = 1; index < argv.length; index += 1) {
    const option = argv[index];
    const value = argv[index + 1];
    if (
      (option !== "--history-file" &&
        option !== "--other-history-file" &&
        option !== "--base-history-file" &&
        option !== "--require-pcr-file") ||
      value === undefined
    ) {
      throw new Error(`invalid argument ${option}`);
    }
    if (option === "--history-file") {
      historyFile = value;
    } else if (option === "--other-history-file") {
      otherHistoryFile = value;
    } else if (option === "--base-history-file") {
      baseHistoryFile = value;
    } else {
      requiredPcrFile = value;
    }
    index += 1;
  }

  return {
    environment,
    historyFile,
    otherHistoryFile,
    baseHistoryFile,
    requiredPcrFile,
  };
}

function readContents(file, label) {
  try {
    return fs.readFileSync(file, "utf8");
  } catch (error) {
    throw new Error(`could not read ${label} ${file}: ${error.message}`);
  }
}

function parseJson(contents, label) {
  try {
    return JSON.parse(contents);
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
}

function readJson(file, label) {
  return parseJson(readContents(file, label), `${label} ${file}`);
}

function parseCanonicalHistory(contents, label) {
  const history = parseJson(contents, label);
  const canonical = `${JSON.stringify(history, null, 2)}\n`;
  if (contents !== canonical) {
    throw new Error(
      `${label} is not canonical JSON; reformatting, reordered keys, and duplicate shadow keys are forbidden`,
    );
  }
  return history;
}

function readCanonicalHistory(file, label) {
  return parseCanonicalHistory(
    readContents(file, label),
    `${label} ${file}`,
  );
}

function verifyFiles({
  environment,
  historyFile,
  otherHistoryFile,
  baseHistoryFile,
  requiredPcrFile,
}) {
  const history = readCanonicalHistory(historyFile, "legacy PCR history");
  const otherHistory = readCanonicalHistory(
    otherHistoryFile,
    "other legacy PCR history",
  );
  const baseHistory =
    baseHistoryFile === undefined
      ? undefined
      : readCanonicalHistory(baseHistoryFile, "base legacy PCR history");
  const requiredMeasurements =
    requiredPcrFile === undefined
      ? undefined
      : readJson(requiredPcrFile, "required PCR file");
  const histories =
    environment === "dev"
      ? validateLegacyHistories(history, otherHistory)
      : validateLegacyHistories(otherHistory, history);
  const result = validateLegacyHistory(history, requiredMeasurements);
  const transition =
    baseHistory === undefined
      ? undefined
      : validateAppendOnlyTransition(baseHistory, history);
  return {
    ...result,
    ...transition,
    environment,
    historyFile,
    otherHistoryFile,
    baseHistoryFile,
    requiredPcrFile,
    ...histories,
  };
}

function main() {
  const options = parseArguments(process.argv.slice(2));
  const result = verifyFiles(options);
  const requirement =
    result.requiredPcrFile === undefined
      ? ""
      : ` and contains the complete tuple from ${result.requiredPcrFile}`;
  console.log(
    `Verified ${result.entries} ${result.environment} legacy entries in ${result.historyFile}${requirement}.`,
  );
  console.log(
    `Verified dev/prod PCR0 separation with ${result.otherHistoryFile}.`,
  );
  if (result.baseHistoryFile !== undefined) {
    console.log(
      `Append-only transition from ${result.baseHistoryFile}: ${result.addedEntries} added entries.`,
    );
  }
  console.log(
    `Pinned legacy public key: ${LEGACY_PCR_VERIFICATION_PUBLIC_KEY_B64.slice(0, 20)}...`,
  );
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    usage();
    console.error(`Legacy PCR verification failed: ${error.message}`);
    process.exitCode = 1;
  }
}

module.exports = {
  parseCanonicalHistory,
  parseArguments,
  readJson,
  verifyFiles,
  verifyPcr0Signature: verifyLegacyPcr0Signature,
};
