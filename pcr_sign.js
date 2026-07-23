#!/usr/bin/env node

// DEPRECATED: pcrDevHistory.json and pcrProdHistory.json are frozen legacy data.
// New measurements are published by .github/workflows/release-nitro-eif.yml.

const fs = require('fs');
const crypto = require('crypto');

/**
 * Generates a P-384 (secp384r1) ECDSA keypair.
 * @returns {Object} An object containing the keypair information
 */
function generateKeypair() {
  // Generate an EC key pair on the P-384 curve
  const { publicKey, privateKey } = crypto.generateKeyPairSync('ec', {
    namedCurve: 'secp384r1',
  });

  // Export the private key in PKCS#8 DER format
  const privateKeyDer = privateKey.export({
    type: 'pkcs8',
    format: 'der',
  });

  // Export the public key in SPKI DER format
  const publicKeyDer = publicKey.export({
    type: 'spki',
    format: 'der',
  });

  // Base64-encode them for easy storage
  const privateKeyBase64 = privateKeyDer.toString('base64');
  const publicKeyBase64 = publicKeyDer.toString('base64');

  // Create PEM format for better human readability
  const privatePem = [
    '-----BEGIN PRIVATE KEY-----',
    ...privateKeyBase64.match(/.{1,64}/g) || [],
    '-----END PRIVATE KEY-----',
  ].join('\n');

  const publicPem = [
    '-----BEGIN PUBLIC KEY-----',
    ...publicKeyBase64.match(/.{1,64}/g) || [],
    '-----END PUBLIC KEY-----',
  ].join('\n');

  return {
    privateKeyBase64,
    publicKeyBase64,
    privatePem,
    publicPem,
    privateKey,
    publicKey
  };
}

/**
 * Signs data using ECDSA+SHA384 with P1363 (raw) encoding
 * @param {Buffer|string} privateKeyData - Private key in base64 DER format
 * @param {string} dataToSign - Data to sign
 * @returns {string} Base64-encoded signature
 */
function signData(privateKeyData, dataToSign) {
  // If privateKeyData is a string, assume it's base64 and convert to Buffer
  const privateKeyBuffer = Buffer.isBuffer(privateKeyData) 
    ? privateKeyData
    : Buffer.from(privateKeyData, 'base64');

  try {
    // Create the private key object
    const privateKey = crypto.createPrivateKey({
      key: privateKeyBuffer,
      format: 'der',
      type: 'pkcs8'
    });

    // Verify data is a string
    if (typeof dataToSign !== 'string') {
      throw new Error('Data to sign must be a string');
    }

    // Create the signer
    const signer = crypto.createSign('SHA384');
    signer.update(dataToSign);

    // Sign in "ieee-p1363" raw format which produces the raw r|s signature
    // This is directly compatible with Web Crypto API's verify method
    const signature = signer.sign({
      key: privateKey,
      dsaEncoding: 'ieee-p1363'
    });

    // Verify signature length (should be 96 bytes for P-384)
    if (signature.length !== 96) {
      console.warn(`Warning: Generated signature is ${signature.length} bytes (expected 96 for P-384)`);
    }

    return signature.toString('base64');
  } catch (error) {
    console.error("Error during signing:", error.message);
    throw error;
  }
}

/**
 * Main function to handle CLI commands
 */
function main() {
  console.error(
    "Legacy PCR0 signing is disabled. Publish measurements with the manually approved Nitro EIF release workflow."
  );
  process.exit(1);
}

// Run the main function if this script is executed directly
if (require.main === module) {
  main();
}

// Export functions for potential module usage
module.exports = {
  generateKeypair,
  signData
};
