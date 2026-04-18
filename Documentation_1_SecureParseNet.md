# SecureImage v2 Algorithm Documentation

## Overview

SecureImage v2 improves both confidentiality and tamper resistance for steganography and full-file encryption.

Main upgrades:
- Versioned binary payload format (`SIM2`) for steganography.
- Authenticated encryption with AES-256-GCM using header-bound AAD.
- Stronger key derivation: PBKDF2-HMAC-SHA256 with 600,000 iterations.
- Password-seeded randomized embedding positions (for encrypted stego payloads).
- Binary encrypted file format (`SIF1`) with integrity-bound metadata.
- Backward compatibility for legacy JSON `.simg` blobs.

## Threat Model

SecureImage v2 is designed to protect:
- Message confidentiality against unauthorized readers.
- Payload integrity against accidental corruption or active tampering.
- Predictable LSB location leakage (for password-protected mode) via deterministic permutation.

Not guaranteed:
- Resistance to advanced steganalysis by well-funded adversaries.
- Survival under lossy recompression (JPEG/social media pipelines).

## Steganography v2 (`SIM2`)

### Header format (fixed length)

`>4sBBHI16s12s`

Fields:
- `magic` (4): `SIM2`
- `version` (1): `2`
- `flags` (1): bit field
  - `0x01`: encrypted
  - `0x02`: permuted embedding positions
- `reserved` (2): future use
- `payload_len` (4): payload byte length
- `salt` (16): PBKDF2 salt (zeroed in plaintext mode)
- `nonce` (12): AES-GCM nonce (zeroed in plaintext mode)

### Inner message structure

Before encryption, text is converted to:
1. UTF-8 bytes
2. SHA-256 digest of plaintext
3. Packed as: `msg_len(4) || message || digest(32)`
4. Compressed with `zlib(level=9)`

### Encrypted mode workflow

1. Generate `salt` and `nonce`.
2. Derive keys from password using PBKDF2-HMAC-SHA256 (600k iterations, 64-byte output).
3. Domain-separate into:
- `enc_key`: AES-256-GCM key
- `perm_key`: seed material for randomized embedding
4. Build header with known ciphertext length (`compressed_len + 16` GCM tag).
5. Encrypt compressed inner blob with AES-GCM using header bytes as AAD.
6. Embed header sequentially in first header bits.
7. Embed encrypted payload at pseudorandom bit positions derived from `perm_key + header`.

### Plain mode workflow

When no password is provided:
- Payload remains compressed but not encrypted.
- Header and payload are embedded sequentially.
- Inner SHA-256 still verifies accidental corruption.

## Randomized Embedding

For encrypted stego payloads, SecureImage v2 shuffles payload bit positions over the available post-header region.

Benefits:
- Removes fixed contiguous payload pattern.
- Makes simple extraction by naive LSB scanners harder.
- Keeps deterministic decode with correct password.

## File Encryption v1 (`SIF1`)

### Binary format

Header struct: `>4sB16s12sI`
- `magic`: `SIF1`
- `version`: `1`
- `salt`: 16 bytes
- `nonce`: 12 bytes
- `payload_len`: ciphertext length

Data: `header || ciphertext`

### Process

1. Derive file key from password + salt (PBKDF2 600k, domain-separated).
2. Encrypt raw image bytes with AES-256-GCM.
3. Bind header as AAD to prevent metadata tampering.
4. Store compact binary output.

## Validation and Failure Modes

Decode/decrypt fails safely if:
- magic/version mismatch
- header/payload length mismatch
- wrong password
- modified ciphertext/tag
- invalid compressed inner payload
- checksum mismatch in inner message structure

## Backward Compatibility

`decrypt_image_file_bytes` still supports old JSON envelope blobs used in earlier versions.

## Performance Notes

- PBKDF2 600k increases brute-force cost but adds latency on low-power devices.
- Compression level 9 improves capacity for text-heavy messages.
- Randomized embedding has low overhead compared to cryptographic steps.

## Operational Guidance

- Prefer PNG for steganography output.
- Do not re-encode stego images through lossy pipelines.
- Use long unique passwords (at least 14+ chars recommended).
- Share password over a different channel than encrypted media.
