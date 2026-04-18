from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import zlib
from io import BytesIO

import numpy as np
from PIL import Image
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

STEGO_MAGIC = b"SIM2"
STEGO_VERSION = 2
STEGO_FLAGS_ENCRYPTED = 0x01
STEGO_FLAGS_PERMUTED = 0x02
STEGO_HEADER_STRUCT = struct.Struct(">4sBBHI16s12s")

FILE_MAGIC = b"SIF1"
FILE_VERSION = 1
FILE_HEADER_STRUCT = struct.Struct(">4sB16s12sI")

PBKDF2_ITERATIONS = 600_000
LEGACY_PBKDF2_ITERATIONS = 250_000
SALT_SIZE = 16
NONCE_SIZE = 12
INNER_STRUCT = struct.Struct(">I")
DIGEST_SIZE = 32


class SecureImageError(Exception):
    """Base error for secure image operations."""


class CapacityError(SecureImageError):
    """Raised when the message does not fit inside the cover image."""


class InvalidPayloadError(SecureImageError):
    """Raised when extracted bytes are not a valid SecureImage payload."""


class AuthenticationError(SecureImageError):
    """Raised when password-based decryption fails."""


def _pbkdf2(password: str, salt: bytes, iterations: int, dklen: int) -> bytes:
    if not password:
        raise ValueError("Password is required for this operation.")
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=dklen)


def _derive_stego_keys(password: str, salt: bytes) -> tuple[bytes, bytes]:
    # Derive 64 bytes and split into domain-separated keys.
    material = _pbkdf2(password, salt, PBKDF2_ITERATIONS, 64)
    enc_key = hashlib.sha256(material[:32] + b":stego:enc:v2").digest()
    perm_key = hashlib.sha256(material[32:] + b":stego:perm:v2").digest()
    return enc_key, perm_key


def _derive_file_key(password: str, salt: bytes) -> bytes:
    material = _pbkdf2(password, salt, PBKDF2_ITERATIONS, 32)
    return hashlib.sha256(material + b":file:enc:v1").digest()


def _image_to_rgb_array(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise SecureImageError("Unable to process image as RGB.")
    return arr


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size % 8 != 0:
        raise InvalidPayloadError("Corrupted bitstream length.")
    return np.packbits(bits).tobytes()


def _extract_bits(flat: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return (flat[positions] & 1).astype(np.uint8)


def _embed_bits(flat: np.ndarray, positions: np.ndarray, bits: np.ndarray) -> None:
    flat[positions] = (flat[positions] & 0xFE) | bits


def _make_payload_positions(
    total_bits: int,
    payload_bits: int,
    perm_key: bytes | None,
    header_bytes: bytes,
) -> np.ndarray:
    start = STEGO_HEADER_STRUCT.size * 8
    available = total_bits - start
    if payload_bits > available:
        raise CapacityError(f"Payload needs {payload_bits} bits but image only holds {available} bits after header.")

    if perm_key is None:
        return np.arange(start, start + payload_bits, dtype=np.int64)

    seed = hashlib.sha256(perm_key + header_bytes + b":perm-seed").digest()
    rng = np.random.default_rng(int.from_bytes(seed[:8], "big", signed=False))
    universe = np.arange(start, total_bits, dtype=np.int64)
    rng.shuffle(universe)
    return universe[:payload_bits]


def _pack_inner_message(message: str) -> bytes:
    message_bytes = message.encode("utf-8")
    digest = hashlib.sha256(message_bytes).digest()
    inner = INNER_STRUCT.pack(len(message_bytes)) + message_bytes + digest
    return zlib.compress(inner, level=9)


def _unpack_inner_message(blob: bytes) -> str:
    try:
        inner = zlib.decompress(blob)
    except zlib.error as exc:
        raise InvalidPayloadError("Failed to decompress embedded payload.") from exc

    if len(inner) < INNER_STRUCT.size + DIGEST_SIZE:
        raise InvalidPayloadError("Inner payload is too short.")

    (msg_len,) = INNER_STRUCT.unpack(inner[: INNER_STRUCT.size])
    expected = INNER_STRUCT.size + msg_len + DIGEST_SIZE
    if len(inner) != expected:
        raise InvalidPayloadError("Inner payload length mismatch.")

    msg_start = INNER_STRUCT.size
    msg_end = msg_start + msg_len
    message_bytes = inner[msg_start:msg_end]
    digest = inner[msg_end:]

    if hashlib.sha256(message_bytes).digest() != digest:
        raise InvalidPayloadError("Inner checksum mismatch. Data may be corrupted.")

    try:
        return message_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidPayloadError("Recovered payload is not valid UTF-8 text.") from exc


def _build_stego_payload(message: str, password: str | None) -> tuple[bytes, bytes, bytes | None]:
    if not message:
        raise ValueError("Message cannot be empty.")

    compressed_inner = _pack_inner_message(message)

    if password:
        flags = STEGO_FLAGS_ENCRYPTED | STEGO_FLAGS_PERMUTED
        salt = os.urandom(SALT_SIZE)
        nonce = os.urandom(NONCE_SIZE)
        enc_key, perm_key = _derive_stego_keys(password, salt)

        # AES-GCM adds a 16-byte tag, so we know ciphertext length before encryption.
        payload_len = len(compressed_inner) + 16
        header = STEGO_HEADER_STRUCT.pack(
            STEGO_MAGIC,
            STEGO_VERSION,
            flags,
            0,
            payload_len,
            salt,
            nonce,
        )
        ciphertext = AESGCM(enc_key).encrypt(nonce, compressed_inner, header)
        return header, ciphertext, perm_key

    flags = 0
    salt = bytes(SALT_SIZE)
    nonce = bytes(NONCE_SIZE)
    header = STEGO_HEADER_STRUCT.pack(
        STEGO_MAGIC,
        STEGO_VERSION,
        flags,
        0,
        len(compressed_inner),
        salt,
        nonce,
    )
    return header, compressed_inner, None


def _parse_stego_payload(header: bytes, payload: bytes, password: str | None) -> str:
    try:
        magic, version, flags, _reserved, payload_len, salt, nonce = STEGO_HEADER_STRUCT.unpack(header)
    except struct.error as exc:
        raise InvalidPayloadError("Invalid SecureImage header format.") from exc

    if magic != STEGO_MAGIC:
        raise InvalidPayloadError("Signature mismatch. No SecureImage payload found.")
    if version != STEGO_VERSION:
        raise InvalidPayloadError("Unsupported SecureImage payload version.")
    if payload_len != len(payload):
        raise InvalidPayloadError("Payload length mismatch. Image may be corrupted.")

    encrypted = bool(flags & STEGO_FLAGS_ENCRYPTED)

    if encrypted:
        if not password:
            raise AuthenticationError("Password required to decode this image.")

        enc_key, _ = _derive_stego_keys(password, salt)
        try:
            compressed_inner = AESGCM(enc_key).decrypt(nonce, payload, header)
        except Exception as exc:
            raise AuthenticationError("Incorrect password or modified image data.") from exc
    else:
        compressed_inner = payload

    return _unpack_inner_message(compressed_inner)


def capacity_for_text(image: Image.Image) -> int:
    arr = _image_to_rgb_array(image)
    total_bytes = arr.size // 8
    # Conservative estimate for encrypted mode overhead.
    reserve = STEGO_HEADER_STRUCT.size + 80
    return max(total_bytes - reserve, 0)


def encode_message_into_image(image: Image.Image, message: str, password: str | None = None) -> Image.Image:
    arr = _image_to_rgb_array(image)
    flat = arr.reshape(-1).copy()

    header, payload, perm_key = _build_stego_payload(message=message, password=password)
    header_bits = _bytes_to_bits(header)
    payload_bits = _bytes_to_bits(payload)

    total_bits = flat.size
    header_positions = np.arange(header_bits.size, dtype=np.int64)
    if header_bits.size > total_bits:
        raise CapacityError("Image is too small to store the SecureImage header.")

    payload_positions = _make_payload_positions(
        total_bits=total_bits,
        payload_bits=payload_bits.size,
        perm_key=perm_key,
        header_bytes=header,
    )

    _embed_bits(flat, header_positions, header_bits)
    _embed_bits(flat, payload_positions, payload_bits)

    encoded = flat.reshape(arr.shape)
    return Image.fromarray(encoded, mode="RGB")


def decode_message_from_image(image: Image.Image, password: str | None = None) -> str:
    arr = _image_to_rgb_array(image)
    flat = arr.reshape(-1)

    header_bits_count = STEGO_HEADER_STRUCT.size * 8
    if header_bits_count > flat.size:
        raise InvalidPayloadError("Image is too small to contain a SecureImage payload.")

    header_positions = np.arange(header_bits_count, dtype=np.int64)
    header_bits = _extract_bits(flat, header_positions)
    header = _bits_to_bytes(header_bits)

    try:
        magic, version, flags, _reserved, payload_len, salt, _nonce = STEGO_HEADER_STRUCT.unpack(header)
    except struct.error as exc:
        raise InvalidPayloadError("Invalid SecureImage header.") from exc

    if magic != STEGO_MAGIC:
        raise InvalidPayloadError("Signature mismatch. No SecureImage payload found.")
    if version != STEGO_VERSION:
        raise InvalidPayloadError("Unsupported SecureImage payload version.")

    payload_bits_count = payload_len * 8
    available = flat.size - header_bits_count
    if payload_bits_count > available:
        raise InvalidPayloadError("Header payload length exceeds image capacity.")

    perm_key = None
    if flags & STEGO_FLAGS_PERMUTED:
        if not password:
            raise AuthenticationError("Password required to decode this image.")
        _, perm_key = _derive_stego_keys(password, salt)

    payload_positions = _make_payload_positions(
        total_bits=flat.size,
        payload_bits=payload_bits_count,
        perm_key=perm_key,
        header_bytes=header,
    )
    payload_bits = _extract_bits(flat, payload_positions)
    payload = _bits_to_bytes(payload_bits)

    return _parse_stego_payload(header, payload, password=password)


def _decrypt_legacy_json_blob(encrypted_blob: bytes, password: str) -> bytes:
    try:
        envelope = json.loads(encrypted_blob.decode("utf-8"))
        salt = base64.b64decode(envelope["salt_b64"])
        nonce = base64.b64decode(envelope["nonce_b64"])
        ciphertext = base64.b64decode(envelope["ciphertext_b64"])
    except Exception as exc:
        raise InvalidPayloadError("Invalid encrypted file format.") from exc

    legacy_key = _pbkdf2(password, salt, LEGACY_PBKDF2_ITERATIONS, 32)
    try:
        return AESGCM(legacy_key).decrypt(nonce, ciphertext, None)
    except Exception as exc:
        raise AuthenticationError("Incorrect password or corrupted encrypted file.") from exc


def encrypt_image_file_bytes(image_bytes: bytes, password: str) -> bytes:
    if not password:
        raise ValueError("Password is required for encryption.")
    if not image_bytes:
        raise ValueError("Image bytes are empty.")

    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    file_key = _derive_file_key(password, salt)

    payload_len = len(image_bytes) + 16
    header = FILE_HEADER_STRUCT.pack(FILE_MAGIC, FILE_VERSION, salt, nonce, payload_len)
    ciphertext = AESGCM(file_key).encrypt(nonce, image_bytes, header)
    return header + ciphertext


def decrypt_image_file_bytes(encrypted_blob: bytes, password: str) -> bytes:
    if not password:
        raise ValueError("Password is required for decryption.")

    # Backward compatibility for older JSON-based blobs.
    if encrypted_blob.startswith(b"{"):
        return _decrypt_legacy_json_blob(encrypted_blob, password)

    if len(encrypted_blob) < FILE_HEADER_STRUCT.size:
        raise InvalidPayloadError("Invalid encrypted file format.")

    header = encrypted_blob[: FILE_HEADER_STRUCT.size]
    payload = encrypted_blob[FILE_HEADER_STRUCT.size :]

    try:
        magic, version, salt, nonce, payload_len = FILE_HEADER_STRUCT.unpack(header)
    except struct.error as exc:
        raise InvalidPayloadError("Invalid encrypted file header.") from exc

    if magic != FILE_MAGIC or version != FILE_VERSION:
        raise InvalidPayloadError("Unknown encrypted file signature.")
    if payload_len != len(payload):
        raise InvalidPayloadError("Encrypted payload length mismatch.")

    file_key = _derive_file_key(password, salt)
    try:
        return AESGCM(file_key).decrypt(nonce, payload, header)
    except Exception as exc:
        raise AuthenticationError("Incorrect password or corrupted encrypted file.") from exc


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
