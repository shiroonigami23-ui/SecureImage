from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import zlib
from dataclasses import dataclass
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"SIMG"
HEADER_STRUCT = struct.Struct(">4sI")  # magic + payload length in bytes
PBKDF2_ITERATIONS = 250_000
SALT_SIZE = 16
NONCE_SIZE = 12


class SecureImageError(Exception):
    """Base error for secure image operations."""


class CapacityError(SecureImageError):
    """Raised when the message does not fit inside the cover image."""


class InvalidPayloadError(SecureImageError):
    """Raised when extracted bytes are not a valid SecureImage payload."""


class AuthenticationError(SecureImageError):
    """Raised when password-based decryption fails."""


@dataclass
class MessagePayload:
    encrypted: bool
    compressed: bool
    checksum: str
    message: str | None = None
    salt_b64: str | None = None
    nonce_b64: str | None = None
    ciphertext_b64: str | None = None


def _derive_key(password: str, salt: bytes) -> bytes:
    if not password:
        raise ValueError("Password is required for this operation.")
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS, dklen=32)


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


def _embed_bits_in_array(arr: np.ndarray, bits: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1).copy()
    capacity = flat.size
    if bits.size > capacity:
        raise CapacityError(f"Payload needs {bits.size} bits but image only holds {capacity} bits.")
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    return flat.reshape(arr.shape)


def _extract_bits_from_array(arr: np.ndarray, bit_count: int) -> np.ndarray:
    flat = arr.reshape(-1)
    if bit_count > flat.size:
        raise InvalidPayloadError("Requested more bits than image capacity.")
    return (flat[:bit_count] & 1).astype(np.uint8)


def _build_payload(message: str, password: str | None) -> bytes:
    message_bytes = message.encode("utf-8")
    compressed = zlib.compress(message_bytes, level=9)
    checksum = hashlib.sha256(message_bytes).hexdigest()

    if password:
        salt = os.urandom(SALT_SIZE)
        nonce = os.urandom(NONCE_SIZE)
        key = _derive_key(password, salt)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, compressed, None)
        payload = MessagePayload(
            encrypted=True,
            compressed=True,
            checksum=checksum,
            salt_b64=base64.b64encode(salt).decode("ascii"),
            nonce_b64=base64.b64encode(nonce).decode("ascii"),
            ciphertext_b64=base64.b64encode(ciphertext).decode("ascii"),
        )
    else:
        payload = MessagePayload(
            encrypted=False,
            compressed=True,
            checksum=checksum,
            message=base64.b64encode(compressed).decode("ascii"),
        )

    body = json.dumps(payload.__dict__, separators=(",", ":")).encode("utf-8")
    header = HEADER_STRUCT.pack(MAGIC, len(body))
    return header + body


def _parse_payload(payload_bytes: bytes, password: str | None) -> str:
    if len(payload_bytes) < HEADER_STRUCT.size:
        raise InvalidPayloadError("Image does not contain a SecureImage payload.")

    magic, body_len = HEADER_STRUCT.unpack(payload_bytes[: HEADER_STRUCT.size])
    if magic != MAGIC:
        raise InvalidPayloadError("Signature mismatch. This image was not encoded by SecureImage.")

    body_end = HEADER_STRUCT.size + body_len
    if body_end > len(payload_bytes):
        raise InvalidPayloadError("Payload length is invalid or image is corrupted.")

    try:
        body = json.loads(payload_bytes[HEADER_STRUCT.size:body_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidPayloadError("Failed to decode embedded payload.") from exc

    encrypted = bool(body.get("encrypted"))
    checksum = body.get("checksum")
    if not checksum:
        raise InvalidPayloadError("Missing payload checksum.")

    if encrypted:
        salt_b64 = body.get("salt_b64")
        nonce_b64 = body.get("nonce_b64")
        ct_b64 = body.get("ciphertext_b64")
        if not (salt_b64 and nonce_b64 and ct_b64):
            raise InvalidPayloadError("Encrypted payload is incomplete.")
        if not password:
            raise AuthenticationError("Password required to decode this image.")

        salt = base64.b64decode(salt_b64)
        nonce = base64.b64decode(nonce_b64)
        ciphertext = base64.b64decode(ct_b64)

        key = _derive_key(password, salt)
        aesgcm = AESGCM(key)
        try:
            compressed = aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as exc:  # cryptography raises InvalidTag
            raise AuthenticationError("Incorrect password or modified image data.") from exc
    else:
        msg_b64 = body.get("message")
        if not msg_b64:
            raise InvalidPayloadError("Plain payload is missing message bytes.")
        compressed = base64.b64decode(msg_b64)

    try:
        message_bytes = zlib.decompress(compressed)
    except zlib.error as exc:
        raise InvalidPayloadError("Failed to decompress extracted message.") from exc

    actual_checksum = hashlib.sha256(message_bytes).hexdigest()
    if actual_checksum != checksum:
        raise InvalidPayloadError("Integrity verification failed. Payload may be damaged.")

    return message_bytes.decode("utf-8")


def capacity_for_text(image: Image.Image) -> int:
    arr = _image_to_rgb_array(image)
    total_bits = arr.size
    header_bits = HEADER_STRUCT.size * 8
    usable_bits = max(total_bits - header_bits, 0)
    return usable_bits // 8


def encode_message_into_image(image: Image.Image, message: str, password: str | None = None) -> Image.Image:
    if not message:
        raise ValueError("Message cannot be empty.")

    arr = _image_to_rgb_array(image)
    payload = _build_payload(message=message, password=password)
    payload_bits = _bytes_to_bits(payload)

    encoded = _embed_bits_in_array(arr, payload_bits)
    return Image.fromarray(encoded, mode="RGB")


def decode_message_from_image(image: Image.Image, password: str | None = None) -> str:
    arr = _image_to_rgb_array(image)

    header_bits = HEADER_STRUCT.size * 8
    header_stream = _extract_bits_from_array(arr, header_bits)
    header_bytes = _bits_to_bytes(header_stream)

    magic, body_len = HEADER_STRUCT.unpack(header_bytes)
    if magic != MAGIC:
        raise InvalidPayloadError("Signature mismatch. No SecureImage payload found.")

    total_bytes = HEADER_STRUCT.size + body_len
    total_bits = total_bytes * 8
    bit_stream = _extract_bits_from_array(arr, total_bits)
    payload_bytes = _bits_to_bytes(bit_stream)
    return _parse_payload(payload_bytes, password=password)


def encrypt_image_file_bytes(image_bytes: bytes, password: str) -> bytes:
    if not password:
        raise ValueError("Password is required for encryption.")
    if not image_bytes:
        raise ValueError("Image bytes are empty.")

    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key = _derive_key(password, salt)
    ciphertext = AESGCM(key).encrypt(nonce, image_bytes, None)

    envelope = {
        "version": 1,
        "salt_b64": base64.b64encode(salt).decode("ascii"),
        "nonce_b64": base64.b64encode(nonce).decode("ascii"),
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
    }
    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


def decrypt_image_file_bytes(encrypted_blob: bytes, password: str) -> bytes:
    if not password:
        raise ValueError("Password is required for decryption.")

    try:
        envelope = json.loads(encrypted_blob.decode("utf-8"))
        salt = base64.b64decode(envelope["salt_b64"])
        nonce = base64.b64decode(envelope["nonce_b64"])
        ciphertext = base64.b64decode(envelope["ciphertext_b64"])
    except Exception as exc:
        raise InvalidPayloadError("Invalid encrypted file format.") from exc

    key = _derive_key(password, salt)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, None)
    except Exception as exc:
        raise AuthenticationError("Incorrect password or corrupted encrypted file.") from exc


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
