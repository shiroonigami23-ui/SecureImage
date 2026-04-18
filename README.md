# SecureImage

SecureImage is a privacy-first Streamlit app for:

- Steganography: hide secret text inside an image using LSB embedding.
- Authenticated encryption: protect hidden messages with AES-256-GCM.
- Full file encryption: encrypt/decrypt entire image files with password-based keys.

Live app: https://secureimage.streamlit.app/

Repository: https://github.com/shiroonigami23-ui/SecureImage

## Why this upgrade

The project now uses a production-ready algorithm instead of a research mock flow. The new pipeline includes:

- Capacity-aware LSB embedding in RGB channels.
- Structured payload framing with `SIMG` signature and length header.
- PBKDF2-HMAC-SHA256 key derivation (250,000 iterations).
- AES-256-GCM encryption for confidentiality + integrity.
- SHA-256 checksum validation after extraction.
- Zlib compression before embedding to improve effective capacity.

## Features

- Encode secret text into PNG/JPG/JPEG/BMP/WebP cover images.
- Optional password protection for hidden messages.
- Decode hidden messages and verify payload integrity.
- Encrypt raw image files to `.simg` format.
- Decrypt `.simg` files back to original image bytes.
- Fully local processing in the running app session.

## Tech stack

- Python 3.10+
- Streamlit
- Pillow
- NumPy
- Cryptography (AES-GCM)

## Installation

```bash
git clone https://github.com/shiroonigami23-ui/SecureImage.git
cd SecureImage
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Usage

### 1) Steganography (hide message)

1. Open the `Steganography` tab.
2. Choose `Encode message`.
3. Upload a cover image.
4. Enter the secret message.
5. (Optional) add a password for AES-GCM protection.
6. Download the generated PNG.

### 2) Steganography (reveal message)

1. Open `Steganography` -> `Decode message`.
2. Upload the encoded image.
3. Enter password if one was used.
4. Decode and read recovered text.

### 3) Full image encryption

1. Open `File Encryption`.
2. Choose `Encrypt image file`.
3. Upload an image + set password.
4. Download encrypted `.simg` file.
5. Use `Decrypt .simg file` with the same password to recover bytes.

## Security notes

- For steganography output, prefer PNG (lossless). Lossy recompression (like social media JPEG re-encode) can destroy hidden data.
- Use strong passwords for encrypted payloads.
- Hidden payload existence is not deniable against advanced steganalysis; encryption protects message content.
- No system is perfect. Treat this as strong practical security, not military-grade deniability.

## Suggested repository metadata

Description:
`SecureImage: Streamlit app for AES-GCM protected image steganography and image file encryption.`

Topics:
`steganography, image-encryption, aes-gcm, cybersecurity, streamlit, python, privacy, lsb, cryptography`

## License

Add your preferred license file (MIT is a common default for open-source projects).
