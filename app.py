from __future__ import annotations

import math
import secrets
import string
from datetime import datetime
from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from core_logic import (
    AuthenticationError,
    CapacityError,
    InvalidPayloadError,
    capacity_for_text,
    decode_message_from_image,
    decrypt_image_file_bytes,
    encode_message_into_image,
    encrypt_image_file_bytes,
    image_to_png_bytes,
)

st.set_page_config(page_title="SecureImage", page_icon="??", layout="wide")


CSS = """
<style>
:root {
  --bg-top: #0b1220;
  --bg-mid: #0f1b2e;
  --bg-bottom: #101827;
  --card: rgba(17, 25, 40, 0.78);
  --line: rgba(255, 255, 255, 0.12);
  --text: #e8eefc;
  --muted: #aebad3;
  --accent: #36d6b4;
  --accent2: #7cf5dc;
  --danger: #ff6b6b;
}

.stApp {
  background: radial-gradient(circle at 20% 10%, #172a44 0%, var(--bg-top) 45%),
              radial-gradient(circle at 80% 0%, #1f2945 0%, transparent 35%),
              linear-gradient(145deg, var(--bg-mid), var(--bg-bottom));
  color: var(--text);
}

.hero {
  position: relative;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.2rem 1.2rem 1rem 1.2rem;
  background: linear-gradient(120deg, rgba(54,214,180,0.15), rgba(124,245,220,0.08), rgba(255,255,255,0.02));
  overflow: hidden;
  animation: fadeIn 0.8s ease-out;
}

.hero::after {
  content: "";
  position: absolute;
  inset: -60% -15% auto -15%;
  height: 200px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
  transform: rotate(8deg);
  animation: sweep 6s linear infinite;
}

.card {
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1rem;
  background: var(--card);
  backdrop-filter: blur(6px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
  animation: rise 0.5s ease-out;
}

.small-note {
  color: var(--muted);
  font-size: 0.92rem;
}

.metric-pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.32rem 0.75rem;
  display: inline-block;
  margin-right: 0.4rem;
  margin-top: 0.2rem;
  background: rgba(255,255,255,0.03);
}

.history-item {
  border-left: 2px solid var(--accent);
  padding: 0.5rem 0.7rem;
  margin-bottom: 0.4rem;
  background: rgba(255,255,255,0.03);
  border-radius: 8px;
}

@keyframes sweep {
  from { transform: translateX(-120%) rotate(8deg); }
  to { transform: translateX(120%) rotate(8deg); }
}

@keyframes rise {
  from { transform: translateY(6px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.stButton > button {
  border-radius: 10px !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  transition: transform 0.18s ease, box-shadow 0.2s ease !important;
}

.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 18px rgba(0, 0, 0, 0.28);
}

[data-testid="stMetricValue"] {
  color: #f5f8ff;
}
</style>
"""


def _init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("encode_password", "")
    st.session_state.setdefault("file_encrypt_password", "")


def _add_history(action: str, detail: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["history"].insert(0, {"ts": ts, "action": action, "detail": detail})
    st.session_state["history"] = st.session_state["history"][:12]


def _open_image_from_upload(uploaded_file) -> Image.Image:
    try:
        return Image.open(uploaded_file)
    except UnidentifiedImageError as exc:
        raise ValueError("Unsupported image format.") from exc


def _password_strength(password: str) -> tuple[str, float]:
    if not password:
        return "No password", 0.0

    pool_bonus = 0
    if any(c.islower() for c in password):
        pool_bonus += 1
    if any(c.isupper() for c in password):
        pool_bonus += 1
    if any(c.isdigit() for c in password):
        pool_bonus += 1
    if any(c in string.punctuation for c in password):
        pool_bonus += 1

    length_score = min(len(password) / 20.0, 1.0)
    variety_score = pool_bonus / 4.0
    score = (length_score * 0.6) + (variety_score * 0.4)

    if score < 0.35:
        return "Weak", score
    if score < 0.7:
        return "Moderate", score
    return "Strong", score


def _generate_password(length: int = 18) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        label, _ = _password_strength(pwd)
        if label == "Strong":
            return pwd


def _human_size(size: int) -> str:
    if size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    idx = min(int(math.log(size, 1024)), len(units) - 1)
    value = size / (1024 ** idx)
    return f"{value:.2f} {units[idx]}"


def _history_panel() -> None:
    st.sidebar.markdown("### Activity")
    if not st.session_state["history"]:
        st.sidebar.caption("No actions yet.")
        return

    for item in st.session_state["history"]:
        st.sidebar.markdown(
            f"<div class='history-item'><b>{item['action']}</b><br><span class='small-note'>{item['detail']} | {item['ts']}</span></div>",
            unsafe_allow_html=True,
        )


_init_state()
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
  <h1 style="margin:0;">SecureImage</h1>
  <p class="small-note" style="margin:0.25rem 0 0.5rem 0;">Next-gen local privacy toolkit for steganography and image encryption.</p>
  <span class="metric-pill">AES-256-GCM</span>
  <span class="metric-pill">LSB Steganography</span>
  <span class="metric-pill">PBKDF2 250K</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("Live app: [https://secureimage.streamlit.app/](https://secureimage.streamlit.app/)")
st.info("Everything runs locally in your session. For hidden-message images, always download as PNG.")

with st.sidebar:
    st.markdown("## Control Center")
    st.caption("Generate passwords, review activity, and monitor operation safety.")

    if st.button("Generate Strong Password", use_container_width=True):
        generated = _generate_password()
        st.session_state["encode_password"] = generated
        st.session_state["file_encrypt_password"] = generated
        st.success("New strong password generated and autofilled.")

    st.markdown("---")
    _history_panel()


tab1, tab2, tab3 = st.tabs(["Steganography", "File Encryption", "Security Guide"])

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Hide or reveal secret text")
    mode = st.radio("Select action", ["Encode message", "Decode message"], horizontal=True)

    if mode == "Encode message":
        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            uploaded = st.file_uploader(
                "Upload cover image",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                key="encode_upload",
            )
            message = st.text_area("Secret message", placeholder="Type your hidden message...")

        with col_right:
            password = st.text_input(
                "Password (optional, recommended)",
                type="password",
                key="encode_password",
            )
            strength_label, strength_score = _password_strength(password)
            st.caption(f"Password strength: {strength_label}")
            st.progress(strength_score)

            if uploaded is not None:
                try:
                    cover = _open_image_from_upload(uploaded)
                    cap = capacity_for_text(cover)
                    msg_bytes = len(message.encode("utf-8")) if message else 0
                    usage = min(msg_bytes / cap, 1.0) if cap > 0 else 1.0

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Capacity", _human_size(cap))
                    c2.metric("Message", _human_size(msg_bytes))
                    c3.metric("Usage", f"{usage * 100:.1f}%")
                    st.progress(usage)
                except Exception:
                    st.warning("Could not analyze capacity for this file.")

        if uploaded:
            try:
                cover = _open_image_from_upload(uploaded)
                st.image(cover, caption="Cover preview", use_container_width=True)
            except Exception as exc:
                st.error(f"Preview failed: {exc}")

        if st.button("Encode and download", type="primary"):
            if not uploaded:
                st.error("Please upload an image first.")
            elif not message.strip():
                st.error("Please enter a message.")
            else:
                try:
                    with st.spinner("Embedding payload with integrity protection..."):
                        cover = _open_image_from_upload(uploaded)
                        encoded = encode_message_into_image(cover, message.strip(), password=password.strip() or None)
                        png_data = image_to_png_bytes(encoded)

                    st.success("Message encoded successfully.")
                    _add_history("Encoded message", f"{_human_size(len(message.encode('utf-8')))} hidden")
                    st.download_button(
                        "Download encoded image (PNG)",
                        data=png_data,
                        file_name="secureimage_encoded.png",
                        mime="image/png",
                    )
                except CapacityError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Encoding failed: {exc}")

    else:
        uploaded = st.file_uploader(
            "Upload encoded image",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="decode_upload",
        )
        password = st.text_input("Password (if used during encode)", type="password", key="decode_password")

        if st.button("Decode message", type="primary"):
            if not uploaded:
                st.error("Please upload an image first.")
            else:
                try:
                    with st.spinner("Extracting and verifying payload..."):
                        encoded_image = _open_image_from_upload(uploaded)
                        decoded = decode_message_from_image(encoded_image, password=password.strip() or None)

                    st.success("Message extracted successfully.")
                    _add_history("Decoded message", f"Recovered {_human_size(len(decoded.encode('utf-8')))}")
                    st.text_area("Recovered message", value=decoded, height=180)
                except AuthenticationError as exc:
                    st.error(str(exc))
                except InvalidPayloadError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Decoding failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Encrypt or decrypt complete image files")
    mode = st.radio("Select action", ["Encrypt image file", "Decrypt .simg file"], horizontal=True, key="file_mode")

    if mode == "Encrypt image file":
        uploaded = st.file_uploader(
            "Upload image to encrypt",
            type=["png", "jpg", "jpeg", "bmp", "gif", "webp"],
            key="file_encrypt",
        )
        password = st.text_input("Encryption password", type="password", key="file_encrypt_password")
        strength_label, strength_score = _password_strength(password)
        st.caption(f"Password strength: {strength_label}")
        st.progress(strength_score)

        if uploaded:
            st.caption(f"Selected file: `{uploaded.name}` ({_human_size(len(uploaded.getvalue()))})")

        if st.button("Encrypt file", type="primary", key="encrypt_btn"):
            if not uploaded:
                st.error("Please upload an image first.")
            elif not password.strip():
                st.error("Password is required.")
            else:
                try:
                    with st.spinner("Encrypting file with AES-GCM..."):
                        encrypted_blob = encrypt_image_file_bytes(uploaded.getvalue(), password.strip())
                        original_name = uploaded.name.rsplit(".", 1)[0]

                    st.success("Image file encrypted successfully.")
                    _add_history("Encrypted file", f"{uploaded.name} -> {original_name}.simg")
                    st.download_button(
                        "Download encrypted file",
                        data=encrypted_blob,
                        file_name=f"{original_name}.simg",
                        mime="application/octet-stream",
                    )
                except Exception as exc:
                    st.error(f"Encryption failed: {exc}")

    else:
        uploaded = st.file_uploader("Upload .simg encrypted file", type=["simg", "json", "txt"], key="file_decrypt")
        password = st.text_input("Decryption password", type="password", key="file_decrypt_password")
        output_ext = st.text_input("Recovered file extension", value="png", help="Example: png, jpg, jpeg")

        if st.button("Decrypt file", type="primary", key="decrypt_btn"):
            if not uploaded:
                st.error("Please upload an encrypted file first.")
            elif not password.strip():
                st.error("Password is required.")
            else:
                try:
                    with st.spinner("Decrypting and validating file..."):
                        decrypted_bytes = decrypt_image_file_bytes(uploaded.getvalue(), password.strip())

                    ext = output_ext.strip().lower().replace(".", "") or "png"
                    st.success("File decrypted successfully.")
                    _add_history("Decrypted file", f"Output as .{ext}")
                    st.download_button(
                        "Download recovered image",
                        data=decrypted_bytes,
                        file_name=f"secureimage_recovered.{ext}",
                        mime=f"image/{ext}",
                    )

                    try:
                        preview = Image.open(BytesIO(decrypted_bytes))
                        st.image(preview, caption="Recovered image preview", use_container_width=True)
                    except Exception:
                        st.warning("Recovered bytes were validly decrypted, but no image preview is available.")
                except AuthenticationError as exc:
                    st.error(str(exc))
                except InvalidPayloadError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Decryption failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Security best practices")
    st.markdown(
        """
- Prefer PNG for encoded images. JPEG recompression can destroy hidden payload bits.
- Use unique passwords with uppercase, lowercase, numbers, and symbols.
- Share encrypted files and passwords over different channels.
- Keep in mind: steganography hides content but does not guarantee deniability against advanced analysis.
- If decode fails after social media upload, try the original uncompressed file.
"""
    )

    if st.button("Clear activity history"):
        st.session_state["history"] = []
        st.success("Activity history cleared.")

    st.markdown("</div>", unsafe_allow_html=True)

