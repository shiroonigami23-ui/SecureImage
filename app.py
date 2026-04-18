from __future__ import annotations

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

st.title("SecureImage")
st.caption("Hide secret messages inside images and encrypt image files locally.")
st.markdown("Live app: [https://secureimage.streamlit.app/](https://secureimage.streamlit.app/)")

st.info("All processing runs locally in this Streamlit session. Use PNG for steganography output to avoid lossy compression.")


def _open_image_from_upload(uploaded_file) -> Image.Image:
    try:
        return Image.open(uploaded_file)
    except UnidentifiedImageError as exc:
        raise ValueError("Unsupported image format.") from exc


tab1, tab2 = st.tabs(["Steganography", "File Encryption"])

with tab1:
    st.subheader("Hide or reveal messages")
    mode = st.radio("Select action", ["Encode message", "Decode message"], horizontal=True)

    if mode == "Encode message":
        uploaded = st.file_uploader("Upload cover image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="encode_upload")
        message = st.text_area("Secret message", placeholder="Type your hidden message...")
        password = st.text_input("Password (optional, recommended)", type="password")

        if uploaded:
            cover = _open_image_from_upload(uploaded)
            cap = capacity_for_text(cover)
            st.write(f"Estimated max payload capacity: **{cap} bytes**")
            st.image(cover, caption="Cover image", use_container_width=True)

        if st.button("Encode and download", type="primary"):
            if not uploaded:
                st.error("Please upload an image first.")
            elif not message.strip():
                st.error("Please enter a message.")
            else:
                try:
                    cover = _open_image_from_upload(uploaded)
                    encoded = encode_message_into_image(cover, message.strip(), password=password.strip() or None)
                    png_data = image_to_png_bytes(encoded)
                    st.success("Message encoded successfully.")
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
        uploaded = st.file_uploader("Upload encoded image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="decode_upload")
        password = st.text_input("Password (if used during encode)", type="password", key="decode_password")

        if st.button("Decode message", type="primary"):
            if not uploaded:
                st.error("Please upload an image first.")
            else:
                try:
                    encoded_image = _open_image_from_upload(uploaded)
                    decoded = decode_message_from_image(encoded_image, password=password.strip() or None)
                    st.success("Message extracted successfully.")
                    st.text_area("Recovered message", value=decoded, height=180)
                except AuthenticationError as exc:
                    st.error(str(exc))
                except InvalidPayloadError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Decoding failed: {exc}")

with tab2:
    st.subheader("Encrypt or decrypt full image files")
    mode = st.radio("Select action", ["Encrypt image file", "Decrypt .simg file"], horizontal=True, key="file_mode")

    if mode == "Encrypt image file":
        uploaded = st.file_uploader("Upload image to encrypt", type=["png", "jpg", "jpeg", "bmp", "gif", "webp"], key="file_encrypt")
        password = st.text_input("Encryption password", type="password", key="file_encrypt_password")

        if st.button("Encrypt file", type="primary", key="encrypt_btn"):
            if not uploaded:
                st.error("Please upload an image first.")
            elif not password.strip():
                st.error("Password is required.")
            else:
                try:
                    encrypted_blob = encrypt_image_file_bytes(uploaded.getvalue(), password.strip())
                    original_name = uploaded.name.rsplit(".", 1)[0]
                    st.success("Image file encrypted successfully.")
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
                    decrypted_bytes = decrypt_image_file_bytes(uploaded.getvalue(), password.strip())
                    ext = output_ext.strip().lower().replace(".", "") or "png"
                    st.success("File decrypted successfully.")
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
                        st.warning("Recovered bytes were decrypted, but preview could not be rendered as an image.")
                except AuthenticationError as exc:
                    st.error(str(exc))
                except InvalidPayloadError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Decryption failed: {exc}")
