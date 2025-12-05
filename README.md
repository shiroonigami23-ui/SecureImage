# 🔒 SecureImage

![Status](https://img.shields.io/badge/Status-Live-success)
![Security](https://img.shields.io/badge/Security-Client_Side-green)
![Privacy](https://img.shields.io/badge/Privacy-No_Server_Uploads-blue)

> **A secure web-based tool for Image Steganography and Encryption.**

**SecureImage** is a privacy-focused application that allows users to hide sensitive messages inside images (Steganography) or encrypt image files directly in the browser. Whether you want to send secret messages or protect your photos from prying eyes, SecureImage performs all operations locally on your device, ensuring your data never leaves your browser.

---

## 🔗 Live Demo

**Protect your images now:**
### [🛡️ Launch SecureImage](https://shiroonigami23-ui.github.io/SecureImage/)

---

## ✨ Key Features

### 🕵️‍♂️ Image Steganography (Hide Text)
- **Hide Data:** Embed secret text messages or passwords invisibly inside standard image files (PNG/JPG).
- **Extract Data:** Decrypt and reveal hidden messages from processed images using a passkey.
- **LSB Algorithm:** Uses Least Significant Bit manipulation for minimal visual distortion.

### 🔐 Image Encryption
- **Scramble & Lock:** Encrypt the entire visual content of an image using a password.
- **Secure Decryption:** Restore the original image only with the correct credentials.
- **Client-Side Processing:** All encryption happens in your browser via WebCrypto API or JavaScript—no images are ever uploaded to a server.

### 📂 File Management
- **Drag & Drop:** Easy upload interface.
- **Instant Download:** Save your secured images immediately after processing.
- **Format Support:** Works with standard web image formats (PNG, JPEG).

---

## 🎮 How to Use

### To Hide a Message (Steganography)
1. **Upload:** Drag and drop an image into the "Encode" section.
2. **Type:** Enter the secret message you want to hide.
3. **Password (Optional):** Set a password for extra security.
4. **Encode:** Click the button to generate the new image.
5. **Download:** Save the resulting image. It will look identical to the original!

### To Reveal a Message
1. **Upload:** Select the image containing the secret message in the "Decode" section.
2. **Unlock:** Enter the password (if one was set).
3. **Reveal:** Click "Decode" to see the hidden text.

---

## 📸 Screenshots

| Encoding (Hiding) | Decoding (Revealing) |
|:---:|:---:|
| *Interface for hiding text* | *Interface for extracting text* |

---

## 💻 Local Installation

To run this tool locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shiroonigami23-ui/SecureImage.git](https://github.com/shiroonigami23-ui/SecureImage.git)
   
