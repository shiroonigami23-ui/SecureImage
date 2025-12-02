import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from core_logic import SecureParseNet_Inference, NeuroSpikeProcessor, QuantumLayer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Neuro-Secure Research Suite",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Good UI/UX) ---
st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    h1 { color: #1a237e; font-family: 'Roboto', sans-serif; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #3f51b5; color: white; }
    .metric-container { background-color: white; padding: 15px; border-radius: 10px; border-left: 5px solid #3f51b5; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("🔬 Research Controls")
module = st.sidebar.radio("Select Research Module", 
    ["1. SecureParseNet (Security)", "2. Neuro-Spike SegNet (Energy)", "3. Quantum Attention (Novelty)"])

st.sidebar.markdown("---")
st.sidebar.caption(f"System: {'GPU Active 🚀' if torch.cuda.is_available() else 'CPU Optimization 💻'}")

# --- MODULE 1: SECURE PARSENET ---
if module == "1. SecureParseNet (Security)":
    st.title("🛡️ SecureParseNet: Integrity Verification")
    st.markdown("### Goal: Detect Adversarial Attacks using Differentiable Hashing")
    
    @st.cache_resource
    def load_secure_model():
        return SecureParseNet_Inference()
    
    model = load_secure_model()
    
    uploaded_file = st.file_uploader("Upload Image for Semantic Analysis", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        # Preprocessing
        tfms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = tfms(image).unsqueeze(0)
        
        with col1:
            st.image(image, caption="Original Input", use_container_width=True)
            with st.spinner("Running MobileNetV3 Backbone..."):
                with torch.no_grad():
                    out, hash_clean = model(input_tensor)
            
            mask = out.argmax(1).squeeze().numpy()
            st.image(mask, caption="Semantic Segmentation Mask", use_container_width=True, clamp=True)
            st.info(f"Clean Hash Signature: {hash_clean[0, :6].numpy().round(3)}...")

        with col2:
            st.subheader("Adversarial Simulation")
            epsilon = st.slider("Attack Strength (Epsilon)", 0.0, 0.15, 0.05)
            
            if st.button("RUN ATTACK"):
                # FGSM Logic
                input_tensor.requires_grad = True
                out_adv, _ = model(input_tensor)
                loss = out_adv.mean()
                loss.backward()
                
                noise = input_tensor.grad.sign() * epsilon
                adv_tensor = (input_tensor + noise).detach()
                
                with torch.no_grad():
                    _, hash_adv = model(adv_tensor)
                
                # Verify
                mse = torch.nn.functional.mse_loss(hash_clean, hash_adv).item()
                adv_img_show = transforms.ToPILImage()(adv_tensor.squeeze(0))
                st.image(adv_img_show, caption=f"Adversarial Input (Epsilon={epsilon})", use_container_width=True)
                
                st.metric("Integrity Loss (MSE)", f"{mse:.5f}")
                
                if mse > 0.02:
                    st.error("🚨 TAMPERING DETECTED: Semantic Integrity Violated!")
                else:
                    st.success("✅ SECURE: Hash Invariant.")

# --- MODULE 2: NEURO-SPIKE SEGNET ---
elif module == "2. Neuro-Spike SegNet (Energy)":
    st.title("🧠 Neuro-Spike SegNet")
    st.markdown("### Goal: Reduce Computation via Sleep-Wake Gating")
    
    processor = NeuroSpikeProcessor()
    
    uploaded_file = st.file_uploader("Upload Image for Neuromorphic Encoding", type=['jpg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L").resize((128, 128))
        img_tensor = transforms.ToTensor()(image)
        
        st.image(image, caption="Input for Spike Generation", width=200)
        
        if st.button("RUN NEUROMORPHIC SIMULATION"):
            wake_data, sleep_data = processor.process_real_image(img_tensor)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            time_steps = np.arange(len(wake_data) + len(sleep_data))
            data = wake_data + sleep_data
            
            ax.plot(time_steps[:30], data[:30], color='green', label='Wake Phase (Active)')
            ax.plot(time_steps[30:], data[30:], color='red', label='Sleep Phase (Gating)')
            ax.axvline(x=30, color='black', linestyle='--')
            ax.text(32, max(data)*0.9, "Gating ON", fontsize=12)
            ax.set_ylabel("Total Network Spikes")
            ax.set_xlabel("Time (ms)")
            ax.set_title("Energy Reduction Proof: Sleep-Wake Cycle")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Export Logic
            fn = "spike_analysis.png"
            plt.savefig(fn)
            with open(fn, "rb") as f:
                st.download_button("Download Spike Graph", f, file_name=fn)
            
            # Calculation
            avg_wake = np.mean(wake_data)
            avg_sleep = np.mean(sleep_data)
            savings = ((avg_wake - avg_sleep) / avg_wake) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Wake Spikes (Avg)", int(avg_wake))
            c2.metric("Sleep Spikes (Avg)", int(avg_sleep))
            c3.metric("Energy Saved", f"{savings:.1f}%")

# --- MODULE 3: QUANTUM ATTENTION ---
elif module == "3. Quantum Attention (Novelty)":
    st.title("⚛️ Quantum Superposition Attention")
    st.markdown("### Goal: Visualize Wave Function Collapse")
    
    model = QuantumLayer(dim=128)
    
    if st.button("GENERATE QUANTUM STATE"):
        # Simulate feature vector
        feats = torch.randn(1, 128)
        probs = model(feats).detach().numpy()[0]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(128), probs, color='purple', alpha=0.7)
        ax.set_title(" collapsed Wave Function (|Psi|^2)")
        ax.set_xlabel("Feature Dimension")
        ax.set_ylabel("Probability Amplitude")
        
        st.pyplot(fig)
        
        fn = "quantum_state.png"
        plt.savefig(fn)
        with open(fn, "rb") as f:
            st.download_button("Download Quantum Plot", f, file_name=fn)