import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models, transforms

# ==========================================
# 1. SECURE PARSENET (The Upgrade)
# ==========================================
class SecureParseNet_Inference(nn.Module):
    def __init__(self):
        super().__init__()
        # REAL SOTA BACKBONE: MobileNetV3-Large (Pre-trained on COCO)
        # We use this to prove it works on real images immediately.
        self.model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
        self.model.eval()
        
        # YOUR NOVELTY: Differentiable Integrity Layer
        self.integrity_layer = DiffPHash(hash_size=8)

    def forward(self, x):
        # 1. Segmentation Pass
        output = self.model(x)['out']
        
        # 2. Integrity Hashing (novelty)
        # We hash the feature distribution, not just pixel values
        semantic_probs = F.softmax(output, dim=1)
        integrity_hash = self.integrity_layer(semantic_probs)
        return output, integrity_hash

class DiffPHash(nn.Module):
    """
    Your Enhanced Differentiable Perceptual Hash.
    Uses Soft-Thresholding (Tanh) to allow gradient flow during training.
    """
    def __init__(self, hash_size=8):
        super().__init__()
        self.hash_size = hash_size
        self.register_buffer('dct_matrix', self._make_dct(32))

    def _make_dct(self, N):
        n = torch.arange(N).float()
        k = torch.arange(N).float().unsqueeze(1)
        return torch.cos(math.pi / N * (n + 0.5) * k)

    def forward(self, x):
        # Flatten channels to create an intensity map
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Bilinear resize to canonical hash size
        x_small = F.interpolate(x, (32, 32), mode='bilinear', align_corners=False)
        
        # DCT Transformation
        freq_domain = self.dct_matrix @ x_small.squeeze(1) @ self.dct_matrix.T
        
        # Extract Low-Frequency Features (Structural Content)
        low_freq = freq_domain[:, :self.hash_size, :self.hash_size].flatten(1)
        
        # Soft Hashing (Tanh) for differentiability
        return torch.tanh(low_freq)

# ==========================================
# 2. NEURO-SPIKE SEGNET (The Invention)
# ==========================================
class NeuroSpikeProcessor:
    def __init__(self):
        self.neuron = SleepWakeNeuron(threshold=0.5, decay=0.8)

    def process_real_image(self, image_tensor):
        """
        Converts a real user image into spike trains and applies gating.
        """
        # 1. Poisson Rate Coding (Pixel Intensity -> Spike Probability)
        # Flatten image: (1, 1, H, W) -> (1, H*W)
        flat_img = image_tensor.view(1, -1)
        T = 60 # Time steps
        
        # Generate spike train based on pixel values
        # Shape: (T, Total_Pixels)
        input_spikes = torch.rand(T, flat_img.shape[1]) < flat_img.repeat(T, 1)
        input_spikes = input_spikes.float()

        # 2. Run Biological Simulation
        wake_activity = []
        sleep_activity = []
        
        # Phase 1: WAKE (Encoding)
        self.neuron.set_sleep(False)
        for t in range(30):
            s = self.neuron(input_spikes[t])
            wake_activity.append(s.sum().item())
            
        # Phase 2: SLEEP (Gating/Pruning)
        self.neuron.set_sleep(True)
        for t in range(30, 60):
            s = self.neuron(input_spikes[t])
            sleep_activity.append(s.sum().item())
            
        return wake_activity, sleep_activity

class SleepWakeNeuron(nn.Module):
    """ The Novel Biomimetic Neuron """
    def __init__(self, threshold=1.0, decay=0.5):
        super().__init__()
        self.base_thresh = threshold
        self.base_decay = decay
        self.mem = 0
        self.is_sleep = False

    def set_sleep(self, state: bool):
        self.is_sleep = state
        self.mem = 0 # Reset potential on phase switch

    def forward(self, input_current):
        # Dynamic Parameter Modulation (The Novelty)
        # Sleep Mode = Higher Leak (Faster decay) + Higher Threshold
        decay = self.base_decay * 0.5 if self.is_sleep else self.base_decay
        thresh = self.base_thresh * 1.5 if self.is_sleep else self.base_thresh
        
        # LIF Dynamics
        self.mem = self.mem * decay + input_current
        spike = (self.mem > thresh).float()
        self.mem = self.mem * (1 - spike) # Reset mechanism
        return spike

# ==========================================
# 3. QUANTUM SUPERPOSITION (Completeness)
# ==========================================
class QuantumLayer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.real_w = nn.Linear(dim, dim)
        self.imag_w = nn.Linear(dim, dim)

    def forward(self, feature_vector):
        # Simulate Complex-Valued Neural Network
        r = self.real_w(feature_vector)
        i = self.imag_w(feature_vector)
        
        # Wave Function Collapse: Magnitude Squared
        collapse = torch.sqrt(r**2 + i**2)
        return F.softmax(collapse, dim=1)