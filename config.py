# config.py - Application configuration and resources

# Model Configuration
DEFAULT_MODEL = "Ateeqq/ai-vs-human-image-detector"
SUPPORTED_MODELS = {
    "Ateeqq/ai-vs-human-image-detector": {
        "name": "SigLIP High-Res Classifier (Recommended)",
        "description": "State-of-the-art SigLIP model fine-tuned on 120,000 high-resolution images (60k Real, 60k AI). Highly accurate and robust on real photographs.",
        "input_size": (224, 224)
    },
    "dima806/ai_vs_real_image_detection": {
        "name": "ViT CIFAKE Classifier (Legacy)",
        "description": "Vision Transformer fine-tuned on the CIFAKE dataset (upscaled 32x32 images). Prone to false-positives on high-res photos.",
        "input_size": (224, 224)
    }
}

# Forensic Metrics Thresholds
CONFIDENCE_THRESHOLD = 0.50  # Above 50% means AI

# Sample Images for Quick Testing
# Contains high-quality, stable public URLs with fallbacks
SAMPLE_IMAGES = {
    "Real Human Portrait": {
        "url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=500",
        "type": "REAL",
        "description": "A high-resolution photograph of a person with natural skin textures, hair strands, and lighting reflections."
    },
    "Real Nature Landscape": {
        "url": "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?q=80&w=500",
        "type": "REAL",
        "description": "Authentic nature photography showing complex natural fractals and light dispersion."
    },
    "AI Generated Landscape": {
        "url": "https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/images/stable_diffusion_15.png",
        "type": "AI",
        "description": "Stable Diffusion generated artistic landscape. Look for smooth blending and geometric inconsistencies."
    },
    "AI Generated Portrait": {
        "url": "https://raw.githubusercontent.com/taki0112/Diffusion-Models-in-Practice/main/assets/sd_sample.png",
        "type": "AI",
        "description": "AI-generated face. Common artifacts include blended ear shapes and mismatched background patterns."
    }
}

# Custom Premium CSS Styling for Glassmorphism & Neon theme
CUSTOM_CSS = """
<style>
/* Core layout overrides */
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}

/* Glassmorphism containers */
div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
    background: rgba(17, 25, 40, 0.55);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.085);
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Neon title gradients */
.neon-text-blue {
    background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

.neon-text-purple {
    background: linear-gradient(90deg, #d946ef 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Results banners */
.banner-real {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
    border-left: 5px solid #10b981;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.banner-ai {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.25) 100%);
    border-left: 5px solid #ef4444;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Progress bar color overrides */
div[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 700;
}

/* Style tabs headers */
button[data-baseweb="tab"] {
    font-size: 1rem;
    font-weight: 600;
    color: #888888;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #a855f7 !important;
    border-bottom-color: #a855f7 !important;
}

/* Footer style */
.footer {
    text-align: center;
    padding: 2rem;
    color: #555555;
    font-size: 0.85rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}
</style>
"""

# Educational content (LaTeX and text)
ELA_EXPLANATION = r"""
### 🔍 Error Level Analysis (ELA)

**Error Level Analysis (ELA)** is a classical digital image forensics technique. It works by intentionally resaving an image at a known error level (e.g., 90% JPEG quality) and computing the absolute difference between the original and the resaved image.

#### 📈 The Mathematics
When an image is saved in the lossy JPEG format, the image is compressed in $8 \times 8$ pixel grids. Each compression step reduces the variance within these grids. 

If an image is modified (e.g., splicing a face, combining parts, or generated in blocks by a neural network), the modified parts will have undergone a different number of compression cycles compared to the original areas. 

When we calculate the absolute difference:
$$D(x,y) = |I_{original}(x,y) - I_{resaved}(x,y)|$$

And scale it to $[0, 255]$:
$$S(x,y) = D(x,y) \times \frac{255}{\max(D)}$$

**Interpretations:**
- **Uniform Brightness / Darkness:** The image has uniform compression history (likely untouched).
- **Localized Bright Highlights:** Areas of high brightness indicate sections that have been edited, modified, or contain high-frequency synthetic details that do not match the surrounding compression rate.
"""

FFT_EXPLANATION = r"""
### 📊 Fast Fourier Transform (FFT) Spectral Analysis

**FFT Spectral Analysis** translates the image from the spatial domain $(x, y)$ to the frequency domain $(u, v)$. 

#### 📈 The Mathematics
The 2D Discrete Fourier Transform (DFT) of an image $f(x, y)$ of size $M \times N$ is defined as:
$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$

We shift the zero-frequency component to the center and compute the logarithmic magnitude spectrum:
$$|F_{log}(u, v)| = \log(1 + |F(u, v)|)$$

#### 🛡️ AI Artifact Detection
Generative AI models (GANs, Diffusion models) generate images using **transposed convolutions** or **upsampling filters** to scale up low-resolution latent vectors. These mathematical upsampling operations introduce **periodic grid artifacts** (similar to the checkerboard effect) in the spatial domain.

While these grids are often invisible to the human eye, they stand out in the frequency domain as:
1. **Bright symmetric dots/peaks** away from the center.
2. **Artificial horizontal or vertical lines** in the high-frequency quadrants.
3. **Regular grid patterns** overlaying the entire spectrum.

Real camera photographs show a smooth, star-like continuous decay from the center outwards with no artificial periodic dots.
"""
