# forensics.py - Classical image forensics operations
import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt

def compute_ela(image: Image.Image, quality: int = 90) -> Image.Image:
    """
    Computes Error Level Analysis (ELA) on the provided PIL image.
    Saves the image in-memory at a lower JPEG quality, compares it 
    with the original, and scales the difference for visualization.
    """
    # ELA works on RGB images
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Save to an in-memory JPEG buffer at the target quality
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    
    # Reload the compressed image
    compressed_image = Image.open(buffer)
    
    # Calculate pixel-wise absolute difference
    ela_image = ImageChops.difference(image, compressed_image)
    
    # Calculate the scale factor based on the maximum pixel difference
    extrema = ela_image.getextrema()
    # extrema is a tuple of (min, max) for R, G, B channels
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
        
    # Scale difference to maximize contrast (0-255 range)
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

def compute_fft(image: Image.Image) -> Image.Image:
    """
    Computes the 2D Fast Fourier Transform (FFT) magnitude spectrum
    of the image and colors it using a high-contrast colormap.
    """
    # Convert image to grayscale for frequency analysis
    gray_image = image.convert("L")
    img_array = np.array(gray_image)
    
    # 2D Fast Fourier Transform
    f_transform = np.fft.fft2(img_array)
    # Shift zero-frequency component to center of spectrum
    f_shift = np.fft.fftshift(f_transform)
    
    # Compute magnitude spectrum (logarithmic scaling to make details visible)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # Normalize magnitude spectrum to [0, 255]
    min_val = magnitude_spectrum.min()
    max_val = magnitude_spectrum.max()
    if max_val > min_val:
        normalized = (magnitude_spectrum - min_val) / (max_val - min_val) * 255.0
    else:
        normalized = magnitude_spectrum * 0.0
        
    # Apply a high-end colormap (magma/inferno) using matplotlib for aesthetic appeal
    colormap = plt.get_cmap("inferno")
    colored_array = colormap(normalized / 255.0)  # Maps to float [0, 1] RGBA
    
    # Drop alpha channel and scale back to uint8 [0, 255]
    rgb_array = (colored_array[:, :, :3] * 255.0).astype(np.uint8)
    
    return Image.fromarray(rgb_array)

def extract_metadata(image: Image.Image) -> dict:
    """
    Extracts EXIF metadata from the image and performs analysis
    to detect AI signatures or camera hardware parameters.
    """
    try:
        exif_info = image.getexif()
    except Exception:
        exif_info = None
        
    if not exif_info:
        return {
            "has_exif": False,
            "details": {},
            "ai_signatures": [],
            "verdict_impact": "ℹ️ No metadata headers found. This is typical of screenshots, social media downloads, and AI generations."
        }
        
    details = {}
    ai_signatures = []
    
    # Extract standard tags
    for tag_id, value in exif_info.items():
        tag_name = TAGS.get(tag_id, tag_id)
        
        # Decode byte values safely
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="ignore").strip()
            except Exception:
                pass
                
        details[str(tag_name)] = str(value)
        
        # Check value content for AI generator keywords
        val_lower = str(value).lower()
        ai_keywords = ["stable diffusion", "midjourney", "dall-e", "dalle", "firefly", "adobe firefly", "generative ai", "gan", "sdxl", "flux.1"]
        for kw in ai_keywords:
            if kw in val_lower:
                ai_signatures.append(f"AI marker '{kw}' found in metadata tag '{tag_name}'")

    # Double check common tags that contain generator software names
    software_val = details.get("Software", "").lower()
    if any(x in software_val for x in ["firefly", "midjourney", "dall-e", "dalle", "sdxl", "stable diffusion"]):
        ai_signatures.append(f"AI-signature Software tag: '{details['Software']}'")
        
    # Check if there is camera hardware info (tends to indicate real photo)
    camera_tags = ["Make", "Model", "LensModel", "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"]
    has_camera_data = any(tag in details for tag in camera_tags)
    
    if ai_signatures:
        impact = "🚨 AI Software signature detected directly in EXIF metadata tags! (Highly likely AI)"
    elif has_camera_data:
        impact = "📸 Authentic camera hardware metadata (Make/Model) detected. (Highly likely Real)"
    else:
        impact = "ℹ️ Metadata present but contains no camera hardware parameters (Neutral)."
        
    return {
        "has_exif": True,
        "details": details,
        "ai_signatures": ai_signatures,
        "verdict_impact": impact
    }

def generate_forensic_reasons(classification: dict, metadata: dict, ela_image: Image.Image, fft_image: Image.Image) -> list:
    """
    Generates structured reason objects for classification based on 
    the model prediction, EXIF metadata tags, and image pixel statistical analysis.
    """
    reasons = []
    verdict = classification.get("verdict")
    fake_prob = classification.get("fake_score", 0.5)
    real_prob = classification.get("real_score", 0.5)
    
    # 1. Neural Transformer Check
    if verdict == "AI":
        reasons.append({
            "type": "ai",
            "icon": "🤖",
            "label": "Neural Check",
            "detail": f"The Transformer model flagged generative textures and pixel patch anomalies with {fake_prob:.1%} confidence."
        })
    else:
        reasons.append({
            "type": "real",
            "icon": "🟢",
            "label": "Neural Check",
            "detail": f"The Transformer model verified natural skin, hair, or landscape fractal structures with {real_prob:.1%} confidence."
        })
        
    # 2. EXIF Metadata Auditor
    if metadata["has_exif"]:
        if metadata["ai_signatures"]:
            reasons.append({
                "type": "ai",
                "icon": "🚨",
                "label": "EXIF Audit",
                "detail": f"Direct AI generator software signatures detected in metadata tags: {', '.join(metadata['ai_signatures'])}."
            })
        elif "Make" in metadata["details"] or "Model" in metadata["details"]:
            make = metadata["details"].get("Make", "Unknown")
            model = metadata["details"].get("Model", "Unknown")
            reasons.append({
                "type": "real",
                "icon": "📸",
                "label": "EXIF Audit",
                "detail": f"Found authentic camera hardware metadata (Make: {make}, Model: {model}), indicating physical capture."
            })
        else:
            reasons.append({
                "type": "neutral",
                "icon": "ℹ️",
                "label": "EXIF Audit",
                "detail": "Metadata tags are present but contain no physical camera hardware parameters."
            })
    else:
        reasons.append({
            "type": "neutral",
            "icon": "ℹ️",
            "label": "EXIF Audit",
            "detail": "No metadata headers found. This is typical of screenshots, web downloads, or direct AI exports."
        })
        
    # 3. ELA Pixel Statistics (Standard Deviation of Grayscale ELA highlights)
    try:
        ela_gray = ela_image.convert("L")
        ela_arr = np.array(ela_gray)
        ela_std = ela_arr.std()
        
        if ela_std > 12.0:
            reasons.append({
                "type": "ai",
                "icon": "🔍",
                "label": "Error Level Analysis",
                "detail": f"High compression variance detected (std={ela_std:.1f}). This highlights regional compression offsets typical of digital editing or localized patch rendering."
            })
        else:
            reasons.append({
                "type": "real",
                "icon": "real",
                "label": "Error Level Analysis",
                "detail": f"Compression error levels are uniform (std={ela_std:.1f}). No localized editing or patch splicing detected."
            })
    except Exception:
         pass
         
    # 4. FFT Checkerboard Grid Auditor
    try:
        fft_gray = fft_image.convert("L")
        fft_arr = np.array(fft_gray)
        fft_std = fft_arr.std()
        
        if fft_std > 42.0:
            reasons.append({
                "type": "ai",
                "icon": "📊",
                "label": "FFT Spectrum",
                "detail": f"Bright periodic frequencies detected (std={fft_std:.1f}). This indicates checkerboard upsampling patterns left by neural generator filters."
            })
        else:
            reasons.append({
                "type": "real",
                "icon": "real",
                "label": "FFT Spectrum",
                "detail": f"Grayscale frequency spectrum decays smoothly (std={fft_std:.1f}), characteristic of natural camera exposure."
            })
    except Exception:
        pass
        
    return reasons
