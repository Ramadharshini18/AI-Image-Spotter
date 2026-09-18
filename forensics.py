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
        impact = "📸 Camera hardware metadata (Make/Model) detected. (Highly likely Real)"
    else:
        impact = "ℹ️ Metadata present but contains no camera hardware parameters (Neutral)."
        
    return {
        "has_exif": True,
        "details": details,
        "ai_signatures": ai_signatures,
        "verdict_impact": impact
    }

def generate_diagnostic_report(classification: dict, metadata: dict, ela_image: Image.Image, fft_image: Image.Image) -> dict:
    """
    Generates a human-centered, explainable diagnostic report.
    Explains the finding in simple, cautious language ('Likely AI-Generated' or 'Likely Real')
    with 4 structured evidence cards and a collapsible technical details payload.
    """
    fake_prob = float(classification.get("fake_score", 0.5))
    real_prob = float(classification.get("real_score", 0.5))
    is_ai = fake_prob >= 0.5

    # 1. Cautious Verdict Label & Summary
    if is_ai:
        verdict_label = "Likely AI-Generated"
        if fake_prob >= 0.85:
            summary = "Our AI model found strong visual patterns that are more consistent with AI-generated images."
        elif fake_prob >= 0.65:
            summary = "Our AI model found visual patterns that are more consistent with AI-generated images."
        else:
            summary = "Our AI model found visual patterns that lean towards AI-generated images, though the margin is close."
    else:
        verdict_label = "Likely Real"
        if real_prob >= 0.85:
            summary = "Our AI model found visual patterns that are more consistent with a natural photograph than with the AI-generated images it has learned from."
        elif real_prob >= 0.65:
            summary = "Our AI model found visual patterns that are more consistent with a natural photograph than with synthetic images."
        else:
            summary = "Our AI model found visual patterns that lean towards a natural photograph, though the margin is close."

    # 2. Measured forensic values
    ela_std = 0.0
    try:
        ela_gray = ela_image.convert("L")
        ela_std = float(np.array(ela_gray).std())
    except Exception:
        pass

    fft_std = 0.0
    try:
        fft_gray = fft_image.convert("L")
        fft_std = float(np.array(fft_gray).std())
    except Exception:
        pass

    # 3. 4 User-Friendly Evidence Cards
    evidence_cards = []

    # Card 1: Visual Pattern Check
    if is_ai:
        v_status = "Supports AI suspicion"
        v_type = "ai"
        v_icon = "🤖"
        if fake_prob >= 0.85:
            v_explanation = f"The visual characteristics show strong patterns commonly produced by generative AI systems recognized by our model (confidence: {fake_prob:.1%})."
        else:
            v_explanation = f"The visual characteristics lean closer to AI-generated examples recognized by our model (confidence: {fake_prob:.1%})."
    else:
        v_status = "Supports Real"
        v_type = "real"
        v_icon = "📷"
        if real_prob >= 0.85:
            v_explanation = f"The image's visual characteristics are strongly consistent with the real photographs recognized by our model (confidence: {real_prob:.1%})."
        else:
            v_explanation = f"The image's visual characteristics are more consistent with natural photographs recognized by our model (confidence: {real_prob:.1%})."

    evidence_cards.append({
        "category": "Visual Pattern Check",
        "title": "Visual Pattern Check",
        "status": v_status,
        "status_type": v_type,
        "icon": v_icon,
        "explanation": v_explanation
    })

    # Card 2: Image Information
    details = metadata.get("details", {})
    ai_sigs = metadata.get("ai_signatures", [])
    has_camera_data = any(tag in details for tag in ["Make", "Model", "LensModel", "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"])

    if ai_sigs:
        info_status = "Supports AI suspicion"
        info_type = "ai"
        info_icon = "🚨"
        info_explanation = f"Direct AI generator software signatures ({', '.join(ai_sigs)}) were detected in the file's information headers."
    elif has_camera_data:
        make = details.get("Make", "")
        model = details.get("Model", "")
        camera_desc = f"{make} {model}".strip() or "Standard Camera"
        info_status = "Supports Real"
        info_type = "real"
        info_icon = "📸"
        info_explanation = f"Camera hardware information was found ({camera_desc}). While this is typical of a physical camera capture, metadata can sometimes be preserved or edited."
    else:
        info_status = "Inconclusive"
        info_type = "neutral"
        info_icon = "ℹ️"
        info_explanation = "No camera information was found in this file. This does not establish whether the image is real or AI-generated, as web platforms, social media, and screenshots commonly remove metadata."

    evidence_cards.append({
        "category": "Image Information",
        "title": "Image Information",
        "status": info_status,
        "status_type": info_type,
        "icon": info_icon,
        "explanation": info_explanation
    })

    # Card 3: Editing & Compression Check
    if ela_std > 12.0:
        c_status = "Needs attention"
        c_type = "warning"
        c_icon = "⚠️"
        c_explanation = f"Some areas show different compression patterns (measured variance: {ela_std:.1f}). This may be due to editing, repeated saving, or selective image processing."
    else:
        c_status = "Supports Real"
        c_type = "real"
        c_icon = "✅"
        c_explanation = f"Compression patterns appear uniform across the image (measured variance: {ela_std:.1f}), showing no clear signs of localized editing or spliced areas."

    evidence_cards.append({
        "category": "Editing & Compression Check",
        "title": "Editing & Compression Check",
        "status": c_status,
        "status_type": c_type,
        "icon": c_icon,
        "explanation": c_explanation
    })

    # Card 4: Image Pattern Analysis
    if fft_std > 42.0:
        p_status = "Supports AI suspicion"
        p_type = "ai"
        p_icon = "⚠️"
        p_explanation = f"Unusual repeating pixel or frequency patterns were detected (measured variance: {fft_std:.1f}), which can be associated with synthetic or digitally processed images."
    else:
        p_status = "No clear evidence"
        p_type = "neutral"
        p_icon = "✅"
        p_explanation = f"No strong unusual repeating frequency patterns were detected by this check (measured variance: {fft_std:.1f})."

    evidence_cards.append({
        "category": "Image Pattern Analysis",
        "title": "Image Pattern Analysis",
        "status": p_status,
        "status_type": p_type,
        "icon": p_icon,
        "explanation": p_explanation
    })

    # Contradiction / Limitation Analysis
    contradiction_note = None
    if not is_ai and (ela_std > 12.0 or fft_std > 42.0):
        contradiction_note = "Notice: Although the visual model leaned towards a real photo, compression or frequency patterns show localized variations. This frequently happens with images shared on social media, repeatedly saved, or lightly retouched."
    elif is_ai and has_camera_data:
        contradiction_note = "Notice: Camera hardware metadata was found, yet the visual model flagged synthetic generation patterns. Camera metadata can sometimes be preserved when editing or simulating an image."

    # Technical Details Block
    technical_details = {
        "model_name": classification.get("model", "SigLIP High-Res Classifier"),
        "raw_scores": classification.get("raw", []),
        "fake_score": fake_prob,
        "real_score": real_prob,
        "ela_std": round(ela_std, 2),
        "ela_threshold": 12.0,
        "fft_std": round(fft_std, 2),
        "fft_threshold": 42.0,
        "has_exif": metadata.get("has_exif", False),
        "metadata_count": len(details),
        "metadata_tags": details,
        "ai_signatures": ai_sigs
    }

    return {
        "verdict_banner": {
            "verdict": verdict_label,
            "summary": summary,
            "ai_confidence": fake_prob,
            "real_confidence": real_prob
        },
        "evidence_cards": evidence_cards,
        "contradiction_note": contradiction_note,
        "technical_details": technical_details
    }

def generate_forensic_reasons(classification: dict, metadata: dict, ela_image: Image.Image, fft_image: Image.Image) -> list:
    """
    Backward-compatible adapter that maps generate_diagnostic_report cards
    into the legacy reason list structure.
    """
    report = generate_diagnostic_report(classification, metadata, ela_image, fft_image)
    reasons = []
    for card in report["evidence_cards"]:
        reasons.append({
            "type": card["status_type"],
            "icon": card["icon"],
            "label": f"{card['title']} — {card['status']}",
            "detail": card["explanation"]
        })
    return reasons
