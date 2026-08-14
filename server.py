# server.py - FastAPI Backend Server
import io
import base64
import requests
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Import workspace modules
from forensics import compute_ela, compute_fft, extract_metadata, generate_forensic_reasons
from models import classify_image, classify_real_category
from config import DEFAULT_MODEL

app = FastAPI(title="AI Image Spotter Forensics API")

# Configure CORS so the React app on port 5173 can call this server on port 8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits requests from React development port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache to store the last uploaded image
# This allows instant ELA slider recalculation without re-uploading the file
last_uploaded_image = None

def pil_to_base64(img: Image.Image, format: str = "JPEG", quality: int = 95) -> str:
    """Helper to convert a PIL image to a base64 data URI string."""
    buffer = io.BytesIO()
    if img.mode != "RGB" and format == "JPEG":
        img = img.convert("RGB")
    img.save(buffer, format=format, quality=quality)
    img_bytes = buffer.getvalue()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    model_id: str = Query(DEFAULT_MODEL)
):
    global last_uploaded_image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save to cache
        last_uploaded_image = image.copy()
        
        # 1. Run model prediction
        classification = classify_image(image, model_id)
        if not classification.get("success", False):
            raise HTTPException(status_code=500, detail=f"Model error: {classification.get('error')}")
            
        # 2. Run forensics
        metadata = extract_metadata(image)
        ela_img = compute_ela(image, quality=90)
        fft_img = compute_fft(image)
        
        # 3. Generate diagnostic reasoning report
        reasons = generate_forensic_reasons(classification, metadata, ela_img, fft_img)
        
        # Convert visual assets to base64 URIs
        ela_b64 = pil_to_base64(ela_img, format="JPEG", quality=90)
        fft_b64 = pil_to_base64(fft_img, format="JPEG", quality=95)
        
        return {
            "success": True,
            "filename": file.filename,
            "model": model_id,
            "verdict": classification["verdict"],
            "fake_score": classification["fake_score"],
            "real_score": classification["real_score"],
            "raw": classification["raw"],
            "metadata": metadata,
            "reasons": reasons,
            "ela_base64": ela_b64,
            "fft_base64": fft_b64
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")

@app.post("/api/analyze-url")
async def analyze_url_image(
    body: dict,
    model_id: str = Query(DEFAULT_MODEL)
):
    global last_uploaded_image
    url = body.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing image URL")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        
        # Save to cache
        last_uploaded_image = image.copy()
        
        # 1. Run model prediction
        classification = classify_image(image, model_id)
        if not classification.get("success", False):
            raise HTTPException(status_code=500, detail=f"Model error: {classification.get('error')}")
            
        # 2. Run forensics
        metadata = extract_metadata(image)
        ela_img = compute_ela(image, quality=90)
        fft_img = compute_fft(image)
        
        # 3. Generate diagnostic reasoning report
        reasons = generate_forensic_reasons(classification, metadata, ela_img, fft_img)
        
        # Convert visual assets to base64 URIs
        ela_b64 = pil_to_base64(ela_img, format="JPEG", quality=90)
        fft_b64 = pil_to_base64(fft_img, format="JPEG", quality=95)
        
        return {
            "success": True,
            "filename": url.split("/")[-1].split("?")[0] or "url_image.jpg",
            "model": model_id,
            "verdict": classification["verdict"],
            "fake_score": classification["fake_score"],
            "real_score": classification["real_score"],
            "raw": classification["raw"],
            "metadata": metadata,
            "reasons": reasons,
            "ela_base64": ela_b64,
            "fft_base64": fft_b64
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process URL: {str(e)}")

@app.post("/api/ela")
async def update_ela(quality: int = Query(90)):
    global last_uploaded_image
    if last_uploaded_image is None:
        raise HTTPException(status_code=400, detail="No active image cached. Upload an image first.")
        
    try:
        ela_img = compute_ela(last_uploaded_image, quality=quality)
        ela_b64 = pil_to_base64(ela_img, format="JPEG", quality=quality)
        return {"success": True, "ela_base64": ela_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ELA recalculation error: {str(e)}")

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "model": DEFAULT_MODEL}

# Mount static files folder (if frontend is built)
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    
    @app.get("/{fallback:path}")
    def read_index(fallback: str):
        index_path = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"error": "Frontend build files missing."}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    # Enable reload only when running locally on localhost
    uvicorn.run("server:app", host=host, port=port, reload=(host == "127.0.0.1"))
