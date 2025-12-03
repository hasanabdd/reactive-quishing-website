from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np

from qr_model import classify_url

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UrlRequest(BaseModel):
    url: str

class ImageRequest(BaseModel):
    image: str  # base64 image

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Quishing backend running"}

@app.post("/predict")
def predict_url(req: UrlRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is empty")

    label_str, raw_pred, proba = classify_url(url)
    backend_label = "phishing" if raw_pred == 1 else "safe"

    return {
        "url": url,
        "label": backend_label,
        "model_label": label_str,
        "score": proba,
    }

@app.post("/scan-qr")
def scan_qr(req: ImageRequest):
    data = req.image

    # Handle data URLs (data:image/png;base64,...)
    if "," in data:
        _, data = data.split(",", 1)

    try:
        img_bytes = base64.b64decode(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Image decode failed")

    detector = cv2.QRCodeDetector()
    qr_text, points, _ = detector.detectAndDecode(img)

    if not qr_text:
        return {"label": "suspicious", "url": None}

    label_str, raw_pred, proba = classify_url(qr_text)
    backend_label = "phishing" if raw_pred == 1 else "safe"

    return {
        "label": backend_label,
        "url": qr_text,
        "model_label": label_str,
        "score": proba,
    }
