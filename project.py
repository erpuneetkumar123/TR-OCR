import io
import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from starlette.middleware.cors import CORSMiddleware

# === CONFIG ===
MODEL_NAME = "microsoft/trocr-large-handwritten"
BEST_MODEL_PATH = "C:/Users\ASUS\Downloads/best_model.pth"
IMAGE_SIZE = (384, 384)
MAX_LENGTH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model & Processor ===
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Preprocess Function ===
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_cv = np.array(image)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    img_thresh = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 10)
    img_pil = Image.fromarray(img_thresh).convert("RGB")
    img_pil.thumbnail(IMAGE_SIZE, Image.BILINEAR)
    dw, dh = IMAGE_SIZE[0] - img_pil.size[0], IMAGE_SIZE[1] - img_pil.size[1]
    img_padded = ImageOps.expand(img_pil, (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2), fill="white")
    return img_padded

# === Post-processing Correction (if needed) ===
def post_process_with_lm(pred):
    corrections = {
        "acetq": "aceta",
        "acetm": "aceta"
    }
    return corrections.get(pred.strip().lower(), pred.strip().lower())

# === Inference ===
def predict_text(image_pil):
    pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=MAX_LENGTH)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        final_text = post_process_with_lm(decoded)
        return final_text

# === FastAPI Setup ===
app = FastAPI(title="OCR API with TrOCR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "🧠 TrOCR Inference API is running."}

@app.post(" redict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)
        prediction = predict_text(image)
        return {"filename": file.filename, "prediction": prediction}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})