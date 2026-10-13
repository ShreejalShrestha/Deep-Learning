"""
FastAPI service for the Nepali Sign Language CNN (Week 4 "Low LR" model).

The model, the preprocessing and the class mapping are copied unchanged from
Shreejal.Shrestha.ipynb so that the live demo sees exactly what the notebook
evaluated. Nothing is trained here: the checkpoint is only loaded.
"""

import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Paths - resolved from the location of this file, not from the shell's cwd.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CKPT_PATH = PROJECT_ROOT / "models" / "low_lr.pt"   # original Week 4 model

IMG_SIZE = 128        # notebook Section 14.2: 300x300 -> 128x128, bilinear
N_CLASSES = 36
DROPOUT = 0.3         # the "Low LR" experiment's dropout
BASE_FILTERS = 32     # run_experiment default

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Class labels (notebook Section 2) - IDs and order must not change.
# ---------------------------------------------------------------------------
CHAR_MAP = {
    0: "\u0915", 1: "\u0916", 2: "\u0917", 3: "\u0918", 4: "\u0919",
    5: "\u091a", 6: "\u091b", 7: "\u091c", 8: "\u091d", 9: "\u091e",
    10: "\u091f", 11: "\u0920", 12: "\u0921", 13: "\u0922", 14: "\u0923",
    15: "\u0924", 16: "\u0925", 17: "\u0926", 18: "\u0927", 19: "\u0928",
    20: "\u092a", 21: "\u092b", 22: "\u092c", 23: "\u092d", 24: "\u092e",
    25: "\u092f", 26: "\u0930", 27: "\u0932", 28: "\u0935", 29: "\u0936",
    30: "\u0937", 31: "\u0938", 32: "\u0939",
    33: "\u0915\u094d\u0937", 34: "\u0924\u094d\u0930", 35: "\u091c\u094d\u091e",
}
ROMAN_MAP = {
    0: "ka", 1: "kha", 2: "ga", 3: "gha", 4: "nga",
    5: "cha", 6: "chha", 7: "ja", 8: "jha", 9: "yna",
    10: "Ta", 11: "Tha", 12: "Da", 13: "Dha", 14: "Na",
    15: "ta", 16: "tha", 17: "da", 18: "dha", 19: "na",
    20: "pa", 21: "pha", 22: "ba", 23: "bha", 24: "ma",
    25: "ya", 26: "ra", 27: "la", 28: "wa", 29: "sha",
    30: "Sha", 31: "sa", 32: "ha", 33: "ksha", 34: "tra", 35: "gya",
}


# ---------------------------------------------------------------------------
# Architecture - copied verbatim from the notebook (Section 14.4).
# ---------------------------------------------------------------------------
class SignCNN(nn.Module):
    """Small VGG-style CNN for the 36 Nepali Sign Language character classes."""

    def __init__(self, n_classes=36, dropout=0.3, base_filters=32):
        super().__init__()
        layers = []
        c_in = 3  # RGB input
        for block in range(4):
            c_out = base_filters * (2 ** block)
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # halve height and width
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GlobalAveragePooling -> one value per filter
            nn.Flatten(),
            nn.Linear(c_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),  # logits; softmax lives in the loss function
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Load the trained weights once, at import time (teacher's pattern).
# ---------------------------------------------------------------------------
if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

model = SignCNN(n_classes=N_CLASSES, dropout=DROPOUT, base_filters=BASE_FILTERS)
state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=True)   # strict: any mismatch must raise
model.to(DEVICE)
model.eval()
print(f"Loaded {CKPT_PATH.name} on {DEVICE} "
      f"({sum(p.numel() for p in model.parameters()):,} parameters)")


# ---------------------------------------------------------------------------
# Inference preprocessing - the notebook's path with augmentation left out.
#   PIL -> RGB -> 128x128 bilinear -> uint8 HWC -> float32 CHW / 255 -> batch
# ---------------------------------------------------------------------------
def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.uint8)                    # HWC uint8 (a copy)
    x = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)  # CHW [0,1]
    return x.unsqueeze(0).to(DEVICE)                         # NCHW on device


def describe(class_id: int, confidence: float) -> dict:
    return {
        "class_id": class_id,
        "roman": ROMAN_MAP[class_id],
        "devanagari": CHAR_MAP[class_id],
        "confidence": round(confidence, 4),
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Nepali Sign Language CNN API")


@app.get("/")
def root():
    return {"message": "Nepali Sign Language CNN API", "status": "ready"}


@app.get("/health")
def health():
    return {
        "status": "ready",
        "model": "SignCNN",
        "checkpoint": CKPT_PATH.name,
        "device": str(DEVICE),
        "classes": N_CLASSES,
        "input_size": [IMG_SIZE, IMG_SIZE],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. read the upload
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 2. decode it as an image
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400,
                            detail=f"Could not read the uploaded image: {exc}")

    # 3-5. preprocess, forward pass, softmax
    try:
        x = preprocess(image)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as a 500
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # 6-8. best class, its confidence, and the top 3
    top_conf, top_idx = torch.topk(probs, k=3)
    top3 = [describe(int(i), float(c)) for c, i in zip(top_conf, top_idx)]
    best = dict(top3[0])
    best["top3"] = top3
    return best


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
