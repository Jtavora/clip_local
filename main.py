import io
import os
from typing import Any

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# =========================
# Config
# =========================
MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-large-patch14")
NORMALIZE = os.getenv("EMBEDDING_L2_NORMALIZE", "true").lower() == "true"

app = FastAPI(title="CLIP Embeddings API", version="1.1.1")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda") and (os.getenv("USE_AMP_FP16", "true").lower() == "true")

model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)
EMBED_DIM = int(model.config.projection_dim)


# =========================
# Schemas
# =========================
class TextEmbeddingsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


# =========================
# Helpers
# =========================
def _ensure_tensor(x: Any, *, name: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    raise TypeError(f"{name} must be a torch.Tensor, got: {type(x)}")


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_tensor(x, name="feats").float()
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def _maybe_normalize(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_tensor(x, name="feats").float()
    return _l2_normalize(x) if NORMALIZE else x


def _to_device(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in d.items()}


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "normalize": NORMALIZE,
        "amp_fp16": use_amp,
    }


@app.post("/v1/embeddings/text")
@torch.inference_mode()
def embeddings_text(req: TextEmbeddingsRequest):
    """
    Retorna embeddings de texto (Tensor) no espaço multimodal do CLIP.
    """
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")

    inputs = processor(
        text=req.texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Pegue APENAS o que o get_text_features usa
    text_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    text_inputs = _to_device(text_inputs)

    if use_amp:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            feats = model.get_text_features(**text_inputs)  # Tensor (N, D)
    else:
        feats = model.get_text_features(**text_inputs)  # Tensor (N, D)

    feats = _maybe_normalize(feats)

    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "list",
        "data": [
            {"index": i, "object": "embedding", "embedding": emb}
            for i, emb in enumerate(feats.detach().cpu().tolist())
        ],
        "usage": {"input_count": len(req.texts)},
    }


@app.post("/v1/embeddings/image")
@torch.inference_mode()
async def embeddings_image(file: UploadFile = File(...)):
    """
    Retorna embeddings de imagem (Tensor) no espaço multimodal do CLIP.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    inputs = processor(images=image, return_tensors="pt")

    # Pegue APENAS o que o get_image_features usa
    image_inputs = {"pixel_values": inputs["pixel_values"]}
    image_inputs = _to_device(image_inputs)

    if use_amp:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            feats = model.get_image_features(**image_inputs)  # Tensor (1, D)
    else:
        feats = model.get_image_features(**image_inputs)  # Tensor (1, D)

    feats = _maybe_normalize(feats)

    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "embedding",
        "embedding": feats.detach().cpu().tolist()[0],
    }