# main.py
#
# Rodar local:
#   pip install fastapi uvicorn torch transformers pillow
#   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
# Endpoints:
#   GET  /health
#   POST /v1/embeddings/image        multipart form: file=@img.jpg
#   POST /v1/embeddings/image/multi  multipart form: file=@img.jpg
#   POST /v1/embeddings/text         (NÃO SUPORTADO no DINOv2)
#
# Env vars:
#   DINO_MODEL_ID (default: facebook/dinov2-base)
#   EMBEDDING_L2_NORMALIZE (default: "true")
#   USE_AMP_FP16 (default: "true")

import io
import os
from typing import Any

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

MODEL_ID = os.getenv("DINO_MODEL_ID", "facebook/dinov2-base")
NORMALIZE = os.getenv("EMBEDDING_L2_NORMALIZE", "true").lower() == "true"

app = FastAPI(title="DINOv2 Embeddings API (HF)", version="2.0.0")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda") and (os.getenv("USE_AMP_FP16", "true").lower() == "true")


class TextEmbeddingsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


def _autocast_ctx():
    if use_amp:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def _maybe_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if not NORMALIZE:
        return x
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def _make_crops(img: Image.Image) -> dict[str, Image.Image]:
    w, h = img.size
    if w <= 0 or h <= 0:
        return {"full": img}

    full = img
    upper = img.crop((0, 0, w, int(h * 0.55)))
    lower = img.crop((0, int(h * 0.40), w, h))
    center = img.crop((0, int(h * 0.20), w, int(h * 0.85)))
    return {"full": full, "upper": upper, "lower": lower, "center": center}


# Load 1x
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = Dinov2Model.from_pretrained(MODEL_ID).to(device).eval()

# hidden_size do DINOv2 no HF é o dim do embedding que vamos usar (CLS)
EMBED_DIM = int(model.config.hidden_size)


def _embed_images_batch(images: list[Image.Image]) -> torch.Tensor:
    """
    Retorna embeddings (N, D) usando CLS token de last_hidden_state[:, 0, :].
    """
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with _autocast_ctx(), torch.inference_mode():
        out = model(pixel_values=pixel_values)

    if not hasattr(out, "last_hidden_state") or out.last_hidden_state is None:
        raise RuntimeError("Unexpected DINOv2 output: missing last_hidden_state")

    # CLS token
    feats = out.last_hidden_state[:, 0, :]  # (N, D)
    feats = _maybe_normalize(feats)
    return feats


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "normalize": NORMALIZE,
        "amp_fp16": use_amp,
        "dtype": str(next(model.parameters()).dtype),
    }


@app.post("/v1/embeddings/text")
def embeddings_text(_: TextEmbeddingsRequest):
    raise HTTPException(
        status_code=501,
        detail="DINOv2 não suporta embeddings de texto. Use CLIP para /v1/embeddings/text.",
    )


@app.post("/v1/embeddings/image")
async def embeddings_image(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    feats = _embed_images_batch([image])  # (1, D)
    emb = feats[0].detach().cpu().tolist()

    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "embedding",
        "embedding": emb,
    }


@app.post("/v1/embeddings/image/multi")
async def embeddings_image_multi(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    crops = _make_crops(image)
    names = list(crops.keys())
    images = [crops[n] for n in names]

    feats = _embed_images_batch(images)  # (N, D)
    if feats.ndim != 2 or feats.shape[0] != len(names):
        raise RuntimeError("Unexpected embedding batch shape")

    embeddings = {names[i]: feats[i].detach().cpu().tolist() for i in range(len(names))}

    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "embedding_multi",
        "embeddings": embeddings,
    }