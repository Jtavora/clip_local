# marqoFashionClip.py
#
# Rodar local:
#   uv run uvicorn marqoFashionClip:app --host 0.0.0.0 --port 8000 --reload
#
# Endpoints:
#   GET  /health
#   POST /v1/embeddings/text         {"texts": ["...","..."]}
#   POST /v1/embeddings/image        multipart form: file=@img.jpg
#   POST /v1/embeddings/image/batch  multipart form: files=@img1.jpg files=@img2.jpg
#   POST /v1/embeddings/image/multi  multipart form: file=@img.jpg
#
# Env vars:
#   MARQO_FASHION_CLIP_MODEL_ID (default: Marqo/marqo-fashionCLIP)
#   CLIP_MODEL_ID (fallback opcional)
#   EMBEDDING_L2_NORMALIZE (default: "true")
#   USE_AMP_FP16 (default: "true")
#   MAX_IMAGE_SIDE (default: 2048)
#   MAX_TEXTS_PER_REQUEST (default: 128)
#   MAX_IMAGES_PER_REQUEST (default: 16)
#   HF_TOKEN (opcional)

import io
import os
from contextlib import nullcontext
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_ID = os.getenv(
    "MARQO_FASHION_CLIP_MODEL_ID",
    os.getenv("CLIP_MODEL_ID", "Marqo/marqo-fashionCLIP"),
)
HF_TOKEN = os.getenv("HF_TOKEN")
NORMALIZE = os.getenv("EMBEDDING_L2_NORMALIZE", "true").lower() == "true"
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2048"))
MAX_TEXTS_PER_REQUEST = int(os.getenv("MAX_TEXTS_PER_REQUEST", "128"))
MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "16"))

app = FastAPI(title="Marqo FashionCLIP Embeddings API (Local)", version="1.0.0")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda") and (os.getenv("USE_AMP_FP16", "true").lower() == "true")

# O modelo usa custom code no Hub (AutoModel/AutoProcessor + trust_remote_code=True).
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=HF_TOKEN or None,
).to(device).eval()
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=HF_TOKEN or None,
)


class TextEmbeddingsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


def _autocast_ctx():
    if use_amp:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _maybe_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if not NORMALIZE:
        return x
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def _extract_tensor(out: Any, kind: str) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out

    embed_attr = f"{kind}_embeds"
    embeds = getattr(out, embed_attr, None)
    if embeds is not None:
        return embeds

    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled

    raise TypeError(
        f"Unexpected output from get_{kind}_features: {type(out)} "
        f"(no Tensor, no {embed_attr}, no pooler_output)"
    )


def _limit_image(pil_img: Image.Image) -> Image.Image:
    if MAX_IMAGE_SIDE and max(pil_img.size) > MAX_IMAGE_SIDE:
        pil_img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return pil_img


def _make_crops(img: Image.Image) -> dict[str, Image.Image]:
    w, h = img.size
    if w <= 0 or h <= 0:
        return {"full": img}
    return {
        "full": img,
        "upper": img.crop((0, 0, w, int(h * 0.55))),
        "lower": img.crop((0, int(h * 0.40), w, h)),
        "center": img.crop((0, int(h * 0.20), w, int(h * 0.85))),
    }


def _embed_texts(texts: list[str]) -> torch.Tensor:
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)

    with torch.inference_mode():
        with _autocast_ctx():
            try:
                out = model.get_text_features(input_ids=input_ids, normalize=False)
            except TypeError:
                out = model.get_text_features(input_ids=input_ids)

    feats = _extract_tensor(out, kind="text")
    return _maybe_normalize(feats)


def _embed_images_batch(images: list[Image.Image]) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        with _autocast_ctx():
            try:
                out = model.get_image_features(pixel_values=pixel_values, normalize=False)
            except TypeError:
                out = model.get_image_features(pixel_values=pixel_values)

    feats = _extract_tensor(out, kind="image")
    return _maybe_normalize(feats)


def _infer_embedding_dim() -> int:
    config = getattr(model, "config", None)
    if config is not None:
        projection_dim = getattr(config, "projection_dim", None)
        if isinstance(projection_dim, int) and projection_dim > 0:
            return projection_dim

        hidden_size = getattr(config, "hidden_size", None)
        if isinstance(hidden_size, int) and hidden_size > 0:
            return hidden_size

    try:
        with torch.inference_mode():
            feats = _embed_texts(["test"])
        if feats.ndim == 2 and feats.shape[-1] > 0:
            return int(feats.shape[-1])
    except Exception:
        pass

    return 0


EMBED_DIM = _infer_embedding_dim()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "normalize": NORMALIZE,
        "amp_fp16": use_amp,
        "max_image_side": MAX_IMAGE_SIDE,
        "max_texts_per_request": MAX_TEXTS_PER_REQUEST,
        "max_images_per_request": MAX_IMAGES_PER_REQUEST,
        "trust_remote_code": True,
    }


@app.post("/v1/embeddings/text")
@torch.inference_mode()
def embeddings_text(req: TextEmbeddingsRequest):
    texts = [t for t in (req.texts or []) if t and t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    if len(texts) > MAX_TEXTS_PER_REQUEST:
        raise HTTPException(status_code=413, detail="too many texts")

    feats = _embed_texts(texts)
    data = feats.detach().cpu().tolist()

    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "list",
        "data": [
            {"index": i, "object": "embedding", "embedding": data[i]}
            for i in range(len(data))
        ],
        "usage": {"input_count": len(data)},
    }


@app.post("/v1/embeddings/image")
@torch.inference_mode()
async def embeddings_image(file: UploadFile = File(...)):
    b = await file.read()
    if not b:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        img = _limit_image(img)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    feats = _embed_images_batch([img])
    if feats.ndim != 2 or feats.shape[0] != 1:
        raise RuntimeError("Unexpected embedding batch shape")

    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "embedding",
        "embedding": feats[0].detach().cpu().tolist(),
    }


@app.post("/v1/embeddings/image/batch")
@torch.inference_mode()
async def embeddings_image_batch(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="no files provided")
    if len(files) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(status_code=413, detail="too many images")

    imgs = []
    for f in files:
        b = await f.read()
        if not b:
            raise HTTPException(status_code=400, detail=f"empty: {f.filename}")
        try:
            im = Image.open(io.BytesIO(b)).convert("RGB")
            imgs.append(_limit_image(im))
        except Exception:
            raise HTTPException(status_code=400, detail=f"invalid: {f.filename}")

    feats = _embed_images_batch(imgs)
    data = feats.detach().cpu().tolist()
    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "list",
        "data": [
            {"index": i, "object": "embedding", "embedding": data[i]}
            for i in range(len(data))
        ],
        "usage": {"input_count": len(data)},
    }


@app.post("/v1/embeddings/image/multi")
@torch.inference_mode()
async def embeddings_image_multi(file: UploadFile = File(...)):
    b = await file.read()
    if not b:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        img = _limit_image(img)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    crops = _make_crops(img)
    names = list(crops.keys())
    images = [crops[n] for n in names]

    feats = _embed_images_batch(images)
    if feats.ndim != 2 or feats.shape[0] != len(names):
        raise RuntimeError("Unexpected embedding batch shape")

    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "embedding_multi",
        "embeddings": {
            names[i]: feats[i].detach().cpu().tolist() for i in range(len(names))
        },
    }

