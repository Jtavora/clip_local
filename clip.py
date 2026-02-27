# main.py
#
# Rodar local:
#   pip install fastapi uvicorn torch transformers pillow
#   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
# Endpoints:
#   GET  /health
#   POST /v1/embeddings/text         {"texts": ["...","..."]}
#   POST /v1/embeddings/image        multipart form: file=@img.jpg
#   POST /v1/embeddings/image/multi  multipart form: file=@img.jpg
#
# Env vars:
#   CLIP_MODEL_ID (default: openai/clip-vit-large-patch14)
#   EMBEDDING_L2_NORMALIZE (default: "true")
#   USE_AMP_FP16 (default: "true")

import io
import os
from typing import Any

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-large-patch14")
NORMALIZE = os.getenv("EMBEDDING_L2_NORMALIZE", "true").lower() == "true"

app = FastAPI(title="CLIP Embeddings API (Local)", version="1.0.4")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda") and (os.getenv("USE_AMP_FP16", "true").lower() == "true")

# Load 1x
model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)
EMBED_DIM = int(model.config.projection_dim)


class TextEmbeddingsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


def _to_device(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in d.items()}


def _maybe_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if not NORMALIZE:
        return x
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def _autocast_ctx():
    if use_amp:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def _pick_or_project(
    *,
    out: Any,
    proj_layer: torch.nn.Linear,
    embed_dim: int,
    kind: str,  # "text" ou "image" (só pra msg de erro)
) -> torch.Tensor:
    """
    Compatível com diferenças de versões do transformers:
    - Se out já é Tensor => retorna.
    - Se existir {text,image}_embeds => retorna.
    - Senão usa pooler_output:
        * Se já tem embed_dim => retorna (já projetado)
        * Se tem in_features da projeção => projeta
        * Caso contrário => erro explicando shapes
    """
    if isinstance(out, torch.Tensor):
        return out

    embeds_attr = f"{kind}_embeds"  # text_embeds / image_embeds
    embeds = getattr(out, embeds_attr, None)
    if embeds is not None:
        return embeds

    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        raise TypeError(
            f"Unexpected output from get_{kind}_features: {type(out)} "
            f"(no Tensor, no {embeds_attr}, no pooler_output)"
        )

    last_dim = pooled.shape[-1]
    if last_dim == embed_dim:
        # Já está no espaço final (ex.: 768)
        return pooled

    in_feats = proj_layer.in_features
    if last_dim == in_feats:
        # Ainda está no hidden size (ex.: 1024 no vision), então projeta
        proj_dtype = proj_layer.weight.dtype
        pooled = pooled.to(dtype=proj_dtype)
        return proj_layer(pooled)

    raise RuntimeError(
        f"{kind}: pooler_output has dim={last_dim}, but expected either "
        f"embed_dim={embed_dim} (already projected) or in_features={in_feats} "
        f"(pre-projection)."
    )


def _make_crops(img: Image.Image) -> dict[str, Image.Image]:
    """
    Gera crops geométricos para reduzir o "efeito da modelo" no embedding.
    Isso aumenta robustez para fotos variadas dos clientes (corpo todo, foco na perna, etc.).
    """
    w, h = img.size
    if w <= 0 or h <= 0:
        return {"full": img}

    # full: imagem inteira
    full = img

    # upper: 0% -> 55% (top/peito/ombros)
    upper = img.crop((0, 0, w, int(h * 0.55)))

    # lower: 40% -> 100% (cintura/pernas - legging)
    lower = img.crop((0, int(h * 0.40), w, h))

    # center: 20% -> 85% (textura/miolo)
    center = img.crop((0, int(h * 0.20), w, int(h * 0.85)))

    return {"full": full, "upper": upper, "lower": lower, "center": center}


def _embed_images_batch(images: list[Image.Image]) -> torch.Tensor:
    """
    Gera embeddings para uma lista de imagens em batch (N, D).
    Reutiliza autocast + pick_or_project + normalização.
    """
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with _autocast_ctx():
        out = model.get_image_features(pixel_values=pixel_values)

    feats = _pick_or_project(
        out=out,
        proj_layer=model.visual_projection,
        embed_dim=EMBED_DIM,
        kind="image",
    )
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
        "vision_proj_in": int(model.visual_projection.in_features),
        "text_proj_in": int(model.text_projection.in_features),
    }


@app.post("/v1/embeddings/text")
@torch.inference_mode()
def embeddings_text(req: TextEmbeddingsRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")

    inputs = processor(
        text=req.texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = _to_device(
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
    )

    with _autocast_ctx():
        out = model.get_text_features(**inputs)

    feats = _pick_or_project(
        out=out,
        proj_layer=model.text_projection,
        embed_dim=EMBED_DIM,
        kind="text",
    )
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
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    feats = _embed_images_batch([image])  # (1, D)

    feats_out = feats[0] if feats.ndim == 2 and feats.shape[0] == 1 else feats

    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "embedding",
        "embedding": feats_out.detach().cpu().tolist(),
    }


@app.post("/v1/embeddings/image/multi")
@torch.inference_mode()
async def embeddings_image_multi(file: UploadFile = File(...)):
    """
    Retorna múltiplos embeddings (full/upper/lower/center) para robustez de busca.
    Ideal quando o usuário final pode enviar fotos em ângulos/cortes variados.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    crops = _make_crops(image)
    names = list(crops.keys())
    images = [crops[name] for name in names]

    feats = _embed_images_batch(images)  # (N, D)
    if feats.ndim != 2 or feats.shape[0] != len(names):
        raise RuntimeError("Unexpected embedding batch shape")

    embeddings: dict[str, list[float]] = {
        names[i]: feats[i].detach().cpu().tolist() for i in range(len(names))
    }

    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "embedding_multi",
        "embeddings": embeddings,
    }