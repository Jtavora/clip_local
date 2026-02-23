import io
import os

import torch
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# =========================
# Config
# =========================
MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-large-patch14")

# IMPORTANTE:
# Não force 1536. Use a dimensão nativa do modelo (projection_dim).
# Se você precisa padronizar dimensões com outros modelos, faça isso no seu pipeline
# com um projetor treinado, não com padding de zeros.
NORMALIZE = os.getenv("EMBEDDING_L2_NORMALIZE", "true").lower() == "true"

app = FastAPI(title="CLIP Embeddings API", version="1.1.0")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Para estabilidade numérica no output, vamos normalizar em float32.
# Em GPU você pode inferir em fp16/bf16, mas normalize/retorne float32.
use_amp = (device == "cuda") and (os.getenv("USE_AMP_FP16", "true").lower() == "true")

model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)

BASE_DIM = int(model.config.projection_dim)


# =========================
# Schemas
# =========================
class TextEmbeddingsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


# =========================
# Helpers
# =========================
def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    # normalize sempre em float32 para reduzir instabilidade
    x = x.float()
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def _maybe_normalize(x: torch.Tensor) -> torch.Tensor:
    return _l2_normalize(x) if NORMALIZE else x.float()


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model": MODEL_ID,
        "embedding_dim": BASE_DIM,
        "normalize": NORMALIZE,
        "amp_fp16": use_amp,
    }


@app.post("/v1/embeddings/text")
@torch.inference_mode()
def embeddings_text(req: TextEmbeddingsRequest):
    """
    Gera embeddings de texto no mesmo espaço multimodal do CLIP.
    Usa o caminho correto do modelo: get_text_features (pooling+projeção corretos).
    """
    inputs = processor(
        text=req.texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if use_amp:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            feats = model.get_text_features(**inputs)  # (N, BASE_DIM)
    else:
        feats = model.get_text_features(**inputs)  # (N, BASE_DIM)

    feats = _maybe_normalize(feats)

    return {
        "model": MODEL_ID,
        "embedding_dim": BASE_DIM,
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
    Gera embeddings de imagem no mesmo espaço multimodal do CLIP.
    Usa o caminho correto do modelo: get_image_features (pooling+projeção corretos).
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if use_amp:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            feats = model.get_image_features(**inputs)  # (1, BASE_DIM)
    else:
        feats = model.get_image_features(**inputs)  # (1, BASE_DIM)

    feats = _maybe_normalize(feats)

    return {
        "model": MODEL_ID,
        "embedding_dim": BASE_DIM,
        "object": "embedding",
        "embedding": feats.detach().cpu().tolist()[0],
    }