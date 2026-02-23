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

app = FastAPI(title="CLIP Embeddings API", version="1.2.1")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda") and (os.getenv("USE_AMP_FP16", "true").lower() == "true")

model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)
EMBED_DIM = int(model.config.projection_dim)


class TextEmbeddingsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


def _ensure_tensor(x: Any, *, name: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    raise TypeError(f"{name} must be a torch.Tensor, got: {type(x)}")


def _maybe_normalize(x: torch.Tensor) -> torch.Tensor:
    x = _ensure_tensor(x, name="embeds").float()
    if not NORMALIZE:
        return x
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def _to_device(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in d.items()}


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
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")

    inputs = processor(
        text=req.texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    text_inputs = _to_device(
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
    )

    if use_amp:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            text_outputs = model.text_model(**text_inputs)  # BaseModelOutputWithPooling
    else:
        text_outputs = model.text_model(**text_inputs)

    # last_hidden_state: (N, T, H)
    last_hidden = text_outputs.last_hidden_state

    # Posição do último token válido em cada sequência:
    # attention_mask soma os 1s (tokens válidos) => último índice é sum-1
    last_token_pos = text_inputs["attention_mask"].sum(dim=-1) - 1  # (N,)

    pooled = last_hidden[torch.arange(last_hidden.size(0), device=device), last_token_pos]  # (N, H)

    # Projeção pro espaço multimodal CLIP: (N, D)
    feats = model.text_projection(pooled)

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

    inputs = processor(images=image, return_tensors="pt")
    image_inputs = _to_device({"pixel_values": inputs["pixel_values"]})

    if use_amp:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            vision_outputs = model.vision_model(**image_inputs)  # BaseModelOutputWithPooling
    else:
        vision_outputs = model.vision_model(**image_inputs)

    pooled = vision_outputs.pooler_output  # (1, H)
    feats = model.visual_projection(pooled)  # (1, D)

    feats = _maybe_normalize(feats)

    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "embedding",
        "embedding": feats.detach().cpu().tolist()[0],
    }