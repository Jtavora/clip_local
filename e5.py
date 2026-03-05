# e5.py
#
# Rodar local:
#   pip install fastapi uvicorn torch transformers
#   uvicorn e5:app --host 0.0.0.0 --port 8000 --reload
#
# Endpoints:
#   GET  /health
#   POST /v1/embeddings/text   {"texts": ["...","..."]}
#
# Env vars:
#   TEXT_MODEL_ID (default: intfloat/multilingual-e5-large)
#   EMBEDDING_L2_NORMALIZE (default: "true")
#   USE_AMP_FP16 (default: "true")  # só faz efeito se estiver em CUDA
#   E5_MAX_LENGTH (default: 512)
#   E5_DEFAULT_PREFIX (default: query)

import os
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

MODEL_ID = os.getenv("TEXT_MODEL_ID", "intfloat/multilingual-e5-large")
NORMALIZE = os.getenv("EMBEDDING_L2_NORMALIZE", "true").lower() == "true"
MAX_LENGTH = int(os.getenv("E5_MAX_LENGTH", "512"))
DEFAULT_PREFIX = os.getenv("E5_DEFAULT_PREFIX", "query").strip().lower()
if DEFAULT_PREFIX not in ("query", "passage"):
    DEFAULT_PREFIX = "query"

app = FastAPI(title="Text Embeddings API (Multilingual E5)", version="1.0.0")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda") and (os.getenv("USE_AMP_FP16", "true").lower() == "true")


class TextEmbeddingsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


def _autocast_ctx():
    if use_amp:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def _to_device(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in d.items()}


def _average_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / torch.clamp(attention_mask.sum(dim=1)[..., None], min=1e-9)


def _maybe_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if not NORMALIZE:
        return x
    return F.normalize(x, p=2, dim=-1)


def _format_e5_text(text: str) -> str:
    t = text.strip()
    tl = t.lower()
    if tl.startswith("query:") or tl.startswith("passage:"):
        return t
    return f"{DEFAULT_PREFIX}: {t}"


# Load 1x
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
EMBED_DIM = int(getattr(model.config, "hidden_size", 1024))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "normalize": NORMALIZE,
        "amp_fp16": use_amp,
        "max_length": MAX_LENGTH,
        "default_prefix": DEFAULT_PREFIX,
    }


@app.post("/v1/embeddings/text")
@torch.inference_mode()
def embeddings_text(req: TextEmbeddingsRequest):
    texts = [t for t in (req.texts or []) if t and t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")

    formatted = [_format_e5_text(t) for t in texts]

    encoded = tokenizer(
        formatted,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = _to_device(
        {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
    )

    with _autocast_ctx():
        out = model(**encoded)

    feats = _average_pooling(out.last_hidden_state, encoded["attention_mask"])
    feats = _maybe_normalize(feats)

    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "list",
        "data": [
            {"index": i, "object": "embedding", "embedding": emb}
            for i, emb in enumerate(feats.detach().cpu().tolist())
        ],
        "usage": {"input_count": len(texts)},
    }
