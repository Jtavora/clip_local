# legalBert.py
#
# Rodar local:
#   pip install fastapi uvicorn torch transformers
#   uvicorn legalBert:app --host 0.0.0.0 --port 8000 --reload
#
# Endpoints:
#   GET  /health
#   POST /v1/embeddings/text   {"texts": ["...","..."]}
#
# Env vars:
#   TEXT_MODEL_ID (default: stjiris/bert-large-portuguese-cased-legal-tsdae-sts-v1)
#   EMBEDDING_L2_NORMALIZE (default: "true")
#   USE_AMP_FP16 (default: "true")  # só faz efeito se estiver em CUDA
#   LEGAL_MAX_LENGTH (default: 514)

import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

MODEL_ID = os.getenv("TEXT_MODEL_ID", "stjiris/bert-large-portuguese-cased-legal-tsdae-sts-v1")
NORMALIZE = os.getenv("EMBEDDING_L2_NORMALIZE", "true").lower() == "true"
MAX_LENGTH = int(os.getenv("LEGAL_MAX_LENGTH", "514"))

app = FastAPI(title="Text Embeddings API (Legal BERTimbau)", version="1.0.0")

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


def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    denom = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / denom


def _maybe_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if not NORMALIZE:
        return x
    return F.normalize(x, p=2, dim=-1)


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
    }


@app.post("/v1/embeddings/text")
@torch.inference_mode()
def embeddings_text(req: TextEmbeddingsRequest):
    texts = [t for t in (req.texts or []) if t and t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")

    encoded = tokenizer(
        texts,
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

    feats = _mean_pooling(out, encoded["attention_mask"])
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
