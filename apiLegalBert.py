# modal_legalbert_api.py
import os
from typing import Optional

import modal

app = modal.App("legalbert-text-embeddings-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]==0.115.0",
        "pydantic==2.8.2",
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "huggingface_hub==0.24.6",
        "hf_transfer==0.1.8",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="L4",  # opcional: troque por None para CPU
    timeout=60 * 10,
    scaledown_window=60 * 5,
    max_containers=10,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name("clip-api-secret")],
)
@modal.concurrent(max_inputs=64)
class LegalBertAPI:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.MODEL_ID = os.getenv("TEXT_MODEL_ID", "stjiris/bert-large-portuguese-cased-legal-tsdae-sts-v1")
        self.NORMALIZE = os.getenv("EMBEDDING_L2_NORMALIZE", "true").lower() == "true"
        self.USE_AMP = os.getenv("USE_AMP_FP16", "true").lower() == "true"
        self.MAX_LENGTH = int(os.getenv("LEGAL_MAX_LENGTH", "514"))

        self.API_KEY = os.getenv("EMBEDDINGS_API_KEY")
        self.HF_TOKEN = os.getenv("HF_TOKEN")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = (self.device == "cuda") and self.USE_AMP

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            token=self.HF_TOKEN or None,
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            token=self.HF_TOKEN or None,
        ).to(self.device).eval()

        self.EMBED_DIM = int(getattr(self.model.config, "hidden_size", 1024))
        self.MAX_TEXTS_PER_REQUEST = int(os.getenv("MAX_TEXTS_PER_REQUEST", "128"))

    def _require_key(self, provided: Optional[str]) -> None:
        if not self.API_KEY:
            return
        from fastapi import HTTPException

        if not provided or provided != self.API_KEY:
            raise HTTPException(status_code=401, detail="unauthorized")

    def _autocast_ctx(self):
        import torch

        if self.use_amp:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return torch.autocast(device_type="cpu", enabled=False)

    def _maybe_normalize(self, x):
        import torch

        x = x.float()
        if not self.NORMALIZE:
            return x
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def _mean_pooling(self, model_output, attention_mask):
        import torch

        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        denom = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return summed / denom

    def _embed_texts(self, texts: list[str]):
        import torch

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}

        with torch.inference_mode():
            with self._autocast_ctx():
                out = self.model(**encoded)

        feats = self._mean_pooling(out, encoded["attention_mask"])
        feats = self._maybe_normalize(feats)
        return feats

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel, Field

        api = FastAPI(title="Legal BERTimbau Text Embeddings API (Modal)", version="1.0.0")

        class TextEmbeddingsRequest(BaseModel):
            texts: list[str] = Field(..., min_length=1)

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "device": self.device,
                "model": self.MODEL_ID,
                "embedding_dim": self.EMBED_DIM,
                "normalize": self.NORMALIZE,
                "amp_fp16": self.use_amp,
                "max_texts_per_request": self.MAX_TEXTS_PER_REQUEST,
                "max_length": self.MAX_LENGTH,
            }

        @api.post("/v1/embeddings/text")
        def embeddings_text(
            req: TextEmbeddingsRequest,
            x_api_key: Optional[str] = Header(default=None),
        ):
            self._require_key(x_api_key)

            texts = [t for t in (req.texts or []) if t and t.strip()]
            if not texts:
                raise HTTPException(status_code=400, detail="texts must not be empty")
            if len(texts) > self.MAX_TEXTS_PER_REQUEST:
                raise HTTPException(status_code=413, detail="too many texts")

            feats = self._embed_texts(texts)
            data = feats.detach().cpu().tolist()

            return {
                "model": self.MODEL_ID,
                "embedding_dim": self.EMBED_DIM,
                "object": "list",
                "data": [
                    {"index": i, "object": "embedding", "embedding": data[i]}
                    for i in range(len(data))
                ],
                "usage": {"input_count": len(data)},
            }

        return api
