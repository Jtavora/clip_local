# apiMarqoFashionClip.py
import io
import os
from contextlib import nullcontext
from typing import Any, Optional

import modal

app = modal.App("marqo-fashion-clip-embeddings-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]==0.115.0",
        "pydantic==2.8.2",
        "pillow==10.4.0",
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "huggingface_hub==0.24.6",
        "hf_transfer==0.1.8",
        "open-clip-torch==2.32.0",
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
    gpu="L4",
    timeout=60 * 10,
    scaledown_window=60 * 5,
    max_containers=10,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name("clip-api-secret")],
)
@modal.concurrent(max_inputs=32)
class MarqoFashionCLIPAPI:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.MODEL_ID = os.getenv(
            "MARQO_FASHION_CLIP_MODEL_ID",
            os.getenv("CLIP_MODEL_ID", "Marqo/marqo-fashionCLIP"),
        )
        self.NORMALIZE = os.getenv("EMBEDDING_L2_NORMALIZE", "true").lower() == "true"
        self.USE_AMP = os.getenv("USE_AMP_FP16", "true").lower() == "true"
        self.MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2048"))
        self.MAX_TEXTS_PER_REQUEST = int(os.getenv("MAX_TEXTS_PER_REQUEST", "128"))
        self.MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "16"))

        self.API_KEY = os.getenv("EMBEDDINGS_API_KEY")
        self.HF_TOKEN = os.getenv("HF_TOKEN")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = (self.device == "cuda") and self.USE_AMP

        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            token=self.HF_TOKEN or None,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            token=self.HF_TOKEN or None,
        )
        self.EMBED_DIM = self._infer_embedding_dim()

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
        return nullcontext()

    def _maybe_normalize(self, x):
        import torch

        x = x.float()
        if not self.NORMALIZE:
            return x
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def _extract_tensor(self, out: Any, kind: str):
        import torch

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

    def _limit_image(self, pil_img):
        if self.MAX_IMAGE_SIDE and max(pil_img.size) > self.MAX_IMAGE_SIDE:
            pil_img.thumbnail((self.MAX_IMAGE_SIDE, self.MAX_IMAGE_SIDE))
        return pil_img

    def _make_crops(self, img):
        w, h = img.size
        if w <= 0 or h <= 0:
            return {"full": img}
        return {
            "full": img,
            "upper": img.crop((0, 0, w, int(h * 0.55))),
            "lower": img.crop((0, int(h * 0.40), w, h)),
            "center": img.crop((0, int(h * 0.20), w, int(h * 0.85))),
        }

    def _embed_texts(self, texts: list[str]):
        import torch

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)

        with torch.inference_mode():
            with self._autocast_ctx():
                try:
                    out = self.model.get_text_features(input_ids=input_ids, normalize=False)
                except TypeError:
                    out = self.model.get_text_features(input_ids=input_ids)

        feats = self._extract_tensor(out, kind="text")
        return self._maybe_normalize(feats)

    def _embed_images_pil_batch(self, images):
        import torch

        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.inference_mode():
            with self._autocast_ctx():
                try:
                    out = self.model.get_image_features(pixel_values=pixel_values, normalize=False)
                except TypeError:
                    out = self.model.get_image_features(pixel_values=pixel_values)

        feats = self._extract_tensor(out, kind="image")
        return self._maybe_normalize(feats)

    def _infer_embedding_dim(self) -> int:
        config = getattr(self.model, "config", None)
        if config is not None:
            projection_dim = getattr(config, "projection_dim", None)
            if isinstance(projection_dim, int) and projection_dim > 0:
                return projection_dim

            hidden_size = getattr(config, "hidden_size", None)
            if isinstance(hidden_size, int) and hidden_size > 0:
                return hidden_size

        try:
            feats = self._embed_texts(["test"])
            if feats.ndim == 2 and feats.shape[-1] > 0:
                return int(feats.shape[-1])
        except Exception:
            pass

        return 0

    @modal.asgi_app()
    def web(self):
        import torch
        from fastapi import FastAPI, File, Header, HTTPException, UploadFile
        from pydantic import BaseModel, Field
        from PIL import Image

        api = FastAPI(title="Marqo FashionCLIP Embeddings API (Modal)", version="1.0.0")

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
                "max_image_side": self.MAX_IMAGE_SIDE,
                "max_texts_per_request": self.MAX_TEXTS_PER_REQUEST,
                "max_images_per_request": self.MAX_IMAGES_PER_REQUEST,
                "trust_remote_code": True,
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

        @api.post("/v1/embeddings/image")
        async def embeddings_image(
            file: UploadFile = File(...),
            x_api_key: Optional[str] = Header(default=None),
        ):
            self._require_key(x_api_key)

            b = await file.read()
            if not b:
                raise HTTPException(status_code=400, detail="empty file")

            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
                img = self._limit_image(img)
            except Exception:
                raise HTTPException(status_code=400, detail="invalid image file")

            feats = self._embed_images_pil_batch([img])
            if feats.ndim != 2 or feats.shape[0] != 1:
                raise RuntimeError("Unexpected embedding batch shape")

            return {
                "model": self.MODEL_ID,
                "embedding_dim": self.EMBED_DIM,
                "object": "embedding",
                "embedding": feats[0].detach().cpu().tolist(),
            }

        @api.post("/v1/embeddings/image/batch")
        async def embeddings_image_batch(
            files: list[UploadFile] = File(...),
            x_api_key: Optional[str] = Header(default=None),
        ):
            self._require_key(x_api_key)

            if not files:
                raise HTTPException(status_code=400, detail="no files provided")
            if len(files) > self.MAX_IMAGES_PER_REQUEST:
                raise HTTPException(status_code=413, detail="too many images")

            imgs = []
            for f in files:
                b = await f.read()
                if not b:
                    raise HTTPException(status_code=400, detail=f"empty: {f.filename}")
                try:
                    im = Image.open(io.BytesIO(b)).convert("RGB")
                    imgs.append(self._limit_image(im))
                except Exception:
                    raise HTTPException(status_code=400, detail=f"invalid: {f.filename}")

            feats = self._embed_images_pil_batch(imgs)
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

        @api.post("/v1/embeddings/image/multi")
        async def embeddings_image_multi(
            file: UploadFile = File(...),
            x_api_key: Optional[str] = Header(default=None),
        ):
            self._require_key(x_api_key)

            b = await file.read()
            if not b:
                raise HTTPException(status_code=400, detail="empty file")

            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
                img = self._limit_image(img)
            except Exception:
                raise HTTPException(status_code=400, detail="invalid image file")

            crops = self._make_crops(img)
            names = list(crops.keys())
            imgs = [crops[n] for n in names]

            feats = self._embed_images_pil_batch(imgs)
            if feats.ndim != 2 or feats.shape[0] != len(names):
                raise RuntimeError("Unexpected embedding batch shape")

            return {
                "model": self.MODEL_ID,
                "embedding_dim": self.EMBED_DIM,
                "object": "embedding_multi",
                "embeddings": {
                    names[i]: feats[i].detach().cpu().tolist() for i in range(len(names))
                },
            }

        return api

