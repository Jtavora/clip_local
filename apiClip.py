# modal_clip_api.py
import io
import os
from typing import Any, Optional

import modal

app = modal.App("clip-embeddings-api")

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
class CLIPAPI:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-large-patch14")
        self.NORMALIZE = os.getenv("EMBEDDING_L2_NORMALIZE", "true").lower() == "true"
        self.USE_AMP = os.getenv("USE_AMP_FP16", "true").lower() == "true"
        self.MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2048"))
        self.MAX_TEXTS_PER_REQUEST = int(os.getenv("MAX_TEXTS_PER_REQUEST", "128"))
        self.MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "16"))

        # vindo do Secret:
        self.API_KEY = os.getenv("EMBEDDINGS_API_KEY")
        self.HF_TOKEN = os.getenv("HF_TOKEN")  # opcional, mas bom ter

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = (self.device == "cuda") and self.USE_AMP

        self.model = CLIPModel.from_pretrained(self.MODEL_ID).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.EMBED_DIM = int(self.model.config.projection_dim)

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

    def _pick_or_project(
        self,
        *,
        out: Any,
        proj_layer,
        embed_dim: int,
        kind: str,
    ):
        import torch

        if isinstance(out, torch.Tensor):
            return out

        embeds_attr = f"{kind}_embeds"
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
            return pooled

        in_feats = proj_layer.in_features
        if last_dim == in_feats:
            pooled = pooled.to(dtype=proj_layer.weight.dtype)
            return proj_layer(pooled)

        raise RuntimeError(
            f"{kind}: pooler_output dim={last_dim}, expected {embed_dim} or {in_feats}"
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

    def _embed_images_pil_batch(self, images):
        import torch

        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.inference_mode():
            with self._autocast_ctx():
                out = self.model.get_image_features(pixel_values=pixel_values)

        feats = self._pick_or_project(
            out=out,
            proj_layer=self.model.visual_projection,
            embed_dim=self.EMBED_DIM,
            kind="image",
        )
        return self._maybe_normalize(feats)

    @modal.asgi_app()
    def web(self):
        import torch
        from fastapi import FastAPI, File, Header, HTTPException, UploadFile
        from pydantic import BaseModel, Field
        from PIL import Image

        api = FastAPI(title="CLIP Embeddings API (Modal)", version="1.0.4")

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
            }

        @api.post("/v1/embeddings/text")
        @torch.inference_mode()
        def embeddings_text(
            req: TextEmbeddingsRequest,
            x_api_key: Optional[str] = Header(default=None),
        ):
            self._require_key(x_api_key)

            if not req.texts:
                raise HTTPException(status_code=400, detail="texts must not be empty")
            if len(req.texts) > self.MAX_TEXTS_PER_REQUEST:
                raise HTTPException(status_code=413, detail="too many texts")

            inputs = self.processor(
                text=req.texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {
                "input_ids": inputs["input_ids"].to(self.device),
                "attention_mask": inputs["attention_mask"].to(self.device),
            }

            with self._autocast_ctx():
                out = self.model.get_text_features(**inputs)

            feats = self._pick_or_project(
                out=out,
                proj_layer=self.model.text_projection,
                embed_dim=self.EMBED_DIM,
                kind="text",
            )
            feats = self._maybe_normalize(feats)

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
