# apiFashionClip.py
import io
import os
from typing import Any, Optional

import modal

app = modal.App("fashion-clip-embeddings-api")

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
class FashionCLIPAPI:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.MODEL_ID = os.getenv(
            "FASHION_CLIP_MODEL_ID",
            os.getenv("CLIP_MODEL_ID", "patrickjohncyh/fashion-clip"),
        )
        self.NORMALIZE = os.getenv("EMBEDDING_L2_NORMALIZE", "true").lower() == "true"
        self.USE_AMP = os.getenv("USE_AMP_FP16", "true").lower() == "true"
        self.MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2048"))
        self.DESC_TOP_K = int(os.getenv("FASHION_DESC_TOP_K", "4"))
        self.DESC_PROMPT_PREFIX = os.getenv(
            "FASHION_DESC_PROMPT_PREFIX",
            "a women gymwear product photo of",
        ).strip()

        default_desc_labels = [
            # --- Leggings ---
            "women gym leggings",
            "high-waisted leggings",
            "seamless leggings",
            "compression leggings",
            "scrunch butt leggings",
            "v-waist leggings",
            "flare leggings",
            # --- Shorts ---
            "women biker shorts",
            "women gym shorts",
            "high-waisted shorts",
            # --- Tops / Bras ---
            "sports bra",
            "high support sports bra",
            "longline sports bra",
            "crop top",
            "zip-up crop top",
            "tank top",
            "cropped tank top",
            "racerback tank top",
            "training t-shirt",
            "oversized gym t-shirt",
            "long sleeve workout top",
            "thumb hole long sleeve",
            # --- Blusas / Regatas ---
            "dry-fit t-shirt",
            "mesh workout top",
            "muscle tank top",
            # --- Conjuntos ---
            "matching workout set",
            "zip-up workout set",
            "matching set with pocket",
            "matching short set",
            "matching legging set",
            # --- Macacões / Bodies / Macaquinhos ---
            "one-piece workout jumpsuit",
            "short workout romper",
            "workout bodysuit",
            # --- Vestidos ---
            "fitness dress",
            # --- Jaquetas ---
            "zip hoodie",
            "workout jacket",
            "slim fit jacket",
            # --- Detalhes estruturais ---
            "front zipper",
            "side pocket",
            "mesh panel detail",
            "cutout detail",
            "dual strap detail",
            "ribbed texture",
            "seamless knit",
            # --- Estampas ---
            "logo print",
            "floral pattern",
            "color block",
            "bicolor pattern",
            "solid color",
            # --- Cores ---
            "black color",
            "white color",
            "gray color",
            "beige color",
            "nude color",
            "brown color",
            "coffee color",
            "wine color",
            "marsala color",
            "red color",
            "coral color",
            "orange color",
            "yellow color",
            "green color",
            "olive green color",
            "mint green color",
            "emerald green color",
            "blue color",
            "navy color",
            "royal blue color",
            "aqua color",
            "turquoise color",
            "purple color",
            "lilac color",
            "pink color",
            "hot pink color",
            "rose color",
            "multicolor",
        ]
        raw_desc_labels = os.getenv("FASHION_DESC_LABELS", "").strip()
        self.DESC_LABELS = (
            [x.strip() for x in raw_desc_labels.split(",") if x.strip()]
            if raw_desc_labels
            else default_desc_labels
        )
        self.DESC_PROMPTS = [f"{self.DESC_PROMPT_PREFIX} {label}" for label in self.DESC_LABELS]
        self._desc_text_embeddings = None

        self.API_KEY = os.getenv("EMBEDDINGS_API_KEY")
        self.HF_TOKEN = os.getenv("HF_TOKEN")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = (self.device == "cuda") and self.USE_AMP

        self.model = CLIPModel.from_pretrained(
            self.MODEL_ID,
            token=self.HF_TOKEN or None,
        ).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(
            self.MODEL_ID,
            token=self.HF_TOKEN or None,
        )
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

    def _embed_texts_batch(self, texts: list[str]):
        import torch

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
        }

        with torch.inference_mode():
            with self._autocast_ctx():
                out = self.model.get_text_features(**inputs)

        feats = self._pick_or_project(
            out=out,
            proj_layer=self.model.text_projection,
            embed_dim=self.EMBED_DIM,
            kind="text",
        )
        return self._maybe_normalize(feats)

    def _ensure_description_index(self):
        if self._desc_text_embeddings is None:
            self._desc_text_embeddings = self._embed_texts_batch(self.DESC_PROMPTS)
        return self._desc_text_embeddings

    def _is_color_label(self, label: str) -> bool:
        return label.endswith(" color") or label == "multicolor"

    def _build_description(self, image_embedding, top_k: int):
        import torch

        text_embs = self._ensure_description_index()
        if text_embs.ndim != 2:
            raise RuntimeError("Unexpected description embedding shape")

        k = max(1, min(int(top_k), int(text_embs.shape[0])))
        scores, indices = torch.topk(text_embs @ image_embedding, k=k, dim=0)

        tags = [self.DESC_LABELS[int(i)] for i in indices.detach().cpu().tolist()]
        top_scores = [float(s) for s in scores.detach().cpu().tolist()]

        color_hits = [
            {"label": tags[i], "score": top_scores[i]}
            for i in range(len(tags))
            if self._is_color_label(tags[i])
        ]
        non_color_hits = [
            {"label": tags[i], "score": top_scores[i]}
            for i in range(len(tags))
            if not self._is_color_label(tags[i])
        ]
        non_color_tags = [hit["label"] for hit in non_color_hits]

        color_names = []
        for hit in color_hits:
            name = hit["label"].replace(" color", "")
            if name not in color_names:
                color_names.append(name)

        if non_color_tags:
            description = non_color_tags[0]
            if color_names:
                description = f"{description} in {', '.join(color_names[:2])}"
            if len(non_color_tags) > 1:
                description = f"{description} with {', '.join(non_color_tags[1:])}"
        elif color_names:
            description = f"fashion item in {', '.join(color_names[:2])}"
        else:
            description = tags[0]

        return {
            "description": description,
            "labels": non_color_hits,
            "colors": color_hits,
        }

    @modal.asgi_app()
    def web(self):
        import torch
        from fastapi import FastAPI, File, Header, HTTPException, UploadFile
        from PIL import Image

        api = FastAPI(title="FashionCLIP Embeddings API (Modal)", version="1.0.0")

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
                "description_labels_count": len(self.DESC_LABELS),
                "description_color_labels_count": sum(
                    1 for label in self.DESC_LABELS if self._is_color_label(label)
                ),
                "description_top_k_default": self.DESC_TOP_K,
                "description_prompt_prefix": self.DESC_PROMPT_PREFIX,
            }

        @api.post("/v1/embeddings/image/with-description")
        async def embeddings_image_with_description(
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

            img_emb = feats[0]
            desc = self._build_description(img_emb, top_k=self.DESC_TOP_K)

            return {
                "model": self.MODEL_ID,
                "embedding_dim": self.EMBED_DIM,
                "object": "embedding_with_description",
                "embedding": img_emb.detach().cpu().tolist(),
                "description": desc["description"],
                "labels": desc["labels"],
                "colors": desc["colors"],
            }

        return api
