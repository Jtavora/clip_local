# apiFashnParser.py
import io
import os
from typing import Optional

import modal

app = modal.App("fashn-clothes-cutout-api")

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
@modal.concurrent(max_inputs=16)
class FashnParserAPI:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.MODEL_ID = os.getenv("FASHN_MODEL_ID", "fashn-ai/fashn-human-parser")
        self.MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2048"))
        self.USE_AMP = os.getenv("USE_AMP_FP16", "true").lower() == "true"
        self.API_KEY = os.getenv("EMBEDDINGS_API_KEY")
        self.HF_TOKEN = os.getenv("HF_TOKEN")

        # top, dress, skirt, pants, belt, scarf
        self.CLOTHES_CLASS_IDS = [3, 4, 5, 6, 7, 10]

        self.BACKGROUND_PRESETS = {
            "transparent": None,
            "gray": (217, 217, 217),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = (self.device == "cuda") and self.USE_AMP

        self.processor = SegformerImageProcessor.from_pretrained(
            self.MODEL_ID,
            token=self.HF_TOKEN or None,
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.MODEL_ID,
            token=self.HF_TOKEN or None,
        ).to(self.device).eval()

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

    def _load_image(self, contents):
        from fastapi import HTTPException
        from PIL import Image

        if not contents:
            raise HTTPException(status_code=400, detail="empty file")
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="invalid image file")

        if self.MAX_IMAGE_SIDE > 0 and max(image.size) > self.MAX_IMAGE_SIDE:
            image = image.copy()
            image.thumbnail((self.MAX_IMAGE_SIDE, self.MAX_IMAGE_SIDE))
        return image

    def _predict_class_mask(self, image):
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.inference_mode():
            with self._autocast_ctx():
                outputs = self.model(pixel_values=pixel_values)

        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        return upsampled.argmax(dim=1).squeeze(0).to(torch.uint8).cpu()

    def _build_clothes_mask(self, class_mask):
        import torch

        binary = torch.zeros_like(class_mask, dtype=torch.bool)
        for class_id in self.CLOTHES_CLASS_IDS:
            binary |= class_mask == int(class_id)
        return binary.to(torch.uint8)

    def _resolve_background(self, background: str):
        from fastapi import HTTPException

        value = background.strip().lower()
        if value in self.BACKGROUND_PRESETS:
            return self.BACKGROUND_PRESETS[value]

        if value.startswith("#") and len(value) == 7:
            try:
                r = int(value[1:3], 16)
                g = int(value[3:5], 16)
                b = int(value[5:7], 16)
                return (r, g, b)
            except ValueError:
                pass

        valid = ", ".join(list(self.BACKGROUND_PRESETS.keys()) + ["#RRGGBB"])
        raise HTTPException(status_code=400, detail=f"background invalido: {background}. validos: {valid}")

    def _png_bytes(self, img) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _make_cutout_png_bytes(
        self,
        image,
        clothes_mask,
        crop: bool,
        padding: int,
        background: str,
    ) -> bytes:
        import torch
        from fastapi import HTTPException
        from PIL import Image

        if padding < 0:
            raise HTTPException(status_code=400, detail="padding deve ser >= 0")

        alpha = Image.fromarray((clothes_mask * 255).to(torch.uint8).numpy(), mode="L")
        rgba = image.convert("RGBA")
        rgba.putalpha(alpha)

        if crop:
            bbox = alpha.getbbox()
            if bbox is None:
                raise HTTPException(status_code=422, detail="nenhuma roupa detectada na imagem")
            left, top, right, bottom = bbox
            if padding > 0:
                left = max(0, left - padding)
                top = max(0, top - padding)
                right = min(rgba.width, right + padding)
                bottom = min(rgba.height, bottom + padding)
            rgba = rgba.crop((left, top, right, bottom))

        bg_rgb = self._resolve_background(background)
        if bg_rgb is None:
            return self._png_bytes(rgba)

        bg = Image.new("RGBA", rgba.size, (bg_rgb[0], bg_rgb[1], bg_rgb[2], 255))
        bg.alpha_composite(rgba)
        return self._png_bytes(bg.convert("RGB"))

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, File, Header, UploadFile
        from fastapi.responses import Response

        api = FastAPI(title="FASHN Clothes Cutout API (Modal)", version="1.0.0")

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "device": self.device,
                "model": self.MODEL_ID,
                "amp_fp16": self.use_amp,
                "max_image_side": self.MAX_IMAGE_SIDE,
                "clothes_class_ids": self.CLOTHES_CLASS_IDS,
            }

        @api.post("/v1/segmentation/human/clothes/cutout.png")
        async def segment_clothes_cutout_png(
            file: UploadFile = File(...),
            x_api_key: Optional[str] = Header(default=None),
        ):
            self._require_key(x_api_key)

            contents = await file.read()
            image = self._load_image(contents)

            class_mask = self._predict_class_mask(image)
            clothes_mask = self._build_clothes_mask(class_mask)
            png_bytes = self._make_cutout_png_bytes(
                image=image,
                clothes_mask=clothes_mask,
                crop=True,
                padding=0,
                background="transparent",
            )

            return Response(
                content=png_bytes,
                media_type="image/png",
                headers={
                    "X-Model": self.MODEL_ID,
                    "X-Endpoint": "clothes-cutout",
                    "X-Crop": "true",
                    "X-Padding": "0",
                    "X-Background": "transparent",
                },
            )

        return api
