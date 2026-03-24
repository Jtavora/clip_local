# fashnParser.py
#
# Rodar local:
#   uv run uvicorn fashnParser:app --host 0.0.0.0 --port 8000 --reload
#
# Endpoints:
#   POST /v1/segmentation/human/clothes/cutout.png
#   POST /v1/segmentation/human/clothes/cutout-white.png
#   POST /v1/segmentation/human/model/cutout-white.png
#
# Env vars:
#   FASHN_MODEL_ID (default: fashn-ai/fashn-human-parser)
#   MAX_IMAGE_SIDE (default: 2048)
#   USE_AMP_FP16 (default: "true")  # só faz efeito se estiver em CUDA
#
# Chamada:
#   curl -X POST "http://localhost:8000/v1/segmentation/human/clothes/cutout.png" \
#     -F "file=@./testebody.jpeg" \
#     --output roupa_recortada.png
#
#   curl -X POST "http://localhost:8000/v1/segmentation/human/clothes/cutout-white.png" \
#     -F "file=@./testebody.jpeg" \
#     --output roupa_fundo_branco.png
#
#   curl -X POST "http://localhost:8000/v1/segmentation/human/model/cutout-white.png" \
#     -F "file=@./testebody.jpeg" \
#     --output modelo_fundo_branco.png

import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_ID = os.getenv("FASHN_MODEL_ID", "fashn-ai/fashn-human-parser")
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2048"))
USE_AMP = os.getenv("USE_AMP_FP16", "true").lower() == "true"

# Classes de roupa no esquema de 18 classes do FASHN parser
CLOTHES_CLASS_IDS = [3, 4, 5, 6, 7, 10]  # top, dress, skirt, pants, belt, scarf
BACKGROUND_CLASS_ID = 0

BACKGROUND_PRESETS = {
    "transparent": None,
    "gray": (217, 217, 217),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

app = FastAPI(title="FASHN Clothes Cutout API (Local)", version="1.0.0")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda") and USE_AMP

processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(device).eval()


def _autocast_ctx():
    if use_amp:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def _load_image(contents: bytes) -> Image.Image:
    if not contents:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    if MAX_IMAGE_SIDE > 0 and max(image.size) > MAX_IMAGE_SIDE:
        image = image.copy()
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return image


def _predict_class_mask(image: Image.Image) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        with _autocast_ctx():
            outputs = model(pixel_values=pixel_values)

    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1).squeeze(0).to(torch.uint8).cpu()


def _build_clothes_mask(class_mask: torch.Tensor) -> torch.Tensor:
    binary = torch.zeros_like(class_mask, dtype=torch.bool)
    for class_id in CLOTHES_CLASS_IDS:
        binary |= class_mask == int(class_id)
    return binary.to(torch.uint8)


def _build_model_mask(class_mask: torch.Tensor) -> torch.Tensor:
    return (class_mask != BACKGROUND_CLASS_ID).to(torch.uint8)


def _resolve_background(background: str) -> tuple[int, int, int] | None:
    value = background.strip().lower()
    if value in BACKGROUND_PRESETS:
        return BACKGROUND_PRESETS[value]

    if value.startswith("#") and len(value) == 7:
        try:
            r = int(value[1:3], 16)
            g = int(value[3:5], 16)
            b = int(value[5:7], 16)
            return (r, g, b)
        except ValueError:
            pass

    valid = ", ".join(list(BACKGROUND_PRESETS.keys()) + ["#RRGGBB"])
    raise HTTPException(status_code=400, detail=f"background invalido: {background}. validos: {valid}")


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_cutout_png_bytes(
    image: Image.Image,
    cutout_mask: torch.Tensor,
    crop: bool,
    padding: int,
    background: str,
) -> bytes:
    if padding < 0:
        raise HTTPException(status_code=400, detail="padding deve ser >= 0")

    alpha = Image.fromarray((cutout_mask * 255).to(torch.uint8).numpy(), mode="L")
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)

    if crop:
        bbox = alpha.getbbox()
        if bbox is None:
            raise HTTPException(status_code=422, detail="nenhuma regiao detectada na imagem")
        left, top, right, bottom = bbox
        if padding > 0:
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(rgba.width, right + padding)
            bottom = min(rgba.height, bottom + padding)
        rgba = rgba.crop((left, top, right, bottom))

    bg_rgb = _resolve_background(background)
    if bg_rgb is None:
        return _png_bytes(rgba)

    bg = Image.new("RGBA", rgba.size, (bg_rgb[0], bg_rgb[1], bg_rgb[2], 255))
    bg.alpha_composite(rgba)
    return _png_bytes(bg.convert("RGB"))


@app.post("/v1/segmentation/human/clothes/cutout.png")
async def segment_clothes_cutout_png(
    file: UploadFile = File(...),
):
    contents = await file.read()
    image = _load_image(contents)

    class_mask = _predict_class_mask(image)
    clothes_mask = _build_clothes_mask(class_mask)
    png_bytes = _make_cutout_png_bytes(
        image=image,
        cutout_mask=clothes_mask,
        crop=True,
        padding=0,
        background="transparent",
    )

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Model": MODEL_ID,
            "X-Endpoint": "clothes-cutout",
            "X-Crop": "true",
            "X-Padding": "0",
            "X-Background": "transparent",
        },
    )


@app.post("/v1/segmentation/human/model/cutout-white.png")
async def segment_model_cutout_white_png(
    file: UploadFile = File(...),
):
    contents = await file.read()
    image = _load_image(contents)

    class_mask = _predict_class_mask(image)
    model_mask = _build_model_mask(class_mask)
    png_bytes = _make_cutout_png_bytes(
        image=image,
        cutout_mask=model_mask,
        crop=True,
        padding=0,
        background="white",
    )

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Model": MODEL_ID,
            "X-Endpoint": "model-cutout-white",
            "X-Crop": "true",
            "X-Padding": "0",
            "X-Background": "white",
        },
    )


@app.post("/v1/segmentation/human/clothes/cutout-white.png")
async def segment_clothes_cutout_white_png(
    file: UploadFile = File(...),
):
    contents = await file.read()
    image = _load_image(contents)

    class_mask = _predict_class_mask(image)
    clothes_mask = _build_clothes_mask(class_mask)
    png_bytes = _make_cutout_png_bytes(
        image=image,
        cutout_mask=clothes_mask,
        crop=True,
        padding=0,
        background="white",
    )

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Model": MODEL_ID,
            "X-Endpoint": "clothes-cutout-white",
            "X-Crop": "true",
            "X-Padding": "0",
            "X-Background": "white",
        },
    )
