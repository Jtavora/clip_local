# fashionClip.py
#
# Rodar local:
#   uv run uvicorn fashionClip:app --host 0.0.0.0 --port 8000 --reload
#
# Endpoints:
#   GET  /health
#   POST /v1/embeddings/image/with-description  multipart form: file=@img.jpg
#
# Env vars:
#   FASHION_CLIP_MODEL_ID (default: patrickjohncyh/fashion-clip)
#   EMBEDDING_L2_NORMALIZE (default: "true")
#   USE_AMP_FP16 (default: "true")
#   MAX_IMAGE_SIDE (default: 2048)
#   HF_TOKEN (opcional)
#   FASHION_DESC_TOP_K (default: 4)
#   FASHION_DESC_LABELS (opcional, csv)
#   FASHION_DESC_PROMPT_PREFIX (default: a women gymwear product photo of)

import io
import os
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = os.getenv(
    "FASHION_CLIP_MODEL_ID",
    os.getenv("CLIP_MODEL_ID", "patrickjohncyh/fashion-clip"),
)
HF_TOKEN = os.getenv("HF_TOKEN")
NORMALIZE = os.getenv("EMBEDDING_L2_NORMALIZE", "true").lower() == "true"
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2048"))
DESC_TOP_K = int(os.getenv("FASHION_DESC_TOP_K", "4"))
DESC_PROMPT_PREFIX = os.getenv(
    "FASHION_DESC_PROMPT_PREFIX",
    "a women gymwear product photo of",
).strip()

DEFAULT_FASHION_DESC_LABELS = [
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
    "tank top",
    "cropped tank top",
    "training t-shirt",
    "oversized gym t-shirt",
    "long sleeve workout top",
    # --- Conjuntos ---
    "matching workout set",
    # --- Macacões / Bodies ---
    "one-piece workout jumpsuit",
    "workout bodysuit",
    # --- Vestidos ---
    "fitness dress",
    # --- Texturas / Detalhes ---
    "seamless knit",
    "ribbed texture",
    "mesh panel detail",
    "cutout detail",
    "logo print",
    "floral pattern",
    "solid color",
    # --- Cores ---
    "black color",
    "white color",
    "gray color",
    "beige color",
    "brown color",
    "coffee color",
    "wine color",
    "red color",
    "orange color",
    "yellow color",
    "green color",
    "olive green color",
    "blue color",
    "navy color",
    "aqua color",
    "purple color",
    "pink color",
    "multicolor",
]

desc_labels_raw = os.getenv("FASHION_DESC_LABELS", "").strip()
DESC_LABELS = (
    [x.strip() for x in desc_labels_raw.split(",") if x.strip()]
    if desc_labels_raw
    else DEFAULT_FASHION_DESC_LABELS
)
DESC_PROMPTS = [f"{DESC_PROMPT_PREFIX} {label}" for label in DESC_LABELS]
_DESC_TEXT_EMBEDDINGS: torch.Tensor | None = None

app = FastAPI(title="FashionCLIP Embeddings API (Local)", version="1.0.0")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda") and (os.getenv("USE_AMP_FP16", "true").lower() == "true")

model = CLIPModel.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN or None,
).to(device).eval()
processor = CLIPProcessor.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN or None,
)
EMBED_DIM = int(model.config.projection_dim)


def _autocast_ctx():
    if use_amp:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def _maybe_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if not NORMALIZE:
        return x
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def _pick_or_project(
    *,
    out: Any,
    proj_layer: torch.nn.Linear,
    embed_dim: int,
    kind: str,
) -> torch.Tensor:
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
        proj_dtype = proj_layer.weight.dtype
        pooled = pooled.to(dtype=proj_dtype)
        return proj_layer(pooled)

    raise RuntimeError(
        f"{kind}: pooler_output has dim={last_dim}, but expected either "
        f"embed_dim={embed_dim} (already projected) or in_features={in_feats} "
        f"(pre-projection)."
    )


def _limit_image(pil_img: Image.Image) -> Image.Image:
    if MAX_IMAGE_SIDE and max(pil_img.size) > MAX_IMAGE_SIDE:
        pil_img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return pil_img


def _embed_images_batch(images: list[Image.Image]) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        with _autocast_ctx():
            out = model.get_image_features(pixel_values=pixel_values)

    feats = _pick_or_project(
        out=out,
        proj_layer=model.visual_projection,
        embed_dim=EMBED_DIM,
        kind="image",
    )
    return _maybe_normalize(feats)


def _embed_texts_batch(texts: list[str]) -> torch.Tensor:
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
    }

    with torch.inference_mode():
        with _autocast_ctx():
            out = model.get_text_features(**inputs)

    feats = _pick_or_project(
        out=out,
        proj_layer=model.text_projection,
        embed_dim=EMBED_DIM,
        kind="text",
    )
    return _maybe_normalize(feats)


def _ensure_description_index() -> torch.Tensor:
    global _DESC_TEXT_EMBEDDINGS

    if _DESC_TEXT_EMBEDDINGS is None:
        with torch.inference_mode():
            _DESC_TEXT_EMBEDDINGS = _embed_texts_batch(DESC_PROMPTS)
    return _DESC_TEXT_EMBEDDINGS


def _is_color_label(label: str) -> bool:
    return label.endswith(" color") or label == "multicolor"


def _build_description(image_embedding: torch.Tensor, top_k: int) -> dict[str, object]:
    text_embs = _ensure_description_index()
    if text_embs.ndim != 2:
        raise RuntimeError("Unexpected description embedding shape")

    k = max(1, min(int(top_k), text_embs.shape[0]))
    scores, indices = torch.topk(text_embs @ image_embedding, k=k, dim=0)

    tags = [DESC_LABELS[int(i)] for i in indices.detach().cpu().tolist()]
    top_scores = [float(s) for s in scores.detach().cpu().tolist()]

    color_hits = [
        {"label": tags[i], "score": top_scores[i]}
        for i in range(len(tags))
        if _is_color_label(tags[i])
    ]
    non_color_hits = [
        {"label": tags[i], "score": top_scores[i]}
        for i in range(len(tags))
        if not _is_color_label(tags[i])
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "normalize": NORMALIZE,
        "amp_fp16": use_amp,
        "max_image_side": MAX_IMAGE_SIDE,
        "description_labels_count": len(DESC_LABELS),
        "description_color_labels_count": sum(1 for label in DESC_LABELS if _is_color_label(label)),
        "description_top_k_default": DESC_TOP_K,
        "description_prompt_prefix": DESC_PROMPT_PREFIX,
        "vision_proj_in": int(model.visual_projection.in_features),
        "text_proj_in": int(model.text_projection.in_features),
    }


@app.post("/v1/embeddings/image/with-description")
async def embeddings_image_with_description(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = _limit_image(image)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    feats = _embed_images_batch([image])
    if feats.ndim != 2 or feats.shape[0] != 1:
        raise RuntimeError("Unexpected embedding batch shape")

    img_emb = feats[0]
    desc = _build_description(img_emb, top_k=DESC_TOP_K)

    return {
        "model": MODEL_ID,
        "embedding_dim": EMBED_DIM,
        "object": "embedding_with_description",
        "embedding": img_emb.detach().cpu().tolist(),
        "description": desc["description"],
        "labels": desc["labels"],
        "colors": desc["colors"],
    }
