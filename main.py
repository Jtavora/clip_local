import io
import os

import torch
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-large-patch14")
OUTPUT_DIM = int(os.getenv("EMBEDDING_OUTPUT_DIM", "1536"))

app = FastAPI(title="CLIP Embeddings API", version="1.0.0")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)
BASE_DIM = int(model.config.projection_dim)

if OUTPUT_DIM < BASE_DIM:
    raise ValueError(
        f"EMBEDDING_OUTPUT_DIM ({OUTPUT_DIM}) must be >= model projection dim ({BASE_DIM})"
    )


class TextEmbeddingsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model": MODEL_ID,
        "base_dim": BASE_DIM,
        "output_dim": OUTPUT_DIM,
    }


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def _to_output_dim(x: torch.Tensor) -> torch.Tensor:
    if x.size(-1) == OUTPUT_DIM:
        return x

    pad = OUTPUT_DIM - x.size(-1)
    return torch.nn.functional.pad(x, (0, pad))


@app.post("/v1/embeddings/text")
@torch.inference_mode()
def embeddings_text(req: TextEmbeddingsRequest):
    inputs = processor(
        text=req.texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Encoder de texto (CLIPTextTransformer)
    text_outputs = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = text_outputs.last_hidden_state  # (N, T, H)

    # Pega o embedding do token EOS/EOT (padrão CLIP)
    # Para o tokenizer do CLIP, o EOT costuma ser o maior id na sequência
    eos_pos = input_ids.argmax(dim=-1)  # (N,)
    pooled = last_hidden[torch.arange(last_hidden.size(0)), eos_pos]  # (N, H)

    # Projeção para o espaço multimodal
    feats = model.text_projection(pooled)  # (N, D)

    feats = feats.to(dtype)
    feats = _to_output_dim(feats)
    feats = _l2_normalize(feats)

    return {
        "model": MODEL_ID,
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
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Encoder de visão (CLIPVisionTransformer)
    vision_outputs = model.vision_model(pixel_values=pixel_values)

    # pooler_output já vem como (B, H)
    pooled = vision_outputs.pooler_output  # (1, H)

    # Projeção para o espaço multimodal
    feats = model.visual_projection(pooled)  # (1, D)

    feats = feats.to(dtype)
    feats = _to_output_dim(feats)
    feats = _l2_normalize(feats)

    return {"model": MODEL_ID, "object": "embedding", "embedding": feats.detach().cpu().tolist()[0]}
