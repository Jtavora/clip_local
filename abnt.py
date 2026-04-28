# abnt.py
#
# Rodar local:
#   uv run uvicorn abnt:app --host 0.0.0.0 --port 8001 --reload
#
# Endpoints:
#   GET  /health
#   POST /v1/abnt/extract/text   {"text": "...", "source_name": "arquivo.md"}
#   POST /v1/abnt/extract/file   multipart form: file=@livro.pdf
#
# Env vars:
#   ABNT_MODEL_ID (default: Qwen/Qwen3.5-2B)
#   HF_TOKEN (opcional para modelos gated)
#   ABNT_MAX_PAGES (default: 24)            # só para PDF
#   ABNT_MAX_INPUT_CHARS (default: 30000)
#   ABNT_MAX_INPUT_TOKENS (default: 4096)
#   ABNT_MAX_NEW_TOKENS (default: 256)
#   ABNT_TEMPERATURE (default: 0.0)
#   ABNT_TOP_P (default: 1.0)
#   ABNT_TOP_K (default: 20)
#   ABNT_REPAIR_MAX_NEW_TOKENS (default: 256)
#   ABNT_DEVICE_MAP (default: cuda:0 se CUDA disponível, senão cpu)
#   ABNT_TORCH_DTYPE (default: fp16 se CUDA disponível, senão fp32)  # auto|bf16|fp16|fp32

from __future__ import annotations

import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """
Você é um especialista em normalização bibliográfica brasileira (ABNT NBR 6023:2018).

Sua tarefa é analisar o conteúdo textual de uma obra, geralmente extraído da capa,
folha de rosto, verso da folha de rosto e ficha catalográfica, e extrair com precisão
todas as informações necessárias para montar uma referência bibliográfica completa no
padrão ABNT.

Regras importantes:
- Sobrenomes devem ser fornecidos em CAIXA ALTA.
- Se houver mais de 3 autores, inclua apenas o primeiro e marque et_al como true.
- A edição só deve ser preenchida se for a 2ª ou posterior.
- Use '[S.l.]' para local desconhecido, '[s.n.]' para editora desconhecida e '[s.d.]'
  para ano desconhecido.
- Extraia o ISBN se visível no conteúdo.
- Se houver ficha catalográfica/CIP, trate-a como fonte prioritária.
- Seja criterioso: prefira não preencher um campo a preencher com dado incorreto.
"""

JSON_SCHEMA_HINT = """
Retorne apenas JSON válido, sem markdown e sem texto extra, no formato:
{
  "autores": [{"sobrenome": "SILVA", "prenomes": "João"}],
  "et_al": false,
  "titulo": "Título principal",
  "subtitulo": "Subtítulo ou null",
  "edicao": "2. ed." ou null,
  "local_publicacao": "Cidade ou [S.l.]",
  "editora": "Editora ou [s.n.]",
  "ano_publicacao": "2024 ou [s.d.]",
  "isbn": "978... ou null"
}
"""


class AutorABNT(BaseModel):
    sobrenome: str = ""
    prenomes: str = ""

    def formatar(self) -> str:
        sobrenome = (self.sobrenome or "").strip().upper()
        prenomes = (self.prenomes or "").strip()
        if sobrenome and prenomes:
            return f"{sobrenome}, {prenomes}"
        return sobrenome or prenomes


class ReferenciaLivroABNT(BaseModel):
    autores: list[AutorABNT] = Field(default_factory=list)
    et_al: bool = False
    titulo: str = ""
    subtitulo: Optional[str] = None
    edicao: Optional[str] = None
    local_publicacao: str = "[S.l.]"
    editora: str = "[s.n.]"
    ano_publicacao: str = "[s.d.]"
    isbn: Optional[str] = None

    def formatar_referencia(self) -> str:
        autores = [a.formatar() for a in self.autores if a.formatar()]
        if autores:
            if self.et_al:
                autores_txt = f"{autores[0]} et al."
            else:
                autores_txt = "; ".join(autores)
            autores_txt = f"{autores_txt}. "
        else:
            autores_txt = ""

        titulo = (self.titulo or "").strip() or "[Título não identificado]"
        subtitulo = (self.subtitulo or "").strip()
        titulo_completo = f"{titulo}: {subtitulo}" if subtitulo else titulo

        edicao = (self.edicao or "").strip()
        edicao_txt = f" {edicao}" if edicao else ""

        local_pub = (self.local_publicacao or "").strip() or "[S.l.]"
        editora = (self.editora or "").strip() or "[s.n.]"
        ano = (self.ano_publicacao or "").strip() or "[s.d.]"

        ref = f"{autores_txt}{titulo_completo}.{edicao_txt} {local_pub}: {editora}, {ano}."
        if self.isbn and self.isbn.strip():
            ref = f"{ref} ISBN {self.isbn.strip()}."
        return re.sub(r"\s+", " ", ref).strip()


class TextExtractionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_name: str = "documento.txt"


def _split_markdown_pages(texto: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?mi)^##\s+Página\s+(\d+)\s*$", texto))
    if not matches:
        return [("1", texto.strip())] if texto.strip() else []

    pages: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
        bloco = texto[start:end].strip()
        if bloco:
            pages.append((match.group(1), bloco))
    return pages


def _selecionar_trecho_bibliografico_markdown(texto: str, max_pages: int = 6) -> str:
    pages = _split_markdown_pages(texto)
    if not pages:
        return ""

    keywords = [
        "ficha catalografica",
        "ficha catalográfica",
        "dados internacionais de catalogação",
        "dados internacionais de catalogacao",
        "cip",
        "isbn",
        "autor",
    ]

    selected: set[int] = set()
    for idx, (_, page_text) in enumerate(pages):
        normalized = page_text.lower()
        if any(keyword in normalized for keyword in keywords):
            selected.add(idx)
            if idx > 0:
                selected.add(idx - 1)
            if idx + 1 < len(pages):
                selected.add(idx + 1)

    if not selected:
        selected = set(range(min(max_pages, len(pages))))

    return "\n\n".join(pages[idx][1] for idx in sorted(selected))


def _extrair_texto_pdf(pdf_bytes: bytes, max_pages: int) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    total = len(reader.pages)
    if total == 0:
        return ""

    n = min(max(1, max_pages), total)
    pages_text: list[str] = []
    for i in range(n):
        page_text = reader.pages[i].extract_text() or ""
        pages_text.append(f"## Página {i + 1}\n{page_text.strip()}")
    return "\n\n".join(pages_text).strip()


def _strip_think_blocks(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<\|channel\|>thought.*?<\|/?channel\|>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _sanitize_json_controls(text: str) -> str:
    """
    Escapa/remov​e caracteres de controle que quebram json.loads.
    - Dentro de string JSON: converte para \\n, \\r, \\t ou \\u00xx
    - Fora de string JSON: remove controles não-whitespace
    """
    out: list[str] = []
    in_string = False
    escaped = False

    for ch in text:
        code = ord(ch)

        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if code < 0x20:
                if ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(f"\\u{code:04x}")
                continue
            out.append(ch)
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            continue
        if code < 0x20 and ch not in ("\n", "\r", "\t"):
            continue
        out.append(ch)

    return "".join(out)


def _parse_json_object(text: str) -> dict:
    cleaned = _strip_think_blocks(text).replace("\ufeff", "")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates: list[str] = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])

    last_err: Optional[Exception] = None
    for candidate in candidates:
        for probe in (candidate, _sanitize_json_controls(candidate)):
            try:
                return json.loads(probe, strict=False)
            except Exception as exc:
                last_err = exc

    if last_err is not None:
        raise last_err
    raise ValueError("Não foi possível parsear JSON da saída do modelo.")


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload or {})

    autores = normalized.get("autores")
    if not isinstance(autores, list):
        autores = []
    normalized["autores"] = autores

    normalized["et_al"] = bool(normalized.get("et_al", False))

    def _opt_str(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    normalized["titulo"] = _opt_str(normalized.get("titulo")) or ""
    normalized["subtitulo"] = _opt_str(normalized.get("subtitulo"))
    normalized["edicao"] = _opt_str(normalized.get("edicao"))
    normalized["isbn"] = _opt_str(normalized.get("isbn"))
    normalized["local_publicacao"] = _opt_str(normalized.get("local_publicacao")) or "[S.l.]"
    normalized["editora"] = _opt_str(normalized.get("editora")) or "[s.n.]"
    normalized["ano_publicacao"] = _opt_str(normalized.get("ano_publicacao")) or "[s.d.]"

    return normalized


def _sanitize_referencia(ref: ReferenciaLivroABNT) -> ReferenciaLivroABNT:
    autores = [a for a in ref.autores if (a.sobrenome or a.prenomes)]
    for autor in autores:
        autor.sobrenome = (autor.sobrenome or "").strip().upper()
        autor.prenomes = (autor.prenomes or "").strip()

    if len(autores) > 3:
        autores = [autores[0]]
        ref.et_al = True
    if ref.et_al and len(autores) > 1:
        autores = [autores[0]]

    ref.autores = autores
    ref.titulo = (ref.titulo or "").strip()
    ref.subtitulo = (ref.subtitulo or "").strip() or None
    ref.edicao = (ref.edicao or "").strip() or None

    if ref.edicao and re.match(r"^1[\D_]*", ref.edicao):
        ref.edicao = None

    ref.local_publicacao = (ref.local_publicacao or "").strip() or "[S.l.]"
    ref.editora = (ref.editora or "").strip() or "[s.n.]"
    ref.ano_publicacao = (ref.ano_publicacao or "").strip() or "[s.d.]"
    ref.isbn = (ref.isbn or "").strip() or None

    return ref


def _resolve_torch_dtype(dtype_name: str) -> Any:
    name = (dtype_name or "auto").strip().lower()
    if name == "auto":
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(name, "auto")


def _normalize_device_label(dev: Any) -> str:
    if isinstance(dev, int):
        return f"cuda:{dev}"
    return str(dev)


def _summarize_model_device(model: AutoModelForCausalLM) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        unique = []
        for dev in device_map.values():
            label = _normalize_device_label(dev)
            if label not in unique:
                unique.append(label)
        if len(unique) == 1:
            return unique[0]
        return f"mixed({','.join(unique)})"

    devices = []
    for p in model.parameters():
        label = str(p.device)
        if label not in devices:
            devices.append(label)
        if len(devices) > 1:
            break
    if not devices:
        return "unknown"
    if len(devices) == 1:
        return devices[0]
    return f"mixed({','.join(devices)})"


def _pick_inference_device(model: AutoModelForCausalLM) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for dev in device_map.values():
            label = _normalize_device_label(dev)
            if label.startswith("cuda"):
                return torch.device(label)

    for p in model.parameters():
        if p.device.type == "cuda":
            return p.device

    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


class ABNTExtractor:
    def __init__(self) -> None:
        self.model_id = os.getenv("ABNT_MODEL_ID", "Qwen/Qwen3.5-2B")
        self.hf_token = os.getenv("HF_TOKEN")
        self.max_pages = int(os.getenv("ABNT_MAX_PAGES", "24"))
        self.max_input_chars = int(os.getenv("ABNT_MAX_INPUT_CHARS", "30000"))
        self.max_input_tokens = int(os.getenv("ABNT_MAX_INPUT_TOKENS", "4096"))
        self.max_new_tokens = int(os.getenv("ABNT_MAX_NEW_TOKENS", "256"))
        self.repair_max_new_tokens = int(os.getenv("ABNT_REPAIR_MAX_NEW_TOKENS", "256"))
        self.temperature = float(os.getenv("ABNT_TEMPERATURE", "0.0"))
        self.top_p = float(os.getenv("ABNT_TOP_P", "1.0"))
        self.top_k = int(os.getenv("ABNT_TOP_K", "20"))

        default_device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        default_dtype = "fp16" if torch.cuda.is_available() else "fp32"
        self.device_map = os.getenv("ABNT_DEVICE_MAP", default_device_map)
        self.torch_dtype = _resolve_torch_dtype(os.getenv("ABNT_TORCH_DTYPE", default_dtype))

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.model_device: str = "unknown"
        self.load_error: Optional[str] = None

        self._load_model()

    def _load_model(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.hf_token or None,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=self.hf_token or None,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
            ).eval()
            self.model_device = _summarize_model_device(self.model)
        except Exception as exc:
            self.load_error = str(exc)

    def _require_model(self) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        if self.load_error:
            raise RuntimeError(f"Falha ao carregar modelo Hugging Face: {self.load_error}")
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Modelo não inicializado.")
        return self.tokenizer, self.model

    def _build_prompt(self, text: str, source_name: str) -> str:
        tokenizer, _ = self._require_model()
        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{JSON_SCHEMA_HINT}"},
            {
                "role": "user",
                "content": (
                    f"Arquivo de origem: {source_name}\n\n"
                    "Analise o conteúdo abaixo e extraia os dados ABNT. "
                    "Retorne somente JSON:\n\n"
                    f"{text}"
                ),
            },
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _repair_json_payload(self, broken_output: str, source_name: str) -> dict[str, Any]:
        tokenizer, model = self._require_model()
        repair_prompt = (
            "Converta o conteúdo abaixo para JSON estritamente válido. "
            "Retorne SOMENTE JSON, sem texto adicional.\n\n"
            f"Arquivo de origem: {source_name}\n"
            f"Conteúdo para corrigir:\n{broken_output[:6000]}"
        )
        encoded = tokenizer(
            repair_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(self.max_input_tokens, 2048),
        )
        device = _pick_inference_device(model)
        if str(device) != "meta":
            encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            repaired_outputs = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.repair_max_new_tokens,
            )
        repair_input_len = encoded["input_ids"].shape[-1]
        repaired = tokenizer.decode(
            repaired_outputs[0][repair_input_len:],
            skip_special_tokens=True,
        )
        return _normalize_payload(_parse_json_object(repaired))

    def extract_from_text(self, text: str, *, source_name: str) -> tuple[ReferenciaLivroABNT, dict]:
        tokenizer, model = self._require_model()

        body = (text or "").strip()
        if not body:
            raise ValueError("Texto vazio para extração ABNT.")

        if len(body) > self.max_input_chars:
            body = body[: self.max_input_chars]

        prompt = self._build_prompt(body, source_name)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        device = _pick_inference_device(model)
        if str(device) != "meta":
            encoded = {k: v.to(device) for k, v in encoded.items()}

        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = model.generate(**encoded, **gen_kwargs)

        input_len = encoded["input_ids"].shape[-1]
        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        try:
            payload = _normalize_payload(_parse_json_object(generated))
        except Exception:
            payload = self._repair_json_payload(generated, source_name)
        ref = ReferenciaLivroABNT.model_validate(payload)
        ref = _sanitize_referencia(ref)

        usage = {
            "input_tokens": int(input_len),
            "output_tokens": int(outputs.shape[-1] - input_len),
            "total_tokens": int(outputs.shape[-1]),
        }
        return ref, usage


def _build_response(ref: ReferenciaLivroABNT, usage: dict, model: str, source_name: str) -> dict:
    return {
        "model": model,
        "object": "abnt_reference",
        "source_name": source_name,
        "data": ref.model_dump(),
        "formatted_reference": ref.formatar_referencia(),
        "usage": usage,
    }


app = FastAPI(title="ABNT Reference Extraction API (Local)", version="1.0.0")
extractor = ABNTExtractor()


@app.get("/health")
def health():
    return {
        "status": "ok" if extractor.load_error is None else "error",
        "model": extractor.model_id,
        "model_device": extractor.model_device,
        "model_loaded": extractor.load_error is None,
        "load_error": extractor.load_error,
        "max_pages": extractor.max_pages,
        "max_input_chars": extractor.max_input_chars,
        "max_input_tokens": extractor.max_input_tokens,
        "max_new_tokens": extractor.max_new_tokens,
        "repair_max_new_tokens": extractor.repair_max_new_tokens,
    }


@app.post("/v1/abnt/extract/text")
def extract_text(req: TextExtractionRequest):
    try:
        ref, usage = extractor.extract_from_text(req.text, source_name=req.source_name)
        return _build_response(ref, usage, extractor.model_id, req.source_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha na extração ABNT: {exc}")


@app.post("/v1/abnt/extract/file")
async def extract_file(file: UploadFile = File(...), max_pages: Optional[int] = None):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")

    source_name = file.filename or "arquivo"
    suffix = Path(source_name).suffix.lower()

    try:
        if suffix == ".pdf":
            pages = max_pages if (max_pages is not None and max_pages > 0) else extractor.max_pages
            extracted = _extrair_texto_pdf(content, pages)
        elif suffix == ".md":
            decoded = content.decode("utf-8", errors="ignore")
            pages = max_pages if (max_pages is not None and max_pages > 0) else 6
            extracted = _selecionar_trecho_bibliografico_markdown(decoded, max_pages=pages)
        else:
            extracted = content.decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Falha ao ler arquivo: {exc}")

    if not extracted.strip():
        raise HTTPException(status_code=400, detail="não foi possível extrair texto útil do arquivo")

    try:
        ref, usage = extractor.extract_from_text(extracted, source_name=source_name)
        payload = _build_response(ref, usage, extractor.model_id, source_name)
        payload["input_chars"] = len(extracted)
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha na extração ABNT: {exc}")
