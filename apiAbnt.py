# modal_abnt_api.py
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import modal
from pydantic import BaseModel, Field

app = modal.App("abnt-reference-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]==0.115.0",
        "pydantic==2.8.2",
        "torch>=2.10.0",
        "git+https://github.com/huggingface/transformers.git",
        "accelerate>=0.33.0",
        "huggingface_hub>=0.24.6",
        "hf_transfer>=0.1.8",
        "sentencepiece>=0.2.0",
        "pypdf>=4.3.1",
        "python-multipart>=0.0.22",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

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
    from pypdf import PdfReader

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
    import torch

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


def _build_response(ref: ReferenciaLivroABNT, usage: dict, model: str, source_name: str) -> dict:
    return {
        "model": model,
        "object": "abnt_reference",
        "source_name": source_name,
        "data": ref.model_dump(),
        "formatted_reference": ref.formatar_referencia(),
        "usage": usage,
    }


@app.cls(
    image=image,
    gpu="L4",
    timeout=60 * 20,
    scaledown_window=60 * 5,
    max_containers=10,
    volumes={"/cache": hf_cache},
    secrets=[modal.Secret.from_name("clip-api-secret")],  # EMBEDDINGS_API_KEY + HF_TOKEN
)
@modal.concurrent(max_inputs=8)
class ABNTAPI:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.MODEL_ID = os.getenv("ABNT_MODEL_ID", "Qwen/Qwen3.5-2B")
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.API_KEY = os.getenv("EMBEDDINGS_API_KEY")

        self.MAX_PAGES = int(os.getenv("ABNT_MAX_PAGES", "24"))
        self.MAX_INPUT_CHARS = int(os.getenv("ABNT_MAX_INPUT_CHARS", "30000"))
        self.MAX_INPUT_TOKENS = int(os.getenv("ABNT_MAX_INPUT_TOKENS", "4096"))
        self.MAX_NEW_TOKENS = int(os.getenv("ABNT_MAX_NEW_TOKENS", "256"))
        self.REPAIR_MAX_NEW_TOKENS = int(os.getenv("ABNT_REPAIR_MAX_NEW_TOKENS", "256"))
        self.TEMPERATURE = float(os.getenv("ABNT_TEMPERATURE", "0.0"))
        self.TOP_P = float(os.getenv("ABNT_TOP_P", "1.0"))
        self.TOP_K = int(os.getenv("ABNT_TOP_K", "20"))

        default_device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        default_dtype = "fp16" if torch.cuda.is_available() else "fp32"
        self.DEVICE_MAP = os.getenv("ABNT_DEVICE_MAP", default_device_map)
        self.TORCH_DTYPE = _resolve_torch_dtype(os.getenv("ABNT_TORCH_DTYPE", default_dtype))

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, token=self.HF_TOKEN or None)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            token=self.HF_TOKEN or None,
            device_map=self.DEVICE_MAP,
            torch_dtype=self.TORCH_DTYPE,
        ).eval()

        try:
            self.MODEL_DEVICE = str(next(self.model.parameters()).device)
        except Exception:
            self.MODEL_DEVICE = str(getattr(self.model, "device", "unknown"))

    def _require_key(self, provided: Optional[str]) -> None:
        if not self.API_KEY:
            return
        from fastapi import HTTPException

        if not provided or provided != self.API_KEY:
            raise HTTPException(status_code=401, detail="unauthorized")

    def _build_prompt(self, text: str, source_name: str) -> str:
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
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _extract_from_text(self, text: str, source_name: str) -> tuple[ReferenciaLivroABNT, dict]:
        import torch
        from fastapi import HTTPException

        body = (text or "").strip()
        if not body:
            raise HTTPException(status_code=400, detail="Texto vazio para extração ABNT.")

        if len(body) > self.MAX_INPUT_CHARS:
            body = body[: self.MAX_INPUT_CHARS]

        prompt = self._build_prompt(body, source_name)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.MAX_INPUT_TOKENS,
        )

        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        if str(device) != "meta":
            encoded = {k: v.to(device) for k, v in encoded.items()}

        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.MAX_NEW_TOKENS}
        if self.TEMPERATURE > 0:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.TEMPERATURE,
                    "top_p": self.TOP_P,
                    "top_k": self.TOP_K,
                }
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = self.model.generate(**encoded, **gen_kwargs)

        input_len = encoded["input_ids"].shape[-1]
        generated = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        try:
            payload = _normalize_payload(_parse_json_object(generated))
        except Exception:
            repair_prompt = (
                "Converta o conteúdo abaixo para JSON estritamente válido. "
                "Retorne SOMENTE JSON, sem texto adicional.\n\n"
                f"Arquivo de origem: {source_name}\n"
                f"Conteúdo para corrigir:\n{generated[:6000]}"
            )
            repair_encoded = self.tokenizer(
                repair_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(self.MAX_INPUT_TOKENS, 2048),
            )
            if str(device) != "meta":
                repair_encoded = {k: v.to(device) for k, v in repair_encoded.items()}
            with torch.inference_mode():
                repair_outputs = self.model.generate(
                    **repair_encoded,
                    do_sample=False,
                    max_new_tokens=self.REPAIR_MAX_NEW_TOKENS,
                )
            repair_input_len = repair_encoded["input_ids"].shape[-1]
            repaired = self.tokenizer.decode(
                repair_outputs[0][repair_input_len:],
                skip_special_tokens=True,
            )
            payload = _normalize_payload(_parse_json_object(repaired))
        ref = ReferenciaLivroABNT.model_validate(payload)
        ref = _sanitize_referencia(ref)

        usage = {
            "input_tokens": int(input_len),
            "output_tokens": int(outputs.shape[-1] - input_len),
            "total_tokens": int(outputs.shape[-1]),
        }
        return ref, usage

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, File, Header, HTTPException, UploadFile

        api = FastAPI(title="ABNT Reference API (Modal)", version="1.0.0")

        class TextExtractionRequest(BaseModel):
            text: str = Field(..., min_length=1)
            source_name: str = "documento.txt"

        @api.get("/health")
        def health():
            return {
                "status": "ok",
                "model": self.MODEL_ID,
                "model_device": self.MODEL_DEVICE,
                "max_pages": self.MAX_PAGES,
                "max_input_chars": self.MAX_INPUT_CHARS,
                "max_input_tokens": self.MAX_INPUT_TOKENS,
                "max_new_tokens": self.MAX_NEW_TOKENS,
                "repair_max_new_tokens": self.REPAIR_MAX_NEW_TOKENS,
            }

        @api.post("/v1/abnt/extract/text")
        def extract_text(
            req: TextExtractionRequest,
            x_api_key: Optional[str] = Header(default=None),
        ):
            self._require_key(x_api_key)
            try:
                ref, usage = self._extract_from_text(req.text, req.source_name)
                return _build_response(ref, usage, self.MODEL_ID, req.source_name)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Falha na extração ABNT: {exc}")

        @api.post("/v1/abnt/extract/file")
        async def extract_file(
            file: UploadFile = File(...),
            max_pages: Optional[int] = None,
            x_api_key: Optional[str] = Header(default=None),
        ):
            self._require_key(x_api_key)

            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="empty file")

            source_name = file.filename or "arquivo"
            suffix = Path(source_name).suffix.lower()

            try:
                if suffix == ".pdf":
                    pages = max_pages if (max_pages is not None and max_pages > 0) else self.MAX_PAGES
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
                ref, usage = self._extract_from_text(extracted, source_name)
                payload = _build_response(ref, usage, self.MODEL_ID, source_name)
                payload["input_chars"] = len(extracted)
                return payload
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Falha na extração ABNT: {exc}")

        return api
