import asyncio
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import modal

APP_NAME = "ollama-gemma4-auth"
MODEL_DIR = "/ollama_models"
OLLAMA_PORT = 11434

# Escolha aqui o modelo
MODELS_TO_DOWNLOAD = [
    "gemma4:e4b",  # bom ponto de partida
    # "gemma4:26b",
    # "gemma4:31b",
]

OLLAMA_VERSION = "0.6.5"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .pip_install(
        "fastapi[standard]==0.115.0",
        "httpx==0.28.1",
        "pydantic==2.10.6",
    )
    .run_commands(
        f"OLLAMA_VERSION={OLLAMA_VERSION} curl -fsSL https://ollama.com/install.sh | sh",
        f"mkdir -p {MODEL_DIR}",
    )
    .env(
        {
            "OLLAMA_HOST": f"127.0.0.1:{OLLAMA_PORT}",
            "OLLAMA_MODELS": MODEL_DIR,
        }
    )
)

model_volume = modal.Volume.from_name("ollama-gemma4-models", create_if_missing=True)


@app.cls(
    image=image,
    gpu="L40S",
    timeout=60 * 30,
    scaledown_window=60 * 10,
    volumes={MODEL_DIR: model_volume},
    secrets=[
        modal.Secret.from_name("ollama-api-secret"),  # OLLAMA_API_KEY
    ],
)
@modal.concurrent(max_inputs=20)
class OllamaGateway:
    def _wait_for_port(self, host: str, port: int, timeout: int = 120) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=2):
                    return
            except OSError:
                time.sleep(1)
        raise RuntimeError(f"Ollama não ficou pronto em {host}:{port}")

    @modal.enter()
    def start_ollama(self):
        self.api_key = os.getenv("OLLAMA_API_KEY", "").strip()

        # Sobe o servidor do Ollama em background
        self.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        self._wait_for_port("127.0.0.1", OLLAMA_PORT, timeout=180)

        listed = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
        )
        if listed.returncode != 0:
            raise RuntimeError(f"Falha ao listar modelos: {listed.stderr}")

        existing = listed.stdout
        pulled_any = False

        for model_name in MODELS_TO_DOWNLOAD:
            model_tag = model_name if ":" in model_name else f"{model_name}:latest"
            if model_tag not in existing:
                proc = subprocess.run(
                    ["ollama", "pull", model_name],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"Falha ao baixar {model_name}: {proc.stderr}")
                pulled_any = True

        if pulled_any:
            model_volume.commit()

    @modal.exit()
    def stop_ollama(self):
        if getattr(self, "ollama_process", None) and self.ollama_process.poll() is None:
            self.ollama_process.terminate()
            try:
                self.ollama_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.ollama_process.kill()

    def _extract_api_key(
        self,
        authorization: Optional[str],
        x_api_key: Optional[str],
    ) -> Optional[str]:
        if authorization and authorization.startswith("Bearer "):
            return authorization.replace("Bearer ", "", 1).strip()
        if x_api_key:
            return x_api_key.strip()
        return None

    def _require_key(
        self,
        authorization: Optional[str],
        x_api_key: Optional[str],
    ) -> None:
        from fastapi import HTTPException

        # Se não configurar OLLAMA_API_KEY, deixa sem auth.
        # Em prod, configure.
        if not self.api_key:
            return

        provided = self._extract_api_key(authorization, x_api_key)
        if not provided or provided != self.api_key:
            raise HTTPException(status_code=401, detail="unauthorized")

    @modal.asgi_app()
    def web(self):
        import httpx
        from fastapi import FastAPI, Header, HTTPException, Request, Response

        api = FastAPI(title="Authenticated Ollama Gateway", version="1.0.0")

        OLLAMA_BASE = f"http://127.0.0.1:{OLLAMA_PORT}"

        @api.get("/health")
        async def health():
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    resp = await client.get(f"{OLLAMA_BASE}/api/tags")
                    resp.raise_for_status()
                    return {
                        "status": "ok",
                        "models": resp.json().get("models", []),
                        "backend": "ollama",
                    }
                except Exception as exc:
                    raise HTTPException(
                        status_code=500, detail=f"healthcheck failed: {exc}"
                    )

        async def proxy_request(
            request: Request,
            path: str,
            authorization: Optional[str],
            x_api_key: Optional[str],
        ) -> Response:
            self._require_key(authorization, x_api_key)

            # Repassa quase tudo, removendo host/content-length
            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("content-length", None)

            # Não encaminha sua chave custom para o Ollama
            headers.pop("x-api-key", None)

            body = await request.body()
            query = request.url.query
            target_url = f"{OLLAMA_BASE}/{path}"
            if query:
                target_url = f"{target_url}?{query}"

            timeout = httpx.Timeout(600.0, connect=30.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                upstream = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=body,
                )

                excluded = {
                    "content-encoding",
                    "transfer-encoding",
                    "connection",
                }

                response_headers = {
                    k: v
                    for k, v in upstream.headers.items()
                    if k.lower() not in excluded
                }

                return Response(
                    content=upstream.content,
                    status_code=upstream.status_code,
                    headers=response_headers,
                    media_type=upstream.headers.get("content-type"),
                )

        # Rotas mais usadas
        @api.api_route("/v1/chat/completions", methods=["POST"])
        async def chat_completions(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_api_key: Optional[str] = Header(default=None),
        ):
            return await proxy_request(
                request, "v1/chat/completions", authorization, x_api_key
            )

        @api.api_route("/v1/responses", methods=["POST"])
        async def responses_api(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_api_key: Optional[str] = Header(default=None),
        ):
            return await proxy_request(
                request, "v1/responses", authorization, x_api_key
            )

        @api.api_route("/api/tags", methods=["GET"])
        async def api_tags(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_api_key: Optional[str] = Header(default=None),
        ):
            return await proxy_request(request, "api/tags", authorization, x_api_key)

        # Catch-all para outros endpoints do Ollama/OpenAI-compatible
        @api.api_route(
            "/{full_path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        )
        async def catch_all(
            full_path: str,
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_api_key: Optional[str] = Header(default=None),
        ):
            return await proxy_request(request, full_path, authorization, x_api_key)

        return api
