from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from PIL import Image

from .logging_config import get_logger

log = get_logger("ollama")


DEFAULT_OLLAMA_URL = ""
DEFAULT_VISION_MODEL = "qwen3-vl:8b-instruct"


class OllamaError(RuntimeError):
    """Erro de comunicação ou resposta inválida do servidor Ollama."""


@dataclass(frozen=True)
class LlmInspection:
    verdict: str
    confidence: float
    summary: str
    suspect_leds: list[str]
    observations: list[str]
    model: str


def _base_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise OllamaError("O endereço do Ollama deve começar com http:// ou https://.")
    return normalized


def _request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="GET" if payload is None else "POST",
    )

    # Ignora proxies do ambiente para alcançar diretamente o servidor da rede local.
    opener = build_opener(ProxyHandler({}))
    try:
        log.debug("HTTP %s %s", request.method, url)
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        log.error("Ollama HTTP %d: %s", exc.code, detail[:300])
        raise OllamaError(f"Ollama respondeu HTTP {exc.code}: {detail[:300]}") from exc
    except URLError as exc:
        log.error("Falha conexão Ollama: %s", exc.reason)
        raise OllamaError(f"Não foi possível conectar ao Ollama: {exc.reason}") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        log.error("Timeout ou resposta inválida do Ollama: %s", exc)
        raise OllamaError("O Ollama demorou demais ou devolveu uma resposta inválida.") from exc


def list_vision_models(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 8.0) -> list[str]:
    response = _request_json(f"{_base_url(base_url)}/api/tags", None, timeout)
    models = response.get("models")
    if not isinstance(models, list):
        raise OllamaError("A resposta de /api/tags não contém uma lista de modelos.")

    vision_models = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        capabilities = item.get("capabilities", [])
        if isinstance(name, str) and "vision" in capabilities:
            vision_models.append(name)
    return vision_models


def _image_as_base64(image: Image.Image, max_size: int = 1280) -> str:
    prepared = image.convert("RGB").copy()
    prepared.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    output = BytesIO()
    prepared.save(output, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def analyze_with_vision(
    reference: Image.Image,
    candidate: Image.Image,
    model: str,
    local_results: list[dict[str, Any]],
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 180.0,
) -> LlmInspection:
    """Compara a golden e a captura usando um modelo multimodal do Ollama."""
    if not model.strip():
        raise OllamaError("Selecione um modelo de visão.")

    log.info("Enviando para LLM: modelo=%s, %d LEDs locais", model, len(local_results))
    local_summary = json.dumps(local_results, ensure_ascii=False)
    prompt = (
        "INSPEÇÃO DE INTEGRIDADE DO LED\n\n"
        "Sua tarefa é identificar qualquer dano físico nos LEDs destacados na imagem "
        "'Câmera ao vivo', comparando-os com a referência 'Golden'. A PRIMEIRA imagem "
        "anexada é a Golden e a SEGUNDA imagem é a Câmera ao vivo.\n\n"
        "REJEITAR se houver: quebra de corpo, rachadura no fósforo, queima, mancha, "
        "fragmento, delaminação ou qualquer diferença estrutural visível na peça. "
        "Priorize danos físicos e diferenças de textura ou estrutura. Não rejeite apenas "
        "por pequenas mudanças de iluminação, cor, brilho, foco ou posição causadas pela captura.\n\n"
        "RELATÓRIO DE SAÍDA:\n"
        "Veredito: [OK / REJEITADO]\n"
        "Análise: se REJEITADO, descreva exatamente o defeito visível na imagem de teste "
        "em relação à Golden e informe o identificador do LED (por exemplo: 'O LED LD1 "
        "tem uma rachadura no fósforo perto do centro que não existe na Golden.'). "
        "Se OK, informe que não foi observada diferença estrutural relevante.\n\n"
        "Use as medições locais abaixo apenas como apoio; faça sua própria comparação visual. "
        "Os identificadores seguem a ordem LD1, LD2, ... da esquerda para a direita e de cima para baixo.\n\n"
        f"Medições locais: {local_summary}"
    )
    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["OK", "REJEITADO", "REVISAR"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 100},
            "summary": {"type": "string"},
            "suspect_leds": {"type": "array", "items": {"type": "string"}},
            "observations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["verdict", "confidence", "summary", "suspect_leds", "observations"],
    }
    payload = {
        "model": model,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0},
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [_image_as_base64(reference), _image_as_base64(candidate)],
            }
        ],
    }
    response = _request_json(f"{_base_url(base_url)}/api/chat", payload, timeout)
    try:
        content = response["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
        verdict = str(parsed["verdict"]).upper()
        if verdict == "REVISAR":
            verdict = "REJEITADO"
        if verdict not in {"OK", "REJEITADO"}:
            raise ValueError("veredito desconhecido")
        confidence = float(parsed["confidence"])
        if 0.0 <= confidence <= 1.0:
            confidence *= 100.0
        confidence = max(0.0, min(100.0, confidence))
        suspect_leds = [str(item) for item in parsed.get("suspect_leds", [])]
        observations = [str(item) for item in parsed.get("observations", [])]
        summary = str(parsed["summary"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        log.error("Resposta da LLM inválida: %s", exc)
        raise OllamaError("O modelo devolveu um parecer fora do formato esperado.") from exc

    log.info("LLM resposta: verdict=%s, confidence=%.1f%%, suspeitos=%s", verdict, confidence, suspect_leds)
    return LlmInspection(
        verdict=verdict,
        confidence=round(confidence, 1),
        summary=summary,
        suspect_leds=suspect_leds,
        observations=observations,
        model=model,
    )
