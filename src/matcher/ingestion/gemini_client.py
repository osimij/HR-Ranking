"""Helpers for calling Google's Gemini API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


try:  # pragma: no cover - optional dependency until installed
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]


DEFAULT_MODEL = "gemini-flash-lite-latest"


class GeminiClientError(RuntimeError):
    """Raised when Gemini client prerequisites are missing."""


def _ensure_dependency() -> None:
    if genai is None or types is None:
        raise GeminiClientError(
            "google-genai package is not installed. Install it via `pip install google-genai`."
        )


def _ensure_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise GeminiClientError("GEMINI_API_KEY environment variable is not set.")
    return key


def build_client(api_key: Optional[str] = None) -> "genai.Client":
    """Instantiate a Gemini client using the provided API key or environment variable."""
    _ensure_dependency()
    key = _ensure_api_key(api_key)
    return genai.Client(api_key=key)


@dataclass
class GeminiConfig:
    """Configuration for Gemini content generation."""

    model: str = DEFAULT_MODEL
    thinking_budget: int = 0
    image_size: str = "1K"

    def to_generate_config(self) -> "types.GenerateContentConfig":
        _ensure_dependency()
        config_kwargs = {
            "thinking_config": types.ThinkingConfig(thinking_budget=self.thinking_budget),
        }
        try:
            config_kwargs["image_config"] = types.ImageConfig(image_size=self.image_size)
        except AttributeError:
            pass
        return types.GenerateContentConfig(**config_kwargs)


def stream_content_completion(
    parts: Sequence["types.Part"],
    *,
    model: Optional[str] = None,
    client: Optional["genai.Client"] = None,
    config: Optional[GeminiConfig] = None,
) -> str:
    """Generate a response from Gemini given arbitrary content parts."""

    _ensure_dependency()

    client = client or build_client()
    cfg = config or GeminiConfig(model=model or DEFAULT_MODEL)

    contents = [types.Content(role="user", parts=list(parts))]

    generate_config = cfg.to_generate_config()
    chunks: Iterable["types.GenerateContentResponse"] = client.models.generate_content_stream(
        model=cfg.model if cfg.model else DEFAULT_MODEL,
        contents=contents,
        config=generate_config,
    )

    output_parts: list[str] = []
    for chunk in chunks:
        text = getattr(chunk, "text", None)
        if text:
            output_parts.append(text)
    return "".join(output_parts)


def stream_text_completion(
    prompt: str,
    *,
    model: Optional[str] = None,
    client: Optional["genai.Client"] = None,
    config: Optional[GeminiConfig] = None,
) -> str:
    """Generate a text response from Gemini using streaming API."""

    part = types.Part.from_text(text=prompt)
    return stream_content_completion([part], model=model, client=client, config=config)


def gemini_completion(prompt: str) -> str:
    """Convenience wrapper returning non-streaming text for prompt."""
    return stream_text_completion(prompt)


def generate_json_from_pdf(
    pdf_bytes: bytes,
    prompt: str,
    *,
    model: Optional[str] = None,
    client: Optional["genai.Client"] = None,
    config: Optional[GeminiConfig] = None,
) -> str:
    """Generate JSON text given a PDF document and instruction prompt."""

    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    prompt_part = types.Part.from_text(text=prompt)
    return stream_content_completion([prompt_part, pdf_part], model=model, client=client, config=config)


__all__ = [
    "GeminiConfig",
    "GeminiClientError",
    "build_client",
    "stream_text_completion",
    "stream_content_completion",
    "gemini_completion",
    "generate_json_from_pdf",
]
