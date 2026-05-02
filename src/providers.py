"""LLM provider abstraction.

The submitter chooses a model name; we route to the right backend:
  - Names starting with `gpt-` or `o1`/`o3`/`o4` → OpenAI Responses API
  - Names starting with `gemini-` → Google Gemini API
  - Override with LLM_PROVIDER env var if needed.
"""
from __future__ import annotations

import base64
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def detect_provider(model: str) -> str:
    override = os.environ.get("LLM_PROVIDER", "").lower()
    if override in ("openai", "gemini"):
        return override
    m = (model or "").lower()
    if m.startswith("gemini"):
        return "gemini"
    return "openai"


class LLMProvider(ABC):
    @abstractmethod
    def respond(
        self,
        *,
        system: str,
        text: str,
        images: list[tuple[str, bytes]],
        output_format: str,
        effort: str,
        max_output_tokens: int,
    ) -> str:
        ...


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str) -> None:
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def respond(
        self,
        *,
        system: str,
        text: str,
        images: list[tuple[str, bytes]],
        output_format: str,
        effort: str,
        max_output_tokens: int,
    ) -> str:
        content: list[dict] = [{"type": "input_text", "text": text}]
        for _name, jpeg in images:
            b64 = base64.b64encode(jpeg).decode("ascii")
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
                "detail": "high",
            })

        kwargs: dict = dict(
            model=self.model,
            instructions=system,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": effort},
            max_output_tokens=max_output_tokens,
        )
        # GPT-5 reasoning models ignore both `temperature` and `seed` on the
        # Responses API; we leave decoding params at their defaults.
        if output_format == "json":
            kwargs["text"] = {"format": {"type": "json_object"}}
        resp = self.client.responses.create(**kwargs)
        return (resp.output_text or "").strip()


class GeminiProvider(LLMProvider):
    """Google Gemini via google-genai SDK.

    Notes:
    - Auth: uses GEMINI_API_KEY (or GOOGLE_API_KEY as fallback). The SDK strips
      a leading `=` if the env was set with an extra `=` character.
    - Reasoning effort is mapped to thinking_budget for Gemini 2.5+.
    - JSON output uses response_mime_type="application/json".
    """

    EFFORT_TO_BUDGET = {
        "minimal": 0,
        "low": 1024,
        "medium": 4096,
        "high": 16384,
    }

    def __init__(self, model: str) -> None:
        from google import genai
        from google.genai import types as gtypes  # noqa: F401
        self._genai = genai
        self._gtypes = gtypes
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        api_key = api_key.lstrip("=").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def respond(
        self,
        *,
        system: str,
        text: str,
        images: list[tuple[str, bytes]],
        output_format: str,
        effort: str,
        max_output_tokens: int,
    ) -> str:
        gtypes = self._gtypes
        parts: list = [gtypes.Part.from_text(text=text)]
        for _name, jpeg in images:
            parts.append(gtypes.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))

        config_kwargs: dict = dict(
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            temperature=0.0,
            seed=0,
        )
        # Thinking budget — Gemini 2.5+ and 3.x support this.
        if any(v in self.model for v in ("2.5", "3-pro", "3.0", "3.1", "3-flash")):
            budget = self.EFFORT_TO_BUDGET.get(effort, 4096)
            try:
                config_kwargs["thinking_config"] = gtypes.ThinkingConfig(thinking_budget=budget)
            except Exception as e:  # noqa: BLE001
                logger.warning("ThinkingConfig not accepted for %s: %s", self.model, e)
        if output_format == "json":
            config_kwargs["response_mime_type"] = "application/json"

        resp = self.client.models.generate_content(
            model=self.model,
            contents=[gtypes.Content(role="user", parts=parts)],
            config=gtypes.GenerateContentConfig(**config_kwargs),
        )
        return (resp.text or "").strip()


def make_provider(model: str) -> LLMProvider:
    kind = detect_provider(model)
    logger.info("LLM provider: %s (model=%s)", kind, model)
    if kind == "gemini":
        return GeminiProvider(model)
    return OpenAIProvider(model)
