"""LLM provider — OpenAI Responses API."""
from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _extract_text(resp) -> str:
        text = (getattr(resp, "output_text", None) or "").strip()
        if text:
            return text

        chunks: list[str] = []
        for item in getattr(resp, "output", None) or []:
            for content in getattr(item, "content", None) or []:
                if getattr(content, "type", None) != "output_text":
                    continue
                value = getattr(content, "text", None)
                if value:
                    chunks.append(value)
        return "".join(chunks).strip()

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
        text_out = self._extract_text(resp)

        incomplete = getattr(resp, "incomplete_details", None)
        reason = getattr(incomplete, "reason", None)
        if output_format == "json" and getattr(resp, "status", None) == "incomplete" and reason == "max_output_tokens":
            retry_effort = "low" if effort in ("medium", "high") else effort
            if retry_effort != effort:
                logger.warning(
                    "OpenAI response exhausted max_output_tokens before visible output; retrying with effort=%s",
                    retry_effort,
                )
                kwargs["reasoning"] = {"effort": retry_effort}
                retry_resp = self.client.responses.create(**kwargs)
                retry_text = self._extract_text(retry_resp)
                if retry_text:
                    return retry_text

        if text_out:
            return text_out

        logger.warning(
            "OpenAI response produced no visible output (status=%s, incomplete_reason=%s)",
            getattr(resp, "status", None),
            reason,
        )
        return text_out


def make_provider(model: str) -> LLMProvider:
    logger.info("LLM provider: openai (model=%s)", model)
    return OpenAIProvider(model)
