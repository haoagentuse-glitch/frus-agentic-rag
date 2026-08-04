"""Ollama access with schema-enforced JSON.

Structured output goes through Ollama's native `format` parameter with the
Pydantic JSON schema, not `with_structured_output` and not regex extraction:
a 4B model will happily emit prose around a JSON blob, and we would rather get
a hard validation error than a silently truncated plan.
"""

from __future__ import annotations

import asyncio
import json
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from frus_agentic_rag import observability as obs
from frus_agentic_rag.config import get_settings

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """The model produced something the schema rejects, twice."""


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self.num_ctx = settings.ollama_num_ctx
        self.temperature = settings.ollama_temperature
        self.timeout = settings.ollama_timeout_s
        self.num_predict = settings.ollama_num_predict
        self.calls = 0

    async def _post(self, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()

    def _payload(
        self, system: str, user: str, fmt: dict | None, num_predict: int | None = None
    ) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "top_p": 1.0,
                # Backstop for runaway grammar-constrained decoding. The schema
                # maxItems bounds should make this unreachable; it is here so a
                # schema change can never cost a 5-minute timeout again.
                "num_predict": num_predict or self.num_predict,
            },
        }
        if fmt is not None:
            payload["format"] = fmt
        return payload

    async def generate(self, system: str, user: str) -> str:
        self.calls += 1
        with obs.llm_span(self.model, system, user) as sp:
            data = await self._post(self._payload(system, user, None))
            content = data["message"]["content"]
            obs.record_llm_output(sp, content, data)
        return content

    async def structured(
        self, system: str, user: str, schema: type[T], num_predict: int | None = None
    ) -> T:
        """One retry with the validation error fed back, then give up."""
        fmt = schema.model_json_schema()
        last: Exception | None = None
        prompt = user

        for attempt in range(2):
            self.calls += 1
            with obs.llm_span(self.model, system, prompt, schema=schema.__name__) as sp:
                data = await self._post(self._payload(system, prompt, fmt, num_predict))
                raw = data["message"]["content"]
                obs.record_llm_output(sp, raw, data)
                try:
                    parsed = schema.model_validate_json(raw)
                except (ValidationError, json.JSONDecodeError) as exc:
                    last = exc
                    if sp is not None:
                        sp.set_attribute("frus.schema_valid", False)
                else:
                    if sp is not None:
                        sp.set_attribute("frus.schema_valid", True)
                    return parsed
            if attempt == 0:
                prompt = (
                    f"{user}\n\nYour previous reply was rejected by the schema:\n"
                    f"{str(last)[:600]}\nReturn only valid JSON for the schema."
                )
        raise StructuredOutputError(f"{schema.__name__} invalid after retry: {last}")

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{self.host}/api/tags")
            r.raise_for_status()
            tags = [m["name"] for m in r.json().get("models", [])]
        return {
            "host": self.host,
            "models": tags,
            "target": self.model,
            "target_present": self.model in tags,
        }


_CLIENT: OllamaClient | None = None


def get_client() -> OllamaClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OllamaClient()
    return _CLIENT


def check_sync() -> dict:
    return asyncio.run(get_client().health())
