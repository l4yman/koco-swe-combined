import json
import logging
import os
import threading
import time
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from moatless.completion.completion import CompletionModel


DEFAULT_AIDP_URL = "https://aidp.bytedance.net/api/modelhub/online/v2/crawl"
logger = logging.getLogger(__name__)
_USAGE_LOG_LOCK = threading.Lock()


class AIDPMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None


class AIDPChoice(BaseModel):
    index: int = 0
    message: AIDPMessage
    finish_reason: str | None = None


class AIDPUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AIDPCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"aidp-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[AIDPChoice]
    usage: AIDPUsage = Field(default_factory=AIDPUsage)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class AIDPCompletionModel(CompletionModel):
    """CompletionModel backed by ByteDance AIDP ModelHub chat endpoint.

    The rest of SWE-Exp expects LiteLLM/OpenAI-shaped responses with
    ``choices[0].message.content`` and ``usage``. This class preserves that
    shape while sending requests to the AIDP endpoint shown in the KOCO task.
    """

    aidp_ak: str | None = Field(default=None, exclude=True)
    aidp_url: str = Field(default=DEFAULT_AIDP_URL)
    usage_stage: str = Field(default="")
    usage_log: str | None = Field(default=None)

    def _with_ak(self) -> str:
        ak = self.aidp_ak or self.model_api_key or os.getenv("AIDP_AK")
        if not ak:
            raise ValueError("AIDP AK is required. Pass --aidp-ak or set AIDP_AK.")

        parsed = urlparse(self.model_base_url or self.aidp_url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query.setdefault("ak", ak)
        return urlunparse(parsed._replace(query=urlencode(query)))

    def _extract_content(self, response: dict[str, Any]) -> str:
        candidates = [
            response,
            response.get("data") if isinstance(response.get("data"), dict) else {},
            response.get("result") if isinstance(response.get("result"), dict) else {},
        ]
        for candidate in candidates:
            choices = candidate.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict) and message.get("content") is not None:
                        return str(message["content"])
                    if first.get("text") is not None:
                        return str(first["text"])
            if candidate.get("content") is not None:
                return str(candidate["content"])
            if candidate.get("message") is not None and isinstance(candidate["message"], str):
                return candidate["message"]

        raise ValueError(f"Could not extract assistant content from AIDP response keys: {list(response.keys())}")

    def _find_usage(self, value: Any, depth: int = 0) -> dict[str, Any] | None:
        if depth > 4:
            return None
        if isinstance(value, dict):
            usage = value.get("usage")
            if isinstance(usage, dict):
                return usage
            usage_keys = {
                "prompt_tokens",
                "input_tokens",
                "completion_tokens",
                "output_tokens",
                "total_tokens",
            }
            if usage_keys & set(value):
                return value
            for nested in value.values():
                found = self._find_usage(nested, depth + 1)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = self._find_usage(item, depth + 1)
                if found is not None:
                    return found
        return None

    def _usage_int(self, usage: dict[str, Any], *keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return 0

    def _extract_usage(self, response: dict[str, Any]) -> AIDPUsage:
        usage = self._find_usage(response) or {}
        prompt_tokens = self._usage_int(usage, "prompt_tokens", "input_tokens")
        completion_tokens = self._usage_int(usage, "completion_tokens", "output_tokens")
        total_tokens = self._usage_int(usage, "total_tokens") or prompt_tokens + completion_tokens
        return AIDPUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def _write_usage_record(
        self,
        *,
        call_id: str,
        messages: list[dict],
        char_count: int,
        elapsed_seconds: float,
        usage: AIDPUsage | None = None,
        error: str | None = None,
    ) -> None:
        usage_log = self.usage_log or os.getenv("KOCO_USAGE_LOG")
        if not usage_log:
            return

        record = {
            "timestamp": int(time.time()),
            "call_id": call_id,
            "stage": self.usage_stage or os.getenv("KOCO_USAGE_STAGE") or "infer",
            "model": self.model,
            "messages": len(messages),
            "chars": char_count,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "elapsed_seconds": elapsed_seconds,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "error": error or "",
        }
        path = os.path.abspath(usage_log)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with _USAGE_LOG_LOCK:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write AIDP usage record to %s: %s", path, exc)

    def _litellm_base_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        response_format: dict | None = None,
    ) -> AIDPCompletionResponse:
        if tools:
            raise ValueError("AIDPCompletionModel supports JSON/ReAct prompting, not tool-call transport.")

        payload: dict[str, Any] = {
            "stream": False,
            "messages": messages,
            "model": self.model,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if response_format:
            payload["response_format"] = response_format
        if tool_choice:
            payload["tool_choice"] = tool_choice

        char_count = sum(len(str(message.get("content", ""))) for message in messages)
        call_id = f"swe-exp-koco-{uuid.uuid4().hex}"
        logger.info(
            "AIDP request model=%s messages=%d chars=%d max_tokens=%d timeout=%s",
            self.model,
            len(messages),
            char_count,
            self.max_tokens,
            self.timeout,
        )

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self._with_ak(),
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-TT-LOGID": call_id,
            },
        )

        start_time = time.time()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_text = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            self._write_usage_record(
                call_id=call_id,
                messages=messages,
                char_count=char_count,
                elapsed_seconds=time.time() - start_time,
                error=f"AIDP HTTP {exc.code}: {error_body[:1000]}",
            )
            raise RuntimeError(f"AIDP HTTP {exc.code}: {error_body[:1000]}") from exc
        except URLError as exc:
            self._write_usage_record(
                call_id=call_id,
                messages=messages,
                char_count=char_count,
                elapsed_seconds=time.time() - start_time,
                error=f"AIDP request failed: {exc}",
            )
            raise RuntimeError(f"AIDP request failed: {exc}") from exc

        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self._write_usage_record(
                call_id=call_id,
                messages=messages,
                char_count=char_count,
                elapsed_seconds=time.time() - start_time,
                error=f"AIDP returned non-JSON response: {raw_text[:1000]}",
            )
            raise RuntimeError(f"AIDP returned non-JSON response: {raw_text[:1000]}") from exc

        usage = self._extract_usage(raw)
        try:
            content = self._extract_content(raw)
        except Exception as exc:
            self._write_usage_record(
                call_id=call_id,
                messages=messages,
                char_count=char_count,
                elapsed_seconds=time.time() - start_time,
                usage=usage,
                error=f"AIDP response parse failed: {exc}",
            )
            raise
        self._write_usage_record(
            call_id=call_id,
            messages=messages,
            char_count=char_count,
            elapsed_seconds=time.time() - start_time,
            usage=usage,
        )
        logger.info("AIDP response model=%s chars=%d", self.model, len(content))
        return AIDPCompletionResponse(
            model=self.model,
            choices=[AIDPChoice(message=AIDPMessage(content=content))],
            usage=usage,
            raw_response=raw,
        )
