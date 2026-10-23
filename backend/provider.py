from __future__ import annotations

import re
import time
from typing import Any

import httpx

from .config import Settings

_VISION_PATTERN = re.compile(
    r"(llava|bakllava|vision|minicpm-v|moondream|pixtral|qwen[^/]*vl|gemma3|granite[^/]*vision)",
    re.IGNORECASE,
)


def supports_vision(model_name: str) -> bool:
    return bool(model_name and _VISION_PATTERN.search(model_name))


class ModelUnavailable(RuntimeError):
    code = "model_unavailable"


class ModelNotConfigured(ModelUnavailable):
    code = "model_not_configured"


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, str):
        detail = error
    elif isinstance(error, dict):
        detail = str(error.get("message") or error.get("detail") or "")
    else:
        detail = str(payload.get("message") or payload.get("detail") or "")
    return " ".join(detail.split())[:300]


class ChatProvider:
    def __init__(
        self,
        config: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport

    @property
    def configured(self) -> bool:
        return self.config.model_configured

    @property
    def vision_supported(self) -> bool:
        if self.config.model_supports_vision is not None:
            return self.config.model_supports_vision
        return supports_vision(self.config.model_name)

    @property
    def endpoint(self) -> str:
        return self.config.chat_completions_url

    def _headers(self) -> dict[str, str]:
        if not self.config.model_api_key:
            return {}
        return {"Authorization": f"Bearer {self.config.model_api_key}"}

    def _timeout(self, total: float | None = None) -> httpx.Timeout:
        seconds = total or self.config.model_timeout_seconds
        return httpx.Timeout(seconds, connect=min(10.0, seconds))

    def _http_error(self, response: httpx.Response) -> str:
        model = self.config.model_name
        detail = _response_detail(response)
        lowered = detail.lower()
        if response.status_code in {401, 403}:
            return "The model endpoint rejected authentication. Check MODEL_API_KEY."
        if response.status_code == 404:
            if "not found" in lowered or "pull" in lowered:
                return (
                    f'Model "{model}" is not available on the endpoint. '
                    f"Run: ollama pull {model}"
                )
            return (
                "The chat-completions endpoint was not found. Verify MODEL_BASE_URL "
                "ends with /v1 and that the model server is running."
            )
        if response.status_code == 400:
            suffix = f" Endpoint detail: {detail}" if detail else ""
            return f"The model endpoint rejected the request (HTTP 400).{suffix}"
        if response.status_code == 429:
            return "The model endpoint is rate limiting requests. Try again shortly."
        suffix = f" Endpoint detail: {detail}" if detail else ""
        return f"The model endpoint returned HTTP {response.status_code}.{suffix}"

    def _transport_error(self, exc: httpx.HTTPError) -> str:
        base_url = self.config.model_base_url
        if isinstance(exc, httpx.TimeoutException):
            return (
                f"The model endpoint timed out after "
                f"{self.config.model_timeout_seconds:g} seconds. The model may still be "
                "loading; try again, or increase MODEL_TIMEOUT_SECONDS."
            )
        if isinstance(exc, httpx.ConnectError):
            return (
                f"Could not connect to the model endpoint at {base_url}. "
                "If using Ollama, start it with 'ollama serve' and verify MODEL_BASE_URL."
            )
        return f"Could not communicate with the model endpoint at {base_url}."

    async def generate(
        self,
        *,
        system_prompt: str,
        message: str,
        history: list[dict[str, Any]],
        image_data_url: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not self.configured:
            missing = ", ".join(self.config.missing_model_settings) or "MODEL_BASE_URL, MODEL_NAME"
            raise ModelNotConfigured(
                f"Model is not configured. Missing: {missing}. Add them to the project "
                ".env file and restart the API."
            )
        if image_data_url and not self.vision_supported:
            raise ModelUnavailable(
                f'The configured model "{self.config.model_name}" does not support image '
                "input. Remove the image or configure a vision-capable model such as "
                "Qwen2.5-VL or LLaVA."
            )

        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        for item in history[-20:]:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content})

        if image_data_url:
            content: str | list[dict[str, Any]] = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        else:
            content = message
        messages.append({"role": "user", "content": content})

        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": 0.2,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout(),
                transport=self._transport,
            ) as client:
                response = await client.post(
                    self.endpoint,
                    headers=self._headers(),
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise ModelUnavailable(self._transport_error(exc)) from exc

        if response.status_code >= 400:
            raise ModelUnavailable(self._http_error(response))

        try:
            body = response.json()
        except ValueError as exc:
            raise ModelUnavailable(
                "The model endpoint returned invalid JSON instead of a chat response."
            ) from exc

        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ModelUnavailable(
                "The model endpoint returned an invalid chat response (missing choices)."
            )
        first = choices[0]
        response_message = first.get("message") if isinstance(first, dict) else None
        answer = response_message.get("content") if isinstance(response_message, dict) else None
        if isinstance(answer, list):
            answer = "".join(
                str(part.get("text", ""))
                for part in answer
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if not isinstance(answer, str) or not answer.strip():
            raise ModelUnavailable(
                "The model endpoint returned an empty or malformed assistant response."
            )
        return answer.strip()

    async def probe(self, *, live: bool = False) -> dict[str, Any]:
        info: dict[str, Any] = {
            "configured": self.configured,
            "status": "not_configured",
            "reachable": False,
            "model_available": False,
            "model": self.config.model_name or None,
            "base_url": self.config.model_base_url or None,
            "endpoint": self.endpoint or None,
            "vision_supported": self.vision_supported,
            "live_checked": False,
            "live_ok": False,
            "detail": "",
        }
        if not self.configured:
            missing = ", ".join(self.config.missing_model_settings)
            info["detail"] = (
                f"Missing {missing}. Add them to the project .env file and restart the API."
            )
            return info

        models_url = f"{self.config.model_base_url}/models"
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout(min(8.0, self.config.model_timeout_seconds)),
                transport=self._transport,
            ) as client:
                response = await client.get(models_url, headers=self._headers())
        except httpx.HTTPError as exc:
            info.update(status="unreachable", detail=self._transport_error(exc))
            return info

        info["reachable"] = True
        if response.status_code >= 400:
            info.update(status="error", detail=self._http_error(response))
            return info

        try:
            body = response.json()
        except ValueError:
            info.update(status="error", detail="The endpoint returned an invalid model list.")
            return info
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            info.update(status="error", detail="The endpoint returned an invalid model list.")
            return info

        model_ids = {
            item.get("id")
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        available = self.config.model_name in model_ids
        if not available and ":" not in self.config.model_name:
            available = any(
                model_id.split(":", 1)[0] == self.config.model_name for model_id in model_ids
            )
        info["model_available"] = available
        if not available:
            info.update(
                status="model_missing",
                detail=(
                    f'Model "{self.config.model_name}" is not available at the endpoint. '
                    f"Run: ollama pull {self.config.model_name}"
                ),
            )
            return info

        info.update(status="available", detail="Model endpoint is reachable and the model is listed.")
        if not live:
            return info

        started = time.perf_counter()
        info["live_checked"] = True
        try:
            reply = await self.generate(
                system_prompt="You are a connection test. Reply with exactly: OK",
                message="Reply with exactly: OK",
                history=[],
                max_tokens=8,
            )
        except ModelUnavailable as exc:
            info.update(status="error", live_ok=False, detail=str(exc))
            return info
        info.update(
            live_ok=bool(reply.strip()),
            live_latency_seconds=round(time.perf_counter() - started, 2),
            detail="The model returned a real chat-completions response.",
        )
        return info
