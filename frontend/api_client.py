from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0


def configured_timeout(
    environ: Mapping[str, str],
    env_values: Mapping[str, str | None],
    name: str,
    default: float,
) -> float:
    raw = environ.get(name)
    if raw is None:
        raw = env_values.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def vision_timeout_seconds(
    environ: Mapping[str, str],
    env_values: Mapping[str, str | None],
) -> float:
    return configured_timeout(
        environ, env_values, "VISION_TIMEOUT_SECONDS", 900.0
    )


def image_analyze_timeout_seconds(
    environ: Mapping[str, str],
    env_values: Mapping[str, str | None],
) -> float:
    vision_timeout = vision_timeout_seconds(environ, env_values)
    default = max(300.0, vision_timeout + 30.0)
    return configured_timeout(
        environ, env_values, "IMAGE_ANALYZE_TIMEOUT_SECONDS", default
    )


def chat_timeout_seconds(
    environ: Mapping[str, str],
    env_values: Mapping[str, str | None],
) -> float:
    model_timeout = configured_timeout(
        environ, env_values, "MODEL_TIMEOUT_SECONDS", 60.0
    )
    vision_timeout = vision_timeout_seconds(environ, env_values)
    default = max(180.0, model_timeout, vision_timeout) + 30.0
    return configured_timeout(environ, env_values, "CHAT_TIMEOUT_SECONDS", default)


def error_detail(response: httpx.Response) -> tuple[str, str | None]:
    try:
        payload = response.json()
    except ValueError:
        return f"Backend returned HTTP {response.status_code}.", None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        message = str(
            detail.get("message") or f"Backend returned HTTP {response.status_code}."
        )
        code = detail.get("code")
        return message, str(code) if code else None
    if isinstance(detail, str) and detail.strip():
        return detail, None
    return f"Backend returned HTTP {response.status_code}.", None


def request_json(
    api: str,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    transport: httpx.BaseTransport | None = None,
    request_label: str = "Request",
    timeout_setting: str | None = None,
) -> tuple[object | None, str, str | None]:
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.request(
                method,
                f"{api}{path}",
                params=params,
                json=json_body,
            )
    except httpx.ConnectTimeout:
        return (
            None,
            f"Timed out connecting to the backend at {api} after "
            f"{timeout:g} seconds. Start it with: "
            "uvicorn backend.app:app --reload --port 8000.",
            "backend_connect_timeout",
        )
    except httpx.TimeoutException:
        hint = (
            f"Increase {timeout_setting} in .env only if this repeats after the "
            "model has loaded."
            if timeout_setting
            else "Check the backend logs if this repeats."
        )
        return (
            None,
            f"{request_label} did not receive a backend response within "
            f"{timeout:g} seconds. The backend may still be processing it; wait "
            "before retrying so it does not start duplicate work. A local model "
            f"may still be loading. {hint}",
            "backend_timeout",
        )
    except httpx.ConnectError:
        return (
            None,
            f"Could not connect to the backend at {api}. Start it with: "
            "uvicorn backend.app:app --reload --port 8000.",
            "backend_unreachable",
        )
    except httpx.HTTPError:
        return (
            None,
            f"The connection to the backend at {api} failed before a complete "
            "response was received. Check the API logs and retry.",
            "backend_connection_error",
        )

    if response.status_code >= 400:
        message, code = error_detail(response)
        return None, message, code or f"backend_http_{response.status_code}"
    try:
        return response.json(), "", None
    except ValueError:
        return None, "The backend returned an invalid response.", "invalid_response"
