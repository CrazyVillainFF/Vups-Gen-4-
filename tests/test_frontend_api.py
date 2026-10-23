from __future__ import annotations

import json
from typing import Any

import httpx

from frontend.api_client import (
    chat_timeout_seconds,
    image_analyze_timeout_seconds,
    request_json,
    vision_timeout_seconds,
)


def mock_transport(handler: Any) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_chat_timeout_tracks_model_settings_and_allows_override() -> None:
    assert chat_timeout_seconds({}, {}) == 930.0
    assert chat_timeout_seconds({}, {"MODEL_TIMEOUT_SECONDS": "1000"}) == 1030.0
    assert (
        chat_timeout_seconds(
            {"CHAT_TIMEOUT_SECONDS": "1200"},
            {"CHAT_TIMEOUT_SECONDS": "600"},
        )
        == 1200.0
    )
    assert chat_timeout_seconds({}, {"MODEL_TIMEOUT_SECONDS": "invalid"}) == 930.0


def test_image_timeout_always_exceeds_the_backend_vision_timeout() -> None:
    assert vision_timeout_seconds({}, {}) == 900.0
    assert image_analyze_timeout_seconds({}, {}) == 930.0
    assert (
        image_analyze_timeout_seconds(
            {},
            {"VISION_TIMEOUT_SECONDS": "1200"},
        )
        == 1230.0
    )
    assert (
        image_analyze_timeout_seconds(
            {"IMAGE_ANALYZE_TIMEOUT_SECONDS": "1500"},
            {"VISION_TIMEOUT_SECONDS": "1200"},
        )
        == 1500.0
    )


def test_successful_chat_response_is_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://api.test/chat"
        assert json.loads(request.content) == {"message": "hello"}
        return httpx.Response(
            200,
            json={
                "conversation_id": "conversation-1",
                "answer": "A real answer.",
                "skill": "auto",
                "model": "qwen2.5-coder:7b",
            },
        )

    payload, message, code = request_json(
        "http://api.test",
        "POST",
        "/chat",
        json_body={"message": "hello"},
        timeout=1030.0,
        transport=mock_transport(handler),
    )

    assert message == ""
    assert code is None
    assert payload["answer"] == "A real answer."


def test_read_timeout_reports_actionable_timeout_without_duplicate_retry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    payload, message, code = request_json(
        "http://api.test",
        "POST",
        "/chat",
        json_body={"message": "hello"},
        timeout=12.5,
        transport=mock_transport(handler),
        request_label="General chat",
        timeout_setting="CHAT_TIMEOUT_SECONDS",
    )

    assert payload is None
    assert code == "backend_timeout"
    assert "General chat" in message
    assert "12.5 seconds" in message
    assert "duplicate work" in message
    assert "CHAT_TIMEOUT_SECONDS" in message


def test_connect_error_is_reported_separately_from_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    payload, message, code = request_json(
        "http://api.test",
        "POST",
        "/chat",
        json_body={"message": "hello"},
        timeout=10.0,
        transport=mock_transport(handler),
    )

    assert payload is None
    assert code == "backend_unreachable"
    assert "Could not connect" in message
    assert "uvicorn backend.app:app" in message


def test_connect_timeout_is_reported_separately_from_read_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow connect", request=request)

    payload, message, code = request_json(
        "http://api.test",
        "GET",
        "/health",
        timeout=3.0,
        transport=mock_transport(handler),
    )

    assert payload is None
    assert code == "backend_connect_timeout"
    assert "connecting" in message


def test_backend_http_error_uses_detail_and_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            json={
                "detail": {
                    "message": "The model is still loading.",
                    "code": "model_unavailable",
                }
            },
        )

    payload, message, code = request_json(
        "http://api.test",
        "POST",
        "/chat",
        json_body={"message": "hello"},
        timeout=10.0,
        transport=mock_transport(handler),
    )

    assert payload is None
    assert message == "The model is still loading."
    assert code == "model_unavailable"


def test_backend_status_error_without_detail_keeps_http_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"not-json")

    payload, message, code = request_json(
        "http://api.test",
        "GET",
        "/health",
        timeout=3.0,
        transport=mock_transport(handler),
    )

    assert payload is None
    assert message == "Backend returned HTTP 503."
    assert code == "backend_http_503"
