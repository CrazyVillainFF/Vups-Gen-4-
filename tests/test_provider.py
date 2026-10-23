from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from backend.config import Settings
from backend.provider import ChatProvider, ModelNotConfigured, ModelUnavailable


def build_config(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "model_base_url": "http://ollama.test/v1",
        "model_name": "qwen2.5-coder:7b",
        "model_api_key": "ollama",
        "model_timeout_seconds": 5.0,
    }
    values.update(overrides)
    return Settings(**values)


def run_generate(provider: ChatProvider, **kwargs: object) -> str:
    options: dict[str, object] = {
        "system_prompt": "system",
        "message": "hello",
        "history": [],
    }
    options.update(kwargs)
    return asyncio.run(provider.generate(**options))


def test_unconfigured_provider_does_not_invent_an_answer() -> None:
    provider = ChatProvider(Settings(model_base_url="", model_name=""))
    with pytest.raises(ModelNotConfigured) as caught:
        run_generate(provider)
    assert "not configured" in str(caught.value)
    assert "no model endpoint is configured" not in str(caught.value)


def test_valid_chat_completion_request_and_response() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["payload"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Hello"}}]},
        )

    provider = ChatProvider(build_config(), transport=httpx.MockTransport(handler))
    answer = run_generate(
        provider,
        history=[
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "reply"},
        ],
    )
    payload = json.loads(str(seen["payload"]))

    assert answer == "Hello"
    assert str(seen["url"]) == "http://ollama.test/v1/chat/completions"
    assert seen["authorization"] == "Bearer ollama"
    assert payload["model"] == "qwen2.5-coder:7b"
    assert payload["stream"] is False
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_text_model_rejects_image_before_http_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("A text-only model must not receive an image request")

    provider = ChatProvider(build_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable, match="does not support image"):
        run_generate(provider, image_data_url="data:image/png;base64,AAAA")


def test_vision_support_can_be_enabled_explicitly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Image read"}}]},
        )

    provider = ChatProvider(
        build_config(model_supports_vision=True),
        transport=httpx.MockTransport(handler),
    )

    assert asyncio.run(provider.generate(
        system_prompt="system",
        message="what is this?",
        history=[],
        image_data_url="data:image/png;base64,AAAA",
    )) == "Image read"


def test_vision_model_receives_image_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.read()))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "A cat"}}]},
        )

    provider = ChatProvider(
        build_config(model_name="qwen2.5-vl:7b"),
        transport=httpx.MockTransport(handler),
    )
    answer = run_generate(provider, image_data_url="data:image/png;base64,AAAA")

    content = captured["messages"][-1]["content"]
    assert answer == "A cat"
    assert content[0] == {"type": "text", "text": "hello"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_connection_error_is_safe_and_actionable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = ChatProvider(build_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable) as caught:
        run_generate(provider)
    message = str(caught.value)
    assert "Could not connect" in message
    assert "ollama serve" in message


def test_timeout_is_reported_as_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    provider = ChatProvider(build_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable, match="timed out"):
        run_generate(provider)


def test_missing_ollama_model_suggests_pull() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": "model 'qwen2.5-coder:7b' not found, try pulling it first"},
        )

    provider = ChatProvider(build_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable) as caught:
        run_generate(provider)
    assert "ollama pull qwen2.5-coder:7b" in str(caught.value)


def test_authentication_error_never_contains_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    provider = ChatProvider(
        build_config(model_api_key="super-secret"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ModelUnavailable) as caught:
        run_generate(provider)
    message = str(caught.value)
    assert "Authentication" in message or "authentication" in message
    assert "super-secret" not in message


def test_invalid_json_response_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    provider = ChatProvider(build_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable, match="invalid JSON"):
        run_generate(provider)


def test_empty_choices_response_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    provider = ChatProvider(build_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable, match="missing choices"):
        run_generate(provider)


def test_empty_assistant_content_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "  "}}]},
        )

    provider = ChatProvider(build_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(ModelUnavailable, match="empty or malformed"):
        run_generate(provider)


def test_probe_reports_listed_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://ollama.test/v1/models"
        return httpx.Response(200, json={"data": [{"id": "qwen2.5-coder:7b"}]})

    provider = ChatProvider(build_config(), transport=httpx.MockTransport(handler))
    status = asyncio.run(provider.probe())

    assert status["status"] == "available"
    assert status["reachable"] is True
    assert status["model_available"] is True
    assert status["live_checked"] is False


def test_probe_reports_missing_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "llama3.2:3b"}]})

    provider = ChatProvider(build_config(), transport=httpx.MockTransport(handler))
    status = asyncio.run(provider.probe())

    assert status["status"] == "model_missing"
    assert status["model_available"] is False
    assert "ollama pull qwen2.5-coder:7b" in status["detail"]


def test_probe_reports_unreachable_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    provider = ChatProvider(build_config(), transport=httpx.MockTransport(handler))
    status = asyncio.run(provider.probe())

    assert status["status"] == "unreachable"
    assert status["reachable"] is False
