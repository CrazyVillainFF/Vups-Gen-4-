from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings
from backend.provider import ChatProvider, ModelUnavailable


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, str, str]] = []
        self.available = True

    def save_turn(
        self, conversation_id: str, user_message: str, assistant_message: str, skill: str
    ) -> None:
        self.saved.append((conversation_id, user_message, assistant_message, skill))

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return []

    def ping(self) -> bool:
        return self.available


class FakeProvider:
    def __init__(self, answer: str = "A real answer", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.configured = True
        self.vision_supported = False

    async def generate(self, **kwargs: Any) -> str:
        if self.error:
            raise self.error
        return self.answer

    async def probe(self, *, live: bool = False) -> dict[str, Any]:
        return {
            "configured": True,
            "status": "available",
            "reachable": True,
            "model_available": True,
            "model": "qwen2.5-coder:7b",
            "base_url": "http://127.0.0.1:11434/v1",
            "endpoint": "http://127.0.0.1:11434/v1/chat/completions",
            "vision_supported": False,
            "live_checked": live,
            "live_ok": live,
            "detail": "",
        }


def configured_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_name": "Vups Test",
        "model_base_url": "http://127.0.0.1:11434/v1",
        "model_name": "qwen2.5-coder:7b",
    }
    values.update(overrides)
    return Settings(**values)


def make_client(
    provider: Any,
    store: FakeStore | None = None,
    config: Settings | None = None,
) -> tuple[TestClient, FakeStore]:
    store = store or FakeStore()
    app = create_app(config=config or configured_settings(), provider=provider, store=store)
    return TestClient(app), store


def test_health_reports_application_only() -> None:
    client, _ = make_client(FakeProvider())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_distinguishes_database_from_model_configuration() -> None:
    provider = ChatProvider(Settings(model_base_url="", model_name=""))
    client, _ = make_client(
        provider,
        config=Settings(app_name="Vups Test", model_base_url="", model_name=""),
    )

    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["database"] == "ok"
    assert body["model_configured"] is False
    assert body["model_status"] == "not_configured"


def test_model_info_reports_missing_configuration_without_secrets() -> None:
    config = Settings(
        app_name="Vups Test",
        model_base_url="",
        model_name="",
        model_api_key="super-secret",
    )
    client, _ = make_client(ChatProvider(config), config=config)

    response = client.get("/model-info")
    body = response.json()

    assert response.status_code == 200
    assert body["provider_configured"] is False
    assert body["missing_settings"] == ["MODEL_BASE_URL", "MODEL_NAME"]
    assert "super-secret" not in response.text


def test_chat_without_model_returns_503_and_does_not_save_turn() -> None:
    config = Settings(app_name="Vups Test", model_base_url="", model_name="")
    client, store = make_client(ChatProvider(config), config=config)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["code"] == "model_not_configured"
    assert "answer" not in body
    assert store.saved == []


def test_provider_failure_returns_502_and_no_fake_answer() -> None:
    client, store = make_client(FakeProvider(error=ModelUnavailable("Ollama is not running.")))

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502
    body = response.json()
    assert body["detail"]["code"] == "model_unavailable"
    assert "Ollama is not running." in body["detail"]["message"]
    assert "answer" not in body
    assert store.saved == []


def test_successful_chat_persists_the_real_answer() -> None:
    client, store = make_client(FakeProvider(answer="The real model answer."))

    response = client.post(
        "/chat",
        json={
            "message": "hello",
            "conversation_id": "conversation-1",
            "skill": "coding",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The real model answer."
    assert body["model"] == "qwen2.5-coder:7b"
    assert store.saved == [
        ("conversation-1", "hello", "The real model answer.", "coding")
    ]


def test_model_status_endpoint_returns_probe_result() -> None:
    client, _ = make_client(FakeProvider())

    response = client.get("/model-status")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "available"
    assert body["live_checked"] is False


def test_message_length_limit_is_enforced() -> None:
    client, store = make_client(
        FakeProvider(),
        config=configured_settings(max_message_chars=5),
    )

    response = client.post("/chat", json={"message": "too long"})

    assert response.status_code == 413
    assert store.saved == []
