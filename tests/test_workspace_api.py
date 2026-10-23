from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings
from backend.provider import ModelNotConfigured
from backend.workspace import WorkspaceService


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, str, str]] = []

    def save_turn(
        self, conversation_id: str, user_message: str, assistant_message: str, skill: str
    ) -> None:
        self.saved.append((conversation_id, user_message, assistant_message, skill))

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return []

    def ping(self) -> bool:
        return True


class ScriptedProvider:
    def __init__(
        self, *responses: str, configured: bool = True
    ) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.config = SimpleNamespace(model_name="fake-model")
        self.configured = configured
        self.vision_supported = False

    async def generate(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if not self.configured:
            raise ModelNotConfigured("Configure MODEL_BASE_URL and MODEL_NAME.")
        if not self.responses:
            raise AssertionError("The provider was called more times than expected")
        return self.responses.pop(0)

    async def probe(self, *, live: bool = False) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "status": "available" if self.configured else "not_configured",
            "reachable": self.configured,
            "model_available": self.configured,
            "model": "fake-model",
            "base_url": "http://127.0.0.1:11434/v1",
            "endpoint": "http://127.0.0.1:11434/v1/chat/completions",
            "vision_supported": False,
            "live_checked": live,
            "live_ok": live and self.configured,
            "detail": "",
        }


def proposal_json(path: str = "src/app.py", content: str = "print('hi')\n") -> str:
    return json.dumps(
        {
            "summary": "Create an app",
            "files": [{"path": path, "content": content}],
        }
    )


def make_client(
    tmp_path,
    *responses: str,
    configured: bool = True,
    **settings_overrides: Any,
) -> tuple[TestClient, ScriptedProvider, WorkspaceService]:
    config = Settings(
        app_name="Vups Workspace Test",
        model_base_url="http://127.0.0.1:11434/v1",
        model_name="fake-model",
        workspace_root=tmp_path / "workspaces",
        **settings_overrides,
    )
    provider = ScriptedProvider(*responses, configured=configured)
    service = WorkspaceService.from_settings(config)
    app = create_app(
        config=config,
        provider=provider,
        store=FakeStore(),
        workspaces=service,
    )
    return TestClient(app), provider, service


def create_workspace(client: TestClient, name: str = "demo") -> None:
    response = client.post("/workspaces", json={"name": name})
    assert response.status_code == 201, response.text


def test_workspace_and_file_crud_api(tmp_path) -> None:
    client, _, _ = make_client(tmp_path)

    created = client.post("/workspaces", json={"name": "demo"})
    listed = client.get("/workspaces")
    opened = client.get("/workspaces/demo")
    saved = client.put(
        "/workspaces/demo/file",
        json={"path": "src/main.py", "content": "print('hello')\n"},
    )
    files = client.get("/workspaces/demo/files")
    content = client.get("/workspaces/demo/file", params={"path": "src/main.py"})

    assert created.status_code == 201
    assert created.json()["name"] == "demo"
    assert listed.json()["workspaces"][0]["name"] == "demo"
    assert opened.status_code == 200
    assert saved.status_code == 200
    assert saved.json()["language"] == "python"
    assert files.json() == [{"path": "src/main.py", "size": 15, "language": "python"}]
    assert content.json()["content"] == "print('hello')\n"
    assert client.post("/workspaces", json={"name": "demo"}).status_code == 409
    assert client.post("/workspaces", json={"name": "../bad"}).status_code == 400
    assert client.get("/workspaces/missing").status_code == 404


def test_file_api_rejects_traversal_secrets_and_missing_files(tmp_path) -> None:
    client, _, service = make_client(tmp_path)
    create_workspace(client)
    (service.root / "demo" / "binary.bin").write_bytes(b"\x00\xff")

    assert (
        client.put(
            "/workspaces/demo/file",
            json={"path": ".env", "content": "TOKEN=secret\n"},
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/workspaces/demo/file", params={"path": "../outside.txt"}
        ).status_code
        == 400
    )
    assert (
        client.get(
            "/workspaces/demo/file", params={"path": "missing.txt"}
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/workspaces/demo/file", params={"path": "binary.bin"}
        ).status_code
        == 415
    )


def test_proposal_preview_approval_and_apply_flow(tmp_path) -> None:
    client, provider, _ = make_client(tmp_path, proposal_json())
    create_workspace(client)

    proposed = client.post(
        "/workspaces/demo/proposals",
        json={"message": "Build a small app"},
    )

    assert proposed.status_code == 200, proposed.text
    body = proposed.json()
    assert body["created_count"] == 1
    assert body["updated_count"] == 0
    assert body["requires_overwrite_approval"] is False
    assert body["model"] == "fake-model"
    assert body["files"][0]["path"] == "src/app.py"
    assert client.get("/workspaces/demo/files").json() == []

    proposal_id = body["proposal_id"]
    refused = client.post(
        f"/workspaces/demo/proposals/{proposal_id}/apply",
        json={"approved": False},
    )
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "approval_required"
    assert client.get("/workspaces/demo/files").json() == []

    applied = client.post(
        f"/workspaces/demo/proposals/{proposal_id}/apply",
        json={"approved": True, "approve_overwrites": False},
    )
    assert applied.status_code == 200, applied.text
    assert applied.json()["created"] == ["src/app.py"]
    assert client.get("/workspaces/demo/files").json()[0]["path"] == "src/app.py"
    assert len(provider.calls) == 1

    reused = client.post(
        f"/workspaces/demo/proposals/{proposal_id}/apply",
        json={"approved": True},
    )
    assert reused.status_code == 404


def test_existing_files_require_separate_overwrite_approval(tmp_path) -> None:
    client, _, _ = make_client(
        tmp_path, proposal_json("main.py", "new content\n")
    )
    create_workspace(client)
    client.put(
        "/workspaces/demo/file",
        json={"path": "main.py", "content": "old content\n"},
    )

    proposed = client.post(
        "/workspaces/demo/proposals", json={"message": "Update main.py"}
    )
    body = proposed.json()
    assert body["requires_overwrite_approval"] is True
    assert body["updated_count"] == 1

    refused = client.post(
        f"/workspaces/demo/proposals/{body['proposal_id']}/apply",
        json={"approved": True, "approve_overwrites": False},
    )
    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "overwrite_approval_required"
    assert (
        client.get("/workspaces/demo/file", params={"path": "main.py"}).json()[
            "content"
        ]
        == "old content\n"
    )

    applied = client.post(
        f"/workspaces/demo/proposals/{body['proposal_id']}/apply",
        json={"approved": True, "approve_overwrites": True},
    )
    assert applied.status_code == 200
    assert applied.json()["modified"] == ["main.py"]
    assert (
        client.get("/workspaces/demo/file", params={"path": "main.py"}).json()[
            "content"
        ]
        == "new content\n"
    )


def test_workspace_zip_export_excludes_secrets(tmp_path) -> None:
    client, _, service = make_client(tmp_path)
    create_workspace(client)
    service.save_file("demo", "src/main.py", "print('ok')\n")
    secret = service.root / "demo" / ".env"
    secret.write_text("TOKEN=secret\n", encoding="utf-8")

    response = client.get("/workspaces/demo/export")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "demo.zip" in response.headers["content-disposition"]
    assert "Review generated code" in response.headers["x-vups-warning"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == ["src/main.py"]
        assert b"Review" in archive.comment


def test_workspace_ask_returns_answer_and_uses_context(tmp_path) -> None:
    client, provider, service = make_client(tmp_path, "  Looks good.  ")
    create_workspace(client)
    service.save_file("demo", "main.py", "print('hi')\n")

    response = client.post(
        "/workspaces/demo/ask",
        json={"message": "Is this correct?", "context_files": ["main.py"]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "Looks good."
    assert body["context_files"] == ["main.py"]
    assert "print('hi')" in provider.calls[0]["message"]
    assert provider.calls[0]["max_tokens"] == 2_048


def test_propose_returns_503_when_model_is_not_configured(tmp_path) -> None:
    client, _, _ = make_client(tmp_path, configured=False)
    create_workspace(client)

    response = client.post(
        "/workspaces/demo/proposals", json={"message": "Build it"}
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "model_not_configured"


def test_invalid_model_output_returns_502_without_writing(tmp_path) -> None:
    client, provider, _ = make_client(tmp_path, "not json", "still not json")
    create_workspace(client)

    response = client.post(
        "/workspaces/demo/proposals", json={"message": "Build it"}
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "invalid_model_output"
    assert len(provider.calls) == 2
    assert client.get("/workspaces/demo/files").json() == []


def test_workspace_message_length_limit(tmp_path) -> None:
    client, _, _ = make_client(tmp_path, max_message_chars=5)
    create_workspace(client)

    response = client.post(
        "/workspaces/demo/proposals", json={"message": "123456"}
    )

    assert response.status_code == 413
    assert (
        client.post(
            "/workspaces/demo/proposals", json={"message": "   "}
        ).status_code
        == 422
    )
