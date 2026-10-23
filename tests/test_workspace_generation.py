from __future__ import annotations

import json
from asyncio import run
from types import SimpleNamespace
from typing import Any

import pytest

from backend.workspace import WorkspaceNotFound, WorkspaceService
from backend.workspace_generation import ModelOutputError, ProjectGenerator


class ScriptedProvider:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.config = SimpleNamespace(model_name="scripted-model")
        self.configured = True
        self.vision_supported = False

    async def generate(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("The provider was called more times than expected")
        return self.responses.pop(0)


def make_generator(
    tmp_path, provider: ScriptedProvider, **kwargs: int
) -> tuple[ProjectGenerator, WorkspaceService]:
    service = WorkspaceService(tmp_path / "workspaces", **kwargs)
    service.create_workspace("demo")
    return ProjectGenerator(provider, service), service


def valid_output(path: str = "src/main.py", content: str = "print('hi')\n") -> str:
    return json.dumps(
        {
            "summary": "Create a small program",
            "files": [{"path": path, "content": content}],
        }
    )


def test_propose_parses_project_and_does_not_write(tmp_path) -> None:
    provider = ScriptedProvider(valid_output())
    generator, service = make_generator(tmp_path, provider)

    proposal = run(generator.propose("demo", "Build a hello-world app"))

    assert proposal.summary == "Create a small program"
    assert proposal.model == "scripted-model"
    assert proposal.files[0].path == "src/main.py"
    assert proposal.files[0].action == "create"
    assert proposal.requires_overwrite_approval is False
    assert service.list_files("demo") == []
    assert len(provider.calls) == 1
    assert provider.calls[0]["max_tokens"] == 4_096
    assert "Return exactly one JSON object" in provider.calls[0]["system_prompt"]
    assert "Build a hello-world app" in provider.calls[0]["message"]
    assert "(empty project)" in provider.calls[0]["message"]


def test_propose_includes_selected_context_and_warnings(tmp_path) -> None:
    provider = ScriptedProvider(valid_output())
    generator, service = make_generator(tmp_path, provider)
    service.save_file("demo", "src/app.py", "existing = True\n")
    (service.root / "demo" / ".env").write_text("TOKEN=secret\n", encoding="utf-8")

    proposal = run(
        generator.propose(
            "demo", "Change the app", context_files=["src/app.py", ".env"]
        )
    )

    message = provider.calls[0]["message"]
    assert "existing = True" in message
    assert "TOKEN=secret" not in message
    assert "src/app.py" in message
    assert any("sensitive" in warning.lower() for warning in proposal.warnings)


def test_invalid_json_triggers_one_bounded_repair(tmp_path) -> None:
    provider = ScriptedProvider("This is not JSON at all.", valid_output())
    generator, _ = make_generator(tmp_path, provider)

    proposal = run(generator.propose("demo", "Build it"))

    assert proposal.files[0].path == "src/main.py"
    assert len(provider.calls) == 2
    assert "previous response" in provider.calls[1]["message"].lower()
    assert "invalid json" in provider.calls[1]["message"].lower()


def test_invalid_structure_triggers_repair(tmp_path) -> None:
    invalid = json.dumps({"summary": "bad", "files": []})
    provider = ScriptedProvider(invalid, valid_output())
    generator, _ = make_generator(tmp_path, provider)

    proposal = run(generator.propose("demo", "Build it"))

    assert proposal.files[0].path == "src/main.py"
    assert len(provider.calls) == 2
    assert "non-empty files list" in provider.calls[1]["message"]


def test_markdown_fenced_json_is_accepted(tmp_path) -> None:
    fenced = f"```json\n{valid_output()}\n```"
    provider = ScriptedProvider(fenced)
    generator, _ = make_generator(tmp_path, provider)

    proposal = run(generator.propose("demo", "Build it"))

    assert proposal.files[0].content == "print('hi')\n"


def test_two_invalid_outputs_raise_and_write_nothing(tmp_path) -> None:
    provider = ScriptedProvider("not json", "still not json")
    generator, service = make_generator(tmp_path, provider)

    with pytest.raises(ModelOutputError) as caught:
        run(generator.propose("demo", "Build it"))

    assert "No files were written" in str(caught.value)
    assert len(provider.calls) == 2
    assert service.list_files("demo") == []


def test_unsafe_paths_are_repaired_not_written(tmp_path) -> None:
    unsafe = json.dumps(
        {"files": [{"path": "../escape.py", "content": "print(1)\n"}]}
    )
    provider = ScriptedProvider(unsafe, valid_output())
    generator, service = make_generator(tmp_path, provider)

    proposal = run(generator.propose("demo", "Build it"))

    assert proposal.files[0].path == "src/main.py"
    assert len(provider.calls) == 2
    assert service.list_files("demo") == []
    assert not (tmp_path / "escape.py").exists()


def test_ask_uses_context_and_returns_clean_answer(tmp_path) -> None:
    provider = ScriptedProvider("  It looks correct.  ")
    generator, service = make_generator(tmp_path, provider)
    service.save_file("demo", "main.py", "print('hi')\n")

    answer, used, warnings = run(
        generator.ask("demo", "Is this correct?", context_files=["main.py"])
    )

    assert answer == "It looks correct."
    assert used == ["main.py"]
    assert warnings == []
    assert "print('hi')" in provider.calls[0]["message"]
    assert "Do not claim to have edited" in provider.calls[0]["system_prompt"]
    assert provider.calls[0]["max_tokens"] == 2_048


def test_ask_rejects_missing_workspace(tmp_path) -> None:
    provider = ScriptedProvider("unused")
    generator, _ = make_generator(tmp_path, provider)

    with pytest.raises(WorkspaceNotFound):
        run(generator.ask("missing", "Question"))

    assert provider.calls == []


def test_sensitive_context_is_skipped_before_model_call(tmp_path) -> None:
    provider = ScriptedProvider(valid_output())
    generator, service = make_generator(tmp_path, provider)
    (service.root / "demo" / ".env").write_text("TOKEN=secret\n", encoding="utf-8")

    proposal = run(generator.propose("demo", "Build it", context_files=[".env"]))

    assert provider.calls
    assert "TOKEN=secret" not in provider.calls[0]["message"]
    assert any("sensitive" in warning.lower() for warning in proposal.warnings)
