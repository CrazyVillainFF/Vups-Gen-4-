from __future__ import annotations

import io
import os
import subprocess
import time
import zipfile
from pathlib import Path

import pytest

from backend.workspace import (
    ApprovalRequired,
    FileTooLarge,
    InvalidPath,
    InvalidProposal,
    InvalidWorkspaceName,
    LimitExceeded,
    OverwriteApprovalRequired,
    Proposal,
    ProposalExpired,
    ProposalNotFound,
    ProposalStore,
    SensitivePath,
    WorkspaceError,
    WorkspaceExists,
    WorkspaceService,
)


def make_service(tmp_path: Path, **kwargs: int) -> WorkspaceService:
    return WorkspaceService(tmp_path / "workspaces", **kwargs)


def create_project(service: WorkspaceService, name: str = "demo") -> Path:
    service.create_workspace(name)
    return service.root / name


def test_create_list_and_open_workspace(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    created = service.create_workspace("Demo-App")

    assert created["name"] == "Demo-App"
    assert service.list_workspaces()[0]["name"] == "Demo-App"
    assert service.get_workspace("Demo-App")["file_count"] == 0

    with pytest.raises(WorkspaceExists):
        service.create_workspace("Demo-App")
    with pytest.raises(InvalidWorkspaceName):
        service.create_workspace("../escape")
    with pytest.raises(InvalidWorkspaceName):
        service.create_workspace("CON")


def test_save_list_and_read_text_file(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    create_project(service)

    saved = service.save_file("demo", "src/app.py", "print('hello')\n")
    entries = service.list_files("demo")
    content = service.read_file("demo", "src/app.py")

    assert saved["saved"] is True
    assert entries == [{"path": "src/app.py", "size": 15, "language": "python"}]
    assert content["content"] == "print('hello')\n"
    assert content["language"] == "python"


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "src/../../outside.txt",
        "/etc/passwd",
        "C:/Windows/win.ini",
        r"C:\Windows\win.ini",
        r"\\server\share\file.txt",
        "~/.ssh/id_rsa",
        "src/\x00app.py",
        "src/CON.txt",
        "src/aux.py",
        "src/file.txt:stream",
        "src/bad?name.py",
        "src/trailing.",
        "src/trailing /x.py",
    ],
)
def test_traversal_absolute_and_invalid_paths_are_rejected(
    tmp_path: Path, path: str
) -> None:
    service = make_service(tmp_path)
    create_project(service)

    with pytest.raises(WorkspaceError):
        service.read_file("demo", path)
    assert not (tmp_path / "outside.txt").exists()


def test_save_rejects_outside_target(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    project = create_project(service)
    (project / "src").mkdir()

    with pytest.raises(WorkspaceError):
        service.save_file("demo", "../escape.txt", "bad")
    assert not (tmp_path / "escape.txt").exists()
    with pytest.raises(InvalidPath):
        service.save_file("demo", "src", "not a file")
    with pytest.raises(InvalidPath):
        service.build_proposal(
            "demo", {"files": [{"path": "src", "content": "not a file"}]}, "test"
        )


def test_link_and_junction_escapes_are_rejected(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    project = create_project(service)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("private", encoding="utf-8")
    link = project / "linked"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        if os.name != "nt":
            pytest.skip("Directory links are not available in this environment")
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("Could not create a directory junction")

    with pytest.raises(InvalidPath):
        service.read_file("demo", "linked/secret.txt")
    with pytest.raises(InvalidPath):
        service.save_file("demo", "linked/new.txt", "bad")
    assert service.list_files("demo") == []
    data, _ = service.export_zip("demo")
    assert zipfile.ZipFile(io.BytesIO(data)).namelist() == []


def test_secret_files_are_hidden_from_listing_context_and_export(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    project = create_project(service)
    service.save_file("demo", "src/main.py", "print('ok')\n")
    secrets = {
        ".env": "TOKEN=secret\n",
        "prod.env": "TOKEN=secret\n",
        ".git/config": "[remote]\n",
        ".venv/lib.py": "hidden\n",
        "node_modules/package/index.js": "hidden\n",
        "credentials.json": "{}\n",
        "app.db": "not really a database\n",
        "private.pem": "key\n",
        "server.log": "secret log\n",
    }
    for relative, content in secrets.items():
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    paths = {entry["path"] for entry in service.list_files("demo")}
    assert paths == {"src/main.py"}

    context, used, warnings = service.context_for_files(
        "demo", [".env", "credentials.json", "src/main.py"]
    )
    assert used == ["src/main.py"]
    assert "TOKEN=secret" not in context
    assert any("sensitive" in warning.lower() for warning in warnings)

    with pytest.raises(SensitivePath):
        service.read_file("demo", ".env")
    with pytest.raises(SensitivePath):
        service.save_file("demo", ".env", "TOKEN=new\n")

    data, filename = service.export_zip("demo")
    names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
    assert filename == "demo.zip"
    assert names == {"src/main.py"}


def test_zip_export_avoids_unsafe_archive_paths(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    project = create_project(service)
    service.save_file("demo", "src/main.py", "print('ok')\n")
    (project / "docs").mkdir()
    (project / "docs" / "README.md").write_text("# docs\n", encoding="utf-8")

    data, _ = service.export_zip("demo")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        assert "src/main.py" in names
        assert "docs/README.md" in names
        assert all(not name.startswith("/") and ".." not in name.split("/") for name in names)
        assert b"Review" in archive.comment


def test_proposal_limits_and_invalid_structures(tmp_path: Path) -> None:
    service = make_service(
        tmp_path,
        max_files=2,
        max_file_bytes=10,
        max_total_bytes=15,
    )
    create_project(service)

    with pytest.raises(LimitExceeded):
        service.build_proposal(
            "demo",
            {
                "files": [
                    {"path": "a.txt", "content": "a"},
                    {"path": "b.txt", "content": "b"},
                    {"path": "c.txt", "content": "c"},
                ]
            },
            "test",
        )
    with pytest.raises(LimitExceeded):
        service.build_proposal(
            "demo",
            {"files": [{"path": "a.txt", "content": "x" * 11}]},
            "test",
        )
    with pytest.raises(LimitExceeded):
        service.build_proposal(
            "demo",
            {
                "files": [
                    {"path": "a.txt", "content": "x" * 8},
                    {"path": "b.txt", "content": "y" * 8},
                ]
            },
            "test",
        )
    with pytest.raises(InvalidProposal):
        service.build_proposal("demo", {"files": []}, "test")
    with pytest.raises(InvalidProposal):
        service.build_proposal("demo", {"files": ["bad"]}, "test")
    with pytest.raises(InvalidProposal):
        service.build_proposal(
            "demo",
            {"files": [{"path": "a.txt", "content": "a"}, {"path": "A.TXT", "content": "b"}]},
            "test",
        )
    with pytest.raises(InvalidPath):
        service.build_proposal(
            "demo",
            {"files": [{"path": "../escape.txt", "content": "a"}]},
            "test",
        )


def test_proposal_requires_approval_and_overwrite_approval(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    create_project(service)
    service.save_file("demo", "main.py", "old\n")
    update = service.build_proposal(
        "demo",
        {"summary": "update", "files": [{"path": "main.py", "content": "new\n"}]},
        "test",
    )
    assert update.files[0].action == "update"
    assert update.requires_overwrite_approval is True

    with pytest.raises(ApprovalRequired):
        service.apply_proposal(
            "demo", update, approved=False, approve_overwrites=False
        )
    assert service.read_file("demo", "main.py")["content"] == "old\n"

    with pytest.raises(OverwriteApprovalRequired):
        service.apply_proposal(
            "demo", update, approved=True, approve_overwrites=False
        )
    assert service.read_file("demo", "main.py")["content"] == "old\n"

    result = service.apply_proposal(
        "demo", update, approved=True, approve_overwrites=True
    )
    assert result["modified"] == ["main.py"]
    assert service.read_file("demo", "main.py")["content"] == "new\n"

    create = service.build_proposal(
        "demo",
        {"files": [{"path": "src/new.py", "content": "print(1)\n"}]},
        "test",
    )
    result = service.apply_proposal(
        "demo", create, approved=True, approve_overwrites=False
    )
    assert result["created"] == ["src/new.py"]
    assert service.read_file("demo", "src/new.py")["content"] == "print(1)\n"


def test_atomic_write_failure_preserves_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make_service(tmp_path)
    project = create_project(service)
    service.save_file("demo", "main.py", "original\n")

    def fail_replace(source: str, destination: str) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("backend.workspace.os.replace", fail_replace)

    with pytest.raises(OSError):
        service.save_file("demo", "main.py", "replacement\n")

    assert (project / "main.py").read_text(encoding="utf-8") == "original\n"
    assert list(project.glob(".vups-*.tmp")) == []


def test_workspace_limits_apply_to_file_saves(tmp_path: Path) -> None:
    service = make_service(tmp_path, max_file_bytes=5)
    create_project(service)

    with pytest.raises(FileTooLarge):
        service.save_file("demo", "large.txt", "123456")


def test_total_workspace_size_limit_applies_to_saves_and_apply(tmp_path: Path) -> None:
    service = make_service(tmp_path, max_file_bytes=10, max_total_bytes=12)
    create_project(service)
    service.save_file("demo", "a.txt", "123456")

    with pytest.raises(LimitExceeded):
        service.save_file("demo", "b.txt", "7890123")

    proposal = service.build_proposal(
        "demo",
        {"files": [{"path": "b.txt", "content": "1234567"}]},
        "test",
    )
    with pytest.raises(LimitExceeded):
        service.apply_proposal(
            "demo", proposal, approved=True, approve_overwrites=False
        )
    assert service.list_files("demo") == [{"path": "a.txt", "size": 6, "language": "text"}]


def test_proposal_store_distinguishes_missing_and_expired() -> None:
    store = ProposalStore(ttl_seconds=300, max_items=2)
    active = Proposal(
        id="active",
        workspace="demo",
        summary="active",
        files=(),
        model="test",
        requires_overwrite_approval=False,
        warnings=(),
        created_at=time.time(),
    )
    expired = Proposal(
        id="expired",
        workspace="demo",
        summary="expired",
        files=(),
        model="test",
        requires_overwrite_approval=False,
        warnings=(),
        created_at=time.time() - 120,
    )
    store.add(active)
    store.add(expired)
    store.ttl_seconds = 30

    assert store.get("active", "demo") is active
    with pytest.raises(ProposalNotFound):
        store.get("missing", "demo")
    with pytest.raises(ProposalNotFound):
        store.get("active", "other")
    with pytest.raises(ProposalExpired):
        store.get("expired", "demo")
    with pytest.raises(ProposalNotFound):
        store.get("expired", "demo")
