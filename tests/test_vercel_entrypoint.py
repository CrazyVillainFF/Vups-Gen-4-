from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import json
import os
import sys

import app  # noqa: F401

print(json.dumps({
    "database_url": os.environ.get("DATABASE_URL", ""),
    "workspace_root": os.environ.get("WORKSPACE_ROOT", ""),
    "image_output_dir": os.environ.get("IMAGE_OUTPUT_DIR", ""),
    "streamlit_loaded": "streamlit" in sys.modules,
    "routes": len(app.app.routes),
}))
"""


def _clean_environment(tmp_path: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "DATABASE_URL",
            "WORKSPACE_ROOT",
            "IMAGE_OUTPUT_DIR",
            "TURSO_DATABASE_URL",
            "TURSO_AUTH_TOKEN",
            "VERCEL",
            "VERCEL_ENV",
        }
    }
    env.update(
        {
            "VERCEL": "1",
            "VERCEL_ENV": "production",
            "DATABASE_BACKEND": "sqlite",
            "TURSO_DATABASE_URL": "",
            "TURSO_AUTH_TOKEN": "",
            "TEMP": str(tmp_path),
            "TMP": str(tmp_path),
            "TMPDIR": str(tmp_path),
        }
    )
    return env


def _import_entrypoint(env: dict[str, str], cwd: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_vercel_entrypoint_uses_writable_runtime_paths(tmp_path: Path) -> None:
    result = _import_entrypoint(_clean_environment(tmp_path), PROJECT_ROOT)

    runtime_root = tmp_path / "vups-gen4"
    assert result["database_url"] == f"sqlite:///{(runtime_root / 'vups_gen4.db').as_posix()}"
    assert result["workspace_root"] == str(runtime_root / "workspaces")
    assert result["image_output_dir"] == str(runtime_root / "generated_images")
    assert result["streamlit_loaded"] is False
    assert int(result["routes"]) > 0

    # The writable runtime paths are actually used during import.
    assert (runtime_root / "vups_gen4.db").is_file()
    assert (runtime_root / "workspaces").is_dir()


def test_vercel_entrypoint_keeps_explicit_storage_paths(tmp_path: Path) -> None:
    env = _clean_environment(tmp_path)
    explicit_database = f"sqlite:///{(tmp_path / 'explicit.db').as_posix()}"
    explicit_workspaces = str(tmp_path / "explicit-workspaces")
    explicit_images = str(tmp_path / "explicit-images")
    env.update(
        {
            "DATABASE_URL": explicit_database,
            "WORKSPACE_ROOT": explicit_workspaces,
            "IMAGE_OUTPUT_DIR": explicit_images,
        }
    )

    result = _import_entrypoint(env, PROJECT_ROOT)

    assert result["database_url"] == explicit_database
    assert result["workspace_root"] == explicit_workspaces
    assert result["image_output_dir"] == explicit_images


def test_local_entrypoint_keeps_project_relative_defaults(tmp_path: Path) -> None:
    env = _clean_environment(tmp_path)
    env.pop("VERCEL", None)
    env.pop("VERCEL_ENV", None)

    result = _import_entrypoint(env, PROJECT_ROOT)

    # Local development must keep the project's configured defaults instead of
    # being redirected to Vercel's temporary runtime root.
    assert str(tmp_path) not in str(result["database_url"])
    assert result["workspace_root"] == ""
    assert result["image_output_dir"] == ""
    assert result["streamlit_loaded"] is False
    assert (PROJECT_ROOT / "workspaces").is_dir()
