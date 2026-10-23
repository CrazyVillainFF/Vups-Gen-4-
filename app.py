"""ASGI entrypoint used by Vercel and by local ``uvicorn app:app`` runs.

Vercel mounts the deployment under a read-only directory. ``create_app()`` runs
at import time and initializes SQLite plus the workspace root, so the default
project-relative paths cannot be created there and every request fails with
FUNCTION_INVOCATION_FAILED. Point those defaults at writable temporary storage
when running on Vercel. Explicit paths (including Turso) are always respected.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

if os.environ.get("VERCEL"):
    runtime_root = Path(tempfile.gettempdir()) / "vups-gen4"
    os.environ.setdefault(
        "DATABASE_URL", f"sqlite:///{(runtime_root / 'vups_gen4.db').as_posix()}"
    )
    os.environ.setdefault("WORKSPACE_ROOT", str(runtime_root / "workspaces"))
    os.environ.setdefault("IMAGE_OUTPUT_DIR", str(runtime_root / "generated_images"))

from backend.app import app  # noqa: E402

__all__ = ["app"]
