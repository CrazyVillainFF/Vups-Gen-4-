from __future__ import annotations

import io
import os
import re
import stat
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "site-packages",
    ".streamlit",
    ".vscode",
    ".idea",
    ".cache",
    "dist",
    "build",
}
_SENSITIVE_FILE = re.compile(
    r"^(\.env(\..*)?|"
    r"id_(rsa|dsa|ecdsa|ed25519)(\.pub)?|"
    r"\.npmrc|\.pypirc|credentials(\..*)?|secrets(\..*)?|"
    r"service-account.*\.json|"
    r".+\.(pem|key|p12|pfx|kdbx|jks|keystore|db|db-wal|db-shm|sqlite|sqlite3|log)|"
    r".*\.env(\..*)?)$",
    re.IGNORECASE,
)
_WORKSPACE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    "conin$",
    "conout$",
    "clock$",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".json": "json",
    ".jsonc": "json",
    ".md": "markdown",
    ".markdown": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".ps1": "powershell",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".xml": "xml",
    ".ini": "ini",
    ".cfg": "ini",
    ".txt": "text",
    ".csv": "text",
    ".env": "dotenv",
}
_SPECIAL_LANGUAGES = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "procfile": "text",
    "gemfile": "ruby",
    "requirements.txt": "text",
}


class WorkspaceError(RuntimeError):
    code = "workspace_error"


class WorkspaceNotFound(WorkspaceError):
    code = "workspace_not_found"


class WorkspaceExists(WorkspaceError):
    code = "workspace_exists"


class InvalidWorkspaceName(WorkspaceError):
    code = "invalid_workspace_name"


class InvalidPath(WorkspaceError):
    code = "invalid_path"


class SensitivePath(WorkspaceError):
    code = "sensitive_path"


class FileTooLarge(WorkspaceError):
    code = "file_too_large"


class BinaryFile(WorkspaceError):
    code = "binary_file"


class LimitExceeded(WorkspaceError):
    code = "limit_exceeded"


class InvalidProposal(WorkspaceError):
    code = "invalid_proposal"


class ApprovalRequired(WorkspaceError):
    code = "approval_required"


class OverwriteApprovalRequired(WorkspaceError):
    code = "overwrite_approval_required"


class ProposalNotFound(WorkspaceError):
    code = "proposal_not_found"


class ProposalExpired(WorkspaceError):
    code = "proposal_expired"


def language_for_path(path: str) -> str:
    name = PurePosixPath(path).name.lower()
    if name in _SPECIAL_LANGUAGES:
        return _SPECIAL_LANGUAGES[name]
    return _LANGUAGES.get(PurePosixPath(name).suffix.lower(), "text")


def _is_link_like(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (OSError, AttributeError):
        return False
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_point)


def is_sensitive_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if not parts:
        return True
    if any(part.lower() in _EXCLUDED_DIRS for part in parts[:-1]):
        return True
    return bool(_SENSITIVE_FILE.match(parts[-1]))


def normalize_relative_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidPath("A file path is required.")
    text = raw.strip()
    if "\x00" in text:
        raise InvalidPath("The file path contains a null byte.")
    if "\\" in text:
        raise InvalidPath("Use forward slashes in file paths.")
    if text.startswith("~"):
        raise InvalidPath("Home-directory paths are not allowed.")
    pure = PurePosixPath(text)
    if pure.is_absolute() or text.startswith("/"):
        raise InvalidPath("Absolute paths are not allowed.")
    parts = pure.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise InvalidPath("Path traversal is not allowed.")
    if ":" in parts[0]:
        raise InvalidPath("Drive-qualified paths are not allowed.")
    for part in parts:
        if part.endswith((" ", ".")):
            raise InvalidPath("File names cannot end with a space or dot.")
        if part.split(".")[0].lower() in _RESERVED_NAMES:
            raise InvalidPath("Reserved Windows file names are not allowed.")
        if any(character in part for character in '<>"|?*:') or any(
            ord(character) < 32 for character in part
        ):
            raise InvalidPath("File names contain unsupported characters.")
    normalized = pure.as_posix()
    if len(normalized) > 240:
        raise InvalidPath("The file path is too long.")
    if is_sensitive_path(normalized):
        raise SensitivePath("Sensitive files and directories are not allowed.")
    return normalized


def validate_workspace_name(raw: str) -> str:
    name = raw.strip() if isinstance(raw, str) else ""
    if not _WORKSPACE_NAME.fullmatch(name) or name.endswith("."):
        raise InvalidWorkspaceName(
            "Workspace names must be 1-64 characters: letters, digits, dots, "
            "underscores, or hyphens, and must start with a letter or digit."
        )
    if name.lower() in _RESERVED_NAMES:
        raise InvalidWorkspaceName("That workspace name is reserved by Windows.")
    return name


@dataclass(frozen=True)
class ProposedFile:
    path: str
    content: str
    action: str
    size: int
    language: str


@dataclass(frozen=True)
class Proposal:
    id: str
    workspace: str
    summary: str
    files: tuple[ProposedFile, ...]
    model: str
    requires_overwrite_approval: bool
    warnings: tuple[str, ...]
    created_at: float


class ProposalStore:
    def __init__(self, *, ttl_seconds: int = 1_800, max_items: int = 100) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._items: dict[str, Proposal] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        expired = [
            key
            for key, proposal in self._items.items()
            if now - proposal.created_at > self.ttl_seconds
        ]
        for key in expired:
            self._items.pop(key, None)
        while len(self._items) > self.max_items:
            oldest = min(self._items, key=lambda key: self._items[key].created_at)
            self._items.pop(oldest, None)

    def add(self, proposal: Proposal) -> Proposal:
        with self._lock:
            now = time.time()
            self._prune(now)
            self._items[proposal.id] = proposal
        return proposal

    def get(self, proposal_id: str, workspace: str) -> Proposal:
        with self._lock:
            now = time.time()
            proposal = self._items.get(proposal_id)
            if proposal is None or proposal.workspace != workspace:
                raise ProposalNotFound("The proposed changes were not found.")
            if now - proposal.created_at > self.ttl_seconds:
                self._items.pop(proposal_id, None)
                raise ProposalExpired(
                    "The proposed changes expired. Generate a new proposal."
                )
            return proposal

    def pop(self, proposal_id: str) -> None:
        with self._lock:
            self._items.pop(proposal_id, None)


class WorkspaceService:
    def __init__(
        self,
        root: str | Path,
        *,
        max_files: int = 24,
        max_file_bytes: int = 65_536,
        max_total_bytes: int = 524_288,
        max_context_files: int = 8,
        max_context_bytes: int = 48_000,
        max_read_bytes: int = 262_144,
        max_export_bytes: int = 50_000_000,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self.max_context_files = max_context_files
        self.max_context_bytes = max_context_bytes
        self.max_read_bytes = max_read_bytes
        self.max_export_bytes = max_export_bytes

    @classmethod
    def from_settings(cls, config: Any) -> "WorkspaceService":
        return cls(
            config.workspace_root,
            max_files=config.workspace_max_files,
            max_file_bytes=config.workspace_max_file_bytes,
            max_total_bytes=config.workspace_max_total_bytes,
            max_context_files=config.workspace_max_context_files,
            max_context_bytes=config.workspace_max_context_bytes,
            max_read_bytes=config.workspace_max_read_bytes,
        )

    def _workspace_dir(self, name: str) -> Path:
        safe_name = validate_workspace_name(name)
        candidate = self.root / safe_name
        if _is_link_like(candidate):
            raise InvalidPath("Workspace directories cannot be links or junctions.")
        workspace = candidate.resolve()
        if workspace.parent != self.root:
            raise InvalidPath("The workspace path escapes the workspace root.")
        if not workspace.is_dir():
            raise WorkspaceNotFound(f'Workspace "{safe_name}" does not exist.')
        return workspace

    def _safe_target(self, workspace: Path, relative: str) -> Path:
        path = normalize_relative_path(relative)
        target = (workspace / path).resolve(strict=False)
        if target != workspace and workspace not in target.parents:
            raise InvalidPath("The path escapes the workspace root.")
        return target

    def _iter_files(self, workspace: Path) -> Iterable[Path]:
        for directory, dirnames, filenames in os.walk(workspace, followlinks=False):
            base = Path(directory)
            dirnames[:] = sorted(
                dirname
                for dirname in dirnames
                if dirname.lower() not in _EXCLUDED_DIRS
                and not _is_link_like(base / dirname)
            )
            for filename in sorted(filenames):
                candidate = base / filename
                if _is_link_like(candidate) or not candidate.is_file():
                    continue
                relative = candidate.relative_to(workspace).as_posix()
                if is_sensitive_path(relative):
                    continue
                try:
                    resolved = candidate.resolve(strict=True)
                except OSError:
                    continue
                if workspace not in resolved.parents:
                    continue
                yield candidate

    def _total_size(self, workspace: Path) -> int:
        return sum(file.stat().st_size for file in self._iter_files(workspace))

    def list_workspaces(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for entry in sorted(self.root.iterdir(), key=lambda item: item.name.lower()):
            if _is_link_like(entry) or not entry.is_dir():
                continue
            try:
                name = validate_workspace_name(entry.name)
            except InvalidWorkspaceName:
                continue
            files = list(self._iter_files(entry))
            items.append(
                {
                    "name": name,
                    "file_count": len(files),
                    "size_bytes": sum(file.stat().st_size for file in files),
                    "updated_at": datetime.fromtimestamp(
                        entry.stat().st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                }
            )
        return items

    def create_workspace(self, name: str) -> dict[str, Any]:
        safe_name = validate_workspace_name(name)
        target = self.root / safe_name
        try:
            target.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise WorkspaceExists(f'Workspace "{safe_name}" already exists.') from exc
        return {
            "name": safe_name,
            "file_count": 0,
            "size_bytes": 0,
            "updated_at": datetime.fromtimestamp(
                target.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds"),
        }

    def get_workspace(self, name: str) -> dict[str, Any]:
        workspace = self._workspace_dir(name)
        files = list(self._iter_files(workspace))
        return {
            "name": validate_workspace_name(name),
            "file_count": len(files),
            "size_bytes": sum(file.stat().st_size for file in files),
            "updated_at": datetime.fromtimestamp(
                workspace.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds"),
        }

    def list_files(self, name: str) -> list[dict[str, Any]]:
        workspace = self._workspace_dir(name)
        entries: list[dict[str, Any]] = []
        for file in self._iter_files(workspace):
            relative = file.relative_to(workspace).as_posix()
            entries.append(
                {
                    "path": relative,
                    "size": file.stat().st_size,
                    "language": language_for_path(relative),
                }
            )
        return entries

    def read_file(self, name: str, relative: str) -> dict[str, Any]:
        workspace = self._workspace_dir(name)
        relative = normalize_relative_path(relative)
        target = self._safe_target(workspace, relative)
        if not target.is_file():
            raise WorkspaceNotFound(f'File "{relative}" does not exist.')
        size = target.stat().st_size
        if size > self.max_read_bytes:
            raise FileTooLarge(
                f'File "{relative}" is too large to open ({size} bytes; '
                f"limit {self.max_read_bytes})."
            )
        raw = target.read_bytes()
        if b"\x00" in raw:
            raise BinaryFile(f'File "{relative}" is binary and cannot be displayed.')
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise BinaryFile(f'File "{relative}" is not valid UTF-8 text.') from exc
        return {
            "path": relative,
            "content": content,
            "size": size,
            "language": language_for_path(relative),
        }

    def save_file(self, name: str, relative: str, content: str) -> dict[str, Any]:
        workspace = self._workspace_dir(name)
        relative = normalize_relative_path(relative)
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise FileTooLarge(
                f'File "{relative}" exceeds the {self.max_file_bytes}-byte limit.'
            )
        target = self._safe_target(workspace, relative)
        if target.exists() and not target.is_file():
            raise InvalidPath(f'"{relative}" is not a regular file.')
        existing_size = target.stat().st_size if target.is_file() else 0
        projected = self._total_size(workspace) - existing_size + len(encoded)
        if projected > self.max_total_bytes:
            raise LimitExceeded(
                f'Saving "{relative}" would exceed the '
                f"{self.max_total_bytes}-byte workspace limit."
            )
        self._atomic_write(workspace, target, content)
        return {
            "path": relative,
            "size": len(encoded),
            "language": language_for_path(relative),
            "saved": True,
        }

    def _ensure_parent(self, workspace: Path, target: Path) -> None:
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        resolved = parent.resolve(strict=True)
        if resolved != workspace and workspace not in resolved.parents:
            raise InvalidPath("The file directory escapes the workspace root.")

    def _stage_write(self, workspace: Path, target: Path, content: str) -> Path:
        self._ensure_parent(workspace, target)
        file_descriptor, temporary = tempfile.mkstemp(
            prefix=".vups-", suffix=".tmp", dir=target.parent
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return temporary_path

    def _atomic_write(self, workspace: Path, target: Path, content: str) -> None:
        temporary = self._stage_write(workspace, target, content)
        try:
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def context_for_files(
        self, name: str, paths: Iterable[str]
    ) -> tuple[str, list[str], list[str]]:
        workspace = self._workspace_dir(name)
        warnings: list[str] = []
        used: list[str] = []
        blocks: list[str] = []
        seen: set[str] = set()
        total_bytes = 0
        for raw in paths:
            if len(used) >= self.max_context_files:
                warnings.append(
                    f"Only the first {self.max_context_files} selected files were included."
                )
                break
            try:
                relative = normalize_relative_path(raw)
            except SensitivePath:
                warnings.append(f"Skipped sensitive path: {raw}")
                continue
            except InvalidPath as exc:
                warnings.append(f"Skipped invalid path {raw!r}: {exc}")
                continue
            if relative in seen:
                continue
            seen.add(relative)
            try:
                entry = self.read_file(name, relative)
            except FileTooLarge as exc:
                warnings.append(str(exc))
                continue
            except (WorkspaceError, OSError) as exc:
                warnings.append(f'Skipped "{relative}": {exc}')
                continue
            remaining = self.max_context_bytes - total_bytes
            if remaining <= 0:
                warnings.append(
                    f"Project context stopped at {self.max_context_bytes} bytes."
                )
                break
            content = entry["content"]
            encoded_size = len(content.encode("utf-8"))
            if encoded_size > remaining:
                content = content.encode("utf-8")[:remaining].decode("utf-8", "ignore")
                content += "\n... [truncated by Vups context limit]"
                warnings.append(f'Truncated "{relative}" to the context limit.')
            blocks.append(f"=== FILE: {relative} ===\n{content}\n=== END FILE ===")
            used.append(relative)
            total_bytes += len(content.encode("utf-8"))
        return "\n\n".join(blocks), used, warnings

    def build_proposal(
        self, name: str, model_output: dict[str, Any], model: str
    ) -> Proposal:
        workspace = self._workspace_dir(name)
        if not isinstance(model_output, dict):
            raise InvalidProposal("The model output must be a JSON object.")
        files = model_output.get("files")
        if not isinstance(files, list) or not files:
            raise InvalidProposal("The model output must contain a non-empty files list.")
        if len(files) > self.max_files:
            raise LimitExceeded(
                f"The proposal contains {len(files)} files; the limit is {self.max_files}."
            )
        summary = model_output.get("summary", "")
        if not isinstance(summary, str):
            summary = ""
        summary = " ".join(summary.split())[:2_000]
        proposed: list[ProposedFile] = []
        seen: set[str] = set()
        total_bytes = 0
        for item in files:
            if not isinstance(item, dict):
                raise InvalidProposal("Each proposed file must be an object with path and content.")
            path = item.get("path")
            content = item.get("content")
            if not isinstance(path, str) or not isinstance(content, str):
                raise InvalidProposal("Each proposed file needs a string path and string content.")
            relative = normalize_relative_path(path)
            key = relative.casefold()
            if key in seen:
                raise InvalidProposal(f'The proposal contains a duplicate path: "{relative}".')
            seen.add(key)
            size = len(content.encode("utf-8"))
            if size > self.max_file_bytes:
                raise LimitExceeded(
                    f'File "{relative}" exceeds the {self.max_file_bytes}-byte limit.'
                )
            total_bytes += size
            if total_bytes > self.max_total_bytes:
                raise LimitExceeded(
                    f"The proposal exceeds the {self.max_total_bytes}-byte total limit."
                )
            target = self._safe_target(workspace, relative)
            if target.exists() and not target.is_file():
                raise InvalidPath(f'"{relative}" is not a regular file.')
            action = "update" if target.exists() else "create"
            proposed.append(
                ProposedFile(
                    path=relative,
                    content=content,
                    action=action,
                    size=size,
                    language=language_for_path(relative),
                )
            )
        if not summary:
            summary = f"Proposed {len(proposed)} file{'s' if len(proposed) != 1 else ''}."
        return Proposal(
            id=uuid.uuid4().hex,
            workspace=validate_workspace_name(name),
            summary=summary,
            files=tuple(proposed),
            model=model,
            requires_overwrite_approval=any(file.action == "update" for file in proposed),
            warnings=(),
            created_at=time.time(),
        )

    def apply_proposal(
        self,
        name: str,
        proposal: Proposal,
        *,
        approved: bool,
        approve_overwrites: bool,
    ) -> dict[str, Any]:
        workspace = self._workspace_dir(name)
        if proposal.workspace != validate_workspace_name(name):
            raise ProposalNotFound("The proposal belongs to a different workspace.")
        if not approved:
            raise ApprovalRequired("Explicit approval is required before writing files.")
        targets: list[tuple[ProposedFile, Path]] = []
        seen: set[str] = set()
        overwrites: list[str] = []
        for file in proposal.files:
            relative = normalize_relative_path(file.path)
            if relative.casefold() in seen:
                raise InvalidProposal(f'Duplicate path in proposal: "{relative}".')
            seen.add(relative.casefold())
            size = len(file.content.encode("utf-8"))
            if size > self.max_file_bytes:
                raise FileTooLarge(f'File "{relative}" exceeds the file-size limit.')
            target = self._safe_target(workspace, relative)
            if target.exists() and not target.is_file():
                raise InvalidPath(f'"{relative}" is not a regular file.')
            if target.exists():
                overwrites.append(relative)
            targets.append((file, target))
        if overwrites and not approve_overwrites:
            names = ", ".join(sorted(overwrites))
            raise OverwriteApprovalRequired(
                f"Explicit overwrite approval is required for: {names}"
            )
        projected = self._total_size(workspace)
        for file, target in targets:
            if target.is_file():
                projected -= target.stat().st_size
            projected += len(file.content.encode("utf-8"))
        if projected > self.max_total_bytes:
            raise LimitExceeded(
                f"Applying this proposal would exceed the "
                f"{self.max_total_bytes}-byte workspace limit."
            )

        staged: list[tuple[Path, Path, bool]] = []
        try:
            for file, target in targets:
                temporary = self._stage_write(workspace, target, file.content)
                staged.append((temporary, target, target.exists()))
            created: list[str] = []
            modified: list[str] = []
            for temporary, target, existed in staged:
                os.replace(temporary, target)
                (modified if existed else created).append(
                    target.relative_to(workspace).as_posix()
                )
        except BaseException:
            for temporary, _, _ in staged:
                temporary.unlink(missing_ok=True)
            raise
        return {
            "created": created,
            "modified": modified,
            "summary": (
                f"Applied {len(created) + len(modified)} file(s): "
                f"{len(created)} created, {len(modified)} modified."
            ),
        }

    def export_zip(self, name: str) -> tuple[bytes, str]:
        workspace = self._workspace_dir(name)
        buffer = io.BytesIO()
        total = 0
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in self._iter_files(workspace):
                relative = file.relative_to(workspace).as_posix()
                pure = PurePosixPath(relative)
                if pure.is_absolute() or ".." in pure.parts:
                    continue
                raw = file.read_bytes()
                total += len(raw)
                if total > self.max_export_bytes:
                    raise FileTooLarge("The workspace is too large to export as one ZIP.")
                archive.writestr(relative, raw)
            archive.comment = (
                b"Generated by Vups Gen 4. Review all code before execution."
            )
        return buffer.getvalue(), f"{validate_workspace_name(name)}.zip"
