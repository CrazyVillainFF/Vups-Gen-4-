from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT / "workspaces"
DEFAULT_IMAGE_OUTPUT_DIR = PROJECT_ROOT / "generated_images"
IMAGE_GENERATION_PROVIDERS = {"none", "comfyui"}


def load_env_file(
    path: str | Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> Path | None:
    env_path = Path(path) if path is not None else ENV_FILE
    target = os.environ if environ is None else environ
    if not env_path.is_file():
        return None
    for key, value in dotenv_values(env_path).items():
        if value is not None:
            target.setdefault(key, value)
    return env_path


def _text(values: Mapping[str, str], key: str, default: str) -> str:
    value = values.get(key, default)
    return value.strip() if isinstance(value, str) else default


def _positive_float(values: Mapping[str, str], key: str, default: float) -> float:
    raw = _text(values, key, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{key} must be greater than zero.")
    return value


def _positive_int(values: Mapping[str, str], key: str, default: int) -> int:
    raw = _text(values, key, "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{key} must be greater than zero.")
    return value


def _optional_bool(values: Mapping[str, str], key: str) -> bool | None:
    raw = _text(values, key, "").lower()
    if not raw or raw == "auto":
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be auto, true, or false, got {raw!r}.")


def _project_path(values: Mapping[str, str], key: str, default: Path) -> Path:
    raw = _text(values, key, "")
    if not raw:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _generation_provider(values: Mapping[str, str]) -> str:
    provider = _text(values, "IMAGE_GENERATION_PROVIDER", "none").lower()
    if provider not in IMAGE_GENERATION_PROVIDERS:
        allowed = ", ".join(sorted(IMAGE_GENERATION_PROVIDERS))
        raise ValueError(
            f"IMAGE_GENERATION_PROVIDER must be one of {allowed}, got {provider!r}."
        )
    return provider


@dataclass(frozen=True)
class Settings:
    app_name: str = "Vups Gen 4"
    database_url: str = "sqlite:///./vups_gen4.db"
    turso_database_url: str = ""
    turso_auth_token: str = ""
    model_base_url: str = ""
    model_name: str = ""
    model_api_key: str = ""
    model_timeout_seconds: float = 60.0
    max_message_chars: int = 12000
    model_supports_vision: bool | None = None
    vision_model_name: str = "qwen2.5vl:7b"
    vision_base_url: str = ""
    vision_timeout_seconds: float = 900.0
    vision_model_max_tokens: int = 2_048
    image_max_upload_bytes: int = 5_242_880
    image_max_pixels: int = 20_000_000
    image_max_dimension: int = 8_192
    image_generation_provider: str = "none"
    comfyui_base_url: str = "http://127.0.0.1:8188"
    comfyui_checkpoint: str = "v1-5-pruned-emaonly.safetensors"
    comfyui_timeout_seconds: float = 300.0
    image_output_dir: Path = DEFAULT_IMAGE_OUTPUT_DIR
    image_generation_max_output_bytes: int = 20_000_000
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT
    workspace_max_files: int = 24
    workspace_max_file_bytes: int = 65_536
    workspace_max_total_bytes: int = 524_288
    workspace_max_context_files: int = 8
    workspace_max_context_bytes: int = 48_000
    workspace_max_read_bytes: int = 262_144
    workspace_model_max_tokens: int = 4_096

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        env_file: str | Path | None = None,
    ) -> "Settings":
        values: dict[str, str] = {}
        load_env_file(env_file, values)
        values.update(os.environ if environ is None else environ)
        return cls(
            app_name=_text(values, "APP_NAME", "Vups Gen 4"),
            database_url=_text(values, "DATABASE_URL", "sqlite:///./vups_gen4.db"),
            turso_database_url=_text(values, "TURSO_DATABASE_URL", ""),
            turso_auth_token=_text(values, "TURSO_AUTH_TOKEN", ""),
            model_base_url=_text(values, "MODEL_BASE_URL", "").rstrip("/"),
            model_name=_text(values, "MODEL_NAME", ""),
            model_api_key=_text(values, "MODEL_API_KEY", ""),
            model_timeout_seconds=_positive_float(values, "MODEL_TIMEOUT_SECONDS", 60.0),
            max_message_chars=_positive_int(values, "MAX_MESSAGE_CHARS", 12000),
            model_supports_vision=_optional_bool(values, "MODEL_SUPPORTS_VISION"),
            vision_model_name=(
                _text(values, "VISION_MODEL_NAME", "")
                or _text(values, "OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
            ),
            vision_base_url=(
                _text(values, "VISION_BASE_URL", "")
                or _text(values, "OLLAMA_BASE_URL", "")
            ).rstrip("/"),
            vision_timeout_seconds=_positive_float(
                values, "VISION_TIMEOUT_SECONDS", 900.0
            ),
            vision_model_max_tokens=_positive_int(
                values, "VISION_MODEL_MAX_TOKENS", 2_048
            ),
            image_max_upload_bytes=_positive_int(
                values, "IMAGE_MAX_UPLOAD_BYTES", 5_242_880
            ),
            image_max_pixels=_positive_int(
                values, "IMAGE_MAX_PIXELS", 20_000_000
            ),
            image_max_dimension=_positive_int(
                values, "IMAGE_MAX_DIMENSION", 8_192
            ),
            image_generation_provider=_generation_provider(values),
            comfyui_base_url=_text(
                values, "COMFYUI_BASE_URL", "http://127.0.0.1:8188"
            ).rstrip("/"),
            comfyui_checkpoint=_text(
                values,
                "COMFYUI_CHECKPOINT",
                "v1-5-pruned-emaonly.safetensors",
            ),
            comfyui_timeout_seconds=_positive_float(
                values, "COMFYUI_TIMEOUT_SECONDS", 300.0
            ),
            image_output_dir=_project_path(
                values, "IMAGE_OUTPUT_DIR", DEFAULT_IMAGE_OUTPUT_DIR
            ),
            image_generation_max_output_bytes=_positive_int(
                values, "IMAGE_GENERATION_MAX_OUTPUT_BYTES", 20_000_000
            ),
            workspace_root=_project_path(values, "WORKSPACE_ROOT", DEFAULT_WORKSPACE_ROOT),
            workspace_max_files=_positive_int(values, "WORKSPACE_MAX_FILES", 24),
            workspace_max_file_bytes=_positive_int(values, "WORKSPACE_MAX_FILE_BYTES", 65_536),
            workspace_max_total_bytes=_positive_int(values, "WORKSPACE_MAX_TOTAL_BYTES", 524_288),
            workspace_max_context_files=_positive_int(values, "WORKSPACE_MAX_CONTEXT_FILES", 8),
            workspace_max_context_bytes=_positive_int(values, "WORKSPACE_MAX_CONTEXT_BYTES", 48_000),
            workspace_max_read_bytes=_positive_int(values, "WORKSPACE_MAX_READ_BYTES", 262_144),
            workspace_model_max_tokens=_positive_int(values, "WORKSPACE_MODEL_MAX_TOKENS", 4_096),
        )

    @property
    def turso_enabled(self) -> bool:
        return bool(self.turso_database_url and self.turso_auth_token)

    @property
    def model_configured(self) -> bool:
        return bool(self.model_base_url and self.model_name)

    @property
    def missing_model_settings(self) -> list[str]:
        missing: list[str] = []
        if not self.model_base_url:
            missing.append("MODEL_BASE_URL")
        if not self.model_name:
            missing.append("MODEL_NAME")
        return missing

    @property
    def chat_completions_url(self) -> str:
        if not self.model_base_url:
            return ""
        if self.model_base_url.endswith("/chat/completions"):
            return self.model_base_url
        return f"{self.model_base_url}/chat/completions"

    @property
    def ollama_api_root(self) -> str:
        base = self.model_base_url
        if base.endswith("/v1/chat/completions"):
            return base[: -len("/v1/chat/completions")]
        if base.endswith("/chat/completions"):
            return base[: -len("/chat/completions")]
        if base.endswith("/v1"):
            return base[: -len("/v1")]
        return base

    @property
    def vision_configured(self) -> bool:
        return bool(self.vision_model_name and self.vision_api_root)

    @property
    def missing_vision_settings(self) -> list[str]:
        missing: list[str] = []
        if not self.vision_api_root:
            missing.append("VISION_BASE_URL (or MODEL_BASE_URL)")
        if not self.vision_model_name:
            missing.append("VISION_MODEL_NAME")
        return missing

    @property
    def vision_api_root(self) -> str:
        base = (self.vision_base_url or self.model_base_url).rstrip("/")
        for suffix in (
            "/api/chat",
            "/v1/chat/completions",
            "/chat/completions",
            "/api",
            "/v1",
        ):
            if base.endswith(suffix):
                return base[: -len(suffix)].rstrip("/")
        return base

    @property
    def vision_chat_url(self) -> str:
        root = self.vision_api_root
        return f"{root}/api/chat" if root else ""

    @property
    def image_generation_configured(self) -> bool:
        return bool(
            self.image_generation_provider == "comfyui"
            and self.comfyui_base_url
            and self.comfyui_checkpoint
        )


load_env_file()
settings = Settings.from_env()
