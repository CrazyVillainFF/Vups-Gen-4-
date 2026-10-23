from __future__ import annotations

from pathlib import Path

import pytest

from backend.config import ENV_FILE, PROJECT_ROOT, Settings, load_env_file


def test_env_file_is_anchored_to_project_root() -> None:
    assert ENV_FILE == PROJECT_ROOT / ".env"


def test_env_file_load_does_not_override_existing_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("MODEL_NAME=from-file\nMODEL_TIMEOUT_SECONDS=45\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    environ = {"MODEL_NAME": "from-os"}

    loaded = load_env_file(env_file, environ)

    assert loaded == env_file
    assert environ["MODEL_NAME"] == "from-os"
    assert environ["MODEL_TIMEOUT_SECONDS"] == "45"


def test_settings_load_from_env_file_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "project.env"
    env_file.write_text(
        "\n".join(
            [
                "APP_NAME=Test Vups",
                "MODEL_BASE_URL=http://127.0.0.1:11434/v1/",
                "MODEL_NAME=qwen2.5-coder:7b",
                "MODEL_API_KEY=ollama",
                "MODEL_TIMEOUT_SECONDS=30",
                "MAX_MESSAGE_CHARS=9000",
            ]
        ),
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    settings = Settings.from_env(environ={}, env_file=env_file)

    assert settings.app_name == "Test Vups"
    assert settings.model_base_url == "http://127.0.0.1:11434/v1"
    assert settings.model_name == "qwen2.5-coder:7b"
    assert settings.model_configured is True
    assert settings.model_timeout_seconds == 30.0
    assert settings.max_message_chars == 9000


def test_environment_variables_take_precedence_over_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "MODEL_BASE_URL=http://file.local/v1\nMODEL_NAME=file-model\n",
        encoding="utf-8",
    )

    settings = Settings.from_env(
        environ={
            "MODEL_BASE_URL": "http://os.local/v1",
            "MODEL_NAME": "os-model",
        },
        env_file=env_file,
    )

    assert settings.model_base_url == "http://os.local/v1"
    assert settings.model_name == "os-model"


def test_missing_model_configuration_is_reported(tmp_path: Path) -> None:
    settings = Settings.from_env(environ={}, env_file=tmp_path / "missing.env")

    assert settings.model_configured is False
    assert settings.missing_model_settings == ["MODEL_BASE_URL", "MODEL_NAME"]
    assert settings.chat_completions_url == ""


def test_chat_completions_url_and_ollama_root() -> None:
    settings = Settings(
        model_base_url="http://127.0.0.1:11434/v1",
        model_name="qwen2.5-coder:7b",
    )

    assert settings.chat_completions_url == "http://127.0.0.1:11434/v1/chat/completions"
    assert settings.ollama_api_root == "http://127.0.0.1:11434"


def test_invalid_timeout_is_rejected_with_variable_name(tmp_path: Path) -> None:
    env_file = tmp_path / "bad.env"
    env_file.write_text("MODEL_TIMEOUT_SECONDS=0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="MODEL_TIMEOUT_SECONDS"):
        Settings.from_env(environ={}, env_file=env_file)


def test_vision_support_can_be_overridden(tmp_path: Path) -> None:
    missing = tmp_path / "missing.env"
    enabled = Settings.from_env(
        environ={"MODEL_SUPPORTS_VISION": "true"},
        env_file=missing,
    )
    disabled = Settings.from_env(
        environ={"MODEL_SUPPORTS_VISION": "false"},
        env_file=missing,
    )

    assert enabled.model_supports_vision is True
    assert disabled.model_supports_vision is False
    with pytest.raises(ValueError, match="MODEL_SUPPORTS_VISION"):
        Settings.from_env(
            environ={"MODEL_SUPPORTS_VISION": "sometimes"},
            env_file=missing,
        )


def test_workspace_settings_parse_relative_root_and_limits(tmp_path: Path) -> None:
    env_file = tmp_path / "workspace.env"
    env_file.write_text(
        "\n".join(
            [
                "WORKSPACE_ROOT=my-workspaces",
                "WORKSPACE_MAX_FILES=4",
                "WORKSPACE_MAX_FILE_BYTES=1024",
                "WORKSPACE_MAX_TOTAL_BYTES=2048",
                "WORKSPACE_MAX_CONTEXT_FILES=2",
                "WORKSPACE_MAX_CONTEXT_BYTES=512",
                "WORKSPACE_MAX_READ_BYTES=4096",
                "WORKSPACE_MODEL_MAX_TOKENS=1024",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings.from_env(environ={}, env_file=env_file)

    assert settings.workspace_root == (PROJECT_ROOT / "my-workspaces").resolve()
    assert settings.workspace_max_files == 4
    assert settings.workspace_max_file_bytes == 1024
    assert settings.workspace_max_total_bytes == 2048
    assert settings.workspace_max_context_files == 2
    assert settings.workspace_max_context_bytes == 512
    assert settings.workspace_max_read_bytes == 4096
    assert settings.workspace_model_max_tokens == 1024


def test_invalid_workspace_limit_is_rejected(tmp_path: Path) -> None:
    env_file = tmp_path / "bad-workspace.env"
    env_file.write_text("WORKSPACE_MAX_FILES=0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="WORKSPACE_MAX_FILES"):
        Settings.from_env(environ={}, env_file=env_file)


def test_vision_settings_use_native_ollama_root_and_aliases(tmp_path: Path) -> None:
    env_file = tmp_path / "vision.env"
    env_file.write_text(
        "\n".join(
            [
                "MODEL_BASE_URL=http://ollama.test/v1",
                "OLLAMA_VISION_MODEL=qwen2.5vl:7b",
                "VISION_BASE_URL=http://vision.test/v1",
                "VISION_TIMEOUT_SECONDS=90",
                "VISION_MODEL_MAX_TOKENS=1024",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings.from_env(environ={}, env_file=env_file)

    assert settings.vision_model_name == "qwen2.5vl:7b"
    assert settings.vision_api_root == "http://vision.test"
    assert settings.vision_chat_url == "http://vision.test/api/chat"
    assert settings.vision_timeout_seconds == 90.0
    assert settings.vision_model_max_tokens == 1024


def test_vision_falls_back_to_model_base_url(tmp_path: Path) -> None:
    settings = Settings.from_env(
        environ={"MODEL_BASE_URL": "http://127.0.0.1:11434/v1"},
        env_file=tmp_path / "missing.env",
    )

    assert settings.vision_configured is True
    assert settings.vision_chat_url == "http://127.0.0.1:11434/api/chat"


def test_image_configuration_parses_output_and_generation_provider(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / "images.env"
    env_file.write_text(
        "\n".join(
            [
                "IMAGE_GENERATION_PROVIDER=comfyui",
                "COMFYUI_BASE_URL=http://comfy.test/",
                "COMFYUI_CHECKPOINT=sd15.safetensors",
                "IMAGE_OUTPUT_DIR=my-images",
                "IMAGE_MAX_UPLOAD_BYTES=4096",
                "IMAGE_MAX_PIXELS=100000",
                "IMAGE_MAX_DIMENSION=2048",
                "IMAGE_GENERATION_MAX_OUTPUT_BYTES=8192",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings.from_env(environ={}, env_file=env_file)

    assert settings.image_generation_provider == "comfyui"
    assert settings.image_generation_configured is True
    assert settings.comfyui_base_url == "http://comfy.test"
    assert settings.image_output_dir == (PROJECT_ROOT / "my-images").resolve()
    assert settings.image_max_upload_bytes == 4096
    assert settings.image_max_pixels == 100_000
    assert settings.image_max_dimension == 2048
    assert settings.image_generation_max_output_bytes == 8192


def test_invalid_image_generation_provider_is_rejected(tmp_path: Path) -> None:
    env_file = tmp_path / "bad-images.env"
    env_file.write_text(
        "IMAGE_GENERATION_PROVIDER=cloud-service\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="IMAGE_GENERATION_PROVIDER"):
        Settings.from_env(environ={}, env_file=env_file)
