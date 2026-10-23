from __future__ import annotations

import argparse
import asyncio
import shutil
import sys

import httpx

from .config import ENV_FILE, Settings
from .image_understanding import ImageAnalyzer
from .provider import ChatProvider, ModelUnavailable


def _print_config(config: Settings) -> None:
    print("Vups Gen 4 model diagnostics")
    print(f"  .env file: {ENV_FILE} ({'found' if ENV_FILE.is_file() else 'missing'})")
    print(f"  MODEL_BASE_URL: {config.model_base_url or '(not set)'}")
    print(f"  MODEL_NAME: {config.model_name or '(not set)'}")
    print(f"  MODEL_API_KEY: {'set (not displayed)' if config.model_api_key else '(not set)'}")
    print(f"  MODEL_TIMEOUT_SECONDS: {config.model_timeout_seconds:g}")
    print(f"  VISION_MODEL_NAME: {config.vision_model_name or '(not set)'}")
    print(f"  VISION_BASE_URL: {config.vision_api_root or '(not set)'}")
    print(f"  VISION_TIMEOUT_SECONDS: {config.vision_timeout_seconds:g}")
    print(f"  IMAGE_GENERATION_PROVIDER: {config.image_generation_provider}")
    if config.image_generation_provider == "comfyui":
        print(f"  COMFYUI_BASE_URL: {config.comfyui_base_url or '(not set)'}")
        print(f"  COMFYUI_CHECKPOINT: {config.comfyui_checkpoint or '(not set)'}")


def _ollama_models(config: Settings) -> tuple[bool | None, list[str], str]:
    if not config.ollama_api_root:
        return None, [], "MODEL_BASE_URL is empty."
    url = f"{config.ollama_api_root}/api/tags"
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
    except httpx.HTTPError:
        return False, [], "Ollama native API is not reachable."
    if response.status_code == 404:
        return None, [], "Not an Ollama native API endpoint; skipped the Ollama check."
    if response.status_code >= 400:
        return False, [], f"Ollama returned HTTP {response.status_code}."
    try:
        payload = response.json()
    except ValueError:
        return False, [], "Ollama returned an invalid model list."
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return False, [], "Ollama returned an invalid model list."
    names = sorted(
        item["name"]
        for item in models
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    )
    return True, names, ""


def _check_vision(config: Settings) -> None:
    print("")
    print("Checking the native Ollama vision model...")
    status = asyncio.run(ImageAnalyzer(config).probe())
    print(f"  Status: {status['status']}")
    print(f"  Endpoint reachable: {status['reachable']}")
    print(f"  Model listed: {status['model_available']}")
    if status.get("detail"):
        print(f"  Detail: {status['detail']}")
    if status["status"] == "model_missing":
        print(f"  Fix: ollama pull {config.vision_model_name}")
    print("  Note: this checks installation only; use Image AI or a vision smoke test "
          "to verify a real image response.")


def _setup_commands(model: str) -> None:
    print("")
    print("Setup commands (Windows PowerShell):")
    print("  if (-not (Test-Path .env)) { Copy-Item .env.example .env }")
    print("  ollama serve")
    print(f"  ollama pull {model}")
    print("  python -m backend.check_model")
    print("  uvicorn backend.app:app --reload --port 8000")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the configured Vups Gen 4 model connection.")
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip the real chat-completions test (avoids loading the model).",
    )
    args = parser.parse_args(argv)

    try:
        config = Settings.from_env()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        print(f"Fix the value in {ENV_FILE} and retry.")
        return 1

    _print_config(config)
    if not config.model_configured:
        print("")
        print("Model is NOT configured.")
        print(f"  Missing: {', '.join(config.missing_model_settings)}")
        _setup_commands(config.model_name or "qwen2.5-coder:7b")
        return 1

    provider = ChatProvider(config)
    print("")
    print("Checking the OpenAI-compatible model list...")
    status = asyncio.run(provider.probe(live=False))
    print(f"  Status: {status['status']}")
    print(f"  Endpoint reachable: {status['reachable']}")
    print(f"  Model listed: {status['model_available']}")
    if status.get("detail"):
        print(f"  Detail: {status['detail']}")

    ollama_ok, ollama_names, ollama_note = _ollama_models(config)
    print("")
    print("Checking Ollama's native API...")
    if ollama_ok:
        print(f"  Ollama reachable; {len(ollama_names)} model(s) installed.")
        if config.model_name in ollama_names:
            print(f"  Configured model is installed: {config.model_name}")
        else:
            matches = [
                name
                for name in ollama_names
                if name.split(":", 1)[0] == config.model_name.split(":", 1)[0]
            ]
            suffix = f" Similar installed tags: {', '.join(matches)}." if matches else ""
            print(f"  Configured model is NOT installed.{suffix}")
            print(f"  Fix: ollama pull {config.model_name}")
    elif ollama_ok is None:
        print(f"  {ollama_note}")
    else:
        print(f"  {ollama_note}")
        if not shutil.which("ollama"):
            print("  Ollama CLI was not found on PATH. Install Ollama or start its desktop app.")
        print("  Fix: start Ollama with 'ollama serve' and retry.")

    _check_vision(config)

    live_ok = True
    if not args.skip_live:
        print("")
        print("Testing a real chat-completions response...")
        if not status["reachable"] and not ollama_ok:
            print("  Skipped: the endpoint is not reachable.")
            live_ok = False
        else:
            try:
                reply = asyncio.run(
                    provider.generate(
                        system_prompt="You are a connection test. Reply with exactly: OK",
                        message="Reply with exactly: OK",
                        history=[],
                        max_tokens=8,
                    )
                )
            except ModelUnavailable as exc:
                print(f"  FAILED: {exc}")
                live_ok = False
            else:
                preview = " ".join(reply.split())[:100]
                print(f"  OK, real response received: {preview!r}")

    model_available = bool(status["model_available"]) or (
        bool(ollama_ok) and config.model_name in ollama_names
    )
    if not model_available:
        print("")
        print(f"Model check failed: {config.model_name} is not available to the endpoint.")
        print(f"Run: ollama pull {config.model_name}")
        return 1
    if not live_ok:
        print("")
        print("Model check failed: the chat-completions test did not succeed.")
        _setup_commands(config.model_name)
        return 1

    print("")
    print("Model check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
