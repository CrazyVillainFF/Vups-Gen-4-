from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.config import Settings
from backend.image_generation import (
    ComfyGenerationRequest,
    ComfyUIUnavailable,
    ComfyUIWorkflowError,
    ImageGenerationError,
    ImageGenerationNotConfigured,
    ImageGenerator,
)
from backend.image_understanding import (
    ImageAnalyzer,
    VisionNotConfigured,
    VisionUnavailable,
)
from backend.image_validation import (
    ImageValidationError,
    inspect_image_bytes,
    validate_image_data_url,
)


def png_bytes(width: int = 8, height: int = 8, color: str = "blue") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(raw: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{base64.b64encode(raw).decode('ascii')}"


def build_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_name": "Vups Image Test",
        "model_base_url": "http://ollama.test/v1",
        "model_name": "qwen2.5-coder:7b",
        "vision_model_name": "qwen2.5vl:7b",
        "vision_base_url": "http://ollama.test",
        "vision_timeout_seconds": 5.0,
        "image_generation_provider": "comfyui",
        "comfyui_base_url": "http://comfy.test",
        "comfyui_checkpoint": "model.safetensors",
        "comfyui_timeout_seconds": 5.0,
        "image_output_dir": tmp_path / "generated",
    }
    values.update(overrides)
    return Settings(**values)


def validate(raw: bytes, **overrides: int) -> Any:
    return validate_image_data_url(
        data_url(raw),
        max_bytes=overrides.get("max_bytes", 5_242_880),
        max_pixels=overrides.get("max_pixels", 20_000_000),
        max_dimension=overrides.get("max_dimension", 8_192),
    )


def test_valid_png_is_inspected_without_trusting_the_filename() -> None:
    raw = png_bytes(width=12, height=9)

    image = validate(raw)

    assert image.media_type == "image/png"
    assert image.image_format == "PNG"
    assert image.width == 12
    assert image.height == 9
    assert image.size_bytes == len(raw)
    assert base64.b64decode(image.base64_data) == raw


def test_invalid_base64_and_declared_type_mismatch_are_rejected() -> None:
    with pytest.raises(ImageValidationError) as malformed:
        validate_image_data_url(
            "data:image/png;base64,not valid!",
            max_bytes=1024,
            max_pixels=100,
            max_dimension=100,
        )
    assert malformed.value.code == "invalid_image"

    with pytest.raises(ImageValidationError) as mismatch:
        validate_image_data_url(
            data_url(png_bytes(), media_type="image/jpeg"),
            max_bytes=1024,
            max_pixels=100,
            max_dimension=100,
        )
    assert mismatch.value.code == "unsupported_image"


def test_unsupported_format_and_unsafe_dimensions_are_rejected() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="GIF")
    with pytest.raises(ImageValidationError) as unsupported:
        inspect_image_bytes(
            buffer.getvalue(),
            max_bytes=1024,
            max_pixels=100,
            max_dimension=100,
        )
    assert unsupported.value.code == "unsupported_image"

    with pytest.raises(ImageValidationError) as too_many_pixels:
        validate(png_bytes(width=8, height=8), max_pixels=4)
    assert too_many_pixels.value.status_code == 422


def test_upload_byte_limit_is_enforced_before_processing() -> None:
    with pytest.raises(ImageValidationError) as too_large:
        validate(png_bytes(width=64, height=64), max_bytes=40)
    assert too_large.value.code == "image_too_large"
    assert too_large.value.status_code == 413


def test_vision_uses_native_ollama_images_payload_and_real_response() -> None:
    raw = png_bytes()
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.read())
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "A small blue image."}},
        )

    analyzer = ImageAnalyzer(
        build_settings(Path(".")),
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(
        analyzer.analyze(
            image_data_url=data_url(raw),
            prompt="What is here?",
            mode="describe",
        )
    )

    payload = seen["payload"]
    user_message = payload["messages"][-1]
    assert seen["url"] == "http://ollama.test/api/chat"
    assert payload["model"] == "qwen2.5vl:7b"
    assert payload["stream"] is True
    assert payload["options"]["num_predict"] == 2_048
    assert user_message["content"] == "What is here?"
    assert base64.b64decode(user_message["images"][0]) == raw
    assert result.answer == "A small blue image."
    assert result.image.width == 8


def test_vision_unconfigured_and_invalid_image_never_call_the_provider() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("The vision provider must not be called")

    missing = ImageAnalyzer(
        Settings(model_base_url="", model_name="", vision_base_url=""),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(VisionNotConfigured):
        asyncio.run(
            missing.analyze(image_data_url=data_url(png_bytes()), prompt="Read")
        )

    configured = ImageAnalyzer(
        build_settings(Path(".")),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ImageValidationError):
        asyncio.run(
            configured.analyze(
                image_data_url="data:image/png;base64,AAAA",
                prompt="Read",
            )
        )


def test_vision_model_errors_are_actionable() -> None:
    def missing_model(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": "model 'qwen2.5vl:7b' not found, try pulling it first"},
        )

    analyzer = ImageAnalyzer(
        build_settings(Path(".")),
        transport=httpx.MockTransport(missing_model),
    )
    with pytest.raises(VisionUnavailable, match="ollama pull qwen2.5vl:7b"):
        asyncio.run(
            analyzer.analyze(image_data_url=data_url(png_bytes()), prompt="Read")
        )


def test_vision_slow_streamed_inference_completes_once() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.03)
        return httpx.Response(
            200,
            content=(
                b'{"message":{"content":"Slow answer."},"done":false}\n'
                b'{"message":{"content":""},"done":true,'
                b'"prompt_eval_count":1234,"eval_count":7}\n'
            ),
        )

    analyzer = ImageAnalyzer(
        build_settings(Path("."), vision_timeout_seconds=0.2),
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(
        analyzer.analyze(image_data_url=data_url(png_bytes()), prompt="Read")
    )

    assert result.answer == "Slow answer."
    assert calls == 1


def test_vision_total_timeout_is_clear_and_does_not_retry() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)
        return httpx.Response(
            200,
            content=b'{"message":{"content":"too late"},"done":true}\n',
        )

    analyzer = ImageAnalyzer(
        build_settings(Path("."), vision_timeout_seconds=0.02),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(VisionUnavailable) as caught:
        asyncio.run(
            analyzer.analyze(image_data_url=data_url(png_bytes()), prompt="Read")
        )

    message = str(caught.value)
    assert calls == 1
    assert "timed out after" in message
    assert "still loading" not in message
    assert "duplicate work" in message
    assert "VISION_TIMEOUT_SECONDS" in message


def test_vision_logs_timing_without_image_or_prompt_contents(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = png_bytes()
    prompt = "PRIVATE-PROMPT-TEXT"
    encoded = base64.b64encode(raw).decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'{"message":{"content":"Logged answer."},"done":false}\n'
                b'{"message":{"content":""},"done":true,'
                b'"prompt_eval_count":321,"eval_count":5}\n'
            ),
        )

    caplog.set_level(logging.INFO, logger="backend.image_understanding")
    result = asyncio.run(
        ImageAnalyzer(
            build_settings(Path(".")),
            transport=httpx.MockTransport(handler),
        ).analyze(
            image_data_url=data_url(raw),
            prompt=prompt,
            mode="describe",
        )
    )

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert result.answer == "Logged answer."
    assert "Vision analysis started" in logs
    assert "response started" in logs
    assert "first content" in logs
    assert "completed" in logs
    assert "prompt_tokens=321" in logs
    assert "completion_tokens=5" in logs
    assert encoded not in logs
    assert prompt not in logs


def test_vision_http_error_preserves_provider_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "vision runner failed"})

    analyzer = ImageAnalyzer(
        build_settings(Path(".")),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(VisionUnavailable) as caught:
        asyncio.run(
            analyzer.analyze(image_data_url=data_url(png_bytes()), prompt="Read")
        )

    assert "HTTP 500" in str(caught.value)
    assert "vision runner failed" in str(caught.value)


def test_vision_probe_distinguishes_missing_model_and_success() -> None:
    def missing(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://ollama.test/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    status = asyncio.run(
        ImageAnalyzer(
            build_settings(Path(".")),
            transport=httpx.MockTransport(missing),
        ).probe()
    )
    assert status["status"] == "model_missing"
    assert status["model_available"] is False

    def available(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen2.5vl:7b"}]})

    status = asyncio.run(
        ImageAnalyzer(
            build_settings(Path(".")),
            transport=httpx.MockTransport(available),
        ).probe()
    )
    assert status["status"] == "available"
    assert status["model_available"] is True


def comfy_handler(seen: dict[str, Any] | None = None) -> Any:
    raw = png_bytes(width=32, height=24)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if seen is not None:
            seen.setdefault("paths", []).append(path)
        if path == "/prompt":
            if seen is not None:
                seen["queue"] = json.loads(request.read())
            return httpx.Response(200, json={"prompt_id": "prompt-1"})
        if path == "/history/prompt-1":
            return httpx.Response(
                200,
                json={
                    "prompt-1": {
                        "outputs": {
                            "9": {
                                "images": [
                                    {
                                        "filename": "result.png",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        }
                    }
                },
            )
        if path == "/view":
            return httpx.Response(200, content=raw, headers={"content-type": "image/png"})
        return httpx.Response(404)

    return handler


def test_comfyui_generation_queues_workflow_saves_and_reads_safe_image(
    tmp_path: Path,
) -> None:
    seen: dict[str, Any] = {}
    generator = ImageGenerator(
        build_settings(tmp_path),
        transport=httpx.MockTransport(comfy_handler(seen)),
        poll_interval=0,
    )

    result = asyncio.run(
        generator.generate(
            ComfyGenerationRequest(
                prompt="A quiet lake",
                negative_prompt="blurry",
                width=640,
                height=384,
                steps=12,
                cfg=6.5,
                seed=42,
            )
        )
    )

    queued = seen["queue"]
    workflow = queued["prompt"]
    assert workflow["4"]["inputs"]["ckpt_name"] == "model.safetensors"
    assert workflow["6"]["inputs"]["text"] == "A quiet lake"
    assert workflow["7"]["inputs"]["text"] == "blurry"
    assert workflow["5"]["inputs"]["width"] == 640
    assert workflow["5"]["inputs"]["height"] == 384
    assert workflow["3"]["inputs"]["seed"] == 42
    assert workflow["3"]["inputs"]["steps"] == 12
    assert result.image_id.endswith(".png")
    assert result.media_type == "image/png"
    assert result.seed == 42
    stored = generator.outputs.get(result.image_id)
    assert stored is not None
    assert stored[0].read_bytes() == png_bytes(width=32, height=24)
    assert generator.outputs.get("../outside.png") is None
    assert generator.outputs.get("a" * 32 + ".png") is None


def test_generation_disabled_or_rejected_never_writes_an_image(
    tmp_path: Path,
) -> None:
    disabled = ImageGenerator(
        build_settings(tmp_path, image_generation_provider="none")
    )
    with pytest.raises(ImageGenerationNotConfigured):
        asyncio.run(disabled.generate(ComfyGenerationRequest(prompt="test")))
    assert list((tmp_path / "generated").glob("*")) == []

    def rejected(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "invalid prompt"}, "node_errors": {"4": {}}},
        )

    generator = ImageGenerator(
        build_settings(tmp_path),
        transport=httpx.MockTransport(rejected),
        poll_interval=0,
    )
    with pytest.raises(ComfyUIWorkflowError, match="COMFYUI_CHECKPOINT"):
        asyncio.run(generator.generate(ComfyGenerationRequest(prompt="test")))
    assert list((tmp_path / "generated").glob("*")) == []


def test_comfyui_connection_failure_is_explicit(tmp_path: Path) -> None:
    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    generator = ImageGenerator(
        build_settings(tmp_path),
        transport=httpx.MockTransport(offline),
        poll_interval=0,
    )
    with pytest.raises(ComfyUIUnavailable, match="Could not connect"):
        asyncio.run(generator.generate(ComfyGenerationRequest(prompt="test")))


def test_generated_image_save_failure_is_reported_safely(tmp_path: Path) -> None:
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("blocked", encoding="utf-8")
    generator = ImageGenerator(
        build_settings(tmp_path, image_output_dir=output_path),
        transport=httpx.MockTransport(comfy_handler()),
        poll_interval=0,
    )

    with pytest.raises(ImageGenerationError, match="could not be saved"):
        asyncio.run(generator.generate(ComfyGenerationRequest(prompt="test")))


def test_comfyui_probe_reports_available_checkpoint(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {}})
        if request.url.path == "/object_info/CheckpointLoaderSimple":
            return httpx.Response(
                200,
                json={
                    "CheckpointLoaderSimple": {
                        "input": {
                            "required": {"ckpt_name": [["model.safetensors"]]}
                        }
                    }
                },
            )
        return httpx.Response(404)

    status = asyncio.run(
        ImageGenerator(
            build_settings(tmp_path),
            transport=httpx.MockTransport(handler),
            poll_interval=0,
        ).probe()
    )

    assert status["status"] == "available"
    assert status["reachable"] is True
    assert status["checkpoint_available"] is True


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, str, str]] = []

    def save_turn(
        self, conversation_id: str, user_message: str, assistant_message: str, skill: str
    ) -> None:
        self.saved.append(
            (conversation_id, user_message, assistant_message, skill)
        )

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return []

    def ping(self) -> bool:
        return True


class FakeProvider:
    def __init__(self) -> None:
        self.configured = True
        self.vision_supported = False
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "text answer"

    async def probe(self, *, live: bool = False) -> dict[str, Any]:
        return {
            "configured": True,
            "status": "available",
            "reachable": True,
            "model_available": True,
            "model": "qwen2.5-coder:7b",
            "base_url": "http://ollama.test/v1",
            "endpoint": "http://ollama.test/v1/chat/completions",
            "vision_supported": False,
            "live_checked": live,
            "live_ok": live,
            "detail": "",
        }


def test_api_chat_routes_images_to_vision_and_never_to_text_provider(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Blue square."}},
        )

    config = build_settings(tmp_path)
    provider = FakeProvider()
    store = FakeStore()
    app = create_app(
        config=config,
        provider=provider,
        store=store,
        vision=ImageAnalyzer(config, transport=httpx.MockTransport(handler)),
    )
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "message": "What is this?",
            "conversation_id": "vision-1",
            "skill": "vision",
            "image_data_url": data_url(png_bytes()),
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "qwen2.5vl:7b"
    assert response.json()["answer"] == "Blue square."
    assert provider.calls == []
    assert store.saved == [
        ("vision-1", "What is this?", "Blue square.", "vision")
    ]


def test_chat_rejects_invalid_images_without_a_fake_answer(tmp_path: Path) -> None:
    provider = FakeProvider()
    store = FakeStore()
    app = create_app(
        config=build_settings(tmp_path),
        provider=provider,
        store=store,
        vision=ImageAnalyzer(build_settings(tmp_path)),
    )
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "message": "Read this",
            "image_data_url": "data:image/png;base64,AAAA",
        },
    )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_image"
    assert provider.calls == []
    assert store.saved == []


def test_image_api_reports_clear_timeout_without_fake_answer(
    tmp_path: Path,
) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)
        return httpx.Response(
            200,
            content=b'{"message":{"content":"too late"},"done":true}\n',
        )

    config = build_settings(tmp_path, vision_timeout_seconds=0.02)
    app = create_app(
        config=config,
        provider=FakeProvider(),
        store=FakeStore(),
        vision=ImageAnalyzer(config, transport=httpx.MockTransport(handler)),
    )
    client = TestClient(app)

    response = client.post(
        "/images/analyze",
        json={"image_data_url": data_url(png_bytes()), "prompt": "Describe"},
    )

    assert calls == 1
    assert response.status_code == 502
    body = response.json()
    assert body["detail"]["code"] == "vision_model_unavailable"
    assert "timed out after" in body["detail"]["message"]
    assert "still loading" not in body["detail"]["message"]
    assert "VISION_TIMEOUT_SECONDS" in body["detail"]["message"]


def test_image_api_analyze_generate_status_and_safe_download(
    tmp_path: Path,
) -> None:
    def vision_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Detected one button."}},
        )

    config = build_settings(tmp_path)
    app = create_app(
        config=config,
        provider=FakeProvider(),
        store=FakeStore(),
        vision=ImageAnalyzer(config, transport=httpx.MockTransport(vision_handler)),
        image_generator=ImageGenerator(
            config,
            transport=httpx.MockTransport(comfy_handler()),
            poll_interval=0,
        ),
    )
    client = TestClient(app)

    analyzed = client.post(
        "/images/analyze",
        json={
            "image_data_url": data_url(png_bytes()),
            "prompt": "What is in this UI?",
            "mode": "ui_review",
        },
    )
    assert analyzed.status_code == 200, analyzed.text
    assert analyzed.json()["answer"] == "Detected one button."
    assert analyzed.json()["image"]["width"] == 8

    generated = client.post(
        "/images/generate",
        json={"prompt": "A lake", "width": 512, "height": 512, "seed": 7},
    )
    assert generated.status_code == 200, generated.text
    body = generated.json()
    assert body["seed"] == 7
    downloaded = client.get(f"/images/generated/{body['image_id']}")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "image/png"
    assert downloaded.content == png_bytes(width=32, height=24)
    assert client.get("/images/generated/not-a-valid-id").status_code == 404


def test_image_api_reports_unconfigured_services_without_fake_results(
    tmp_path: Path,
) -> None:
    config = build_settings(
        tmp_path,
        vision_base_url="",
        model_base_url="",
        image_generation_provider="none",
    )
    app = create_app(
        config=config,
        provider=FakeProvider(),
        store=FakeStore(),
    )
    client = TestClient(app)

    analyzed = client.post(
        "/images/analyze",
        json={"image_data_url": data_url(png_bytes())},
    )
    generated = client.post("/images/generate", json={"prompt": "test"})
    status = client.get("/image-status")

    assert analyzed.status_code == 503
    assert analyzed.json()["detail"]["code"] == "vision_model_not_configured"
    assert generated.status_code == 503
    assert generated.json()["detail"]["code"] == "image_generation_not_configured"
    assert status.status_code == 200
    assert status.json()["vision"]["status"] == "not_configured"
    assert status.json()["generation"]["status"] == "disabled"
