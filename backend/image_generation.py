from __future__ import annotations

import asyncio
import os
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .image_validation import ImageValidationError, ValidatedImage, inspect_image_bytes

_EXTENSIONS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}
_GENERATED_ID = re.compile(r"^[0-9a-f]{32}\.(?:png|jpg|webp)$")
_MEDIA_TYPES = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}


class ImageGenerationError(RuntimeError):
    code = "image_generation_failed"
    status_code = 502


class ImageGenerationNotConfigured(ImageGenerationError):
    code = "image_generation_not_configured"
    status_code = 503


class ComfyUIUnavailable(ImageGenerationError):
    code = "comfyui_unavailable"
    status_code = 503


class ComfyUIWorkflowError(ImageGenerationError):
    code = "comfyui_workflow_failed"
    status_code = 502


@dataclass(frozen=True)
class ComfyGenerationRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg: float = 7.0
    seed: int | None = None


@dataclass(frozen=True)
class GeneratedImage:
    image_id: str
    media_type: str
    image_format: str
    width: int
    height: int
    size_bytes: int
    seed: int
    checkpoint: str
    created_at: str


class ImageOutputStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def save(self, raw: bytes, image: ValidatedImage) -> tuple[str, str]:
        extension = _EXTENSIONS.get(image.image_format)
        if extension is None:
            raise ImageGenerationError("Unsupported generated image format.")
        self.root.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            image_id = f"{uuid.uuid4().hex}.{extension}"
            path = self.root / image_id
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_BINARY", 0)
            try:
                descriptor = os.open(path, flags, 0o600)
            except FileExistsError:
                continue
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                path.unlink(missing_ok=True)
                raise
            return image_id, _MEDIA_TYPES[extension]
        raise ImageGenerationError("Could not allocate a unique image filename.")

    def get(self, image_id: str) -> tuple[Path, str] | None:
        if not _GENERATED_ID.fullmatch(image_id):
            return None
        candidate = self.root / image_id
        try:
            if candidate.is_symlink() or not candidate.is_file():
                return None
            resolved = candidate.resolve(strict=True)
        except OSError:
            return None
        if resolved.parent != self.root:
            return None
        media_type = _MEDIA_TYPES[image_id.rsplit(".", 1)[1]]
        return resolved, media_type


class ImageGenerator:
    def __init__(
        self,
        config: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        poll_interval: float = 0.75,
    ) -> None:
        self.config = config
        self._transport = transport
        self._poll_interval = poll_interval
        self.outputs = ImageOutputStore(config.image_output_dir)

    @property
    def configured(self) -> bool:
        return self.config.image_generation_configured

    @property
    def enabled(self) -> bool:
        return self.config.image_generation_provider == "comfyui"

    @property
    def base_url(self) -> str:
        return self.config.comfyui_base_url

    def _timeout(self, total: float | None = None) -> httpx.Timeout:
        seconds = total or self.config.comfyui_timeout_seconds
        return httpx.Timeout(seconds, connect=min(10.0, seconds))

    async def probe(self) -> dict[str, object]:
        info: dict[str, object] = {
            "enabled": self.enabled,
            "configured": self.configured,
            "status": "disabled" if not self.enabled else "not_configured",
            "reachable": False,
            "checkpoint": self.config.comfyui_checkpoint or None,
            "checkpoint_available": None,
            "base_url": self.base_url or None,
            "generation_tested": False,
            "detail": "",
        }
        if not self.enabled:
            info["detail"] = (
                "Image generation is disabled. Set IMAGE_GENERATION_PROVIDER=comfyui "
                "in .env to use a local ComfyUI server."
            )
            return info
        if not self.config.comfyui_base_url or not self.config.comfyui_checkpoint:
            info["detail"] = (
                "Set COMFYUI_BASE_URL and COMFYUI_CHECKPOINT in .env, then restart "
                "the API."
            )
            return info

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout(min(4.0, self.config.comfyui_timeout_seconds)),
                transport=self._transport,
            ) as client:
                response = await client.get(f"{self.base_url}/system_stats")
        except httpx.HTTPError as exc:
            info.update(
                status="unreachable",
                detail=self._connection_detail(exc),
            )
            return info
        info["reachable"] = True
        if response.status_code >= 400:
            info.update(
                status="error",
                detail=f"ComfyUI returned HTTP {response.status_code}.",
            )
            return info
        try:
            response.json()
        except ValueError:
            info.update(
                status="error",
                detail="ComfyUI returned invalid system information.",
            )
            return info

        checkpoint_available = await self._check_checkpoint()
        info["checkpoint_available"] = checkpoint_available
        if checkpoint_available is False:
            info.update(
                status="checkpoint_missing",
                detail=(
                    f'ComfyUI does not list the checkpoint "{self.config.comfyui_checkpoint}". '
                    "Set COMFYUI_CHECKPOINT to an installed checkpoint filename."
                ),
            )
            return info
        info.update(
            status="available",
            detail=(
                "ComfyUI is reachable"
                + (" and the checkpoint is listed." if checkpoint_available else ".")
                + " Generation itself has not been tested."
            ),
        )
        return info

    async def _check_checkpoint(self) -> bool | None:
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout(min(5.0, self.config.comfyui_timeout_seconds)),
                transport=self._transport,
            ) as client:
                response = await client.get(
                    f"{self.base_url}/object_info/CheckpointLoaderSimple"
                )
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            body = response.json()
            node = body.get("CheckpointLoaderSimple")
            required = node.get("input", {}).get("required", {})
            checkpoint_input = required.get("ckpt_name")
            names = checkpoint_input[0] if isinstance(checkpoint_input, list) else None
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        if not isinstance(names, list):
            return None
        return self.config.comfyui_checkpoint in names

    def _connection_detail(self, exc: httpx.HTTPError) -> str:
        if isinstance(exc, httpx.TimeoutException):
            return "The ComfyUI request timed out."
        if isinstance(exc, httpx.ConnectError):
            return (
                f"Could not connect to ComfyUI at {self.base_url}. Start ComfyUI "
                "or update COMFYUI_BASE_URL."
            )
        return f"Could not communicate with ComfyUI at {self.base_url}."

    async def generate(self, request: ComfyGenerationRequest) -> GeneratedImage:
        if not self.enabled:
            raise ImageGenerationNotConfigured(
                "Image generation is disabled. Set IMAGE_GENERATION_PROVIDER=comfyui "
                "and configure COMFYUI_BASE_URL."
            )
        if not self.configured:
            raise ImageGenerationNotConfigured(
                "ComfyUI is not fully configured. Set COMFYUI_BASE_URL and "
                "COMFYUI_CHECKPOINT in .env, then restart the API."
            )

        seed = request.seed if request.seed is not None else secrets.randbelow(2**32)
        workflow = self._workflow(request, seed)
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout(),
                transport=self._transport,
            ) as client:
                prompt_id = await self._queue_prompt(client, workflow)
                output = await self._wait_for_output(client, prompt_id)
                raw = await self._download_output(client, output)
        except httpx.HTTPError as exc:
            raise ComfyUIUnavailable(self._connection_detail(exc)) from exc

        try:
            validated = inspect_image_bytes(
                raw,
                max_bytes=self.config.image_generation_max_output_bytes,
                max_pixels=self.config.image_max_pixels,
                max_dimension=self.config.image_max_dimension,
            )
        except ImageValidationError as exc:
            raise ComfyUIWorkflowError(
                f"ComfyUI returned an invalid image: {exc}"
            ) from exc
        try:
            image_id, media_type = self.outputs.save(raw, validated)
        except OSError as exc:
            raise ImageGenerationError(
                "The generated image could not be saved. Check that IMAGE_OUTPUT_DIR "
                "is writable."
            ) from exc
        return GeneratedImage(
            image_id=image_id,
            media_type=media_type,
            image_format=validated.image_format,
            width=validated.width,
            height=validated.height,
            size_bytes=validated.size_bytes,
            seed=seed,
            checkpoint=self.config.comfyui_checkpoint,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def _workflow(
        self, request: ComfyGenerationRequest, seed: int
    ) -> dict[str, dict[str, Any]]:
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": request.steps,
                    "cfg": request.cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": self.config.comfyui_checkpoint},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": request.width,
                    "height": request.height,
                    "batch_size": 1,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": request.prompt, "clip": ["4", 1]},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": request.negative_prompt, "clip": ["4", 1]},
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "vups_gen4", "images": ["8", 0]},
            },
        }

    async def _queue_prompt(
        self, client: httpx.AsyncClient, workflow: dict[str, dict[str, Any]]
    ) -> str:
        try:
            response = await client.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow, "client_id": uuid.uuid4().hex},
                timeout=self._timeout(min(30.0, self.config.comfyui_timeout_seconds)),
            )
        except httpx.HTTPError as exc:
            raise ComfyUIUnavailable(self._connection_detail(exc)) from exc
        if response.status_code >= 400:
            raise ComfyUIWorkflowError(self._workflow_error(response))
        try:
            body = response.json()
        except ValueError as exc:
            raise ComfyUIWorkflowError(
                "ComfyUI returned invalid JSON when queuing the workflow."
            ) from exc
        prompt_id = body.get("prompt_id") if isinstance(body, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ComfyUIWorkflowError(
                "ComfyUI did not return a prompt identifier."
            )
        return prompt_id

    def _workflow_error(self, response: httpx.Response) -> str:
        detail = _response_detail(response)
        suffix = f" Detail: {detail}" if detail else ""
        return (
            f"ComfyUI rejected the generation workflow (HTTP {response.status_code}). "
            "Check COMFYUI_CHECKPOINT and the ComfyUI server logs."
            f"{suffix}"
        )

    async def _wait_for_output(
        self, client: httpx.AsyncClient, prompt_id: str
    ) -> dict[str, str]:
        deadline = time.monotonic() + self.config.comfyui_timeout_seconds
        while time.monotonic() < deadline:
            try:
                response = await client.get(
                    f"{self.base_url}/history/{prompt_id}",
                    timeout=self._timeout(min(20.0, self.config.comfyui_timeout_seconds)),
                )
            except httpx.HTTPError as exc:
                raise ComfyUIUnavailable(self._connection_detail(exc)) from exc
            if response.status_code >= 400:
                raise ComfyUIWorkflowError(
                    f"ComfyUI returned HTTP {response.status_code} while checking "
                    "generation progress."
                )
            try:
                body = response.json()
            except ValueError as exc:
                raise ComfyUIWorkflowError(
                    "ComfyUI returned invalid JSON while checking generation progress."
                ) from exc
            entry = body.get(prompt_id) if isinstance(body, dict) else None
            if isinstance(entry, dict):
                status = entry.get("status")
                if isinstance(status, dict) and status.get("status_str") == "error":
                    raise ComfyUIWorkflowError(self._execution_error())
                output = self._find_output(entry.get("outputs"))
                if output is not None:
                    return output
            await asyncio.sleep(self._poll_interval)
        raise ComfyUIUnavailable(
            "ComfyUI did not finish the image within "
            f"{self.config.comfyui_timeout_seconds:g} seconds."
        )

    def _find_output(self, outputs: Any) -> dict[str, str] | None:
        if not isinstance(outputs, dict):
            return None
        for node_output in outputs.values():
            images = node_output.get("images") if isinstance(node_output, dict) else None
            if not isinstance(images, list):
                continue
            for image in images:
                if not isinstance(image, dict):
                    continue
                filename = image.get("filename")
                subfolder = image.get("subfolder", "")
                image_type = image.get("type", "output")
                if (
                    isinstance(filename, str)
                    and filename
                    and isinstance(subfolder, str)
                    and isinstance(image_type, str)
                ):
                    return {
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": image_type,
                    }
        return None

    def _execution_error(self) -> str:
        return (
            "ComfyUI reported a failed generation workflow. Check its server logs, "
            "the configured checkpoint, and available memory."
        )

    async def _download_output(
        self, client: httpx.AsyncClient, output: dict[str, str]
    ) -> bytes:
        try:
            async with client.stream(
                "GET",
                f"{self.base_url}/view",
                params=output,
                timeout=self._timeout(min(60.0, self.config.comfyui_timeout_seconds)),
            ) as response:
                if response.status_code >= 400:
                    raise ComfyUIWorkflowError(
                        f"ComfyUI returned HTTP {response.status_code} for the "
                        "generated image."
                    )
                content_length = response.headers.get("content-length")
                if content_length and content_length.isdigit():
                    if int(content_length) > self.config.image_generation_max_output_bytes:
                        raise ComfyUIWorkflowError(
                            "ComfyUI returned an image larger than the configured "
                            "output limit."
                        )
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > self.config.image_generation_max_output_bytes:
                        raise ComfyUIWorkflowError(
                            "ComfyUI returned an image larger than the configured "
                            "output limit."
                        )
        except httpx.HTTPError as exc:
            raise ComfyUIUnavailable(self._connection_detail(exc)) from exc
        if not content:
            raise ComfyUIWorkflowError("ComfyUI returned an empty generated image.")
        return bytes(content)


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, str):
        detail = error
    elif isinstance(error, dict):
        detail = str(error.get("message") or error.get("detail") or "")
    else:
        detail = str(payload.get("message") or payload.get("detail") or "")
    if not detail and payload.get("node_errors"):
        detail = "One or more workflow nodes were rejected."
    return " ".join(detail.split())[:300]
