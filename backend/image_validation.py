from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

_ALLOWED_MEDIA_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}
_DECLARED_ALIASES = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/webp": "image/webp",
}
_SIGNATURES = {
    "PNG": b"\x89PNG\r\n\x1a\n",
    "JPEG": b"\xff\xd8\xff",
    "WEBP": b"RIFF",
}


class ImageValidationError(ValueError):
    def __init__(self, message: str, *, code: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ValidatedImage:
    __slots__ = (
        "media_type",
        "image_format",
        "width",
        "height",
        "size_bytes",
        "base64_data",
    )

    def __init__(
        self,
        *,
        media_type: str,
        image_format: str,
        width: int,
        height: int,
        size_bytes: int,
        base64_data: str = "",
    ) -> None:
        self.media_type = media_type
        self.image_format = image_format
        self.width = width
        self.height = height
        self.size_bytes = size_bytes
        self.base64_data = base64_data

    def as_dict(self) -> dict[str, object]:
        return {
            "media_type": self.media_type,
            "format": self.image_format,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
        }


def validate_image_data_url(
    value: str,
    *,
    max_bytes: int,
    max_pixels: int,
    max_dimension: int,
) -> ValidatedImage:
    if not isinstance(value, str) or not value.strip():
        raise ImageValidationError(
            "An image data URL is required.",
            code="invalid_image",
            status_code=400,
        )
    value = value.strip()
    header, separator, encoded = value.partition(",")
    if not separator or not header.lower().startswith("data:"):
        raise ImageValidationError(
            "The image must be supplied as a base64 data URL.",
            code="invalid_image",
            status_code=400,
        )

    parts = header[5:].split(";")
    declared = parts[0].strip().lower()
    parameters = {part.strip().lower() for part in parts[1:]}
    media_type = _DECLARED_ALIASES.get(declared)
    if media_type is None:
        raise ImageValidationError(
            "Unsupported image type. Use PNG, JPEG, or WebP.",
            code="unsupported_image",
            status_code=415,
        )
    if "base64" not in parameters:
        raise ImageValidationError(
            "The image data URL must use base64 encoding.",
            code="invalid_image",
            status_code=400,
        )

    maximum_encoded = ((max_bytes + 2) // 3) * 4 + 128
    if len(encoded) > maximum_encoded:
        raise ImageValidationError(
            f"Image exceeds the {max_bytes / (1024 * 1024):g} MB upload limit.",
            code="image_too_large",
            status_code=413,
        )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageValidationError(
            "The image data is not valid base64.",
            code="invalid_image",
            status_code=400,
        ) from exc

    validated = inspect_image_bytes(
        raw,
        max_bytes=max_bytes,
        max_pixels=max_pixels,
        max_dimension=max_dimension,
        declared_media_type=media_type,
    )
    return ValidatedImage(
        media_type=validated.media_type,
        image_format=validated.image_format,
        width=validated.width,
        height=validated.height,
        size_bytes=validated.size_bytes,
        base64_data=base64.b64encode(raw).decode("ascii"),
    )


def inspect_image_bytes(
    raw: bytes,
    *,
    max_bytes: int,
    max_pixels: int,
    max_dimension: int,
    declared_media_type: str | None = None,
) -> ValidatedImage:
    if not raw:
        raise ImageValidationError(
            "The image is empty.",
            code="invalid_image",
            status_code=400,
        )
    if len(raw) > max_bytes:
        raise ImageValidationError(
            f"Image exceeds the {max_bytes / (1024 * 1024):g} MB size limit.",
            code="image_too_large",
            status_code=413,
        )

    image_format = _detect_signature(raw)
    if image_format is None:
        raise ImageValidationError(
            "Unsupported or malformed image. Use PNG, JPEG, or WebP.",
            code="unsupported_image",
            status_code=415,
        )

    try:
        with Image.open(io.BytesIO(raw)) as image:
            actual_format = (image.format or "").upper()
            width, height = image.size
            image.verify()
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            "The image dimensions are too large to process safely.",
            code="image_dimensions_too_large",
            status_code=422,
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(
            "The image file is malformed or unreadable.",
            code="invalid_image",
            status_code=400,
        ) from exc

    media_type = _ALLOWED_MEDIA_TYPES.get(actual_format)
    if media_type is None or actual_format != image_format:
        raise ImageValidationError(
            "Unsupported or malformed image. Use PNG, JPEG, or WebP.",
            code="unsupported_image",
            status_code=415,
        )
    if declared_media_type is not None and declared_media_type != media_type:
        raise ImageValidationError(
            "The image content does not match its declared media type.",
            code="unsupported_image",
            status_code=415,
        )
    if (
        width <= 0
        or height <= 0
        or width > max_dimension
        or height > max_dimension
        or width * height > max_pixels
    ):
        raise ImageValidationError(
            f"Image dimensions must not exceed {max_dimension}px per side or "
            f"{max_pixels:,} total pixels.",
            code="image_dimensions_too_large",
            status_code=422,
        )

    return ValidatedImage(
        media_type=media_type,
        image_format=actual_format,
        width=width,
        height=height,
        size_bytes=len(raw),
    )


def _detect_signature(raw: bytes) -> str | None:
    if raw.startswith(_SIGNATURES["PNG"]):
        return "PNG"
    if raw.startswith(_SIGNATURES["JPEG"]):
        return "JPEG"
    if len(raw) >= 12 and raw.startswith(_SIGNATURES["WEBP"]) and raw[8:12] == b"WEBP":
        return "WEBP"
    return None
