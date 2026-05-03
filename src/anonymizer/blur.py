from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from anonymizer.detectors.common import Detection


def decode_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def encode_jpeg(image_rgb: np.ndarray, quality: int = 92) -> bytes:
    image = Image.fromarray(image_rgb)
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def anonymize_image_bytes(
    image_bytes: bytes,
    detections: list[Detection],
    blur_kernel_ratio: float = 0.18,
    min_blur_kernel: int = 31,
) -> bytes:
    image_rgb = decode_image(image_bytes)
    anonymized = apply_blur(image_rgb, detections, blur_kernel_ratio, min_blur_kernel)
    return encode_jpeg(anonymized)


def apply_blur(
    image_rgb: np.ndarray,
    detections: list[Detection],
    blur_kernel_ratio: float,
    min_blur_kernel: int,
) -> np.ndarray:
    output = Image.fromarray(image_rgb)
    width, height = output.size

    for detection in detections:
        x1, y1, x2, y2 = detection.clamped(width=width, height=height)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = output.crop((x1, y1, x2, y2))
        radius = _blur_radius(max(x2 - x1, y2 - y1), blur_kernel_ratio, min_blur_kernel)
        output.paste(roi.filter(ImageFilter.GaussianBlur(radius=radius)), (x1, y1))

    return np.array(output)


def _blur_radius(size: int, ratio: float, minimum: int) -> float:
    return max(minimum, int(size * ratio)) / 3
