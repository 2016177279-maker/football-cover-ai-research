"""Deterministic, lightweight thumbnail visual features (v1).

All intensity and RGB values use the 0..255 scale. Ratios are in 0..1.
This module deliberately contains no learned models, OCR, API calls, or
engagement/outcome analysis.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, UnidentifiedImageError


FEATURE_NAMES: Final[tuple[str, ...]] = (
    "image_width", "image_height", "aspect_ratio",
    "brightness_mean", "brightness_std",
    "saturation_mean", "saturation_std", "colorfulness", "contrast",
    "sharpness_laplacian", "edge_density", "entropy",
    "center_brightness", "border_brightness", "center_edge_density",
    "dominant_color_r", "dominant_color_g", "dominant_color_b",
    "warm_color_ratio", "dark_pixel_ratio", "bright_pixel_ratio",
)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    ).astype(np.float32)


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """Return gradient magnitudes using deterministic central differences."""
    gy, gx = np.gradient(gray)
    return np.hypot(gx, gy)


def _regions(height: int, width: int) -> tuple[tuple[slice, slice], np.ndarray]:
    # Center is the middle 50% in both dimensions; border is the outer 10%.
    y0, y1 = height // 4, height - height // 4
    x0, x1 = width // 4, width - width // 4
    border_y = max(1, int(round(height * 0.10)))
    border_x = max(1, int(round(width * 0.10)))
    mask = np.ones((height, width), dtype=bool)
    if height > 2 * border_y and width > 2 * border_x:
        mask[border_y:-border_y, border_x:-border_x] = False
    return (slice(y0, y1), slice(x0, x1)), mask


def extract_visual_features(image_path: str | Path) -> dict[str, float | int]:
    """Extract v1 features from one image; raises for missing/corrupt images."""
    path = Path(image_path)
    with Image.open(path) as image:
        image.load()
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32)

    height, width = rgb.shape[:2]
    if width == 0 or height == 0:
        raise ValueError("image has zero width or height")

    gray = _luminance(rgb)
    saturation = hsv[..., 1]
    edge = _edge_map(gray)

    # Four-neighbour discrete Laplacian, evaluated away from the outer pixel.
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1] + padded[2:, 1:-1]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
        - 4.0 * gray
    )

    center, border_mask = _regions(height, width)
    hist = np.bincount(np.clip(np.rint(gray), 0, 255).astype(np.uint8).ravel(), minlength=256)
    probabilities = hist[hist > 0].astype(np.float64) / gray.size
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = math.hypot(float(rg.std()), float(yb.std())) + 0.3 * math.hypot(
        float(rg.mean()), float(yb.mean())
    )

    hue_degrees = hsv[..., 0] * (360.0 / 255.0)
    chromatic = saturation >= 25.5  # exclude nearly gray pixels (10% saturation)
    warm = chromatic & ((hue_degrees <= 60.0) | (hue_degrees >= 300.0))
    dominant = np.median(rgb.reshape(-1, 3), axis=0)
    result: dict[str, float | int] = {
        "image_width": int(width),
        "image_height": int(height),
        "aspect_ratio": float(width / height),
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "colorfulness": float(colorfulness),
        "contrast": float((np.percentile(gray, 95) - np.percentile(gray, 5)) / 255.0),
        "sharpness_laplacian": float(laplacian.var()),
        "edge_density": float((edge >= 30.0).mean()),
        "entropy": entropy,
        "center_brightness": float(gray[center].mean()),
        "border_brightness": float(gray[border_mask].mean()),
        "center_edge_density": float((edge[center] >= 30.0).mean()),
        "dominant_color_r": float(dominant[0]),
        "dominant_color_g": float(dominant[1]),
        "dominant_color_b": float(dominant[2]),
        "warm_color_ratio": float(warm.mean()),
        "dark_pixel_ratio": float((gray < 51.0).mean()),
        "bright_pixel_ratio": float((gray > 204.0).mean()),
    }
    if set(result) != set(FEATURE_NAMES):
        raise RuntimeError("feature schema mismatch")
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("non-finite feature value produced")
    return result


def classify_extraction_error(path: str | Path, error: Exception) -> str:
    """Return a stable status label suitable for batch CSV output."""
    if not Path(path).is_file():
        return "missing"
    if isinstance(error, (UnidentifiedImageError, OSError, ValueError)):
        return "corrupt_or_unsupported"
    return "error"
