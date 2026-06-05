"""
Image ingestion, format verification, and preprocessing.
"""

from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def validate_image_format(content_type: str) -> bool:
    """Return True if content_type is an allowed image format."""
    allowed = {"image/jpeg", "image/png", "image/webp"}
    return content_type.lower() in allowed


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load PIL Image from raw bytes."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # Verify it's a valid image
        # PIL Image.open needs to be reopened after verify()
        img = Image.open(io.BytesIO(data))
        # Convert to RGB to discard alpha channel if present
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.exception("Failed to load image from bytes")
        raise ValueError(f"Invalid image data: {str(e)}")


def load_image_from_url(url: str, timeout_seconds: int = 10) -> Image.Image:
    """Fetch image from a remote URL and return a PIL Image."""
    try:
        logger.debug("Fetching image from URL: %s", url)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GlobalFreight-CV-RAG/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            data = response.read()
        return load_image_from_bytes(data)
    except Exception as e:
        logger.exception("Failed to load image from URL", url=url)
        raise ValueError(f"Failed to fetch image from URL: {str(e)}")


def load_image_from_path(path: Union[str, Path]) -> Image.Image:
    """Load image from local file path."""
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            data = f.read()
        return load_image_from_bytes(data)
    except Exception as e:
        logger.exception("Failed to load image from file path", path=str(path))
        raise ValueError(f"Failed to load image: {str(e)}")


def preprocess_image(img: Image.Image, max_dim: int | None = None) -> Image.Image:
    """
    Resize image if its dimensions exceed max_dim (preserving aspect ratio).
    Ensures optimal token and compute efficiency.
    """
    settings = get_settings()
    max_dim = max_dim or settings.max_image_dimension

    width, height = img.size
    if width > max_dim or height > max_dim:
        scale = max_dim / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        logger.info(
            "Resizing image to fit dimensions",
            original_size=(width, height),
            new_size=(new_width, new_height),
        )
        # Using Resampling.LANCZOS for high quality resizing
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img


def to_opencv_format(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to OpenCV (BGR) format."""
    # Convert PIL Image to RGB numpy array
    rgb_arr = np.array(img.convert("RGB"))
    # Convert RGB to BGR for OpenCV
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    return bgr_arr


def to_pil_format(cv_img: np.ndarray) -> Image.Image:
    """Convert an OpenCV (BGR) image to PIL Image."""
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)
