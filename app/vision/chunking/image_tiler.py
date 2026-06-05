"""
Image chunking (tiling) and video frame extraction utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import cv2
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("image_tiler")


def tile_image(
    img: Image.Image,
    tile_size: int | None = None,
    overlap: int = 64
) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    """
    Split a PIL Image into a list of grid tiles.
    Returns:
        List of tuples: (tile_image, bounding_box_as_ltrb)
        where bounding_box_as_ltrb is (left, top, right, bottom) pixel coordinates.
    """
    settings = get_settings()
    tile_size = tile_size or settings.tile_size
    max_tiles = settings.max_image_tiles

    width, height = img.size
    tiles = []

    # If the image is smaller than the tile size, return the image itself as a single tile
    if width <= tile_size and height <= tile_size:
        return [(img, (0, 0, width, height))]

    stride = tile_size - overlap
    if stride <= 0:
        stride = tile_size // 2

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Check edge bounds
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            # Adjust start coordinates if we're at the bottom/right edges
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)

            box = (x_start, y_start, x_end, y_end)
            tile = img.crop(box)
            tiles.append((tile, box))

            # Guard against too many tiles to control compute cost
            if len(tiles) >= max_tiles:
                logger.warning(
                    "Maximum image tiles reached",
                    max_tiles=max_tiles,
                    image_size=(width, height),
                )
                return tiles

    return tiles


def extract_frames_from_video(
    video_path: str | Path,
    fps_interval: float = 1.0,
    max_frames: int = 20
) -> List[Tuple[Image.Image, float]]:
    """
    Extract keyframes from a video file.
    Args:
        video_path: Local path to the video file.
        fps_interval: Frequency of extraction in seconds (e.g. 1 frame every 1.0s).
        max_frames: Upper limit on number of frames to return.
    Returns:
        List of tuples: (PIL_image, timestamp_seconds)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        fps = 25.0  # Safe default if FPS is metadata is missing

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    logger.info("Extracting frames from video", path=str(video_path), fps=fps, duration=duration)

    frame_step = int(fps * fps_interval)
    if frame_step <= 0:
        frame_step = 1

    extracted = []
    frame_idx = 0

    while cap.isOpened() and len(extracted) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Convert BGR OpenCV frame to RGB PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            timestamp = frame_idx / fps
            extracted.append((pil_img, timestamp))

        frame_idx += 1

    cap.release()
    logger.info("Video frame extraction completed", frames_extracted=len(extracted))
    return extracted
