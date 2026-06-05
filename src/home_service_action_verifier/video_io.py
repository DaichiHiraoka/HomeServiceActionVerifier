"""OpenCV based video metadata and frame extraction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from home_service_action_verifier.schemas import FrameInfo, VideoMetadata

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


def validate_video_path(video_path: str | Path) -> Path:
    path = Path(video_path)
    if not path.exists():
        msg = f"Video file does not exist: {path}"
        raise FileNotFoundError(msg)
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        msg = f"Unsupported video extension: {path.suffix}. Supported: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
        raise ValueError(msg)
    return path


def open_video_capture(video_path: str | Path) -> cv2.VideoCapture:
    path = validate_video_path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        msg = f"Failed to open video: {path}"
        raise ValueError(msg)
    return capture


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    path = validate_video_path(video_path)
    capture = open_video_capture(path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = float(total_frames / fps) if fps > 0 else 0.0
        return VideoMetadata(
            video_path=path,
            fps=fps,
            duration=duration,
            total_frames=total_frames,
            width=width,
            height=height,
        )
    finally:
        capture.release()


def resize_frame(frame: np.ndarray, resize_width: int | None = None) -> np.ndarray:
    if resize_width is None or resize_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= resize_width:
        return frame
    scale = resize_width / width
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resize_width, resized_height), interpolation=cv2.INTER_AREA)


def get_frame(video_path: str | Path, frame_index: int, resize_width: int | None = None) -> np.ndarray:
    metadata = get_video_metadata(video_path)
    if metadata.total_frames <= 0:
        msg = f"Video contains no frames: {video_path}"
        raise ValueError(msg)
    index = min(max(int(frame_index), 0), metadata.total_frames - 1)
    capture = open_video_capture(video_path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            msg = f"Failed to read frame {index} from {video_path}"
            raise ValueError(msg)
        return resize_frame(frame, resize_width)
    finally:
        capture.release()


def save_frame(
    frame: np.ndarray,
    output_dir: str | Path,
    frame_index: int,
    timestamp: float,
    prefix: str = "frame",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{prefix}_{frame_index:06d}_{timestamp:.3f}s.jpg"
    ok = cv2.imwrite(str(file_path), frame)
    if not ok:
        msg = f"Failed to write frame image: {file_path}"
        raise ValueError(msg)
    return file_path


def extract_frames_by_indices(
    video_path: str | Path,
    indices: Iterable[int],
    output_dir: str | Path,
    resize_width: int | None = None,
    motion_scores: dict[int, float] | None = None,
) -> list[FrameInfo]:
    metadata = get_video_metadata(video_path)
    if metadata.fps <= 0:
        fps = 1.0
    else:
        fps = metadata.fps

    unique_indices = sorted({min(max(int(index), 0), max(metadata.total_frames - 1, 0)) for index in indices})
    frames: list[FrameInfo] = []
    for index in unique_indices:
        frame = get_frame(video_path, index, resize_width=resize_width)
        timestamp = index / fps
        path = save_frame(frame, output_dir, index, timestamp)
        frames.append(
            FrameInfo(
                frame_index=index,
                timestamp=timestamp,
                path=path,
                motion_score=(motion_scores or {}).get(index),
            )
        )
    return frames
