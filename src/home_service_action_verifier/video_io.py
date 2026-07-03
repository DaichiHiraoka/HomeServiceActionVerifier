from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np


def iter_video_frames(path: str | Path) -> Iterator[tuple[int, float, np.ndarray]]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            yield frame, frame / fps, image
            frame += 1
    finally:
        cap.release()


def get_video_fps(path: str | Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    finally:
        cap.release()
    return float(fps)


def expand_video_inputs(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        suffixes = {".mp4", ".mov", ".avi", ".mkv"}
        return sorted(child for child in p.iterdir() if child.suffix.lower() in suffixes)
    return [p]
