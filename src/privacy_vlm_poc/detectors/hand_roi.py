"""Manual and event-linked hand/object ROI extraction helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from privacy_vlm_poc.schemas import EventToken, ROI
from privacy_vlm_poc.video_io import get_video_metadata, open_video_capture


def crop_roi_from_frame(frame: np.ndarray, roi: ROI) -> np.ndarray:
    height, width = frame.shape[:2]
    clipped = roi.clipped(width, height)
    return frame[clipped.y1 : clipped.y2, clipped.x1 : clipped.x2].copy()


def save_event_roi_frames(
    video_path: str | Path,
    event: EventToken,
    roi: ROI,
    output_dir: str | Path,
    num_frames: int = 4,
) -> list[Path]:
    if num_frames <= 0:
        msg = "num_frames must be positive"
        raise ValueError(msg)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = get_video_metadata(video_path)
    start_frame = max(0, int(round(event.start_sec * metadata.fps)))
    end_frame = min(max(metadata.total_frames - 1, 0), int(round(event.end_sec * metadata.fps)))
    if end_frame < start_frame:
        end_frame = start_frame
    if num_frames == 1:
        indices = [start_frame]
    else:
        span = max(end_frame - start_frame, 1)
        indices = sorted({start_frame + round(span * i / (num_frames - 1)) for i in range(num_frames)})

    capture = open_video_capture(video_path)
    saved: list[Path] = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            cropped = crop_roi_from_frame(frame, roi)
            path = output_path / f"{event.event_id}_frame_{index:06d}.jpg"
            if not cv2.imwrite(str(path), cropped):
                msg = f"Failed to write ROI frame: {path}"
                raise ValueError(msg)
            saved.append(path)
    finally:
        capture.release()
    return saved
