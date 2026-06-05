"""Frame sampling strategies for comparing limited visual inputs."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from home_service_action_verifier.schemas import FrameInfo
from home_service_action_verifier.video_io import extract_frames_by_indices, get_video_metadata, open_video_capture, resize_frame


def _default_output_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="privacy_vlm_frames_"))


def _unique_sorted(indices: Iterable[int], total_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    return sorted({min(max(int(index), 0), total_frames - 1) for index in indices})


def _uniform_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0 or num_frames <= 0:
        return []
    count = min(num_frames, total_frames)
    if count == 1:
        return [total_frames // 2]
    return _unique_sorted(np.linspace(0, total_frames - 1, count).round().astype(int), total_frames)


def _motion_scores(video_path: str | Path, resize_width: int | None = None) -> dict[int, float]:
    metadata = get_video_metadata(video_path)
    if metadata.total_frames <= 0:
        return {}

    capture = open_video_capture(video_path)
    scores: dict[int, float] = {0: 0.0}
    previous_gray: np.ndarray | None = None
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            small = resize_frame(frame, min(resize_width or 320, 320))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if previous_gray is None:
                scores[index] = 0.0
            else:
                diff = cv2.absdiff(gray, previous_gray)
                scores[index] = float(np.mean(diff))
            previous_gray = gray
            index += 1
    finally:
        capture.release()

    return scores


def _select_motion_indices(
    scores: dict[int, float],
    total_frames: int,
    fps: float,
    num_frames: int,
    min_gap_seconds: float,
) -> list[int]:
    if total_frames <= 0 or num_frames <= 0:
        return []
    min_gap_frames = max(1, int(round(max(min_gap_seconds, 0.0) * max(fps, 1.0))))
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    selected: list[int] = []

    for index, _score in ranked:
        if len(selected) >= min(num_frames, total_frames):
            break
        if all(abs(index - existing) >= min_gap_frames for existing in selected):
            selected.append(index)

    if len(selected) < min(num_frames, total_frames):
        for index, _score in ranked:
            if len(selected) >= min(num_frames, total_frames):
                break
            if index not in selected:
                selected.append(index)

    return _unique_sorted(selected, total_frames)


def _save_selected(
    video_path: str | Path,
    indices: Iterable[int],
    resize_width: int | None,
    output_dir: str | Path | None,
    motion_scores: dict[int, float] | None = None,
) -> list[FrameInfo]:
    target_dir = Path(output_dir) if output_dir is not None else _default_output_dir()
    return extract_frames_by_indices(video_path, indices, target_dir, resize_width, motion_scores)


def uniform_sampling(
    video_path: str | Path,
    num_frames: int,
    resize_width: int | None = None,
    output_dir: str | Path | None = None,
) -> list[FrameInfo]:
    """Select up to N frames evenly across the full video."""

    metadata = get_video_metadata(video_path)
    indices = _uniform_indices(metadata.total_frames, num_frames)
    return _save_selected(video_path, indices, resize_width, output_dir)


def motion_sampling(
    video_path: str | Path,
    num_frames: int,
    resize_width: int | None = None,
    min_gap_seconds: float = 0.5,
    output_dir: str | Path | None = None,
) -> list[FrameInfo]:
    """Select frames with large adjacent-frame pixel changes."""

    metadata = get_video_metadata(video_path)
    scores = _motion_scores(video_path, resize_width)
    indices = _select_motion_indices(scores, metadata.total_frames, metadata.fps, num_frames, min_gap_seconds)
    return _save_selected(video_path, indices, resize_width, output_dir, scores)


def hybrid_sampling(
    video_path: str | Path,
    num_frames: int,
    resize_width: int | None = None,
    min_gap_seconds: float = 0.5,
    output_dir: str | Path | None = None,
) -> list[FrameInfo]:
    """Combine global context from uniform sampling with high-motion moments."""

    metadata = get_video_metadata(video_path)
    if metadata.total_frames <= 0 or num_frames <= 0:
        return []

    uniform_count = int(math.ceil(num_frames / 2))
    motion_count = max(num_frames - uniform_count, 0)
    uniform_indices = _uniform_indices(metadata.total_frames, uniform_count)
    scores = _motion_scores(video_path, resize_width)
    motion_indices = _select_motion_indices(
        scores,
        metadata.total_frames,
        metadata.fps,
        motion_count,
        min_gap_seconds,
    )

    combined = _unique_sorted([*uniform_indices, *motion_indices], metadata.total_frames)
    if len(combined) < min(num_frames, metadata.total_frames):
        fill = _uniform_indices(metadata.total_frames, num_frames)
        combined = _unique_sorted([*combined, *fill], metadata.total_frames)
    return _save_selected(video_path, combined[:num_frames], resize_width, output_dir, scores)


def event_window_sampling(
    video_path: str | Path,
    num_frames: int,
    resize_width: int | None = None,
    min_gap_seconds: float = 1.0,
    before_seconds: float = 0.5,
    after_seconds: float = 0.5,
    output_dir: str | Path | None = None,
) -> list[FrameInfo]:
    """Keep before/during/after context around high-motion frames."""

    metadata = get_video_metadata(video_path)
    if metadata.total_frames <= 0 or num_frames <= 0:
        return []

    scores = _motion_scores(video_path, resize_width)
    anchor_count = max(1, min(max(num_frames // 3, 1), metadata.total_frames))
    anchors = _select_motion_indices(scores, metadata.total_frames, metadata.fps, anchor_count, min_gap_seconds)
    before_gap = max(1, int(round(before_seconds * max(metadata.fps, 1.0))))
    after_gap = max(1, int(round(after_seconds * max(metadata.fps, 1.0))))

    indices: list[int] = []
    for anchor in anchors:
        for candidate in (anchor - before_gap, anchor, anchor + after_gap):
            if candidate not in indices:
                indices.append(candidate)

    indices = _unique_sorted(indices, metadata.total_frames)
    if len(indices) < min(num_frames, metadata.total_frames):
        fill = _uniform_indices(metadata.total_frames, num_frames)
        indices = _unique_sorted([*indices, *fill], metadata.total_frames)
    if len(indices) > num_frames:
        anchor_set = set(anchors)

        def rank(index: int) -> tuple[int, int]:
            nearest_anchor_distance = min(abs(index - anchor) for anchor in anchor_set) if anchor_set else index
            return nearest_anchor_distance, index

        indices = sorted(sorted(indices, key=rank)[:num_frames])
    return _save_selected(video_path, indices, resize_width, output_dir, scores)
