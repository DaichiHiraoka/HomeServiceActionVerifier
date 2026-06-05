from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from home_service_action_verifier.sampling import event_window_sampling, hybrid_sampling, uniform_sampling


def _make_test_video(path: Path, frames: int = 30, fps: int = 10) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert writer.isOpened()
    try:
        for index in range(frames):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            if index >= frames // 2:
                cv2.rectangle(frame, (50, 40), (105, 85), (255, 255, 255), thickness=-1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_uniform_sampling_returns_at_most_requested_frames(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path / "sample.mp4")
    frames = uniform_sampling(video, num_frames=8, output_dir=tmp_path / "uniform")
    assert len(frames) <= 8
    assert all(frame.path and frame.path.exists() for frame in frames)


def test_hybrid_sampling_removes_duplicate_frames(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path / "sample.mp4", frames=5)
    frames = hybrid_sampling(video, num_frames=8, output_dir=tmp_path / "hybrid")
    indices = [frame.frame_index for frame in frames]
    assert len(indices) == len(set(indices))
    assert len(indices) <= 5


def test_event_window_sampling_includes_before_and_after_frames(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path / "sample.mp4", frames=30, fps=10)
    frames = event_window_sampling(
        video,
        num_frames=5,
        before_seconds=0.2,
        after_seconds=0.2,
        output_dir=tmp_path / "event",
    )
    indices = [frame.frame_index for frame in frames]
    assert any(index < 15 for index in indices)
    assert any(index > 15 for index in indices)
