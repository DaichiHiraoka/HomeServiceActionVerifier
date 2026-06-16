"""Generate simple synthetic videos for local PoC verification."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
WIDTH = 640
HEIGHT = 360
FPS = 10
NUM_FRAMES = 80
OBJECT_SIZE = (68, 44)


def _interpolate(start: tuple[int, int], end: tuple[int, int], ratio: float) -> tuple[int, int]:
    x = int(round(start[0] + (end[0] - start[0]) * ratio))
    y = int(round(start[1] + (end[1] - start[1]) * ratio))
    return x, y


def _draw_scene(frame: np.ndarray, hand_x: int, object_pos: tuple[int, int] | None) -> None:
    cv2.rectangle(frame, (90, 185), (560, 315), (145, 145, 145), thickness=-1)
    cv2.rectangle(frame, (470, 220), (560, 300), (20, 20, 20), thickness=-1)

    hand_y = 165
    cv2.rectangle(frame, (hand_x, hand_y), (hand_x + 42, hand_y + 90), (210, 80, 40), thickness=-1)
    cv2.rectangle(frame, (hand_x + 30, hand_y + 70), (hand_x + 72, hand_y + 95), (210, 80, 40), thickness=-1)

    if object_pos is not None:
        ox, oy = object_pos
        ow, oh = OBJECT_SIZE
        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (20, 20, 230), thickness=-1)
        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (240, 240, 240), thickness=2)


def _write_video(path: Path, suspicious: bool) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        msg = f"Failed to create video writer: {path}"
        raise RuntimeError(msg)

    object_start = (205, 235)
    object_end = (485, 238)
    try:
        for index in range(NUM_FRAMES):
            frame = np.full((HEIGHT, WIDTH, 3), 180, dtype=np.uint8)
            if suspicious:
                hand_x = int(np.interp(index, [0, 30, NUM_FRAMES - 1], [65, 195, 410]))
            else:
                hand_x = int(np.interp(index, [0, 30, NUM_FRAMES - 1], [65, 120, 155]))

            if suspicious:
                if index < 24:
                    object_pos = object_start
                elif index < 48:
                    ratio = (index - 24) / 24
                    object_pos = _interpolate(object_start, object_end, ratio)
                elif index < 58:
                    object_pos = object_end
                else:
                    object_pos = None
            else:
                object_pos = object_start

            _draw_scene(frame, hand_x, object_pos)
            label = "sample_suspicious" if suspicious else "sample_normal"
            cv2.putText(frame, label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    suspicious_path = SAMPLE_DIR / "sample_suspicious.mp4"
    normal_path = SAMPLE_DIR / "sample_normal.mp4"
    labels_path = SAMPLE_DIR / "labels.csv"

    _write_video(suspicious_path, suspicious=True)
    _write_video(normal_path, suspicious=False)

    with labels_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["video_path", "label", "evidence_frames"])
        writer.writerow(["data/sample/sample_suspicious.mp4", 1, "24;36;60"])
        writer.writerow(["data/sample/sample_normal.mp4", 0, ""])

    print(f"Generated {suspicious_path}")
    print(f"Generated {normal_path}")
    print(f"Generated {labels_path}")


if __name__ == "__main__":
    main()
