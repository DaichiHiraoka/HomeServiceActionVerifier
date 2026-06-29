"""Prepare a router-work trial from a raw MP4.

This is a data-preparation helper, not part of the desktop app runtime.
It uses MediaPipe Pose when available and writes the three files consumed
by the current app: context.json, skeleton.csv, and object_tracks.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROUTER_BBOX = (475, 245, 250, 285)
ROUTER_ZONE = "router_work_area"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare router normal-work trial data")
    parser.add_argument("--video", type=Path, required=True, help="Source MP4 path")
    parser.add_argument("--out", type=Path, required=True, help="Trial output directory")
    parser.add_argument("--sample-seconds", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--crop-margin", type=int, default=140, help="Object crop margin in pixels")
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def open_video(video: Path):
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        msg = f"Could not open video: {video}"
        raise RuntimeError(msg)
    return capture


def read_frame_at(capture, timestamp: float):
    import cv2

    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = capture.read()
    if not ok:
        return None
    return frame


def get_metadata(capture) -> dict[str, float | int]:
    import cv2

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0.0
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


def clamp_crop(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def write_context(path: Path) -> None:
    context = {
        "task_name": "router_normal_img_7852",
        "work_areas": [ROUTER_ZONE, "desk_area", "tool_area"],
        "private_areas": ["private_storage", "private_desk"],
        "target_objects": ["router", "lan_cable", "power_cable"],
        "worker_objects": ["screwdriver", "tape", "tool_bag"],
        "private_objects": ["wallet", "key", "resident_phone"],
        "high_risk_objects": ["wallet", "key", "resident_phone"],
        "thresholds": {
            "touch_distance": 180,
            "return_distance": 80,
            "movement_distance": 100,
            "body_distance": 450,
            "review_threshold": 0.25,
            "suspicious_threshold": 0.55,
            "high_risk_threshold": 0.8,
        },
    }
    path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pose_label(
    left_wrist_x: float,
    left_wrist_y: float,
    right_wrist_x: float,
    right_wrist_y: float,
    torso_x: float,
    torso_y: float,
) -> str:
    rx, ry, rw, rh = ROUTER_BBOX
    center_x = rx + rw / 2
    center_y = ry + rh / 2
    left = math.hypot(left_wrist_x - center_x, left_wrist_y - center_y)
    right = math.hypot(right_wrist_x - center_x, right_wrist_y - center_y)
    if min(left, right) <= 180:
        return "operate_router"
    if abs(torso_x - center_x) <= 450 and torso_y >= 250:
        return "near_router"
    return "in_work_area"


def pose_row(timestamp: float, frame, pose) -> dict[str, str] | None:
    import cv2
    import mediapipe as mp

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image)
    if not result.pose_landmarks:
        return None

    height, width = frame.shape[:2]
    landmarks = result.pose_landmarks.landmark
    names = mp.solutions.pose.PoseLandmark

    left_wrist = landmarks[names.LEFT_WRIST]
    right_wrist = landmarks[names.RIGHT_WRIST]
    left_shoulder = landmarks[names.LEFT_SHOULDER]
    right_shoulder = landmarks[names.RIGHT_SHOULDER]
    left_hip = landmarks[names.LEFT_HIP]
    right_hip = landmarks[names.RIGHT_HIP]

    left_wrist_x = left_wrist.x * width
    left_wrist_y = left_wrist.y * height
    right_wrist_x = right_wrist.x * width
    right_wrist_y = right_wrist.y * height
    torso_x = ((left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4) * width
    torso_y = ((left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4) * height

    return {
        "timestamp": fmt(timestamp),
        "left_wrist_x": fmt(left_wrist_x),
        "left_wrist_y": fmt(left_wrist_y),
        "right_wrist_x": fmt(right_wrist_x),
        "right_wrist_y": fmt(right_wrist_y),
        "torso_x": fmt(torso_x),
        "torso_y": fmt(torso_y),
        "pose_label": pose_label(left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y, torso_x, torso_y),
    }


def main() -> int:
    import cv2
    import mediapipe as mp

    args = parse_args()
    video = args.video.resolve()
    output = args.out
    crops_dir = output / "object_crops" / "router_1"
    raw_dir = output / "raw_videos"
    output.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (crops_dir / ".gitkeep").write_text("", encoding="utf-8")
    (raw_dir / ".gitkeep").write_text("", encoding="utf-8")

    capture = open_video(video)
    metadata = get_metadata(capture)
    width = int(metadata["width"])
    height = int(metadata["height"])
    duration = float(metadata["duration_seconds"])
    rx, ry, rw, rh = ROUTER_BBOX

    skeleton_rows: list[dict[str, str]] = []
    object_rows: list[dict[str, str]] = []

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        timestamp = 0.0
        index = 0
        while timestamp <= duration:
            frame = read_frame_at(capture, timestamp)
            if frame is None:
                break

            crop_x, crop_y, crop_w, crop_h = clamp_crop(
                rx - args.crop_margin,
                ry - args.crop_margin,
                rw + args.crop_margin * 2,
                rh + args.crop_margin * 2,
                width,
                height,
            )
            crop = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            crop_rel = Path("object_crops") / "router_1" / f"router_{index:04d}.jpg"
            cv2.imwrite(str(output / crop_rel), crop)

            object_rows.append(
                {
                    "timestamp": fmt(timestamp),
                    "object_id": "router_1",
                    "label": "router",
                    "role": "target",
                    "bbox_x": fmt(rx),
                    "bbox_y": fmt(ry),
                    "bbox_w": fmt(rw),
                    "bbox_h": fmt(rh),
                    "zone": ROUTER_ZONE,
                    "visible": "true",
                    "crop_path": str(crop_rel).replace("\\", "/"),
                }
            )

            row = pose_row(timestamp, frame, pose)
            if row is not None:
                skeleton_rows.append(row)

            timestamp += args.sample_seconds
            index += 1
    finally:
        pose.close()
        capture.release()

    write_context(output / "context.json")
    write_csv(output / "skeleton.csv", skeleton_rows, skeleton_fieldnames())
    write_csv(output / "object_tracks.csv", object_rows, object_fieldnames())
    source_video = portable_path(video)
    (output / "video_path.txt").write_text(source_video + "\n", encoding="utf-8")
    write_notes(output / "analysis_notes.md", source_video, metadata, len(skeleton_rows), len(object_rows))
    metadata_payload = {
        **metadata,
        "source_video": source_video,
        "sample_seconds": args.sample_seconds,
        "router_bbox": {"x": rx, "y": ry, "w": rw, "h": rh},
        "router_zone": ROUTER_ZONE,
        "skeleton_rows": len(skeleton_rows),
        "object_track_rows": len(object_rows),
    }
    (output / "video_metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote trial: {output}")
    print(f"Skeleton rows: {len(skeleton_rows)}")
    print(f"Object rows: {len(object_rows)}")
    return 0


def skeleton_fieldnames() -> list[str]:
    return [
        "timestamp",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
        "torso_x",
        "torso_y",
        "pose_label",
    ]


def object_fieldnames() -> list[str]:
    return [
        "timestamp",
        "object_id",
        "label",
        "role",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "zone",
        "visible",
        "crop_path",
    ]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_notes(
    path: Path,
    video: str,
    metadata: dict[str, float | int],
    skeleton_rows: int,
    object_rows: int,
) -> None:
    lines = [
        "# router_normal_img_7852 analysis notes",
        "",
        "この試行データは、正常系のルーター向け作業動画から作成したものです。",
        "",
        "## 入力動画",
        "",
        f"- 動画: `{video}`",
        f"- 解像度: {metadata['width']} x {metadata['height']}",
        f"- 長さ: {float(metadata['duration_seconds']):.2f} 秒",
        f"- FPS: {float(metadata['fps']):.3f}",
        "",
        "## 作成したファイル",
        "",
        "- `context.json`: ルーター作業を正常な作業文脈として定義",
        "- `skeleton.csv`: MediaPipe Poseで推定した手首と胴体中心の時系列",
        "- `object_tracks.csv`: ルーター本体を作業対象として固定bboxで追跡した時系列",
        "- `object_crops/router_1/`: ルーター本体と周辺のみを切り出したフレーム画像",
        "- `video_metadata.json`: 動画と生成処理のメタデータ",
        "",
        "## 注意",
        "",
        "ルーターのbboxは今回の動画に対する固定注釈です。物体検出器が自動でルーターを認識しているわけではありません。",
        "このデータは、現行アプリの正常系入力を作るための初期データとして使います。",
        "",
        "## 件数",
        "",
        f"- skeleton rows: {skeleton_rows}",
        f"- object track rows: {object_rows}",
        "",
        "## 実行例",
        "",
        "試行データを再生成する。",
        "",
        "```powershell",
        "uv run --no-project --python 3.11 --with \"mediapipe==0.10.21\" --with opencv-python-headless python scripts/prepare_router_trial.py --video data/raw_videos/IMG_7852.mp4 --out data/trials/router_normal_img_7852 --sample-seconds 1.0",
        "```",
        "",
        "通常のアプリCLIで分析する。",
        "",
        "```powershell",
        "uv run home-service-verifier-app --context data/trials/router_normal_img_7852/context.json --skeleton data/trials/router_normal_img_7852/skeleton.csv --objects data/trials/router_normal_img_7852/object_tracks.csv",
        "```",
        "",
        "既存 `.venv` がロックされている場合は、リポジトリの仮想環境を触らずに直接実行する。",
        "",
        "```powershell",
        "$env:PYTHONPATH=(Resolve-Path .\\src).Path; python -m home_service_action_verifier.app --context data/trials/router_normal_img_7852/context.json --skeleton data/trials/router_normal_img_7852/skeleton.csv --objects data/trials/router_normal_img_7852/object_tracks.csv",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
