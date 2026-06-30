from __future__ import annotations

from pathlib import Path
from typing import Optional

from .constants import BBox
from .runtime import cv2, np

# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def open_capture(
    source: int | str,
    camera_width: int,
    camera_height: int,
) -> "cv2.VideoCapture":
    """
    カメラまたは動画を開きます。
    """

    capture = cv2.VideoCapture(source)

    if isinstance(source, int):
        # カメラの場合のみ、希望解像度を要求します。
        # 実際の解像度はカメラドライバ側で近い値へ調整される場合があります。
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    if not capture.isOpened():
        raise RuntimeError(f"映像入力を開けませんでした: {source}")

    return capture


def create_video_writer(
    output_path: Path,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> "cv2.VideoWriter":
    """
    MP4形式の動画出力を作成します。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"動画出力を作成できませんでした: {output_path}"
        )

    return writer


def select_object_roi(frame_bgr: "np.ndarray") -> Optional[BBox]:
    """OpenCVのROI選択UIで初期追跡物体を選ばせます。"""

    roi = cv2.selectROI(
        "Select target object, then press Enter/Space",
        frame_bgr,
        showCrosshair=True,
        fromCenter=False,
    )
    try:
        cv2.destroyWindow("Select target object, then press Enter/Space")
    except Exception:
        pass

    x, y, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0:
        return None

    return (x, y, width, height)
