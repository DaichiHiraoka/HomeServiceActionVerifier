from __future__ import annotations

from typing import Any, Sequence, Tuple

from .constants import (
    HAND_CONNECTIONS,
    INDEX_TIP,
    MIDDLE_TIP,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
)
from .geometry import clamp, clamp01, mirror_bbox
from .models import DetectorConfig, GripDecision, TrackedObject
from .runtime import cv2, np

# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------

def normalized_to_pixel(
    landmark: Any,
    frame_width: int,
    frame_height: int,
    mirror: bool,
) -> Tuple[int, int]:
    """
    正規化座標を画素座標へ変換します。

    mirror=True の場合は表示だけ左右反転するため、x座標も反転します。
    """

    normalized_x = 1.0 - landmark.x if mirror else landmark.x

    x = int(clamp(normalized_x, 0.0, 1.0) * (frame_width - 1))
    y = int(clamp(landmark.y, 0.0, 1.0) * (frame_height - 1))

    return x, y


def draw_hand_skeleton(
    frame: "np.ndarray",
    landmarks: Sequence[Any],
    mirror: bool,
    grasping: bool,
) -> None:
    """
    手の21点と接続線を描画します。

    把持状態では緑、非把持状態では橙系で描画します。
    """

    height, width = frame.shape[:2]

    line_color = (60, 210, 70) if grasping else (0, 170, 255)
    point_color = (255, 255, 255)
    tip_color = (80, 80, 255)

    pixel_points = [
        normalized_to_pixel(
            landmark,
            frame_width=width,
            frame_height=height,
            mirror=mirror,
        )
        for landmark in landmarks
    ]

    # 骨格線を先に描きます。
    for start_id, end_id in HAND_CONNECTIONS:
        cv2.line(
            frame,
            pixel_points[start_id],
            pixel_points[end_id],
            line_color,
            2,
            cv2.LINE_AA,
        )

    fingertip_ids = {
        THUMB_TIP,
        INDEX_TIP,
        MIDDLE_TIP,
        RING_TIP,
        PINKY_TIP,
    }

    # 21点を円で描き、指先だけ色を変えます。
    for landmark_id, point in enumerate(pixel_points):
        color = tip_color if landmark_id in fingertip_ids else point_color

        cv2.circle(
            frame,
            point,
            4,
            color,
            -1,
            cv2.LINE_AA,
        )


def draw_tracked_objects(
    frame: "np.ndarray",
    tracked_objects: Sequence[TrackedObject],
    held_object_ids: set[str],
    mirror: bool,
) -> None:
    """追跡中の物体矩形を描画します。"""

    frame_width = frame.shape[1]

    for tracked_object in tracked_objects:
        bbox = tracked_object.bbox
        if mirror:
            bbox = mirror_bbox(bbox, frame_width)

        x, y, width, height = bbox

        if tracked_object.object_id in held_object_ids:
            color = (70, 230, 90)
            label_prefix = "HELD"
        elif tracked_object.missed_frames > 0:
            color = (0, 190, 255)
            label_prefix = "TRACK"
        else:
            color = (255, 180, 40)
            label_prefix = "OBJ"

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            color,
            2,
            cv2.LINE_AA,
        )

        label = (
            f"{label_prefix} {tracked_object.object_id} "
            f"{tracked_object.confidence:.2f}"
        )
        if tracked_object.label:
            label = f"{label} {tracked_object.label}"
        text_y = max(18, y - 8)
        cv2.putText(
            frame,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_score_bar(
    frame: "np.ndarray",
    x: int,
    y: int,
    width: int,
    height: int,
    score: float,
    enter_threshold: float,
    exit_threshold: float,
) -> None:
    """
    把持スコアと開始・解除閾値を横棒で表示します。
    """

    score = clamp01(score)

    # 背景バー
    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (70, 70, 70),
        -1,
    )

    # スコア部分
    filled_width = int(width * score)
    cv2.rectangle(
        frame,
        (x, y),
        (x + filled_width, y + height),
        (70, 200, 90),
        -1,
    )

    # 解除閾値を青線、開始閾値を赤線で表示します。
    exit_x = x + int(width * exit_threshold)
    enter_x = x + int(width * enter_threshold)

    cv2.line(
        frame,
        (exit_x, y - 2),
        (exit_x, y + height + 2),
        (255, 180, 60),
        2,
    )
    cv2.line(
        frame,
        (enter_x, y - 2),
        (enter_x, y + height + 2),
        (60, 60, 255),
        2,
    )

    # 外枠
    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (230, 230, 230),
        1,
    )


def draw_status_panel(
    frame: "np.ndarray",
    decisions: Sequence[GripDecision],
    tracked_objects: Sequence[TrackedObject],
    config: DetectorConfig,
    fps: float,
) -> None:
    """
    画面左上へ判定状態と主要特徴量を表示します。
    """

    panel_width = min(720, frame.shape[1] - 20)
    row_height = 128
    panel_height = 62 + max(1, len(decisions)) * row_height

    # 半透明パネルを作るため、複製画像へ矩形を描いて合成します。
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0.0,
        frame,
    )

    cv2.putText(
        frame,
        (
            f"Object-aware grip detector | FPS: {fps:5.1f} | "
            f"objects: {len(tracked_objects)}"
        ),
        (22, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    if not decisions:
        cv2.putText(
            frame,
            "No hand detected",
            (22, 83),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        return

    for row_index, decision in enumerate(decisions):
        row_top = 58 + row_index * row_height

        state_text = decision.mode if decision.is_grasping else "OPEN / NO_GRASP"
        state_color = (
            (70, 230, 90)
            if decision.is_grasping
            else (0, 180, 255)
        )

        cv2.putText(
            frame,
            (
                f"{decision.hand_id} "
                f"({decision.handedness_score:.2f}) : {state_text}"
            ),
            (22, row_top + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            state_color,
            2,
            cv2.LINE_AA,
        )

        # 平滑化スコアのバー
        draw_score_bar(
            frame,
            x=22,
            y=row_top + 34,
            width=min(280, panel_width - 40),
            height=15,
            score=decision.smoothed_score,
            enter_threshold=config.enter_threshold,
            exit_threshold=config.exit_threshold,
        )

        cv2.putText(
            frame,
            (
                f"object_raw={decision.raw_score:.3f}  "
                f"smooth={decision.smoothed_score:.3f}  "
                f"pose={decision.pose_score:.3f}  "
                f"contact={decision.object_contact_score:.3f}"
            ),
            (22, row_top + 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            (
                f"object={decision.object_id or '-'}  "
                f"overlap={decision.object_overlap_score:.2f}  "
                f"tips_in={decision.object_fingertip_inside_ratio:.2f}  "
                f"power={decision.features.power_score:.2f}  "
                f"pinch={decision.features.pinch_score:.2f}"
            ),
            (22, row_top + 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            (
                "curl "
                f"I:{decision.features.finger_curl['index']:.2f} "
                f"M:{decision.features.finger_curl['middle']:.2f} "
                f"R:{decision.features.finger_curl['ring']:.2f} "
                f"P:{decision.features.finger_curl['pinky']:.2f}"
            ),
            (22, row_top + 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


def draw_help(frame: "np.ndarray") -> None:
    """操作キーを画面下部へ表示します。"""

    height = frame.shape[0]

    cv2.putText(
        frame,
        "q/Esc: quit | p: pause | r: reset hand/object state",
        (14, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
