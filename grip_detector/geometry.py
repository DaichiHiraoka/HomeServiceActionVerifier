from __future__ import annotations

import math
from typing import Tuple

from .constants import BBox
from .runtime import np

# ---------------------------------------------------------------------------
# 基本数学関数
# ---------------------------------------------------------------------------

def clamp(value: float, minimum: float, maximum: float) -> float:
    """値を minimum～maximum の範囲へ制限します。"""
    return max(minimum, min(maximum, value))


def clamp01(value: float) -> float:
    """値を 0.0～1.0 の範囲へ制限します。"""
    return clamp(value, 0.0, 1.0)


def vector_distance(a: "np.ndarray", b: "np.ndarray") -> float:
    """
    3次元点 a, b 間のユークリッド距離を返します。

    世界座標を使う場合はメートル単位ですが、
    このプログラムでは掌幅で割るため最終的には無次元量になります。
    """
    return float(np.linalg.norm(a - b))


def angle_deg(
    point_a: "np.ndarray",
    vertex_b: "np.ndarray",
    point_c: "np.ndarray",
) -> float:
    """
    3点 A-B-C の角度を度数法で計算します。

    Bが関節位置です。
    指が真っすぐなら約180度、強く曲がるほど角度が小さくなります。
    """

    # BからA、BからCへ向かう2本のベクトルを作ります。
    vector_ba = point_a - vertex_b
    vector_bc = point_c - vertex_b

    # ゼロ除算を避けるため、各ベクトル長を確認します。
    norm_ba = float(np.linalg.norm(vector_ba))
    norm_bc = float(np.linalg.norm(vector_bc))

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        # 点が重なった異常値の場合、直線扱いにして屈曲度を上げません。
        return 180.0

    # 内積から cos(theta) を求めます。
    cosine = float(np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc))

    # 浮動小数点誤差で -1～1 を僅かに外れる場合があるため制限します。
    cosine = clamp(cosine, -1.0, 1.0)

    # arccosでラジアン角を求め、度へ変換します。
    return math.degrees(math.acos(cosine))


def flexion_score_from_angle(
    angle: float,
    straight_angle: float = 170.0,
    bent_angle: float = 70.0,
) -> float:
    """
    関節角度を 0～1 の屈曲度へ変換します。

    - straight_angle 以上: 0.0
    - bent_angle 以下: 1.0
    - その中間: 線形補間

    固定ピクセル距離ではなく角度を使うため、
    手の大きさやカメラとの距離変化に比較的強くなります。
    """

    denominator = straight_angle - bent_angle

    if denominator <= 0.0:
        raise ValueError("straight_angle は bent_angle より大きくする必要があります。")

    return clamp01((straight_angle - angle) / denominator)


def closeness_score(
    normalized_distance: float,
    near_distance: float,
    far_distance: float,
) -> float:
    """
    距離が近いほど1、遠いほど0になるスコアへ変換します。

    - near_distance 以下: 1.0
    - far_distance 以上: 0.0
    - その中間: 線形補間
    """

    denominator = far_distance - near_distance

    if denominator <= 0.0:
        raise ValueError("far_distance は near_distance より大きくする必要があります。")

    return clamp01((far_distance - normalized_distance) / denominator)


def bbox_area(bbox: BBox) -> float:
    """矩形面積を返します。"""

    return float(max(0, bbox[2]) * max(0, bbox[3]))


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    """矩形中心座標を返します。"""

    x, y, width, height = bbox
    return (x + width * 0.5, y + height * 0.5)


def clamp_bbox(bbox: BBox, frame_width: int, frame_height: int) -> BBox:
    """矩形を画像内へ収めます。"""

    x, y, width, height = bbox
    x = int(clamp(float(x), 0.0, float(max(0, frame_width - 1))))
    y = int(clamp(float(y), 0.0, float(max(0, frame_height - 1))))
    width = int(clamp(float(width), 1.0, float(max(1, frame_width - x))))
    height = int(clamp(float(height), 1.0, float(max(1, frame_height - y))))
    return (x, y, width, height)


def expand_bbox(
    bbox: BBox,
    margin_px: int,
    frame_width: int,
    frame_height: int,
) -> BBox:
    """矩形を指定ピクセルだけ広げ、画像内へ収めます。"""

    x, y, width, height = bbox
    return clamp_bbox(
        (
            x - margin_px,
            y - margin_px,
            width + 2 * margin_px,
            height + 2 * margin_px,
        ),
        frame_width,
        frame_height,
    )


def mirror_bbox(bbox: BBox, frame_width: int) -> BBox:
    """表示用に左右反転した矩形を返します。"""

    x, y, width, height = bbox
    return (frame_width - x - width, y, width, height)


def bbox_intersection_area(a: BBox, b: BBox) -> float:
    """2つの矩形の交差面積を返します。"""

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    return float(max(0, right - left) * max(0, bottom - top))


def bbox_iou(a: BBox, b: BBox) -> float:
    """2つの矩形のIoUを返します。"""

    intersection = bbox_intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def bbox_distance(a: BBox, b: BBox) -> float:
    """2つの矩形中心間の距離を返します。"""

    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def point_inside_bbox(point: Tuple[float, float], bbox: BBox) -> bool:
    """点が矩形内にあるかを返します。"""

    x, y = point
    bx, by, width, height = bbox
    return bx <= x <= bx + width and by <= y <= by + height


def point_bbox_distance(point: Tuple[float, float], bbox: BBox) -> float:
    """
    点から矩形までの最短距離を返します。

    点が矩形内にある場合は0です。
    """

    x, y = point
    bx, by, width, height = bbox
    dx = max(bx - x, 0.0, x - (bx + width))
    dy = max(by - y, 0.0, y - (by + height))
    return math.hypot(dx, dy)
