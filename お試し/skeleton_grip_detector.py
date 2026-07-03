#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
skeleton_grip_detector.py

MediaPipe Hand Landmarker が出力する「手の21点骨格」と、
OpenCVで検出・追跡した画面内の物体候補を使い、カメラ映像または動画から
次の状態をリアルタイム判定する単一ファイル実装です。

    OPEN / NO_GRASP : 把持姿勢ではない
    POWER_GRASP     : 指全体で包み込む把持姿勢
    PINCH_GRASP     : 親指と人差し指によるつまみ把持姿勢
    *_OBJECT         : 追跡中の物体に対する把持

重要な制約
----------
このプログラムは汎用物体認識モデルを使いません。
物体名やカテゴリを理解するのではなく、写っている物体候補を画像上の矩形として
検出・追跡し、その矩形と手骨格の位置関係から「その物体を握っているか」を判定します。

自動検出は背景差分と輪郭に基づくため、静止物体だけを最初から確実に見つける用途では
--object-roi または --select-object で対象物体を指定してください。

骨格だけでは区別できなかった以下の誤判定を、追跡物体との接触条件で抑えます。

- 何も持っていない握り拳
- 指同士を接触させただけのピンチ
- 対象物から離れた場所で作った把持姿勢

使用手法
--------
1. MediaPipe Hand Landmarker で手の21点3次元骨格を取得
2. 各指の PIP/DIP 関節角度から「指の屈曲度」を計算
3. 指先と掌中心の正規化距離から「手の閉じ具合」を計算
4. 親指と他指の距離から「母指対向」を計算
5. 背景差分、輪郭、テンプレート照合で物体候補を検出・追跡
6. 指先、掌、手矩形と追跡物体矩形の接触度を計算
7. 骨格把持スコアと物体接触スコアを統合して「物体把持スコア」を計算
8. EMA平滑化、開始遅延、解除遅延、ヒステリシスで瞬間的な誤判定を抑制

実行例
------
依存パッケージの導入:
    python -m pip install mediapipe opencv-python numpy

Webカメラ:
    python skeleton_grip_detector.py

起動後にIP WebcamのURLを入力:
    python skeleton_grip_detector.py
    映像入力ソース> 192.168.1.20:8080

カメラ番号1:
    python skeleton_grip_detector.py --source 1

IP WebcamのURLを引数指定:
    python skeleton_grip_detector.py --source http://192.168.1.20:8080/video

動画ファイル:
    python skeleton_grip_detector.py --source input.mp4

静止している対象物を矩形で指定:
    python skeleton_grip_detector.py --source input.mp4 --object-roi 520,310,80,60

初回フレームで対象物をマウス選択:
    python skeleton_grip_detector.py --source input.mp4 --select-object

判定結果をCSVへ保存:
    python skeleton_grip_detector.py --csv grip_log.csv

描画済み動画を保存:
    python skeleton_grip_detector.py --output annotated.mp4

終了キー:
    q または Esc

一時停止:
    p

状態リセット:
    r
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 外部依存は、未導入時にも説明付きエラーを出せるように try/except で読み込みます。
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# MediaPipe公式配布モデル
# ---------------------------------------------------------------------------

# Google AI Edge が公式配布している Hand Landmarker の full モデルです。
# 初回実行時だけユーザーのキャッシュディレクトリへ自動保存します。
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# モデルの既定保存先です。
# Pythonファイルの隣へ勝手にファイルを置かず、ユーザー領域のキャッシュへ保存します。
DEFAULT_MODEL_PATH = (
    Path.home()
    / ".cache"
    / "skeleton_grip_detector"
    / "hand_landmarker.task"
)


# ---------------------------------------------------------------------------
# MediaPipe Hand Landmarker の21点番号
# ---------------------------------------------------------------------------

# 21点の意味:
#  0: 手首
#  1-4: 親指 CMC, MCP, IP, TIP
#  5-8: 人差し指 MCP, PIP, DIP, TIP
#  9-12: 中指 MCP, PIP, DIP, TIP
# 13-16: 薬指 MCP, PIP, DIP, TIP
# 17-20: 小指 MCP, PIP, DIP, TIP

WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


# 各指の4点を、根元から先端までの順に定義します。
# 親指は関節構造が異なるため、主な4指とは別処理にします。
FINGER_JOINTS: Dict[str, Tuple[int, int, int, int]] = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}


# 骨格を描画するための接続関係です。
# MediaPipeの21点番号を線で結び、手骨格を画面上へ表示します。
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    # 掌
    (WRIST, THUMB_CMC),
    (WRIST, INDEX_MCP),
    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, PINKY_MCP),
    (PINKY_MCP, WRIST),

    # 親指
    (THUMB_CMC, THUMB_MCP),
    (THUMB_MCP, THUMB_IP),
    (THUMB_IP, THUMB_TIP),

    # 人差し指
    (INDEX_MCP, INDEX_PIP),
    (INDEX_PIP, INDEX_DIP),
    (INDEX_DIP, INDEX_TIP),

    # 中指
    (MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_DIP, MIDDLE_TIP),

    # 薬指
    (RING_MCP, RING_PIP),
    (RING_PIP, RING_DIP),
    (RING_DIP, RING_TIP),

    # 小指
    (PINKY_MCP, PINKY_PIP),
    (PINKY_PIP, PINKY_DIP),
    (PINKY_DIP, PINKY_TIP),
)


# OpenCVの矩形を (x, y, width, height) で扱います。
BBox = Tuple[int, int, int, int]


# 物体との接触判定で使う指先IDです。
FINGERTIP_IDS: Tuple[int, ...] = (
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
)


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass
class GripFeatures:
    """
    1フレーム、1手分の骨格特徴量を保持します。

    値は原則として 0.0～1.0 へ正規化します。
    0.0 は把持らしくない状態、1.0 は把持らしい状態です。
    """

    # 各指の屈曲度
    finger_curl: Dict[str, float]

    # 各指先が掌へ近づいている度合い
    finger_closure: Dict[str, float]

    # 指の骨長合計に対して MCP-TIP 間が縮んだ度合い
    # 浅い把持では指先が掌中心まで届かないため、この特徴量が特に重要です。
    finger_contraction: Dict[str, float]

    # 親指が他指へ対向している度合い
    thumb_opposition: float

    # 親指先と人差し指先の近接度
    pinch_proximity: float

    # パワーグリップ専用の統合スコア
    power_score: float

    # ピンチグリップ専用の統合スコア
    pinch_score: float

    # 最終的に時間平滑化へ渡す生スコア
    raw_score: float

    # 生スコア時点で優勢な把持方式
    raw_mode: str

    # デバッグと研究ログ用の関節角度
    pip_angles_deg: Dict[str, float]
    dip_angles_deg: Dict[str, float]

    # 正規化に使った掌幅
    palm_width: float


@dataclass
class TrackedObject:
    """
    画面内で追跡している物体候補です。

    object_id はフレーム間で維持されます。bbox は元画像座標系の矩形で、
    mirror 表示の有無には影響されません。
    """

    object_id: str
    bbox: BBox
    confidence: float
    source: str
    first_seen_time: float
    last_seen_time: float
    last_update_time: float
    age_frames: int = 1
    missed_frames: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)
    template: Optional["np.ndarray"] = None


@dataclass
class ObjectEvidence:
    """
    1つの手と、最も関連が強い追跡物体の関係を保持します。
    """

    object_id: str = ""
    bbox: Optional[BBox] = None
    contact_score: float = 0.0
    overlap_score: float = 0.0
    fingertip_inside_ratio: float = 0.0
    proximity_score: float = 0.0


@dataclass
class TemporalGripState:
    """
    時系列判定の内部状態です。

    1フレームだけスコアが閾値を超えても、即座には把持にしません。
    一定時間継続した場合だけ状態遷移させます。
    """

    # EMAで平滑化した把持スコア
    smoothed_score: float = 0.0

    # 現在、把持状態として確定しているか
    is_grasping: bool = False

    # 現在確定している把持方式
    mode: str = "NONE"

    # 把持候補が始まった時刻
    enter_candidate_since: Optional[float] = None

    # 解除候補が始まった時刻
    exit_candidate_since: Optional[float] = None

    # 最後にこの手を検出した時刻
    last_seen_time: Optional[float] = None

    # EMAの時間差計算に使う直前更新時刻
    last_update_time: Optional[float] = None


@dataclass
class GripDecision:
    """
    描画、CSV出力、後段処理へ渡す最終判定結果です。
    """

    hand_id: str
    handedness: str
    handedness_score: float
    is_grasping: bool
    mode: str
    raw_score: float
    pose_score: float
    smoothed_score: float
    object_id: str
    object_bbox: Optional[BBox]
    object_contact_score: float
    object_overlap_score: float
    object_fingertip_inside_ratio: float
    features: GripFeatures


@dataclass
class DetectorConfig:
    """
    判定閾値を一か所に集約します。

    CLI引数から変更できるため、実験データに基づく閾値調整が可能です。
    """

    # 把持開始と判定する平滑化スコア
    enter_threshold: float = 0.50

    # 把持解除と判定する平滑化スコア
    exit_threshold: float = 0.36

    # 開始閾値を超え続ける必要時間
    enter_delay_sec: float = 0.18

    # 解除閾値を下回り続ける必要時間
    exit_delay_sec: float = 0.15

    # EMAの時定数。大きいほど滑らかだが応答が遅くなります。
    ema_time_constant_sec: float = 0.10

    # 手が見失われてから状態を破棄するまでの時間
    missing_reset_sec: float = 0.50

    # 物体追跡を有効化するか
    object_tracking_enabled: bool = True

    # 背景差分で物体候補として採用する最小面積
    object_min_area: float = 500.0

    # フレーム全面に対する物体候補の最大面積比
    object_max_area_ratio: float = 0.35

    # 極端に細長い輪郭を除外するためのアスペクト比範囲
    object_min_aspect_ratio: float = 0.15
    object_max_aspect_ratio: float = 6.0

    # 同時に保持する最大物体数
    object_max_tracks: int = 6

    # 検出できないフレームが続いた物体を削除するまでの猶予
    object_max_missed_frames: int = 24

    # 物体IDのフレーム間対応付けに使う距離上限
    object_association_distance_px: float = 110.0

    # テンプレート照合の採用閾値
    object_template_match_threshold: float = 0.58

    # 直前bboxの何倍の領域をテンプレート探索するか
    object_template_search_scale: float = 2.8

    # 手領域を背景差分候補から除くときの余白
    object_hand_mask_padding_px: int = 18

    # 背景差分の履歴長と閾値
    object_background_history: int = 180
    object_background_threshold: float = 32.0
    object_background_learning_rate: float = -1.0

    # 手と物体が接触しているとみなすスコア目安
    object_contact_threshold: float = 0.42


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


# ---------------------------------------------------------------------------
# 骨格特徴量の計算
# ---------------------------------------------------------------------------

def landmarks_to_numpy(
    landmarks: Sequence[Any],
) -> "np.ndarray":
    """
    MediaPipeのランドマーク列を shape=(21, 3) のNumPy配列へ変換します。
    """

    return np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in landmarks],
        dtype=np.float64,
    )


def select_geometry_points(
    image_landmarks: Sequence[Any],
    world_landmarks: Optional[Sequence[Any]],
) -> "np.ndarray":
    """
    幾何計算に使う座標系を選びます。

    優先:
        MediaPipeの世界座標

    代替:
        画像正規化座標

    世界座標は x, y, z が同一の物理尺度なので、
    関節角度や距離比の計算に適しています。
    """

    if world_landmarks is not None and len(world_landmarks) == 21:
        world_points = landmarks_to_numpy(world_landmarks)

        # NaNやInfがなく、掌幅が極端に小さくなければ世界座標を採用します。
        if np.all(np.isfinite(world_points)):
            palm_width = vector_distance(
                world_points[INDEX_MCP],
                world_points[PINKY_MCP],
            )
            if palm_width > 1e-6:
                return world_points

    # 世界座標が使えない場合だけ画像座標へフォールバックします。
    return landmarks_to_numpy(image_landmarks)


def landmarks_to_pixel_points(
    landmarks: Sequence[Any],
    frame_width: int,
    frame_height: int,
) -> List[Tuple[int, int]]:
    """
    MediaPipeの正規化ランドマークを元画像座標系の画素列へ変換します。
    """

    points: List[Tuple[int, int]] = []

    for landmark in landmarks:
        x = int(clamp(landmark.x, 0.0, 1.0) * (frame_width - 1))
        y = int(clamp(landmark.y, 0.0, 1.0) * (frame_height - 1))
        points.append((x, y))

    return points


def hand_bbox_from_landmarks(
    landmarks: Sequence[Any],
    frame_width: int,
    frame_height: int,
    margin_px: int = 0,
) -> BBox:
    """手ランドマーク全体を含む矩形を返します。"""

    points = landmarks_to_pixel_points(landmarks, frame_width, frame_height)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    bbox = (
        min(xs) - margin_px,
        min(ys) - margin_px,
        max(xs) - min(xs) + 1 + 2 * margin_px,
        max(ys) - min(ys) + 1 + 2 * margin_px,
    )
    return clamp_bbox(bbox, frame_width, frame_height)


def estimate_palm_width_pixels(
    landmarks: Sequence[Any],
    frame_width: int,
    frame_height: int,
) -> float:
    """人差し指MCPと小指MCPの距離から掌幅を画素単位で推定します。"""

    points = landmarks_to_pixel_points(landmarks, frame_width, frame_height)
    x0, y0 = points[INDEX_MCP]
    x1, y1 = points[PINKY_MCP]
    return max(math.hypot(x0 - x1, y0 - y1), 1.0)


def calculate_grip_features(
    image_landmarks: Sequence[Any],
    world_landmarks: Optional[Sequence[Any]],
) -> GripFeatures:
    """
    21点骨格から把持判定用の特徴量を計算します。

    物体の画素、色、輪郭、検出結果は一切参照しません。
    """

    # 幾何計算に適した座標列を取得します。
    points = select_geometry_points(image_landmarks, world_landmarks)

    # 掌幅を、人差し指MCPと小指MCPの距離として定義します。
    # 以後の距離を掌幅で割ることで、手の大きさや撮影距離の影響を弱めます。
    palm_width = vector_distance(points[INDEX_MCP], points[PINKY_MCP])
    palm_width = max(palm_width, 1e-6)

    # 掌中心は、手首と4本のMCPの平均として近似します。
    # 厳密な解剖学的中心ではありませんが、骨格のみから安定して計算できます。
    palm_center = np.mean(
        points[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]],
        axis=0,
    )

    finger_curl: Dict[str, float] = {}
    finger_closure: Dict[str, float] = {}
    finger_contraction: Dict[str, float] = {}
    pip_angles_deg: Dict[str, float] = {}
    dip_angles_deg: Dict[str, float] = {}

    # 人差し指、中指、薬指、小指について個別に処理します。
    for finger_name, (mcp_id, pip_id, dip_id, tip_id) in FINGER_JOINTS.items():
        # PIP関節角: MCP-PIP-DIP
        pip_angle = angle_deg(
            points[mcp_id],
            points[pip_id],
            points[dip_id],
        )

        # DIP関節角: PIP-DIP-TIP
        dip_angle = angle_deg(
            points[pip_id],
            points[dip_id],
            points[tip_id],
        )

        # 浅い把持を早い段階で捉えるため、屈曲度を2種類計算します。
        #
        # early_flexion:
        #   130～178度付近を主対象とし、軽く曲がり始めた段階から反応します。
        # strong_flexion:
        #   深い屈曲を評価し、握り拳に近い状態では高得点になります。
        #
        # 早期屈曲だけでは関節推定ノイズに反応しやすいため、
        # 強い屈曲も40%残して安定性を確保します。
        pip_early_flexion = flexion_score_from_angle(
            pip_angle,
            straight_angle=178.0,
            bent_angle=125.0,
        )
        dip_early_flexion = flexion_score_from_angle(
            dip_angle,
            straight_angle=178.0,
            bent_angle=130.0,
        )
        pip_strong_flexion = flexion_score_from_angle(
            pip_angle,
            straight_angle=170.0,
            bent_angle=75.0,
        )
        dip_strong_flexion = flexion_score_from_angle(
            dip_angle,
            straight_angle=170.0,
            bent_angle=85.0,
        )

        early_curl = clamp01(
            0.65 * pip_early_flexion
            + 0.35 * dip_early_flexion
        )
        strong_curl = clamp01(
            0.65 * pip_strong_flexion
            + 0.35 * dip_strong_flexion
        )

        curl_score = clamp01(
            0.60 * early_curl
            + 0.40 * strong_curl
        )

        # 指を構成する3本の骨区間の長さを合計します。
        # 指が伸びている場合、MCP-TIP直線距離はこの合計に近くなります。
        # 指が曲がると、骨長合計はほぼ不変のまま直線距離だけ短くなります。
        finger_path_length = (
            vector_distance(points[mcp_id], points[pip_id])
            + vector_distance(points[pip_id], points[dip_id])
            + vector_distance(points[dip_id], points[tip_id])
        )
        finger_path_length = max(finger_path_length, 1e-8)

        finger_chord_ratio = (
            vector_distance(points[mcp_id], points[tip_id])
            / finger_path_length
        )

        # 直線距離比が0.97以上ならほぼ伸展、0.72以下なら十分収縮とします。
        # 掌中心への近さとは異なり、浅い円柱把持でも反応しやすい特徴です。
        contraction_score = closeness_score(
            normalized_distance=finger_chord_ratio,
            near_distance=0.72,
            far_distance=0.97,
        )

        # 指先と掌中心の距離を掌幅で正規化します。
        normalized_tip_distance = (
            vector_distance(points[tip_id], palm_center)
            / palm_width
        )

        # 旧版より判定範囲を広げます。
        # 指先が掌中心まで深く入り込まなくても、補助的な閉鎖証拠になります。
        closure_score = closeness_score(
            normalized_distance=normalized_tip_distance,
            near_distance=0.70,
            far_distance=1.75,
        )

        finger_curl[finger_name] = curl_score
        finger_closure[finger_name] = closure_score
        finger_contraction[finger_name] = contraction_score
        pip_angles_deg[finger_name] = pip_angle
        dip_angles_deg[finger_name] = dip_angle

    # 親指のIP関節角度を計算します。
    thumb_ip_angle = angle_deg(
        points[THUMB_MCP],
        points[THUMB_IP],
        points[THUMB_TIP],
    )
    thumb_flexion = flexion_score_from_angle(
        thumb_ip_angle,
        straight_angle=165.0,
        bent_angle=75.0,
    )

    # 親指先から、人差し指・中指・薬指の先端までの最短距離を計算します。
    # 親指が他指側へ回り込む「母指対向」を近似する特徴です。
    thumb_to_finger_distances = [
        vector_distance(points[THUMB_TIP], points[INDEX_TIP]) / palm_width,
        vector_distance(points[THUMB_TIP], points[MIDDLE_TIP]) / palm_width,
        vector_distance(points[THUMB_TIP], points[RING_TIP]) / palm_width,
    ]
    minimum_thumb_distance = min(thumb_to_finger_distances)

    # 近いほど母指対向が強いとみなします。
    thumb_opposition = closeness_score(
        normalized_distance=minimum_thumb_distance,
        near_distance=0.25,
        far_distance=1.05,
    )

    # ピンチ判定では親指先と人差し指先の距離を直接使います。
    thumb_index_distance = (
        vector_distance(points[THUMB_TIP], points[INDEX_TIP])
        / palm_width
    )
    pinch_proximity = closeness_score(
        normalized_distance=thumb_index_distance,
        near_distance=0.16,
        far_distance=0.72,
    )

    # -----------------------------------------------------------------------
    # パワーグリップスコア
    # -----------------------------------------------------------------------

    # 物体を包み込む把持では、人差し指と中指の屈曲が特に重要です。
    # 小指だけ曲げた姿勢などが高得点にならないよう、重みを非均等にします。
    weighted_curl = (
        0.35 * finger_curl["index"]
        + 0.30 * finger_curl["middle"]
        + 0.20 * finger_curl["ring"]
        + 0.15 * finger_curl["pinky"]
    )

    # 指の収縮率を重み付き平均します。
    # これは浅い把持で最も効く追加特徴量です。
    weighted_contraction = (
        0.35 * finger_contraction["index"]
        + 0.30 * finger_contraction["middle"]
        + 0.20 * finger_contraction["ring"]
        + 0.15 * finger_contraction["pinky"]
    )

    # 指先と掌中心の近さも同じ順序で重み付けします。
    weighted_closure = (
        0.35 * finger_closure["index"]
        + 0.30 * finger_closure["middle"]
        + 0.20 * finger_closure["ring"]
        + 0.15 * finger_closure["pinky"]
    )

    # 4本中2番目に高い値を採用します。
    # これにより、少なくとも2本の指が同時に曲がった場合だけ
    # 「協調した把持」の証拠が強くなります。
    # 旧版の min(index, middle) と異なり、人差し指が物体表面に沿って
    # やや伸びている把持でも、他の複数指から判定できます。
    coordinated_curl = sorted(
        finger_curl.values(),
        reverse=True,
    )[1]
    coordinated_contraction = sorted(
        finger_contraction.values(),
        reverse=True,
    )[1]

    # まず、各特徴量を0～1の証拠値として重み付き統合します。
    # 浅い把持への感度を上げるため、掌中心距離の重みを旧版の28%から14%へ下げ、
    # 代わりに屈曲、収縮率、複数指協調を重視します。
    power_evidence = clamp01(
        0.38 * weighted_curl
        + 0.24 * weighted_contraction
        + 0.14 * weighted_closure
        + 0.10 * thumb_opposition
        + 0.09 * coordinated_curl
        + 0.05 * coordinated_contraction
    )

    # 実際の把持姿勢では、すべての特徴が同時に1.0になることは稀です。
    # そのため0.08～0.63の実用範囲を0～1へ再正規化します。
    # 単なる閾値低下ではなく、特徴空間の実用域を広げる処理です。
    calibrated_power_score = clamp01(
        (power_evidence - 0.08) / 0.55
    )

    # 複数指の協調が弱い場合は、単一指ジェスチャーの誤判定を抑えるため減衰します。
    coordination_gate = clamp01(
        (coordinated_curl - 0.12) / 0.48
    )
    power_score = clamp01(
        calibrated_power_score
        * (0.72 + 0.28 * coordination_gate)
    )

    # -----------------------------------------------------------------------
    # ピンチグリップスコア
    # -----------------------------------------------------------------------

    # ピンチでは親指先と人差し指先の近さが最重要です。
    # ただし単なる指先接触を完全には除外できません。
    pinch_score = clamp01(
        0.68 * pinch_proximity
        + 0.20 * finger_curl["index"]
        + 0.12 * thumb_flexion
    )

    # 2種類の把持方式のうち、より強い方を最終生スコアに採用します。
    if power_score >= pinch_score:
        raw_score = power_score
        raw_mode = "POWER_GRASP"
    else:
        raw_score = pinch_score
        raw_mode = "PINCH_GRASP"

    return GripFeatures(
        finger_curl=finger_curl,
        finger_closure=finger_closure,
        finger_contraction=finger_contraction,
        thumb_opposition=thumb_opposition,
        pinch_proximity=pinch_proximity,
        power_score=power_score,
        pinch_score=pinch_score,
        raw_score=raw_score,
        raw_mode=raw_mode,
        pip_angles_deg=pip_angles_deg,
        dip_angles_deg=dip_angles_deg,
        palm_width=palm_width,
    )


# ---------------------------------------------------------------------------
# 物体検出・追跡と手物体関係
# ---------------------------------------------------------------------------

class ObjectTracker:
    """
    背景差分とテンプレート照合で、画面内の物体候補をID付きで追跡します。

    汎用物体認識ではないため物体名は出しません。画面上の矩形として追跡し、
    手骨格との位置関係を後段で評価します。
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.tracks: Dict[str, TrackedObject] = {}
        self.next_object_index = 1
        self.background_subtractor = self._create_background_subtractor()

    def _create_background_subtractor(self) -> Any:
        return cv2.createBackgroundSubtractorMOG2(
            history=self.config.object_background_history,
            varThreshold=self.config.object_background_threshold,
            detectShadows=True,
        )

    def reset(self) -> None:
        """追跡状態と背景モデルを初期化します。"""

        self.tracks.clear()
        self.background_subtractor = self._create_background_subtractor()

    def add_manual_track(
        self,
        frame_bgr: "np.ndarray",
        bbox: BBox,
        timestamp_sec: float,
        source: str = "manual",
    ) -> Optional[TrackedObject]:
        """ユーザー指定の矩形から追跡を開始します。"""

        frame_height, frame_width = frame_bgr.shape[:2]
        bbox = clamp_bbox(bbox, frame_width, frame_height)

        if bbox_area(bbox) < 4.0:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        object_id = self._new_object_id()
        track = TrackedObject(
            object_id=object_id,
            bbox=bbox,
            confidence=1.0,
            source=source,
            first_seen_time=timestamp_sec,
            last_seen_time=timestamp_sec,
            last_update_time=timestamp_sec,
            template=self._extract_template(gray, bbox),
        )
        self.tracks[object_id] = track
        return track

    def update(
        self,
        frame_bgr: "np.ndarray",
        hand_landmarks_list: Sequence[Sequence[Any]],
        timestamp_sec: float,
    ) -> List[TrackedObject]:
        """現在フレームから物体候補を検出し、既存トラックへ対応付けます。"""

        if not self.config.object_tracking_enabled:
            return []

        frame_height, frame_width = frame_bgr.shape[:2]
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detections = self._detect_candidates(
            frame_bgr=frame_bgr,
            hand_landmarks_list=hand_landmarks_list,
        )
        unmatched_detection_indices = set(range(len(detections)))

        for track in list(self.tracks.values()):
            match_index = self._best_detection_index(
                track=track,
                detections=detections,
                candidate_indices=unmatched_detection_indices,
            )

            if match_index is not None:
                unmatched_detection_indices.remove(match_index)
                self._update_track(
                    track=track,
                    bbox=detections[match_index],
                    frame_gray=frame_gray,
                    timestamp_sec=timestamp_sec,
                    confidence=1.0,
                )
                continue

            template_match = self._template_match(
                track=track,
                frame_gray=frame_gray,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            if template_match is not None:
                bbox, match_score = template_match
                self._update_track(
                    track=track,
                    bbox=bbox,
                    frame_gray=frame_gray,
                    timestamp_sec=timestamp_sec,
                    confidence=match_score,
                    update_template=False,
                )
                continue

            track.missed_frames += 1
            track.confidence = max(0.0, track.confidence * 0.82)

        for index in sorted(unmatched_detection_indices):
            if len(self.tracks) >= self.config.object_max_tracks:
                break
            self.add_manual_track(
                frame_bgr=frame_bgr,
                bbox=detections[index],
                timestamp_sec=timestamp_sec,
                source="auto",
            )

        self._drop_stale_tracks()
        return self.objects()

    def objects(self) -> List[TrackedObject]:
        """追跡中の物体を、信頼度が高い順に返します。"""

        return sorted(
            self.tracks.values(),
            key=lambda item: (item.missed_frames, -item.confidence, item.object_id),
        )

    def _detect_candidates(
        self,
        frame_bgr: "np.ndarray",
        hand_landmarks_list: Sequence[Sequence[Any]],
    ) -> List[BBox]:
        frame_height, frame_width = frame_bgr.shape[:2]
        frame_area = float(frame_width * frame_height)

        foreground = self.background_subtractor.apply(
            frame_bgr,
            learningRate=self.config.object_background_learning_rate,
        )

        # MOG2の影画素は127付近になるため、明確な前景だけ残します。
        _, foreground = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)

        for landmarks in hand_landmarks_list:
            hand_bbox = hand_bbox_from_landmarks(
                landmarks=landmarks,
                frame_width=frame_width,
                frame_height=frame_height,
                margin_px=self.config.object_hand_mask_padding_px,
            )
            x, y, width, height = hand_bbox
            cv2.rectangle(
                foreground,
                (x, y),
                (x + width, y + height),
                0,
                -1,
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            foreground,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections: List[BBox] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))

            if area < self.config.object_min_area:
                continue
            if area > frame_area * self.config.object_max_area_ratio:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width <= 0 or height <= 0:
                continue

            aspect_ratio = width / height
            if aspect_ratio < self.config.object_min_aspect_ratio:
                continue
            if aspect_ratio > self.config.object_max_aspect_ratio:
                continue

            detections.append(clamp_bbox((x, y, width, height), frame_width, frame_height))

        detections.sort(key=bbox_area, reverse=True)
        return detections

    def _best_detection_index(
        self,
        track: TrackedObject,
        detections: Sequence[BBox],
        candidate_indices: set[int],
    ) -> Optional[int]:
        best_index: Optional[int] = None
        best_score = 0.0

        for index in candidate_indices:
            bbox = detections[index]
            iou_score = bbox_iou(track.bbox, bbox)
            distance = bbox_distance(track.bbox, bbox)
            distance_score = closeness_score(
                normalized_distance=distance,
                near_distance=0.0,
                far_distance=self.config.object_association_distance_px,
            )
            score = max(iou_score, 0.80 * distance_score)

            if score > best_score:
                best_score = score
                best_index = index

        if best_score < 0.20:
            return None

        return best_index

    def _update_track(
        self,
        track: TrackedObject,
        bbox: BBox,
        frame_gray: "np.ndarray",
        timestamp_sec: float,
        confidence: float,
        update_template: bool = True,
    ) -> None:
        old_center = bbox_center(track.bbox)
        new_center = bbox_center(bbox)
        delta_time = max(timestamp_sec - track.last_update_time, 1e-6)

        track.velocity = (
            (new_center[0] - old_center[0]) / delta_time,
            (new_center[1] - old_center[1]) / delta_time,
        )
        track.bbox = bbox
        track.confidence = clamp01(confidence)
        track.last_seen_time = timestamp_sec
        track.last_update_time = timestamp_sec
        track.age_frames += 1
        track.missed_frames = 0

        if update_template:
            template = self._extract_template(frame_gray, bbox)
            if template is not None:
                track.template = template

    def _template_match(
        self,
        track: TrackedObject,
        frame_gray: "np.ndarray",
        frame_width: int,
        frame_height: int,
    ) -> Optional[Tuple[BBox, float]]:
        if track.template is None:
            return None

        template_height, template_width = track.template.shape[:2]
        if template_width < 4 or template_height < 4:
            return None

        x, y, width, height = track.bbox
        search_margin_x = int(width * self.config.object_template_search_scale)
        search_margin_y = int(height * self.config.object_template_search_scale)
        search_bbox = expand_bbox(
            bbox=(x, y, width, height),
            margin_px=max(search_margin_x, search_margin_y),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        sx, sy, sw, sh = search_bbox
        search_region = frame_gray[sy: sy + sh, sx: sx + sw]

        if search_region.shape[1] < template_width:
            return None
        if search_region.shape[0] < template_height:
            return None

        match = cv2.matchTemplate(
            search_region,
            track.template,
            cv2.TM_CCOEFF_NORMED,
        )
        _, max_value, _, max_location = cv2.minMaxLoc(match)

        if max_value < self.config.object_template_match_threshold:
            return None

        mx, my = max_location
        bbox = clamp_bbox(
            (sx + mx, sy + my, template_width, template_height),
            frame_width,
            frame_height,
        )
        return bbox, float(max_value)

    def _extract_template(
        self,
        frame_gray: "np.ndarray",
        bbox: BBox,
    ) -> Optional["np.ndarray"]:
        frame_height, frame_width = frame_gray.shape[:2]
        x, y, width, height = clamp_bbox(bbox, frame_width, frame_height)

        if width < 4 or height < 4:
            return None

        return frame_gray[y: y + height, x: x + width].copy()

    def _drop_stale_tracks(self) -> None:
        stale_ids = [
            object_id
            for object_id, track in self.tracks.items()
            if track.missed_frames > self.config.object_max_missed_frames
        ]

        for object_id in stale_ids:
            del self.tracks[object_id]

    def _new_object_id(self) -> str:
        object_id = f"object_{self.next_object_index}"
        self.next_object_index += 1
        return object_id


def evaluate_hand_object_evidence(
    image_landmarks: Sequence[Any],
    features: GripFeatures,
    tracked_objects: Sequence[TrackedObject],
    frame_width: int,
    frame_height: int,
    config: DetectorConfig,
) -> ObjectEvidence:
    """
    手と追跡物体の位置関係から、最も把持されていそうな物体を選びます。
    """

    if not tracked_objects:
        return ObjectEvidence()

    palm_width = estimate_palm_width_pixels(
        image_landmarks,
        frame_width,
        frame_height,
    )
    contact_margin = max(8, int(palm_width * 0.22))
    hand_bbox = hand_bbox_from_landmarks(
        image_landmarks,
        frame_width,
        frame_height,
        margin_px=contact_margin,
    )
    pixel_points = landmarks_to_pixel_points(
        image_landmarks,
        frame_width,
        frame_height,
    )

    fingertip_points = [pixel_points[landmark_id] for landmark_id in FINGERTIP_IDS]
    thumb_tip = pixel_points[THUMB_TIP]
    index_tip = pixel_points[INDEX_TIP]
    middle_tip = pixel_points[MIDDLE_TIP]
    pinch_midpoint = (
        (thumb_tip[0] + index_tip[0]) * 0.5,
        (thumb_tip[1] + index_tip[1]) * 0.5,
    )

    best = ObjectEvidence()
    best_score = 0.0

    for tracked_object in tracked_objects:
        bbox = tracked_object.bbox
        expanded_bbox = expand_bbox(
            bbox,
            contact_margin,
            frame_width,
            frame_height,
        )

        tip_distance_scores = [
            closeness_score(
                normalized_distance=point_bbox_distance(point, expanded_bbox) / palm_width,
                near_distance=0.0,
                far_distance=0.95,
            )
            for point in fingertip_points
        ]
        top_tip_scores = sorted(tip_distance_scores, reverse=True)[:3]
        power_contact = sum(top_tip_scores) / max(1, len(top_tip_scores))

        thumb_score = closeness_score(
            normalized_distance=point_bbox_distance(thumb_tip, expanded_bbox) / palm_width,
            near_distance=0.0,
            far_distance=0.90,
        )
        index_score = closeness_score(
            normalized_distance=point_bbox_distance(index_tip, expanded_bbox) / palm_width,
            near_distance=0.0,
            far_distance=0.90,
        )
        middle_score = closeness_score(
            normalized_distance=point_bbox_distance(middle_tip, expanded_bbox) / palm_width,
            near_distance=0.0,
            far_distance=0.90,
        )
        pinch_midpoint_score = closeness_score(
            normalized_distance=point_bbox_distance(pinch_midpoint, expanded_bbox) / palm_width,
            near_distance=0.0,
            far_distance=0.80,
        )
        pinch_contact = clamp01(
            0.55 * min(thumb_score, index_score)
            + 0.25 * pinch_midpoint_score
            + 0.20 * middle_score
        )

        fingertip_inside_ratio = (
            sum(1 for point in fingertip_points if point_inside_bbox(point, expanded_bbox))
            / len(fingertip_points)
        )
        overlap_score = clamp01(
            bbox_intersection_area(hand_bbox, expanded_bbox)
            / max(bbox_area(bbox), 1.0)
        )
        object_center = bbox_center(bbox)
        center_inside_hand = 1.0 if point_inside_bbox(object_center, hand_bbox) else 0.0

        if features.raw_mode == "PINCH_GRASP":
            contact_score = clamp01(
                0.46 * pinch_contact
                + 0.22 * power_contact
                + 0.20 * fingertip_inside_ratio
                + 0.12 * overlap_score
            )
        else:
            contact_score = clamp01(
                0.36 * power_contact
                + 0.20 * pinch_contact
                + 0.20 * fingertip_inside_ratio
                + 0.16 * overlap_score
                + 0.08 * center_inside_hand
            )

        confidence_gate = 0.70 + 0.30 * clamp01(tracked_object.confidence)
        contact_score = clamp01(contact_score * confidence_gate)

        if contact_score > best_score:
            best_score = contact_score
            best = ObjectEvidence(
                object_id=tracked_object.object_id,
                bbox=bbox,
                contact_score=contact_score,
                overlap_score=overlap_score,
                fingertip_inside_ratio=fingertip_inside_ratio,
                proximity_score=max(max(tip_distance_scores), pinch_contact),
            )

    return best


def combine_pose_and_object_scores(
    features: GripFeatures,
    evidence: ObjectEvidence,
    config: DetectorConfig,
) -> float:
    """
    骨格把持スコアと物体接触スコアを統合します。

    物体接触が弱い場合は、握り拳姿勢だけでは高得点にならないよう強く減衰します。
    """

    if not evidence.object_id:
        return 0.0

    contact_gate = clamp01(
        (evidence.contact_score - config.object_contact_threshold * 0.35)
        / max(config.object_contact_threshold * 0.95, 1e-6)
    )
    base_score = clamp01(
        0.62 * features.raw_score
        + 0.38 * evidence.contact_score
    )
    return clamp01(base_score * contact_gate)


def object_grasp_mode(features: GripFeatures, evidence: ObjectEvidence) -> str:
    """表示とCSV用の物体把持モード名を返します。"""

    if not evidence.object_id:
        return "NO_TRACKED_OBJECT"

    if features.raw_mode == "PINCH_GRASP":
        return "PINCH_OBJECT"

    return "POWER_OBJECT"


# ---------------------------------------------------------------------------
# 時系列判定
# ---------------------------------------------------------------------------

class GripTemporalFilter:
    """
    手ごとの時系列状態を管理します。

    ヒステリシス:
        開始閾値 > 解除閾値

    これにより、閾値付近で OPEN と GRASP が高速反転する現象を抑えます。
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.states: Dict[str, TemporalGripState] = {}

    def reset(self) -> None:
        """全手の状態を初期化します。"""
        self.states.clear()

    def update(
        self,
        hand_id: str,
        raw_score: float,
        raw_mode: str,
        timestamp_sec: float,
    ) -> TemporalGripState:
        """
        1手分の生スコアを受け取り、平滑化と状態遷移を行います。
        """

        state = self.states.setdefault(hand_id, TemporalGripState())

        # 初回だけ、生スコアをそのまま平滑化スコアの初期値にします。
        if state.last_update_time is None:
            state.smoothed_score = raw_score
            delta_time = 0.0
        else:
            delta_time = max(0.0, timestamp_sec - state.last_update_time)

            # フレームレートに依存しにくいEMA係数を時定数から計算します。
            # alpha = 1 - exp(-dt / tau)
            tau = max(self.config.ema_time_constant_sec, 1e-6)
            alpha = 1.0 - math.exp(-delta_time / tau)

            state.smoothed_score += alpha * (
                raw_score - state.smoothed_score
            )

        state.last_update_time = timestamp_sec
        state.last_seen_time = timestamp_sec

        # 現在が非把持状態の場合、開始閾値を一定時間超えたか確認します。
        if not state.is_grasping:
            state.exit_candidate_since = None

            if state.smoothed_score >= self.config.enter_threshold:
                if state.enter_candidate_since is None:
                    state.enter_candidate_since = timestamp_sec

                elapsed = timestamp_sec - state.enter_candidate_since

                if elapsed >= self.config.enter_delay_sec:
                    state.is_grasping = True
                    state.mode = raw_mode
                    state.enter_candidate_since = None
            else:
                # 閾値を下回ったら、継続時間の計測をやり直します。
                state.enter_candidate_since = None
                state.mode = "NONE"

        # 現在が把持状態の場合、解除閾値を一定時間下回ったか確認します。
        else:
            state.enter_candidate_since = None

            if state.smoothed_score <= self.config.exit_threshold:
                if state.exit_candidate_since is None:
                    state.exit_candidate_since = timestamp_sec

                elapsed = timestamp_sec - state.exit_candidate_since

                if elapsed >= self.config.exit_delay_sec:
                    state.is_grasping = False
                    state.mode = "NONE"
                    state.exit_candidate_since = None
            else:
                # 解除条件を満たしていない間は、現在の優勢方式へ更新します。
                state.exit_candidate_since = None
                state.mode = raw_mode

        return state

    def remove_missing_hands(
        self,
        detected_hand_ids: Iterable[str],
        timestamp_sec: float,
    ) -> None:
        """
        一定時間見失った手の状態を削除します。

        手を画面外へ出した後、再入場時に以前の把持状態が残ることを防ぎます。
        """

        detected = set(detected_hand_ids)
        delete_targets: List[str] = []

        for hand_id, state in self.states.items():
            if hand_id in detected:
                continue

            if state.last_seen_time is None:
                delete_targets.append(hand_id)
                continue

            missing_duration = timestamp_sec - state.last_seen_time

            if missing_duration >= self.config.missing_reset_sec:
                delete_targets.append(hand_id)

        for hand_id in delete_targets:
            del self.states[hand_id]


# ---------------------------------------------------------------------------
# 依存関係とモデル取得
# ---------------------------------------------------------------------------

def require_dependencies() -> None:
    """
    必須パッケージを確認し、足りない場合は具体的な導入コマンドを示します。
    """

    missing: List[str] = []

    if cv2 is None:
        missing.append("opencv-python")
    if mp is None:
        missing.append("mediapipe")
    if np is None:
        missing.append("numpy")

    if missing:
        package_list = " ".join(missing)
        raise RuntimeError(
            "必須パッケージが不足しています。\n"
            f"次を実行してください:\n"
            f"  {sys.executable} -m pip install {package_list}"
        )


def download_with_progress(url: str, destination: Path) -> None:
    """
    モデルを一時ファイルへダウンロードし、完了後に正式名へ置き換えます。

    途中で停止しても壊れたモデルファイルが残りにくい構成です。
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    # 同じディレクトリに一時ファイルを作り、os.replaceで原子的に置換します。
    with tempfile.NamedTemporaryFile(
        prefix="hand_landmarker_",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        print(f"MediaPipe Hand Landmarkerモデルを取得します:\n  {url}")

        def report_progress(
            block_count: int,
            block_size: int,
            total_size: int,
        ) -> None:
            if total_size <= 0:
                return

            downloaded = min(block_count * block_size, total_size)
            percentage = 100.0 * downloaded / total_size

            print(
                f"\rダウンロード中: {percentage:6.2f}% "
                f"({downloaded / 1024 / 1024:.1f} MB)",
                end="",
                flush=True,
            )

        urllib.request.urlretrieve(
            url,
            temporary_path,
            reporthook=report_progress,
        )
        print()

        # 異常に小さいファイルはHTMLエラー等の可能性があるため拒否します。
        if temporary_path.stat().st_size < 1_000_000:
            raise RuntimeError(
                "取得したモデルファイルが異常に小さいため使用を中止しました。"
            )

        os.replace(temporary_path, destination)

    except (urllib.error.URLError, OSError, RuntimeError) as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Hand Landmarkerモデルの取得に失敗しました。\n"
            f"URL: {url}\n"
            f"保存先: {destination}\n"
            f"原因: {error}"
        ) from error


def ensure_model(model_path: Path) -> Path:
    """
    モデルが存在すればそのまま使い、なければ公式配布元から取得します。
    """

    if model_path.exists() and model_path.stat().st_size >= 1_000_000:
        return model_path

    download_with_progress(MODEL_URL, model_path)
    return model_path


# ---------------------------------------------------------------------------
# MediaPipe結果の解釈
# ---------------------------------------------------------------------------

def get_handedness(
    hand_result: Sequence[Any],
) -> Tuple[str, float]:
    """
    MediaPipeの handedness 結果から左右ラベルと信頼度を取得します。
    """

    if not hand_result:
        return "Unknown", 0.0

    category = hand_result[0]

    # MediaPipeのバージョン差を吸収するため、候補属性を順番に確認します。
    name = (
        getattr(category, "category_name", None)
        or getattr(category, "display_name", None)
        or "Unknown"
    )
    score = float(getattr(category, "score", 0.0))

    return str(name), score


def build_hand_id(
    handedness: str,
    duplicate_count: int,
) -> str:
    """
    左右ラベルを時系列追跡IDとして使います。

    同一ラベルが同じフレームに複数出た場合のみ連番を付けます。
    """
    if duplicate_count == 0:
        return handedness

    return f"{handedness}_{duplicate_count + 1}"


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


# ---------------------------------------------------------------------------
# CSVロギング
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "timestamp_sec",
    "frame_index",
    "hand_id",
    "handedness",
    "handedness_score",
    "is_grasping",
    "mode",
    "object_raw_score",
    "pose_raw_score",
    "smoothed_score",
    "object_id",
    "object_bbox_x",
    "object_bbox_y",
    "object_bbox_w",
    "object_bbox_h",
    "object_contact_score",
    "object_overlap_score",
    "object_fingertip_inside_ratio",
    "power_score",
    "pinch_score",
    "thumb_opposition",
    "pinch_proximity",
    "index_curl",
    "middle_curl",
    "ring_curl",
    "pinky_curl",
    "index_closure",
    "middle_closure",
    "ring_closure",
    "pinky_closure",
    "index_contraction",
    "middle_contraction",
    "ring_contraction",
    "pinky_contraction",
    "index_pip_angle_deg",
    "middle_pip_angle_deg",
    "ring_pip_angle_deg",
    "pinky_pip_angle_deg",
    "index_dip_angle_deg",
    "middle_dip_angle_deg",
    "ring_dip_angle_deg",
    "pinky_dip_angle_deg",
    "palm_width",
]


def decision_to_csv_row(
    timestamp_sec: float,
    frame_index: int,
    decision: GripDecision,
) -> List[Any]:
    """GripDecisionをCSVの1行へ変換します。"""

    features = decision.features

    return [
        f"{timestamp_sec:.6f}",
        frame_index,
        decision.hand_id,
        decision.handedness,
        f"{decision.handedness_score:.6f}",
        int(decision.is_grasping),
        decision.mode,
        f"{decision.raw_score:.6f}",
        f"{decision.pose_score:.6f}",
        f"{decision.smoothed_score:.6f}",
        decision.object_id,
        "" if decision.object_bbox is None else decision.object_bbox[0],
        "" if decision.object_bbox is None else decision.object_bbox[1],
        "" if decision.object_bbox is None else decision.object_bbox[2],
        "" if decision.object_bbox is None else decision.object_bbox[3],
        f"{decision.object_contact_score:.6f}",
        f"{decision.object_overlap_score:.6f}",
        f"{decision.object_fingertip_inside_ratio:.6f}",
        f"{features.power_score:.6f}",
        f"{features.pinch_score:.6f}",
        f"{features.thumb_opposition:.6f}",
        f"{features.pinch_proximity:.6f}",
        f"{features.finger_curl['index']:.6f}",
        f"{features.finger_curl['middle']:.6f}",
        f"{features.finger_curl['ring']:.6f}",
        f"{features.finger_curl['pinky']:.6f}",
        f"{features.finger_closure['index']:.6f}",
        f"{features.finger_closure['middle']:.6f}",
        f"{features.finger_closure['ring']:.6f}",
        f"{features.finger_closure['pinky']:.6f}",
        f"{features.finger_contraction['index']:.6f}",
        f"{features.finger_contraction['middle']:.6f}",
        f"{features.finger_contraction['ring']:.6f}",
        f"{features.finger_contraction['pinky']:.6f}",
        f"{features.pip_angles_deg['index']:.3f}",
        f"{features.pip_angles_deg['middle']:.3f}",
        f"{features.pip_angles_deg['ring']:.3f}",
        f"{features.pip_angles_deg['pinky']:.3f}",
        f"{features.dip_angles_deg['index']:.3f}",
        f"{features.dip_angles_deg['middle']:.3f}",
        f"{features.dip_angles_deg['ring']:.3f}",
        f"{features.dip_angles_deg['pinky']:.3f}",
        f"{features.palm_width:.8f}",
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def prompt_source() -> str:
    """
    CLI引数で映像入力が指定されていない場合、実行後に入力させます。
    """

    print("映像入力ソースを指定してください。")
    print("  Webカメラ: 0")
    print("  動画ファイル: C:\\path\\input.mp4")
    print("  IP Webcam: 192.168.1.20:8080 または http://192.168.1.20:8080/video")

    try:
        source_text = input("映像入力ソース> ").strip()
    except EOFError as error:
        raise RuntimeError(
            "--source が未指定で、実行後入力も読み取れませんでした。"
        ) from error

    if not source_text:
        print("未入力のため Webカメラ 0 を使います。")
        return "0"

    return source_text


def normalize_stream_source(source_text: str) -> str:
    """
    IP Webcam向けの省略入力をOpenCVで開けるURLへ補正します。
    """

    stripped = source_text.strip()

    if not stripped:
        return "0"

    # Windowsの絶対パス C:\... はURL扱いしません。
    if len(stripped) >= 3 and stripped[1] == ":" and stripped[2] in ("\\", "/"):
        return stripped

    parsed = urllib.parse.urlparse(stripped)

    if parsed.scheme in {"http", "https"}:
        if parsed.path in {"", "/"}:
            return urllib.parse.urlunparse(parsed._replace(path="/video"))
        return stripped

    # IP Webcamでは host:port 形式を入力することが多いので /video を補います。
    if ":" in stripped and "/" not in stripped and "\\" not in stripped:
        return f"http://{stripped}/video"

    return stripped


def parse_source(source_text: str) -> int | str:
    """
    source が整数文字列ならカメラ番号、その他なら動画パスまたはURLとして扱います。
    """

    stripped = normalize_stream_source(source_text)

    if stripped.lstrip("+-").isdigit():
        return int(stripped)

    return stripped


def parse_bbox_argument(bbox_text: str) -> BBox:
    """x,y,w,h 形式の文字列をOpenCV矩形へ変換します。"""

    parts = [part.strip() for part in bbox_text.split(",")]

    if len(parts) != 4:
        raise ValueError("--object-roi は x,y,w,h の4整数で指定してください。")

    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError("--object-roi は x,y,w,h の4整数で指定してください。") from error

    if width <= 0 or height <= 0:
        raise ValueError("--object-roi の width と height は1以上にしてください。")

    return (x, y, width, height)


def build_argument_parser() -> argparse.ArgumentParser:
    """コマンドライン引数を定義します。"""

    parser = argparse.ArgumentParser(
        description=(
            "手の21点骨格と追跡物体から、物体を握っているかをリアルタイム判定します。"
        )
    )

    parser.add_argument(
        "--source",
        default=None,
        help=(
            "カメラ番号、動画ファイル、IP Webcam URL。"
            "未指定なら起動後に入力します。"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Hand Landmarkerモデルの保存先。存在しなければ公式配布元から取得します。"
        ),
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="同時に検出する最大手数。既定値: 2",
    )
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="表示だけ左右反転します。推論座標は反転しません。既定値: 有効",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="カメラ入力時に要求する横解像度。既定値: 1280",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="カメラ入力時に要求する縦解像度。既定値: 720",
    )
    parser.add_argument(
        "--enter-threshold",
        type=float,
        default=0.50,
        help="把持開始閾値。既定値: 0.50",
    )
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=0.36,
        help="把持解除閾値。既定値: 0.36",
    )
    parser.add_argument(
        "--enter-delay",
        type=float,
        default=0.18,
        help="把持開始に必要な継続秒数。既定値: 0.18",
    )
    parser.add_argument(
        "--exit-delay",
        type=float,
        default=0.15,
        help="把持解除に必要な継続秒数。既定値: 0.15",
    )
    parser.add_argument(
        "--ema-time-constant",
        type=float,
        default=0.10,
        help="スコア平滑化EMAの時定数秒。既定値: 0.10",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.55,
        help="MediaPipeの手検出信頼度閾値。既定値: 0.55",
    )
    parser.add_argument(
        "--presence-confidence",
        type=float,
        default=0.55,
        help="MediaPipeの手存在信頼度閾値。既定値: 0.55",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.55,
        help="MediaPipeの追跡信頼度閾値。既定値: 0.55",
    )
    parser.add_argument(
        "--object-tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="OpenCVによる物体検出・追跡を有効化します。既定値: 有効",
    )
    parser.add_argument(
        "--object-roi",
        default=None,
        help="初期追跡物体の矩形 x,y,w,h。静止物体を追う場合に指定します。",
    )
    parser.add_argument(
        "--select-object",
        action="store_true",
        help="初回フレームで追跡対象をマウス選択します。",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=6,
        help="同時に追跡する最大物体数。既定値: 6",
    )
    parser.add_argument(
        "--object-min-area",
        type=float,
        default=500.0,
        help="自動検出で採用する最小物体面積px。既定値: 500",
    )
    parser.add_argument(
        "--object-contact-threshold",
        type=float,
        default=0.42,
        help="手と物体が接触しているとみなす目安スコア。既定値: 0.42",
    )
    parser.add_argument(
        "--object-match-threshold",
        type=float,
        default=0.58,
        help="テンプレート追跡の採用閾値。既定値: 0.58",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="指定した場合、フレームごとの特徴量と判定をCSV保存します。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="指定した場合、描画済み動画を保存します。",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="画面表示を無効化します。動画処理やCSV生成向けです。",
    )

    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> DetectorConfig:
    """CLI引数の値域と相互関係を検証します。"""

    probability_arguments = {
        "--enter-threshold": args.enter_threshold,
        "--exit-threshold": args.exit_threshold,
        "--detection-confidence": args.detection_confidence,
        "--presence-confidence": args.presence_confidence,
        "--tracking-confidence": args.tracking_confidence,
        "--object-contact-threshold": args.object_contact_threshold,
        "--object-match-threshold": args.object_match_threshold,
    }

    for argument_name, value in probability_arguments.items():
        if not 0.0 <= value <= 1.0:
            parser.error(f"{argument_name} は 0.0～1.0 で指定してください。")

    if args.enter_threshold <= args.exit_threshold:
        parser.error(
            "--enter-threshold は --exit-threshold より大きくしてください。"
        )

    if args.enter_delay < 0.0 or args.exit_delay < 0.0:
        parser.error("遅延時間は0秒以上で指定してください。")

    if args.ema_time_constant <= 0.0:
        parser.error("--ema-time-constant は0より大きくしてください。")

    if args.max_hands < 1:
        parser.error("--max-hands は1以上で指定してください。")

    if args.camera_width < 1 or args.camera_height < 1:
        parser.error("カメラ解像度は1以上で指定してください。")

    if args.max_objects < 1:
        parser.error("--max-objects は1以上で指定してください。")

    if args.object_min_area < 1.0:
        parser.error("--object-min-area は1以上で指定してください。")

    if args.object_roi is not None:
        try:
            parse_bbox_argument(args.object_roi)
        except ValueError as error:
            parser.error(str(error))

    if args.select_object and args.no_window:
        parser.error("--select-object は --no-window と同時に使えません。")

    if args.no_window and args.output is None and args.csv is None:
        parser.error(
            "--no-window を使う場合は --output または --csv を指定してください。"
        )

    return DetectorConfig(
        enter_threshold=args.enter_threshold,
        exit_threshold=args.exit_threshold,
        enter_delay_sec=args.enter_delay,
        exit_delay_sec=args.exit_delay,
        ema_time_constant_sec=args.ema_time_constant,
        object_tracking_enabled=args.object_tracking,
        object_min_area=args.object_min_area,
        object_max_tracks=args.max_objects,
        object_template_match_threshold=args.object_match_threshold,
        object_contact_threshold=args.object_contact_threshold,
    )


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


def main() -> int:
    """
    プログラム全体を実行します。

    戻り値:
        0: 正常終了
        1: 実行時エラー
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        require_dependencies()
        config = validate_arguments(parser, args)

        # MediaPipe公式モデルを確認し、なければ初回だけ取得します。
        model_path = ensure_model(args.model.expanduser().resolve())

        # 入力元をカメラ番号、動画パス、またはIP Webcam URLへ変換します。
        source_text = args.source if args.source is not None else prompt_source()
        source = parse_source(source_text)

        capture = open_capture(
            source=source,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
        )

        # 動画の場合は取得FPSを使い、取得不能なら30FPSとします。
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 1.0:
            source_fps = 30.0

        # MediaPipe Hand LandmarkerをVIDEOモードで作成します。
        # VIDEOモードは前フレームの追跡結果を使うため、毎フレームの掌検出を省略でき、
        # 単純なIMAGEモードより時系列処理へ適しています。
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=args.max_hands,
            min_hand_detection_confidence=args.detection_confidence,
            min_hand_presence_confidence=args.presence_confidence,
            min_tracking_confidence=args.tracking_confidence,
        )

        temporal_filter = GripTemporalFilter(config)
        object_tracker = (
            ObjectTracker(config)
            if config.object_tracking_enabled
            else None
        )
        initial_object_roi = (
            parse_bbox_argument(args.object_roi)
            if args.object_roi is not None
            else None
        )
        manual_object_initialized = False

        csv_file = None
        csv_writer = None
        video_writer = None

        if args.csv is not None:
            csv_path = args.csv.expanduser().resolve()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = csv_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            )
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_HEADER)

        frame_index = 0
        previous_mediapipe_timestamp_ms = -1
        start_monotonic = time.monotonic()
        fps_measure_start = start_monotonic
        fps_measure_frames = 0
        measured_fps = 0.0
        paused = False
        last_display_frame = None

        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                if paused and last_display_frame is not None:
                    # 一時停止中は推論を進めず、同じ画像を表示します。
                    if not args.no_window:
                        cv2.imshow(
                            "Skeleton Grip Detector",
                            last_display_frame,
                        )
                        key = cv2.waitKey(30) & 0xFF

                        if key in (ord("q"), 27):
                            break
                        if key == ord("p"):
                            paused = False
                        if key == ord("r"):
                            temporal_filter.reset()
                            if object_tracker is not None:
                                object_tracker.reset()
                            manual_object_initialized = False

                    continue

                success, frame_bgr = capture.read()

                if not success:
                    # 動画末尾またはカメラ読取失敗で終了します。
                    break

                frame_index += 1

                # カメラでは実時間、動画ではフレーム番号/FPSを時刻として使います。
                if isinstance(source, int):
                    timestamp_sec = time.monotonic() - start_monotonic
                else:
                    timestamp_sec = (frame_index - 1) / source_fps

                # MediaPipe VIDEOモードは単調増加するミリ秒時刻を要求します。
                mediapipe_timestamp_ms = int(timestamp_sec * 1000.0)
                mediapipe_timestamp_ms = max(
                    previous_mediapipe_timestamp_ms + 1,
                    mediapipe_timestamp_ms,
                )
                previous_mediapipe_timestamp_ms = mediapipe_timestamp_ms

                if (
                    object_tracker is not None
                    and not manual_object_initialized
                    and initial_object_roi is not None
                ):
                    object_tracker.add_manual_track(
                        frame_bgr=frame_bgr,
                        bbox=initial_object_roi,
                        timestamp_sec=timestamp_sec,
                        source="manual",
                    )
                    manual_object_initialized = True

                if (
                    object_tracker is not None
                    and not manual_object_initialized
                    and args.select_object
                ):
                    selected_bbox = select_object_roi(frame_bgr)
                    if selected_bbox is not None:
                        object_tracker.add_manual_track(
                            frame_bgr=frame_bgr,
                            bbox=selected_bbox,
                            timestamp_sec=timestamp_sec,
                            source="manual",
                        )
                    manual_object_initialized = True

                # OpenCVのBGR画像をMediaPipe用のRGB画像へ変換します。
                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                # MediaPipeが要求するImageオブジェクトへ変換します。
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(frame_rgb),
                )

                # 現在フレームの手骨格を推論します。
                result = landmarker.detect_for_video(
                    mp_image,
                    mediapipe_timestamp_ms,
                )

                if object_tracker is not None:
                    tracked_objects = object_tracker.update(
                        frame_bgr=frame_bgr,
                        hand_landmarks_list=result.hand_landmarks,
                        timestamp_sec=timestamp_sec,
                    )
                else:
                    tracked_objects = []

                decisions: List[GripDecision] = []
                detected_hand_ids: List[str] = []
                handedness_counts: Dict[str, int] = {}
                frame_height, frame_width = frame_bgr.shape[:2]

                # 各検出手について特徴量計算と時系列判定を行います。
                for hand_index, image_landmarks in enumerate(
                    result.hand_landmarks
                ):
                    world_landmarks = None

                    if hand_index < len(result.hand_world_landmarks):
                        world_landmarks = result.hand_world_landmarks[
                            hand_index
                        ]

                    handedness_result = (
                        result.handedness[hand_index]
                        if hand_index < len(result.handedness)
                        else []
                    )
                    handedness, handedness_score = get_handedness(
                        handedness_result
                    )

                    duplicate_count = handedness_counts.get(
                        handedness,
                        0,
                    )
                    handedness_counts[handedness] = duplicate_count + 1

                    hand_id = build_hand_id(
                        handedness,
                        duplicate_count,
                    )
                    detected_hand_ids.append(hand_id)

                    features = calculate_grip_features(
                        image_landmarks=image_landmarks,
                        world_landmarks=world_landmarks,
                    )

                    if object_tracker is not None:
                        object_evidence = evaluate_hand_object_evidence(
                            image_landmarks=image_landmarks,
                            features=features,
                            tracked_objects=tracked_objects,
                            frame_width=frame_width,
                            frame_height=frame_height,
                            config=config,
                        )
                        object_raw_score = combine_pose_and_object_scores(
                            features=features,
                            evidence=object_evidence,
                            config=config,
                        )
                        raw_mode = object_grasp_mode(
                            features=features,
                            evidence=object_evidence,
                        )
                    else:
                        object_evidence = ObjectEvidence()
                        object_raw_score = features.raw_score
                        raw_mode = features.raw_mode

                    temporal_state = temporal_filter.update(
                        hand_id=hand_id,
                        raw_score=object_raw_score,
                        raw_mode=raw_mode,
                        timestamp_sec=timestamp_sec,
                    )

                    decision = GripDecision(
                        hand_id=hand_id,
                        handedness=handedness,
                        handedness_score=handedness_score,
                        is_grasping=temporal_state.is_grasping,
                        mode=temporal_state.mode,
                        raw_score=object_raw_score,
                        pose_score=features.raw_score,
                        smoothed_score=temporal_state.smoothed_score,
                        object_id=object_evidence.object_id,
                        object_bbox=object_evidence.bbox,
                        object_contact_score=object_evidence.contact_score,
                        object_overlap_score=object_evidence.overlap_score,
                        object_fingertip_inside_ratio=object_evidence.fingertip_inside_ratio,
                        features=features,
                    )
                    decisions.append(decision)

                    if csv_writer is not None:
                        csv_writer.writerow(
                            decision_to_csv_row(
                                timestamp_sec=timestamp_sec,
                                frame_index=frame_index,
                                decision=decision,
                            )
                        )

                # 一定時間見失った手の履歴を削除します。
                temporal_filter.remove_missing_hands(
                    detected_hand_ids=detected_hand_ids,
                    timestamp_sec=timestamp_sec,
                )

                # 表示用フレームだけ左右反転します。
                # 推論は元画像で行っているため、左右ラベルや座標計算を壊しません。
                if args.mirror:
                    display_frame = cv2.flip(frame_bgr, 1)
                else:
                    display_frame = frame_bgr.copy()

                held_object_ids = {
                    decision.object_id
                    for decision in decisions
                    if decision.is_grasping and decision.object_id
                }
                draw_tracked_objects(
                    frame=display_frame,
                    tracked_objects=tracked_objects,
                    held_object_ids=held_object_ids,
                    mirror=args.mirror,
                )

                # 判定結果と手骨格を同じ順番で描画します。
                for hand_index, decision in enumerate(decisions):
                    image_landmarks = result.hand_landmarks[hand_index]

                    draw_hand_skeleton(
                        frame=display_frame,
                        landmarks=image_landmarks,
                        mirror=args.mirror,
                        grasping=decision.is_grasping,
                    )

                # 実測FPSを約0.5秒ごとに更新します。
                fps_measure_frames += 1
                fps_elapsed = time.monotonic() - fps_measure_start

                if fps_elapsed >= 0.5:
                    measured_fps = fps_measure_frames / fps_elapsed
                    fps_measure_start = time.monotonic()
                    fps_measure_frames = 0

                draw_status_panel(
                    frame=display_frame,
                    decisions=decisions,
                    tracked_objects=tracked_objects,
                    config=config,
                    fps=measured_fps,
                )
                draw_help(display_frame)

                # 出力動画は最初のフレーム寸法が分かった時点で作成します。
                if args.output is not None and video_writer is None:
                    frame_height, frame_width = display_frame.shape[:2]
                    output_fps = (
                        source_fps
                        if not isinstance(source, int)
                        else 30.0
                    )
                    video_writer = create_video_writer(
                        output_path=args.output.expanduser().resolve(),
                        fps=output_fps,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )

                if video_writer is not None:
                    video_writer.write(display_frame)

                last_display_frame = display_frame.copy()

                if not args.no_window:
                    cv2.imshow(
                        "Skeleton Grip Detector",
                        display_frame,
                    )
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), 27):
                        break
                    if key == ord("p"):
                        paused = True
                    if key == ord("r"):
                        temporal_filter.reset()
                        if object_tracker is not None:
                            object_tracker.reset()
                        manual_object_initialized = False

        # while終了後にリソースを閉じます。
        capture.release()

        if video_writer is not None:
            video_writer.release()

        if csv_file is not None:
            csv_file.flush()
            csv_file.close()

        if not args.no_window:
            cv2.destroyAllWindows()

        return 0

    except KeyboardInterrupt:
        print("\nユーザー操作により終了しました。", file=sys.stderr)
        return 0

    except Exception as error:
        print(f"\nエラー: {error}", file=sys.stderr)

        # OpenCVウィンドウが残らないよう、可能なら閉じます。
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
