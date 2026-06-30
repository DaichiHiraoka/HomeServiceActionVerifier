from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .constants import BBox
from .runtime import np

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
    label: str = ""


@dataclass
class ObjectDetection:
    """1フレーム内で検出された物体候補です。"""

    bbox: BBox
    confidence: float
    source: str
    label: str = ""


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

    # 物体候補の検出方式: yolo, motion, hybrid
    object_detector: str = "yolo"

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

    # YOLO設定
    yolo_model: str = "yolo11n.pt"
    yolo_confidence: float = 0.35
    yolo_iou: float = 0.45
    yolo_imgsz: int = 640
    yolo_classes: Optional[Tuple[str, ...]] = None
    yolo_ignore_classes: Tuple[str, ...] = ("person",)
