from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .constants import (
    BBox,
    FINGER_JOINTS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    RING_MCP,
    RING_TIP,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
)
from .geometry import (
    angle_deg,
    clamp,
    clamp01,
    clamp_bbox,
    closeness_score,
    flexion_score_from_angle,
    vector_distance,
)
from .models import GripFeatures
from .runtime import np

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
