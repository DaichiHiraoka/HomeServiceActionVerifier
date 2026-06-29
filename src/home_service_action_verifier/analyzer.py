"""Contextual suspicious object-manipulation scoring."""

from __future__ import annotations

import math
from collections import defaultdict

from home_service_action_verifier.models import DetectionResult, ObjectFrame, SkeletonFrame, TaskContext, Thresholds


def classify_score(score: float, thresholds: Thresholds) -> str:
    if score < thresholds.review_threshold:
        return "normal"
    if score < thresholds.suspicious_threshold:
        return "review"
    if score < thresholds.high_risk_threshold:
        return "suspicious"
    return "high_risk"


def nearest_skeleton(frame: ObjectFrame, skeleton_frames: list[SkeletonFrame]) -> SkeletonFrame | None:
    if not skeleton_frames:
        return None
    return min(skeleton_frames, key=lambda item: abs(item.timestamp - frame.timestamp))


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _hand_distance(frame: ObjectFrame, skeleton: SkeletonFrame | None) -> float | None:
    if skeleton is None:
        return None
    left = _distance(frame.center_x, frame.center_y, skeleton.left_wrist_x, skeleton.left_wrist_y)
    right = _distance(frame.center_x, frame.center_y, skeleton.right_wrist_x, skeleton.right_wrist_y)
    return min(left, right)


def _body_distance(frame: ObjectFrame, skeleton: SkeletonFrame | None) -> float | None:
    if skeleton is None:
        return None
    return _distance(frame.center_x, frame.center_y, skeleton.torso_x, skeleton.torso_y)


def _resolve_role(frame: ObjectFrame, context: TaskContext) -> str:
    role = frame.role.strip().lower()
    if role:
        return role
    if frame.label in context.target_objects:
        return "target"
    if frame.label in context.worker_objects:
        return "worker"
    if frame.label in context.private_objects:
        return "private"
    if frame.label in context.high_risk_objects:
        return "high_risk"
    return "unknown"


def _track_movement(track: list[ObjectFrame]) -> float:
    visible = [frame for frame in track if frame.visible]
    if len(visible) < 2:
        return 0.0
    first = visible[0]
    last = visible[-1]
    return _distance(first.center_x, first.center_y, last.center_x, last.center_y)


def _returned_to_start(track: list[ObjectFrame], threshold: float) -> bool:
    visible = [frame for frame in track if frame.visible]
    if len(visible) < 2:
        return True
    first = visible[0]
    last = visible[-1]
    return _distance(first.center_x, first.center_y, last.center_x, last.center_y) <= threshold


def _left_work_area(track: list[ObjectFrame], context: TaskContext) -> bool:
    if not context.work_areas:
        return False
    visible = [frame for frame in track if frame.visible]
    if not visible:
        return False
    return any(frame.zone and frame.zone not in context.work_areas for frame in visible)


def _entered_private_area(track: list[ObjectFrame], context: TaskContext) -> bool:
    if not context.private_areas:
        return False
    return any(frame.zone in context.private_areas for frame in track if frame.visible)


def _disappeared(track: list[ObjectFrame]) -> bool:
    if not track:
        return False
    visible_count = sum(1 for frame in track if frame.visible)
    return visible_count > 0 and not track[-1].visible


def analyze_tracks(
    skeleton_frames: list[SkeletonFrame],
    object_frames: list[ObjectFrame],
    context: TaskContext,
) -> list[DetectionResult]:
    tracks: dict[str, list[ObjectFrame]] = defaultdict(list)
    for frame in object_frames:
        tracks[frame.object_id].append(frame)

    results = [
        analyze_one_track(track_id, sorted(track, key=lambda item: item.timestamp), skeleton_frames, context)
        for track_id, track in sorted(tracks.items())
    ]
    return sorted(results, key=lambda item: (-item.suspicion_score, item.object_id))


def analyze_one_track(
    object_id: str,
    track: list[ObjectFrame],
    skeleton_frames: list[SkeletonFrame],
    context: TaskContext,
) -> DetectionResult:
    if not track:
        msg = "track must contain at least one frame"
        raise ValueError(msg)

    thresholds = context.thresholds
    first = track[0]
    role = _resolve_role(first, context)
    movement = _track_movement(track)
    returned = _returned_to_start(track, thresholds.return_distance)
    left_work_area = _left_work_area(track, context)
    private_area = _entered_private_area(track, context)
    disappeared = _disappeared(track)

    hand_distances: list[tuple[float, float]] = []
    body_distances: list[tuple[float, float]] = []
    for frame in track:
        if not frame.visible:
            continue
        skeleton = nearest_skeleton(frame, skeleton_frames)
        hand_distance = _hand_distance(frame, skeleton)
        body_distance = _body_distance(frame, skeleton)
        if hand_distance is not None:
            hand_distances.append((frame.timestamp, hand_distance))
        if body_distance is not None:
            body_distances.append((frame.timestamp, body_distance))

    first_touch_time = next(
        (timestamp for timestamp, distance in hand_distances if distance <= thresholds.touch_distance),
        None,
    )
    close_to_body = any(distance <= thresholds.body_distance for _, distance in body_distances)
    moved = movement >= thresholds.movement_distance
    touched_and_moved = first_touch_time is not None and moved

    score = 0.0
    reasons: list[str] = []

    if role in {"private", "resident"} or first.label in context.private_objects:
        score += 0.22
        reasons.append("対象が作業対象外の私物である")
    if role == "high_risk" or first.label in context.high_risk_objects:
        score += 0.20
        reasons.append("高リスク物体に該当する")
    if private_area:
        score += 0.18
        reasons.append("私的エリアで物体操作が起きている")
    if touched_and_moved:
        score += 0.18
        reasons.append("手先が近づいた後に物体が移動している")
    if moved and not returned:
        score += 0.15
        reasons.append("物体が元の位置に戻っていない")
    if left_work_area:
        score += 0.14
        reasons.append("物体が作業エリア外へ移動している")
    if close_to_body and touched_and_moved:
        score += 0.10
        reasons.append("物体が身体側に近づいている")
    if disappeared:
        score += 0.12
        reasons.append("追跡中の物体が見えなくなっている")

    if role in {"target", "worker"} and returned and not private_area and not left_work_area:
        score = max(0.0, score - 0.18)
        reasons.append("作業対象または作業者所有物で、元の位置に戻っている")

    final_score = max(0.0, min(score, 1.0))
    predicted_label = classify_score(final_score, thresholds)
    first_alert_time = _first_alert_time(track, skeleton_frames, context, thresholds.suspicious_threshold)
    time_to_detection = None
    if first_touch_time is not None and first_alert_time is not None:
        time_to_detection = round(first_alert_time - first_touch_time, 3)

    if not reasons:
        reasons.append("作業文脈から大きく外れる動きは見つからない")

    evidence = {
        "movement_distance": round(movement, 3),
        "returned_to_start": returned,
        "left_work_area": left_work_area,
        "private_area": private_area,
        "disappeared": disappeared,
        "first_touch_time": first_touch_time,
    }

    return DetectionResult(
        object_id=object_id,
        object_label=first.label,
        role=role,
        predicted_label=predicted_label,
        suspicion_score=round(final_score, 3),
        first_touch_time=first_touch_time,
        first_alert_time=first_alert_time,
        time_to_detection=time_to_detection,
        reasons=reasons,
        evidence=evidence,
    )


def _first_alert_time(
    track: list[ObjectFrame],
    skeleton_frames: list[SkeletonFrame],
    context: TaskContext,
    threshold: float,
) -> float | None:
    if not track:
        return None
    for index in range(1, len(track) + 1):
        partial = track[:index]
        result = analyze_one_track_without_alert(partial[0].object_id, partial, skeleton_frames, context)
        if result.suspicion_score >= threshold:
            return partial[-1].timestamp
    return None


def analyze_one_track_without_alert(
    object_id: str,
    track: list[ObjectFrame],
    skeleton_frames: list[SkeletonFrame],
    context: TaskContext,
) -> DetectionResult:
    """Score a partial track without recursively computing first alert time."""

    thresholds = context.thresholds
    first = track[0]
    role = _resolve_role(first, context)
    movement = _track_movement(track)
    returned = _returned_to_start(track, thresholds.return_distance)
    left_work_area = _left_work_area(track, context)
    private_area = _entered_private_area(track, context)
    disappeared = _disappeared(track)

    hand_distances = []
    body_distances = []
    for frame in track:
        if not frame.visible:
            continue
        skeleton = nearest_skeleton(frame, skeleton_frames)
        hand_distance = _hand_distance(frame, skeleton)
        body_distance = _body_distance(frame, skeleton)
        if hand_distance is not None:
            hand_distances.append((frame.timestamp, hand_distance))
        if body_distance is not None:
            body_distances.append((frame.timestamp, body_distance))

    first_touch_time = next(
        (timestamp for timestamp, distance in hand_distances if distance <= thresholds.touch_distance),
        None,
    )
    close_to_body = any(distance <= thresholds.body_distance for _, distance in body_distances)
    moved = movement >= thresholds.movement_distance
    touched_and_moved = first_touch_time is not None and moved

    score = 0.0
    reasons: list[str] = []
    if role in {"private", "resident"} or first.label in context.private_objects:
        score += 0.22
        reasons.append("対象が作業対象外の私物である")
    if role == "high_risk" or first.label in context.high_risk_objects:
        score += 0.20
        reasons.append("高リスク物体に該当する")
    if private_area:
        score += 0.18
        reasons.append("私的エリアで物体操作が起きている")
    if touched_and_moved:
        score += 0.18
        reasons.append("手先が近づいた後に物体が移動している")
    if moved and not returned:
        score += 0.15
        reasons.append("物体が元の位置に戻っていない")
    if left_work_area:
        score += 0.14
        reasons.append("物体が作業エリア外へ移動している")
    if close_to_body and touched_and_moved:
        score += 0.10
        reasons.append("物体が身体側に近づいている")
    if disappeared:
        score += 0.12
        reasons.append("追跡中の物体が見えなくなっている")
    if role in {"target", "worker"} and returned and not private_area and not left_work_area:
        score = max(0.0, score - 0.18)
        reasons.append("作業対象または作業者所有物で、元の位置に戻っている")

    final_score = max(0.0, min(score, 1.0))
    if not reasons:
        reasons.append("作業文脈から大きく外れる動きは見つからない")
    return DetectionResult(
        object_id=object_id,
        object_label=first.label,
        role=role,
        predicted_label=classify_score(final_score, thresholds),
        suspicion_score=round(final_score, 3),
        first_touch_time=first_touch_time,
        first_alert_time=None,
        time_to_detection=None,
        reasons=reasons,
        evidence={
            "movement_distance": round(movement, 3),
            "returned_to_start": returned,
            "left_work_area": left_work_area,
            "private_area": private_area,
            "disappeared": disappeared,
            "first_touch_time": first_touch_time,
        },
    )

