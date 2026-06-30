from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .constants import BBox, FINGERTIP_IDS, INDEX_TIP, MIDDLE_TIP, THUMB_TIP
from .geometry import (
    bbox_area,
    bbox_center,
    bbox_distance,
    bbox_intersection_area,
    bbox_iou,
    clamp01,
    clamp_bbox,
    closeness_score,
    expand_bbox,
    point_bbox_distance,
    point_inside_bbox,
)
from .grip import (
    estimate_palm_width_pixels,
    hand_bbox_from_landmarks,
    landmarks_to_pixel_points,
)
from .models import (
    DetectorConfig,
    GripFeatures,
    ObjectDetection,
    ObjectEvidence,
    TrackedObject,
)
from .runtime import cv2, np

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
        self.yolo_model: Optional[Any] = None

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
            label="manual" if source == "manual" else "",
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
                    detection=detections[match_index],
                    frame_gray=frame_gray,
                    timestamp_sec=timestamp_sec,
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
                    detection=ObjectDetection(
                        bbox=bbox,
                        confidence=match_score,
                        source=track.source,
                        label=track.label,
                    ),
                    frame_gray=frame_gray,
                    timestamp_sec=timestamp_sec,
                    update_template=False,
                )
                continue

            track.missed_frames += 1
            track.confidence = max(0.0, track.confidence * 0.82)

        for index in sorted(unmatched_detection_indices):
            if len(self.tracks) >= self.config.object_max_tracks:
                break
            added_track = self.add_manual_track(
                frame_bgr=frame_bgr,
                bbox=detections[index].bbox,
                timestamp_sec=timestamp_sec,
                source=detections[index].source,
            )
            if added_track is not None:
                added_track.confidence = detections[index].confidence
                added_track.label = detections[index].label

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
    ) -> List[ObjectDetection]:
        detections: List[ObjectDetection] = []

        if self.config.object_detector in {"yolo", "hybrid"}:
            detections.extend(self._detect_yolo_candidates(frame_bgr))

        if self.config.object_detector in {"motion", "hybrid"}:
            detections.extend(
                self._detect_motion_candidates(
                    frame_bgr=frame_bgr,
                    hand_landmarks_list=hand_landmarks_list,
                )
            )

        detections.sort(
            key=lambda detection: (
                detection.confidence,
                bbox_area(detection.bbox),
            ),
            reverse=True,
        )
        return detections

    def _detect_motion_candidates(
        self,
        frame_bgr: "np.ndarray",
        hand_landmarks_list: Sequence[Sequence[Any]],
    ) -> List[ObjectDetection]:
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

        detections: List[ObjectDetection] = []

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

            detections.append(
                ObjectDetection(
                    bbox=clamp_bbox((x, y, width, height), frame_width, frame_height),
                    confidence=0.50,
                    source="motion",
                    label="motion",
                )
            )

        detections.sort(key=lambda detection: bbox_area(detection.bbox), reverse=True)
        return detections

    def _detect_yolo_candidates(
        self,
        frame_bgr: "np.ndarray",
    ) -> List[ObjectDetection]:
        model = self._get_yolo_model()
        frame_height, frame_width = frame_bgr.shape[:2]

        results = model.predict(
            frame_bgr,
            conf=self.config.yolo_confidence,
            iou=self.config.yolo_iou,
            imgsz=self.config.yolo_imgsz,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", None) or getattr(model, "names", {})
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        detections: List[ObjectDetection] = []
        for box, confidence, class_id in zip(xyxy, confidences, classes):
            label = self._class_label(names, int(class_id))
            if not self._class_allowed(int(class_id), label):
                continue

            x1, y1, x2, y2 = box.tolist()
            bbox = clamp_bbox(
                (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2 - x1)),
                    int(round(y2 - y1)),
                ),
                frame_width,
                frame_height,
            )
            if bbox_area(bbox) < self.config.object_min_area:
                continue

            detections.append(
                ObjectDetection(
                    bbox=bbox,
                    confidence=float(confidence),
                    source="yolo",
                    label=label,
                )
            )

        return detections

    def _get_yolo_model(self) -> Any:
        if self.yolo_model is not None:
            return self.yolo_model

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "YOLO検出には ultralytics が必要です。"
                " `uv sync` または `uv add ultralytics` を実行してください。"
            ) from error

        self.yolo_model = YOLO(self.config.yolo_model)
        return self.yolo_model

    def _class_allowed(self, class_id: int, label: str) -> bool:
        label_key = label.lower()
        id_key = str(class_id)

        ignored = self.config.yolo_ignore_classes
        if label_key in ignored or id_key in ignored:
            return False

        allowed = self.config.yolo_classes
        if allowed is None:
            return True

        return label_key in allowed or id_key in allowed

    @staticmethod
    def _class_label(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _best_detection_index(
        self,
        track: TrackedObject,
        detections: Sequence[ObjectDetection],
        candidate_indices: set[int],
    ) -> Optional[int]:
        best_index: Optional[int] = None
        best_score = 0.0

        for index in candidate_indices:
            bbox = detections[index].bbox
            iou_score = bbox_iou(track.bbox, bbox)
            distance = bbox_distance(track.bbox, bbox)
            distance_score = closeness_score(
                normalized_distance=distance,
                near_distance=0.0,
                far_distance=self.config.object_association_distance_px,
            )
            confidence_gate = 0.70 + 0.30 * clamp01(detections[index].confidence)
            score = max(iou_score, 0.80 * distance_score) * confidence_gate

            if score > best_score:
                best_score = score
                best_index = index

        if best_score < 0.20:
            return None

        return best_index

    def _update_track(
        self,
        track: TrackedObject,
        detection: ObjectDetection,
        frame_gray: "np.ndarray",
        timestamp_sec: float,
        update_template: bool = True,
    ) -> None:
        bbox = detection.bbox
        old_center = bbox_center(track.bbox)
        new_center = bbox_center(bbox)
        delta_time = max(timestamp_sec - track.last_update_time, 1e-6)

        track.velocity = (
            (new_center[0] - old_center[0]) / delta_time,
            (new_center[1] - old_center[1]) / delta_time,
        )
        track.bbox = bbox
        track.confidence = clamp01(detection.confidence)
        track.source = detection.source
        track.label = detection.label
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
