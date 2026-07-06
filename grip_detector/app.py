from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grip_detector.cli import build_argument_parser, parse_bbox_argument, parse_source, prompt_source, validate_arguments
from grip_detector.csv_logging import CSV_HEADER, decision_to_csv_row
from grip_detector.drawing import draw_hand_skeleton, draw_help, draw_status_panel, draw_tracked_objects
from grip_detector.grip import calculate_grip_features
from grip_detector.hand_utils import build_hand_id, get_handedness
from grip_detector.model_io import ensure_model, require_dependencies
from grip_detector.models import GripDecision, ObjectEvidence
from grip_detector.object_tracking import ObjectTracker, combine_pose_and_object_scores, evaluate_hand_object_evidence, object_grasp_mode
from grip_detector.runtime import cv2, mp, np
from grip_detector.temporal import GripTemporalFilter
from grip_detector.video_io import create_video_writer, open_capture, select_object_roi

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
                        object_raw_score = 0.0
                        raw_mode = "NO_OBJECT_TRACKING"

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
                        object_motion_score=object_evidence.motion_score,
                        object_speed_px_s=object_evidence.speed_px_s,
                        object_displacement_px=object_evidence.displacement_px,
                        object_settled=object_evidence.settled,
                        object_vacancy_similarity=object_evidence.vacancy_similarity,
                        object_identity_similarity=object_evidence.identity_similarity,
                        object_motion_valid=object_evidence.motion_valid,
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
