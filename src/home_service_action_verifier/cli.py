from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .annotation import read_annotation
from .config import load_config
from .dataset import build_dataset, write_dataset
from .early import build_early_curves, summarize_score_curves
from .evaluate import POS_LABEL, compute_metrics, evaluate_lopo, write_evaluation
from .experiment import run_grid
from .features import compute_feature_row
from .pose import extract_pose, pose_missing_rate
from .tracking import track_annotation
from .video_io import expand_video_inputs, get_video_fps
from .windows import build_contact_windows

LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return int(args.func(args) or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hsav")
    sub = parser.add_subparsers(dest="command", required=True)

    pose = sub.add_parser("pose", help="extract MediaPipe pose to skeleton.parquet")
    pose.add_argument("--video", required=True)
    pose.add_argument("--out", default="data/interim")
    pose.add_argument("--config", default="configs/default.yaml")
    pose.set_defaults(func=cmd_pose)

    track = sub.add_parser("track", help="interpolate object bbox keyframes")
    track.add_argument("--annotation", required=True)
    track.add_argument("--out", default="data/interim")
    track.set_defaults(func=cmd_track)

    features = sub.add_parser("features", help="build window-level features")
    features.add_argument("--annotation", default="data/annotations")
    features.add_argument("--seq", default="all")
    features.add_argument("--ctx", type=float, nargs="+", default=None)
    features.add_argument("--interim", default="data/interim")
    features.add_argument("--out", default="data/processed/features.parquet")
    features.add_argument("--config", default="configs/default.yaml")
    features.add_argument(
        "--truncate",
        type=float,
        nargs="+",
        default=None,
        help="also emit truncated windows for early detection, e.g. -1 0 1 2 3 5 8",
    )
    features.set_defaults(func=cmd_features)

    dataset = sub.add_parser("dataset", help="select condition-specific feature groups")
    dataset.add_argument("--features", default="data/processed/features.parquet")
    dataset.add_argument("--condition", required=True)
    dataset.add_argument("--ctx", type=float, required=True)
    dataset.add_argument("--out", default=None)
    dataset.set_defaults(func=cmd_dataset)

    train = sub.add_parser("train", help="run LOPO training/evaluation")
    train.add_argument("--dataset", default=None)
    train.add_argument("--features", default="data/processed/features.parquet")
    train.add_argument("--condition", default="E")
    train.add_argument("--ctx", type=float, default=5)
    train.add_argument("--model", default="rf", choices=["logreg", "rf", "lgbm"])
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--out", default="outputs/experiments")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="compute metrics from predictions or dataset")
    evaluate.add_argument("--exp", default=None)
    evaluate.add_argument("--predictions", default=None)
    evaluate.add_argument("--dataset", default=None)
    evaluate.add_argument("--model", default="rf", choices=["logreg", "rf", "lgbm"])
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--out", default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    grid = sub.add_parser("grid", help="run condition x context x seed grid")
    grid.add_argument("--features", default="data/processed/features.parquet")
    grid.add_argument("--config", default="configs/default.yaml")
    grid.add_argument("--out", default="outputs/experiments")
    grid.add_argument("--models", nargs="+", default=None, choices=["logreg", "rf", "lgbm"])
    grid.set_defaults(func=cmd_grid)

    early_curves = sub.add_parser("early-curves", help="score truncated windows with fold-local thresholds")
    early_curves.add_argument("--features", default="data/processed/features.parquet")
    early_curves.add_argument("--condition", default="E")
    early_curves.add_argument("--ctx", type=float, default=5)
    early_curves.add_argument("--model", default="rf", choices=["logreg", "rf", "lgbm"])
    early_curves.add_argument("--seed", type=int, default=0)
    early_curves.add_argument("--config", default="configs/default.yaml")
    early_curves.add_argument("--out", default=None)
    early_curves.set_defaults(func=cmd_early_curves)

    early = sub.add_parser("early", help="summarize early-detection score curves")
    early.add_argument("--exp", default=None)
    early.add_argument("--curves", default=None)
    early.add_argument("--threshold", type=float, default=None)
    early.add_argument("--out", default=None)
    early.set_defaults(func=cmd_early)

    qc = sub.add_parser("qc", help="validate annotations and skeleton missing rates")
    qc.add_argument("--annotation", default="data/annotations")
    qc.add_argument("--seq", default="all")
    qc.add_argument("--interim", default="data/interim")
    qc.set_defaults(func=cmd_qc)

    return parser


def cmd_pose(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    videos = expand_video_inputs(args.video)
    for video in videos:
        seq_id = video.stem
        out = Path(args.out) / seq_id / "skeleton.parquet"
        df = extract_pose(
            video,
            out,
            config.pose.max_gap_frames,
            config.pose.model_complexity,
            config.pose.min_detection_confidence,
        )
        LOG.info("pose %s frames=%d missing_rate=%.3f -> %s", seq_id, len(df), pose_missing_rate(df), out)
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    paths = _annotation_paths(args.annotation, "all")
    for path in paths:
        annotation = read_annotation(path)
        out = Path(args.out) / annotation["seq_id"] / "objects.parquet"
        df = track_annotation(path, out)
        LOG.info("track %s rows=%d -> %s", annotation["seq_id"], len(df), out)
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ctx_seconds = args.ctx or config.windows.ctx_seconds
    rows: list[dict[str, object]] = []
    for path in _annotation_paths(args.annotation, args.seq):
        annotation = read_annotation(path)
        seq_dir = Path(args.interim) / annotation["seq_id"]
        skeleton = pd.read_parquet(seq_dir / "skeleton.parquet")
        objects_path = seq_dir / "objects.parquet"
        objects = pd.read_parquet(objects_path) if objects_path.exists() else track_annotation(path, objects_path)
        windows = build_contact_windows(annotation, ctx_seconds)
        if args.truncate is not None:
            windows.extend(build_contact_windows(annotation, ctx_seconds, args.truncate))
        for window in windows:
            rows.append(compute_feature_row(skeleton, objects, annotation, window, config.features))
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    LOG.info("features rows=%d cols=%d -> %s", len(df), len(df.columns), out)
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    features = pd.read_parquet(args.features)
    dataset = build_dataset(features, args.condition, args.ctx)
    out = args.out or f"data/processed/dataset_{args.condition}_ctx{args.ctx:g}.parquet"
    write_dataset(dataset, out)
    LOG.info("dataset rows=%d features=%d -> %s", len(dataset), len(dataset.columns), out)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    if args.dataset:
        dataset = pd.read_parquet(args.dataset)
    else:
        dataset = build_dataset(pd.read_parquet(args.features), args.condition, args.ctx)
    result = evaluate_lopo(dataset, args.model, args.seed)
    exp_id = f"{args.condition}_ctx{args.ctx:g}_{args.model}_seed{args.seed}"
    out = Path(args.out) / exp_id
    write_evaluation(result, out)
    LOG.info("train/eval %s metrics=%s -> %s", exp_id, json.dumps(result["metrics"], allow_nan=True), out)
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    features = pd.read_parquet(args.features)
    summary = run_grid(features, args.out, args.config, models=args.models)
    LOG.info("grid experiments=%d -> %s", len(summary), Path(args.out) / "summary.csv")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    if args.dataset:
        result = evaluate_lopo(pd.read_parquet(args.dataset), args.model, args.seed)
        out = Path(args.out or "outputs/experiments/evaluate")
        write_evaluation(result, out)
        LOG.info("evaluate dataset metrics=%s -> %s", json.dumps(result["metrics"], allow_nan=True), out)
        return 0

    pred_path = Path(args.predictions) if args.predictions else None
    if args.exp:
        exp_dir = Path(args.exp)
        pred_path = exp_dir / "predictions.csv"
    if pred_path is None:
        raise ValueError("evaluate requires --dataset, --predictions, or --exp")
    predictions = pd.read_csv(pred_path)
    if "pred_label" not in predictions and "score" in predictions:
        predictions["pred_label"] = predictions["score"].map(lambda score: POS_LABEL if score >= 0.5 else "normal")
    metrics = compute_metrics(predictions["label"], predictions["pred_label"], predictions["score"])
    out = Path(args.out) if args.out else pred_path.parent / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    LOG.info("evaluate predictions metrics=%s -> %s", json.dumps(metrics, allow_nan=True), out)
    return 0


def cmd_early(args: argparse.Namespace) -> int:
    curves_path = Path(args.curves) if args.curves else None
    if args.exp:
        curves_path = Path(args.exp) / "early_curves.parquet"
    if curves_path is None:
        raise ValueError("early requires --curves or --exp")
    curves = pd.read_parquet(curves_path)
    summary = summarize_score_curves(curves, args.threshold)
    out = Path(args.out) if args.out else curves_path.parent / "early_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    LOG.info("early rows=%d threshold=%.3f -> %s", len(summary), args.threshold, out)
    return 0


def cmd_early_curves(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    features = pd.read_parquet(args.features)
    curves = build_early_curves(
        features,
        args.condition,
        args.ctx,
        args.model,
        args.seed,
        config.eval.target_fpr,
    )
    exp_id = f"{args.condition}_ctx{args.ctx:g}_{args.model}_seed{args.seed}"
    out = Path(args.out) if args.out else Path("outputs/experiments") / exp_id / "early_curves.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    curves.to_parquet(out, index=False)
    LOG.info("early-curves rows=%d -> %s", len(curves), out)
    return 0


def cmd_qc(args: argparse.Namespace) -> int:
    rows: list[dict[str, object]] = []
    for path in _annotation_paths(args.annotation, args.seq):
        annotation = read_annotation(path)
        seq_dir = Path(args.interim) / annotation["seq_id"]
        skeleton_path = seq_dir / "skeleton.parquet"
        objects_path = seq_dir / "objects.parquet"
        missing_rate = None
        skeleton_exists = skeleton_path.exists()
        objects_exists = objects_path.exists()
        if skeleton_path.exists():
            missing_rate = pose_missing_rate(pd.read_parquet(skeleton_path))
        video_path = Path(annotation["video"])
        video_exists = video_path.exists()
        video_fps = None
        fps_match = None
        if video_exists:
            try:
                video_fps = get_video_fps(video_path)
                fps_match = abs(video_fps - float(annotation["fps"])) < 0.01
            except Exception as exc:
                LOG.warning("cannot read fps for %s: %s", video_path, exc)
        contacts_within_track = None
        if objects_exists:
            objects = pd.read_parquet(objects_path)
            if objects.empty:
                contacts_within_track = False
            else:
                max_frame = int(objects["frame"].max())
                contacts_within_track = all(
                    int(contact["end_frame"]) <= max_frame for contact in annotation["contacts"]
                )
        rows.append(
            {
                "seq_id": annotation["seq_id"],
                "actor": annotation["actor"],
                "label": annotation["label"],
                "seq_id_matches_video": Path(annotation["video"]).stem == annotation["seq_id"],
                "video_exists": video_exists,
                "video_fps": video_fps,
                "fps_match": fps_match,
                "skeleton_exists": skeleton_exists,
                "objects_exists": objects_exists,
                "contacts_within_track": contacts_within_track,
                "contacts": len(annotation["contacts"]),
                "objects": len(annotation["objects"]),
                "pose_missing_rate": missing_rate,
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return 0


def _annotation_paths(path: str, seq: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        paths = [p]
    else:
        paths = sorted(p.glob("*.json"))
    if seq != "all":
        paths = [candidate for candidate in paths if candidate.stem == seq]
    if not paths:
        raise FileNotFoundError(f"no annotation files matched path={path} seq={seq}")
    return paths
