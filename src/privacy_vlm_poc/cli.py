"""Command line interface for the PoC pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from privacy_vlm_poc.baselines import proposed, vlm_direct
from privacy_vlm_poc.analyzer import analyze_video
from privacy_vlm_poc.event_evaluation import evaluate_event_predictions
from privacy_vlm_poc.evaluation import evaluate_labels
from privacy_vlm_poc.model_selection import UI_OLLAMA_MODELS, ensure_ollama_models, model_candidates, ollama_doctor
from privacy_vlm_poc.rule_engine import run_rule_based
from privacy_vlm_poc.scenario import load_event_tokens, load_work_order, load_zone_config
from privacy_vlm_poc.schemas import DetectionResult, EventToken, MaskMethod, ROI, SamplingMethod, VLMBackend, WorkOrder

console = Console()
SCENARIO_METHODS = {"rule_based", "vlm_direct_full", "vlm_direct_roi", "proposed"}


def _parse_roi(value: str | None) -> ROI | None:
    if value is None or value.strip() == "":
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        msg = "--roi must be formatted as x1,y1,x2,y2"
        raise argparse.ArgumentTypeError(msg)
    try:
        x1, y1, x2, y2 = [int(part) for part in parts]
    except ValueError as exc:
        msg = "--roi values must be integers"
        raise argparse.ArgumentTypeError(msg) from exc
    return ROI(x1=x1, y1=y1, x2=x2, y2=y2)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sampling", choices=[item.value for item in SamplingMethod], default=SamplingMethod.HYBRID.value)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--mask", choices=[item.value for item in MaskMethod], default=MaskMethod.NONE.value)
    parser.add_argument("--roi", type=_parse_roi, default=None, help="Optional ROI formatted as x1,y1,x2,y2")
    parser.add_argument("--vlm-backend", choices=[item.value for item in VLMBackend], default=VLMBackend.MOCK.value)
    parser.add_argument("--vlm-model", default=None, help="Optional per-run model override for the selected backend")
    parser.add_argument("--resize-width", type=int, default=None)


def analyze_command(args: argparse.Namespace) -> int:
    result = analyze_video(
        video_path=args.video,
        sampling_method=args.sampling,
        num_frames=args.num_frames,
        mask_method=args.mask,
        roi=args.roi,
        vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model,
        resize_width=args.resize_width,
    )
    table = Table(title="Analysis Complete")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("run_dir", str(result.run_dir))
    table.add_row("grid", str(result.grid_path))
    table.add_row("result", str(result.result_path))
    table.add_row("report", str(result.report_path))
    table.add_row("selected_frames", ", ".join(str(frame.frame_index) for frame in result.selected_frames))
    console.print(table)
    console.print_json(json.dumps(result.vlm_response.model_dump(mode="json"), ensure_ascii=False))
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    metrics = evaluate_labels(
        labels_csv=args.labels,
        sampling_method=args.sampling,
        num_frames=args.num_frames,
        mask_method=args.mask,
        roi=args.roi,
        vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model,
        resize_width=args.resize_width,
    )
    console.print_json(json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False))
    return 0


def doctor_command(_args: argparse.Namespace) -> int:
    table = Table(title="Research VLM Model Selection")
    table.add_column("Role")
    table.add_column("Model")
    table.add_column("Size")
    table.add_column("Command")
    for candidate in model_candidates():
        table.add_row(candidate.role, candidate.name, candidate.expected_download_size, candidate.command)
    console.print(table)

    result = ollama_doctor()
    console.print_json(json.dumps(result.to_dict(), ensure_ascii=False))
    if result.host_reachable and result.configured_model_present and result.ollama_command_available:
        console.print("[green]Ollama VLM backend is ready.[/green]")
        return 0
    console.print("[yellow]Ollama VLM backend is not fully ready. See notes above.[/yellow]")
    return 1


def bootstrap_command(args: argparse.Namespace) -> int:
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    result = ensure_ollama_models(models)
    console.print_json(json.dumps(result.to_dict(), ensure_ascii=False))
    if result.host_reachable and result.sample_data_ready:
        console.print("[green]Runtime assets are ready.[/green]")
        return 0
    console.print("[yellow]Runtime assets are not fully ready. See notes above.[/yellow]")
    return 1


def _make_scenario_output_dir(root: str | Path = "outputs/runs") -> Path:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    counter = 1
    while path.exists():
        path = base / f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter:03d}"
        counter += 1
    path.mkdir(parents=True)
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _load_predictions(path: str | Path) -> list[DetectionResult]:
    predictions_path = Path(path)
    results: list[DetectionResult] = []
    for line_number, raw_line in enumerate(predictions_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            results.append(DetectionResult.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            msg = f"Invalid prediction JSONL at {predictions_path}:{line_number}: {exc}"
            raise ValueError(msg) from exc
    return results


def _latest_dir() -> Path:
    path = Path("outputs/runs/latest")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_scenario_method(method: str, events: list[EventToken], work_order: WorkOrder) -> list[DetectionResult]:
    if method == "rule_based":
        return run_rule_based(events, work_order)
    if method == "proposed":
        return proposed.run(events, work_order)
    if method in {"vlm_direct_full", "vlm_direct_roi"}:
        return vlm_direct.run_stub(events, method)
    msg = f"Unsupported scenario method: {method}. Supported methods: {sorted(SCENARIO_METHODS)}"
    raise ValueError(msg)


def _write_scenario_summary(
    output_dir: Path,
    method: str,
    events: list[EventToken],
    results: list[DetectionResult],
    predictions_path: Path,
) -> None:
    event_by_id = {event.event_id: event for event in events}
    lines = [
        "# Scenario Analysis Summary",
        "",
        f"- method: `{method}`",
        f"- predictions: `{predictions_path}`",
        "",
        "| event_id | truth | prediction | score | reasons |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for result in results:
        event = event_by_id.get(result.event_id)
        truth = event.ground_truth_label if event else None
        reasons = "; ".join(result.reasons)
        lines.append(
            f"| {result.event_id} | {truth} | {result.predicted_label} | "
            f"{result.suspicion_score:.2f} | {reasons} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def analyze_scenario_command(args: argparse.Namespace) -> int:
    work_order = load_work_order(args.work_order)
    load_zone_config(args.zones)
    events = load_event_tokens(args.annotations)
    results = _run_scenario_method(args.method, events, work_order)

    output_dir = Path(args.output_dir) if args.output_dir else _make_scenario_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "event_predictions.jsonl"
    rows = [result.model_dump(mode="json") for result in results]
    _write_jsonl(predictions_path, rows)
    _write_scenario_summary(output_dir, args.method, events, results, predictions_path)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "video": str(args.video) if args.video else None,
                "work_order": str(args.work_order),
                "zones": str(args.zones),
                "annotations": str(args.annotations),
                "method": args.method,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    latest = _latest_dir()
    _write_jsonl(latest / "event_predictions.jsonl", rows)
    _write_scenario_summary(latest, args.method, events, results, latest / "event_predictions.jsonl")

    console.print_json(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "predictions": str(predictions_path),
                "summary": str(output_dir / "summary.md"),
                "num_events": len(results),
            },
            ensure_ascii=False,
        )
    )
    return 0


def evaluate_events_command(args: argparse.Namespace) -> int:
    events = load_event_tokens(args.annotations)
    predictions = _load_predictions(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else _make_scenario_output_dir("outputs/evaluations")
    metrics = evaluate_event_predictions(events, predictions, output_dir=output_dir, review_policy=args.review_policy)
    console.print_json(json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False))
    return 0


def compare_methods_command(args: argparse.Namespace) -> int:
    work_order = load_work_order(args.work_order)
    load_zone_config(args.zones)
    events = load_event_tokens(args.annotations)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unsupported = sorted(set(methods) - SCENARIO_METHODS)
    if unsupported:
        msg = f"Unsupported methods: {unsupported}. Supported methods: {sorted(SCENARIO_METHODS)}"
        raise ValueError(msg)

    output_dir = Path(args.output_dir) if args.output_dir else _make_scenario_output_dir("outputs/evaluations")
    all_prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    for method in methods:
        method_dir = output_dir / method
        results = _run_scenario_method(method, events, work_order)
        prediction_rows = [result.model_dump(mode="json") for result in results]
        _write_jsonl(method_dir / "event_predictions.jsonl", prediction_rows)
        metrics = evaluate_event_predictions(events, results, output_dir=method_dir, review_policy=args.review_policy)
        metric = metrics.model_dump(mode="json")
        metric["method"] = method
        metric_rows.append(metric)
        for row in prediction_rows:
            row["method"] = method
            all_prediction_rows.append(row)

    pd.DataFrame(metric_rows).to_csv(output_dir / "per_method_metrics.csv", index=False)
    pd.DataFrame(all_prediction_rows).to_csv(output_dir / "per_event_predictions.csv", index=False)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "video": str(args.video) if args.video else None,
                "work_order": str(args.work_order),
                "zones": str(args.zones),
                "annotations": str(args.annotations),
                "methods": methods,
                "review_policy": args.review_policy,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        "# Method Comparison Summary\n\n"
        + _markdown_table(
            metric_rows,
            [
                "method",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "average_precision",
                "false_alarm_rate",
                "num_events",
            ],
        )
        + "\n\nVLM Direct baselines are placeholders until event-window frame extraction is connected.\n",
        encoding="utf-8",
    )
    console.print_json(json.dumps({"output_dir": str(output_dir), "methods": methods}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="privacy-vlm-poc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze one video")
    analyze_parser.add_argument("--video", required=True, type=Path)
    _add_common_options(analyze_parser)
    analyze_parser.set_defaults(func=analyze_command)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a labels.csv file")
    evaluate_parser.add_argument("--labels", required=True, type=Path)
    _add_common_options(evaluate_parser)
    evaluate_parser.set_defaults(func=evaluate_command)

    doctor_parser = subparsers.add_parser("doctor", help="Check local VLM research readiness")
    doctor_parser.set_defaults(func=doctor_command)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Create local defaults, sample data, and pull UI Ollama models",
    )
    bootstrap_parser.add_argument(
        "--models",
        default=",".join(UI_OLLAMA_MODELS),
        help="Comma-separated Ollama models to ensure locally",
    )
    bootstrap_parser.set_defaults(func=bootstrap_command)

    analyze_scenario_parser = subparsers.add_parser("analyze-scenario", help="Analyze annotated scenario events")
    analyze_scenario_parser.add_argument("--video", type=Path, default=None)
    analyze_scenario_parser.add_argument("--work-order", required=True, type=Path)
    analyze_scenario_parser.add_argument("--zones", required=True, type=Path)
    analyze_scenario_parser.add_argument("--annotations", required=True, type=Path)
    analyze_scenario_parser.add_argument("--method", choices=sorted(SCENARIO_METHODS), default="rule_based")
    analyze_scenario_parser.add_argument("--output-dir", type=Path, default=None)
    analyze_scenario_parser.set_defaults(func=analyze_scenario_command)

    evaluate_events_parser = subparsers.add_parser("evaluate-events", help="Evaluate event-level predictions")
    evaluate_events_parser.add_argument("--annotations", required=True, type=Path)
    evaluate_events_parser.add_argument("--predictions", required=True, type=Path)
    evaluate_events_parser.add_argument("--output-dir", type=Path, default=None)
    evaluate_events_parser.add_argument(
        "--review-policy",
        choices=["exclude", "positive", "negative"],
        default="exclude",
    )
    evaluate_events_parser.set_defaults(func=evaluate_events_command)

    compare_methods_parser = subparsers.add_parser("compare-methods", help="Compare event-level scenario methods")
    compare_methods_parser.add_argument("--video", type=Path, default=None)
    compare_methods_parser.add_argument("--work-order", required=True, type=Path)
    compare_methods_parser.add_argument("--zones", required=True, type=Path)
    compare_methods_parser.add_argument("--annotations", required=True, type=Path)
    compare_methods_parser.add_argument("--methods", default="rule_based,vlm_direct_full,proposed")
    compare_methods_parser.add_argument("--output-dir", type=Path, default=None)
    compare_methods_parser.add_argument(
        "--review-policy",
        choices=["exclude", "positive", "negative"],
        default="exclude",
    )
    compare_methods_parser.set_defaults(func=compare_methods_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
