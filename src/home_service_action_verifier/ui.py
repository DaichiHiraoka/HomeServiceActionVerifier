"""Streamlit UI for event-level home-service action verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from home_service_action_verifier.baselines import proposed, token_only
from home_service_action_verifier.event_evaluation import evaluate_event_predictions
from home_service_action_verifier.rule_engine import run_rule_based
from home_service_action_verifier.scenario import load_event_tokens, load_work_order, load_zone_config
from home_service_action_verifier.schemas import (
    DetectionResult,
    EventEvaluationMetrics,
    EventToken,
    WorkOrder,
    ZoneConfig,
)

UPLOADFILES_ROOT = Path("uploadfiles")
TEMPLATE_SCENARIO_DIR = UPLOADFILES_ROOT / "template"
SCENARIO_FILE_NAMES = {
    "work_order": "work_order.json",
    "zones": "zones.json",
    "annotations": "annotations.jsonl",
    "video_path": "video_path.txt",
}
LATEST_DIR = Path("outputs/runs/latest")
UI_METHODS = ("rule_based", "token_only", "proposed")


@dataclass(frozen=True)
class ScenarioAssetPaths:
    scenario_dir: Path
    work_order: Path
    zones: Path
    annotations: Path
    video: Path | None
    video_path_file: Path


@dataclass
class UIExperimentResult:
    output_dir: Path
    asset_paths: ScenarioAssetPaths | None
    work_order: WorkOrder
    zone_config: ZoneConfig
    events: list[EventToken]
    results_by_method: dict[str, list[DetectionResult]]
    metrics_by_method: dict[str, EventEvaluationMetrics]
    prediction_paths: dict[str, Path]
    evaluation_dirs: dict[str, Path]


def _make_ui_output_dir(root: str | Path = "outputs/runs") -> Path:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"ui_scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    counter = 1
    while path.exists():
        path = base / f"ui_scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter:03d}"
        counter += 1
    path.mkdir(parents=True)
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def list_upload_scenario_dirs(root: str | Path = UPLOADFILES_ROOT) -> list[Path]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(path for path in base.iterdir() if path.is_dir())


def _required_file(scenario_dir: Path, key: str) -> Path:
    path = scenario_dir / SCENARIO_FILE_NAMES[key]
    if not path.exists():
        expected = ", ".join(SCENARIO_FILE_NAMES.values())
        msg = f"{scenario_dir} must contain {SCENARIO_FILE_NAMES[key]}. Expected files: {expected}"
        raise FileNotFoundError(msg)
    return path


def _read_video_path(scenario_dir: Path, video_path_file: Path) -> Path | None:
    for raw_line in video_path_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        return path if path.is_absolute() else (scenario_dir / path)
    return None


def resolve_scenario_asset_paths(scenario_dir: str | Path) -> ScenarioAssetPaths:
    base = Path(scenario_dir)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Scenario directory does not exist: {base}")
    work_order = _required_file(base, "work_order")
    zones = _required_file(base, "zones")
    annotations = _required_file(base, "annotations")
    video_path_file = _required_file(base, "video_path")
    video = _read_video_path(base, video_path_file)
    return ScenarioAssetPaths(
        scenario_dir=base,
        work_order=work_order,
        zones=zones,
        annotations=annotations,
        video=video,
        video_path_file=video_path_file,
    )


def _run_method(method: str, events: list[EventToken], work_order: WorkOrder) -> list[DetectionResult]:
    if method == "rule_based":
        return run_rule_based(events, work_order)
    if method == "token_only":
        return token_only.run(events, work_order)
    if method == "proposed":
        return proposed.run(events, work_order)
    msg = f"Unsupported UI method: {method}. UI supports: {list(UI_METHODS)}"
    raise ValueError(msg)


def _write_analysis_summary(
    path: Path,
    method: str,
    events: list[EventToken],
    results: list[DetectionResult],
) -> None:
    event_by_id = {event.event_id: event for event in events}
    lines = [
        "# UI Scenario Analysis",
        "",
        f"- method: `{method}`",
        "",
        "| event_id | truth | prediction | score | reasons |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for result in results:
        event = event_by_id.get(result.event_id)
        truth = event.ground_truth_label if event else ""
        reasons = "; ".join(result.reasons)
        lines.append(
            f"| {result.event_id} | {truth} | {result.predicted_label} | "
            f"{result.suspicion_score:.2f} | {reasons} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_comparison_summary(output_dir: Path, rows: list[dict]) -> None:
    columns = [
        "method",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "false_alarm_rate",
        "same_action_different_context_accuracy",
        "num_events",
    ]
    lines = [
        "# UI Method Comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = ["" if row.get(column) is None else str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_latest_prediction(path: Path) -> None:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    (LATEST_DIR / "event_predictions.jsonl").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def run_ui_experiment(
    work_order_path: str | Path,
    zones_path: str | Path,
    annotations_path: str | Path,
    methods: list[str],
    output_root: str | Path = "outputs/runs",
    review_policy: str = "exclude",
    video_path: str | Path | None = None,
    asset_paths: ScenarioAssetPaths | None = None,
) -> UIExperimentResult:
    if not methods:
        raise ValueError("At least one method must be selected.")
    unsupported = sorted(set(methods) - set(UI_METHODS))
    if unsupported:
        raise ValueError(f"Unsupported UI methods: {unsupported}. UI supports: {list(UI_METHODS)}")

    work_order = load_work_order(work_order_path)
    zone_config = load_zone_config(zones_path)
    events = load_event_tokens(annotations_path)
    output_dir = _make_ui_output_dir(output_root)

    results_by_method: dict[str, list[DetectionResult]] = {}
    metrics_by_method: dict[str, EventEvaluationMetrics] = {}
    prediction_paths: dict[str, Path] = {}
    evaluation_dirs: dict[str, Path] = {}
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for method in methods:
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        results = _run_method(method, events, work_order)
        prediction_path = method_dir / "event_predictions.jsonl"
        _write_jsonl(prediction_path, [result.model_dump(mode="json") for result in results])
        _write_analysis_summary(method_dir / "analysis_summary.md", method, events, results)

        evaluation_dir = method_dir / "evaluation"
        metrics = evaluate_event_predictions(events, results, output_dir=evaluation_dir, review_policy=review_policy)
        metric = metrics.model_dump(mode="json")
        metric["method"] = method
        metric_rows.append(metric)
        for result in results:
            row = result.model_dump(mode="json")
            row["method"] = method
            prediction_rows.append(row)

        results_by_method[method] = results
        metrics_by_method[method] = metrics
        prediction_paths[method] = prediction_path
        evaluation_dirs[method] = evaluation_dir

    pd.DataFrame(metric_rows).to_csv(output_dir / "per_method_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "per_event_predictions.csv", index=False)
    _write_comparison_summary(output_dir, metric_rows)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "video": str(video_path) if video_path else None,
                "work_order": str(work_order_path),
                "zones": str(zones_path),
                "annotations": str(annotations_path),
                "methods": methods,
                "review_policy": review_policy,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _copy_latest_prediction(prediction_paths[methods[0]])

    return UIExperimentResult(
        output_dir=output_dir,
        asset_paths=asset_paths,
        work_order=work_order,
        zone_config=zone_config,
        events=events,
        results_by_method=results_by_method,
        metrics_by_method=metrics_by_method,
        prediction_paths=prediction_paths,
        evaluation_dirs=evaluation_dirs,
    )


def run_ui_directory_experiment(
    scenario_dir: str | Path,
    methods: list[str],
    output_root: str | Path = "outputs/runs",
    review_policy: str = "exclude",
) -> UIExperimentResult:
    assets = resolve_scenario_asset_paths(scenario_dir)
    return run_ui_experiment(
        work_order_path=assets.work_order,
        zones_path=assets.zones,
        annotations_path=assets.annotations,
        methods=methods,
        output_root=output_root,
        review_policy=review_policy,
        video_path=assets.video,
        asset_paths=assets,
    )


def _event_rows(events: list[EventToken]) -> pd.DataFrame:
    return pd.DataFrame([event.model_dump(mode="json") for event in events])


def _prediction_rows(events: list[EventToken], results: list[DetectionResult]) -> pd.DataFrame:
    event_by_id = {event.event_id: event for event in events}
    rows = []
    for result in results:
        event = event_by_id.get(result.event_id)
        rows.append(
            {
                "event_id": result.event_id,
                "truth": event.ground_truth_label if event else None,
                "prediction": result.predicted_label,
                "score": result.suspicion_score,
                "action": event.action if event else None,
                "zone": event.zone if event else None,
                "object_class": event.object_class if event else None,
                "object_owner": event.object_owner if event else None,
                "same_action_pair_id": event.same_action_pair_id if event else None,
                "reasons": "; ".join(result.reasons),
            }
        )
    return pd.DataFrame(rows)


def _metric_rows(result: UIExperimentResult) -> pd.DataFrame:
    rows = []
    for method, metrics in result.metrics_by_method.items():
        row = metrics.model_dump(mode="json")
        row["method"] = method
        rows.append(row)
    return pd.DataFrame(rows)


def _render_metric_cards(metrics: EventEvaluationMetrics) -> None:
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metrics.accuracy:.2f}")
    cols[1].metric("Precision", f"{metrics.precision:.2f}")
    cols[2].metric("Recall", f"{metrics.recall:.2f}")
    cols[3].metric("F1", f"{metrics.f1:.2f}")
    cols[4].metric("False Alarm", f"{metrics.false_alarm_rate:.2f}")
    cols = st.columns(3)
    cols[0].metric("ROC-AUC", "-" if metrics.roc_auc is None else f"{metrics.roc_auc:.2f}")
    cols[1].metric("AP", "-" if metrics.average_precision is None else f"{metrics.average_precision:.2f}")
    value = metrics.same_action_different_context_accuracy
    cols[2].metric("Same Action Context", "-" if value is None else f"{value:.2f}")


def _render_scenario_overview(result: UIExperimentResult) -> None:
    work_order = result.work_order
    st.subheader("Scenario")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenario ID", work_order.scenario_id)
    c2.metric("Events", str(len(result.events)))
    c3.metric("Zones", str(len(result.zone_config.zones)))
    c4.metric("Methods", str(len(result.results_by_method)))
    if result.asset_paths is not None:
        st.markdown("#### Loaded Folder")
        st.code(str(result.asset_paths.scenario_dir), language="text")
        st.write(
            {
                "work_order": str(result.asset_paths.work_order),
                "zones": str(result.asset_paths.zones),
                "annotations": str(result.asset_paths.annotations),
                "video": str(result.asset_paths.video) if result.asset_paths.video else None,
            }
        )

    st.markdown("#### Work Order")
    st.json(work_order.model_dump(mode="json"))
    st.markdown("#### Zones")
    st.dataframe(pd.DataFrame([zone.model_dump(mode="json") for zone in result.zone_config.zones]), use_container_width=True)


def _render_method_view(result: UIExperimentResult) -> None:
    method = st.selectbox("表示する手法", list(result.results_by_method.keys()))
    metrics = result.metrics_by_method[method]
    _render_metric_cards(metrics)
    st.markdown("#### Predictions")
    st.dataframe(_prediction_rows(result.events, result.results_by_method[method]), use_container_width=True)

    summary_path = result.evaluation_dirs[method] / "summary.md"
    if summary_path.exists():
        with st.expander("Evaluation summary.md"):
            st.markdown(summary_path.read_text(encoding="utf-8"))


def _render_comparison(result: UIExperimentResult) -> None:
    metrics = _metric_rows(result)
    st.subheader("Method Comparison")
    st.dataframe(metrics, use_container_width=True)
    chart_data = metrics.set_index("method")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("F1 by method")
        st.bar_chart(chart_data[["f1"]])
    with c2:
        st.caption("False alarm rate by method")
        st.bar_chart(chart_data[["false_alarm_rate"]])


def _render_files(result: UIExperimentResult) -> None:
    st.subheader("Output Files")
    st.code(str(result.output_dir), language="text")
    files = sorted(path for path in result.output_dir.rglob("*") if path.is_file())
    for path in files:
        rel = path.relative_to(result.output_dir)
        st.write(f"`{rel}`")
        if path.suffix.lower() in {".json", ".jsonl", ".csv", ".md"}:
            st.download_button(
                label=f"Download {rel}",
                data=path.read_text(encoding="utf-8"),
                file_name=path.name,
                mime="text/plain",
                key=f"download-{rel}",
            )


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
              radial-gradient(circle at top left, rgba(28, 78, 216, 0.16), transparent 34rem),
              linear-gradient(135deg, #f8f5ee 0%, #edf3f0 52%, #f7efe0 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #13231f 0%, #263b33 100%);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f9f4e8;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: #13231f;
        }
        .hsv-hero {
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(19, 35, 31, 0.16);
            border-radius: 1.2rem;
            background: rgba(255, 252, 244, 0.78);
            box-shadow: 0 1.2rem 3rem rgba(20, 32, 28, 0.08);
        }
        .hsv-hero h1 {
            margin-bottom: 0.2rem;
            letter-spacing: -0.04em;
        }
        .hsv-note {
            color: #4b5c54;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_app() -> None:
    st.set_page_config(page_title="Home Service Action Verifier", layout="wide")
    _inject_style()
    st.markdown(
        """
        <div class="hsv-hero">
          <h1>Home Service Action Verifier</h1>
          <div class="hsv-note">
            作業票、ゾーン、イベントトークンを照合し、イベント単位で許可外行動候補を評価します。
            Rule-Based / Token Only / Proposed の研究比較UIです。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Scenario Folder")
        scenario_dirs = list_upload_scenario_dirs()
        options = [str(path) for path in scenario_dirs] or [str(TEMPLATE_SCENARIO_DIR)]
        selected_dir = st.selectbox("uploadfiles/<folder>", options, index=0)
        scenario_dir_text = st.text_input("Scenario directory", value=selected_dir)
        with st.expander("Required files"):
            st.markdown(
                "\n".join(
                    [
                        f"- `{SCENARIO_FILE_NAMES['work_order']}`",
                        f"- `{SCENARIO_FILE_NAMES['zones']}`",
                        f"- `{SCENARIO_FILE_NAMES['annotations']}`",
                        f"- `{SCENARIO_FILE_NAMES['video_path']}`",
                    ]
                )
            )

        st.header("Experiment")
        methods = st.multiselect("Methods", list(UI_METHODS), default=["rule_based", "token_only", "proposed"])
        review_policy = st.selectbox("Review policy", ["exclude", "positive", "negative"], index=0)
        output_root = st.text_input("Output root", value="outputs/runs")
        run_button = st.button("Run Event-Level Experiment", type="primary", use_container_width=True)

    if run_button:
        try:
            with st.spinner("イベント単位の解析・評価を実行中..."):
                st.session_state["ui_experiment_result"] = run_ui_directory_experiment(
                    scenario_dir=scenario_dir_text,
                    methods=list(methods),
                    output_root=output_root,
                    review_policy=review_policy,
                )
        except Exception as exc:  # noqa: BLE001 - UI boundary should show recoverable errors.
            st.exception(exc)

    result = st.session_state.get("ui_experiment_result")
    if not isinstance(result, UIExperimentResult):
        st.info(
            "`uploadfiles/template` と同じ構造のフォルダを `uploadfiles/` 配下に置き、"
            "`Run Event-Level Experiment` を押してください。"
        )
        return

    tab_overview, tab_events, tab_method, tab_compare, tab_files = st.tabs(
        ["Overview", "Events", "Method Detail", "Comparison", "Files"]
    )
    with tab_overview:
        _render_scenario_overview(result)
    with tab_events:
        st.subheader("Event Tokens")
        st.dataframe(_event_rows(result.events), use_container_width=True)
    with tab_method:
        _render_method_view(result)
    with tab_compare:
        _render_comparison(result)
    with tab_files:
        _render_files(result)


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
