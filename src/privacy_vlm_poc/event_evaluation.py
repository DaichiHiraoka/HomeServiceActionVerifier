"""Event-level evaluation for scenario experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from privacy_vlm_poc.schemas import DetectionResult, EventEvaluationMetrics, EventToken

POSITIVE_LABELS = {"suspicious", "high_risk"}
NEGATIVE_LABELS = {"normal"}
REVIEW_POLICIES = {"exclude", "positive", "negative"}


def _binary_label(label: str | None, review_policy: str) -> int | None:
    if label in POSITIVE_LABELS:
        return 1
    if label in NEGATIVE_LABELS:
        return 0
    if label == "review":
        if review_policy == "positive":
            return 1
        if review_policy == "negative":
            return 0
        return None
    return None


def _make_output_dir(output_dir: str | Path | None) -> Path | None:
    if output_dir is None:
        return None
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _score_metric(metric_name: str, y_true: list[int], scores: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    try:
        if metric_name == "roc_auc":
            return float(roc_auc_score(y_true, scores))
        if metric_name == "average_precision":
            return float(average_precision_score(y_true, scores))
    except ValueError:
        return None
    return None


def _same_action_lines(rows: list[dict]) -> list[str]:
    dataframe = pd.DataFrame(rows)
    if dataframe.empty or "same_action_pair_id" not in dataframe.columns:
        return ["_No same-action pairs._"]
    pair_rows = dataframe.dropna(subset=["same_action_pair_id"])
    if pair_rows.empty:
        return ["_No same-action pairs._"]

    lines = ["| pair_id | events | correct |", "| --- | --- | --- |"]
    for pair_id, group in pair_rows.groupby("same_action_pair_id"):
        events = ", ".join(
            f"{row.event_id}:{row.ground_truth_label}->{row.predicted_label}" for row in group.itertuples()
        )
        correct = sum(bool(row.is_correct) for row in group.itertuples())
        lines.append(f"| {pair_id} | {events} | {correct}/{len(group)} |")
    return lines


def evaluate_event_predictions(
    events: list[EventToken],
    results: list[DetectionResult],
    output_dir: str | Path | None = None,
    review_policy: str = "exclude",
) -> EventEvaluationMetrics:
    if review_policy not in REVIEW_POLICIES:
        msg = f"review_policy must be one of {sorted(REVIEW_POLICIES)}"
        raise ValueError(msg)

    result_by_id = {result.event_id: result for result in results}
    rows: list[dict] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[float] = []

    for event in events:
        result = result_by_id.get(event.event_id)
        if result is None:
            continue
        true_binary = _binary_label(event.ground_truth_label, review_policy)
        pred_binary = _binary_label(result.predicted_label, review_policy)
        included = true_binary is not None and pred_binary is not None
        is_correct = event.ground_truth_label == result.predicted_label
        rows.append(
            {
                "event_id": event.event_id,
                "ground_truth_label": event.ground_truth_label,
                "predicted_label": result.predicted_label,
                "suspicion_score": result.suspicion_score,
                "included_in_binary_metrics": included,
                "is_correct": is_correct,
                "same_action_pair_id": event.same_action_pair_id,
                "action": event.action,
                "zone": event.zone,
                "object_class": event.object_class,
                "reasons": ";".join(result.reasons),
            }
        )
        if included:
            y_true.append(int(true_binary))
            y_pred.append(int(pred_binary))
            scores.append(result.suspicion_score)

    tn = fp = fn = tp = 0
    if y_true:
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    metrics = EventEvaluationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        precision=float(precision_score(y_true, y_pred, zero_division=0)) if y_true else 0.0,
        recall=float(recall_score(y_true, y_pred, zero_division=0)) if y_true else 0.0,
        f1=float(f1_score(y_true, y_pred, zero_division=0)) if y_true else 0.0,
        roc_auc=_score_metric("roc_auc", y_true, scores),
        average_precision=_score_metric("average_precision", y_true, scores),
        false_alarm_rate=float(fp / (fp + tn)) if (fp + tn) else 0.0,
        num_events=len(y_true),
        num_positive_events=sum(y_true),
        num_negative_events=len(y_true) - sum(y_true),
        notes=f"review_policy={review_policy}; generated_at={datetime.now().isoformat(timespec='seconds')}",
    )

    out_dir = _make_output_dir(output_dir)
    if out_dir is not None:
        metrics.output_dir = out_dir
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(rows).to_csv(out_dir / "per_event.csv", index=False)
        pd.DataFrame(
            [
                {"actual": "negative", "predicted_negative": tn, "predicted_positive": fp},
                {"actual": "positive", "predicted_negative": fn, "predicted_positive": tp},
            ]
        ).to_csv(out_dir / "confusion_matrix.csv", index=False)
        summary = [
            "# Event Evaluation Summary",
            "",
            "```json",
            json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Same Action Different Context",
            "",
            *_same_action_lines(rows),
            "",
        ]
        (out_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return metrics
