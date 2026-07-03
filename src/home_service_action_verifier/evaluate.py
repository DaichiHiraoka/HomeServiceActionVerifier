from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
from sklearn.model_selection import LeaveOneGroupOut

from .dataset import split_xy
from .models import make_model

POS_LABEL = "unnatural"
NEG_LABEL = "normal"


def evaluate_lopo(dataset: pd.DataFrame, model_name: str = "rf", seed: int = 0) -> dict[str, object]:
    X, y, groups = split_xy(dataset)
    logo = LeaveOneGroupOut()
    predictions: list[pd.DataFrame] = []
    per_fold: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        model = make_model(model_name, seed)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pd.Series(model.predict(X.iloc[test_idx]), index=y.iloc[test_idx].index)
        score = _positive_scores(model, X.iloc[test_idx])
        fold_pred = dataset.iloc[test_idx][["seq_id", "actor", "scenario", "label", "contact_id"]].copy()
        fold_pred["fold"] = fold
        fold_pred["pred_label"] = pred.to_numpy()
        fold_pred["score"] = score
        predictions.append(fold_pred)
        per_fold.append({"fold": fold, "actor": groups.iloc[test_idx].iloc[0], **compute_metrics(y.iloc[test_idx], pred, score)})
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return {
        "metrics": compute_metrics(pred_df["label"], pred_df["pred_label"], pred_df["score"]) if not pred_df.empty else {},
        "per_fold": pd.DataFrame(per_fold),
        "predictions": pred_df,
    }


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, scores: pd.Series | np.ndarray) -> dict[str, float]:
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    y_bin = (y_true == POS_LABEL).astype(int)
    score_arr = np.asarray(scores, dtype=float)
    cm = confusion_matrix(y_true, y_pred, labels=[NEG_LABEL, POS_LABEL])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_unnatural": float(precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "recall_unnatural": float(recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "f1_unnatural": float(f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    if len(set(y_bin)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_bin, score_arr))
        metrics["pr_auc"] = float(average_precision_score(y_bin, score_arr))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def bootstrap_f1_ci(
    predictions: pd.DataFrame, seed: int = 0, n: int = 1000, unit_col: str = "seq_id"
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    units = predictions[unit_col].drop_duplicates().to_numpy()
    if len(units) == 0:
        return (float("nan"), float("nan"))
    values = []
    for _ in range(n):
        sampled = rng.choice(units, size=len(units), replace=True)
        sample = pd.concat([predictions[predictions[unit_col] == unit] for unit in sampled], ignore_index=True)
        values.append(f1_score(sample["label"], sample["pred_label"], pos_label=POS_LABEL, zero_division=0))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def write_evaluation(result: dict[str, object], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, allow_nan=True)
        f.write("\n")
    result["per_fold"].to_csv(out / "per_fold.csv", index=False)
    result["predictions"].to_csv(out / "predictions.csv", index=False)


def _positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if POS_LABEL in classes:
            return model.predict_proba(X)[:, classes.index(POS_LABEL)]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1 / (1 + np.exp(-raw))
    return np.asarray(model.predict(X) == POS_LABEL, dtype=float)
