from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from .config import load_conditions
from .dataset import build_dataset, feature_columns_for_groups, split_xy
from .evaluate import NEG_LABEL, POS_LABEL, compute_metrics
from .models import make_model
from .schemas import META_COLUMNS


def choose_threshold_for_fpr(
    train_scores: pd.Series | np.ndarray,
    train_labels: pd.Series | np.ndarray,
    target_fpr: float = 0.20,
) -> float:
    """Choose a threshold using only training-fold scores under a target FPR."""
    scores = np.asarray(train_scores, dtype=float)
    labels = np.asarray(train_labels)
    candidates = np.unique(scores)[::-1]
    if len(candidates) == 0:
        return 1.0
    chosen = float(candidates[0])
    for threshold in candidates:
        pred = np.where(scores >= threshold, POS_LABEL, NEG_LABEL)
        metrics = compute_metrics(pd.Series(labels), pd.Series(pred), scores)
        if metrics["fpr"] <= target_fpr:
            chosen = float(threshold)
        else:
            break
    return chosen


def build_early_curves(
    features: pd.DataFrame,
    condition: str,
    ctx_sec: float,
    model_name: str = "rf",
    seed: int = 0,
    target_fpr: float = 0.20,
    conditions_path: str = "configs/conditions.yaml",
) -> pd.DataFrame:
    """Fit LOPO models on full windows and score truncated test-fold windows."""
    full = build_dataset(features, condition, ctx_sec, conditions_path, t_end_rel=float("inf"))
    conditions = load_conditions(conditions_path)
    early = _build_early_dataset(features, conditions[condition], ctx_sec)
    if early.empty:
        raise ValueError("no truncated windows found; run features with --truncate first")

    X, y, groups = split_xy(full)
    logo = LeaveOneGroupOut()
    rows: list[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        model = make_model(model_name, seed)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_scores = _positive_scores(model, X.iloc[train_idx])
        threshold = choose_threshold_for_fpr(train_scores, y.iloc[train_idx], target_fpr)
        test_actors = set(groups.iloc[test_idx])
        early_fold = early[early["actor"].isin(test_actors)].copy()
        if early_fold.empty:
            continue
        feature_cols = [c for c in early_fold.columns if "__" in c]
        scores = _positive_scores(model, early_fold[feature_cols])
        early_fold["fold"] = fold
        early_fold["score"] = scores
        early_fold["threshold"] = threshold
        early_fold["pred_label"] = np.where(scores >= threshold, POS_LABEL, NEG_LABEL)
        rows.append(early_fold)
    if not rows:
        raise ValueError("no early score rows were produced")
    return pd.concat(rows, ignore_index=True)


def summarize_score_curves(curves: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Return per-sequence Time-to-detection in seconds from contact start."""
    rows: list[dict[str, object]] = []
    for seq_id, group in curves.sort_values("t_end_rel").groupby("seq_id"):
        if threshold is None:
            thresholds = group["threshold"] if "threshold" in group else pd.Series(0.5, index=group.index)
            hits = group[group["score"] >= thresholds]
            threshold_value = float(thresholds.iloc[0])
        else:
            hits = group[group["score"] >= threshold]
            threshold_value = threshold
        rows.append(
            {
                "seq_id": seq_id,
                "label": group["label"].iloc[0],
                "threshold": threshold_value,
                "time_to_detection": float(hits["t_end_rel"].iloc[0]) if not hits.empty else np.nan,
                "detected": bool(not hits.empty),
            }
        )
    return pd.DataFrame(rows)


def _build_early_dataset(features: pd.DataFrame, groups: list[str], ctx_sec: float) -> pd.DataFrame:
    selected = features[
        (features["ctx_sec"].astype(float) == float(ctx_sec))
        & np.isfinite(features["t_end_rel"].astype(float))
    ].copy()
    feature_cols = feature_columns_for_groups(selected, groups)
    cols = [c for c in META_COLUMNS if c in selected.columns]
    return selected[cols + feature_cols]


def _positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if POS_LABEL in classes:
            return model.predict_proba(X)[:, classes.index(POS_LABEL)]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1 / (1 + np.exp(-raw))
    return np.asarray(model.predict(X) == POS_LABEL, dtype=float)
