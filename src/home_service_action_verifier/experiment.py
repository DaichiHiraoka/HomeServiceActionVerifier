from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import yaml

from .config import load_conditions, load_config, snapshot_config
from .dataset import build_dataset
from .evaluate import bootstrap_f1_ci, evaluate_lopo, write_evaluation


def run_grid(
    features: pd.DataFrame,
    out_root: str | Path = "outputs/experiments",
    config_path: str | Path = "configs/default.yaml",
    conditions_path: str | Path = "configs/conditions.yaml",
    models: list[str] | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    conditions = load_conditions(conditions_path)
    models = models or ["rf", "lgbm"]
    rows: list[dict[str, object]] = []
    for condition in conditions:
        for ctx in config.windows.ctx_seconds:
            for seed in config.eval.seeds:
                for model_name in models:
                    dataset = build_dataset(features, condition, ctx, conditions_path)
                    exp_id = f"{condition}_ctx{ctx:g}_{model_name}_seed{seed}"
                    out_dir = Path(out_root) / exp_id
                    result = evaluate_lopo(dataset, model_name, seed)
                    if config.eval.bootstrap_n > 0:
                        ci_low, ci_high = bootstrap_f1_ci(
                            result["predictions"], seed=seed, n=config.eval.bootstrap_n
                        )
                        result["metrics"]["f1_unnatural_ci95_low"] = ci_low
                        result["metrics"]["f1_unnatural_ci95_high"] = ci_high
                    write_evaluation(result, out_dir)
                    with (out_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
                        yaml.safe_dump(snapshot_config(config, conditions), f, sort_keys=False, allow_unicode=True)
                    (out_dir / "git_sha.txt").write_text(_git_sha(), encoding="utf-8")
                    rows.append({"exp_id": exp_id, "condition": condition, "ctx_sec": ctx, "model": model_name, "seed": seed, **result["metrics"]})
    summary = pd.DataFrame(rows)
    Path(out_root).mkdir(parents=True, exist_ok=True)
    summary.to_csv(Path(out_root) / "summary.csv", index=False)
    _write_seed_summary(summary, Path(out_root) / "summary_by_seed.csv")
    return summary


def _write_seed_summary(summary: pd.DataFrame, path: Path) -> None:
    metric_cols = [
        col
        for col in summary.select_dtypes(include="number").columns
        if col not in {"ctx_sec", "seed"}
    ]
    if not metric_cols:
        return
    grouped = summary.groupby(["condition", "ctx_sec", "model"], as_index=False)[metric_cols].agg(
        ["mean", "std"]
    )
    grouped.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]
    grouped.to_csv(path, index=False)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
