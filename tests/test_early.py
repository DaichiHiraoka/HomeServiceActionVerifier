from __future__ import annotations

import pandas as pd

from home_service_action_verifier.early import build_early_curves, summarize_score_curves


def test_build_early_curves_scores_truncated_windows_with_fold_thresholds() -> None:
    rows = []
    for actor_idx, actor in enumerate(["a1", "a2", "a3"]):
        rows.append(_row(f"s{actor_idx}n", actor, "normal", float("inf"), 0.0))
        rows.append(_row(f"s{actor_idx}u", actor, "unnatural", float("inf"), 1.0))
        for t_end_rel in [0.0, 1.0]:
            rows.append(_row(f"s{actor_idx}n", actor, "normal", t_end_rel, 0.0))
            rows.append(_row(f"s{actor_idx}u", actor, "unnatural", t_end_rel, 1.0))
    curves = build_early_curves(pd.DataFrame(rows), "E", 5.0, model_name="rf", seed=0)
    assert set(curves["actor"]) == {"a1", "a2", "a3"}
    assert set(curves["t_end_rel"]) == {0.0, 1.0}
    assert "threshold" in curves.columns
    summary = summarize_score_curves(curves)
    assert set(summary["seq_id"]) == {f"s{i}{suffix}" for i in range(3) for suffix in ["n", "u"]}


def _row(seq_id: str, actor: str, label: str, t_end_rel: float, value: float) -> dict[str, object]:
    return {
        "seq_id": seq_id,
        "actor": actor,
        "scenario": "synthetic",
        "label": label,
        "contact_id": f"{seq_id}:0",
        "ctx_sec": 5.0,
        "t_end_rel": t_end_rel,
        "SKEL_SEQ__value": value,
        "OBJ__value": value,
        "REL__value": value,
        "CTX__value": value,
    }
