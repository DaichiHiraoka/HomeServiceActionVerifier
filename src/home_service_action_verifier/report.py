from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_summary_table(rows: list[dict[str, object]], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def plot_score_curve(curves: pd.DataFrame, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for seq_id, group in curves.groupby("seq_id"):
        ax.plot(group["t_end_rel"], group["score"], marker="o", label=str(seq_id))
    ax.set_xlabel("Seconds from contact start")
    ax.set_ylabel("Unnatural score")
    ax.set_ylim(0, 1)
    if curves["seq_id"].nunique() <= 8:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
