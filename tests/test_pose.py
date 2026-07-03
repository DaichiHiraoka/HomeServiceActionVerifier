from __future__ import annotations

import math

import numpy as np
import pandas as pd

from home_service_action_verifier.pose import interpolate_skeleton_gaps


def test_interpolate_skeleton_gaps_fills_short_gaps_only() -> None:
    df = pd.DataFrame(
        {
            "frame": range(10),
            "time_s": range(10),
            "lm00_x": [0.0, np.nan, np.nan, 3.0, 4.0, np.nan, np.nan, np.nan, 8.0, 9.0],
        }
    )
    out = interpolate_skeleton_gaps(df, max_gap_frames=2)
    assert out["lm00_x"].iloc[1] == 1.0
    assert out["lm00_x"].iloc[2] == 2.0
    assert math.isnan(out["lm00_x"].iloc[5])
    assert math.isnan(out["lm00_x"].iloc[7])
