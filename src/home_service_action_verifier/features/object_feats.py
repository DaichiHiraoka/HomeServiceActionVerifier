from __future__ import annotations

import numpy as np
import pandas as pd

from home_service_action_verifier.config import FeaturesConfig
from home_service_action_verifier.schemas import AnnotationJson, feature_column
from home_service_action_verifier.windows import WindowSpec

from .common import aggregate_series, object_slice


def compute_object_features(
    skeleton: pd.DataFrame,
    objects: pd.DataFrame,
    annotation: AnnotationJson,
    window: WindowSpec,
    config: FeaturesConfig,
) -> dict[str, float]:
    df = object_slice(objects, window)
    features: dict[str, float] = {}
    if df.empty:
        for name in ("disp_from_initial", "speed", "area"):
            features.update({feature_column("OBJ", k): v for k, v in aggregate_series(name, []).items()})
        features[feature_column("OBJ", "visible_rate")] = np.nan
        return features
    visible = df[df["visible"].astype(bool)]
    features[feature_column("OBJ", "visible_rate")] = float(df["visible"].mean())
    if visible.empty:
        for name in ("disp_from_initial", "speed", "area"):
            features.update({feature_column("OBJ", k): v for k, v in aggregate_series(name, []).items()})
        return features
    first = visible.iloc[0]
    disp = np.sqrt((visible["cx"] - first["cx"]) ** 2 + (visible["cy"] - first["cy"]) ** 2)
    step = np.sqrt(visible["cx"].diff() ** 2 + visible["cy"].diff() ** 2)
    speed = step * float(annotation["fps"])
    area = visible["w"] * visible["h"]
    for name, values in {
        "disp_from_initial": disp,
        "speed": speed,
        "area": area,
    }.items():
        features.update({feature_column("OBJ", k): v for k, v in aggregate_series(name, values).items()})
    return features
