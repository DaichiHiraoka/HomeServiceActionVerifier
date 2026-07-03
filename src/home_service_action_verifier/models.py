from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, seed: int = 0) -> Pipeline:
    if name == "logreg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=seed
                    ),
                ),
            ]
        )
    if name == "rf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=seed,
                        min_samples_leaf=1,
                    ),
                ),
            ]
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        class_weight="balanced",
                        random_state=seed,
                        n_estimators=100,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown model: {name}")
