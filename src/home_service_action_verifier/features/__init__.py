from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from home_service_action_verifier.config import FeaturesConfig
from home_service_action_verifier.schemas import AnnotationJson
from home_service_action_verifier.windows import WindowSpec

FeatureFunc = Callable[[pd.DataFrame, pd.DataFrame, AnnotationJson, WindowSpec, FeaturesConfig], dict[str, float]]


@dataclass(frozen=True)
class FeatureDefinition:
    group: str
    name: str
    func: FeatureFunc


class FeatureRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, FeatureDefinition] = {}

    def register(self, group: str, name: str, func: FeatureFunc) -> None:
        key = f"{group}.{name}"
        self._definitions[key] = FeatureDefinition(group=group, name=name, func=func)

    def by_groups(self, groups: list[str]) -> list[FeatureDefinition]:
        wanted = set(groups)
        return [definition for definition in self._definitions.values() if definition.group in wanted]

    def all(self) -> list[FeatureDefinition]:
        return list(self._definitions.values())


def default_registry() -> FeatureRegistry:
    from .context_feats import compute_context_features
    from .crop_feats import compute_crop_features
    from .object_feats import compute_object_features
    from .relation_feats import compute_relation_features
    from .skeleton_feats import compute_skeleton_point_features, compute_skeleton_sequence_features

    registry = FeatureRegistry()
    registry.register("SKEL_POINT", "point", compute_skeleton_point_features)
    registry.register("SKEL_SEQ", "sequence", compute_skeleton_sequence_features)
    registry.register("OBJ", "object", compute_object_features)
    registry.register("REL", "relation", compute_relation_features)
    registry.register("CTX", "context", compute_context_features)
    registry.register("CROP", "crop", compute_crop_features)
    return registry


def compute_feature_row(
    skeleton: pd.DataFrame,
    objects: pd.DataFrame,
    annotation: AnnotationJson,
    window: WindowSpec,
    config: FeaturesConfig,
    registry: FeatureRegistry | None = None,
) -> dict[str, object]:
    registry = registry or default_registry()
    row: dict[str, object] = {
        "seq_id": window.seq_id,
        "actor": window.actor,
        "scenario": window.scenario,
        "label": window.label,
        "contact_id": window.contact_id,
        "ctx_sec": window.ctx_sec,
        "t_end_rel": window.t_end_rel,
    }
    for definition in registry.all():
        row.update(definition.func(skeleton, objects, annotation, window, config))
    return row
