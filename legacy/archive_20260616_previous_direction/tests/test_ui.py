from __future__ import annotations

from pathlib import Path

from home_service_action_verifier.ui import (
    UI_METHODS,
    resolve_scenario_asset_paths,
    run_ui_directory_experiment,
    run_ui_experiment,
)


def test_ui_methods_are_research_methods_only() -> None:
    assert UI_METHODS == ("rule_based", "token_only", "proposed")
    assert "vlm_direct_full" not in UI_METHODS
    assert "vlm_direct_roi" not in UI_METHODS


def test_run_ui_experiment_writes_predictions_and_metrics(tmp_path) -> None:
    result = run_ui_experiment(
        work_order_path="configs/scenarios/router_repair.json",
        zones_path="configs/zones/router_repair_zones.json",
        annotations_path="data/real/router_trial_001_annotations.example.jsonl",
        methods=["rule_based", "token_only"],
        output_root=tmp_path,
    )

    assert result.output_dir.exists()
    assert set(result.results_by_method) == {"rule_based", "token_only"}
    assert result.prediction_paths["rule_based"].exists()
    assert (result.evaluation_dirs["rule_based"] / "metrics.json").exists()
    assert (result.output_dir / "per_method_metrics.csv").exists()
    assert (result.output_dir / "per_event_predictions.csv").exists()
    assert (result.output_dir / "summary.md").exists()


def test_resolve_template_upload_folder() -> None:
    assets = resolve_scenario_asset_paths("uploadfiles/template")

    assert assets.work_order.name == "work_order.json"
    assert assets.zones.name == "zones.json"
    assert assets.annotations.name == "annotations.jsonl"
    assert assets.video_path_file.name == "video_path.txt"
    assert assets.video is None


def test_run_ui_directory_experiment_reads_folder_files(tmp_path) -> None:
    result = run_ui_directory_experiment(
        scenario_dir="uploadfiles/template",
        methods=["rule_based"],
        output_root=tmp_path,
    )

    assert result.asset_paths is not None
    assert result.asset_paths.scenario_dir == Path("uploadfiles/template")
    assert set(result.results_by_method) == {"rule_based"}
    assert result.prediction_paths["rule_based"].exists()
    assert (result.output_dir / "config.json").exists()
