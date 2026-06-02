from __future__ import annotations

import json

from privacy_vlm_poc.cli import main


def test_analyze_scenario_rule_based_outputs_jsonl_and_summary(tmp_path) -> None:
    output_dir = tmp_path / "scenario_run"

    exit_code = main(
        [
            "analyze-scenario",
            "--work-order",
            "configs/scenarios/router_repair.json",
            "--zones",
            "configs/zones/router_repair_zones.json",
            "--annotations",
            "data/real/router_trial_001_annotations.example.jsonl",
            "--method",
            "rule_based",
            "--output-dir",
            str(output_dir),
        ]
    )

    predictions_path = output_dir / "event_predictions.jsonl"
    assert exit_code == 0
    assert predictions_path.exists()
    assert (output_dir / "summary.md").exists()
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["event_id"] == "S12"
    assert rows[-1]["predicted_label"] == "high_risk"
