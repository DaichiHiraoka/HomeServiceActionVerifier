from __future__ import annotations

import csv
import json

from home_service_action_verifier.cli import main


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


def test_evaluate_events_outputs_metrics(tmp_path) -> None:
    run_dir = tmp_path / "scenario_run"
    eval_dir = tmp_path / "scenario_eval"
    main(
        [
            "analyze-scenario",
            "--work-order",
            "configs/scenarios/router_repair.json",
            "--zones",
            "configs/zones/router_repair_zones.json",
            "--annotations",
            "data/real/router_trial_001_annotations.example.jsonl",
            "--method",
            "token_only",
            "--output-dir",
            str(run_dir),
        ]
    )

    exit_code = main(
        [
            "evaluate-events",
            "--annotations",
            "data/real/router_trial_001_annotations.example.jsonl",
            "--predictions",
            str(run_dir / "event_predictions.jsonl"),
            "--output-dir",
            str(eval_dir),
        ]
    )

    assert exit_code == 0
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "per_event.csv").exists()
    assert (eval_dir / "confusion_matrix.csv").exists()
    assert (eval_dir / "summary.md").exists()


def test_evaluate_events_outputs_all_review_policies(tmp_path) -> None:
    eval_dir = tmp_path / "scenario_eval"
    annotations_path = tmp_path / "annotations.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    annotations_path.write_text(
        "\n".join(
            [
                '{"event_id":"N","start_sec":0,"end_sec":1,"action":"inspect","label":"normal"}',
                '{"event_id":"R","start_sec":1,"end_sec":2,"action":"inspect","label":"review"}',
                '{"event_id":"S","start_sec":2,"end_sec":3,"action":"open","label":"suspicious"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        "\n".join(
            [
                '{"event_id":"N","method":"test","predicted_label":"normal","suspicion_score":0.0}',
                '{"event_id":"R","method":"test","predicted_label":"review","suspicion_score":0.3}',
                '{"event_id":"S","method":"test","predicted_label":"suspicious","suspicion_score":0.7}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "evaluate-events",
            "--annotations",
            str(annotations_path),
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(eval_dir),
            "--all-review-policies",
        ]
    )

    payload = json.loads((eval_dir / "metrics_by_policy.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert sorted(payload) == ["exclude", "negative", "positive"]
    assert len({payload[policy]["num_events"] for policy in payload}) > 1


def test_compare_methods_outputs_non_identical_method_rows(tmp_path) -> None:
    output_dir = tmp_path / "method_comparison"

    exit_code = main(
        [
            "compare-methods",
            "--work-order",
            "configs/scenarios/router_repair.json",
            "--zones",
            "configs/zones/router_repair_zones.json",
            "--annotations",
            "data/real/router_trial_001_annotations.example.jsonl",
            "--methods",
            "rule_based,token_only,rule_based_no_owner,rule_based_no_zone,rule_based_no_photo_target",
            "--output-dir",
            str(output_dir),
        ]
    )

    rows = list(csv.DictReader((output_dir / "per_method_metrics.csv").open(encoding="utf-8")))
    signatures = {
        (
            row["method"],
            row["review_rate"],
            row["same_action_different_context_accuracy"],
            row["num_events"],
        )
        for row in rows
    }
    methods = {row["method"] for row in rows}

    assert exit_code == 0
    assert methods == {
        "rule_based",
        "token_only",
        "rule_based_no_owner",
        "rule_based_no_zone",
        "rule_based_no_photo_target",
    }
    assert len(signatures) > 1


def test_analyze_scenario_records_custom_rule_weights(tmp_path) -> None:
    output_dir = tmp_path / "weighted_run"
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(
        json.dumps(
            {
                "forbidden_zone": 0.4,
                "resident_private_object": 0.4,
                "resident_object_action": 0.3,
                "resident_into_worker_container": 0.8,
                "disallowed_photo_target": 0.5,
                "unexpected_action": 0.25,
                "high_risk_object": 0.2,
                "review_threshold": 0.1,
                "suspicious_threshold": 0.25,
                "high_risk_threshold": 0.9,
            }
        ),
        encoding="utf-8",
    )

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
            "token_only",
            "--rule-weights",
            str(weights_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output_dir / "event_predictions.jsonl").read_text(encoding="utf-8").splitlines()]

    assert exit_code == 0
    assert config["rule_weights_path"] == str(weights_path)
    assert config["rule_weights"]["suspicious_threshold"] == 0.25
    assert next(row for row in rows if row["event_id"] == "S05")["predicted_label"] == "suspicious"
