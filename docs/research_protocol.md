# Research Protocol

This protocol turns the repository from a pipeline PoC into a repeatable experiment harness.

## 1. Setup

```bash
uv sync
uv run python scripts/generate_synthetic_video.py
```

For pipeline-only verification:

```bash
uv run pytest
uv run python -m privacy_vlm_poc.cli analyze --video data/sample/sample_suspicious.mp4 --sampling hybrid --num-frames 8 --mask background_blur_with_roi --vlm-backend mock
```

For real VLM verification, install Ollama and pull the selected local model:

```bash
ollama pull gemma3:4b
ollama pull gemma3:12b
```

Set `.env`:

```env
OLLAMA_ENABLED=true
OLLAMA_MODEL=gemma3:4b
```

Then verify readiness:

```bash
uv run python -m privacy_vlm_poc.cli doctor
```

For event-level scenario assets:

```bash
uv run privacy-vlm-poc bootstrap
```

## 2. Single-Video Real VLM Run

```bash
uv run python -m privacy_vlm_poc.cli analyze \
  --video data/sample/sample_suspicious.mp4 \
  --sampling event_window \
  --num-frames 8 \
  --mask background_blur_with_roi \
  --vlm-backend ollama
```

Inspect:

- `outputs/runs/<timestamp>/grid.jpg`
- `outputs/runs/<timestamp>/result.json`
- `outputs/runs/<timestamp>/report.md`
- `outputs/runs/<timestamp>/config.json`

## 3. Sampling x Masking Matrix

Quick smoke matrix:

```bash
uv run python scripts/run_research_matrix.py --quick --vlm-backend mock
```

Real VLM quick matrix:

```bash
uv run python scripts/run_research_matrix.py --quick --vlm-backend ollama
```

Full matrix:

```bash
uv run python scripts/run_research_matrix.py --vlm-backend ollama
```

The matrix writes:

- `summary.csv`: per-run prediction, confidence, selected frames, explanation, limitations
- `by_condition.csv`: average confidence, selected-frame recall, processing time by sampling/mask condition
- `summary.md`: readable summary
- `config.json`: experiment configuration

## 4. What To Compare

Primary comparisons:

- Sampling method vs. `selected_frame_recall`
- Sampling method vs. VLM explanation changes in `reason` and `limitations`
- Mask method vs. confidence and false/low-confidence outcomes
- Mask method vs. privacy-sensitive output flag

Do not report the result as theft detection accuracy. The permitted label is `unauthorized_object_interaction_suspected`.

## 5. Event-Level Scenario Protocol

The fixed `router_repair` scenario keeps the work order, allowed zones, forbidden zones, and ownership context stable. This makes the central comparison clear: whether a method can distinguish the same visible action under different work-order contexts.

Primary event labels:

- `normal`: authorized work behavior
- `review`: ambiguous behavior that should be checked by a human
- `suspicious`: likely unauthorized behavior
- `high_risk`: behavior involving private objects, forbidden areas, or resident objects entering worker containers

Event-level evaluation is preferred over video-level evaluation because a whole-video label hides which action triggered the decision. The experiment should report per-event predictions, reasons, and evidence.

Comparison conditions:

- `Rule-Based`: `EventToken + WorkOrder`
- `VLM Direct Full`: selected full RGB event frames, initially a placeholder until event-window frame extraction is wired
- `VLM Direct ROI`: hand/object ROI event frames, initially a placeholder
- `Token Only`: event token structure without images
- `Proposed`: Rule-Based first, then VLM confirmation for ambiguous events

Metrics:

- Accuracy shows overall event-label correctness after binary conversion.
- Precision shows how often positive alerts are correct.
- Recall shows how many suspicious/high-risk events are caught.
- F1 balances precision and recall.
- ROC-AUC and Average Precision evaluate score separation when both classes are present.
- False Alarm Rate shows how often normal events become positive alerts.
- Same Action Different Context rows in `summary.md` show whether matching action pairs, such as worker tool into bag versus resident key into bag, are separated correctly.

Privacy constraints should be treated as experimental conditions, not afterthoughts. Compare full RGB, ROI-limited, and token-only inputs to measure performance changes under reduced visual information. Do not add face recognition, personal identification, age, gender, body-type, or clothing-attribute inference.

## 6. Scenario Commands

Rule-Based baseline:

```bash
uv run python -m privacy_vlm_poc.cli analyze-scenario --work-order configs/scenarios/router_repair.json --zones configs/zones/router_repair_zones.json --annotations data/real/router_trial_001_annotations.example.jsonl --method rule_based
```

Evaluate predictions:

```bash
uv run python -m privacy_vlm_poc.cli evaluate-events --annotations data/real/router_trial_001_annotations.example.jsonl --predictions outputs/runs/latest/event_predictions.jsonl
```

Compare methods:

```bash
uv run python -m privacy_vlm_poc.cli compare-methods --work-order configs/scenarios/router_repair.json --zones configs/zones/router_repair_zones.json --annotations data/real/router_trial_001_annotations.example.jsonl --methods rule_based,vlm_direct_full,proposed
```

The system must avoid crime conclusions. Use wording such as unauthorized interaction suspected, review, suspicious, and high risk.

## 7. Minimum Evidence For Graduation Research Demo

The repository is ready for an initial research demo when all of the following are true:

- `uv run pytest` passes
- `uv run python -m privacy_vlm_poc.cli doctor` reports the selected Ollama model present
- one suspicious and one normal synthetic video run succeed with `--vlm-backend ollama`
- `scripts/run_research_matrix.py --quick --vlm-backend ollama` produces `summary.csv`
- `analyze-scenario --method rule_based` produces `event_predictions.jsonl`
- `evaluate-events` produces `metrics.json`, `per_event.csv`, `confusion_matrix.csv`, and `summary.md`
- report discussion focuses on uncertainty and limited visual information, not crime determination
