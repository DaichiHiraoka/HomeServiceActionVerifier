# Research Protocol

## 1. 目的

このプロトコルは、Home Service Action Verifier をイベント単位の研究評価として再現するための手順です。中心は VLM ではなく、`EventToken`、`WorkOrder`、Rule-Based検知、イベント単位評価、手法比較です。

検知結果は研究上の `normal` / `review` / `suspicious` / `high_risk` ラベルであり、犯罪や不正行為を断定しません。

## 2. セットアップ

```powershell
uv sync
uv run home-service-verifier bootstrap
uv run pytest
```

Ollama を補助確認器として使う場合のみ、ローカルモデルを取得します。

```powershell
ollama pull gemma3:4b
ollama pull gemma3:12b
uv run python -m home_service_action_verifier.cli doctor
```

## 3. シナリオ資産

初期シナリオは `router_repair` です。

- 作業票: `configs/scenarios/router_repair.json`
- ゾーン定義: `configs/zones/router_repair_zones.json`
- 注釈例: `data/real/router_trial_001_annotations.example.jsonl`

実動画を使う場合は、注釈例をコピーして実際の `start_sec` / `end_sec` / `action` / `zone` / `object_class` / `object_owner` / `label` に更新します。ゾーン座標も実動画の画角へ合わせます。

## 4. Rule-Based baseline

```powershell
uv run python -m home_service_action_verifier.cli analyze-scenario --video data/real/router_trial_001.mp4 --work-order configs/scenarios/router_repair.json --zones configs/zones/router_repair_zones.json --annotations data/real/router_trial_001_annotations.example.jsonl --method rule_based
```

出力:

- `outputs/runs/<timestamp>/event_predictions.jsonl`
- `outputs/runs/<timestamp>/summary.md`
- `outputs/runs/<timestamp>/config.json`
- `outputs/runs/latest/event_predictions.jsonl`

## 5. イベント単位評価

```powershell
uv run python -m home_service_action_verifier.cli evaluate-events --annotations data/real/router_trial_001_annotations.example.jsonl --predictions outputs/runs/latest/event_predictions.jsonl
```

評価では `normal` を negative、`suspicious` と `high_risk` を positive とし、`review` はデフォルトで除外します。`review_policy` で `exclude`、`positive`、`negative` を選べます。

主要指標:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `average_precision`
- `false_alarm_rate`
- `same_action_different_context_accuracy`

## 6. 複数手法比較

```powershell
uv run python -m home_service_action_verifier.cli compare-methods --video data/real/router_trial_001.mp4 --work-order configs/scenarios/router_repair.json --zones configs/zones/router_repair_zones.json --annotations data/real/router_trial_001_annotations.example.jsonl --methods rule_based,token_only,proposed
```

比較表は `outputs/evaluations/<timestamp>/per_method_metrics.csv` と `summary.md` に保存されます。

初期比較:

| Method | Input | Status |
| --- | --- | --- |
| Rule-Based | EventToken + WorkOrder | implemented |
| Token Only | EventToken | implemented as initial deterministic baseline |
| Proposed | EventToken + WorkOrder + future ROI/VLM | initial Rule-Based scaffold |
| VLM Direct Full | Full RGB event frames | not connected |
| VLM Direct ROI | Hand/Object ROI event frames | not connected |

## 7. Streamlit UI

本目的用UIでは、`uploadfiles/<folder>` を1つ指定し、`rule_based`、`token_only`、`proposed` をイベント単位で実行・評価・比較できます。個別アップロードではなく、フォルダ内の4ファイルを同時に読み込みます。

```powershell
uv run streamlit run src/home_service_action_verifier/ui.py --server.address 127.0.0.1 --server.port 8501
```

テンプレート:

```text
uploadfiles/template/
  work_order.json
  zones.json
  annotations.jsonl
  video_path.txt
```

実証用フォルダは `uploadfiles/template` をコピーし、`uploadfiles/router_trial_001` のように同階層へ配置します。`video_path.txt` には、同フォルダに置いた動画なら `video.mp4` と書きます。動画を使わない評価ではコメントのみでも構いません。

## 8. Same Action Different Context

同じ動作でも文脈が異なるイベントを `same_action_pair_id` で結びます。

- `photo_context`: ルーター型番の撮影と私的書類の撮影。
- `bag_context`: 作業者工具をバッグへ戻す行動と住人所有物を作業者バッグへ入れる行動。

`same_action_different_context_accuracy` は、同一ペア内で異なる正解ラベルを持つイベントを正しく分けられた割合です。

## 9. Privacy And Safety Constraints

- raw video を外部APIへ送信しない。
- 顔認識、個人識別、年齢、性別、体型、服装属性の推定を行わない。
- 実運用の防犯システムとして扱わない。
- 出力は研究用の許可外行動候補であり、断定表現を使わない。

## 10. Legacy Video Analysis

既存の動画読み込み、フレーム抽出、マスク処理、VLM backend 接続は補助機能として残しています。研究評価の主手順ではありません。

```powershell
uv run python -m home_service_action_verifier.cli analyze --video data/sample/sample_suspicious.mp4 --sampling hybrid --num-frames 8 --mask background_blur_with_roi --vlm-backend mock
uv run python scripts/run_research_matrix.py --quick --vlm-backend mock
```

legacy `mock` backend はパイプライン接続確認用であり、精度評価の根拠にはしません。

## 11. Minimum Evidence For Demo

- `uv run pytest` が通る。
- `analyze-scenario --method rule_based` が `event_predictions.jsonl` を出す。
- `evaluate-events` が `metrics.json`、`per_event.csv`、`confusion_matrix.csv`、`summary.md` を出す。
- `compare-methods --methods rule_based,token_only,proposed` が比較表を出す。
- レポートでは、怪しい行動の定義、使った情報、Rule-Basedだけの性能、入力制限条件、同一動作・異文脈ペアの評価を説明する。
