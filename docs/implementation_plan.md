# Implementation Plan

## 実装目的

旧来のVLM補助デモから、作業票、イベントトークン、Rule-Based検知、イベント単位評価、複数手法比較を中心にした研究用実験システムへ整理する。

## 旧構成からの変更点

- 名称を `HomeServiceActionVerifier` / `home_service_action_verifier` / `home-service-verifier` に変更した。
- 研究評価の中心を動画単位のlegacy解析からイベント単位のシナリオ評価へ移した。
- `router_repair` の作業票、ゾーン、イベント注釈例を追加した。
- `normal`、`review`、`suspicious`、`high_risk` の4ラベルへ整理した。

## 削除したもの

- 旧Pythonパッケージディレクトリ `src/privacy_vlm_poc`。
- 旧CLI entry point `privacy-vlm-poc`。
- README上の旧名称とVLM主役の説明。

## 残したもの

- 動画読み込み、フレーム抽出、マスク処理、legacy `analyze` / `evaluate` CLI。
- Ollama/OpenAI compatible backend 接続。
- Streamlit UI は本目的用のイベント単位実験画面として再構成した。
- 既存の sampling/masking 系テスト。

## 追加したもの

- `WorkOrder`、`ZoneConfig`、`EventToken`、`DetectionResult`、`EventEvaluationMetrics`。
- `scenario.py` による作業票、ゾーン、JSONLイベント読み込み。
- `rule_engine.py` による作業票照合ベースのスコアリング。
- `event_evaluation.py` によるイベント単位評価と同一動作・異文脈指標。
- `baselines/rule_based.py`、`baselines/token_only.py`、`baselines/proposed.py`。
- ROI helper と detector stub。
- `analyze-scenario`、`evaluate-events`、`compare-methods`。
- Streamlit UI を本目的用のイベント単位実験画面へ再構成。
- `uploadfiles/template` を追加し、UIで同構造の実証用フォルダを1つ指定して4ファイルを同時に読み込む方式へ変更。

## 今後実装するもの

- 実動画に合わせたゾーン座標の調整。
- イベント窓ごとのフレーム抽出。
- 手元/物体ROIの自動または半自動推定。
- 曖昧イベントだけをVLM補助確認へ回す proposed の統合。
- 実験データを増やした複数試行評価。

## 未実装のbaseline

- `vlm_direct_full`: Full RGB のイベント窓画像を使う比較枠。現時点では明示的に未実装。
- `vlm_direct_roi`: 手元/物体ROI画像を使う比較枠。現時点では明示的に未実装。
- Streamlit UI からは VLM direct 用の入力やbackend選択を外している。

## 研究評価に必要な次ステップ

1. 実験用動画を撮影する。
2. `router_trial_001_annotations.example.jsonl` を実測イベントで置き換える。
3. ゾーン座標を実動画に合わせる。
4. `rule_based`、`token_only`、`proposed` の比較表を生成する。
5. Same Action Different Context のペアを増やして、文脈差の判定性能を確認する。
