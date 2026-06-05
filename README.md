# Home Service Action Verifier

## 概要

Home Service Action Verifier は、一般家庭内で許可済み作業者が修理・点検を行う場面を対象にした卒業研究用の実験システムです。動画全体を直接分類するのではなく、イベントアノテーション、作業票、許可エリア、対象物、所有者情報を照合し、イベント単位で `normal` / `review` / `suspicious` / `high_risk` を出力します。

VLM は主判定器ではなく、必要なイベントだけを確認する補助手段です。最低限の評価は Rule-Based baseline だけで実行できます。

## 研究目的

目的は、許可された作業の範囲から外れた行動をどの情報で検出できるかを数値比較することです。特に、作業票、所有者、許可エリア、イベントトークンが有効か、同じ動作でも文脈が異なる場合に正しく判定できるかを評価します。

このシステムは犯罪や不正行為を断定しません。顔認識、個人識別、人物属性推定、自動通報は実装しません。

## 対象シナリオ

初期シナリオは `router_repair` です。Wi-Fiルーターの通信不良確認を行う作業者を想定し、`configs/scenarios/router_repair.json` に作業票、`configs/zones/router_repair_zones.json` に仮ゾーン、`data/real/router_trial_001_annotations.example.jsonl` にイベント注釈例を置いています。

## ラベル定義

- `normal`: 作業票・許可エリア・作業対象と整合する通常行動。
- `review`: スコアが低中程度で、人手確認に回す行動。
- `suspicious`: 作業票や許可文脈から外れる可能性が高い行動。
- `high_risk`: 住人所有物を作業者側のバッグ等へ入れるなど、強い確認が必要な行動。

## 作業票とは

作業票 `WorkOrder` は、許可ゾーン、禁止ゾーン、作業対象物、作業者所有物、住人私物、許可行動、禁止行動、撮影許可対象、高リスク物体を定義する JSON です。検知器はイベントをこの作業票と照合してスコア化します。

## イベントトークンとは

`EventToken` は、動画から切り出したイベント単位の構造化情報です。開始/終了時刻、行動、ゾーン、物体クラス、所有者、収納先、撮影対象、正解ラベル、同一動作ペアIDなどを持ちます。読み込み時に JSONL の `label` は `ground_truth_label` に変換されます。

## Rule-Based baseline

Rule-Based baseline は、`EventToken + WorkOrder` だけで動く基準手法です。禁止ゾーン、住人私物、撮影許可外対象、作業票にない行動、高リスク物体などに加点し、最終スコアを `0.0` から `1.0` に丸めます。

## 比較対象手法

- `rule_based`: 作業票とイベントトークンを使う基準線。
- `token_only`: 初期実装では Rule-Based と同じ判定を使い、将来の token-only heuristic の比較枠を確保します。
- `proposed`: Rule-Based を起点に、曖昧イベントのみ VLM 補助へ回す提案手法の初期形です。現時点では VLM 補助は未実装です。
- `vlm_direct_full`: Full RGB 入力の比較枠です。イベント窓フレーム抽出未接続のため、現時点では明示的に未実装エラーを返します。
- `vlm_direct_roi`: 手元/物体ROI入力の比較枠です。ROI生成連携未接続のため、現時点では明示的に未実装エラーを返します。

## CLI使用例

セットアップ:

```powershell
uv sync
uv run home-service-verifier bootstrap
uv run pytest
```

イベント単位解析:

```powershell
uv run python -m home_service_action_verifier.cli analyze-scenario --video data/real/router_trial_001.mp4 --work-order configs/scenarios/router_repair.json --zones configs/zones/router_repair_zones.json --annotations data/real/router_trial_001_annotations.example.jsonl --method rule_based
```

イベント単位評価:

```powershell
uv run python -m home_service_action_verifier.cli evaluate-events --annotations data/real/router_trial_001_annotations.example.jsonl --predictions outputs/runs/latest/event_predictions.jsonl
```

複数手法比較:

```powershell
uv run python -m home_service_action_verifier.cli compare-methods --video data/real/router_trial_001.mp4 --work-order configs/scenarios/router_repair.json --zones configs/zones/router_repair_zones.json --annotations data/real/router_trial_001_annotations.example.jsonl --methods rule_based,token_only,proposed
```

legacy 動画解析:

```powershell
uv run python -m home_service_action_verifier.cli analyze --video data/sample/sample_suspicious.mp4 --sampling hybrid --num-frames 8 --mask background_blur_with_roi --vlm-backend mock
```

Streamlit UI は本目的用のイベント単位実験画面です。`uploadfiles/<folder>` を1つ指定すると、作業票、ゾーン、アノテーション、動画パスを同時に読み込み、`rule_based`、`token_only`、`proposed` の解析・評価・比較を同じ画面で実行できます。

```powershell
uv run streamlit run src/home_service_action_verifier/ui.py --server.address 127.0.0.1 --server.port 8501
```

`uploadfiles/template` をコピーして、実証用フォルダを `uploadfiles/` 直下に並列配置します。

```text
uploadfiles/
  template/
    work_order.json
    zones.json
    annotations.jsonl
    video_path.txt
  router_trial_001/
    work_order.json
    zones.json
    annotations.jsonl
    video_path.txt
    video.mp4
```

`video_path.txt` は、動画を記録する場合に `video.mp4` のような相対パスを1行だけ書きます。イベントトークン評価だけならコメントのみでも動きます。

## 出力ファイル

`analyze-scenario` は `outputs/runs/<timestamp>/event_predictions.jsonl`、`summary.md`、`config.json` を出力し、最新予測を `outputs/runs/latest/event_predictions.jsonl` にも保存します。

`evaluate-events` は `outputs/evaluations/<timestamp>/metrics.json`、`per_event.csv`、`confusion_matrix.csv`、`summary.md` を出力します。

`compare-methods` は `per_method_metrics.csv`、`per_event_predictions.csv`、`summary.md`、`config.json` を出力します。

## 評価指標

評価はイベント単位です。`normal` を negative、`suspicious` と `high_risk` を positive とし、`review` はデフォルトで除外します。`accuracy`、`precision`、`recall`、`f1`、`roc_auc`、`average_precision`、`false_alarm_rate`、`same_action_different_context_accuracy`、confusion matrix を出力します。

## 制限事項

- 実動画からの物体検出、手検出、所有者推定は未実装または stub です。
- ゾーン定義は固定矩形の仮座標で、実動画に合わせた調整が必要です。
- `vlm_direct_full` と `vlm_direct_roi` は比較枠のみで、現時点では未接続です。
- `proposed` は Rule-Based 結果を採用し、曖昧イベントを将来の VLM 確認対象として記録します。
- legacy VLM backend は補助機能であり、研究評価の中心ではありません。

## 禁止事項

- 顔認識、個人識別、年齢・性別・体型・服装属性の推定をしない。
- 犯罪関与や犯罪事実の断定をしない。
- 警察や外部機関への自動通報をしない。
- raw video を外部APIへ送信する設計にしない。
- 実運用の防犯システムであるかのように扱わない。
