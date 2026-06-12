# Researcher Next Steps

このディレクトリは、研究者が今すぐ進める作業を置くための場所です。

## 今すべきこと

1. README と docs を UTF-8 で開き、研究目的、制約、実験手順が読めることを確認する。
2. `filming_plan.md` の撮影計画に従い、約50秒の動画を2本撮影する(正常版 `router_trial_normal_001` と要件外行動版 `router_trial_violation_001`)。撮影後は `uploadfiles/` 配下の実験用フォルダに配置する。
3. 実動画に合わせて `data/real/router_trial_001_annotations.example.jsonl` をコピーし、実測の `start_sec`、`end_sec`、`action`、`zone`、`object_class`、`object_owner`、`label` に更新する。
4. 実動画の画角に合わせて `configs/zones/router_repair_zones.json` または実験用フォルダの `zones.json` の座標を調整する。
5. `uv run python -m home_service_action_verifier.cli compare-methods --work-order configs/scenarios/router_repair.json --zones configs/zones/router_repair_zones.json --annotations data/real/router_trial_001_annotations.example.jsonl --methods rule_based,token_only,proposed` を実行し、比較表を生成する。
6. `outputs/evaluations/<timestamp>/per_event_predictions.csv` を確認し、誤判定イベントの理由とスコアを記録する。
7. 同じ動作でも文脈が違うペアを `same_action_pair_id` で増やし、`same_action_different_context_accuracy` を評価できるデータを増やす。
8. `review` になった曖昧イベントを集め、将来 VLM で確認すべき event window と ROI の要件を整理する。
9. raw video を外部 API に送らない、人物属性推定をしない、実運用の防犯判定として扱わない、という制約を実験メモに明記する。
10. 卒論・発表用に、Rule-Based だけで説明できる成果と、VLM 連携が今後課題である部分を分けて整理する。

## 確認コマンド

```powershell
$env:UV_LINK_MODE = "copy"
uv run pytest
```

Windows 環境で uv の hardlink エラーが出る場合は、上記のように `UV_LINK_MODE=copy` を指定して実行する。
