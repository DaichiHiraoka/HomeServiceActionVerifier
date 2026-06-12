# Researcher Next Steps

研究者が次に進める作業、撮影計画、外部レビューの記録をまとめる作業用ドキュメントです。

## Files

- [TODO.md](TODO.md): 実験前に進める作業リスト。
- [filming_plan.md](filming_plan.md): 正常動画と違反動画の撮影計画。
- [claude_output.md](claude_output.md): Claude デスクトップアプリで表示されていた研究評価・改善案の記録。

## Immediate Focus

1. 研究スコープを「イベント情報が与えられた条件下での文脈逸脱判定」と明確化する。
2. `token_only` を作業票なしの対照手法として再実装する。
3. 正解ラベル付与とイベントトークン記入の手続きを分離する。
4. `same_action_different_context_accuracy` を二値検出評価と深刻度分類評価に分けて扱う。
5. 撮影動画が現状では主にデモ・記録用途であり、rule 系手法の入力ではないことを明記する。
