# Codex 作業指示書: 循環構造と空虚な比較の改善(卒研前・最小変更)

## 背景と目的

現状の問題は2点ある。

1. **空虚な比較**: `token_only` と `proposed` が `run_rule_based` のラッパーであり、`compare-methods` の比較表が必ず同値になる。
2. **循環構造**: 正解ラベルと判定入力(EventToken のフィールド)を同じ人が同時に記入するため、Rule-Based の精度が設計上の必然になる。

本指示は、**研究目的(どの情報で許可外行動を検出できるかの数値比較)と主張は変えずに**、既存の EventToken / WorkOrder / rule_engine / evaluate-events / compare-methods の範囲だけでこれを改善する。VLM 接続、フレーム抽出、撮影関連の実装は行わない。

## 全体制約

- 新規依存パッケージを追加しない。
- 既存 CLI のコマンド名・既存引数・既存出力ファイル名を壊さない(フィールド・引数の追加は可)。
- 既存テストはすべて通り続けること。デフォルト動作(引数省略時)は現行と同一の数値を出すこと(回帰保証)。
- すべてのファイルは UTF-8。ドキュメントは既存の文体(です・ます調、断定回避の研究表現)に合わせる。
- 「犯罪を断定しない」「raw video を外部に送らない」等の既存制約文を弱める変更をしない。

---

## 優先度1: token_only を「作業票なしの対照手法」として再実装

**狙い**: 「作業票(許可文脈)がないと same-action different-context を判別できない」ことを実験で示せる対照を作る。これが研究の問いに対する対照条件になる。

### 変更ファイル

- `src/home_service_action_verifier/baselines/token_only.py`(全面書き換え)

### 実装内容

`run(events, work_order)` のシグネチャは CLI 互換のため維持するが、**WorkOrder の内容を一切参照しない**。docstring に「work_order は互換のため受け取るが使用しない」と明記する。

EventToken だけから判断できる汎用事前則のみでスコア化する:

| 規則 | 条件 | 加点 |
| --- | --- | --- |
| 所有者跨ぎの収納 | `action == "place_into_container"` かつ `object_owner` と `container_owner` が両方非 null かつ不一致 | +0.8 |
| 他者所有物の操作 | `action in {"pick_up", "move", "hold", "open"}` かつ `object_owner == "resident"` | +0.3 |
| 文脈不明の撮影 | `action == "photograph"`(対象を問わず一律。許可対象リストを持たないため判断不能) | +0.3 |

`initial_state` は 0.0。スコアは 1.0 で打ち切り、`label_from_score` で既存閾値を使う。`evidence["input"] = "event_token_only"` は維持し、`reasons` には「作業票を参照しないため文脈は判定できない」旨を含める。

**期待される挙動**(設計意図。テストで固定する):

- 型番撮影(正常)と書類撮影(不審)が**同スコア(0.3 → review)になる** = photo_context ペアを構造的に判別できない。
- 鍵をバッグへ(B06 相当)は所有者不一致で 0.8 以上 → 検出できる。
- 禁止ゾーンの引き出し操作は 0.3(review)止まり = ゾーン許可情報なしでは確信できない。

### 追加テスト

`tests/test_token_only.py`(新規):

1. photograph イベントは `target_object` が `router_label` でも `document` でも同一スコアになる。
2. 同じイベント列に対し、rule_based と token_only の予測ラベルが photo ペアで異なる(rule_based は normal / token_only は review)。
3. 所有者不一致の `place_into_container` が `high_risk` になる。
4. **WorkOrder 非依存の回帰テスト**: WorkOrder の `allowed_photo_targets` や `forbidden_zones` を変更しても token_only の出力が完全に不変である(作業票情報の漏れ込み検知)。

### ドキュメント追記

- `README.md` 比較対象手法: token_only を「EventToken のみを使い WorkOrder を参照しない対照手法。許可文脈なしで何が判定できないかを示す」に更新。
- `docs/research_protocol.md` §6 の表: Token Only の Status を `implemented (no work-order context)` に更新。

---

## 優先度2: rule_based のアブレーション変種を追加

**狙い**: 「作業票・所有者・許可エリアのどの情報が有効か」という README の研究目的を、そのまま実験条件にする。条件間の**差分**は循環の影響を受けないため、これが本研究の主結果になる。

### 変更ファイル

- `src/home_service_action_verifier/rule_engine.py`
- `src/home_service_action_verifier/cli.py`

### 実装内容

`rule_based_detect(event, work_order, *, ablation: frozenset[str] = frozenset())` と `run_rule_based(events, work_order, ablation=...)` に拡張する。ablation フラグと停止する規則の対応:

| フラグ | 停止する規則(rule_engine.py の現行行) |
| --- | --- |
| `"owner"` | 住人私物加点(44行)、住人所有物操作加点(49行)、所有者跨ぎ収納加点(54行)、および `_is_contextually_allowed` 内の owner 条件(25行)— owner 条件は「常に許可」側に倒す |
| `"zone"` | 禁止ゾーン加点(39行) |
| `"photo_target"` | 撮影対象加点(63行)と `_is_contextually_allowed` の撮影許可判定(23行)— photograph は許可行動として扱う |
| `"action_allowlist"` | 「作業票にない行動」加点(68行)を停止 |

`ablation` が空のときは現行と完全に同一の数値を返すこと(回帰保証)。`DetectionResult.method` には変種名(下記)を入れ、`evidence["ablation"]` に停止したフラグを記録する。

CLI 側: `SCENARIO_METHODS` に以下を追加し、`_run_scenario_method`(165行)でディスパッチする。

- `rule_based_no_owner`
- `rule_based_no_zone`
- `rule_based_no_photo_target`
- `rule_based_no_action_allowlist`

`compare-methods --methods rule_based,token_only,rule_based_no_owner,...` で混在実行できること。

### 追加テスト

`tests/test_rule_engine_ablation.py`(新規):

1. `ablation=frozenset()` で全イベントのスコアが現行実装と一致する(回帰)。
2. `no_owner` で「住人の鍵を作業者バッグへ」イベントのスコアが 0.8 を下回る(high_risk 判定が崩れる)。
3. `no_zone` で「private_desk で引き出しを開ける」イベントのスコアが suspicious 閾値を下回る。
4. `no_photo_target` で書類撮影と型番撮影のスコア差が縮む(または消える)。
5. CLI 経由で `rule_based_no_owner` が受理され、`method` フィールドに変種名が入る。

### ドキュメント追記

- `README.md` 比較対象手法にアブレーション変種の説明を1段落追加。
- `docs/research_protocol.md` に「§6.5 アブレーション実験」節を追加し、**検証可能な仮説**を明記する:
  - H1: owner 情報を除くと bag_context ペアの判別精度が低下する。
  - H2: zone 情報を除くと禁止エリア系イベントの recall が低下する。
  - H3: photo_target 情報を除くと photo_context ペアの判別精度が低下する。

---

## 優先度3: 評価指標の整合化

**狙い**: 指標間で正解の定義が食い違っている問題と、不確実イベントの除外・無意味な AUC・予測欠落の黙殺を直し、出てくる数値をすべて説明可能にする。

### 変更ファイル

- `src/home_service_action_verifier/event_evaluation.py`
- `src/home_service_action_verifier/schemas.py`(`EventEvaluationMetrics` へのフィールド追加)
- `src/home_service_action_verifier/cli.py`(`evaluate-events` への引数追加)

### 実装内容

1. **ペア指標の二重化**: 既存の `same_action_different_context_accuracy`(4クラス完全一致)は互換のため残し、`same_action_different_context_binary_accuracy` を追加する。二値版は `_binary_label` と同じ写像で正解/予測を二値化して一致判定し、どちらかが None(review 除外)のイベントはペア集計から除外して `notes` に件数を残す。
2. **review 率の可視化**: `review_rate`(predicted_label == "review" の割合、分母は全予測イベント)と `num_review_predictions` を追加する。
3. **予測欠落の可視化**: 現在 `continue` で黙ってスキップしている「予測が見つからないイベント」(117行)を数え、`num_events_without_prediction` として出力し、>0 なら `notes` に警告を含める。
4. **AUC / AP の適用条件**: `num_events < 30`、またはスコアのユニーク値が 4 未満の場合は `roc_auc` と `average_precision` を None とし、`notes` に理由(サンプル不足/スコア離散)を記録する。
5. **全ポリシー一括評価**: `evaluate-events` に `--all-review-policies` フラグを追加。指定時は exclude / positive / negative の3通りで評価し、`metrics_by_policy.json` を出力ディレクトリに追加で書く(主出力ファイルは従来どおり)。

### 追加テスト

`tests/test_event_evaluation.py` に追加:

1. 正解 suspicious / 予測 high_risk のペアイベントで、exact 版は不正解・binary 版は正解になる。
2. review 予測を含むケースで `review_rate` が正しく出る。
3. 予測欠落イベントが `num_events_without_prediction` に計上される。
4. 小サンプルで `roc_auc` が None になり、notes に理由が入る。
5. `--all-review-policies` で `metrics_by_policy.json` が3ポリシー分を含む。

### ドキュメント追記

- `README.md` 評価指標節: 二値版ペア指標、review_rate、AUC の適用条件を反映。
- `docs/research_protocol.md` §5: 「review_policy は3通りすべてを報告する」と明記。

---

## 優先度4: 正解ラベル付与手続きの分離(ドキュメントのみ・コード変更なし)

**狙い**: 循環構造の手続き的切断。コストゼロで実験の妥当性が一段上がる。

### 変更ファイル

- `docs/research_protocol.md`
- `README.md`

### 追記内容

1. `docs/research_protocol.md` に「§3.5 正解ラベル付与手続き」を追加:
   - 正解ラベル(`label`)は**動画のみを見て**「この行動は依頼された作業の範囲か」を判断して付与する。
   - ラベル付与時に作業票 JSON と EventToken の他フィールドを参照しない。
   - トークンのフィールド記入(時刻・行動・ゾーン・物体・所有者)とラベル付与は**別パス**で行い、付与日を記録する。
   - 可能なら別人が付与する。同一人の場合は時間を空けた別パスであることを記録する。
2. `README.md` 研究目的の直後にスコープ宣言を1段落追加:
   - 「本研究は、イベント情報(EventToken)が与えられた条件下での判定可能性(推論)を評価対象とし、映像からのイベント情報抽出(知覚)は対象外とする。」
   - これは目的の変更ではなく精密化であり、既存の目的文・主張は変更しない。
3. `README.md` 制限事項に追記: 「rule_based / token_only / アブレーション変種は動画ファイルを入力として使用しない。動画はデモ・記録・将来の VLM 実験用である。」
4. `README.md` 比較対象手法の `proposed` に追記: 「現時点の proposed は rule_based と同一の結果を返すため、初期実験の主比較からは除外し、VLM 接続後に比較対象へ加える。」

---

## 優先度5: ルール重み・閾値の設定ファイル化

**狙い**: マジックナンバーの外出し。アブレーションと将来の感度分析を1つの仕組みで支える。

### 変更ファイル

- `src/home_service_action_verifier/schemas.py`(`RuleWeights` モデル追加)
- `src/home_service_action_verifier/rule_engine.py`
- `src/home_service_action_verifier/cli.py`(`--rule-weights` オプション)
- `configs/rules/default_rule_weights.json`(新規)

### 実装内容

`RuleWeights` pydantic モデルを追加し、デフォルト値は現行のハードコード値と完全一致させる:

```json
{
  "forbidden_zone": 0.4,
  "resident_private_object": 0.4,
  "resident_object_action": 0.3,
  "resident_into_worker_container": 0.8,
  "disallowed_photo_target": 0.5,
  "unexpected_action": 0.25,
  "high_risk_object": 0.2,
  "review_threshold": 0.2,
  "suspicious_threshold": 0.5,
  "high_risk_threshold": 0.8
}
```

`rule_based_detect` と `label_from_score` が weights を受け取る(省略時はデフォルト)。`analyze-scenario` / `compare-methods` に `--rule-weights <path>` を追加し、使用した重みを `config.json` に記録する。

### 追加テスト

`tests/test_rule_engine.py` に追加: デフォルト weights が現行スコアと一致する回帰テスト、カスタム weights でスコアと閾値判定が変わるテスト。

### ドキュメント追記

- `README.md`: 重みは設定値であり、値の根拠は今後の感度分析で検証する旨を1文追加。

---

## 優先度6: 語彙整合性チェック

**狙い**: 自由文字列の typo・語彙不一致が黙って素通りする問題を警告化する。既知の実例として、注釈例の `zone: "entrance"` は `zones.json` に定義がない(これが最初の検出例になるはず)。

### 変更ファイル

- `src/home_service_action_verifier/scenario.py`
- `src/home_service_action_verifier/cli.py`

### 実装内容

`validate_vocabulary(events, work_order, zone_config) -> list[str]` を追加する。**警告リストを返すだけでエラーにしない**(実験を止めない)。チェック項目:

1. `event.zone` が `zone_config` の zone_id 集合 ∪ `work_order.authorized_zones` ∪ `work_order.forbidden_zones` に含まれない。
2. `event.object_class` が work_order のいずれの物体リスト(target / worker_owned / resident_private / high_risk)にも含まれない(null は除外)。
3. `event.action` が `work_order.allowed_actions` ∪ 既知の汎用動作集合 `{"initial_state", "photograph", "place_into_container", "pick_up", "move", "hold", "open"}` に含まれない。
4. `action == "photograph"` なのに `target_object` が null。

`analyze-scenario` / `compare-methods` の実行時に警告を console 表示し、`config.json` に `vocabulary_warnings` として記録する。

### 追加テスト

`tests/test_scenario_loading.py` に追加: 未知 zone・未知 action・target 欠落でそれぞれ警告が出る、完全整合データで警告ゼロ。

---

## 完了確認(全タスク後)

```powershell
$env:UV_LINK_MODE = "copy"
uv run pytest
uv run python -m home_service_action_verifier.cli compare-methods --work-order configs/scenarios/router_repair.json --zones configs/zones/router_repair_zones.json --annotations data/real/router_trial_001_annotations.example.jsonl --methods rule_based,token_only,rule_based_no_owner,rule_based_no_zone,rule_based_no_photo_target
```

受け入れ条件:

1. 既存テストを含め pytest が全件通る。
2. 上記 compare-methods の `per_method_metrics.csv` で、**5手法の行が同値にならない**(特に token_only と rule_based_no_owner が rule_based と異なる)。
3. token_only の結果が WorkOrder の内容に依存しない。
4. デフォルト引数での rule_based / evaluate-events の数値が変更前と一致する。
5. 注釈例の `zone: "entrance"` に対する語彙警告が表示される。
