# Luna 6への接続互換対応

2026-09-24。利用者の依頼は、Luna 5.6に固定されていた検索・画像の判定と、Luna 6へ送るパラメーターの修正。モデル設定の変更や新しい有料API評価は実行していない。

## 変更内容と根拠

- `application.py`: 検索・画像を生成する条件から `model == "gpt-5.6-luna"` を撤去。既存のOpenAI公式URL・Provider種類の条件は維持し、現在のモデルを渡す。任意モデルの機能対応を保証する意味ではなく、未対応の場合はAPIエラーを返す。モデルを無断で差し替えたり、追加の探索要求を送ったりしない。
- `providers/openai_compatible.py`: GPT-6も `max_completion_tokens` を使用し、既定の推論と非互換な `temperature` を省く。GPT-5の挙動と設定上限は維持。公式ホストはURLのhostnameで判定し、他社互換APIへ同じ形式を押し付けない。
- `services/image_sharing.py`: HTTP 400/404/422を、現在の接続先・モデルで画像解析要求を受け付けられなかった旨として表示。生のエラー・キー・画像は表示せず、自動再試行しない。
- `ui/chat_window.py` と検索アダプターの説明: Luna 5.6専用という表記を撤去。
- 検索・画像のResponses形式、推論強度、人格、音声、PMX/VMD、通常DB、保存形式、環境変数名、依存関係は変更していない。過去の有料評価スクリプトのモデル固定は、その評価の承認上限として維持する。

OpenAI Docsの [Luna 6仕様](https://developers.openai.com/api/docs/models/gpt-6-luna) と [移行ガイド](https://developers.openai.com/api/docs/guides/latest-model) を当日確認。Luna 6のChat Completions、画像入力、ResponsesのWeb検索対応を確認した。推論強度は以前と同じく未指定で、既定値を使う。

## 検証の範囲

検査記録: `local/diagnostics/luna6-compat-20260924/`。修正前の採用範囲・合否条件は `plan.md`。

- 自動テスト: 同期・ストリーミングの送信パラメーター、旧モデルと他社互換APIの回帰、任意モデルへの設定値引き継ぎ、公式以外の接続先を使わないこと、許可OFF維持、検索・画像の必要最小限の送信内容と失敗処理を検査。
- Codex環境の統合確認: 実際のChatWindow・ApplicationBackend・OpenAICompatibleProviderを使い、HTTP応答だけをモック化。Luna 5.6と6の両方で通常会話→検索→出典表示→画像添付→画像の返答と履歴を確認。DB・QSettingsはテスト専用の一時領域。外部への実通信なし。
- 初回対象検査は169件成功・2件失敗。追加した統合検査のHTTPモックが、検索・画像後の非ストリーミング生成にもSSEを返したことが原因。実際の要求形式に応じてJSON/SSEを返すようテストを修正し、移行検査9件が成功。基準の緩和やテストの無効化はしていない。初回ログも保持。
- `test.bat` は画像UIのエラー復帰テスト中、`wait_until` 内で `Fatal Python error: Aborted`。異常停止後に残った今回のpytestプロセスは、コマンド・親PID・ログ作成時刻を照合して終了した。終了コード1。`test-bat.log` を保持し、通常の全体検査が成功したとは扱わない。
- 診断としてテスト内容を変更せず、ログ捕捉を外し、専用一時領域を指定した全件pytest（`-s --basetemp local/diagnostics/luna6-compat-20260924/full-diagnostic-tmp`）は **1245件成功、157.34秒、終了コード0**。`pytest-diagnostic.log` を保持。最初のQt異常終了の根本原因は未解消で、この成功を修正済みの根拠にはしない。
- Ruff全体 `check .` 成功、`format --check .` は266ファイル成功。読み取り専用の別エージェントによる差分・送信境界レビューでも重大な指摘なし。
- 未確認: Luna 6の実API応答、アカウントでの利用可否、費用・速度・会話品質、および利用者の実機受入。HTTPモックの成功を実API検証済みと扱わない。

## 変更ファイルと復元

既存ファイルは次の9件。編集前のコピーは、検査記録の `before/` 以下に同じ相対パスで保存。

- `src/multi_llm_companion/application.py`
- `src/multi_llm_companion/providers/openai_compatible.py`
- `src/multi_llm_companion/services/web_search.py`
- `src/multi_llm_companion/services/image_sharing.py`
- `src/multi_llm_companion/ui/chat_window.py`
- `tests/test_openai_compatible_provider.py`
- `tests/test_web_search.py`
- `tests/test_image_sharing.py`
- `docs/providers_ja.md`

新規ファイルは `tests/test_openai_model_migration.py` と本書。復元はアプリを閉じて対象ファイルだけ編集前コピーと比較し、今回の差分を戻す。その後の編集がある場合はファイル全体を上書きしない。通常の履歴・記憶・設定や原本素材の削除は不要。

## モデルを切り替える操作

切替の目的は、会話・検索・画像共有でLuna 6を使うこと。現在のモデル指定値を控えてから行う。

1. 雨衣のアプリを閉じる。
2. Windowsの検索から「環境変数を編集」を開き、ユーザー環境変数の `OPENAI_MODEL` を `gpt-6-luna` にする。独自のProvider設定なら `local/providers.yaml` の `model_env` が指す変数を変更する。APIキーは変更不要。
3. 起動に使うターミナルを開き直し、いつもの `run.bat` で起動する。
4. 右上の歯車「設定」→「診断情報」の「モデル:」が `gpt-6-luna` なら切替が反映されている。違う場合は送信せず、起動元が引き継いだ環境変数を確認する。
5. 実際の会話確認を行う場合は「こんにちは」を送る。検索は「今日の科学ニュースを一つ調べて」、画像は入力欄のプラスから試し用の画像を選び「この画像について説明して」と送る。これは通常の有料API利用になる。返答が表示され、検索では出典が付き、画像の内容に対応していれば成功。
6. 処理中は「検索・返答を取消」または「画像・返答を取消」、読み上げ中は「音声・動作を停止」で止められる。接続・モデルのエラーが出たら繰り返し送信せず、環境変数を控えた以前の値に戻して起動し直す。報告する場合はモデル名と画面のエラー文だけを伝え、APIキーを含めない。

コード・保存・取消の自動検査を利用者が繰り返す必要はない。実APIでの会話品質は今回未確認なので、切替だけで受入完了とはしない。
