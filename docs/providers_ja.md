# AIの接続先を設定する

初期状態の **Mock** は外部通信をせず、固定応答を返します。AIと会話するには、利用するAPIサービスの契約とAPIキーが必要です。会話・人格設定・関連する記憶などが送信されるため、料金とデータの扱いを確認してください。

## OpenAIを使う

1. アプリを終了します。Windowsのスタートで「環境変数」と検索し、**環境変数を編集** を開きます。
2. 利用者環境変数の **新規**（既存項目は **編集**）で、次を登録します。

| 変数名 | 値 |
| --- | --- |
| `OPENAI_API_KEY` | 自分のAPIキー |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | 基準構成は `gpt-6-luna` |

3. **OK** で閉じ、アプリを起動するフォルダーやターミナルも開き直します。
4. `local/app.yaml` の `active_provider` を `openai` にして保存します。
5. **起動** ショートカットで起動し、**設定 → 診断情報** のプロバイダーとモデルを確認してから文章を送ります。

利用できるモデルはAPIアカウントの権限によります。キーをYAML・コード・ログへ書かず、画面写真にも含めないでください。Windowsの環境変数は `.env` より優先されます。変更が反映されない場合はWindowsへサインインし直します。

## その他の接続先

`local/providers.yaml` にOpenAI互換API用の見本があります。

| `active_provider` | 環境変数 |
| --- | --- |
| `qwen` | `QWEN_API_KEY`・`QWEN_BASE_URL`・`QWEN_MODEL` |
| `deepseek` | `DEEPSEEK_API_KEY`・`DEEPSEEK_BASE_URL`・`DEEPSEEK_MODEL` |
| `mock` | 不要 |

接続URLとモデル名は提供元の資料で確認します。接続先・認識方式・マイクは **設定 → 会話・音声 → LLM接続・マイクの設定** からも選べます。

Web検索・画像共有はOpenAI公式API接続で利用します。設定中のモデルが非対応の場合はエラーになり、別モデルへ自動で切り替えません。操作は [会話のガイド](conversation_ja.md) を参照してください。

## 詳細設定

`local/providers.yaml` の各接続先では `temperature`（回答のばらつき）、`max_output_tokens`（出力上限）、`timeout`（待ち時間・秒）を調整できます。接続先・キー・モデルには値ではなく、それぞれ `base_url_env`・`api_key_env`・`model_env` に環境変数名を書きます。見本は [providers.example.yaml](../config/providers.example.yaml) です。

OpenAI公式APIのGPT-5/6系では `temperature` を送信しないため、この値を変えても反映されません。送信時のAPI差異やHTTP/2接続の再利用はアプリが処理します。通常は送信形式を変更する必要はありません。

## 接続できないとき

モデル名・環境変数名・URL、キーの有効期限、サービスの利用上限を確認します。キーの値を表示して調べる必要はありません。文字会話へ戻すにはアプリを終了し、`local/app.yaml` の `active_provider` を `mock` に戻して起動します。詳細は [復旧手順](backup_and_recovery_ja.md#起動や接続で困ったとき) を参照してください。
