# LLMプロバイダー設定

## Providerとは

Providerは、アプリとLLMサービスの間をつなぐ部品です。サービスごとの違いをProviderへまとめることで、画面や会話保存を作り直さずにLLMを切り替えられます。

第1段階では次を用意します。

- MockProvider: 外部通信なし。APIキー不要
- OpenAICompatibleProvider: OpenAI互換APIへ共通形式で接続

Qwen、DeepSeek、OpenAIは、OpenAICompatibleProviderへそれぞれの設定を渡して利用します。将来、各社固有の機能が必要になった場合は、同じ共通操作を持つ専用Providerを追加できます。

## 共通操作

- chat: 回答をまとめて受け取る
- stream_chat: 回答を可能な範囲で少しずつ受け取る
- health_check: 設定や接続状態を確認する
- get_usage: APIから取得できた入出力トークン数を返す

トークンは、LLMが文章を処理するときの細かな単位です。サービスや応答方式によって使用量を取得できない場合があるため、その場合は無理に推測しません。

## 最初はMockを使う

Mockは次の確認に向いています。

- 画面が起動する
- 入力と回答が表示される
- 逐次表示の流れが動く
- 会話履歴が保存される
- 新しい会話と履歴削除が動く
- テストがAPIキーなしで再現できる

外部LLMで問題が起きたときも、まずMockへ戻して、画面側の問題か接続設定の問題かを切り分けられます。「切り分け」とは、原因がありそうな範囲を順番に狭めることです。

## YAMLへ書く内容

local/app.yamlでは、active_providerへ使用する設定名を書きます。providers_config_pathはProvider設定、persona_pathは人格設定、database_pathは会話データベースの場所です。

local/providers.yamlでは、providersの下にmock、openai、qwen、deepseekなどの名前付き設定を置きます。各設定には次のキーだけを書きます。

- provider: mockまたはopenai_compatibleという実装の種類
- base_url_env: base URLを取得する環境変数名
- api_key_env: APIキーを取得する環境変数名
- model_env: モデル名を取得する環境変数名
- temperature: 温度
- max_output_tokens: 最大出力トークン数
- timeout: タイムアウト

温度は、回答のばらつきを調整する値です。最大出力数は、回答が長くなりすぎないようにする上限です。タイムアウトは、応答を待つ最大時間です。値の範囲や単位は、設定例と実装の検証メッセージを確認してください。

YAMLへ実際のbase URL、APIキー、モデル名を直接書かず、それぞれを読む環境変数の名前を書きます。実際に使用する構造は、config/app.example.yamlとconfig/providers.example.yamlを正としてください。

## 環境変数を一時的に設定する

環境変数とは、アプリの外側から値を渡すWindowsの仕組みです。次は、PowerShellを閉じると消える一時設定の考え方です。

1. プロジェクトフォルダを開きます。
2. フォルダのアドレス欄へ powershell と入力してEnterを押します。
3. 設定例に書かれたAPIキー用の環境変数名を確認します。
4. 次の例の YOUR_API_KEY_ENV を、その環境変数名へ置き換えて実行します。

    $secret = Read-Host "APIキーを入力してください" -AsSecureString
    $env:YOUR_API_KEY_ENV = [System.Net.NetworkCredential]::new("", $secret).Password

5. base URLやモデル名も環境変数から読む設定なら、見本の環境変数名へ値を設定します。

    $env:YOUR_BASE_URL_ENV = "公式資料で確認したbase URL"
    $env:YOUR_MODEL_ENV = "契約で利用できるモデル名"

6. 同じPowerShell画面で次を実行します。

    .\run.bat

7. 終了後、必要なら環境変数を消します。

    Remove-Item Env:YOUR_API_KEY_ENV
    Remove-Item Env:YOUR_BASE_URL_ENV
    Remove-Item Env:YOUR_MODEL_ENV

この例の名前や値をそのまま使うのではなく、providers.example.yamlと各サービスの公式資料に合わせてください。コマンド履歴、画面写真、コピーしたメモにキーを残さないよう注意してください。

永続的な環境変数を使う場合は、Windowsの「環境変数」画面からユーザー環境変数として設定できます。ただし、同じWindows利用者権限で動く別のプログラムから読み取られる可能性があります。共有PCでは設定せず、APIキーを使用後に無効化できる運用を優先してください。

## .env.exampleについて

.env.example は、必要な環境変数名を案内する見本です。実在するAPIキーを含めません。このプロジェクトでは.envをGitの対象外にしますが、「対象外だから安全」とは限りません。今回の安全方針では、実在するAPIキーをプロジェクト内の.envへ保存せず、Windowsの環境変数から渡してください。

## Qwen、DeepSeek、OpenAI

3つの接続先は、OpenAI互換のリクエストと応答を利用できる範囲で共通Providerを使います。

OpenAI公式APIでGPT-5系・GPT-6系を選ぶ場合は、Chat Completionsへ `max_completion_tokens` を送り、既定の推論設定と非互換な `temperature` は省きます。設定ファイルの出力上限はそのまま使います。利用者が送信形式を手作業で変更する必要はありません。他社の互換APIへこの形式を一律に適用しません。

Web検索・画像共有はOpenAI公式の `https://api.openai.com/v1` 接続で、現在設定されているモデルをResponses APIへ渡します。Luna 5.6へのモデル名固定はありません。選んだモデルに必要な機能がない場合や利用が拒否された場合は、エラーを表示し、別モデルへ自動で切り替えません。すべてのモデルの機能対応を保証する意味ではありません。

標準の `openai` 設定でGPT-6 Lunaへ切り替える値は `OPENAI_MODEL=gpt-6-luna` です。独自の設定では `local/providers.yaml` の `model_env` が指す環境変数を使います。モデル設定を変更してアプリを再起動し、右上の歯車「設定」→「診断情報」→「モデル:」で確認してください。Windowsの環境変数が `.env` より優先され、開いたままのターミナルは以前の値を引き継ぐため、起動元も開き直します。APIキーは従来のOpenAI用を使います。アカウントでの利用可否はOpenAI側の契約・権限によります。

根拠（2026-09-24確認）：[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)、[公式移行ガイド](https://developers.openai.com/api/docs/guides/latest-model)。Luna 6はChat Completions、画像入力、ResponsesのWeb検索に対応します。アプリの推論強度は従来同様に明示指定せず、モデルの既定値を使います。今回、実際のAPIへの有料検証は実行していません。

設定時は各サービスについて、次を公式資料で確認してください。

- OpenAI互換APIとして利用する正しいbase URL
- 契約や地域で利用できるモデル名
- APIキーの発行方法
- ストリーミング対応
- 使用量情報の返し方
- 料金、上限、利用規約、データ取り扱い

これらは変更される可能性があるため、この文書に固定値を写すより、公式資料とconfig/providers.example.yamlを照合してください。

## 共通回答形式

アプリ内部では、LLMの返答を次の形へそろえます。

- speech: 発言文字列
- emotion: neutral、happy、sad、angry、embarrassed、surprised
- emotion_intensity: 0から1
- gesture: 文字列
- gaze: 文字列
- memory_candidates: 記憶候補の配列

LLMが正しいJSONを返さなかった場合は、返答全文をspeechとして扱います。第7段階ではspeechを会話欄へ表示し、正しいJSONで返った場合だけemotion、gesture、gazeを動きのあるオリジナル女の子風2Dアバターへ反映します。音声の話し方は変更しません。

## 接続エラーの確認順

1. Mockで送受信できるか確認します。
2. 画面の使用中Provider名とモデル名を確認します。
3. local側の選択名が、Provider設定の名前と一致するか確認します。
4. 必要な環境変数が同じPowerShell内にあるか確認します。
5. base URLとモデル名を公式資料で確認します。
6. APIキーが有効か、利用上限を超えていないか管理画面で確認します。
7. タイムアウト、ネット接続、サービス稼働状況を確認します。

環境変数の存在確認をするときも、APIキーの値自体を画面へ表示しないでください。「設定されている／いない」だけを確認します。

## ログとエラー報告

APIキー、Authorizationヘッダー、秘密を含むURLをログへ出してはいけません。エラーを報告するときは、次だけを安全な範囲で共有します。

- Provider名
- モデル名（秘密でない場合）
- エラーの種類と日本語メッセージ
- 発生時刻
- Mockでは成功するか
- test.batの結果

会話本文やAPI応答に秘密が含まれる場合は、共有前に削除してください。
