# マイク入力と音声読み上げ

`setup_windows.bat` が完了した環境で設定します。マイクと読み上げは別々に有効にできます。

## マイクで入力する

### Whisper

1. アプリを終了し、`download_whisper_model.bat` を開きます。初回はモデルのダウンロードが必要です。
2. **Whisperモデルを利用できます** と出たら、`local/audio.yaml` の `speech_recognition` 内にある `enabled: false` を `enabled: true` にして保存します。
3. `run.bat` で起動し、**設定 → 会話・音声 → LLM接続・マイクの設定** を開きます。**Whisper（高精度・標準）** と使用するマイクを選び、**保存** します。
4. 入力欄右の **マイク** を押して話します。無音になると録音が終わり、認識文が入力欄へ入ります。内容を確認してから送信します。

録音を取り消すときは **録音停止** を押します。認識はPC内で行い、録音音声は外部LLMへ送りません。認識後の文章は送信ボタンを押すと会話として送られます。

モデルは `local/models/faster-whisper-small` へ保存します。ダウンロード中断時は同じバッチを再実行できます。出典・条件は [モデルカード](https://huggingface.co/Systran/faster-whisper-small) を参照してください。

### Voskを使う場合

[Vosk公式モデル一覧](https://alphacephei.com/vosk/models)から `vosk-model-small-ja-0.22` を入手し、展開したフォルダーを `local/models/` へ置きます。終了中に `local/audio.yaml` の `speech_recognition.enabled` を `true` にして起動し、同じ設定画面で **Vosk（軽量）** とマイクを選び保存します。こちらもPC内で認識します。

## VoiSona Talkの声で読み上げる

### 1. VoiSona Talkを準備する

1. VoiSona Talkへログインし、使うボイスを導入します。基準構成は雨衣です。
2. 右上の三本線から **編集 → 環境設定 → API** を開きます。
3. API用パスワードを設定し、**REST APIを有効にする** をオンにします。ポートの見本値は `32766` です。

VoiSona Talkは起動したままにします。ソフト・ボイス・生成音声の条件は [権利表記](rights_and_credits_ja.md) を確認してください。

### 2. 認証情報とボイス名を設定する

Windowsのスタートで「環境変数」と検索し、**環境変数を編集** を開きます。利用者環境変数の **新規**（既存は **編集**）で次を登録します。

| 変数名 | 値 |
| --- | --- |
| `VOISONA_TALK_USERNAME` | API画面のユーザー名 |
| `VOISONA_TALK_PASSWORD` | 自分が設定したAPI用パスワード |

**OK** で閉じてから、アプリのフォルダーを右クリックして **ターミナルで開く** を選びます。PowerShellで次を実行します。

```powershell
.\.venv\Scripts\python.exe -m scripts.list_voisona_voices
```

利用できるボイス名だけが表示されます。使う名前を引用符なしで、利用者環境変数 **`VOISONA_TALK_VOICE_NAME`** に登録します。雨衣が一覧にない場合はVoiSona Talk側の導入を確認してください。このコマンドは音声合成を行いません。

名前が空欄・不一致の場合は合成を止めます。旧設定で名前を省略していた場合も明示設定が必要です。認証情報は設定ファイルや画面写真へ残さないでください。

### 3. 読み上げを有効にする

1. アプリを終了し、`local/audio.yaml` の `voice_output` 内にある `enabled: false` を `enabled: true` にします。
2. アプリとターミナルを開き直し、`run.bat` で起動します。環境変数が反映されない場合はWindowsへサインインし直します。
3. 文章を送り、指定した声で読まれることを確認します。途中で止めるには **音声・動作を停止** を押します。

使わないときは同じ `enabled` を `false` に戻して再起動します。接続先は `local/audio.yaml` の `voice_output.base_url` で、見本は `http://127.0.0.1:32766/api/talk/v1` です。

## 口パクと声の表現

口パクはWAVの実再生位置に合わせます。`voice_output.lip_sync_timing_mode: auto` は音素時刻を優先し、取得できない場合は文章から推定します。

感情に応じた話速・抑揚を使う場合は `voice_output.expression.enabled` を `true` にします。既定では無効です。ボイス固有の `style_weights` は未対応です。通常の調整値は [音声設定の見本](../config/audio.example.yaml) を参照してください。

## 困ったとき

| 症状 | 確認すること |
| --- | --- |
| マイク音量が動かない | 選択した機器、Windowsの入力音量とデスクトップアプリへのマイク許可 |
| 認識モデルを読めない | モデルのフォルダー名・配置。Whisperは取得バッチの完了 |
| VoiSonaへ接続できない | 起動状態、REST API、ポート、認証用環境変数 |
| ボイス名が未設定・見つからない | 一覧コマンドで確認した名前と `VOISONA_TALK_VOICE_NAME` の一致 |
| 音声の先頭が欠ける | `playback_preroll`。通常は見本の `0.12` 秒を使う |

問題が続く場合は音声を無効にして文字会話へ戻します。[復旧手順](backup_and_recovery_ja.md) も参照してください。
