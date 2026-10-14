# 雨衣の音声・MMDを設定する

[初心者ガイド](beginner_guide_ja.md)でMockが起動した後の手順です。基準構成はOpenAIの `gpt-6-luna`、VoiSona Talk「雨衣」、雨衣MMDです。ソフト・ボイス・PMXは [利用条件](rights_and_credits_ja.md) を確認して正規に用意します。

## 1. 表情設定とPMXを導入する

同梱の `config/avatar.reference.example.yaml` は、`雨衣_2025_0516.pmx` 向けの表情・口形・視線・動作設定です。モデル本体とテクスチャは含みません。

1. アプリを終了します。`local/app.yaml` を同じ場所へコピーし、`app.before-reference.yaml` と名前を変えて控えにします。同名ファイルは上書きしません。
2. PMXと付属テクスチャを、フォルダー構造を保って `local/models/mmd/` 内へ置きます。
3. `config/avatar.reference.example.yaml` を `local` へコピーし、`avatar.reference.yaml` と名前を変えます。同名ファイルがあれば上書きせず中止します。既存の `avatar.yaml` は残します。
4. コピーしたファイルをメモ帳で開き、`three_d` 直下の `model_path: ''` を実際のパスへ、同じ階層の `enabled: false` を `enabled: true` へ変更します。`provider: pmx_preview` はそのままです。

```yaml
  model_path: local/models/mmd/モデルのフォルダー/雨衣_2025_0516.pmx
```

5. `local/app.yaml` の `avatar_config_path` を `local/avatar.reference.yaml` に変更します。項目がなければ行頭から1行追加し、重複させません。
6. `run.bat` で起動します。PMXが表示され、**設定 → 診断情報** の3D・表情・VMDに読込みエラーがなければ導入できています。

表情は **設定 → 表示 → 表情を管理** でプリセットを選ぶと確認できます。終了は **プレビューを終了 → 閉じる** です。動きの確認は [VMDの使い方](vmd_catalog_ja.md) を参照してください。

モデルが出ない・対応モーフ不足・大きな表示崩れがある場合は終了し、モデルの版と配置を確認します。`local/app.yaml` の `avatar_config_path` を元の値へ戻せば、以前の表示設定へ戻れます。

## 2. AI・音声・マイクを設定する

1. [LLM設定](providers_ja.md)でAPI接続とモデルを設定します。
2. [音声ガイド](audio_ja.md#voisona-talkの声で読み上げる)でVoiSona Talkを準備し、使用するボイス名を明示します。
3. マイクを使う場合は、同じ音声ガイドの [マイク入力](audio_ja.md#マイクで入力する) を設定します。
4. アプリを再起動します。文章を送り、選んだ声とMMDの口パクを確認します。実APIの会話には料金が発生する場合があります。

問題があれば **音声・動作を停止**、録音中なら **録音停止** を押します。音声を使わない状態へ戻すには、終了後に `local/audio.yaml` の `voice_output.enabled` を `false` にして再起動します。

検索・画像共有・能動的な話しかけは [会話の操作](conversation_ja.md)、保存データの移行は [バックアップと復旧](backup_and_recovery_ja.md) を参照してください。
