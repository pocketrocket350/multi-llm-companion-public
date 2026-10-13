# 音声機能の準備（第2段階）

第2段階では、WhisperまたはVoskによるマイク音声認識と、VoiSona TalkのREST APIによる音声出力を利用できます。音声機能を設定していなくても、従来の文字チャットは動作します。

## 1. 第2段階のパッケージを入れる

1. アプリを閉じます。
2. `setup_windows.bat` をもう一度ダブルクリックします。
3. 「セットアップが正常に終わりました」と表示されるまで待ちます。

既存の会話履歴やlocal設定は上書きされません。新しく `local/audio.yaml` が作成されます。

## 2. Whisperを準備する（推奨）

Whisperは高精度な方式です。音声はPC内で認識し、録音内容を外部へ送信しません。初回だけ認識モデルのダウンロードが必要です。

1. `download_whisper_model.bat` をダブルクリックします。
2. 「Whisperモデルを利用できます」と表示されるまで待ちます。
3. `run.bat` でアプリを起動します。
4. 「設定」を押します。
5. 音声認識で「Whisper（高精度・標準）」を選びます。
6. 使用するマイクを選び、「保存」を押します。

Whisperの `small` モデルをCPUの8ビット計算で使用します。モデルは `local/models/faster-whisper-small` に保存され、Gitへ追加されません。取得が中断しても、同じバッチを再実行すると途中から再開します。

Whisperだけでも利用できます。Voskを残している理由は、Whisperより読み込みと処理が軽く、低性能PCでの予備として使えるためです。

取得対象の出典とライセンス表示は [Systran/faster-whisper-smallの公式モデルカード](https://huggingface.co/Systran/faster-whisper-small) で確認できます。認識モデルは配布ZIPへ同梱せず、利用者のPCで取得します。

## 3. Voskを使う場合（軽量）

Voskは、マイク音声を外部サービスへ送らず、PC内で文字へ変換します。ただし、日本語の学習済みモデルを別途配置する必要があります。

1. [Vosk公式モデル一覧](https://alphacephei.com/vosk/models)から `vosk-model-small-ja-0.22` をダウンロードします。
2. ZIPファイルを展開します。
3. 展開されたフォルダを `local/models/` の中へ置きます。
4. `local/audio.yaml` の `speech_recognition.enabled` を `true` にします。

最終的な場所は `local/models/vosk-model-small-ja-0.22` です。モデルは約48MBで、Apache 2.0 Licenseです。モデル本体はGitへ追加しません。

アプリ起動後、「マイク入力開始」を押して話します。一つの発話を認識すると自動で停止し、認識結果が入力欄へ入ります。誤認識がないか確認してからEnterで送信してください。Shift+Enterは改行です。

設定画面の音声認識で「Vosk（軽量）」を選択します。

## 4. マイク画面の見方

- PTT: 現在のマイク操作状態です。`待機`、`録音中`、`認識済み`を表示します。「マイク入力停止」は録音を取り消して待機へ戻ります。
- マイク音量: 声の大きさを表示します。話しても動かない場合はマイク選択やWindowsの入力音量を確認します。
- 認識途中: 確定前の文章です。Whisperでは一定間隔で更新されるため、PC性能によって表示に時間がかかります。
- 無音終了: 話し終わって約1.2秒間無音になると、自動的に録音を終了します。

## 5. VoiSona Talkを準備する

1. VoiSona Talkを起動してログインします。
2. 利用条件を確認したボイスライブラリを少なくとも一つダウンロードします。
3. 画面右上の三本線から「編集」→「環境設定」→「API」を開きます。
4. 待ち受けポートを確認します。見本設定は既定値の32766です。
5. API用パスワードを設定します。ログイン用パスワードと同じである必要はありません。
6. 「REST APIを有効にする」を有効にします。

REST APIはベータ版です。VoiSona Talk更新後に動かなくなった場合は、公式のAPIリファレンスと更新情報を確認してください。

## 6. 秘密値をWindows環境変数へ設定する

API用パスワードをYAML、`.env`、Pythonコードへ書かないでください。Windowsの「環境変数」画面で、次を利用者環境変数として登録します。

- `VOISONA_TALK_USERNAME`: VoiSona TalkのAPI画面に表示されたユーザー名
- `VOISONA_TALK_PASSWORD`: API画面で自分が設定したAPI用パスワード
- `VOISONA_TALK_VOICE_NAME`: 使用するボイスの `voice_name`（空の場合は取得できた先頭の声）

設定後は、すでに開いているアプリやコマンド画面を閉じてから開き直します。

## 7. 音声出力を有効にする

1. `local/audio.yaml` の `voice_output.enabled` を `true` にします。
2. VoiSona Talkを起動し、REST APIを有効にしたままにします。
3. `run.bat` で会話アプリを起動します。
4. 画面上部の「音声出力: 設定済み」を確認します。
5. 文章を送信し、回答後にVoiSona Talkから音声が出ることを確認します。

## 8. 音声同期口形の確認と調整（第21段階）

このアプリは、VoiSona Talkへ一時WAVの生成を依頼し、そのWAVをアプリ自身で再生します。再生機能がスピーカーへ渡した位置を口パクの時計として使うため、VoiSonaの合成進捗を音声の再生位置とはみなしません。一時WAVはWindowsの一時フォルダーだけに作成し、メモリへ読み込んだ直後に削除します。リポジトリや `local` には保存しません。

VoiSona Talkの音声依頼または状態確認の応答に音素の開始・終了時刻が含まれる場合は、その時刻を優先して「あ・い・う・え・お」の口形を切り替えます。画面のアバター欄が `口形: 音声同期 あ` なら、この方式です。

VoiSona TalkのAPI設定やバージョンによって音素時刻が返らない場合、画面には `口形: 推定 あ` と表示されます。この場合も、開始と終了は実際のWAV再生位置に合わせ、途中の母音だけを文章から推定します。

```yaml
voice_output:
  playback_preroll: 0.12            # Windows音声機器の準備用に先行する無音
  lip_sync_timing_mode: auto        # APIの音素時刻を優先する（推奨）
  lip_sync_start_delay: 0.0         # 通常は0。再生開始後の口形だけを遅らせる秒数
  lip_sync_character_duration: 0.22 # 音素時刻がない場合の文字間隔の比率
```

`playback_preroll`は音声本文を削ったり遅らせたりする値ではなく、Windowsの出力機器を開いた直後に短い無音を先行させる値です。口パクの時計からは除外されます。通常は`0.12`のままで構いません。

通常は `lip_sync_timing_mode: auto` のままで構いません。時刻情報を返さない環境では、まず既定値のまま確認してください。APIの互換性を確認するためだけに `estimated` を指定すると、常に文章推定方式を使えます。

アプリが音声を再生するため、VoiSona Talk側の出力先ではなく、Windowsの既定の出力機器から聞こえます。音が出ない場合は、Windowsの「設定」→「システム」→「サウンド」で既定の出力機器と音量を確認してください。

## 9. 音声・感情・動作表現の統合（第43段階）

VoiSona Talkの音声表現は初期状態で無効です。未設定または`enabled: false`なら`global_parameters`を追加せず、第42段階までと同じ音声リクエストを使います。

```yaml
voice_output:
  expression:
    enabled: false
    max_speed_change: 0.12
    max_intonation_change: 0.30
```

有効時は、平滑化後の感情から作った「音声専用強度」だけを使い、VoiSona Talk REST APIのボイスに依存しない`speed`（話速）と`intonation`（抑揚）を送ります。`max_speed_change`は0～0.25、`max_intonation_change`は0～0.50の範囲で設定でき、最終値も話速0.85～1.15、抑揚0.80～1.20へ制限します。感情強度そのものをそのまま送信しません。

[VoiSona Talk公式REST APIチュートリアル](https://manual.voisona.com/ja/talk/pc/2b6e9bc7efb180ea86ccc6c7347e9ca6)では、REST APIはベータ版で、`global_parameters`による音声表現が案内されています。一方、[公式パラメータ資料](https://manual.voisona.com/ja/talk/pc/2b6e9bc7efb18061b230e8d6235f8ef4)ではスタイルの種類と数はボイスごとに異なります。このアプリは利用中ボイスのスタイル名・順序を確実に取得する経路を持たないため、`style_weights`を決め打ちしません。画面では「ボイス固有スタイル未対応」と表示します。対応しているのは話速と抑揚だけです。

音声合成の受付、WAV再生、口形は従来どおり別状態です。口形の時計もWAVの実再生位置から変更しません。利用者割り込みではUIの発話世代を進め、VoiSonaの監視、WAV再生、口形、PMX表情、視線、ジェスチャー/VMD、非同期姿勢を一回の操作で無効化します。開始前の古い姿勢完了通知や、停止後の古い音声受付完了は再適用されません。

状態欄は、音声表現が無効なら`無効（従来動作）`、適用時は感情、音声専用強度、実際の話速・抑揚を表示します。API用パスワード、ボイス固有の秘密値は表示しません。

利用者は、音声表現を無効にした従来の音声・口パク・表情・VMDと、有効時の実VoiSona・表現統合状態を確認し、第43段階を受け入れ済みです。ボイス依存の`style_weights`、音量、ピッチ、Alpha、Huskinessの自動感情制御は、第43段階の対応済み項目には含めません。

## 10. 文単位の音声先読みと遅延測定

LLMが構造化応答の`speech`を逐次返す場合、`。`、`！`、`？`、改行などで確定した文を待たずにVoiSona Talkへ合成依頼します。一文目の再生中に次の文を最大2件まで並列合成し、回答全体の生成完了を待たずに読み上げを始めます。最初の再生可能なWAVは、次文の合成完了を待たずに直ちに再生します。再生中に合成済みの文を取り込み、現在再生できる文の開始を次文待ちで遅らせないようにします。合成完了順が入れ替わっても文番号順に再生し、割り込み時は未再生の合成を破棄します。文の順序、実WAV再生位置、口形時計、割り込み世代は従来どおり維持します。音声表現は文全体の感情確定前に開始するため、先読み文は基本音声を使い、最終的な表情・視線・動作は回答確定後に更新します。

入力終了からの経過は、本文や認証情報を記録しない診断イベントとして測定できます。Windowsのコマンドプロンプトで次を設定してからアプリを起動してください。

```bat
set MLC_LIP_SYNC_DIAGNOSTIC=1
run.bat
```

`local/diagnostics/lip_sync_trace.jsonl`へ、次のイベントが同じ`monotonic_ms`時計で追記されます。

- `companion/input_finished`: 送信ボタンまたはEnterで入力を確定した時刻
- `companion/ptt_pressed` / `recognition_started` / `recognition_finished`: マイク開始、認識開始、認識結果を確定した時刻
- `recognition/speech_end_detected`: 無音終了により認識器が利用者の発話終了を判断した時刻。認識確定との差で、Whisper・Voskの確定処理を分けて測れます
- `companion/recognition_stop_requested`: 利用者が認識停止を要求した時刻
- `companion/voice_stop_requested` / `voice_stopped`: 新しい発話による旧音声停止の要求と、アプリ状態が停止へ戻った時刻
- `wav_player/playback_stop_completed`: WAV再生器の停止呼出しが戻った時刻。可聴音をマイクで測った時刻ではありません
- `companion/llm_first_chunk`: LLMの最初の本文チャンクを受け取った時刻
- `companion/llm_sentence_ready`: 読み上げ可能な一文が確定した時刻
- `voice/stream_synthesis_requested` / `stream_synthesis_completed`: VoiSona合成の依頼と完了
- `voice/stream_managed_wav_playback_started`: WAVを再生器へ渡した時刻。これは実音声開始ではありません。`sentence_index`で文番号、2文目以降は`playback_gap_ms`で直前の再生終了からの待ち時間を確認できます
- `wav_player/playback_position_started`: Windows音声機器の実再生位置が初めて進んだことを観測した時刻。受入時の再生開始確認にはこちらを使います

`monotonic_ms`の差分から、PTT開始→認識開始・発話終了・確定、旧音声停止要求→停止確認、入力終了→LLM本文、文確定→VoiSona音声データ受信、音声データ受信→再生器投入、実再生位置の観測→最初の口形反映、再生完了→連続録音開始を分けて確認できます。各ターンにはProvider名とモデル名だけを記録し、本文や認証情報は記録しません。診断が不要になったら、コマンドプロンプトを閉じるか`MLC_LIP_SYNC_DIAGNOSTIC`を削除してください。診断ファイルは公開物へ追加しないでください。

記録後は、外部通信なしで次を実行すると、PTT・認識・旧音声停止と、入力からLLM先頭・文確定・音声要求・VoiSona音声データ受信・再生器投入・実再生位置の観測・口形反映・再生完了・連続録音開始までのミリ秒を会話ごとに集計できます。

```bat
python scripts/analyze_lip_sync_trace.py --latest --output local/diagnostics/lip_sync_summary.json
```

本文、会話ID、APIキーは集計結果に含めません。

20ターン以上を同じ条件で記録した後は、`--aggregate`を付けると、各区間の標本数、中央値、95%点（nearest-rank）、最大値を追加で出力します。`playback_not_observed_count`は再生器へ最初のWAVを渡せなかったターン数、`playback_position_not_observed_count`はWindows実再生位置の進行を観測できなかったターン数です。受入時の再生開始時間は後者と`input_to_playback_position_observed_ms`を使います。`cycle_timing`は、60msの口形UIタイマーと33msのPMX軽量・重量タイマーについて、呼出し間隔の中央値・95%点・最大値、処理時間、設定周期を超えた回数を示します。これはUI・PMXのコールバックまでの計測であり、GPU描画完了や実際に聞こえた音を測るものではありません。音声先頭・末尾の欠落、古い表情・視線・VMDの再開は、同じ記録中のWAV・口形・描画イベントと実機の音を別に確認します。

## トラブル確認

- 「Voskモデル未配置」: Voskを選んだ場合はモデルのフォルダ名と場所を確認します。
- Whisperの準備エラー: 初回はネット接続、ディスク空き容量を確認し、`download_whisper_model.bat` を再実行します。
- マイクを開始できない: Windows 11のマイク設定で、デスクトップアプリの使用を許可します。
- 「環境変数未設定」: ユーザー名とAPI用パスワードの環境変数を確認し、アプリを再起動します。
- VoiSona接続エラー: VoiSona Talkが起動中か、REST APIが有効か、ポート番号が一致するか確認します。
- 声が違う: `VOISONA_TALK_VOICE_NAME` を確認します。空の場合は、APIが返す先頭の声を使用します。
- 音声表現が変わらない: 状態欄が`音声表現: 無効（従来動作）`なら、`voice_output.expression.enabled`を確認します。`neutral`または強度0では、有効でも通常値を送ります。

パスワードそのものを画面写真、質問文、ログへ貼らないでください。
