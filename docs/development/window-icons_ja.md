# タイトルバーと会話操作のアイコン

2026-09-24。利用者の指定に合わせ、既存の水色・ミント・濃紺を主画面のタイトルバーと5つの操作へ適用した。編集前の範囲・合否条件は `local/diagnostics/window-icons-20260924/plan.md` に記録。途中の中断依頼では作業を止め、再開依頼後に残る検査を行った。表示変更は実装・関連テスト・Codex環境の統合確認済み。全体テストはQt内部の異常で未合格、利用者実機受入は待ち。

## 変更

- タイトルバーは背景 `#E8F5F9`、文字 `#1A267C`。主画面と、その子である設定・管理画面へ反映する。移動・サイズ変更・最小化・最大化・閉じるはWindows標準枠を維持する。
- 履歴は三本バー、設定は歯車、マイクはマイク、送信は三角矢印、画像添付はプラス。図形は濃紺、マイクの背景はミント、送信は水色。操作できない状態の図形と背景は薄い色で区別する。
- ツールチップとアクセシビリティ名に従来の操作名を残した。Tab・Space・Enterによる既存の操作は維持。録音中は停止の四角と「録音停止」、認識待ちは無効なマイクと「認識中」になり、入力欄上の録音状態・音量・エラーも残す。
- 最前面切替でWindowsのハンドルが変わってもタイトルバー色を再適用する。親の異なるウィンドウには適用しない。
- アイコンは既存PySide6のSVG描画と内製の単純な図形。外部フォント・画像素材・ライブラリの追加はない。保存形式・公開API・依存関係・通常データは変更しない。

Windowsの配色には公式の [DwmSetWindowAttribute](https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/nf-dwmapi-dwmsetwindowattribute) と [DWMWA_CAPTION_COLOR / DWMWA_TEXT_COLOR](https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute) を使用した。対応はWindows 11 Build 22000以降で、未対応OSは標準色のまま動作する。Qtのウィンドウ識別子の変更通知は [QWidget::winId](https://doc.qt.io/qt-6/qwidget.html#winId) に基づく。

## 検証

- 関連自動テスト: **99件成功 / 46.93秒**。アイコン・操作名・キーボード、マイクの録音／認識／通常エラー、履歴・設定、最前面・設定保持、画像送信の既存境界を検査。`focused-final.log`。追加した3件のうち最初は子ダイアログへの適用検査1件が失敗した。Qtのウィンドウ境界を越えて所有者を確認する修正後に成功した。
- 変更したコード・テストのRuff checkとformat確認は成功。全体のRuffも別途実行し、check成功、formatは260ファイル一致。
- Windowsの隔離した実PMXアプリ: Qtの表示倍率 **100％と150％で各12状態**。通常・履歴・送信可能・模擬録音・設定・最前面オン／オフ・760×540・最大化・最小化からの復帰を確認。アイコンと小画面の見え方を目視し、タイトルバーの水色と濃紺の文字は取得画像のピクセルでも照合した。
- 送信はMockProviderによる返答まで、プラスはファイル選択処理への到達までを検査。録音状態は模擬で実マイクは使用せず、外部要求・実音声出力は0件。通常DB・local YAML・通常QSettingsの検査前後は一致。
- `test.bat` は1回実行し、1195件を収集したが、`test_image_sharing_ui.py::test_image_error_returns_to_normal_chat_without_automatic_resend` 中に **Windows fatal exception: access violation** が発生。PythonフレームのないQtスレッドと、UI側の `wait_until` 待機を記録した。240秒の実行上限と終了待ち15秒でも終わらず、実行したテストの親子プロセスを特定して終了した。残存プロセスがないことも確認。**全体成功とは報告しない**。`test-bat.log`、`test-bat-result.json`。
- 停止した1ケースを専用一時フォルダーで比較すると、現行ソースは **1件成功 / 1.43秒**、今回変更前の `chat_window.py` を別のソースコピーへ戻した場合も **1件成功 / 2.04秒**。`current-failure.log`、`baseline-failure.log`、`failure-comparison.json`。全体実行でのQt停止の原因・今回変更との因果関係は未確定。過去にも同じ検査の停止記録があるが、同一原因とは断定しない。テストの削除・無効化・基準緩和は行っていない。

証拠は `local/diagnostics/window-icons-20260924/` 内。100％の実行結果は `native-100-check/result.json`、タイトル色の画像照合は同じ場所の `caption-pixel-verification.json`。150％は `native-150/result.json`。150％はQtプロセスの倍率を指定し、利用者のWindows設定は変更していない。異なるDPIのモニター間の移動は未検査。

当初のDWM属性読取検査は色属性の取得で `E_INVALIDARG` になった。設定APIの成功と、実際の標準タイトルバー画像の配色を確認する方法へ切り替えた。`native-100`、`native-diagnostic` はその診断記録で、成功した統合検査として数えない。`*-frame.png` はWindowsの標準枠を取得しており、OpenGLの主画面内容は白くなることがある。主画面内容は別のQt取得画像 `*.png` で確認し、合成した画面を実キャプチャとしては扱っていない。

実装・自動テスト・Codex環境の統合確認と、利用者環境での見やすさの受入を区別する。実API・実音声の再評価は今回の表示変更には含めない。

## 反映と操作

1. アプリを閉じ、いつもの `run.bat` から起動する。会話履歴・設定は維持する。
2. 左上の三本バーで履歴を開閉し、右上の歯車で設定を開く。マウスを重ねると、それぞれ「履歴」「設定」と出る。設定の「閉じる」で会話へ戻れる。
3. 入力欄右のマイクが録音、右隣の三角矢印が送信、入力欄下のプラスが画像選択。画像選択だけでは送信しない。文字が空の間は送信の矢印が薄くなる。
4. 正常時は、上のバーが画面と同じ水色で、操作が濃紺のアイコンになっている。停止・検索取消など実行中の安全表示は従来の文字付き操作を使う。録音だけは同じマイク位置の四角で停止できる。
5. これらの操作検査はCodexで実施済み。利用者に一式の再検査や追加の有料送信は求めない。使いづらい箇所があれば、そのアイコンと画面倍率を知らせる。表示が欠ける場合は操作を止めて再起動し、会話データや設定は削除しない。

## 変更ファイルと復元

- `src/multi_llm_companion/ui/chat_window.py`: 5ボタンへの反映、マイク状態、タイトルバー表示イベント。
- `src/multi_llm_companion/ui/_window_chrome.py`: 内部用の図形・タイトルバー配色処理（追加）。
- `tests/test_window_chrome.py`: 操作・状態・適用範囲の回帰テスト3件（追加）。
- `docs/development/phase11-user-check_ja.md`、`phase10-user-check_ja.md`: 操作名とアイコンの対応。
- `docs/development/window-icons_ja.md`: この記録（追加）。

既存3ファイルの変更前は `local/diagnostics/window-icons-20260924/before/` にある。戻す場合はアプリを閉じ、後続変更がなければ同名の控えを対応する場所へ戻す。追加ファイルはその後に取り除ける。後続変更があれば `changes.diff` を使い今回の差分だけを戻す。通常DB・QSettings・PMX/VMD・背景を消す必要はない。
