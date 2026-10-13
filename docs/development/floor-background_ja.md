# 床付き背景への差し替え

2026-09-24、利用者の依頼で提供画像へ差し替えた。背景と雨衣の拡大・縮小の連動は利用者確認済みで、その計算は変更していない。今回の床付きの構図は実装・隔離アプリ確認済み、利用者の実機受入は待ち。

## 変更

- 添付された `c212432e-5d1b-4e7d-8b4d-f1a211677698.png` を、加工せず `local/design/phase11-20260924/background-with-floor.png` へコピーした。原本とコピーのSHA256は一致。
- `src/multi_llm_companion/ui/companion_background.py` の読込先だけを新画像へ変更した。旧画像 `local/design/phase11-20260922/revision-02/background-reference.png` と以前のデザイン資料は保持。
- 採用画像の追記先は `docs/development/phase11-ui-options_ja.md` と `docs/development/background-camera-link_ja.md`、今回の記録はこの文書。
- 人物の位置・身長基準、倍率連動、カメラ、モデル、表情、モーション、通常会話・設定は変更しない。背景の配布条件は未確認のまま、画像はlocal内で利用する。

## 検証

- 背景回帰: **22件成功 / 2.19秒**。`local/diagnostics/floor-background-20260924/focused.log`。
- 全体Ruff: 最終のtest.bat内でcheck成功、formatは256ファイル一致。
- 隔離した実PMXアプリ: **10状態成功**。全身で靴が床領域に収まることを目視確認し、バストアップ、ホイールによる縮小・復帰、小画面・横長画面、管理用全身画角からの復帰を確認した。`local/diagnostics/background-camera-20260924/floor-background/` のPNGと `result.json`。通常DB・local YAML・QSettings・旧背景の検査前後ハッシュは一致。通信・実音声は使っていない。
- `test.bat`: **1190件成功 / 123.24秒**、Ruffを含む終了値0。`local/diagnostics/floor-background-20260924/test-bat.log` と `test-bat-result.json`。前回のQt停止は今回再現しなかったが、その原因は未確定のまま。

新規テストは追加していない。差し替えた実画像の読込と表示は隔離アプリで検査し、倍率計算には既存の回帰テストを使った。見た目の好みは利用者実機受入待ち。

## 反映と確認

1. アプリを閉じ、いつもの `run.bat` で起動すると、新しい画像になる。
2. 床のある構図を見たい場合は「設定」→「表示・アバター」→「雨衣の表示範囲」で「全身」を選ぶ。雨衣の靴が床の上に表示されるのが正常。バストアップでは足元と床は画面外になる。
3. ホイールで拡大率を変えた場合は、同じタブの「雨衣を正面に戻す」で標準倍率に戻せる。表示の崩れがあれば操作を止め、再起動しても残るかと、画面の画像を知らせる。

## 復元

今回の編集前の3ファイルは `local/diagnostics/floor-background-20260924/before/` に控えた。背景だけを以前へ戻すには、アプリ終了後、`companion_background.py` の読込先を旧画像へ戻す。後続変更がなければ同名の控えを利用できる。新旧の画像、通常DB・設定・PMX/VMDは削除不要。
