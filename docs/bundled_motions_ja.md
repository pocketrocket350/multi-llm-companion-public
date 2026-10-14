# 同梱VMDの出典と規約

**オトカム (Otokam_0510)** 氏のVMD26件（反応21件・待機5件）を同梱しています。原版23件と補正3件で、カタログの有効19件・無効7件を保持しています。VMDには作者の規約が適用されます。

## 出典と原文規約

| パック | 正式な配布元 | 同梱する原文 |
| --- | --- | --- |
| エモートVol.1 | [BowlRoll file/236370](https://bowlroll.net/file/236370) | [Readme](../bundled/licenses/otokam/vol1/Readme.txt) |
| エモートVol.2 | [BowlRoll file/239977](https://bowlroll.net/file/239977) | [Readme](../bundled/licenses/otokam/vol2/Readme.txt) |
| エモートVol.3 | [BowlRoll file/245803](https://bowlroll.net/file/245803) | [Readme](../bundled/licenses/otokam/vol3/Readme.txt) |
| 待機モーション | [BowlRoll file/237411](https://bowlroll.net/file/237411) | [Readme](../bundled/licenses/otokam/idle/Readme.txt) |

同梱Readmeは原文をバイト一致で保持しています。規約には、改変の有無にかかわらず再配布時に配布元を記載し規約を継承すること、モーション単体の販売禁止、利用・商用・改変は使用ソフトの規約にも従うことが記載されています。利用するPMX・ソフト・音声の条件は別に確認します。

クレジットは「モーション：オトカム (Otokam_0510)／パック名／上記の正式配布URL」とします。補正版には以下の変更内容も添えます。ファイルごとのパック、原文規約、権利記録、SHA256は [同梱一覧](../bundled/FILE_MANIFEST.json) に記録しています。

## 補正版3件

| 識別名 | 原本からの変更 |
| --- | --- |
| sample_otokam_048（両手後ろ） | 左右腕26キーへ親座標X軸-8度を合成。時刻・移動・補間・他ボーン・瞬きを保持 |
| sample_otokam_056（左手腰当て） | 右腕フレームへ親座標Z軸-8度を合成。左腕・他ボーン・時刻・移動・補間・瞬きを保持 |
| sample_otokam_058（猫背） | 左右腕へZ軸+15度/-15度、上半身2へX軸-10度、頭へX軸+10度を合成。原本の20秒、キー時刻・移動・補間・瞬きを保持。開始・終了キーの追加なし |


## 導入

初回の `setup_windows.bat` で、素材を `local/motions/otokam`、規約と権利記録を `local/licenses/otokam`、カタログを `local/vmd_catalog.yaml` へ配置します。既存カタログがある場合は追加・上書きしません。

同名の異なるファイルやハッシュ不一致があれば停止します。競合ファイルを削除せず、新しいフォルダーへ展開してセットアップしてください。途中中断後に同じ内容のファイルだけが残っている場合は、再実行で続きを配置できます。

再生にはPMXの導入が必要です。[基準構成の設定](reference_setup_ja.md) と [VMDの使い方](vmd_catalog_ja.md) を参照してください。

## 最新規約の確認状況

2026年10月15日に4パックの公式配布ZIPを取得し、公式ページのMD5・SHA1と一致すること、各Readmeが同梱原文と全バイト一致することを確認しました。上記の再配布・改変条件に変更はありません。将来更新された場合は、その時点の規約を確認してください。
