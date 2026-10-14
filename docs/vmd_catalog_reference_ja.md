# VMDの追加登録と設定

通常の操作は [VMDの使い方](vmd_catalog_ja.md) を参照してください。以下は別のVMDを追加・調整する場合の手順です。

## 追加する

1. アプリを終了し、`local/vmd_catalog.yaml` と使用中のアバター設定を別名でコピーします。アバター設定は `local/app.yaml` の `avatar_config_path` で確認できます。
2. VMDの正式な配布元と規約で予定する利用を確認し、ファイルを `local/motions/` 内へ置きます。
3. [権利記録の見本](../config/vmd_license.example.yaml)を `local/licenses/` へ別名でコピーし、作者・配布元・規約・取得日・用途・改変・クレジット・再配布可否を記入します。確認できた場合だけ `usage_confirmed: true` にします。
4. `local/vmd_catalog.yaml` の `motions` へ、重複しない識別名で登録します。カタログがない場合だけ [見本](../config/vmd_catalog.example.yaml)をコピーし、既存の内容へ上書きしません。
5. 待機に使う場合はアバター設定の `three_d.vmd_motion_id` に識別名を指定します。会話候補をカタログから選ぶ場合は `three_d.motion_library_enabled: true` にします。
6. **起動** ショートカットで起動し、**設定 → 表示 → モーションを管理** から読込み状態と見た目を確認します。

## カタログ例

```yaml
motions:
  nod_small:
    file: local/motions/nod_small.vmd
    motion_type: reaction
    action: nod
    moods: [neutral, happy]
    energy: low
    weight: 1
    audio_required: false
    category: head
    body_mask: [head, neck]
    duration_seconds: 1.0
    loop: false
    intensity_scale: 0.6
    cooldown_seconds: 8
    license_record: local/licenses/nod_small.yaml
    compatibility:
      model_names: [PMX内部のモデル名]
      required_bones: [頭, 首]
      required_morphs: []
    enabled: true
```

`model_names`・ボーン・モーフ名は使用するPMXと一致させます。音源が必要なVMDは、音声出力を利用できない間は再生候補から外れます。

| 項目 | 意味 |
| --- | --- |
| `motion_type` | `idle`・`talk`・`reaction`・`dance` |
| `action` | 動作の意味。LLMへファイル名は渡さない |
| `moods` / `energy` | 適する感情・強さ（`low`・`medium`・`high`） |
| `weight` | 同条件での選ばれやすさ |
| `body_mask` | 会話中に使うボーンの範囲。管理プレビューは全トラックを使う |
| `intensity_scale` | 動きの強さ。全身動作では感情強度を掛けない |
| `cooldown_seconds` | 同じ動作を再利用するまでの秒数 |
| `category` | 同じ分類の連続動作を抑えるための分類 |

## 選択と連続動作の調整

`motion_library_enabled: false` の場合は、`three_d.gesture_motion_ids` に指定した会話動作だけを使います。待機用IDと会話用IDを同じにすることはできません。VMDの直接パス指定は使わず、カタログへ登録してください。

`three_d.gesture_policy` で最低間隔、同じ意味・分類の待ち時間、大きな動作の1分間上限を調整できます。見本の `max_large_per_minute: 2` は上限2回、`0` は上限なしです。項目と既定値は [アバター設定見本](../config/avatar.reference.example.yaml) を参照してください。

通常画面ではVMD内のカメラ・照明を会話表示へ適用しません。管理プレビューは待機を含め一回再生です。途中停止と通常待機への復帰は **プレビュー停止 → 閉じる** です。

## エラーになった場合

| 表示 | 対処 |
| --- | --- |
| 利用条件の確認が完了していません | 権利記録を見直す。条件不明なら `usage_confirmed: false` のまま使わない |
| 必要なボーンがモデルにありません | PMXとの組合せと必須ボーン名を確認する |
| 対象にしていません | `model_names` とPMX内部名を一致させる |
| 登録されていません | アバター設定とカタログの識別名を一致させる |
| 同じVMDを待機動作と会話ジェスチャーへ同時指定できません | どちらか一方から識別名を外す |
| localフォルダー内へ置いてください | カタログ・素材・権利記録を `local` 内に置く |

個別のVMDが欠落・非互換ならその項目を使わず待機を続けます。カタログ自体のYAMLが不正なら3D連携エラーになります。変更後に起動できなければ終了し、控えた設定から編集箇所を戻してください。
