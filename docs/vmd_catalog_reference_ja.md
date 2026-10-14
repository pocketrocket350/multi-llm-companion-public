# VMDの追加登録と設定詳細

自分でVMDを追加・調整する場合の資料です。同梱VMDの通常操作は [使い方](vmd_catalog_ja.md) を参照してください。

編集前にアプリを終了し、対象のYAMLを別名でコピーして控えを残します。アバター設定は `local/app.yaml` の `avatar_config_path` で選択中のファイルを編集します。通常は `local/avatar.yaml`、採用構成の導入後は `local/avatar.reference.yaml` です。以下の `avatar.yaml` は選択中のファイルを指します。

## 目的

VMDはMMDのモーションファイルです。第三者が作ったVMDには、作者が定めた利用条件があります。ファイルを直接指定せず、用途、抽象動作、対象モデル、権利確認記録をカタログへまとめてから読み込みます。

アプリは利用許諾を法律的に判断できません。利用者自身が公式配布元と規約を確認してください。

## フォルダー

修正版候補の`setup_windows.bat`は、同梱の採用26件と権利記録・原文規約を初回だけlocalへ配置します。既存カタログがあれば追加・上書きを行いません。以下は利用者が別のVMDを追加登録するときのフォルダーと見本です。

```text
local/
├─ avatar.yaml
├─ vmd_catalog.yaml
├─ motions/
│  └─ 利用者が用意したVMD
└─ licenses/
   └─ VMD一件ごとの権利確認記録
```

`local`は`.gitignore`で公開対象から除外されています。

別のVMDを追加するときは、エクスプローラーで`local`を開き、`motions`と`licenses`フォルダーを用意します。`local/vmd_catalog.yaml`が存在しない場合だけ、`config/vmd_catalog.example.yaml`をコピーできます。既存の同梱26件や利用者のカタログを見本で上書きしません。`config/vmd_license.example.yaml`は同名ファイルがない場合だけ`local/licenses/example_nod.yaml`へコピーします。この追加登録用見本自体は実在VMDを含まず、初期状態では無効です。

## 登録手順

1. VMDの公式配布元と利用規約を確認します。
2. VMDを`local/motions`へ置きます。
3. `config/vmd_license.example.yaml`を`local/licenses`へコピーし、VMD専用の名前へ変更します。
4. タイトル、作者、配布元、規約、取得日、許可された用途、変更内容、クレジット、再配布可否を記入します。
5. 記録内容を再確認してから`usage_confirmed: true`へ変更します。
6. `local/vmd_catalog.yaml`へVMDを登録します。
7. 待機VMDなら`local/avatar.yaml`の`vmd_motion_id`へ識別名を書きます。ライブラリ全体を使う場合は`motion_library_enabled: true`にします。

APIキーはどのVMD設定ファイルにも書かないでください。

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

`motion_type`は`idle`、`talk`、`reaction`、`dance`のいずれかです。`action`はLLMが返す抽象動作で、VMD識別名やファイル名ではありません。`moods`は適する感情、`energy`は`low`・`medium`・`high`、`weight`は同条件での選ばれやすさです。`audio_required: true`のVMDは音源が利用できない間は候補から外れます。

`category`は同じ種類の連続抑制、`body_mask`は大きな動作の判定と会話再生時のボーン所有範囲、`intensity_scale`は動きの強さ、`cooldown_seconds`はその動作を再利用するまでの時間に使います。`body_mask`で所有しないセンター・下半身・脚などのトラックは、元VMDを変更せず会話再生時だけ除外します。管理画面のプレビューは元VMDの内容確認なので全トラックを再生します。既存の`intent`も読み込めますが、新規登録では`action`を使用してください。

全身VMD（`body_mask: [full_body]`）の姿勢には感情強度を掛けず、資産側の`intensity_scale`だけを使います。部分動作と表情モーフの強度調整は従来どおりです。全身VMDを弱い感情で再生すると待機姿勢へ薄まる問題を修正したもので、動作の選択頻度や停止条件を変える設定ではありません。

## avatar.yamlの選択

```yaml
three_d:
  vmd_catalog_path: local/vmd_catalog.yaml
  # 待機中に再生するVMDは一件だけ指定します。
  vmd_motion_id: ""
  # 有効なtalk・reaction・danceをカタログから自動登録します。
  motion_library_enabled: true
  # 会話内容に応じて選択できるVMDは配列で指定します。
  gesture_motion_ids:
    - nod_small
    - wave_small
  gesture_policy:
    enabled: true
    min_interval_seconds: 3.5
    same_gesture_cooldown_seconds: 20
    same_category_cooldown_seconds: 6
    max_large_per_minute: 2
    recent_motion_count: 5
```

`motion_library_enabled: true`では、カタログへ有効なメタデータを追加するだけで候補を増やせます。`false`では、従来どおり`gesture_motion_ids`に明示した識別名だけを使います。同じ識別名を`vmd_motion_id`と`gesture_motion_ids`へ同時指定することはできません。会話VMDがなくても、`wave`と`nod`は軽量なボーン動作へ安全に戻ります。以前の`vmd_path`による直接指定は使用できません。

カメラ専用VMDの読み取りと描画は内部互換処理として残しています。2026-09-22の設定整理後、通常UIにVMDカメラ・照明の選択肢はありません。会話ジェスチャーへ登録したVMDのカメラ・照明は通常表示を守るため適用しません。透視カメラまたは照明があれば、ボーン・モーフを持たないVMDも有効です。平行投影カメラだけのVMDは未対応です。

`min_interval_seconds`は、異なる動作も含む最低間隔です。`same_gesture_cooldown_seconds`は同じ意味の動作、`same_category_cooldown_seconds`は同じ分類の動作の待機時間です。`max_large_per_minute`は腕や上半身・全身を使う動作の1分間上限で、`0`なら上限を設けません。

選択時はactionを最初に合わせ、該当候補があればmoodとenergyを優先します。原則として直近`recent_motion_count`件を外し、残った候補を`weight`で重み付き選択します。候補が少ない場合だけ直近除外を緩めます。ローカルVMD識別名とファイル名はLLMへ渡しません。

設定した会話VMDはアプリ起動時に読み込み、ファイル、権利記録、PMX互換性を検査します。回答のたびにPMXやVMDを読み直すことはありません。個別VMDが欠落、形式不正、非互換の場合はその一件だけ候補から外し、状態欄へ`利用不可 N件`と表示して通常待機を維持します。カタログ自体のYAML形式が不正な場合は、設定全体の修正が必要なため3D連携エラーにします。

権利確認とPMX互換性検査へ合格しても、雨衣の人格や発話内容に合わない動作は`enabled: false`にします。実在素材の在庫件数と、会話で使用できる利用者受入済み件数は分けて扱い、数を増やすことだけを採用理由にしません。

## エラーになった場合

- 「利用条件の確認が完了していません」: 権利確認記録を読み直し、確認できた場合だけ`usage_confirmed`をtrueにします。
- 「必要なボーンがモデルにありません」: VMDとPMXの組み合わせが違うか、必須ボーン名が誤っています。
- 「対象にしていません」: `model_names`がPMX内部名と一致していません。
- 「登録されていません」: `avatar.yaml`の識別名とカタログの名前を一致させます。
- 「同じVMDを待機動作と会話ジェスチャーへ同時指定できません」: どちらか一方から識別名を外します。
- 「localフォルダー内へ置いてください」: カタログ、VMD、権利記録を`local`内へ移します。

権利条件が不明な場合は、`usage_confirmed`をfalseのままにして使用しないでください。

設定変更後は `run.bat` で起動します。エラーが出た場合は終了し、変更前の控えから編集箇所を戻してください。詳しくは [バックアップと復旧](backup_and_recovery_ja.md) を参照してください。
