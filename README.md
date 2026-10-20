# multi-llm-companion

Windowsで、文字・音声とMMDアバターを使って会話するPythonアプリです。VoiSona Talk「雨衣」と雨衣MMDを基準にしています。

**最初はAPIキー不要のMockで起動できます。** Mockは動作確認用の固定応答です。AIとの会話には外部LLMの設定が必要です。

**正式版 v1.1.1（記憶・音声処理・キャンセルの修正）**：[ダウンロード](https://github.com/pocketrocket350/multi-llm-companion-public/releases/tag/v1.1.1)

既存利用者は [v1.1.1への更新手順](docs/update_ja.md) を先に確認してください。

## はじめる

Windows 11、Python 3.12 x64、初回セットアップ用のインターネット接続を用意します。

このソースのセットアップは `constraints-windows-py312.txt` に記録した依存版を使います。固定範囲と導入手順は [配布ファイルと導入](docs/release_ja.md) を参照してください。

1. ソースZIPを新しいフォルダーへ展開します。
2. `setup_windows.bat` を開き、「セットアップが正常に終わりました」を待ちます。
3. **起動** ショートカットを開き、入力欄に文章を書いてEnterで送信します。

詳しくは [はじめての起動](docs/beginner_guide_ja.md)。音声とMMDを使う場合は [雨衣の音声・MMDを設定する](docs/reference_setup_ja.md) へ進んでください。

## できること

- 設定画面での接続先管理、保護したAPIキー入力、接続テスト、ボイス一覧
- 記憶・通常案内のフローティング通知と、種類別の表示・時間設定
- 会話の保存・再開、人格設定、長期記憶の確認・編集
- マイク入力、VoiSona Talkによる読み上げ、表情・口パク・VMD動作
- 許可した場合のWeb検索・画像共有・能動的な話しかけ

オトカム氏のVMD **26件**を、出典・原文Readme付きで同梱しています。雨衣PMX・テクスチャ、VoiSona Talk本体・ボイス、音声認識モデルは利用者が用意します。

## ガイド

- [操作・設定ガイド一覧](docs/README.md)
- [バックアップと復旧](docs/backup_and_recovery_ja.md)
- [対応環境と既知の制限](docs/v1_candidate_ja.md)
- [権利とクレジット](docs/rights_and_credits_ja.md)

外部LLMには会話や関連する記憶などを送信し、利用料金が発生する場合があります。サービスの料金とデータの扱いを確認して設定してください。

自作コード・文書には [MIT License](LICENSE)、同梱の第三者素材には [各提供元の条件](THIRD_PARTY_NOTICES.md) が適用されます。脆弱性の報告は [SECURITY.md](SECURITY.md) を参照してください。
