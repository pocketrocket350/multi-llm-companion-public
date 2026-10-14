# multi-llm-companion

Windowsで、文字・音声とMMDアバターを使って会話するPythonアプリです。VoiSona Talk「雨衣」と雨衣MMDを基準にしています。

**最初はAPIキー不要のMockで起動できます。** Mockは動作確認用の固定応答です。AIとの会話には外部LLMの設定が必要です。

## はじめる

Windows 11、Python 3.12、初回セットアップ用のインターネット接続を用意します。

1. ソースZIPを新しいフォルダーへ展開します。
2. `setup_windows.bat` を開き、「セットアップが正常に終わりました」を待ちます。
3. `run.bat` を開き、入力欄に文章を書いてEnterで送信します。

詳しくは [はじめての起動](docs/beginner_guide_ja.md)。音声とMMDを使う場合は [雨衣の音声・MMDを設定する](docs/reference_setup_ja.md) へ進んでください。

## できること

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

自作コード・文書には [MIT License](LICENSE)、同梱の第三者素材には [各提供元の条件](THIRD_PARTY_NOTICES.md) が適用されます。開発への参加は [CONTRIBUTING.md](CONTRIBUTING.md)、脆弱性の報告は [SECURITY.md](SECURITY.md) を参照してください。
