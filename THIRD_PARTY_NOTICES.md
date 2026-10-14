# Third-Party Notices

このプロジェクトの [MIT License](LICENSE) は、自作コードと自作文書を対象とします。第三者のキャラクター・画像・音声・モデル・モーション・商標・外部サービス・生成物の条件を置き換えません。

## 同梱VMD

作者：**オトカム (Otokam_0510)**。26件のVMD、4パックの原文Readme、出典、権利記録を同梱します。原版23件と補正3件の一覧・変更内容・クレジットは [同梱VMD](docs/bundled_motions_ja.md)、ファイルごとのハッシュは [一覧](bundled/FILE_MANIFEST.json) に記載しています。VMDには作者の規約が適用されます。

## 物理計算DLL

`src/multi_llm_companion/ui/native/UMTNativePlugin.dll` はCandidum GamesのUMTNativePlugin（MIT）とBullet 2.75（zlib）を使用しています。対象コミット・DLLとアーカイブのハッシュ・改変パッチ・著作権およびライセンス本文は [licenses/README.md](licenses/README.md) に記載しています。

## セットアップで取得するパッケージ

PySide6、httpx、pydantic、PyYAML、python-dotenv、Vosk、faster-whisper、CTranslate2、sounddevice、Panda3D、pytest、Ruffなどを使用します。パッケージ本体はソースZIPへ含めず、各パッケージのライセンスに従って導入します。Panda3DはBSD-3-Clause、同梱するBullet Physicsはzlibです。

Pythonや依存パッケージを含む実行形式で再配布する場合は、実際に含める版に対応する本文・NOTICEと条件を確認してください。Qt/PySide6などの条件も個別に適用されます。

## 利用者が導入するもの

雨衣の公式画像・MMD、VoiSona Talk本体・ボイスライブラリ・生成音声、Whisper/Voskの認識モデルは同梱しません。ソフト、素材、生成物の利用・公開・再配布には各権利者の条件が適用されます。[権利とクレジット](docs/rights_and_credits_ja.md) を参照してください。
