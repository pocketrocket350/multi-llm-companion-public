# 配布ファイルと導入

最新の修正版は **v1.0.1** です。Windows 11・Python 3.12が必要で、初回セットアップ時に依存パッケージを取得します。

v1.0.1では Windows x64 / Python 3.12 を対象に、`constraints-windows-py312.txt` で直接・間接依存、pip、ビルド用ツールの版を固定しています。`setup_windows.bat` はこの一覧を使い、指定版を取得できない場合はエラーで停止します。Python本体のパッチ版、外部サービス、音声・モデル素材はこの固定の対象外です。旧v1.0.0は履歴として残しており、これらの修正を含みません。

## 入手する

1. [GitHubの配布ページ](https://github.com/pocketrocket350/multi-llm-companion-public/releases)を開き、**v1.0.1** を選びます。
2. **Assets** を開き、下表のZIPと `SHA256SUMS.txt` をダウンロードします。
3. 下の手順でハッシュを確認し、ZIPを右クリックして **すべて展開** を選びます。GitHubが自動作成する `Source code (zip)` は、下表のZIPとはハッシュが異なります。

## ファイルの内容

次の6ファイルを一組として提供します。確認資料5件は、ソースZIPの外に添付します。

| ファイル | 内容 |
| --- | --- |
| `multi-llm-companion-v1.0.1-source.zip` | アプリのソース・設定見本・ガイド・同梱素材 |
| `SHA256SUMS.txt` | 配布ファイルのハッシュ |
| `FILE_MANIFEST.json` | ZIP内のファイル一覧とハッシュ |
| `VERIFICATION.json` | 検証した範囲と結果 |
| `AUDIT.json` | 混入・構文・リンク検査の結果 |
| `RELEASE_NOTES_ja.md` | 修正内容と導入案内 |

導入は [初心者ガイド](beginner_guide_ja.md) の順に進めます。更新時も新しいフォルダーへ展開し、個人データは [バックアップと復元](backup_and_recovery_ja.md) で移します。

## ZIPの破損・取り違えを確認する

1. ZIPのあるフォルダーを右クリックし、**ターミナルで開く** を選びます。
2. PowerShellで次を実行します。

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '.\multi-llm-companion-v1.0.1-source.zip'
```

3. 表示された `Hash` と、`SHA256SUMS.txt` の同じファイル名の値を比べます。大文字・小文字を除いて一致すれば確認完了です。不一致なら実行せず、入手元へ確認してください。

対応範囲は [既知の制限](v1_candidate_ja.md)、第三者素材は [権利とクレジット](rights_and_credits_ja.md) にまとめています。
