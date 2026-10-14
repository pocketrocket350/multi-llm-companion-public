# 配布ファイルと導入

ソース版の候補は **v1.0-rc.3** です。現在は非公開で、公開ダウンロード先はありません。提供された候補ZIPを使ってください。Windows 11・Python 3.12が必要で、初回セットアップ時に依存パッケージを取得します。

## ファイルの内容

| ファイル | 内容 |
| --- | --- |
| `multi-llm-companion-v1.0-rc.3-source.zip` | アプリのソース・設定見本・ガイド・同梱素材 |
| `SHA256SUMS.txt` | 配布ファイルのハッシュ |
| `FILE_MANIFEST.json` | ZIP内のファイル一覧とハッシュ |
| `VERIFICATION.json` | 検証した範囲と結果 |
| `AUDIT.json` | 混入・構文・リンク検査の結果 |
| `REVIEW_NOTES_ja.md` | 候補の変更点 |

導入は [初心者ガイド](beginner_guide_ja.md) の順に進めます。更新時も新しいフォルダーへ展開し、個人データは [バックアップと復元](backup_and_recovery_ja.md) で移します。

## ZIPの破損・取り違えを確認する

1. ZIPのあるフォルダーを右クリックし、**ターミナルで開く** を選びます。
2. PowerShellで次を実行します。

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '.\multi-llm-companion-v1.0-rc.3-source.zip'
```

3. 表示された `Hash` と、`SHA256SUMS.txt` の同じファイル名の値を比べます。大文字・小文字を除いて一致すれば確認完了です。不一致なら実行せず、入手元へ確認してください。

対応範囲は [既知の制限](v1_candidate_ja.md)、第三者素材は [権利とクレジット](rights_and_credits_ja.md) にまとめています。
