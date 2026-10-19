# 公開履歴のメール保護

公開用のリポジトリは、コミットの作者・作成者、注釈付きタグの作成者、
コミット・タグ本文のメールにGitHubのnoreply表記だけを使う。
GitHubが作るコミット用の `noreply@github.com` も許可する。
アプリの設定・認証情報・会話DBを変更する手順ではない。

## GitHubアカウントの設定

GitHubへログインし、右上のプロフィール画像 → Settings → Emails を開く。
次の両方をオンにする。メールそのものをチャットや報告へ貼り付けない。

- **Keep my email addresses private**：Web上のコミットやPRマージでnoreplyを使う。
- **Block command line pushes that expose my email**：アカウントの非公開メールを含むpushを拒否する。

GitHub側のpush検査は直近の作者情報が中心で、全履歴の代わりにはならない。
公式手順：[メール非公開とpushの拒否](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/blocking-command-line-pushes-that-expose-your-personal-email-address)。

## 公開用cloneごとの準備

このフォルダーの通常環境をそのままpushせず、許可リストで作った公開用cloneで行う。
以下は公開用cloneをカレントフォルダーにしたPowerShellの操作。
別のGitフックを使っている場合は上書きせず、既存の `pre-push` から検査スクリプトを呼ぶ。

```powershell
git config user.name pocketrocket350
git config user.email '217594434+pocketrocket350@users.noreply.github.com'
git config user.useConfigOnly true
git config core.hooksPath .githooks
```

Python 3.12とGitを利用できることが前提。アプリのセットアップ済み環境では
`.venv\Scripts\python.exe` を使い、それ以外では `python` を使う。
clone後にフックは自動有効化されないため、この準備を各cloneで行う。
`--no-verify` で迂回しない。浅いcloneは全履歴を確認できないため拒否する。

## コミット・マージ・タグ・リリース

プロセスの環境変数はGitの設定より優先されるため、公開作業のPowerShellで
作者と作成者の両方を明示する。ユーザー全体の環境変数や他プロジェクトは変更しない。

```powershell
$env:GIT_AUTHOR_NAME = 'pocketrocket350'
$env:GIT_COMMITTER_NAME = 'pocketrocket350'
$env:GIT_AUTHOR_EMAIL = '217594434+pocketrocket350@users.noreply.github.com'
$env:GIT_COMMITTER_EMAIL = '217594434+pocketrocket350@users.noreply.github.com'
```

このプロジェクトの自動公開では、GitHub APIのPRマージを使用しない。
検査済みのローカルコミットを通常のfast-forward pushで反映する。
マージコミットが必要ならローカルで作り、全親コミットを含めて検査してからpushする。
アカウント設定が未確認のままGitHubのWeb/APIに作者情報を決めさせない。

```powershell
python scripts/check_public_history.py --repo .
```

成功時は件数とnoreply確認のみ表示する。失敗時はメールの値を表示せず公開を止める。
`pre-push` も送信対象の全祖先とタグを検査する。Releaseを作る前にも、最新の公開用
cloneで同じ検査を実行し、対象タグ、main、配布ZIPの内容・ハッシュを照合する。
この検査はGitメタデータ専用で、ファイル内容の秘密情報検査・許可リスト検査も引き続き必要。
CIでpush後に検出するだけでは、すでに外部へ送信されるので防止にならない。

## 2026-10-20の修正と利用者への影響

v1.1.0のマージコミット1件を修正し、mainとv1.1.0タグを同じ内容の
`bca70386538bd2842a2d6e06c81507de71476e36` へ移した。
コミットIDは変更され、元のコミットの署名は引き継げない。
アプリのファイル、以前のタグ、リリースURL、既存添付6ファイルは保持する。
その後の開発用保護スクリプト追加はmainだけに反映し、v1.1.0の配布内容は固定する。

配布ZIPの利用者は再インストール・設定変更不要。
Gitの利用者は旧フォルダーを保存したまま、別の空フォルダーに新しくcloneする。
必要な未公開の作業だけを新しい履歴へ移し、旧履歴をmerge/pushしない。
GitHub自動生成のSource codeアーカイブのハッシュは変わり得るため、
添付の `multi-llm-companion-v1.1.0-source.zip` と `SHA256SUMS.txt` を使う。
作業が分からない場合は旧フォルダーを消さず、pushを止める。アプリの再受入試験は不要。

履歴を修正しても、古いSHA・PR内の参照、既存clone・外部キャッシュは消えない場合がある。
GitHub上の古い参照の削除はGitHub Supportへの依頼と判断が必要。
既存clone等の回収や完全消去は保証できない。
公式説明：[履歴修正と残存参照](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)。
