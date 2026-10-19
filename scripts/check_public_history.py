"""公開前にGit履歴のメール表記を検査する。値は出力しない。"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

_NOREPLY = re.compile(rb"(?:[A-Za-z0-9+_.-]+@users\.noreply\.github\.com|noreply@github\.com)")
_EMAIL = re.compile(rb"[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_OID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class HistoryError(Exception):
    """公開を止める理由だけを保持する。"""


def _git(repo: Path, *args: str) -> bytes:
    executable = shutil.which("git")
    if executable is None:
        raise HistoryError("Gitが見つからないため公開を停止しました。")
    result = subprocess.run(
        [executable, "--no-replace-objects", "-C", str(repo), *args],
        capture_output=True,
        timeout=60,
    )
    if result.returncode:
        raise HistoryError("Git履歴を確認できないため公開を停止しました。")
    return result.stdout


def check_metadata(raw: bytes, roles: tuple[bytes, ...]) -> None:
    header = raw.split(b"\n\n", 1)[0]
    for role in roles:
        matches = re.findall(rb"^" + role + rb" [^\n]* <([^<>]*)> [^\n]+$", header, re.M)
        if len(matches) != 1 or not _NOREPLY.fullmatch(matches[0]):
            raise HistoryError("作者・作成者・タグ作成者にnoreply以外のメールがあります。")
    # 本文のCo-authored-by等も検査し、メールの値は診断へ含めない。
    if any(not _NOREPLY.fullmatch(email) for email in _EMAIL.findall(raw)):
        raise HistoryError("コミットまたはタグの本文にnoreply以外のメールがあります。")


def check_history(repo: Path, refs: list[str] | None = None) -> int:
    if _git(repo, "rev-parse", "--is-shallow-repository").strip() != b"false":
        raise HistoryError("浅いcloneでは全履歴を検査できないため公開を停止しました。")
    if refs is None:
        refs = _git(repo, "for-each-ref", "--format=%(refname)", "refs/heads", "refs/tags")
        refs = refs.decode().splitlines()
    commits: set[str] = set()
    for ref in refs:
        oid = _git(repo, "rev-parse", "--verify", "--end-of-options", ref).decode().strip()
        if not _OID.fullmatch(oid):
            raise HistoryError("公開対象の識別子を確認できません。")
        while _git(repo, "cat-file", "-t", oid).strip() == b"tag":
            raw = _git(repo, "cat-file", "tag", oid)
            check_metadata(raw, (b"tagger",))
            oid = raw.splitlines()[0].removeprefix(b"object ").decode()
            if not _OID.fullmatch(oid):
                raise HistoryError("タグの参照先を確認できません。")
        if _git(repo, "cat-file", "-t", oid).strip() != b"commit":
            raise HistoryError("コミット以外を参照するタグの公開を停止しました。")
        commits.update(_git(repo, "rev-list", oid).decode().splitlines())
    for oid in sorted(commits):
        check_metadata(_git(repo, "cat-file", "commit", oid), (b"author", b"committer"))
    return len(commits)


def push_refs(lines: str) -> list[str]:
    refs = []
    for line in lines.splitlines():
        fields = line.split()
        if len(fields) != 4 or not _OID.fullmatch(fields[1]):
            raise HistoryError("push対象を確認できないため公開を停止しました。")
        if set(fields[1]) != {"0"}:
            refs.append(fields[1])
    return refs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--pre-push", action="store_true")
    parser.add_argument("refs", nargs="*")
    args = parser.parse_args(argv)
    try:
        refs = push_refs(sys.stdin.read()) if args.pre_push else args.refs or None
        count = check_history(args.repo, refs)
    except (HistoryError, OSError, subprocess.TimeoutExpired) as error:
        message = str(error) if isinstance(error, HistoryError) else "履歴検査を完了できません。"
        print(message, file=sys.stderr)
        return 1
    print(f"公開前の履歴検査: {count}件、メールはnoreply表記です。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
