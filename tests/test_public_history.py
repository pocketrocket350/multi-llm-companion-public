"""ダミーGit履歴で公開前のメール検査を確認する。"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.check_public_history import (
    HistoryError,
    check_history,
    check_metadata,
    main,
    push_refs,
)

SAFE = "217594434+pocketrocket350@users.noreply.github.com"
DUMMY = "private@example.invalid"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    executable = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
    monkeypatch.setenv("PATH", str(Path(executable).parent) + os.pathsep + os.environ["PATH"])
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", SAFE)
    git(tmp_path, "init", "--initial-branch=main")
    git(tmp_path, "commit", "--allow-empty", "-m", "initial")
    return tmp_path


def git(repo, *args):
    return (
        subprocess.run(
            [shutil.which("git"), "-C", str(repo), *args],
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )


def test_valid_history_and_annotated_tag(repo):
    git(
        repo, "-c", f"user.email={SAFE}", "-c", "user.name=Test", "tag", "-a", "v1", "-m", "release"
    )
    assert check_history(repo) == 1


@pytest.mark.parametrize("role", ["AUTHOR", "COMMITTER"])
def test_rejects_bad_ancestor_even_with_safe_tip(repo, monkeypatch, role):
    monkeypatch.setenv(f"GIT_{role}_EMAIL", DUMMY)
    git(repo, "commit", "--allow-empty", "-m", "bad ancestor")
    monkeypatch.setenv(f"GIT_{role}_EMAIL", SAFE)
    git(repo, "commit", "--allow-empty", "-m", "safe tip")
    with pytest.raises(HistoryError):
        check_history(repo)


def test_override_by_environment_is_detected_without_logging_email(repo, monkeypatch, capsys):
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", DUMMY)
    git(repo, "-c", f"user.email={SAFE}", "commit", "--allow-empty", "-m", "override")
    assert main(["--repo", str(repo)]) == 1
    assert DUMMY not in capsys.readouterr().err


def test_rejects_tagger_and_coauthor(repo, monkeypatch):
    git(repo, "commit", "--allow-empty", "-m", f"change\n\nCo-authored-by: Test <{DUMMY}>")
    with pytest.raises(HistoryError):
        check_history(repo, ["HEAD"])
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", DUMMY)
    git(repo, "tag", "-a", "bad-tag", "HEAD~1", "-m", "release")
    with pytest.raises(HistoryError):
        check_history(repo, ["bad-tag"])


def test_push_hook_rejects_before_remote_receives_commit(repo, monkeypatch):
    remote = repo / "remote.git"
    git(repo, "init", "--bare", str(remote))
    hook = Path(__file__).resolve().parents[1] / ".githooks"
    git(repo, "config", "core.hooksPath", str(hook))
    git(repo, "push", str(remote), "HEAD:refs/heads/main")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", DUMMY)
    git(repo, "commit", "--allow-empty", "-m", "must not be sent")
    result = subprocess.run(
        [shutil.which("git"), "-C", str(repo), "push", str(remote), "HEAD:refs/heads/main"],
        capture_output=True,
    )
    assert result.returncode != 0
    assert DUMMY.encode() not in result.stdout + result.stderr
    assert git(remote, "rev-parse", "main") == git(repo, "rev-parse", "HEAD~1")


def test_pre_push_parses_delete_and_rejects_malformed_input():
    assert push_refs(f"(delete) {'0' * 40} refs/heads/old {'a' * 40}\n") == []
    with pytest.raises(HistoryError):
        push_refs("incomplete")


def test_missing_identity_fails_closed():
    with pytest.raises(HistoryError):
        check_metadata(b"tree abc\n\nmessage", (b"author", b"committer"))


def test_shallow_clone_fails_closed(repo):
    shallow = repo / "shallow"
    git(repo, "clone", "--depth=1", repo.as_uri(), str(shallow))
    with pytest.raises(HistoryError):
        check_history(shallow)
