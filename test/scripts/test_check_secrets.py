# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SECRETS = REPO_ROOT / "scripts" / "check_secrets.sh"


def _run_check(tmp_path: Path, *arguments:
               str) -> subprocess.CompletedProcess[str]:
    executable = tmp_path / "mise"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{tmp_path}{os.pathsep}{environment['PATH']}"
    return subprocess.run(
        [str(CHECK_SECRETS), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_history_scan_explicitly_selects_all_commits(tmp_path: Path) -> None:
    result = _run_check(tmp_path, "--history")

    assert result.returncode == 0
    arguments = result.stdout.splitlines()
    assert arguments[:4] == ["exec", "--", "gitleaks", "git"]
    assert "--log-opts=--all" in arguments


def test_history_modes_reject_shallow_checkout(tmp_path: Path) -> None:
    git = tmp_path / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'true\\n'\n",
        encoding="utf-8",
    )
    git.chmod(0o755)

    for arguments in [("--history",),
                      ("--history-report", str(tmp_path / "report"))]:
        result = _run_check(tmp_path, *arguments)
        assert result.returncode == 2
        assert "non-shallow Git checkout" in result.stderr
        assert result.stdout == ""


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_diff_scan_selects_only_the_commit_range(tmp_path: Path) -> None:
    head = _git(REPO_ROOT, "rev-parse", "HEAD")
    base = _git(REPO_ROOT, "rev-parse", "HEAD~1")

    result = _run_check(tmp_path, "--diff", base, head)

    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    assert arguments[:4] == ["exec", "--", "gitleaks", "git"]
    assert f"--log-opts={base}..{head}" in arguments


def test_diff_scan_from_a_new_branch_selects_only_head(tmp_path: Path) -> None:
    head = _git(REPO_ROOT, "rev-parse", "HEAD")

    result = _run_check(tmp_path, "--diff", "0" * 40, head)

    assert result.returncode == 0, result.stderr
    assert f"--log-opts={head}^!" in result.stdout.splitlines()


def test_diff_scan_uses_the_selected_gitleaks_binary(tmp_path: Path) -> None:
    binary = tmp_path / "pinned-gitleaks"
    binary.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'pinned %s\\n' \"$1\"\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    head = _git(REPO_ROOT, "rev-parse", "HEAD")
    environment = os.environ.copy()
    environment["GITLEAKS_BIN"] = str(binary)

    result = subprocess.run(
        [str(CHECK_SECRETS), "--diff", head, head],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["pinned git"]


@pytest.mark.parametrize("arguments", [("", "HEAD"), ("HEAD", ""),
                                       ("HEAD", ), ()])
def test_diff_scan_rejects_missing_references(tmp_path: Path,
                                              arguments) -> None:
    result = _run_check(tmp_path, "--diff", *arguments)

    assert result.returncode == 2
    assert result.stdout == ""


def test_diff_scan_rejects_unavailable_commits(tmp_path: Path) -> None:
    result = _run_check(tmp_path, "--diff", "f" * 40, "HEAD")

    assert result.returncode == 2
    assert "not an available commit" in result.stderr
    assert result.stdout == ""
