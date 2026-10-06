# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared scaffolding for the release-script tests.

Every test in this directory runs a real release shell script, so they all
need the same things: a synthetic git repo to mutate, a stub dir prepended to
PATH, and a `gh` that looks present, modern and authenticated. The per-script
specifics (which stubs, which tag landscape) stay in the test modules.
"""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# `gh` as the scripts expect to find it: new enough for the version gate,
# authenticated, and echoing its argv so calls can be asserted on. Scripts that
# never run `gh auth` are unaffected by that case.
#
# Every subcommand the release scripts may reach is declared. The catch-all
# fails rather than returning 0, so a script that grows a new `gh` call can't
# quietly pass on a stub nobody taught it about.
GH_STUB = """#!/bin/bash
case "$1 $2" in
  --version*) echo "gh version 2.45.0 (2025-07-18)" ;;
  "auth "*) exit 0 ;;
  # Nothing is ever published in the sandbox, so the "has this tag already
  # been released?" probe has to answer no.
  "release view") exit 1 ;;
  "repo view") echo "kumo-ai/kumo" ;;
  # `releases/generate-notes`. A marker line so a test can tell a real
  # changelog from a substitute, plus GH_STUB_NOTES_SIZE bytes of filler so
  # the body-size gate can be driven without a genuine 777-commit range.
  "api "*) echo "* stub changelog entry"; yes x | head -c "${GH_STUB_NOTES_SIZE:-0}" ;;
  # Echo the notes body, not just the temp path, so which body was published
  # is observable.
  "release create")
    echo "gh $*"
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == "--notes-file" ]]; then
        echo "--- release body ---"
        cat "$2"
      fi
      shift
    done
    ;;
  *) echo "unexpected gh invocation: gh $*" >&2; exit 1 ;;
esac
"""


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


def stub(path, body):
    """Drop an executable stub script at `path`."""
    path.write_text(body)
    path.chmod(0o755)


def init_sandbox_repo(tmp_path):
    """A bare origin plus a working clone of it, both under tmp_path."""
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "--bare", "-b", "master", str(origin))

    root = tmp_path / "work"
    git(tmp_path, "clone", str(origin), str(root))
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    return origin, root


def release_branch_with_backport(repo, branch, tag):
    """A release branch carrying one unreleased backport -- what the sweep sees."""
    git(repo.root, "checkout", "-b", branch)
    git(repo.root, "push", "origin", branch)
    git(repo.root, "tag", tag)
    git(repo.root, "push", "origin", tag)
    (repo.root / "backport.txt").write_text("fix\n")
    git(repo.root, "add", "-A")
    git(repo.root, "commit", "-m", "backport a fix")
    git(repo.root, "push", "origin", branch)
    git(repo.root, "fetch", "origin")


@dataclass
class SandboxRepo:
    """A sandbox clone plus the stub dir its scripts run against."""
    root: Path
    origin: Path
    bin: Path

    def _run_script(self, script_rel, args):
        # Prepend the stub dir rather than replacing PATH: only the stubbed
        # tools are shadowed, and everything else the scripts use (git, perl,
        # sort, ...) resolves normally wherever it happens to live. git in
        # particular is real, so a run that mutates anything genuinely creates
        # refs on `origin` and can be asserted on.
        env = dict(os.environ)
        env["PATH"] = f"{self.bin}:{env['PATH']}"
        env["HOME"] = str(self.root.parent)
        return subprocess.run(
            ["bash", script_rel, *args],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
        )

    def origin_refs(self):
        """Every ref that actually made it to the remote."""
        return subprocess.run(["git", "ls-remote", str(self.origin)],
                              capture_output=True, text=True,
                              check=True).stdout
