# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The release-branch sweep must release Kumo 3 and route Kumo 2 elsewhere.

The sweep examines every `origin/vX.Y` branch. Kumo 3 branches must reach
make_kumo3_hotfix_release.sh, while v2.21 must be skipped with a pointer to the
private maintenance repository. One failing Kumo 3 branch must not stop later
branches from being released.

The hotfix script is stubbed. What is under test is the dispatch, not the
release -- the script's own guard is covered in test_release_v3_guards.py.
"""
from test.release_tests.conftest import (
    REPO_ROOT,
    SandboxRepo,
    git,
    init_sandbox_repo,
    stub,
)

import pytest

SWEEP_SCRIPT_REL = "scripts/release/check_updated_release_branches.sh"
KUMO3_HOTFIX_SCRIPT_REL = "scripts/release/make_kumo3_hotfix_release.sh"
V2_MAINTENANCE_REPO = "https://github.com/kumo-ai/kumo-v2-maintenance"


def hotfix_stub(failing_branch=None):
    """A stand-in for the Kumo 3 hotfix script.

    It announces itself on stderr because the sweep captures each script's
    stdout into a variable, which makes stdout unobservable from outside.
    """
    failure = ""
    if failing_branch:
        failure = f"""if [[ "$branch" == "{failing_branch}" ]]; then
  exit 1
fi
"""
    return f"""#!/bin/bash
branch=$(git rev-parse --abbrev-ref HEAD)
echo "RAN kumo3 hotfix on $branch" >&2
{failure}echo "New version is $branch.9"
"""


class SweepRepo(SandboxRepo):
    def run(self, *args):
        return self._run_script(SWEEP_SCRIPT_REL, args)

    def hotfix_versions(self):
        # The sweep appends to ../hotfix_versions.txt, relative to the repo it
        # is run from.
        path = self.root.parent / "hotfix_versions.txt"
        return path.read_text() if path.exists() else ""


@pytest.fixture
def sweep_repo(tmp_path):
    origin, root = init_sandbox_repo(tmp_path)

    (root / "scripts" / "release").mkdir(parents=True)
    (root / SWEEP_SCRIPT_REL).write_text(
        (REPO_ROOT / SWEEP_SCRIPT_REL).read_text())
    stub(root / KUMO3_HOTFIX_SCRIPT_REL, hotfix_stub())

    git(root, "add", "-A")
    git(root, "commit", "-m", "seed")
    git(root, "push", "origin", "master")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    return SweepRepo(root=root, origin=origin, bin=bin_dir)


def _install_hotfix_stub(repo, script_rel, body):
    """Swap in a different hotfix stub, on master, before any branch is cut.

    It has to be committed: `_release_branch` commits whatever is uncommitted
    onto the branch it creates, which would put the stub on the release branch
    and leave master with the original -- and the sweep copies the scripts out
    of master's working tree before it starts.
    """
    stub(repo.root / script_rel, body)
    git(repo.root, "commit", "-am", f"stub {script_rel}")


def _release_branch(repo, branch):
    """A release branch on origin whose tip is untagged -- what the sweep acts on."""
    git(repo.root, "checkout", "-b", branch, "master")
    (repo.root / f"{branch}.txt").write_text("backport\n")
    git(repo.root, "add", "-A")
    git(repo.root, "commit", "-m", f"backport onto {branch}")
    git(repo.root, "push", "origin", branch)
    git(repo.root, "checkout", "master")
    # The sweep runs `git checkout --track origin/<branch>`, which fails if the
    # local branch already exists, so leave only the remote-tracking ref.
    git(repo.root, "branch", "-D", branch)


def test_sweep_routes_v2_to_maintenance_repo_and_releases_v3(sweep_repo):
    _release_branch(sweep_repo, "v2.21")
    _release_branch(sweep_repo, "v3.0")

    proc = sweep_repo.run()

    assert "RAN kumo3 hotfix on v3.0" in proc.stderr
    assert "RAN kumo3 hotfix on v2.21" not in proc.stderr
    assert "Skipping origin/v2.21" in proc.stdout
    assert V2_MAINTENANCE_REPO in proc.stdout
    assert "v2.21.9" not in sweep_repo.hotfix_versions()
    assert "v3.0.9" in sweep_repo.hotfix_versions()


def test_sweep_ignores_kumo_api_release_branches(sweep_repo):
    _release_branch(sweep_repo, "kumo-api-v3.0")
    _release_branch(sweep_repo, "v3.0")

    proc = sweep_repo.run()

    assert "RAN kumo3 hotfix on v3.0" in proc.stderr
    assert "kumo-api-v3.0" not in proc.stderr
    assert "kumo-api-v3.0" not in sweep_repo.hotfix_versions()


def test_a_failed_branch_does_not_end_the_sweep(sweep_repo):
    # 'v10.0' sorts before 'v2.21' in `git branch -r`, so the branch whose
    # hotfix script fails is swept first, and the one after it is only reached
    # if the failure was contained. It is also a v3+ branch whose major only
    # reads as newer than 3 when compared numerically.
    _install_hotfix_stub(sweep_repo, KUMO3_HOTFIX_SCRIPT_REL,
                         hotfix_stub(failing_branch="v10.0"))
    _release_branch(sweep_repo, "v10.0")
    _release_branch(sweep_repo, "v3.0")

    proc = sweep_repo.run()

    assert (proc.stderr.index("RAN kumo3 hotfix on v10.0")
            < proc.stderr.index("RAN kumo3 hotfix on v3.0"))
    # The failing branch produced no version, the one after it did.
    assert "v10.0" not in sweep_repo.hotfix_versions()
    assert "v3.0.9" in sweep_repo.hotfix_versions()
