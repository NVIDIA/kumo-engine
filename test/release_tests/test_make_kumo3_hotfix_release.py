# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The kumo3 hotfix script has to actually cut a hotfix.

Everywhere else this script appears it is either refusing something
(test_release_v3_guards.py) or stubbed out entirely
(test_check_updated_release_branches.py), so nothing runs it to success --
it could be broken end to end with the suite still green. This drives the
real script through a real cut against a synthetic origin.
"""
from test.release_tests.conftest import (
    GH_STUB,
    REPO_ROOT,
    SandboxRepo,
    git,
    init_sandbox_repo,
    release_branch_with_backport,
    stub,
)

import pytest

SCRIPT_REL = "scripts/release/make_kumo3_hotfix_release.sh"

# Two different things reach `python3`: preflight's import check, which only
# has to succeed, and the image builds, which are the point of the test and so
# record their argv.
PYTHON_STUB = """#!/bin/bash
if [[ "$1" == "-c" ]]; then
  exit 0
fi
echo "python3 $*"
"""


class Kumo3HotfixRepo(SandboxRepo):
    def run(self, *args):
        return self._run_script(SCRIPT_REL, args)


@pytest.fixture
def hotfix_repo(tmp_path):
    origin, root = init_sandbox_repo(tmp_path)

    (root / "scripts" / "release").mkdir(parents=True)
    (root / SCRIPT_REL).write_text((REPO_ROOT / SCRIPT_REL).read_text())

    (root / "README.md").write_text("seed\n")
    git(root, "add", "-A")
    git(root, "commit", "-m", "seed")
    git(root, "push", "origin", "master")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub(bin_dir / "gh", GH_STUB)
    stub(bin_dir / "python3", PYTHON_STUB)

    return Kumo3HotfixRepo(root=root, origin=origin, bin=bin_dir)


def test_cuts_a_kumo3_hotfix(hotfix_repo):
    # CI=true and no --dry-run: the mode the sweep runs unattended, and the
    # only one that genuinely mutates. `git` is real, so the tag assertion
    # below is about a ref that truly reached the sandbox remote.
    release_branch_with_backport(hotfix_repo, "v3.0", "v3.0.0")
    proc = hotfix_repo.run("true")

    assert proc.returncode == 0, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert "refs/tags/v3.0.1" in hotfix_repo.origin_refs()
    assert "gh release create v3.0.1 --verify-tag --title v3.0.1" in proc.stdout
    # Both builds, for the new patch tag and for the floating minor tag that
    # keeps 'v3.0' pointing at the newest v3.0.x images.
    for tag in ("v3.0.1", "v3.0"):
        assert (f"python3 scripts/release/build_kumo_images.py {tag} "
                "--app kumo3 --target saas --publish_to_acr") in proc.stdout
    # check_updated_release_branches.sh greps for exactly this line and awks
    # the fourth field out of it; without it the hotfix is never recorded.
    assert "New version is v3.0.1" in proc.stdout
