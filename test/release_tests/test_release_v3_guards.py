# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kumo 3 release entry points must route pre-v3 work out of this repo."""

import subprocess
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

START_KUMO3_RELEASE = (REPO_ROOT / "scripts" / "release" /
                       "start_kumo3_release.sh")
KUMO3_HOTFIX_SCRIPT_REL = "scripts/release/make_kumo3_hotfix_release.sh"
V2_MAINTENANCE_REPO = "https://github.com/kumo-ai/kumo-v2-maintenance"


@pytest.mark.parametrize("version", ["v1.9", "v2.21"])
def test_kumo3_release_routes_pre_v3_to_private_maintenance(version):
    # The version guard runs before preflight or any git mutation.
    proc = subprocess.run(
        ["bash", str(START_KUMO3_RELEASE), "false", version],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "not released from this repository" in proc.stderr
    assert V2_MAINTENANCE_REPO in proc.stderr


class HotfixRepo(SandboxRepo):
    def run(self, *args):
        return self._run_script(KUMO3_HOTFIX_SCRIPT_REL, args)

    def local_tags(self):
        return subprocess.run(
            ["git", "tag", "-l"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout


@pytest.fixture
def hotfix_repo(tmp_path):
    origin, root = init_sandbox_repo(tmp_path)

    script_dir = root / "scripts" / "release"
    script_dir.mkdir(parents=True)
    (root / KUMO3_HOTFIX_SCRIPT_REL).write_text(
        (REPO_ROOT / KUMO3_HOTFIX_SCRIPT_REL).read_text())

    (root / "README.md").write_text("seed\n")
    git(root, "add", "-A")
    git(root, "commit", "-m", "seed")
    git(root, "push", "origin", "master")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub(bin_dir / "gh", GH_STUB)

    return HotfixRepo(root=root, origin=origin, bin=bin_dir)


@pytest.mark.parametrize(("branch", "prev_tag", "would_be_tag", "args"), [
    ("v1.9", "v1.9.4", "v1.9.5", ()),
    ("v2.21", "v2.21.35", "v2.21.36", ()),
    ("v2.21", "v2.21.35", "v2.21.36", ("v2.21.36", "v2.21.35")),
])
def test_kumo3_hotfix_routes_pre_v3_without_tagging(hotfix_repo, branch,
                                                    prev_tag, would_be_tag,
                                                    args):
    release_branch_with_backport(hotfix_repo, branch, prev_tag)
    proc = hotfix_repo.run("true", *args)

    assert proc.returncode != 0
    assert "not released from" in proc.stderr
    assert V2_MAINTENANCE_REPO in proc.stderr
    assert would_be_tag not in hotfix_repo.local_tags()
    assert would_be_tag not in hotfix_repo.origin_refs()
