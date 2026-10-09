# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/release/start_kumo3_release.sh.

The script pushes branches, tags, and public GitHub releases, so every test
here runs against a synthetic git repo with stubbed `gh` and `python3` on
PATH. Nothing leaves tmp_path, and nothing here runs a real cut: the
properties worth pinning are that a dry run describes the cut without
performing it, that a half-finished cut is diagnosed as one, and that a
preflight failure stops before the first push.
"""
import subprocess
from test.release_tests.conftest import (
    GH_STUB,
    REPO_ROOT,
    SandboxRepo,
    git,
    init_sandbox_repo,
    stub,
)

import pytest

SCRIPT_REL = "scripts/release/start_kumo3_release.sh"

PYTHON_STUB = """#!/bin/bash
exit 0
"""

# The preflight import check fails the way a release worker without the venv
# fails it: `packaging` is not stdlib and is not installed.
PYTHON_STUB_MISSING_DEPS = """#!/bin/bash
echo "ModuleNotFoundError: No module named 'packaging'" >&2
exit 1
"""


class ReleaseRepo(SandboxRepo):
    def run(self, *args, expect_ok=True):
        proc = self._run_script(SCRIPT_REL, args)
        if expect_ok and proc.returncode != 0:
            raise AssertionError(
                f"script failed ({proc.returncode})\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        return proc

    def local_refs(self):
        return subprocess.run([
            "git", "for-each-ref", "--format=%(refname)", "refs/heads/v3.0",
            "refs/tags/v3.0.0"
        ], cwd=self.root, capture_output=True, text=True,
                              check=True).stdout.strip()


@pytest.fixture
def release_repo(tmp_path):
    origin, root = init_sandbox_repo(tmp_path)

    script_dir = root / "scripts" / "release"
    script_dir.mkdir(parents=True)
    (script_dir / "start_kumo3_release.sh").write_text(
        (REPO_ROOT / SCRIPT_REL).read_text())

    (root / "README.md").write_text("seed\n")
    git(root, "add", "-A")
    git(root, "commit", "-m", "seed")
    git(root, "push", "origin", "master")

    # Tag landscape mirroring production: .0 minor releases plus a much more
    # recent hotfix tag that must NOT be chosen as the notes baseline.
    for tag in ("v2.20.0", "v2.21.0", "v2.21.35"):
        git(root, "tag", tag)
    git(root, "push", "origin", "--tags")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub(bin_dir / "gh", GH_STUB)
    stub(bin_dir / "python3", PYTHON_STUB)

    return ReleaseRepo(root=root, origin=origin, bin=bin_dir)


def test_dry_run_emits_the_plan_and_mutates_nothing(release_repo):
    before = release_repo.origin_refs()
    proc = release_repo.run("--dry-run", "false", "v3.0")
    out = proc.stdout

    assert "+ git checkout -b v3.0 origin/master" in out
    assert "+ git push origin v3.0" in out
    assert "+ git tag v3.0.0" in out
    assert "+ git push origin v3.0.0" in out
    assert "+ gh release create v3.0.0" in out
    # Notes come from the previous minor release, not the much newer hotfix tag
    # on a maintenance branch, which would produce a misleading changelog.
    assert "Release notes baseline: v2.21.0" in out
    assert "v2.21.35" not in out
    # The dry run prints copy-pasteable build commands. If it also claims the
    # tag was created, an operator can paste them and build against a tag that
    # does not exist.
    assert "and tag 'v3.0.0' created." not in out
    assert "were NOT created" in out

    assert release_repo.local_refs() == ""
    # The local check above would also pass against a script that does nothing
    # at all; the remote is what a bad release actually damages.
    assert release_repo.origin_refs() == before
    assert "v3.0" not in release_repo.origin_refs()


def test_builds_the_kumo3_lineup_and_offers_no_other_target(release_repo):
    # build_kumo_images.py raises ValueError for any target other than saas
    # under --app kumo3, and the kumo3 lineup is the whole reason this script
    # exists. Assert on what IS emitted rather than on the absence of strings:
    # pin the build lines, then require that no --target appears outside them.
    proc = release_repo.run("--dry-run", "true", "v3.0")
    out = proc.stdout

    assert ("+ python3 scripts/release/build_kumo_images.py v3.0.0 "
            "--app kumo3 --target saas --publish_to_acr") in out
    assert ("+ python3 scripts/release/build_kumo_images.py v3.0 "
            "--app kumo3 --target saas --publish_to_acr") in out

    build_lines = [
        line for line in out.splitlines() if "build_kumo_images.py" in line
    ]
    assert len(build_lines) == 2
    assert out.count("--target") == len(build_lines)


def test_local_tag_does_not_mask_an_unpushed_release(release_repo):
    # The shape of a cut that died between `git tag` and `git push origin
    # <tag>`: the branch is public, the tag is not, and a leftover local tag
    # makes it look shipped. Reading refs/tags instead of asking origin sends
    # the operator to backport against a release that does not exist -- the
    # one situation this diagnosis exists for.
    git(release_repo.root, "push", "origin", "master:refs/heads/v3.0")
    git(release_repo.root, "tag", "v3.0.0")
    assert "refs/tags/v3.0.0" in release_repo.local_refs()
    assert "refs/tags/v3.0.0" not in release_repo.origin_refs()

    proc = release_repo.run("--dry-run", "false", "v3.0", expect_ok=False)
    err = proc.stderr

    assert "tag 'v3.0.0' was never" in err
    assert "pushed to origin." in err
    # The resume steps: push the tag origin never got, then publish.
    assert "git push origin v3.0.0" in err
    assert "gh release create v3.0.0" in err
    # Backport advice is only right once branch, tag and release are all
    # public. Here none of the last two are.
    assert "backport_pr.sh" not in err


COMPARE_URL = "https://github.com/kumo-ai/kumo/compare/v2.21.0...v3.0.0"


def test_notes_that_fit_are_inlined(release_repo):
    # The ordinary cut: the generated changelog is the release body. The
    # fallback must not fire just because it exists.
    out = release_repo.run("true", "v3.0").stdout

    assert "gh release create v3.0.0 --verify-tag --title v3.0.0" in out
    assert "* stub changelog entry" in out
    assert COMPARE_URL not in out


def test_oversized_notes_link_the_compare_view_instead_of_truncating(
        release_repo, monkeypatch):
    # v3.0.0's real changelog spanned 777 commits and generated a ~137k body,
    # which GitHub rejected with "HTTP 422 ... body is too long (maximum is
    # 125000 characters)" -- after the branch and tag were already public.
    monkeypatch.setenv("GH_STUB_NOTES_SIZE", "200000")
    out = release_repo.run("true", "v3.0").stdout

    assert "Linking the compare view instead of inlining." in out
    assert COMPARE_URL in out
    assert "125000-character limit" in out
    # The oversized body must be replaced, not shipped and not clipped: a
    # changelog cut off mid-list reads as a complete one.
    assert "* stub changelog entry" not in out
    assert "xxxx" not in out
    # The release still had to be created. That is the whole point -- the cut
    # died here, leaving a public tag with no release behind it.
    assert "gh release create v3.0.0 --verify-tag --title v3.0.0" in out
    assert "refs/tags/v3.0.0" in release_repo.origin_refs()


def test_preflight_failure_pushes_nothing(release_repo):
    # The venv is a real prerequisite -- build_kumo_images.py imports
    # `packaging`. Discovering that after the branch and tag are public
    # strands a half-cut release, so it has to be checked up front.
    #
    # This is the only test here that runs without --dry-run: CI=true with
    # DRY_RUN=false is the mode that genuinely mutates, so it is the only one
    # where "pushed nothing" means anything. `git` is not stubbed, so a missing
    # guard really would reach the sandbox origin.
    before = release_repo.origin_refs()
    stub(release_repo.bin / "python3", PYTHON_STUB_MISSING_DEPS)
    proc = release_repo.run("true", "v3.0", expect_ok=False)

    assert proc.returncode != 0
    assert "packaging" in proc.stderr
    assert release_repo.local_refs() == ""
    assert release_repo.origin_refs() == before
