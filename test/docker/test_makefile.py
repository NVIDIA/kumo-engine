# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``docker/Makefile`` must not expand a GitHub credential into its recipes.

``make -n`` prints the recipe text verbatim, which is the argv these builds would expose.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / 'docker' / 'Makefile'
FAKE_TOKEN = 'ghp_FAKEdoNotUse0000000000000000000000000'


def _expand(target: str, tmp_path: Path, *args: str) -> str:
    env = dict(os.environ, GITHUB_TOKEN=FAKE_TOKEN,
               GITHUB_NPM_ACCESS_TOKEN=FAKE_TOKEN, TMPDIR=str(tmp_path))
    recipe = subprocess.run(
        ['make', '-n', '-f', str(MAKEFILE), target, *args], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, check=True).stdout
    # build-ui-local's $(shell mktemp -d) runs even under -n, and BSD mktemp
    # ignores TMPDIR, so remove the context by the name make chose.
    for context in re.findall(r'^cp -r docker/\. (\S+)/$', recipe, re.M):
        shutil.rmtree(context, ignore_errors=True)
    return recipe


def test_backend_build_uses_no_github_credentials(tmp_path):
    for variant in ('cpu', 'gpu'):
        recipe = _expand('build-backend-local', tmp_path, f'KUMO_BACKEND_VARIANT={variant}')

        assert FAKE_TOKEN not in recipe
        assert 'GITHUB_TOKEN' not in recipe
        assert 'gh auth' not in recipe
        assert '--secret' not in recipe
        assert f'docker/Dockerfile.kumo_backend_{variant}' in recipe


def test_ui_build_uses_in_tree_catalog_without_github_credentials(tmp_path):
    recipe = _expand('build-ui-local', tmp_path)

    assert FAKE_TOKEN not in recipe
    assert 'GITHUB_TOKEN' not in recipe
    assert 'GITHUB_NPM_ACCESS_TOKEN' not in recipe
    assert 'gh auth' not in recipe
    assert '--secret' not in recipe
    assert 'packages/kumo-ui-catalog/' in recipe
    assert '/kumo/packages/kumo-ui-catalog/' in recipe
