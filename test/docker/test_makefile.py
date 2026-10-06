# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``docker/Makefile`` must not resolve a GitHub credential at make-expansion time.

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
BACKEND_TOKEN = '"${GITHUB_TOKEN:-$(gh auth token 2>/dev/null)}"'


def _expand(target: str, tmp_path: Path) -> str:
    env = dict(os.environ, GITHUB_TOKEN=FAKE_TOKEN,
               GITHUB_NPM_ACCESS_TOKEN=FAKE_TOKEN, TMPDIR=str(tmp_path))
    recipe = subprocess.run(
        ['make', '-n', '-f', str(MAKEFILE), target], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, check=True).stdout
    # build-ui-local's $(shell mktemp -d) runs even under -n, and BSD mktemp
    # ignores TMPDIR, so remove the context by the name make chose.
    for context in re.findall(r'^cp -r docker/\. (\S+)/$', recipe, re.M):
        shutil.rmtree(context, ignore_errors=True)
    return recipe


def test_github_token_is_resolved_by_the_recipe_shell_not_by_make(tmp_path):
    target = 'build-backend-local'
    expected = BACKEND_TOKEN
    secret_flag = '--secret id=github_token,env=GITHUB_TOKEN'
    recipe = _expand(target, tmp_path)

    assert FAKE_TOKEN not in recipe, (
        f'{target}: make expanded the token into the recipe text, exposing it '
        'in process argv for the whole build. Resolve it in the shell.')
    assert expected in recipe, (
        f'{target}: the recipe no longer carries the deferred lookup, so the '
        'build loses its credential and the guards below test another string.')
    assert secret_flag in recipe, (
        f'{target}: the BuildKit secret mount is gone; the token would reach '
        'image layers.')


def test_token_lookup_is_empty_only_when_no_credential_is_reachable(tmp_path):
    """Emptiness is tested in the recipe shell because at make level the lookup
    is always non-empty literal text, so an ``ifeq`` would report a token there.
    """
    empty_bin = tmp_path / 'bin'
    empty_bin.mkdir()
    expression = BACKEND_TOKEN
    guard = ['/bin/sh', '-c', f'test -n {expression}']
    unset = {'PATH': str(empty_bin)}
    with_token = dict(unset, GITHUB_TOKEN=FAKE_TOKEN)

    assert subprocess.run(guard, env=unset).returncode != 0, (
        f'{expression} is non-empty with no token set and no gh on PATH, so '
        'the guards would hand an empty credential to the build.')
    assert subprocess.run(guard, env=with_token).returncode == 0, (
        f'{expression} is empty with GITHUB_TOKEN set, so the build would '
        'reject a credential it was given.')


def test_ui_build_uses_in_tree_catalog_without_github_credentials(tmp_path):
    recipe = _expand('build-ui-local', tmp_path)

    assert FAKE_TOKEN not in recipe
    assert 'GITHUB_TOKEN' not in recipe
    assert 'GITHUB_NPM_ACCESS_TOKEN' not in recipe
    assert 'gh auth' not in recipe
    assert '--secret' not in recipe
    assert 'packages/kumo-ui-catalog/' in recipe
    assert '/kumo/packages/kumo-ui-catalog/' in recipe
