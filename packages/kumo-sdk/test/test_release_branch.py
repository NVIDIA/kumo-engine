# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import textwrap
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
WORKFLOW = REPO_ROOT / '.github/workflows/kumo-sdk-release.yml'
RELEASE_BRANCH_ASSIGNMENT = ('release_branch="kumo-sdk-v${version_final%.*}"')
BASH = shutil.which('bash')


def _run_configure_release(tmp_path, *, version, ref, switch):
    assert BASH is not None
    workflow = WORKFLOW.read_text(encoding='utf-8')
    section = workflow.split('      - name: Configure release\n', 1)[1]
    section = section.split('\n      - uses:', 1)[0]
    script = textwrap.dedent(section.split('        run: |\n', 1)[1])
    package = tmp_path / 'kumoai_studio'
    package.mkdir()
    (package / '_version.py').write_text(f'__version__ = {version!r}\n')
    output = tmp_path / 'github-output'
    result = subprocess.run(
        [BASH, '-c', f'set -euo pipefail\n{script}'],
        cwd=tmp_path,
        env={
            **os.environ,
            'GITHUB_OUTPUT': str(output),
            'PHYSICAL_REF': ref,
            'SWITCH_TO_FINAL': str(switch).lower(),
        },
        capture_output=True,
        text=True,
    )
    return result, output.read_text() if output.exists() else ''


@pytest.mark.parametrize(('version', 'expected'), [
    ('3.0.0', 'kumo-sdk-v3.0'),
    ('3.12.0', 'kumo-sdk-v3.12'),
])
@pytest.mark.skipif(BASH is None, reason='release workflow requires Bash')
def test_release_branch_assignment(version, expected):
    assert BASH is not None
    result = subprocess.check_output(
        [
            BASH,
            '-c',
            f'version_final=$1; {RELEASE_BRANCH_ASSIGNMENT}; '
            'printf "%s\\n" "$release_branch"',
            'release-branch-test',
            version,
        ],
        text=True,
    )

    assert result.strip() == expected


def test_release_workflow_uses_safe_ref_derivation_and_atomic_push():
    workflow = WORKFLOW.read_text(encoding='utf-8')

    assert '${version_final//.0}' not in workflow
    assert workflow.count(RELEASE_BRANCH_ASSIGNMENT) == 1
    assert 'git checkout -b "$release_branch"' in workflow
    assert workflow.count("printf 'UPDATED_REFS=%s %s") == 4
    assert 'git push --atomic origin "${updated_refs[@]}"' in workflow
    assert 'group: sdk-release-refs' in workflow


def test_all_created_sdk_refs_are_namespaced():
    workflow = WORKFLOW.read_text(encoding='utf-8')

    assert workflow.count('git tag "kumo-sdk-v$') == 4
    assert 'git tag "v$' not in workflow
    assert RELEASE_BRANCH_ASSIGNMENT in workflow


@pytest.mark.parametrize(('version', 'ref', 'switch', 'expected'), [
    ('3.0.0', 'master', False,
     'create-prerelease-from-main=true\nprerelease-kind=b'),
    ('3.1.0.dev0', 'master', False,
     'create-prerelease-from-main=true\nprerelease-kind=rc'),
    ('3.0.0b1', 'kumo-sdk-v3.0', False,
     'create-prerelease-from-prerelease=true\nprerelease-kind=b'),
    ('3.0.0rc1', 'kumo-sdk-v3.0', False,
     'create-prerelease-from-prerelease=true\nprerelease-kind=rc'),
    ('3.0.0b1', 'kumo-sdk-v3.0', True, 'create-final-from-prerelease=true'),
    ('3.0.0rc1', 'kumo-sdk-v3.0', True, 'create-final-from-prerelease=true'),
    ('3.0.0', 'kumo-sdk-v3.0', False, 'create-final-from-final=true'),
    ('3.0.7', 'kumo-sdk-v3.0', False, 'create-final-from-final=true'),
])
@pytest.mark.skipif(BASH is None, reason='release workflow requires Bash')
def test_configure_release_accepts_supported_state(tmp_path, version, ref,
                                                   switch, expected):
    result, output = _run_configure_release(tmp_path, version=version, ref=ref,
                                            switch=switch)

    assert result.returncode == 0, result.stderr
    assert output.strip() == expected


@pytest.mark.parametrize(('version', 'ref', 'switch', 'message'), [
    ('3.0.0b1', 'kumo-sdk-v3.1', False, 'does not belong to branch'),
    ('3.0.0', 'kumo-sdk-v3.0', True, 'only a beta or RC'),
    ('3.0.0.dev0', 'kumo-sdk-v3.0', True, 'unsupported version'),
    ('3.1.0', 'master', False, 'master must contain'),
])
@pytest.mark.skipif(BASH is None, reason='release workflow requires Bash')
def test_configure_release_rejects_unsafe_state(tmp_path, version, ref, switch,
                                                message):
    result, output = _run_configure_release(tmp_path, version=version, ref=ref,
                                            switch=switch)

    assert result.returncode != 0
    assert output == ''
    assert message in result.stdout


def test_tag_workflow_accepts_beta_as_prerelease():
    workflow = (WORKFLOW.parent /
                'kumo-sdk-release-on-tag.yml').read_text(encoding='utf-8')

    assert "'kumo-sdk-v*.*.*'" in workflow
    assert ('^kumo-sdk-v[0-9]+\\.[0-9]+\\.[0-9]+'
            '(b[0-9]+|rc[0-9]+)?$') in workflow
    assert "contains(github.ref_name, 'b')" in workflow
    assert 'GH_REPO: ${{ github.repository }}' in workflow


def test_sdk_release_tags_are_isolated_from_kumo_helm_releases():
    helm_workflow = (WORKFLOW.parent /
                     'kumo-stack-release.yml').read_text(encoding='utf-8')

    assert 'tags: ["v3.*"]' in helm_workflow
    assert not fnmatchcase('kumo-sdk-v3.1.0', 'v3.*')
