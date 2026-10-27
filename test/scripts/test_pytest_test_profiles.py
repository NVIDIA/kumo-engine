# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the profile plugin in independent, dependency-light pytest runs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from io_utils import fsutil

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GCP_MODULE = 'test/io_utils/fsutil/test_gcs.py'
_AWS_MODULE = 'test/rest/integ/test_bigquery_diskgraph_e2e.py'


def _write(root: Path, name: str, text: str) -> None:
    target = root / name
    fsutil.makedirs(str(target.parent), exist_ok=True)
    fsutil.write_text(str(target), text)


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(('PYTEST_', 'KUMO_TEST_PROFILE'))
    }
    environment['PYTHONPATH'] = str(REPOSITORY_ROOT)
    environment['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-c', str(root / 'pytest.ini'),
         '--rootdir', str(root), '--confcutdir', str(root),
         '-p', 'tools.pytest_test_profiles', '-q', *args],
        cwd=root, env=environment, capture_output=True, text=True, check=False,
    )


@pytest.fixture
def suite(tmp_path: Path) -> Path:
    _write(tmp_path, 'pytest.ini', '[pytest]\n')
    _write(tmp_path, 'test_local.py', 'def test_local(): pass\n')
    _write(tmp_path, _GCP_MODULE, 'def test_gcs(): pass\n')
    _write(tmp_path, _AWS_MODULE, 'def test_training(): pass\n')
    _write(tmp_path, 'test/rest/test_tables.py', '''
def test_validate_size(): pass
def test_local_table(): pass
''')
    _write(tmp_path, 'test/io_utils/fsutil/test_all.py', '''
import os
import pytest

@pytest.fixture(params=[
    "root_s3",
    pytest.param("root_gcs", marks=pytest.mark.requires_credentials("gcp")),
    "root_ucv", "root_adls",
], ids=["s3", "gcs", "ucv", "adls"])
def root(request):
    profile = os.getenv("KUMO_TEST_PROFILE")
    if profile:
        assert (request.param == "root_gcs") == (profile == "gcp")
    return request.param

def test_contract(root): pass
''')
    return tmp_path


def test_partition_and_manifest(suite: Path) -> None:
    baseline = _run(suite, '--collect-only')
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    baseline_nodes = {
        line for line in baseline.stdout.splitlines() if '::' in line
    }
    selected: set[str] = set()
    for profile in ('general', 'gcp', 'gcp-aws'):
        result = _run(suite, '--test-profile', profile,
                      '--test-profile-manifest')
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(next(
            line.removeprefix('KUMO_TEST_PROFILE_MANIFEST=')
            for line in result.stdout.splitlines()
            if line.startswith('KUMO_TEST_PROFILE_MANIFEST=')))
        assert payload['profile'] == profile
        nodes = set(payload['nodeids'])
        assert len(nodes) == len(payload['nodeids'])
        assert not selected.intersection(nodes)
        selected.update(nodes)
    assert selected == baseline_nodes
    assert 'test/io_utils/fsutil/test_all.py::test_contract[gcs]' in selected


def test_ignored_before_import(suite: Path) -> None:
    _write(suite, _GCP_MODULE, 'raise RuntimeError("provider imported")\n')
    _write(suite, _AWS_MODULE, 'raise RuntimeError("AWS imported")\n')
    result = _run(suite, '--test-profile', 'general')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'provider imported' not in result.stdout


@pytest.mark.parametrize('profile,target', [
    ('general', _GCP_MODULE),
    ('general', _AWS_MODULE),
    ('general', 'test/rest/test_tables.py::test_validate_size'),
    ('gcp', _AWS_MODULE),
    ('gcp-aws', _GCP_MODULE),
])
def test_explicit_incompatible_target(suite: Path, profile: str,
                                      target: str) -> None:
    _write(suite, 'conftest.py', 'raise RuntimeError("conftest imported")\n')
    result = _run(suite, '--test-profile', profile, target)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert 'incompatible' in result.stderr
    assert 'conftest imported' not in result.stderr


def test_unknown_profile(suite: Path) -> None:
    _write(suite, 'conftest.py', 'raise RuntimeError("conftest imported")\n')
    result = _run(suite, '--test-profile', 'typo')
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert 'conftest imported' not in result.stderr


def test_profile_precedes_conftest(suite: Path) -> None:
    _write(suite, 'conftest.py', '''
import os
assert os.environ["KUMO_TEST_PROFILE"] == "gcp"
''')
    result = _run(suite, '--test-profile', 'gcp')
    assert result.returncode == 0, result.stdout + result.stderr


def test_unprofiled_unchanged(suite: Path) -> None:
    result = _run(suite)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '9 passed' in result.stdout


def test_unprofiled_manifest(suite: Path) -> None:
    result = _run(suite, '--collect-only', '--test-profile-manifest')
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"profile": "unprofiled"' in result.stdout


def test_missing_capability_fails(suite: Path) -> None:
    _write(suite, _GCP_MODULE, '''
import pytest
@pytest.mark.requires_credentials("gcp", "aws")
def test_gcs(): pass
''')
    result = _run(suite, '--test-profile', 'gcp')
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert 'undeclared profile capabilities: aws' in result.stderr


@pytest.mark.parametrize('profile', ['gcp', 'gcp-aws'])
@pytest.mark.parametrize('target', ['test_local.py',
                                    'test/unrelated/test_import.py'])
def test_positive_explicit_import_guard(suite: Path, profile: str,
                                        target: str) -> None:
    _write(suite, target, 'raise RuntimeError("unowned module imported")\n')
    result = _run(suite, '--test-profile', profile, target)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert 'incompatible' in result.stderr
    assert 'unowned module imported' not in result.stdout + result.stderr


@pytest.mark.parametrize('profile', ['gcp', 'gcp-aws'])
def test_positive_directory_import_guard(suite: Path, profile: str) -> None:
    _write(suite, 'test/unrelated/test_import.py',
           'raise RuntimeError("unowned module imported")\n')
    _write(suite, 'test/unrelated/conftest.py',
           'raise RuntimeError("unowned conftest imported")\n')
    _write(suite, 'test/io_utils/fsutil/test_unowned.py',
           'raise RuntimeError("neighbor module imported")\n')
    result = _run(suite, '--test-profile', profile, 'test')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('profile,target', [
    ('gcp', 'test/rest/test_tables.py::test_local_table'),
    ('gcp-aws', 'test/rest/test_tables.py'),
    ('general', 'test/io_utils/fsutil/test_all.py::test_contract[gcs]'),
    ('gcp', 'test/io_utils/fsutil/test_all.py::test_contract[s3]'),
    ('gcp', 'test/io_utils/fsutil/test_all.py::test_contract[adls]'),
    ('gcp', 'test/io_utils/fsutil/test_all.py::test_contract[ucv]'),
])
def test_mixed_explicit_import_guard(suite: Path, profile: str,
                                     target: str) -> None:
    _write(suite, target.split('::')[0],
           'raise RuntimeError("incompatible mixed module imported")\n')
    result = _run(suite, '--test-profile', profile, target)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert 'incompatible' in result.stderr
    assert 'mixed module imported' not in result.stdout + result.stderr


def test_unregistered_gcp_fails(suite: Path) -> None:
    _write(suite, 'test_new_provider.py', '''
import pytest
@pytest.mark.requires_credentials("gcp")
def test_new_provider(): pass
''')
    result = _run(suite, '--test-profile', 'general')
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert 'not registered' in result.stderr


def test_external_explicit_import_guard(suite: Path) -> None:
    _write(suite.parent, 'external_test.py',
           'raise RuntimeError("external module imported")\n')
    result = _run(suite, '--test-profile', 'gcp',
                  str(suite.parent / 'external_test.py'))
    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert 'incompatible' in result.stderr
    assert 'external module imported' not in result.stdout + result.stderr
