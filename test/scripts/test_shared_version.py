# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One root VERSION for kumo-engine, kumo-sdk, the platform and the charts."""

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


stamp = _load('stamp_version', 'scripts/stamp_version.py')
set_version = _load('set_version', 'packages/kumo-sdk/scripts/set_version.py')

VERSION = (REPO_ROOT / 'VERSION').read_text().strip()


def test_repository_is_stamped_from_version():
    result = subprocess.run([
        sys.executable,
        str(REPO_ROOT / 'scripts/stamp_version.py'), '--check'
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(('path', 'quote'), [
    ('packages/kumo-engine/kumo/engine/_version.py', '"'),
    ('packages/kumo-sdk/kumoai_studio/_version.py', "'"),
])
def test_python_consumers_carry_the_version(path, quote):
    text = (REPO_ROOT / path).read_text()
    assert f'__version__ = {quote}{VERSION}{quote}\n' in text


def test_platform_carries_the_version():
    text = (REPO_ROOT / 'common_utils/version_file.py').read_text()
    assert f'RELEASE_VERSION = "{VERSION}"\n' in text


def test_every_consumer_is_a_stamp_site():
    paths = {site.path for site in stamp.SITES}
    assert {
        'packages/kumo-engine/kumo/engine/_version.py',
        'packages/kumo-sdk/kumoai_studio/_version.py',
        'common_utils/version_file.py',
        'helm/kumo-stack/Chart.yaml',
    } <= paths


def test_sdk_release_workflows_can_still_read_line_one():
    # The release workflows do `exec(input())` on the first line only.
    first = (REPO_ROOT / 'packages/kumo-sdk/kumoai_studio/_version.py'
             ).read_text().split('\n')[0]
    namespace = {}
    exec(first, namespace)
    assert namespace['__version__'] == VERSION


def test_engine_wheel_version_is_read_from_the_stamped_module():
    project = tomllib.loads(
        (REPO_ROOT / 'packages/kumo-engine/pyproject.toml').read_text())
    assert 'version' in project['project']['dynamic']
    assert project['tool']['setuptools']['dynamic']['version'] == {
        'attr': 'kumo.engine._version.__version__'
    }


@pytest.mark.parametrize(('version', 'semver'), [
    ('1.0.0', '1.0.0'),
    ('1.0.0.dev0', '1.0.0-dev.0'),
    ('3.1.0.dev20261006', '3.1.0-dev.20261006'),
    ('3.1.0b1', '3.1.0-b.1'),
    ('3.1.0rc2', '3.1.0-rc.2'),
])
def test_version_grammar(tmp_path, monkeypatch, version, semver):
    monkeypatch.setattr(stamp, 'REPO_ROOT', tmp_path)
    (tmp_path / 'VERSION').write_text(version + '\n')
    assert stamp.read_versions() == (version, semver)


@pytest.mark.parametrize(
    'version', ['1.0', '1.0.0.rc1', '1.0.0rc', '1.0.0+local', 'v1.0.0', ''])
def test_version_grammar_rejects_other_versions(tmp_path, monkeypatch, version):
    monkeypatch.setattr(stamp, 'REPO_ROOT', tmp_path)
    (tmp_path / 'VERSION').write_text(version + '\n')
    with pytest.raises(SystemExit, match='VERSION must be'):
        stamp.read_versions()


def test_stamp_keeps_each_files_quote_style(tmp_path, monkeypatch):
    monkeypatch.setattr(stamp, 'REPO_ROOT', tmp_path)
    (tmp_path / 'VERSION').write_text('2.0.0rc1\n')
    single = tmp_path / 'single.py'
    single.write_text("__version__ = '1.0.0'\n# header\n")
    double = tmp_path / 'double.py'
    double.write_text('__version__ = "1.0.0"\n')
    monkeypatch.setattr(stamp, 'SITES', (
        stamp.Site('single.py', '__version__', stamp.PYTHON_VERSION, False),
        stamp.Site('double.py', '__version__', stamp.PYTHON_VERSION, False),
    ))
    assert len(stamp.stamp(check=True)) == 2
    assert single.read_text() == "__version__ = '1.0.0'\n# header\n"
    stamp.stamp(check=False)
    assert single.read_text() == "__version__ = '2.0.0rc1'\n# header\n"
    assert double.read_text() == '__version__ = "2.0.0rc1"\n'
    assert stamp.stamp(check=True) == []


def test_set_version_accepts_exactly_what_the_stamper_accepts():
    # Load the real stamp script, as the release workflows do.
    pattern = set_version._version_pattern()
    for version in ('1.0.0', '1.0.0.dev0', '3.1.0b1', '3.1.0rc2'):
        assert pattern.fullmatch(version)
    for version in ('1.0.0.dev9+local', '1.0', 'v1.0.0'):
        assert not pattern.fullmatch(version)


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A repo whose stamp script only records that it ran."""
    (tmp_path / 'scripts').mkdir()
    stamp_script = tmp_path / 'scripts/stamp_version.py'
    stamp_script.write_text(
        'import re\n'
        f'VERSION_PATTERN = re.compile({stamp.VERSION_PATTERN.pattern!r})\n'
        'if __name__ == "__main__":\n'
        '    from pathlib import Path\n'
        '    root = Path(__file__).resolve().parent.parent\n'
        '    (root / "stamped").write_text((root / "VERSION").read_text())\n')
    sdk = tmp_path / 'packages/kumo-sdk/kumoai_studio'
    sdk.mkdir(parents=True)
    version_py = sdk / '_version.py'
    version_py.write_text("__version__ = '1.0.0.dev0'\n# header\n")
    (tmp_path / 'VERSION').write_text('1.0.0.dev0\n')
    monkeypatch.setattr(set_version, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(set_version, 'STAMP', stamp_script)
    monkeypatch.setattr(set_version, 'VERSION_FILE', tmp_path / 'VERSION')
    monkeypatch.setattr(set_version, 'SDK_VERSION_FILE', version_py)
    return tmp_path, version_py


@pytest.mark.parametrize('version',
                         ['3.1.0b1', '3.1.0rc1', '3.1.0', '3.2.0.dev0'])
def test_set_version_writes_the_root_version_and_stamps(fake_repo, monkeypatch,
                                                        version):
    root, version_py = fake_repo
    monkeypatch.setattr(sys, 'argv', ['set_version.py', version])
    assert set_version.main() == 0
    assert (root / 'VERSION').read_text() == version + '\n'
    assert (root / 'stamped').read_text() == version + '\n'
    # Stamping is the stamp script's job; set_version leaves the file alone.
    assert version_py.read_text() == "__version__ = '1.0.0.dev0'\n# header\n"


def test_set_version_falls_back_for_a_temporary_build_version(
        fake_repo, monkeypatch):
    root, version_py = fake_repo
    monkeypatch.setattr(sys, 'argv', ['set_version.py', '3.1.0.dev1+abc'])
    assert set_version.main() == 0
    assert (root / 'VERSION').read_text() == '1.0.0.dev0\n'
    assert not (root / 'stamped').exists()
    assert version_py.read_text(
    ) == "__version__ = '3.1.0.dev1+abc'\n# header\n"


def test_set_version_requires_one_argument(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['set_version.py'])
    assert set_version.main() == 2
