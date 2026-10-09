# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import subprocess
from importlib.metadata import PackageNotFoundError

import kumo_core.version as version_module
from kumo_core.version import _is_git_repository, get_kumo_version

import kumo.home


def test_is_git_repository(tmp_path: pathlib.Path):
    root_dir = tmp_path.resolve().as_posix()
    assert not _is_git_repository(root_dir)


def test_get_kumo_version():
    # kumo is a namespace package (no __init__.py, so no __file__); resolve
    # the install dir from a real submodule instead.
    kumo_install_path = kumo.home.__file__
    kumo_install_path = kumo_install_path[:kumo_install_path.rindex('/')]
    expected_version_dev_branch = subprocess.check_output(
        ['git', '-C', kumo_install_path, 'rev-parse', '--abbrev-ref',
         'HEAD']).decode("utf-8").strip("\n")
    expected_version_master = subprocess.check_output(
        ['git', '-C', kumo_install_path, 'rev-parse',
         'HEAD']).decode("utf-8").strip("\n")
    assert (get_kumo_version() == expected_version_dev_branch
            or get_kumo_version() == expected_version_master)


class _EntryPoint:
    def __init__(self, name, load):
        self.name = name
        self.load = load


def _providers(monkeypatch, *entries):
    monkeypatch.setattr(version_module, 'entry_points',
                        lambda group: list(entries))


def test_platform_provider_name_sorts_before_the_engine_provider(monkeypatch):
    # Both distributions can be installed; entry-point names decide who wins.
    _providers(monkeypatch, _EntryPoint('kumo-engine', lambda: lambda: '3.0.0'),
               _EntryPoint('container-file', lambda: lambda: 'image-sha'))
    assert get_kumo_version.__wrapped__() == 'image-sha'


def test_plugin_provider_comes_before_git(monkeypatch):
    _providers(monkeypatch, _EntryPoint('engine', lambda: lambda: '3.0.0'))
    monkeypatch.setattr(version_module, 'get_version_from_git',
                        lambda: 'my-branch')
    assert get_kumo_version.__wrapped__() == '3.0.0'


def test_providers_are_tried_in_name_order_until_one_answers(monkeypatch):
    _providers(
        monkeypatch,
        _EntryPoint('b', lambda: lambda: 'from-b'),
        _EntryPoint('a-declines', lambda: lambda: None),
    )
    assert get_kumo_version.__wrapped__() == 'from-b'


def test_broken_providers_are_skipped(monkeypatch):
    def explode():
        raise RuntimeError('boom')

    def not_loadable():
        raise ImportError('missing')

    _providers(monkeypatch, _EntryPoint('a', lambda: explode),
               _EntryPoint('b', not_loadable),
               _EntryPoint('c', lambda: lambda: '3.0.0'))
    assert get_kumo_version.__wrapped__() == '3.0.0'


def test_git_then_latest_when_no_provider_answers(monkeypatch):
    _providers(monkeypatch)
    monkeypatch.setattr(version_module, 'get_version_from_git',
                        lambda: 'my-branch')
    assert get_kumo_version.__wrapped__() == 'my-branch'
    monkeypatch.setattr(version_module, 'get_version_from_git', lambda: None)
    assert get_kumo_version.__wrapped__() == 'latest'


class _Distribution:
    def __init__(self, version, direct_url=None):
        self.version = version
        self._direct_url = direct_url

    def read_text(self, name):
        assert name == 'direct_url.json'
        return self._direct_url


def _dist(monkeypatch, dist):
    monkeypatch.setattr(version_module, 'distribution', lambda name: dist)


def test_installed_distribution_version(monkeypatch):
    _dist(monkeypatch, _Distribution('1.0.0'))
    assert version_module.installed_distribution_version('x') == '1.0.0'
    _dist(monkeypatch,
          _Distribution('1.0.0', '{"url": "file:///x", "dir_info": {}}'))
    assert version_module.installed_distribution_version('x') == '1.0.0'


def test_installed_distribution_version_defers_for_editable(monkeypatch):
    _dist(
        monkeypatch,
        _Distribution('1.0.0',
                      '{"url": "file:///x", "dir_info": {"editable": true}}'))
    assert version_module.installed_distribution_version('x') is None


def test_installed_distribution_version_without_the_distribution(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(version_module, 'distribution', missing)
    assert version_module.installed_distribution_version('x') is None


def test_git_version_ignores_repository_around_site_packages(
        tmp_path, monkeypatch):
    # A virtualenv inside the user's own git checkout is not a Kumo checkout.
    package = tmp_path / '.venv/lib/python3/site-packages/kumo_core'
    package.mkdir(parents=True)
    subprocess.check_call(['git', 'init', '-q', str(tmp_path)])
    monkeypatch.setattr(version_module, '__file__', str(package / 'version.py'))
    assert version_module._is_git_repository(str(package.parent))
    assert version_module.get_version_from_git() is None


def test_platform_registers_its_version_providers():
    import re
    from pathlib import Path
    setup_py = (Path(__file__).parents[2] / 'setup.py').read_text()
    assert re.search(
        r'"kumo_core\.version_providers": \[\s*'
        r'"container-file = common_utils\.version_file:container_version",\s*'
        r'"release-version = common_utils\.version_file:release_version",',
        setup_py)


def test_container_version_reads_the_version_file(tmp_path, monkeypatch):
    from common_utils import version_file
    monkeypatch.setenv('HOME', str(tmp_path))
    assert version_file.container_version() is None
    (tmp_path / 'kumo_version.txt').write_text('  \n')
    assert version_file.container_version() is None
    (tmp_path / 'kumo_version.txt').write_text('abc123\n')
    assert version_file.container_version() == 'abc123'


def test_release_version_is_the_stamped_version_for_an_installed_platform(
        monkeypatch):
    from common_utils import version_file
    monkeypatch.setattr(version_file, 'installed_distribution_version',
                        lambda name: '1.0.0' if name == 'kumo' else None)
    assert version_file.release_version() == version_file.RELEASE_VERSION
    # Editable or absent: defer, so a checkout reports its git revision.
    monkeypatch.setattr(version_file, 'installed_distribution_version',
                        lambda name: None)
    assert version_file.release_version() is None


def test_provider_names_order_image_id_before_engine_before_release(
        monkeypatch):
    names = ['release-version', 'kumo-engine', 'container-file']
    assert sorted(names) == ['container-file', 'kumo-engine', 'release-version']
