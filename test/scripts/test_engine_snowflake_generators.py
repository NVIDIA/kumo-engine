# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/test/snowflake_notebook'


def test_dependency_inventory_is_reproducible_from_wheel_metadata(
        monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    generator = importlib.import_module('dependency_inventory')
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('example-1.0.dist-info/METADATA',
                         'Name: example\nVersion: 1.0\n')
        archive.writestr('example/_vendor/vendored-2.0.dist-info/METADATA',
                         'Name: vendored\nVersion: 2.0\n')
    contents = wheel.getvalue()
    name = 'example-1.0-py3-none-any.whl'
    path = tmp_path / 'wheelhouse.tar'
    with tarfile.open(path, 'w') as archive:
        member = tarfile.TarInfo('wheelhouse/' + name)
        member.size = len(contents)
        archive.addfile(member, io.BytesIO(contents))
    plan = dict(target={'python': '3.12'}, local_wheels=[name])
    result = generator.inventory(path, plan)
    assert result == generator.inventory(path, plan)
    assert result['archive_sha256'] == hashlib.sha256(
        path.read_bytes()).hexdigest()
    assert result['wheels'][name] == dict(
        name='example', version='1.0', size=len(contents),
        sha256=hashlib.sha256(contents).hexdigest())
    with pytest.raises(ValueError, match='misses an authored local wheel'):
        generator.inventory(path, dict(plan, local_wheels=['missing.whl']))


@pytest.mark.parametrize('changed', [dict(python='3.14'), dict(system='Darwin'),
                                     dict(architecture='aarch64')])
def test_resolver_rejects_wrong_marker_environment_before_pip(
        monkeypatch, tmp_path, changed):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    generator = importlib.import_module('resolve_dependencies')
    actual = dict(python='3.12', system='Linux', architecture='x86_64')
    monkeypatch.setattr(generator, 'runtime_target', lambda: dict(actual, **changed))
    monkeypatch.setattr(generator.subprocess, 'run',
                        lambda *args, **kwargs: pytest.fail('Must reject before pip'))
    destination = tmp_path / 'output'
    with pytest.raises(ValueError, match='Python minor version'):
        generator.resolve(tmp_path / 'candidate.whl', destination, {'target': actual})
    assert not destination.exists()


def test_resolver_rejects_non_cpython_before_side_effects(
        monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    generator = importlib.import_module('resolve_dependencies')
    monkeypatch.setattr(generator.platform, 'python_implementation',
                        lambda: 'PyPy')
    monkeypatch.setattr(
        generator, 'runtime_target',
        lambda: dict(python='3.12', system='Linux', architecture='x86_64'))
    monkeypatch.setattr(
        generator.subprocess, 'run',
        lambda *args, **kwargs: pytest.fail('Must reject before pip'))
    destination = tmp_path / 'output'
    with pytest.raises(ValueError, match='CPython'):
        generator.resolve(tmp_path / 'candidate.whl', destination,
                          {'target': {}})
    assert not destination.exists()
