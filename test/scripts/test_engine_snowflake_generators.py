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


def test_client_lock_generator_preserves_spdx_on_every_run(
        monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    generator = importlib.import_module('compile_client')
    source = tmp_path / 'repo/packages/kumo-engine/test/snowflake_notebook'
    source.mkdir(parents=True)
    monkeypatch.setattr(generator, '__file__',
                        str(source / 'compile_client.py'))
    (source / 'client-requirements.in').write_text('example==1\n')
    calls = []

    def compile_(command, **kwargs):
        calls.append(command)
        Path(command[command.index('--output-file') +
                     1]).write_text('example==1 --hash=sha256:abc\n')

    monkeypatch.setattr(generator.subprocess, 'run', compile_)
    generator.compile_lock('synthetic-uv')
    first = (source / 'client-requirements.txt').read_text()
    generator.compile_lock('synthetic-uv')
    assert (source / 'client-requirements.txt').read_text() == first
    assert first.startswith(generator.HEADER)
    assert first.count('SPDX-License-Identifier') == 1
    assert all('--no-header' in command for command in calls)


def test_resolver_uses_candidate_extra_and_excludes_it_from_archive(
        monkeypatch, tmp_path):
    import json
    monkeypatch.syspath_prepend(str(DIRECTORY))
    generator = importlib.import_module('resolve_dependencies')
    candidate = tmp_path / 'kumo_engine-1.0-py3-none-any.whl'
    candidate.write_bytes(b'candidate')
    name = 'example-1.0-py3-none-any.whl'
    plan = dict(local_wheels=[name], source_index='https://pypi.org/simple',
                target=dict(python='3.12', system='Linux',
                            architecture='x86_64', maximum_manylinux='2.35'))
    monkeypatch.setattr(generator, 'runtime_target', lambda: dict(
        python='3.12', system='Linux', architecture='x86_64'))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if 'wheel' in command:
            directory = Path(command[command.index('--wheel-dir') + 1])
            with zipfile.ZipFile(directory / name, 'w') as wheel:
                wheel.writestr(zipfile.ZipInfo('example-1.0.dist-info/METADATA'),
                               'Name: example\nVersion: 1.0\n')
        else:
            directory = Path(command[command.index('--dest') + 1])
            (directory / candidate.name).write_bytes(candidate.read_bytes())

    monkeypatch.setattr(generator.subprocess, 'run', run)
    first = generator.resolve(candidate, tmp_path / 'first', plan)
    second = generator.resolve(candidate, tmp_path / 'second', plan)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as archive:
        assert archive.getnames() == [name]
    assert calls[0][-1] == 'example==1.0'
    download = calls[1]
    assert download[-1] == str(candidate) + '[snowflake]'
    assert '--only-binary=:all:' in download
    assert download[download.index('--python-version') + 1] == '3.12'
    assert 'manylinux_2_35_x86_64' in download
    assert 'manylinux_2_34_x86_64' in download
    assert json.loads((first.parent / 'dependencies.json').read_text())[
        'wheels'][name]['name'] == 'example'
    offline = importlib.import_module('offline_support')
    manifest = json.loads((first.parent / 'dependencies.json').read_text())
    (first.parent / 'wheelhouse-manifest.json').write_text(json.dumps(manifest))
    unpacked = tmp_path / 'unpacked'
    offline.unpack(first.parent, unpacked)
    assert (unpacked / name).is_file()


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
