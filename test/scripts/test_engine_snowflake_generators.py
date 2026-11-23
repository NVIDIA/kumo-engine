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


@pytest.mark.parametrize(
    'changed',
    [dict(python='3.14'),
     dict(system='Darwin'),
     dict(architecture='aarch64')])
def test_resolver_rejects_wrong_marker_environment_before_pip(
        monkeypatch, tmp_path, changed):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    generator = importlib.import_module('resolve_dependencies')
    actual = dict(python='3.12', system='Linux', architecture='x86_64')
    monkeypatch.setattr(generator, 'runtime_target',
                        lambda: dict(actual, **changed))
    monkeypatch.setattr(
        generator.subprocess, 'run',
        lambda *args, **kwargs: pytest.fail('Must reject before pip'))
    destination = tmp_path / 'output'
    with pytest.raises(ValueError, match='Python minor version'):
        generator.resolve(tmp_path / 'candidate.whl', destination,
                          {'target': actual})
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


def test_resolver_keeps_explicit_native_and_customer_wheels_in_offline_archive(
        monkeypatch, tmp_path):
    import json
    import shutil
    monkeypatch.syspath_prepend(str(DIRECTORY))
    generator = importlib.import_module('resolve_dependencies')
    actual = dict(python='3.12', system='Linux', architecture='x86_64')
    monkeypatch.setattr(generator, 'runtime_target', lambda: actual)

    def wheel(name, package, files):
        path = tmp_path / name
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(package + '.dist-info/METADATA',
                             'Name: ' + package + '\nVersion: 0.2.0\n')
            for filename, content in files.items():
                archive.writestr(filename, content)
        return path

    native = {'diskgraph/_c.abi3.so': b'admitted native build'}
    engine = wheel('kumo_engine-1.0-py3-none-any.whl', 'kumo-engine', native)
    diskgraph = wheel('diskgraph-0.2.0-cp311-abi3-linux_x86_64.whl',
                      'diskgraph', native)
    pytorch = wheel('kumo_pytorch-0.2.0-py3-none-any.whl', 'kumo-pytorch',
                    {'kumo/pytorch/api.py': b'customer API'})
    commands = []

    def download(command, **kwargs):
        commands.append(command)
        destination = Path(command[command.index('--dest') + 1])
        for source in (engine, diskgraph, pytorch):
            shutil.copyfile(source, destination / source.name)

    monkeypatch.setattr(generator.subprocess, 'run', download)
    plan = dict(target=dict(actual, maximum_manylinux='2.35'), local_wheels=[],
                source_index='https://pypi.org/simple')
    archive = generator.resolve(engine, tmp_path / 'resolved', plan,
                                [diskgraph, pytorch])
    result = json.loads(archive.with_name('dependencies.json').read_text())
    assert result['workload_wheels'] == [diskgraph.name, pytorch.name]
    assert set(result['wheels']) == {diskgraph.name, pytorch.name}
    assert str(diskgraph.resolve()) in commands[0] and str(
        pytorch.resolve()) in commands[0]
    assert 'linux_x86_64' in commands[0]
    assert 'manylinux_2_35_x86_64' in commands[0]
    with tarfile.open(archive) as bundle:
        assert set(bundle.getnames()) == {diskgraph.name, pytorch.name}


def test_gpu_resolver_preserves_the_official_cuda_tag_in_the_offline_inventory(monkeypatch, tmp_path):
    import json

    monkeypatch.syspath_prepend(str(DIRECTORY))
    generator = importlib.import_module('resolve_dependencies')
    target = dict(python='3.12', system='Linux', architecture='x86_64', maximum_manylinux='2.35')
    monkeypatch.setattr(generator, 'runtime_target', lambda: {k: target[k] for k in ('python', 'system', 'architecture')})
    wheel = tmp_path / 'engine.whl'
    wheel.write_bytes(b'candidate')
    calls = []

    def pip(command, **kwargs):
        calls.append(command)
        destination = Path(command[command.index('--dest') + 1])
        path = destination / 'torch-2.9.1+cu128-cp312-cp312-manylinux_2_28_x86_64.whl'
        if '--no-deps' in command:
            assert command[command.index('--index-url') + 1] == 'https://download.pytorch.org/whl/cu128'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('torch.dist-info/METADATA', 'Name: torch\nVersion: 2.9.1+cu128\n')
        else:
            assert path.exists()
            assert str(wheel.resolve()) + '[snowflake,pytorch]' in command
            assert command[command.index('--index-url') + 1] == 'https://pypi.org/simple'
        assert 'torch==2.9.1+cu128' in command

    monkeypatch.setattr(generator.subprocess, 'run', pip)
    plan = dict(target=target, local_wheels=[], source_index='https://pypi.org/simple')
    archive = generator.resolve(wheel, tmp_path / 'dependencies', plan, accelerator='gpu', pytorch=True)
    result = json.loads(archive.with_name('dependencies.json').read_text())
    assert len(calls) == 2
    assert result['accelerator'] == 'gpu'
    assert result['pytorch'] is True
    assert list(result['wheels'].values())[0]['version'] == '2.9.1+cu128'
