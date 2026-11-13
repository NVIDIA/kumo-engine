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
