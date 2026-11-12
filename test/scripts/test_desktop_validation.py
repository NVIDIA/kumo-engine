# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure-detection and isolation checks for the desktop qualification harness."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[2] / 'packages/kumo-engine/test'
SPEC = importlib.util.spec_from_file_location(
    'desktop_validation', DIRECTORY / 'desktop_validation.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize('attributes,expected', [
    ('tests="1" failures="0" errors="0" skipped="0"', 'passed'),
    ('tests="0"', 'failed'),
    ('tests="1" skipped="1"', 'blocked'),
    ('tests="1" failures="1"', 'failed'),
    ('tests="1" errors="1"', 'failed'),
])
def test_missing_or_skipped_execution_cannot_qualify(tmp_path, attributes,
                                                     expected):
    path = tmp_path / 'junit.xml'
    path.write_text('<testsuites><testsuite ' + attributes + '/></testsuites>')
    assert runner.junit_verdict(path)['status'] == expected


def test_unavailable_gpu_is_not_a_full_platform_pass():
    cases = {'recovery': {'status': 'passed'}}
    profiles = {'cpu': {'status': 'passed'}, 'mps': {'status': 'blocked'}}
    assert runner.qualification_status(cases, profiles) == 'blocked'
    profiles['mps']['status'] = 'failed'
    assert runner.qualification_status(cases, profiles) == 'failed'
    profiles['mps']['status'] = 'passed'
    assert runner.qualification_status(cases, profiles) == 'passed'


def test_preparation_preserves_quickstart_layout_and_exact_helpers(tmp_path):
    runner.prepare(tmp_path)
    manifest = json.loads((tmp_path / 'recipe.json').read_text())
    assert 'README.md' in manifest
    assert 'packages/kumo-engine/README.md' in manifest
    for name, digest in manifest.items():
        assert runner.helpers().digest(tmp_path / name) == digest
    assert not (tmp_path / 'packages/kumo-engine/kumo').exists()


def test_module_scoped_training_is_observed_before_function_fixtures(tmp_path):
    shutil.copyfile(DIRECTORY / 'desktop_evidence.py',
                    tmp_path / 'desktop_evidence.py')
    (tmp_path / 'conftest.py').write_text('''
import sys
import types
import pytest

class Trainer:
    def __init__(self, accelerator='auto'):
        self.accelerator = accelerator

class CoreTrainer:
    def __init__(self, accelerator):
        self.accelerator = accelerator
    def create_pl_trainer(self):
        self.pl_trainer = types.SimpleNamespace(strategy=types.SimpleNamespace(
            root_device=types.SimpleNamespace(type=self.accelerator)))

for name in ['kumo', 'kumo.engine', 'kumo_core', 'kumo_core.train', 'kumo_core.train.trainer']:
    sys.modules[name] = types.ModuleType(name)
sys.modules['kumo'].engine = sys.modules['kumo.engine']
sys.modules['kumo.engine'].Trainer = Trainer
sys.modules['kumo_core.train.trainer'].Trainer = CoreTrainer

@pytest.fixture(scope='module')
def fitted():
    trainer = Trainer()
    core = CoreTrainer(trainer.accelerator)
    core.create_pl_trainer()
    return trainer
''')
    (tmp_path / 'test_recipe.py').write_text('''
def test_device(fitted):
    assert fitted.accelerator == 'cpu'
''')
    receipt = tmp_path / 'devices.json'
    env = dict(os.environ, PYTHONPATH=str(tmp_path),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               DESKTOP_DEVICE_RECEIPT=str(receipt))
    result = subprocess.run([
        sys.executable, '-m', 'pytest', '-p', 'desktop_evidence',
        '--confcutdir=.', '-q', 'test_recipe.py'
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    devices = json.loads(receipt.read_text())['devices']
    assert devices == [{'test': 'test_recipe.py::test_device', 'device': 'cpu'}]
