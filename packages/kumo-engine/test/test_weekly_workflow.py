# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute workflow configuration and GPU-class admission with local inputs."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / '.github/workflows/kumo-engine-weekly.yml'
_SPEC = importlib.util.spec_from_file_location(
    'weekly_fixtures',
    Path(__file__).with_name('test_weekly_check.py'))
fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)


def program(job):
    value = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    [step] = [step for step in value['jobs'][job]['steps'] if 'run' in step]
    return '\n'.join(step['run'].splitlines()[1:-1])


class WorkflowChecks(unittest.TestCase):
    def test_configuration_requires_both_existing_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = dict(runner=dict(group='existing',
                                       labels='admitted'), input_root=str(root),
                           reference_root=str(root / 'reference'),
                           host_lock=str(root / 'host.lock'),
                           work_root=str(root / 'work'))
            config = dict(manifest_sha256='a' * 64, reference_sha256='b' * 64,
                          profiles=dict(rtx=profile, scale=profile))
            environment = dict(os.environ, MODE='candidate',
                               GITHUB_OUTPUT=str(root / 'outputs'))
            environment['WEEKLY_CONFIG'] = json.dumps(config)
            result = subprocess.run(
                [sys.executable, '-c',
                 program('configure')], cwd=ROOT, env=environment,
                capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            output = dict(
                line.split('=', 1)
                for line in (root / 'outputs').read_text().splitlines())
            self.assertEqual(
                [v['profile'] for v in json.loads(output['matrix'])['include']],
                ['rtx', 'scale'])
            del config['profiles']['scale']
            environment['WEEKLY_CONFIG'] = json.dumps(config)
            rejected = subprocess.run(
                [sys.executable, '-c',
                 program('configure')], cwd=ROOT, env=environment,
                capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)

    def test_admission_rejects_a_different_gpu_class_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures.input_fixture(root)
            binary = root / 'nvidia-smi'
            config = dict(
                profile='rtx', input_root=str(root),
                reference_root=str(root / 'reference'),
                host_lock=str(root / 'host.lock'), work_root=str(root),
                manifest_sha256=fixtures.weekly.digest(root / 'manifest.json'))
            environment = dict(os.environ, PROFILE_JSON=json.dumps(config),
                               MODE='reference', RUNNER_TEMP=str(root),
                               PATH=str(root) + os.pathsep + os.environ['PATH'])
            for hardware, expected in [('12.0, 100000', 0), ('7.5, 100000', 1),
                                       ('12.0, 0', 1)]:
                binary.write_text("#!/bin/sh\nprintf '%s\\n' '" + hardware +
                                  "'\n")
                binary.chmod(0o755)
                result = subprocess.run(
                    [sys.executable, '-c',
                     program('admit')], cwd=ROOT, env=environment,
                    capture_output=True)
                with self.subTest(hardware=hardware):
                    self.assertEqual(result.returncode == 0, expected == 0,
                                     result.stderr.decode())


if __name__ == '__main__':
    unittest.main()
