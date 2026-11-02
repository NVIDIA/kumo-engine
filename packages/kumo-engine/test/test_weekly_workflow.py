# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute workflow configuration and GPU-class admission with local inputs."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            config = dict(
                manifest_sha256='a' * 64, reference_sha256='b' * 64,
                profiles=dict(rtx=profile, scale=profile),
                p1=dict(runner=profile['runner'], input_root=str(root),
                        reference=str(root / 'p1-reference.json'),
                        host_lock=str(root / 'host.lock'),
                        work_root=str(root / 'p1-work'),
                        manifest_sha256='c' * 64, reference_sha256='d' * 64))
            config['b9'] = dict(runner=profile['runner'], input_root=str(root),
                                reference_root=str(root / 'b9-reference'),
                                host_lock=str(root / 'host.lock'),
                                work_root=str(root / 'b9-work'),
                                manifest_sha256='e' * 64,
                                reference_sha256='f' * 64)
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
            self.assertEqual(json.loads(output.get('p1', '{}')), config['p1'])
            self.assertEqual(json.loads(output.get('b9', '{}')), config['b9'])
            del config['profiles']['scale']
            environment['WEEKLY_CONFIG'] = json.dumps(config)
            rejected = subprocess.run(
                [sys.executable, '-c',
                 program('configure')], cwd=ROOT, env=environment,
                capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)

    def test_p1_job_binds_candidate_wheel_and_approved_reference(self):
        workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
        job = workflow['jobs'].get('p1')
        self.assertIsNotNone(job, 'The manual weekly workflow must execute P1')
        [step] = [
            s for s in job['steps']
            if s.get('name') == 'Run P1 against its approved reference'
        ]
        source = '\n'.join(step['run'].splitlines()[1:-1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheels = root / 'weekly-p1-wheel'
            wheels.mkdir()
            wheel = wheels / 'candidate.whl'
            wheel.write_bytes(b'candidate wheel')
            checksum = hashlib.sha256(wheel.read_bytes()).hexdigest()
            fixtures.weekly.save(
                wheels / 'build.json',
                dict(revision='a' * 40, tree='b' * 40, version='2.0.0',
                     wheel_sha256=checksum))
            config = dict(input_root=str(root / 'inputs'),
                          reference=str(root / 'approved-reference.json'),
                          host_lock=str(root / 'shared.lock'),
                          manifest_sha256='c' * 64, reference_sha256='d' * 64)
            environment = dict(RUNNER_TEMP=str(root),
                               WEEKLY_P1_ROOT=str(root / 'job'),
                               P1_CONFIG=json.dumps(config))
            commands = []
            with patch.dict(os.environ, environment), patch.object(
                    subprocess, 'run', side_effect=lambda command, **kwargs:
                    commands.append(command)):
                exec(compile(source, 'p1-workflow', 'exec'), {})
                [command] = commands
                arguments = dict(zip(command[3::2], command[4::2]))
                self.assertEqual(arguments['--mode'], 'candidate')
                self.assertEqual(arguments['--reference'], config['reference'])
                self.assertEqual(arguments['--reference-sha256'], 'd' * 64)
                self.assertEqual(arguments['--manifest-sha256'], 'c' * 64)
                self.assertEqual(arguments['--host-lock'], config['host_lock'])
                self.assertEqual(arguments['--wheel'], str(wheel))
                self.assertEqual(arguments['--wheel-sha256'], checksum)
                wheel.write_bytes(b'changed wheel')
                with self.assertRaises(ValueError):
                    exec(compile(source, 'p1-workflow', 'exec'), {})
                self.assertEqual(len(commands), 1)

    def test_b9_job_uses_the_shared_wheel_and_retained_reference(self):
        workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
        job = workflow['jobs'].get('b9')
        self.assertIsNotNone(job, 'The manual weekly workflow must execute B9')
        [step] = [
            s for s in job['steps']
            if s.get('name') == 'Run retained P2 prediction on one and two GPUs'
        ]
        source = '\n'.join(step['run'].splitlines()[1:-1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheels = root / 'weekly-b9-wheel'
            wheels.mkdir()
            wheel = wheels / 'candidate.whl'
            wheel.write_bytes(b'candidate wheel')
            checksum = hashlib.sha256(wheel.read_bytes()).hexdigest()
            fixtures.weekly.save(
                wheels / 'build.json',
                dict(revision='a' * 40, tree='b' * 40, version='2.0.0',
                     wheel_sha256=checksum))
            config = dict(input_root=str(root / 'inputs'),
                          reference_root=str(root / 'approved-models'),
                          host_lock=str(root / 'shared.lock'),
                          manifest_sha256='c' * 64, reference_sha256='d' * 64)
            environment = dict(RUNNER_TEMP=str(root),
                               WEEKLY_B9_ROOT=str(root / 'job'),
                               B9_CONFIG=json.dumps(config))
            commands = []
            with patch.dict(os.environ, environment), patch.object(
                    subprocess, 'run', side_effect=lambda command, **kwargs:
                    commands.append(command)):
                exec(compile(source, 'b9-workflow', 'exec'), {})
                [command] = commands
                arguments = dict(zip(command[3::2], command[4::2]))
                self.assertEqual(arguments['--reference-root'],
                                 config['reference_root'])
                self.assertEqual(arguments['--reference-sha256'], 'd' * 64)
                self.assertEqual(arguments['--manifest-sha256'], 'c' * 64)
                self.assertEqual(arguments['--host-lock'], config['host_lock'])
                self.assertEqual(arguments['--wheel'], str(wheel))
                self.assertEqual(arguments['--wheel-sha256'], checksum)
                self.assertEqual(arguments['--version'], '2.0.0')
                wheel.write_bytes(b'changed wheel')
                with self.assertRaises(ValueError):
                    exec(compile(source, 'b9-workflow', 'exec'), {})
                self.assertEqual(len(commands), 1)

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
