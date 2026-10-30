# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check remote failure propagation and integration-result identity."""

import argparse
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).parents[2] / 'scripts/engine_nightly_ssm.py'
if SOURCE.is_file():
    SPEC = importlib.util.spec_from_file_location('nightly_ssm', SOURCE)
    controller = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(controller)
else:
    controller = None


class NightlySSMTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(controller, 'Nightly SSM integration is missing')

    def test_waits_for_remote_completion(self):
        states = [
            dict(Status='InProgress', ResponseCode=-1),
            dict(Status='Success', ResponseCode=0)
        ]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(controller, 'aws', side_effect=states), \
                patch.object(controller.time, 'sleep'):
            result = controller.wait_for_command('command', Path(directory), 10)
        self.assertEqual(result['Status'], 'Success')
        self.assertEqual(result['ResponseCode'], 0)

    def test_failed_or_incomplete_invocations_cannot_pass(self):
        for status, code in [('Failed', 0), ('Success', 7), ('TimedOut', -1),
                             ('Cancelled', -1)]:
            with self.subTest(status=status, code=code), \
                    tempfile.TemporaryDirectory() as directory, \
                    patch.object(controller, 'aws', return_value=dict(
                        Status=status, ResponseCode=code)):
                with self.assertRaises(RuntimeError):
                    controller.wait_for_command('command', Path(directory), 10)

    def test_rejects_results_from_another_run_or_model_work(self):
        request = dict(run_id='123-1', source_sha='a' * 40,
                       instance_id='i-06dd0e3c558780731', nonce='original')
        result = dict(request, status='INTEGRATION_PASS', physical_fits=0,
                      gpu_work_started=False, cuda_visible_devices='0')
        controller.validate_result(request, result)
        for change in [
                dict(nonce='stale'),
                dict(run_id='122-1'),
                dict(source_sha='b' * 40),
                dict(physical_fits=1),
                dict(gpu_work_started=True),
                dict(status='InProgress'),
                dict(cuda_visible_devices='0,1')
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                controller.validate_result(request, dict(result, **change))

    def test_poll_timeout_is_a_failure(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(controller.time, 'monotonic', side_effect=[0, 11]):
            with self.assertRaises(TimeoutError):
                controller.wait_for_command('command', Path(directory), 10)

    def test_aws_accepts_successful_operations_without_response_bodies(self):
        completed = subprocess.CompletedProcess([], 0, stdout='', stderr='')
        with patch.object(controller.subprocess, 'run', return_value=completed):
            self.assertEqual(controller.aws('ssm', 'cancel-command'), {})

    def test_cancel_failure_does_not_hide_the_remote_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(run_id='123-1', source_sha='a' * 40,
                                      output_dir=Path(directory), timeout=10)
            with patch.object(controller, 'aws', side_effect=[
                    {}, {'Command': {'CommandId': 'command'}},
                    RuntimeError('Cancellation denied')]), \
                    patch.object(controller, 'wait_for_command',
                                 side_effect=RuntimeError('Remote process failed')):
                with self.assertRaisesRegex(RuntimeError,
                                            'Remote process failed'):
                    controller.run(args)


if __name__ == '__main__':
    unittest.main()
