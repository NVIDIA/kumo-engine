# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require distinct matching GPUs, loaded state, and an enforced score tolerance."""
import copy
import fcntl
import importlib.util
import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).parents[1] / 'scripts/weekly_b9.py'


class RetainedChecks(unittest.TestCase):
    def test_retained_prediction_loads_current_wheel_helpers(self):
        spec = importlib.util.spec_from_file_location('b9', SOURCE)
        b9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b9)
        helper = b9.installation_helper()
        for name, arguments in [('verify_installed_files', (None, None)),
                                ('verify_checkpoint', (None, None, 'a' * 64))]:
            with self.subTest(name=name):
                function = getattr(helper, name, None)
                self.assertTrue(callable(function),
                                'B9 must load the current wheel helper')
                inspect.signature(function).bind(*arguments)

    def test_busy_host_stops_before_runtime_imports_or_input_reads(self):
        spec = importlib.util.spec_from_file_location('b9', SOURCE)
        b9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b9)
        with TemporaryDirectory() as directory:
            lock_path = Path(directory) / 'host.lock'
            with lock_path.open('a') as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(b9, 'run_locked', create=True) as execute:
                    with self.assertRaises(BlockingIOError):
                        b9.run(SimpleNamespace(host_lock=lock_path))
                    execute.assert_not_called()

    def test_two_gpu_proof_rejects_aliases_cpu_fallback_and_training(self):
        self.assertTrue(SOURCE.is_file())
        spec = importlib.util.spec_from_file_location('b9', SOURCE)
        b9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b9)
        rows = [
            dict(status='PASS', pid=10 + device, device=device,
                 uuid=f'gpu-{device}', model='matching GPU',
                 checkpoint_sha256='a' * 64, restored_tensors=100, batches=3,
                 elements=40, fit_attempts=0) for device in (0, 1)
        ]
        b9.check_devices(rows, 2, 'a' * 64)
        for key, value in [('uuid', 'gpu-0'), ('device', 0), ('pid', 10),
                           ('fit_attempts', 1), ('batches', 0),
                           ('restored_tensors', 0),
                           ('checkpoint_sha256', 'b' * 64),
                           ('model', 'different GPU')]:
            changed = copy.deepcopy(rows)
            changed[1][key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    b9.check_devices(changed, 2, 'a' * 64)
        with self.assertRaises(ValueError):
            b9.check_devices(rows[:1], 2, 'a' * 64)


if __name__ == '__main__':
    unittest.main()
