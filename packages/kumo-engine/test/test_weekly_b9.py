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
    def test_drift_aligns_scores_by_class_and_reports_boundary_changes(self):
        spec = importlib.util.spec_from_file_location('b9', SOURCE)
        b9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b9)
        reference = dict(entity_ids=[101, 102], ranked_ids=[[1, 2], [3, 4]],
                         values=[[.9, .8], [.5, .3]])
        observed = dict(entity_ids=[101, 102], ranked_ids=[[2, 1], [3, 5]],
                        values=[[.80001, .89999], [.5001, .29999]])
        result = b9.prediction_drift(reference, observed)
        self.assertEqual(result['rank_changed_entities'], 2)
        self.assertEqual(result['membership_changed_entities'], 1)
        self.assertEqual(result['removed_recommendations'], 1)
        self.assertEqual(result['added_recommendations'], 1)
        self.assertEqual(result['common_score_coordinates'], 3)
        self.assertAlmostEqual(result['max_abs_common_score_difference'], .0001)
        self.assertEqual(result['membership_examples'][0]['entity_id'], 102)
        self.assertEqual(result['membership_examples'][0]['removed'], [[4, .3]])
        self.assertEqual(result['membership_examples'][0]['added'],
                         [[5, .29999]])
        self.assertEqual(
            b9.prediction_drift(reference,
                                reference)['max_abs_common_score_difference'],
            0.)
        for changed in (dict(observed, entity_ids=[101]),
                        dict(observed, values=[[float('nan'), .8], [.5, .3]])):
            with self.assertRaises(ValueError):
                b9.prediction_drift(reference, changed)

    def test_active_gpu_work_stops_before_model_imports(self):
        spec = importlib.util.spec_from_file_location('b9', SOURCE)
        b9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b9)
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(host_lock=Path(directory) / 'host.lock')
            with patch.object(b9, 'run_locked',
                              return_value='completed') as execute:
                with patch.object(b9.subprocess, 'check_output',
                                  return_value='123, gpu-a\n'):
                    with self.assertRaisesRegex(ValueError, 'active GPU'):
                        b9.run(args)
                execute.assert_not_called()
                with patch.object(b9.subprocess, 'check_output',
                                  return_value=''):
                    self.assertEqual(b9.run(args), 'completed')

    def test_both_calls_must_match_the_retained_reference(self):
        spec = importlib.util.spec_from_file_location('b9', SOURCE)
        b9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b9)
        reference = dict(entity_ids=[10, 20], ranked_ids=[[3, 4], [4, 5]],
                         values=[[.8, .7], [.9, .6]])
        candidate = copy.deepcopy(reference)
        candidate['values'][0][0] += .001
        result = b9.check_predictions(reference, candidate, reference,
                                      atol=.002, rtol=0.)
        self.assertEqual(set(result), {'one_gpu', 'two_gpu', 'worker_counts'})
        drifted = copy.deepcopy(reference)
        drifted['values'][0][0] += .01
        # Matching each other does not excuse both calls drifting from the reference.
        with self.assertRaises(ValueError):
            b9.check_predictions(reference, drifted, drifted, atol=.002,
                                 rtol=0.)
        candidate['ranked_ids'][0].reverse()
        with self.assertRaises(ValueError):
            b9.check_predictions(reference, candidate, candidate, atol=.002,
                                 rtol=0.)

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
