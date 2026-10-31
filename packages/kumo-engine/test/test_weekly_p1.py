# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete sampling and changed neighborhood membership on CPU."""
import copy
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / 'scripts/weekly_p1.py'


class SamplingChecks(unittest.TestCase):
    def test_failed_worker_stops_other_workers(self):
        self.assertTrue(SOURCE.is_file())
        spec = importlib.util.spec_from_file_location('p1', SOURCE)
        p1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p1)
        self.assertTrue(callable(getattr(p1, 'join_workers', None)))
        failed = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(1)'])
        running = subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(30)'])
        try:
            with self.assertRaises(ValueError):
                p1.join_workers([failed, running], 5)
            self.assertIsNotNone(running.poll())
        finally:
            if running.poll() is None:
                running.kill()
            running.wait()

    def test_five_complete_operations_must_match_reference_membership(self):
        self.assertTrue(SOURCE.is_file())
        spec = importlib.util.spec_from_file_location('p1', SOURCE)
        p1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p1)
        rows = [
            dict(status='PASS', seeds=300000, batches=1172, nodes=1000000,
                 edges=2000000, fetched_rows=1000000, seconds=10.,
                 membership_sha256='a' * 64) for _ in range(5)
        ]
        self.assertEqual(
            p1.check_operations(rows, rows, atol_seconds=1., rtol=.1)['status'],
            'PASS')
        for field in ('nodes', 'edges', 'fetched_rows'):
            changed = copy.deepcopy(rows)
            changed[0][field] -= 1
            with self.subTest(field=field):
                self.assertEqual(
                    p1.check_operations(changed, rows, atol_seconds=1.,
                                        rtol=.1)['status'], 'FAIL')
        for changed in (rows[:4], copy.deepcopy(rows)):
            if len(changed) == 5:
                if changed[0]['membership_sha256'] == 'a' * 64:
                    changed[0]['membership_sha256'] = 'b' * 64
            self.assertEqual(
                p1.check_operations(changed, rows, atol_seconds=1.,
                                    rtol=.1)['status'], 'FAIL')
        changed = copy.deepcopy(rows)
        changed[0]['seeds'] = 299999
        self.assertEqual(
            p1.check_operations(changed, rows, atol_seconds=1.,
                                rtol=.1)['status'], 'FAIL')
        changed = copy.deepcopy(rows)
        changed[0]['status'] = 'FAIL'
        self.assertEqual(
            p1.check_operations(changed, rows, atol_seconds=1.,
                                rtol=.1)['status'], 'FAIL')

    def test_invalid_timing_budget_stops_before_graph_reads(self):
        spec = importlib.util.spec_from_file_location('p1', SOURCE)
        p1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for budget in (None, {}, dict(atol_seconds=-1., rtol=.1),
                           dict(atol_seconds=1., rtol=.1, extra=True)):
                p1.weekly.save(root / 'p1.json',
                               dict(work={}, time_tolerance=budget))
                with self.subTest(budget=budget):
                    with self.assertRaisesRegex(ValueError, 'timing'):
                        p1.inputs(root, p1.weekly.digest(root / 'p1.json'))

    def test_timing_drift_fails_at_the_approved_median_budget(self):
        spec = importlib.util.spec_from_file_location('p1', SOURCE)
        p1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p1)
        reference = [
            dict(status='PASS', seeds=300000, batches=1172, nodes=1000000,
                 edges=2000000, fetched_rows=1000000, seconds=10.,
                 membership_sha256='a' * 64) for _ in range(5)
        ]
        candidate = copy.deepcopy(reference)
        for row in candidate:
            row['seconds'] = 12.
        result = p1.check_operations(candidate, reference, atol_seconds=1.,
                                     rtol=.1)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['timing']['maximum_median_seconds'], 12.)
        candidate[-1]['seconds'] = 100.
        self.assertEqual(
            p1.check_operations(candidate, reference, atol_seconds=1.,
                                rtol=.1)['status'], 'PASS')
        for row in candidate:
            row['seconds'] = 12.01
        self.assertEqual(
            p1.check_operations(candidate, reference, atol_seconds=1.,
                                rtol=.1)['status'], 'FAIL')
        for bad in (0., float('nan'), float('inf'), None):
            candidate = copy.deepcopy(reference)
            candidate[0]['seconds'] = bad
            with self.subTest(duration=bad):
                self.assertEqual(
                    p1.check_operations(candidate, reference, atol_seconds=1.,
                                        rtol=.1)['status'], 'FAIL')
        for policy in (dict(atol_seconds=-1.,
                            rtol=.1), dict(atol_seconds=1., rtol=float('inf'))):
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    p1.check_operations(reference, reference, **policy)


if __name__ == '__main__':
    unittest.main()
