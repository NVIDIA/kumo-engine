# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete sampling and changed neighborhood membership on CPU."""
import copy
import importlib.util
import subprocess
import sys
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
                 edges=2000000, fetched_rows=1000000,
                 membership_sha256='a' * 64) for _ in range(5)
        ]
        self.assertEqual(p1.check_operations(rows, rows)['status'], 'PASS')
        for field in ('nodes', 'edges', 'fetched_rows'):
            changed = copy.deepcopy(rows)
            changed[0][field] -= 1
            with self.subTest(field=field):
                self.assertEqual(
                    p1.check_operations(changed, rows)['status'], 'FAIL')
        for changed in (rows[:4], copy.deepcopy(rows)):
            if len(changed) == 5:
                if changed[0]['membership_sha256'] == 'a' * 64:
                    changed[0]['membership_sha256'] = 'b' * 64
            self.assertEqual(
                p1.check_operations(changed, rows)['status'], 'FAIL')
        changed = copy.deepcopy(rows)
        changed[0]['seeds'] = 299999
        self.assertEqual(p1.check_operations(changed, rows)['status'], 'FAIL')
        changed = copy.deepcopy(rows)
        changed[0]['status'] = 'FAIL'
        self.assertEqual(p1.check_operations(changed, rows)['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
