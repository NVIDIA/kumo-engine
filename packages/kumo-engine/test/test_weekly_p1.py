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
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

SOURCE = Path(__file__).parents[1] / 'scripts/weekly_p1.py'


class SamplingChecks(unittest.TestCase):
    def test_reference_keeps_measurements_separate_from_the_timing_policy(self):
        spec = importlib.util.spec_from_file_location('p1', SOURCE)
        p1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p1)
        manifest = dict(work=dict(n=300000), files={'index/data': 'a' * 64},
                        time_tolerance=dict(atol_seconds=2., rtol=.1))
        measured = copy.deepcopy(manifest)
        measured['time_tolerance'] = dict(atol_seconds=0., rtol=0.)
        measured['calibration_only'] = True
        rows = [
            dict(status='PASS', seeds=300000, batches=1172, nodes=1000000,
                 edges=2000000, fetched_rows=1000000, seconds=10.,
                 membership_sha256='a' * 64) for _ in range(5)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.json'
            value = dict(format='kumo-engine-weekly-p1-reference-v1',
                         status='CALIBRATED', manifest_sha256='b' * 64,
                         workload_sha256=p1.workload_digest(measured),
                         physical_fits=0, operations=rows,
                         measurement_results_sha256=['c' * 64])
            p1.weekly.save(path, value)
            self.assertEqual(
                p1.reference_operations(path, p1.weekly.digest(path), manifest,
                                        'b' * 64), rows)
            for change in ('bytes', 'work', 'status', 'fits', 'checksum'):
                candidate = copy.deepcopy(manifest)
                reference = copy.deepcopy(value)
                checksum = 'd' * 64 if change == 'checksum' else None
                if change == 'bytes':
                    candidate['files']['index/data'] = 'd' * 64
                if change == 'work':
                    candidate['work']['n'] -= 1
                if change == 'status':
                    reference['status'] = 'FAIL'
                if change == 'fits':
                    reference['physical_fits'] = 1
                p1.weekly.save(path, reference)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    p1.reference_operations(path, checksum
                                            or p1.weekly.digest(path),
                                            candidate, 'b' * 64)

    def test_installed_wheel_is_checked_before_sampling(self):
        spec = importlib.util.spec_from_file_location('p1', SOURCE)
        p1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p1)
        original_flags = sys.flags

        class IsolatedFlags:
            isolated = 1

            def __getattr__(self, name):
                return getattr(original_flags, name)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'inputs'
            metadata = inputs / 'index/ingest_stats.json'
            p1.weekly.save(
                metadata,
                dict(stats=dict(node_counts=dict(customers=100000000))))
            manifest = dict(
                format='kumo-engine-weekly-p1-v1', dataset='commerce/100M',
                index='index', row_counts=dict(customers=100000000),
                files={'index/ingest_stats.json': p1.weekly.digest(metadata)},
                work=dict(n=300000, offset=0, batch=256, procs=1, threads=1,
                          hops=2), time_tolerance=dict(atol_seconds=1.,
                                                       rtol=.1))
            p1.weekly.save(inputs / 'p1.json', manifest)
            installed = root / 'installed'
            installed.mkdir()
            payload = installed / 'module.py'
            payload.write_bytes(b'approved wheel bytes')
            wheel = root / 'reference.whl'
            with ZipFile(wheel, 'w') as archive:
                archive.writestr('module.py', b'approved wheel bytes')
            distribution = SimpleNamespace(
                version='1.0.0', locate_file=lambda name: installed / name)
            args = SimpleNamespace(
                input_root=inputs,
                manifest_sha256=p1.weekly.digest(inputs / 'p1.json'),
                output=root / 'valid', wheel=wheel,
                wheel_sha256=p1.weekly.digest(wheel), reference=None,
                mode='reference', timeout_seconds=1)
            with patch.object(p1.sys, 'flags', IsolatedFlags()), patch.object(
                    p1.importlib.metadata, 'distribution',
                    return_value=distribution), patch.object(
                        p1.subprocess, 'Popen', side_effect=RuntimeError(
                            'Native sampling intentionally stopped')):
                result = p1.run_locked(args)
                self.assertIn('Native sampling intentionally stopped',
                              result['error'])
                payload.write_bytes(b'changed installed bytes')
                args.output = root / 'changed'
                result = p1.run_locked(args)
                self.assertIn('Installed wheel member differs', result['error'])

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
