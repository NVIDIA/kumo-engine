# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import runpy
import unittest
from pathlib import Path

WORKER = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/wheel_gpu_learning.py')


class WheelGpuTest(unittest.TestCase):
    def worker(self):
        if not WORKER.is_file():
            self.fail('The GPU execution entry point is missing')
        return runpy.run_path(str(WORKER))

    def test_gpu_evidence_rejects_fallback_and_missing_execution_phases(self):
        verify = self.worker()['verify_devices']
        result = dict(
            device='cuda', observed_devices=[
                dict(stage=stage, device='cuda:0')
                for stage in ('training', 'prediction', 'reload_and_predict')
            ])
        verify(result, ('training', 'prediction', 'reload_and_predict'))
        for mutation in ('cpu', 'missing', 'empty', 'wrong_gpu', 'requested'):
            changed = copy.deepcopy(result)
            if mutation == 'cpu':
                changed['observed_devices'][1]['device'] = 'cpu'
            elif mutation == 'missing':
                changed['observed_devices'].pop()
            elif mutation == 'empty':
                changed['observed_devices'] = []
            elif mutation == 'wrong_gpu':
                changed['observed_devices'][1]['device'] = 'cuda:1'
            else:
                changed['device'] = 'cpu'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify(changed,
                       ('training', 'prediction', 'reload_and_predict'))

    def test_gpu_consistency_measures_variation_without_inventing_a_limit(self):
        worker = self.worker()
        self.assertIn('measure_consistency', worker)
        result = worker['measure_consistency']([[0.25, 0.75]],
                                               [[0.250005, 0.749995]])
        self.assertEqual(result['status'], 'MEASURED')
        self.assertAlmostEqual(result['max_abs_difference'], 0.000005)
        self.assertTrue(result['cpu_tolerance_exceeded'])
        self.assertIsNone(result['gpu_tolerance'])
        self.assertFalse(result['qualified'])
        for changed in ([[0.25]], [[float('nan'), 0.75]], [], [0.25, 0.75]):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                worker['measure_consistency']([[0.25, 0.75]], changed)

    def test_gpu_reports_require_matched_runtime_and_fresh_guard_source(self):
        worker = self.worker()
        report = dict(
            configuration=dict(device='cuda', gpu='T4', cuda='13.0'),
            gpu_source_sha256=worker['source_identity'](), runs=[
                dict(device='cuda',
                     observed_devices=[dict(stage='training', device='cuda:0')])
            ])
        worker['verify_report'](report, additional=False)
        for mutation in ('source', 'missing', 'cpu'):
            changed = copy.deepcopy(report)
            if mutation == 'source':
                changed['gpu_source_sha256'] = 'a' * 64
            elif mutation == 'missing':
                changed['runs'][0]['observed_devices'] = []
            else:
                changed['configuration']['device'] = 'cpu'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                worker['verify_report'](changed, additional=False)

    def test_additional_gpu_reports_reject_unqualified_or_nonfinite_consistency(
            self):
        worker = self.worker()
        row = dict(
            device='cuda', observed_devices=[
                dict(stage=stage, device='cuda:0')
                for stage in ('training', 'prediction', 'reload_and_predict')
            ], consistency_measurements=[
                dict(stage=stage, status='MEASURED', qualified=False,
                     gpu_tolerance=None, max_abs_difference=0.0)
                for stage in ('prediction', 'reload_and_predict')
            ])
        report = dict(configuration={'device': 'cuda'},
                      gpu_source_sha256=worker['source_identity'](), runs=[row])
        worker['verify_report'](report, additional=True)
        for mutation in ('missing', 'nan', 'approved'):
            changed = copy.deepcopy(report)
            measured = changed['runs'][0]['consistency_measurements']
            if mutation == 'missing':
                measured.pop()
            elif mutation == 'nan':
                measured[0]['max_abs_difference'] = float('nan')
            else:
                measured[0]['qualified'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                worker['verify_report'](changed, additional=True)


if __name__ == '__main__':
    unittest.main()
