# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import runpy
import unittest
from pathlib import Path

WORKER = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/wheel_task_learning.py')


class WheelTasksTest(unittest.TestCase):
    def worker(self):
        self.assertTrue(WORKER.is_file(), 'The seven-task worker is missing')
        return runpy.run_path(str(WORKER))

    def report(self, worker):
        checks = dict.fromkeys(worker['REQUIRED_CHECKS'], 'PASS')
        rows = []
        for task, (metric, task_type) in worker['TASKS'].items():
            targets = [[int(i == j) for j in range(6)] for i in range(6)]
            if metric == 'mae':
                targets = [[float(i)] * (12 if task == 'forecasting' else 1)
                           for i in range(6)]
            for seed in range(5):
                for case in ('learnable', 'independent'):
                    predictions = targets if case == 'learnable' else [
                        [0.] * len(targets[0]) for _ in targets
                    ]
                    rows.append(
                        dict(
                            task=task, metric=metric, task_type=task_type,
                            seed=seed, training_seed=seed, case=case,
                            status='MEASURED', fit_id=f'{task}-{seed}-{case}',
                            checks=dict(checks, chronology='PASS'),
                            data_sha256='a' * 64, plan_sha256='b' * 64,
                            entity_ids=list(range(6)), targets=targets,
                            predictions=predictions,
                            baseline_predictions=targets,
                            score=worker['checks']()['score'](metric, targets,
                                                              predictions),
                            baseline_score=worker['checks']()['score'](
                                metric, targets, targets)))
                    if task == 'static_link':
                        rows[-1]['control_eligible'] = [[True] * 6
                                                        for _ in range(6)]
                    if task in ('forecasting', 'static_link', 'temporal_link'):
                        rows[-1]['stage_sha256'] = dict.fromkeys(
                            ('train', 'val', 'test'), 'e' * 64)
        return dict(status='MEASURED', run_id='reference', seeds=list(range(5)),
                    source_sha256='c' * 64, dependencies_sha256='d' * 64,
                    python='3.12', platform='linux', runs=rows)

    def test_all_seven_tasks_require_matched_complete_execution(self):
        worker = self.worker()
        report = self.report(worker)
        worker['validate_report'](report, list(range(5)))
        for mutation in ('missing', 'duplicate', 'seed', 'score', 'prediction',
                         'reload', 'chronology', 'task_type', 'failed',
                         'horizons', 'metadata', 'eligible', 'python',
                         'platform', 'seed_header'):
            changed = copy.deepcopy(report)
            row = changed['runs'][0]
            if mutation == 'missing':
                changed['runs'].pop()
            elif mutation == 'duplicate':
                changed['runs'][-1] = row
            elif mutation == 'seed':
                row['training_seed'] = 99
            elif mutation == 'score':
                row['score'] += .1
            elif mutation == 'prediction':
                row['predictions'].pop()
            elif mutation == 'task_type':
                row['task_type'] = 'binary_classification'
            elif mutation == 'failed':
                row['status'] = 'FAIL'
            elif mutation == 'metadata':
                del changed['dependencies_sha256']
            elif mutation in ('python', 'platform'):
                del changed[mutation]
            elif mutation == 'seed_header':
                changed['seeds'][0] = False
            elif mutation == 'eligible':
                del next(r for r in changed['runs']
                         if r['task'] == 'static_link')['control_eligible']
            elif mutation == 'horizons':
                forecast = next(r for r in changed['runs']
                                if r['task'] == 'forecasting')
                for field in ('targets', 'predictions', 'baseline_predictions'):
                    forecast[field] = [r[:1] for r in forecast[field]]
            elif mutation == 'chronology':
                next(r for r in changed['runs'] if r['task'] ==
                     'forecasting')['checks']['chronology'] = 'FAIL'
            else:
                row['checks']['reload'] = 'FAIL'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                worker['validate_report'](changed, list(range(5)))

    def test_task_comparison_has_no_unapproved_quality_gate(self):
        worker = self.worker()
        reference = self.report(worker)
        candidate = copy.deepcopy(reference)
        candidate['run_id'] = 'candidate'
        for row in candidate['runs']:
            row['fit_id'] = 'candidate-' + row['fit_id']
            if row['case'] == 'learnable' and row['metric'] == 'mae':
                row['predictions'] = [[x + 20 for x in y]
                                      for y in row['targets']]
                row['score'] = 20.
        result = worker['compare_tasks'](reference, candidate)
        self.assertEqual(result['status'], 'MEASURED')
        self.assertTrue(result['execution_passed'])
        self.assertIsNone(result['quality_limits'])
        self.assertFalse(result['quality_gate_approved'])
        self.assertEqual(set(result['tasks']), set(worker['TASKS']))
        self.assertEqual(result['tasks']['regression']['mean_quality_drop'],
                         20.)
        self.assertEqual(len(result['tasks']['regression']['per_seed']), 5)

    def test_every_run_needs_a_nonempty_fitted_model_identity(self):
        worker = self.worker()
        for identity in (None, '', ' ', 17):
            report = self.report(worker)
            report['runs'][0]['fit_id'] = identity
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                worker['validate_report'](report, list(range(5)))

    def test_temporal_and_link_runs_require_all_actual_stage_hashes(self):
        worker = self.worker()
        for task in ('forecasting', 'static_link', 'temporal_link'):
            for hashes in (None, {}, {
                    'train': 'e' * 64
            }, dict.fromkeys(('train', 'val', 'test'), 'invalid')):
                report = self.report(worker)
                row = next(row for row in report['runs'] if row['task'] == task)
                if hashes is None:
                    del row['stage_sha256']
                else:
                    row['stage_sha256'] = hashes
                with self.subTest(task=task,
                                  hashes=hashes), self.assertRaises(ValueError):
                    worker['validate_report'](report, list(range(5)))

    def test_comparison_rejects_changed_actual_training_stages(self):
        worker = self.worker()
        reference = self.report(worker)
        candidate = copy.deepcopy(reference)
        candidate['run_id'] = 'candidate'
        for row in candidate['runs']:
            row['fit_id'] = 'candidate-' + row['fit_id']
        forecast = next(row for row in candidate['runs']
                        if row['task'] == 'forecasting')
        forecast['stage_sha256']['train'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'stage_sha256'):
            worker['compare_tasks'](reference, candidate)


if __name__ == '__main__':
    unittest.main()
