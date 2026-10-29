# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import runpy
import subprocess
import sys
import tempfile
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

    def test_release_comparison_rejects_calibration_headers_and_rows(self):
        worker = self.worker()
        for marker in ('header', 'row'):
            reference = self.report(worker)
            candidate = copy.deepcopy(reference)
            candidate['run_id'] = 'candidate'
            for row in candidate['runs']:
                row['fit_id'] = 'candidate-' + row['fit_id']
            target = candidate if marker == 'header' else candidate['runs'][0]
            target['calibration_fault'] = dict(kind='reduced_training_budget',
                                               budget=[1, 2])
            with self.subTest(marker=marker), self.assertRaisesRegex(
                    ValueError, 'calibration'):
                worker['compare_tasks'](reference, candidate)

    def test_calibration_requires_complete_matching_fault_descriptions(self):
        worker = self.worker()
        report = self.report(worker)
        report['calibration_fault'] = dict(kind='reduced_training_budget',
                                           budget=[1, 2])
        for row in report['runs']:
            row['calibration_fault'] = dict(
                report['calibration_fault'], original_plan={
                    'optimization':
                    dict(max_epochs=10, min_steps_per_epoch=20,
                         max_steps_per_epoch=20)
                })
        try:
            worker['validate_report'](report, list(range(5)), calibration=True)
        except TypeError as error:
            self.fail(f'Fault measurements need explicit validation: {error}')
        for mutation in ('header', 'row', 'budget', 'original', 'kind'):
            changed = copy.deepcopy(report)
            if mutation == 'header':
                del changed['calibration_fault']
            elif mutation == 'row':
                del changed['runs'][0]['calibration_fault']
            elif mutation == 'budget':
                changed['runs'][0]['calibration_fault']['budget'] = [2, 2]
            elif mutation == 'original':
                del changed['runs'][0]['calibration_fault']['original_plan']
            else:
                changed['calibration_fault']['kind'] = 'edited_predictions'
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    ValueError, 'calibration'):
                worker['validate_report'](changed, list(range(5)),
                                          calibration=True)

    def test_calibration_execution_failure_keeps_fault_and_failed_evidence(
            self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'fault'
            completed = subprocess.run([
                sys.executable,
                str(WORKER), '--version', '0.invalid-calibration-test',
                '--output-dir',
                str(output), '--calibration-budget', '1', '2'
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            result = output / 'task-result.json'
            self.assertTrue(result.is_file(), completed.stderr)
            report = json.loads(result.read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(
                report['calibration_fault'],
                dict(kind='reduced_training_budget', budget=[1, 2]))

    def calibration_pair(self, worker, root):
        reference = self.report(worker)
        candidate = copy.deepcopy(reference)
        candidate['run_id'] = 'candidate'
        candidate['calibration_fault'] = dict(kind='reduced_training_budget',
                                              budget=[1, 2])
        for old, new in zip(reference['runs'], candidate['runs']):
            new['fit_id'] = 'candidate-' + new['fit_id']
            original = dict(
                training_job={'manual_seed': old['seed']},
                optimization=dict(max_epochs=10, min_steps_per_epoch=20,
                                  max_steps_per_epoch=20))
            weakened = copy.deepcopy(original)
            weakened['optimization'].update(max_epochs=1, min_steps_per_epoch=2,
                                            max_steps_per_epoch=2)
            new['calibration_fault'] = dict(candidate['calibration_fault'],
                                            original_plan=original)
            if new['metric'] == 'mae' and new['case'] == 'learnable':
                new['predictions'] = [[value + .25 for value in target]
                                      for target in new['targets']]
                new['score'] = .25
            for arm, row, requested in [('reference', old, original),
                                        ('candidate', new, weakened)]:
                path = root / arm / (
                    f'{row["task"]}-{row["seed"]}-{row["case"]}')
                path.mkdir(parents=True)
                selected = dict(requested, resolved_task=row['task_type'])
                row['plan_sha256'] = worker['checks']()['fingerprint'](selected)
                filename = ('selected-plan.json' if row['task'] in (
                    'forecasting', 'static_link',
                    'temporal_link') else 'selected-model-plan.json')
                (path / 'requested-plan.json').write_text(json.dumps(requested))
                (path / filename).write_text(json.dumps(selected))
        return reference, candidate

    def test_sensitivity_compares_scores_after_verifying_actual_training_plans(
            self):
        worker = self.worker()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, candidate = self.calibration_pair(worker, root)
            preserved = copy.deepcopy([reference, candidate])
            try:
                result = worker['compare_tasks'](
                    reference, candidate,
                    calibration_dirs=(root / 'reference', root / 'candidate'))
            except TypeError as error:
                self.fail(f'Missing audited sensitivity comparison: {error}')
            self.assertEqual([reference, candidate], preserved)
            self.assertEqual(result['status'], 'MEASURED')
            self.assertFalse(result['quality_gate_approved'])
            self.assertEqual(result['calibration_fault'],
                             candidate['calibration_fault'])
            self.assertEqual(len(result['calibration_plan_checks']), 70)
            for task in ('regression', 'forecasting'):
                self.assertEqual(result['tasks'][task]['mean_quality_drop'],
                                 .25)
            for proof in result['calibration_plan_checks']:
                self.assertNotEqual(proof['reference_plan_sha256'],
                                    proof['candidate_plan_sha256'])

    def test_sensitivity_rejects_nonbudget_plan_changes_even_with_new_hashes(
            self):
        worker = self.worker()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, candidate = self.calibration_pair(worker, root)
            row = candidate['runs'][0]
            path = root / 'candidate' / (
                f'{row["task"]}-{row["seed"]}-{row["case"]}')
            selected = path / 'selected-model-plan.json'
            changed = json.loads(selected.read_text())
            changed['optimization']['batch_size'] = [32]
            selected.write_text(json.dumps(changed))
            row['plan_sha256'] = worker['checks']()['fingerprint'](changed)
            try:
                with self.assertRaisesRegex(ValueError, 'calibration'):
                    worker['compare_tasks'](
                        reference, candidate,
                        calibration_dirs=(root / 'reference',
                                          root / 'candidate'))
            except TypeError as error:
                self.fail(f'Missing audited sensitivity comparison: {error}')

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
