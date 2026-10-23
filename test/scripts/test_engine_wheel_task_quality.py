# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic study fixtures test gate behavior, not actual learning quality."""

import copy
import hashlib
import json
import math
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[2] / 'packages/kumo-engine/test'
QUALITY = DIRECTORY / 'wheel_task_quality.py'
METRICS = dict(multiclass='accuracy', multilabel='ap_macro',
               multilabel_ranking='recall@5', regression='mae',
               forecasting='mae', static_link='recall@5',
               temporal_link='recall@5')
TEST_LIMITS = dict(multiclass=.01, multilabel=.02, multilabel_ranking=.03,
                   regression=.04, forecasting=.05, static_link=.06,
                   temporal_link=.07)
TEST_NOISE = dict(multiclass=.002, multilabel=.004, multilabel_ranking=.006,
                  regression=.008, forecasting=.01, static_link=.012,
                  temporal_link=.014)
TEST_MINIMUMS = dict(multiclass=.018, multilabel=.036, multilabel_ranking=.054,
                     regression=.072, forecasting=.09, static_link=.108,
                     temporal_link=.126)


def task_scores_for_tests(drops=None):
    drops = drops or dict.fromkeys(METRICS, 0.)
    tasks = {}
    for task, metric in METRICS.items():
        before = 1. if metric == 'mae' else .8
        after = round(before + (1 if metric == 'mae' else -1) * drops[task], 12)
        tasks[task] = dict(
            metric=metric, per_seed=[
                dict(seed=seed, reference=before, candidate=after,
                     baseline=2. if metric == 'mae' else .5,
                     quality_drop=drops[task]) for seed in range(5)
            ])
    return tasks


def comparison_for_tests(drops=None):
    worker = runpy.run_path(str(DIRECTORY / 'wheel_task_learning.py'))
    wheel = json.loads(
        (DIRECTORY / 'wheel_workflow.json').read_text())['reference']['wheel']
    return dict(status='MEASURED', execution_passed=True,
                tasks=task_scores_for_tests(drops),
                source_sha256=worker['source_identity'](),
                dependencies_sha256='d' * 64, reference_wheel=wheel,
                seeds=list(range(5)))


def profile_for_tests():
    """These arbitrary test limits must never become a release profile."""
    comparison = comparison_for_tests()
    profile = dict(
        status='VALIDATED', source_sha256=comparison['source_sha256'],
        dependencies_sha256='d' * 64,
        reference_wheel=comparison['reference_wheel'], seeds=list(range(5)),
        frozen_at='2026-10-20T01:00:00Z', limits=TEST_LIMITS.copy(),
        frozen_fault=dict(kind='reduced_training_budget', budget=[1, 2]),
        minimum_losses=TEST_MINIMUMS.copy(), rationales={
            task: 'Synthetic fixture for gate tests only.'
            for task in METRICS
        }, reliability=dict(joint_false_failure_max=.05, task_detection_min=.95,
                            simultaneous_confidence=.95, claims=8,
                            reserved_trials=99), calibration=[], validation=[],
        validation_plan={
            kind: [f'validation-{kind}-{index}' for index in range(99)]
            for kind in ('unchanged', 'weakened')
        })
    for phase, count in [('calibration', 6), ('validation', 99)]:
        for kind in ('unchanged', 'weakened'):
            for index in range(count):
                name = f'{phase}-{kind}-{index}'
                if kind == 'weakened':
                    drops = dict.fromkeys(METRICS, .3)
                else:
                    multiplier = [0, 1, -1, 0, .5, -.5
                                  ][index] if phase == 'calibration' else 0
                    drops = {
                        task: round(limit * multiplier, 12)
                        for task, limit in TEST_NOISE.items()
                    }
                profile[phase].append(
                    dict(
                        id=name, kind=kind, status='PASS', audit_passed=True,
                        source_sha256=comparison['source_sha256'],
                        dependencies_sha256='d' * 64,
                        reference_wheel=comparison['reference_wheel'],
                        candidate_wheel=comparison['reference_wheel'],
                        reference_run_id=name + '-reference',
                        candidate_run_id=name + '-candidate', run_ids=[
                            name + suffix
                            for suffix in ('-reference', '-candidate',
                                           '-binary-reference',
                                           '-binary-candidate')
                        ], fit_ids=[f'{name}-fit-{fit}' for fit in range(140)
                                    ], evidence_sha256=hashlib.sha256(
                                        name.encode()).hexdigest(),
                        started_at=('2026-10-20T00:00:00Z'
                                    if phase == 'calibration' else
                                    '2026-10-20T02:00:00Z'),
                        completed_at=('2026-10-20T00:01:00Z'
                                      if phase == 'calibration' else
                                      '2026-10-20T02:01:00Z'),
                        seeds=list(range(5)), fault=(dict(
                            kind='reduced_training_budget',
                            budget=[1, 2]) if kind == 'weakened' else None),
                        tasks=task_scores_for_tests(drops)))
    return profile


class WheelTaskQualityTest(unittest.TestCase):
    def helpers(self):
        self.assertTrue(QUALITY.is_file(),
                        'The seven-task quality gate is missing')
        return runpy.run_path(str(QUALITY))

    def test_validation_cannot_replace_a_planned_trial_with_a_new_success(self):
        profile = profile_for_tests()
        profile['validation'][0]['id'] = 'replacement-after-a-failed-trial'
        with self.assertRaises(ValueError):
            self.helpers()['assess_quality'](comparison_for_tests(), profile)

    def test_validation_plan_requires_complete_unique_trial_ids(self):
        assess = self.helpers()['assess_quality']
        for mutation in ('missing', 'short', 'duplicate', 'empty', 'extra'):
            profile = profile_for_tests()
            plan = profile['validation_plan']
            if mutation == 'missing':
                del profile['validation_plan']
            elif mutation == 'short':
                plan['unchanged'].pop()
            elif mutation == 'duplicate':
                plan['unchanged'][0] = plan['weakened'][0]
            elif mutation == 'empty':
                plan['unchanged'][0] = ''
            elif mutation == 'extra':
                plan['unused'] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                assess(comparison_for_tests(), profile)

    def test_reporter_blocks_unqualified_release_but_keeps_reporting_available(
            self):
        tests = Path(__file__).parent
        binary = runpy.run_path(str(
            tests / 'test_engine_wheel_learning.py'))['WheelLearningTest']()
        binary.setUpClass()
        reference, candidate, baseline = binary.release_pair()
        task_tests = runpy.run_path(str(
            tests / 'test_engine_wheel_tasks.py'))['WheelTasksTest']()
        worker = task_tests.worker()
        for arm, result in [('reference', reference), ('candidate', candidate)]:
            result['task_learning'] = task_tests.report(worker)
            task = result['task_learning']
            task.update(
                run_id=arm, source_sha256=worker['source_identity'](),
                dependencies_sha256=result['learning']['dependencies_sha256'])
            for row in task['runs']:
                row['fit_id'] = arm + '-' + row['fit_id']
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            inputs = root / 'input.json'
            inputs.write_text(json.dumps([reference, candidate, baseline]))
            for release in (False, True):
                output = root / f'{release}.json'
                completed = subprocess.run([
                    sys.executable,
                    str(DIRECTORY / 'wheel_learning_report.py'),
                    str(inputs),
                    str(output), '--tasks', *(['--release'] if release else [])
                ], capture_output=True, text=True)
                with self.subTest(release=release):
                    self.assertEqual(completed.returncode, int(release),
                                     completed.stderr)
                    result = json.loads(output.read_text())
                    self.assertFalse(result['release_approved'])
                    self.assertEqual(result['status'],
                                     'FAIL' if release else 'MEASURED')
                    task = result['additional_tasks']
                    self.assertEqual(task['quality_status'], 'UNVALIDATED')
                    self.assertFalse(task['quality_gate_approved'])
                    self.assertEqual(
                        task['tasks']['regression']['metric_units'],
                        'target_units')

    def test_each_metric_blocks_only_above_its_own_signed_average_limit(self):
        assess = self.helpers()['assess_quality']
        profile = profile_for_tests()
        for task, metric in METRICS.items():
            for offset, passed in [(-.0001, True), (0, True), (.0001, False)]:
                comparison = comparison_for_tests()
                limit = TEST_LIMITS[task]
                after = round((1. + limit + offset) if metric == 'mae' else
                              (.8 - limit - offset), 12)
                for row in comparison['tasks'][task]['per_seed']:
                    row['candidate'] = after
                with self.subTest(task=task, offset=offset):
                    result = assess(comparison, profile)
                    self.assertEqual(result['quality_gate_approved'], passed)
                    self.assertEqual(result['quality_status'],
                                     'PASS' if passed else 'FAIL')
                    detail = result['tasks'][task]
                    self.assertEqual(detail['mean_drop_alert'],
                                     dict(limit=limit, exceeded=not passed))
                    self.assertEqual(detail['higher_is_better'], metric
                                     != 'mae')
                    self.assertEqual(
                        detail['metric_units'],
                        'target_units' if metric == 'mae' else 'fraction')
                    self.assertEqual(detail['mean_candidate'], after)
                    self.assertEqual(len(detail['per_seed']), 5)

    def test_next_float_above_the_limit_does_not_receive_an_epsilon_allowance(
            self):
        assess, profile = self.helpers()['assess_quality'], profile_for_tests()
        for task, metric in METRICS.items():
            comparison = comparison_for_tests()
            edge = round(
                1. + TEST_LIMITS[task] if metric == 'mae' else .8 -
                TEST_LIMITS[task], 12)
            for row in comparison['tasks'][task]['per_seed']:
                row['candidate'] = math.nextafter(
                    edge, math.inf if metric == 'mae' else -math.inf)
            with self.subTest(task=task):
                self.assertFalse(
                    assess(comparison, profile)['quality_gate_approved'])

    def test_exact_decimal_limit_survives_a_nonrepresentable_calibration_mean(
            self):
        profile = profile_for_tests()
        # These scores average 0.30000000000000002 exactly, not 0.3.
        profile['limits']['multiclass'] = '0.54999999999999999'
        profile['minimum_losses']['multiclass'] = .6
        row = profile['calibration'][0]['tasks']['multiclass']['per_seed']
        for entry, value in zip(row, [.1, .2, .3, .4, .5000000000000001]):
            entry['candidate'] = value
        for trial in profile['calibration'] + profile['validation']:
            if trial['kind'] == 'weakened':
                for entry in trial['tasks']['multiclass']['per_seed']:
                    entry['candidate'] = .1
        result = self.helpers()['assess_quality'](comparison_for_tests(),
                                                  profile)
        self.assertTrue(result['quality_gate_approved'])

    def test_improvements_offset_drops_instead_of_using_individual_limits(self):
        comparison, profile = comparison_for_tests(), profile_for_tests()
        for task, metric in METRICS.items():
            rows = comparison['tasks'][task]['per_seed']
            rows[0]['candidate'] = 1.1 if metric == 'mae' else .7
            rows[1]['candidate'] = .9 if metric == 'mae' else .9
        result = self.helpers()['assess_quality'](comparison, profile)
        self.assertTrue(result['quality_gate_approved'])
        self.assertTrue(
            all(task['mean_quality_drop'] == 0
                for task in result['tasks'].values()))

    def test_unapproved_profile_or_fault_measurement_cannot_approve_release(
            self):
        assess = self.helpers()['assess_quality']
        result = assess(comparison_for_tests(), dict(status='PENDING'))
        self.assertFalse(result['quality_gate_approved'])
        self.assertEqual(result['quality_status'], 'UNVALIDATED')
        comparison = comparison_for_tests()
        comparison['calibration_fault'] = dict(kind='reduced_training_budget',
                                               budget=[1, 2])
        self.assertFalse(
            assess(comparison, profile_for_tests())['quality_gate_approved'])

    def test_profile_requires_fresh_complete_reliable_evidence(self):
        assess = self.helpers()['assess_quality']
        for mutation in ('missing', 'short', 'copied', 'audit', 'failure',
                         'early', 'policy', 'normal_loss', 'undetected',
                         'weak_loss', 'limit', 'source', 'dependencies',
                         'seeds', 'model_reuse'):
            profile = profile_for_tests()
            if mutation == 'missing':
                del profile['limits']['forecasting']
            elif mutation == 'short':
                profile['validation'].pop()
            elif mutation == 'copied':
                profile['validation'][0] = copy.deepcopy(
                    profile['calibration'][0])
            elif mutation == 'audit':
                profile['validation'][0]['audit_passed'] = False
            elif mutation == 'failure':
                profile['validation'][0]['status'] = 'FAIL'
            elif mutation == 'early':
                profile['validation'][0]['started_at'] = '2026-10-20T00:59:59Z'
            elif mutation == 'policy':
                profile['reliability']['task_detection_min'] = .5
            elif mutation == 'limit':
                profile['limits']['multiclass'] = .02
            elif mutation in ('normal_loss', 'undetected', 'weak_loss'):
                index = 0 if mutation == 'normal_loss' else 99
                score = .7 if mutation == 'normal_loss' else .8 if mutation == 'undetected' else .785
                for row in profile['validation'][index]['tasks']['multiclass'][
                        'per_seed']:
                    row['candidate'] = score
            elif mutation == 'source':
                profile['source_sha256'] = 'e' * 64
            elif mutation == 'dependencies':
                profile['dependencies_sha256'] = 'e' * 64
            elif mutation == 'seeds':
                profile['validation'][0]['seeds'][0] = False
            else:
                profile['validation'][1]['reference_run_id'] = profile[
                    'validation'][0]['reference_run_id']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                assess(comparison_for_tests(), profile)

    def test_reserved_trials_cannot_change_the_frozen_regression(self):
        profile = profile_for_tests()
        profile['validation'][99]['fault']['budget'] = [1, 1]
        with self.assertRaises(ValueError):
            self.helpers()['assess_quality'](comparison_for_tests(), profile)

    def test_qualification_requires_fresh_models_not_only_fresh_trial_names(
            self):
        assess = self.helpers()['assess_quality']
        for mutation in ('missing', 'short', 'duplicate', 'reused',
                         'binary_reused', 'wrong_run'):
            profile = profile_for_tests()
            first, second = profile['validation'][:2]
            if mutation == 'missing':
                del first['fit_ids']
            elif mutation == 'short':
                first['fit_ids'].pop()
            elif mutation == 'duplicate':
                first['fit_ids'][1] = first['fit_ids'][0]
            elif mutation == 'reused':
                second['fit_ids'][0] = first['fit_ids'][0]
            elif mutation == 'binary_reused':
                second['run_ids'][2] = first['run_ids'][2]
            else:
                first['run_ids'][0] = 'an-unrelated-run'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                assess(comparison_for_tests(), profile)

    def test_blockworthy_loss_policy_can_preserve_an_exact_fraction(self):
        profile = profile_for_tests()
        profile['minimum_losses']['multiclass'] = '9/500'
        result = self.helpers()['assess_quality'](comparison_for_tests(),
                                                  profile)
        self.assertTrue(result['quality_gate_approved'])

    def test_limit_calculation_leaves_equal_margin_to_noise_and_blockworthy_loss(
            self):
        result = self.helpers()['assess_quality'](comparison_for_tests(),
                                                  profile_for_tests())
        calibration = result['qualification']['calibration']
        self.assertEqual(calibration['unchanged_trials'], 6)
        self.assertEqual(calibration['weakened_trials'], 6)
        self.assertEqual(
            calibration['tasks']['multiclass'],
            dict(largest_normal_variation=.002, smallest_weakened_loss=.3,
                 exact_normal_variation='1/500', exact_proposed_limit='1/100',
                 proposed_limit=.01))


if __name__ == '__main__':
    unittest.main()
