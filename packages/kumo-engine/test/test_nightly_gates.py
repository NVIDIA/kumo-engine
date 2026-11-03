# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise learning and prediction failures without fitting a model."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('nightly_gates', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)


class LearningAndPredictions(unittest.TestCase):
    def test_a_scrambled_model_cannot_match_normal_inside_the_chance_band(self):
        root = SOURCE.parents[1] / 'test/nightly'
        rows = nightly.read(root / 'reference.json')['cases']
        for row in rows:
            if row['task'] == 'forecasting':
                row['score'] = .335
                row['baseline'] = .3822739835549215
        policy = nightly.read(root / 'policy.json')['learning']
        checked = nightly.check_learning(rows, policy)['tasks']['forecasting']
        self.assertEqual(checked['status'], 'FAIL')

    def test_reference_calibration_meets_all_task_learning_checks(self):
        root = SOURCE.parents[1] / 'test/nightly'
        reference = nightly.read(root / 'reference.json')
        policy = nightly.read(root / 'policy.json')['learning']
        measured = nightly.check_learning(reference['cases'], policy)
        self.assertEqual(measured['status'], 'PASS')
        historical = copy.deepcopy(reference['cases'])
        next(row for row in historical if row['task'] == 'static_link'
             and row['arm'] == 'normal')['score'] = .007665
        self.assertEqual(
            nightly.check_learning(historical,
                                   policy)['tasks']['static_link']['status'],
            'FAIL')
        for task in policy:
            for fault in ('nonlearning', 'leaked_control'):
                rows = copy.deepcopy(reference['cases'])
                normal = next(r for r in rows
                              if r['task'] == task and r['arm'] == 'normal')
                control = next(r for r in rows
                               if r['task'] == task and r['arm'] == 'control')
                control['score'] = control['baseline']
                if fault == 'nonlearning':
                    normal['score'] = normal['baseline']
                else:
                    if task == 'static_link':
                        normal['score'] = .02
                    control['score'] = normal['score']
                with self.subTest(task=task, fault=fault):
                    self.assertEqual(
                        nightly.check_learning(rows,
                                               policy)['tasks'][task]['status'],
                        'FAIL')

    def test_policy_requires_all_tasks_and_an_exact_reviewed_checksum(self):
        value = dict(
            format='kumo-engine-nightly-policy-v1', learning={
                task: dict(min_normal_gain=.01, max_control_drift=.005)
                for task in nightly.smoke.TASKS
            }, predictions={
                task: dict(atol=.0004)
                for task in nightly.smoke.TASKS
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            path.write_text(json.dumps(value))
            expected = nightly.digest(path)
            self.assertEqual(nightly.load_policy(path, expected), value)
            path.write_text(json.dumps(value) + '\n')
            with self.assertRaises(ValueError):
                nightly.load_policy(path, expected)
            value['learning']['binary']['min_normal_gain'] = 0
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                nightly.load_policy(path, nightly.digest(path))
            del value['learning']['binary']
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                nightly.load_policy(path, nightly.digest(path))

    def test_complete_summary_propagates_learning_failures(self):
        rows, reference = [], {'cases': []}
        for task in nightly.smoke.TASKS:
            lower = task in ('regression', 'forecasting')
            baseline = 1. if lower else .5
            for arm in ('normal', 'control'):
                rows.append(
                    dict(
                        task=task, arm=arm, seed=0, status='MEASURED',
                        physical_fits=1, fit_id=f'{task}-{arm}',
                        checkpoint_sha256='a' * 64, seconds=1.,
                        prediction_checks=dict(status='PASS'),
                        prediction_batches={
                            stage: 1
                            for stage in nightly.PREDICTIONS
                        }, score=baseline
                        if arm == 'control' else .7 if lower else .8))
                reference['cases'].append(
                    dict(task=task, arm=arm, baseline=baseline, score=baseline))
        policy = dict(
            learning={
                task: dict(min_normal_gain=.1, max_control_drift=.02)
                for task in nightly.smoke.TASKS
            })
        self.assertEqual(
            nightly.summarize(rows, reference, policy)['status'], 'PASS')
        rows[0]['prediction_checks']['status'] = 'FAIL'
        self.assertEqual(
            nightly.summarize(rows, reference, policy)['status'], 'FAIL')
        rows[0]['prediction_checks']['status'] = 'PASS'
        rows[0]['score'] = .5
        self.assertEqual(
            nightly.summarize(rows, reference, policy)['status'], 'FAIL')
        with self.assertRaises(ValueError):
            nightly.summarize(rows[:-1], reference, policy)

    def test_learning_rejects_no_gain_and_leaked_controls(self):
        bounds = {'binary': {'min_normal_gain': .1, 'max_control_drift': .02}}
        for normal, control, expected in [(.8, .5, 'PASS'), (.5, .5, 'FAIL'),
                                          (.8, .8, 'FAIL'), (.8, .2, 'FAIL')]:
            with self.subTest(normal=normal, control=control):
                rows = [
                    dict(task='binary', arm=arm, score=score, baseline=.5)
                    for arm, score in [('normal', normal), ('control', control)]
                ]
                result = nightly.check_learning(rows, bounds)
                self.assertEqual(result['status'], expected)
                self.assertEqual(result['tasks']['binary']['status'], expected)

    def test_lower_error_and_small_learning_gaps_use_task_specific_bounds(self):
        rows = [
            dict(task='regression', arm='normal', score=.7, baseline=1.),
            dict(task='regression', arm='control', score=1.01, baseline=1.)
        ]
        result = nightly.check_learning(
            rows,
            {'regression': {
                'min_normal_gain': .2,
                'max_control_drift': .02
            }})
        self.assertEqual(result['status'], 'PASS')
        rows[0]['score'] = .9
        self.assertEqual(
            nightly.check_learning(rows, {
                'regression': {
                    'min_normal_gain': .2,
                    'max_control_drift': .02
                }
            })['status'], 'FAIL')

    def test_learning_never_approves_a_missing_or_nonpositive_margin(self):
        rows = [
            dict(task='binary', arm=arm, score=score, baseline=.5)
            for arm, score in [('normal', .8), ('control', .5)]
        ]
        for bounds in [{}, {
                'min_normal_gain': 0,
                'max_control_drift': .02
        }, {
                'min_normal_gain': .1,
                'max_control_drift': float('nan')
        }]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                nightly.check_learning(rows, {'binary': bounds})

    def test_ranks_only_change_between_close_scores(self):
        before = dict(ranked_ids=[[1, 2, 3]], ranked_scores=[[.9, .5001, .5]])
        after = dict(ranked_ids=[[1, 3, 2]], ranked_scores=[[.9, .5001, .5]])
        result = nightly.ranked_difference(before, after, atol=.0004)
        self.assertEqual(result['status'], 'PASS')
        for damaged in [
                dict(ranked_ids=[[2, 1, 3]], ranked_scores=[[.9, .5001, .5]]),
                dict(ranked_ids=[[1, 3, 4]], ranked_scores=[[.9, .5001, .5]])
        ]:
            with self.subTest(damaged=damaged), self.assertRaises(ValueError):
                nightly.ranked_difference(before, damaged, atol=.0004)

    def test_rank_measurement_retains_failed_score_and_inversion_bounds(self):
        before = dict(ranked_ids=[[1, 2]], ranked_scores=[[.9, .5]])
        after = dict(ranked_ids=[[2, 1]], ranked_scores=[[.9, .5]])
        result = nightly.ranked_difference(before, after, atol=.0004,
                                           measure_only=True)
        self.assertEqual(result['status'], 'FAIL')
        self.assertAlmostEqual(result['max_abs_aligned_score_difference'], .4)
        self.assertAlmostEqual(result['max_inverted_score_gap'], .4)
        with self.assertRaises(ValueError):
            nightly.ranked_difference(before, dict(after, ranked_ids=[[2, 2]]),
                                      atol=.0004, measure_only=True)

    def test_rank_measurement_reports_missing_cutoff_proof_as_failure(self):
        before = dict(ranked_ids=[[1, 2]], ranked_scores=[[.5, .5]],
                      boundary_ids=[[1, 2, 5]], boundary_scores=[[.5, .5, .5]])
        after = dict(ranked_ids=[[3, 4]], ranked_scores=[[.5, .5]],
                     boundary_ids=[[3, 4, 6]], boundary_scores=[[.5, .5, .5]])
        result = nightly.ranked_difference(before, after, atol=.0004,
                                           measure_only=True)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['unproven_membership_changes'], 1)
        with self.assertRaises(ValueError):
            nightly.ranked_difference(before, after, atol=.0004)

    def test_multiple_cutoff_swaps_use_all_retained_boundary_coordinates(self):
        before = dict(ranked_ids=[[1, 2]], ranked_scores=[[.5001, .5]],
                      boundary_ids=[[1, 2, 3, 4]],
                      boundary_scores=[[.5001, .5, .4999, .4998]])
        after = dict(ranked_ids=[[3, 4]], ranked_scores=[[.5001, .5]],
                     boundary_ids=[[3, 4, 1, 2]],
                     boundary_scores=[[.5001, .5, .4999, .4998]])
        self.assertEqual(
            nightly.ranked_difference(before, after, atol=.0004)['status'],
            'PASS')

    def test_rank_cutoff_changes_need_both_sides_of_the_boundary(self):
        before = dict(ranked_ids=[[1, 2]], ranked_scores=[[.9, .5001]],
                      boundary_ids=[[1, 2,
                                     3]], boundary_scores=[[.9, .5001, .5]])
        after = dict(ranked_ids=[[1, 3]], ranked_scores=[[.9, .5001]],
                     boundary_ids=[[1, 3, 2]], boundary_scores=[[.9, .5001,
                                                                 .5]])
        result = nightly.ranked_difference(before, after, atol=.0004)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['added_items'], 1)
        self.assertEqual(result['removed_items'], 1)
        for damage in [
                dict(after, boundary_ids=[[1, 3, 4]]),
                dict(after, boundary_scores=[[.9, .5001, .4]])
        ]:
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                nightly.ranked_difference(before, damage, atol=.0004)

    def test_installation_binds_one_checked_runtime_to_the_build_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nightly.save(
                root / 'setup/build.json',
                dict(revision='a' * 40, tree='b' * 40, version='1.2.0+abc',
                     wheel_sha256='c' * 64))
            runtime = dict(version='1.2.0+abc', wheel_sha256='c' * 64,
                           installed_files=10)
            nightly.check_install(root, runtime)
            for bad in [
                    dict(runtime, installed_files=0),
                    dict(runtime, wheel_sha256='d' * 64)
            ]:
                with self.assertRaises(ValueError):
                    nightly.check_install(root, bad)


if __name__ == '__main__':
    unittest.main()
