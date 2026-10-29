# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import runpy
import unittest
from pathlib import Path

import numpy as np

CHECKS = Path(__file__).resolve().parents[2] / (
    'packages/kumo-engine/test/wheel_task_checks.py')


class WheelTaskChecksTest(unittest.TestCase):
    def helpers(self):
        self.assertTrue(CHECKS.is_file(), 'Shared task checks are missing')
        return runpy.run_path(str(CHECKS))

    def test_calibration_reduces_only_the_real_training_budget(self):
        helpers = self.helpers()
        self.assertIn('apply_training_budget', helpers,
                      'Calibration must change the actual training plan')
        apply_budget = helpers['apply_training_budget']
        for filename, tasks in [
            ('wheel_node_tasks.py',
             ['multiclass', 'multilabel', 'multilabel_ranking', 'regression']),
            ('wheel_temporal_tasks.py',
             ['forecasting', 'static_link', 'temporal_link']),
        ]:
            make_plan = runpy.run_path(str(
                CHECKS.with_name(filename)))['make_plan']
            for task in tasks:
                with self.subTest(task=task):
                    plan = make_plan(task, 3)
                    original = plan.as_json()
                    self.assertIsNone(apply_budget(plan, None))
                    self.assertEqual(plan.as_json(), original)
                    fault = apply_budget(plan, [1, 2])
                    expected = copy.deepcopy(original)
                    expected['optimization'].update(max_epochs=1,
                                                    min_steps_per_epoch=2,
                                                    max_steps_per_epoch=2)
                    self.assertEqual(plan.as_json(), expected)
                    self.assertEqual(
                        fault,
                        dict(kind='reduced_training_budget', budget=[1, 2],
                             original_plan=original))

    def test_invalid_or_unchanged_calibration_budget_cannot_mutate_the_plan(
            self):
        helpers = self.helpers()
        self.assertIn('apply_training_budget', helpers,
                      'Calibration must validate its training budget')
        make_plan = runpy.run_path(str(
            CHECKS.with_name('wheel_node_tasks.py')))['make_plan']
        for budget in ([], [1], [1, 2, 3], [True, 2], [1, False], [1.0, 2],
                       [0, 2], [1, -1], [11, 2], [1, 21], [10, 20], '1,2', {
                           'epochs': 1,
                           'steps': 2
                       }):
            plan = make_plan('regression', 0)
            original = plan.as_json()
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                helpers['apply_training_budget'](plan, budget)
            self.assertEqual(plan.as_json(), original)

    def test_metrics_score_full_outputs_against_known_answers(self):
        score = self.helpers()['score']
        self.assertEqual(
            score('accuracy', [[1, 0, 0], [0, 1, 0]],
                  [[.8, .1, .1], [.7, .2, .1]]), .5)
        # Label 0 has AP 1; label 1 has AP (1/2 + 2/3)/2 = 7/12.
        self.assertAlmostEqual(
            score('ap_macro', [[1, 1], [1, 0], [0, 1]],
                  [[.9, .1], [.8, .9], [.1, .8]]), 19 / 24)
        self.assertEqual(
            score('recall@5', [[0, 0, 0, 0, 1, 1]], [[6, 5, 4, 3, 2, 1]]), .5)
        self.assertEqual(score('mae', [[2, 4], [6, 8]], [[1, 5], [4, 8]]), 1)

    def test_metrics_reject_missing_scores_and_invalid_answers(self):
        score = self.helpers()['score']
        for metric, targets, predictions in [
            ('accuracy', [], []),
            ('accuracy', [[1, 0]], [[.5]]),
            ('accuracy', [[1, 1]], [[.5, .5]]),
            ('accuracy', [[1, -1]], [[.5, .5]]),
            ('ap_macro', [[1, 2]], [[.5, .5]]),
            ('ap_macro', [[1, 0], [0, 0]], [[.5, .5], [.5, .5]]),
            ('recall@5', [[0, 0, 0, 0, 0]], [[1, 2, 3, 4, 5]]),
            ('recall@5', [[1, 0, 0, 0]], [[1, 2, 3, 4]]),
            ('mae', [[1], [2]], [[1, 2]]),
            ('mae', [[1]], [[float('nan')]]),
            ('mae', [[float('inf')]], [[1]]),
            ('mae', [[1], [2, 3]], [[1], [2]]),
            ('unknown', [[1]], [[1]]),
        ]:
            with self.subTest(metric=metric,
                              targets=targets), self.assertRaises(ValueError):
                score(metric, targets, predictions)

    def test_prediction_ids_cannot_hide_missing_duplicate_or_reordered_rows(
            self):
        verify = self.helpers()['verify_predictions']
        self.assertEqual(
            verify([10, 20], [10, 20], [[1], [2]]).tolist(), [[1], [2]])
        for expected, actual, values in [
            ([10, 20], [20, 10], [[1], [2]]),
            ([10, 20], [10, 10], [[1], [2]]),
            ([10, 20], [10], [[1]]),
            ([10, 20], [10, 30], [[1], [2]]),
            ([10, 10], [10, 10], [[1], [2]]),
            ([10, 20], [10, 20], [[1]]),
            ([10], [10], [[float('nan')]]),
            ([], [], []),
        ]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                verify(expected, actual, values)

    def test_reload_checks_shape_finite_values_and_tolerance(self):
        verify = self.helpers()['verify_reload']
        self.assertLess(verify([[.5, 1]], [[.500001, 1.000001]]), 1.1e-6)
        for values in ([[.50001, 1]], [[.5]], [[float('nan'), 1]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                verify([[.5, 1]], values)

    def test_fingerprint_ignores_mapping_order_and_rejects_nonfinite_inputs(
            self):
        fingerprint = self.helpers()['fingerprint']
        self.assertEqual(fingerprint({
            'a': 1,
            'b': [2]
        }), fingerprint({
            'b': [2],
            'a': 1
        }))
        self.assertNotEqual(fingerprint({'a': 1}), fingerprint({'a': 2}))
        with self.assertRaises(ValueError):
            fingerprint({'a': float('nan')})

    def arms(self, *, suspicious=False, metric='accuracy'):
        targets = [[1, 0], [0, 1]] * 30
        predictions = targets if suspicious else [[.5, .5]] * 60
        if metric == 'recall@5':
            targets = [[1] * 6 + [0] * 6, [0] * 6 + [1] * 6] * 30
            predictions = targets if suspicious else [[.5] * 12] * 60
        if metric == 'mae':
            targets = [[i % 2] for i in range(60)]
            predictions = targets if suspicious else [[.5]] * 60
        runs = [
            dict(task='task', seed=seed, case='independent', metric=metric,
                 targets=targets, predictions=predictions) for seed in [0, 1]
        ]
        return [runs, copy.deepcopy(runs)]

    def test_controls_accept_constant_predictions_and_share_shuffles(self):
        verify = self.helpers()['independent_controls']
        result = verify(self.arms())
        self.assertEqual(result['task']['p_value'], 1)
        self.assertAlmostEqual(result['task']['alpha'], .01 / 7)
        self.assertEqual(result, verify(self.arms()))
        # Duplicating an arm must not produce another independent draw.
        arms = self.arms()
        for arm in arms:
            for row in arm:
                row['predictions'] = [[.9, .1], [.1, .9], [.1, .9], [.9, .1]
                                      ] * 15
        result = verify(arms)
        self.assertEqual(result, verify(arms + [copy.deepcopy(arms[0])]))

    def test_controls_reject_leaked_answers_for_higher_and_lower_metrics(self):
        verify = self.helpers()['independent_controls']
        for metric in ['accuracy', 'ap_macro', 'recall@5', 'mae']:
            with self.subTest(metric=metric), self.assertRaisesRegex(
                    ValueError, 'independent'):
                verify(self.arms(suspicious=True, metric=metric))

    def masked_arms(self, suspicious=False):
        targets = [[int(column == row % 12) for column in range(12)]
                   for row in range(120)]
        eligible = [[column // 6 == (row % 12) // 6 for column in range(12)]
                    for row in range(120)]
        predictions = targets if suspicious else eligible
        rows = [
            dict(task='static_link', seed=seed, case='independent',
                 metric='recall@5', targets=targets, predictions=predictions,
                 control_eligible=eligible) for seed in [0, 1]
        ]
        return [rows, copy.deepcopy(rows)]

    def test_static_link_controls_condition_on_each_entities_unknown_items(
            self):
        verify = self.helpers()['independent_controls']
        try:
            result = verify(self.masked_arms())
        except ValueError as error:
            self.fail(f'Valid masked chance predictions were rejected: {error}')
        self.assertGreater(result['static_link']['p_value'], .01 / 7)
        arms = self.masked_arms()
        self.assertEqual(result, verify(arms + [copy.deepcopy(arms[0])]))
        with self.assertRaisesRegex(ValueError, 'independent'):
            verify(self.masked_arms(suspicious=True))

    def test_static_link_null_preserves_each_rows_positive_count(self):
        verify = self.helpers()['independent_controls']
        arms = self.masked_arms()
        for arm in arms:
            for row in arm:
                row.update(targets=[[1, 1, 0, 0, 0,
                                     1]], predictions=[[6, 5, 4, 3, 2, 1]],
                           control_eligible=[[True] * 6])
        # Every three-positive null set has recall >= 2/3 in a six-item catalog.
        self.assertEqual(verify(arms)['static_link']['p_value'], 1)

    def test_static_link_controls_reject_invalid_or_unmatched_eligibility(self):
        verify = self.helpers()['independent_controls']
        for mutation in ['outside', 'missing', 'shape', 'type', 'metric']:
            arms = self.masked_arms()
            if mutation == 'outside':
                for arm in arms:
                    arm[0]['control_eligible'] = [[False] * 12] * 120
            elif mutation == 'missing':
                del arms[1][0]['control_eligible']
            elif mutation == 'shape':
                arms[1][0]['control_eligible'] = [[True]] * 120
            elif mutation == 'type':
                for arm in arms:
                    arm[0]['control_eligible'] = [[1] * 12] * 120
            else:
                for arm in arms:
                    for row in arm:
                        row['metric'] = 'accuracy'
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    ValueError, 'eligible'):
                verify(arms)

    def test_controls_reject_unmatched_missing_or_invalid_evidence(self):
        verify = self.helpers()['independent_controls']
        for mutation in [
                'missing', 'duplicate', 'targets', 'metric', 'scores',
                'seed_metric'
        ]:
            arms = self.arms()
            if mutation == 'missing':
                arms[1].pop()
            elif mutation == 'duplicate':
                arms[1].append(copy.deepcopy(arms[1][0]))
            elif mutation == 'targets':
                arms[1][0]['targets'] = list(reversed(arms[1][0]['targets']))
            elif mutation == 'metric':
                arms[1][0]['metric'] = 'mae'
            elif mutation == 'seed_metric':
                for arm in arms:
                    arm[1]['metric'] = 'mae'
            else:
                arms[1][0]['predictions'] = [[float('nan'), .5]] * 60
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify(arms)

    def test_ranked_link_recall_keeps_full_positive_denominator_and_strict_ids(
            self):
        score = self.helpers()['ranked_link_score']
        row = dict(task='static_link', seed=0, case='learnable',
                   metric='recall@5', target_format='ranked_link_v1',
                   entity_ids=[7], catalog=list(range(7)),
                   target_ids=[list(range(7))], excluded_ids=[[]],
                   ranked_ids=[[0, 1, 2, 3, 4]], ranked_scores=[[1.] * 5])
        self.assertEqual(score(row), 5 / 7)
        for key, value in [('ranked_ids', [[0, 1, 2, 3, 3]]),
                           ('ranked_ids', [[0, 1, 2, 3, 4.]]),
                           ('ranked_ids', [[0, 1, 2, 3]]),
                           ('ranked_ids', [[0, 1, 2, 3, 9]]),
                           ('ranked_ids', [[1, 0, 2, 3, 4]]),
                           ('ranked_scores', [[float('nan')] * 5])]:
            with self.subTest(key=key,
                              value=value), self.assertRaises(ValueError):
                score(dict(row, **{key: value}))

    def test_ranked_static_null_reuses_uniform_subset_law_and_shared_arms(self):
        from itertools import combinations
        from math import comb

        from scipy.stats import multivariate_hypergeom

        helpers = self.helpers()
        colors, masks = helpers['ranked_control_groups']([[[0, 1]], [[1, 2]]],
                                                         [5])
        for selected in combinations(range(5), 2):
            counts = [
                int(0 in selected),
                int(1 in selected),
                int(2 in selected),
                sum(item in selected for item in (3, 4))
            ]
            # Two outside items can yield the same group counts.
            multiplicity = comb(2, counts[-1])
            self.assertAlmostEqual(
                multivariate_hypergeom.pmf(counts, colors[0], 2),
                multiplicity / comb(5, 2))
        self.assertEqual(
            masks.tolist(),
            [[True, False], [True, True], [False, True], [False, False]])
        arms = self.masked_arms()
        for arm in arms:
            for row in arm:
                values = np.asarray(row.pop('predictions'), dtype=float)
                row.update(
                    target_format='ranked_link_v1',
                    entity_ids=list(range(len(values))),
                    catalog=list(range(values.shape[1])), target_ids=[
                        np.flatnonzero(y).tolist() for y in row.pop('targets')
                    ], excluded_ids=[
                        np.flatnonzero(~np.asarray(mask)).tolist()
                        for mask in row.pop('control_eligible')
                    ], ranked_ids=np.argsort(-values, axis=1,
                                             kind='stable')[:, :5].tolist(),
                    ranked_scores=[[1.] * 5] * len(values))
        verify = helpers['independent_controls']
        result = verify(arms)
        self.assertGreater(result['static_link']['p_value'], .01 / 7)
        self.assertEqual(result, verify(arms + [copy.deepcopy(arms[0])]))


if __name__ == '__main__':
    unittest.main()
