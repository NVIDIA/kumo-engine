# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import runpy
import sys
import unittest
from pathlib import Path

WORKER = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/wheel_node_tasks.py')


class WheelNodeTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if WORKER.exists():
            sys.path.insert(0, str(WORKER.parent))
            cls.worker = runpy.run_path(str(WORKER))

    def worker_function(self, name):
        self.assertTrue(WORKER.is_file(), 'The node learning worker is missing')
        self.assertIn(name, self.worker.keys(),
                      f'Missing node learning check: {name}')
        return self.worker[name]

    def test_control_keeps_graph_splits_and_each_split_target_distribution(
            self):
        make_data = self.worker_function('make_data')
        for task in ('multiclass', 'multilabel', 'multilabel_ranking',
                     'regression'):
            for seed in range(5):
                with self.subTest(task=task, seed=seed):
                    known, split, classes = make_data(task, seed, False)
                    control, other_split, other_classes = make_data(
                        task, seed, True)
                    self.assertEqual(split, other_split)
                    self.assertEqual(classes, other_classes)
                    self.assertEqual(known['ID'], control['ID'])
                    self.assertEqual(known['GROUP_ID'], control['GROUP_ID'])
                    self.assertNotEqual(known['LABEL'], control['LABEL'])
                    for stage, count in ((0, 1200), (1, 400), (2, 400)):
                        before = [
                            str(y) for y, s in zip(known['LABEL'], split)
                            if s == stage
                        ]
                        after = [
                            str(y) for y, s in zip(control['LABEL'], split)
                            if s == stage
                        ]
                        self.assertEqual(len(before), count)
                        self.assertEqual(collections.Counter(before),
                                         collections.Counter(after))
                    self.assertEqual((known, split, classes),
                                     make_data(task, seed, False))

    def test_baseline_uses_only_training_answers(self):
        baseline = self.worker_function('baseline_predictions')
        self.assertEqual(
            baseline('multiclass', [[1, 0], [1, 0], [0, 1], [0, 1]],
                     [0, 0, 0, 2], 2), [[1, 0], [1, 0]])
        for task in ('multilabel', 'multilabel_ranking'):
            self.assertEqual(
                baseline(task, [[1, 0], [0, 1], [1, 1]], [0, 0, 2], 2),
                [[.5, .5], [.5, .5]])
        self.assertEqual(
            baseline('regression', [[1], [3], [100]], [0, 0, 2], 2), [[2], [2]])

    def test_multiclass_matrix_uses_entity_and_class_identity(self):
        import numpy as np
        import pandas as pd
        matrix = self.worker_function('prediction_matrix')
        frame = pd.DataFrame({
            'ENTITY': [2, 1, 2, 1, 2, 1],
            'CLASS': ['c', 'b', 'a', 'a', 'b', 'c'],
            'SCORE': [.7, .3, .2, .6, .1, .1],
            'PREDICTED': [True, False, False, True, False, False],
        })
        np.testing.assert_array_equal(
            matrix('multiclass', frame, [1, 2], ['a', 'b', 'c']),
            [[.6, .3, .1], [.2, .1, .7]])
        for broken in (
                frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]),
                frame.assign(ENTITY=[3, 1, 3, 1, 3, 1]),
                frame.assign(CLASS=['x', 'b', 'a', 'a', 'b', 'c']),
                frame.assign(
                    SCORE=[.7, .3, .2, .6, .1, float('nan')]),
                frame.assign(PREDICTED=False)):
            with self.subTest(broken=broken.to_dict()), self.assertRaises(
                (ValueError, AssertionError)):
                matrix('multiclass', broken, [1, 2], ['a', 'b', 'c'])

    def test_multilabel_scores_are_not_normalized_across_labels(self):
        import numpy as np
        import pandas as pd
        matrix = self.worker_function('prediction_matrix')
        frame = pd.DataFrame({
            'ENTITY': [1, 1],
            'CLASS': ['a', 'b'],
            'SCORE': [.8, .7],
            'PREDICTED': [True, True],
        })
        np.testing.assert_array_equal(
            matrix('multilabel', frame, [1], ['a', 'b']), [[.8, .7]])
        with self.assertRaises((ValueError, AssertionError)):
            matrix('multilabel', frame.assign(PREDICTED=False), [1], ['a', 'b'])

    def test_regression_matrix_rejects_missing_duplicate_and_nonfinite_rows(
            self):
        import numpy as np
        import pandas as pd
        matrix = self.worker_function('prediction_matrix')
        frame = pd.DataFrame({'ENTITY': [2, 1], 'TARGET_PRED': [4., -1.]})
        np.testing.assert_array_equal(matrix('regression', frame, [1, 2], []),
                                      [[-1.], [4.]])
        for broken in (frame.iloc[:1], pd.concat([frame, frame.iloc[:1]]),
                       frame.assign(TARGET_PRED=[float('inf'), 1])):
            with self.subTest(broken=broken.to_dict()), self.assertRaises(
                (ValueError, AssertionError)):
                matrix('regression', broken, [1, 2], [])

    def test_hidden_answer_check_rejects_target_features(self):
        check = self.worker_function('verify_hidden_answers')
        schema = {
            'node_types': {
                'entities': {
                    'feature_columns': ['GROUP_ID']
                },
                'groups': {
                    'feature_columns': ['RISK']
                },
            }
        }
        check(schema)
        schema['node_types']['entities']['feature_columns'].append('LABEL')
        with self.assertRaises((ValueError, AssertionError)):
            check(schema)

    def test_holdout_rejects_inconsistent_answers_in_repeated_class_rows(self):
        import pandas as pd
        check = self.worker_function('verify_holdout_answers')
        frame = pd.DataFrame({
            'ENTITY': [1, 1, 2, 2],
            'TARGET': ['a', 'a', 'b', 'b'],
            'CLASS': ['a', 'b', 'a', 'b'],
        })
        check('multiclass', frame, [1, 2], ['a', 'b'], [[1, 0], [0, 1]])
        for targets in (['a', 'b', 'b', 'b'], ['a', 'a', 'a', 'a']):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                check('multiclass', frame.assign(TARGET=targets), [1, 2],
                      ['a', 'b'], [[1, 0], [0, 1]])

    def test_unknown_task_is_rejected_before_data_generation(self):
        with self.assertRaises(ValueError):
            self.worker_function('make_data')('unknown', 0, False)

    def test_task_plans_only_request_supported_public_options(self):
        from kumoapi.model_plan import MissingType
        from kumoapi.pquery.utils import QueryType
        from kumoapi.task import TaskType
        make_plan = self.worker_function('make_plan')
        for task, task_type in (('multiclass',
                                 TaskType.MULTICLASS_CLASSIFICATION),
                                ('multilabel',
                                 TaskType.MULTILABEL_CLASSIFICATION),
                                ('multilabel_ranking',
                                 TaskType.MULTILABEL_RANKING),
                                ('regression', TaskType.REGRESSION)):
            plan = make_plan(task, 0)
            for section in ('training_job', 'optimization', 'column_processing',
                            'neighbor_sampling', 'model_architecture'):
                part = getattr(plan, section)
                for name, value, metadata in part.items():
                    if value != MissingType.VALUE:
                        with self.subTest(task=task, section=section,
                                          name=name):
                            self.assertTrue(
                                part.is_valid_option(name, metadata, task_type,
                                                     QueryType.STATIC))

    def test_multilabel_seed_strings_keep_complete_sets_in_one_entity_row(self):
        convert = self.worker_function('training_targets')
        targets = [['a', 'b', 'g'], ['c', 'd', 'i']]
        for task in ('multilabel', 'multilabel_ranking'):
            self.assertEqual(convert(task, targets), ['a|b|g', 'c|d|i'])
        self.assertEqual(convert('multiclass', ['a', 'b']), ['a', 'b'])
        self.assertEqual(convert('regression', [1.25, 2.5]), [1.25, 2.5])

    def test_regression_truth_survives_target_tensor_float32_roundtrip(self):
        import numpy as np
        values, _, _ = self.worker_function('make_data')('regression', 0, False)
        np.testing.assert_array_equal(
            values['LABEL'], np.asarray(values['LABEL'], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
