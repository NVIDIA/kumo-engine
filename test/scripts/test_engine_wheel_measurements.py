# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
import sys
import tempfile
import time
import unittest
from pathlib import Path

WORKER = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/wheel_measurements.py')


class WheelMeasurementsTest(unittest.TestCase):
    def test_gpu_ranking_measurement_keeps_cutoff_changes_unqualified(self):
        import pandas as pd

        worker = runpy.run_path(str(WORKER))
        self.assertIn('measure_recommendations', worker)
        before = pd.DataFrame({'ENTITY': [1, 1], 'CLASS': ['a', 'b'],
                               'SCORE': [0.2, 0.1]})
        after = pd.DataFrame({'ENTITY': [1, 1], 'CLASS': ['a', 'c'],
                              'SCORE': [0.200001, 0.100001]})
        measured = worker['measure_recommendations'](before, after)
        self.assertFalse(measured['qualified'])
        self.assertEqual(measured['common_recommendations'], 1)
        self.assertEqual(measured['removed_recommendations'], 1)
        self.assertEqual(measured['added_recommendations'], 1)
        self.assertEqual(measured['changed_entities'], 1)
        self.assertAlmostEqual(measured['max_abs_difference'], 0.000001)
        with self.assertRaises(ValueError):
            worker['measure_recommendations'](before, after.assign(ENTITY=2))

    def test_fixed_training_can_continue_after_the_timed_window(self):
        worker = runpy.run_path(str(WORKER))
        self.assertIn('verify_training_work', worker)
        window = worker['TrainingWindow'](lambda: None)
        for batch in range(880):
            window.batch_start()
            window.batch_end(optimizer_steps=(batch + 1) * 2)
        work = [dict(completed_batches=880, optimizer_steps=1760,
                     early_stopped=False)]
        verify = worker['verify_training_work']
        verify(work, window.summary(), expected_batches=880)
        for broken in ([dict(work[0], completed_batches=220)],
                       [dict(work[0], early_stopped=True)], work * 2):
            with self.assertRaises(ValueError):
                verify(broken, window.summary(), expected_batches=880)

    def test_loader_profile_rejects_automatic_batch_or_worker_changes(self):
        worker = runpy.run_path(str(WORKER))
        self.assertIn('verify_loader_profile', worker)
        verify = worker['verify_loader_profile']
        requested = {'batch_size': 256, 'workers': 8}
        observed = {'train_batch_size': 256, 'eval_batch_size': 256,
                    'workers': 8}
        verify(observed, **requested)
        for broken in (dict(observed, train_batch_size=128),
                       dict(observed, eval_batch_size=128),
                       dict(observed, workers=16), {}):
            with self.assertRaises(ValueError):
                verify(broken, **requested)

    def test_training_window_counts_batches_with_two_optimizer_updates(self):
        worker = runpy.run_path(str(WORKER))
        self.assertTrue(callable(worker.get('TrainingWindow')),
                        'Batch-based timing window is missing')
        window = worker['TrainingWindow'](lambda: None)
        for batch in range(220):
            window.batch_start()
            window.batch_end(optimizer_steps=(batch + 1) * 2)
        result = window.summary()
        self.assertEqual(result['completed_batches'], 220)
        self.assertEqual(result['optimizer_steps'], 440)
        self.assertEqual(result['warmup_batches'], 20)
        self.assertEqual(result['timed_batches'], 200)
        self.assertGreater(result['seconds'], 0)

    def test_prediction_population_rejects_wrong_ids_and_incomplete_top_k(self):
        import pandas as pd

        sys.path.insert(0, str(WORKER.parent))
        try:
            worker = runpy.run_path(str(WORKER.with_name('wheel_customer_learning.py')))
        finally:
            sys.path.pop(0)
        self.assertTrue(callable(worker.get('verify_population')),
                        'Prediction population validation is missing')
        verify = worker['verify_population']
        frame = pd.DataFrame({'ENTITY': [1, 1, 2, 2], 'CLASS': ['a', 'b', 'a', 'b']})
        verify(frame, [1, 2], top_k=2)
        for broken in (frame.iloc[:-1], frame.assign(ENTITY=[1, 1, 3, 3]),
                       pd.concat([frame, frame.iloc[:1]])):
            with self.assertRaises(ValueError):
                verify(broken, [1, 2], top_k=2)
        verify(frame, [1, 2], top_k=2, class_ids=['a', 'b'])
        with self.assertRaises(ValueError):
            verify(frame.assign(CLASS=['a', 'z', 'a', 'b']), [1, 2],
                   top_k=2, class_ids=['a', 'b'])

    def test_real_graph_recipe_uses_local_sources_and_target_time_ranges(self):
        sys.path.insert(0, str(WORKER.parent))
        try:
            worker = runpy.run_path(str(WORKER.with_name('wheel_customer_learning.py')))
        finally:
            sys.path.pop(0)
        self.assertIn('local_graph_definition', worker)
        payload = {'train_pquery': {'graph': {'tables': {
            'orders': {'pkey': None, 'time_col': 'ORDER_TIME',
                       'cols': [{'name': 'ID', 'stype': 'ID'}],
                       'source_table': {'s3_path': 's3://original/orders'}}},
            'col_groups': []}}}
        result = worker['local_graph_definition'](payload)
        self.assertEqual(result['tables']['orders']['source_table'], {'path': 'orders'})
        self.assertEqual(result['tables']['orders']['time_col'], 'ORDER_TIME')
        self.assertEqual(payload['train_pquery']['graph']['tables']['orders']['source_table'],
                         {'s3_path': 's3://original/orders'})
        ranges = worker['target_time_ranges']('2026-05-18')
        self.assertEqual(ranges, [['2026-04-04', '2026-05-04'],
                                  ['2026-05-04', '2026-05-11'],
                                  ['2026-05-11', '2026-05-18']])

    def test_customer_scores_use_all_reserved_answers_and_reject_missing_ids(self):
        import pandas as pd

        sys.path.insert(0, str(WORKER.parent))
        try:
            worker = runpy.run_path(str(WORKER.with_name('wheel_customer_learning.py')))
        finally:
            sys.path.pop(0)
        self.assertIn('score_customer', worker)
        score = worker['score_customer']
        stamp = pd.Timestamp('2026-01-01')
        answers = pd.DataFrame({'ENTITY1': [1, 2, 3, 4],
                                'TIMESTAMP': [stamp] * 4,
                                'TARGET1': [True, False, True, False]})
        frame = pd.DataFrame({'ENTITY': [4, 3, 2, 1],
                              'TIMESTAMP': [stamp] * 4,
                              'False_PROB': [0.8, 0.4, 0.3, 0.1],
                              'True_PROB': [0.2, 0.6, 0.7, 0.9]})
        self.assertEqual(score('affirm', frame, answers), 0.75)
        self.assertEqual(score('affirm', frame.drop(columns='TIMESTAMP'), answers), 0.75)
        ambiguous = pd.concat([answers, answers.assign(TIMESTAMP=stamp + pd.Timedelta(days=1))])
        with self.assertRaises(ValueError):
            score('affirm', frame.drop(columns='TIMESTAMP'), ambiguous)
        with self.assertRaises(ValueError):
            score('affirm', frame.iloc[1:], answers)
        with self.assertRaises((ValueError, AssertionError)):
            score('affirm', frame.assign(False_PROB=0.8), answers)
        answers = pd.DataFrame({'ENTITY1': [1, 2], 'TIMESTAMP': [stamp] * 2,
                                'TARGET1': [['a', 'b'], ['c']]})
        frame = pd.DataFrame({'ENTITY': [1, 1, 2], 'TIMESTAMP': [stamp] * 3,
                              'CLASS': ['a', 'x', 'c'], 'SCORE': [0.9, 0.8, 0.7]})
        self.assertEqual(score('sothebys', frame, answers), 0.75)
        with self.assertRaises(ValueError):
            score('sothebys', pd.concat([frame, frame.iloc[:1]]), answers)

    def test_stage_records_elapsed_work_and_preserves_failed_stage(self):
        self.assertTrue(WORKER.is_file(), 'Stage measurement helper is missing')
        worker = runpy.run_path(str(WORKER))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            measurements = worker['Measurements'](output)
            with measurements.stage('table'):
                time.sleep(0.01)
            with self.assertRaisesRegex(ValueError, 'broken'):
                with measurements.stage('fit'):
                    raise ValueError('broken')
            saved = json.loads((output / 'measurements.json').read_text())
            self.assertEqual([row['stage'] for row in saved], ['table', 'fit'])
            self.assertGreaterEqual(saved[0]['seconds'], 0.01)
            self.assertEqual(saved[0]['status'], 'COMPLETE')
            self.assertEqual(saved[1]['status'], 'FAIL')

    def test_binary_split_rejects_single_class_before_training(self):
        self.assertTrue(WORKER.is_file(), 'Stage measurement helper is missing')
        worker = runpy.run_path(str(WORKER))
        verify = worker['binary_split_counts']
        self.assertEqual(verify([False, True, False]),
                         {'rows': 3, 'positive': 1, 'negative': 2})
        for values in ([], [False, False], [True], [False, None, True],
                       [False, 2, True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                verify(values)


if __name__ == '__main__':
    unittest.main()
