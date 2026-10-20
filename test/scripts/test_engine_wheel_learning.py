# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import runpy
import unittest
from pathlib import Path

WORKER = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/wheel_learning.py')


class WheelLearningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = runpy.run_path(str(WORKER))

    def rows(self):
        return [
            {
                'seed': 0,
                'case': 'learnable',
                'auroc': .9
            },
            {
                'seed': 0,
                'case': 'independent',
                'auroc': .48
            },
            {
                'seed': 1,
                'case': 'learnable',
                'auroc': .8
            },
            {
                'seed': 1,
                'case': 'independent',
                'auroc': .52
            },
        ]

    def test_reports_each_case_separately(self):
        result = self.worker['summarize'](self.rows(), [0, 1])
        self.assertEqual(set(result), {'learnable', 'independent'})
        self.assertAlmostEqual(result['learnable'], .85)
        self.assertAlmostEqual(result['independent'], .5)

    def test_constant_guesses_and_leaked_answers_fail_the_pair(self):
        for case, score, reason in [('learnable', .5, 'learnable'),
                                    ('independent', 1.0, 'independent')]:
            with self.subTest(case=case):
                rows = self.rows()
                for row in rows:
                    if row['case'] == case:
                        row['auroc'] = score
                with self.assertRaisesRegex(ValueError, reason):
                    self.worker['summarize'](rows, [0, 1])

    def test_missing_or_duplicate_seed_case_cannot_pass(self):
        rows = self.rows()
        for invalid in (rows[:-1], rows[:-1] + [copy.copy(rows[0])]):
            with self.assertRaisesRegex(ValueError, 'seed'):
                self.worker['summarize'](invalid, [0, 1])

    def test_invalid_scores_cannot_pass(self):
        for score in (float('nan'), float('inf'), -1.0, 2.0):
            with self.subTest(score=score):
                rows = self.rows()
                rows[0]['auroc'] = score
                with self.assertRaisesRegex(ValueError, 'score'):
                    self.worker['summarize'](rows, [0, 1])

    def test_signal_removal_preserves_inputs_and_splits(self):
        signal, split = self.worker['create_data'](0, False)
        independent, other_split = self.worker['create_data'](0, True)
        self.assertIn('ID', signal)
        self.assertEqual(signal['ID'], independent['ID'])
        self.assertEqual(signal['GROUP_ID'], independent['GROUP_ID'])
        self.assertEqual(split, other_split)
        self.assertNotEqual(signal['LABEL'], independent['LABEL'])
        self.assertEqual([split.count(stage) for stage in (0, 1, 2)],
                         [1200, 400, 400])
        for upper, low, high in ((True, .8, 1.0), (False, 0.0, .2)):
            labels = [
                label
                for group, label in zip(signal['GROUP_ID'], signal['LABEL'])
                if (group < 10) == upper
            ]
            self.assertGreater(sum(labels) / len(labels), low)
            self.assertLess(sum(labels) / len(labels), high)
        for upper in (True, False):
            labels = [
                label for group, label in zip(independent['GROUP_ID'],
                                              independent['LABEL'])
                if (group < 10) == upper
            ]
            self.assertGreater(sum(labels) / len(labels), .4)
            self.assertLess(sum(labels) / len(labels), .6)

    def test_scores_probabilities_and_rejects_wrong_evaluation_rows(self):
        import pandas as pd

        frame = pd.DataFrame({
            'ENTITY': [0, 1, 2, 3],
            'TARGET': [False, False, True, True],
            'False_PROB': [.9, .6, .65, .2],
            'True_PROB': [.1, .4, .35, .8],
        })
        expected = {0: False, 1: False, 2: True, 3: True}
        score = self.worker['score_predictions']
        self.assertEqual(score(frame, expected), .75)
        wrong_label = frame.copy()
        wrong_label.loc[0, 'TARGET'] = True
        wrong_probability = frame.copy()
        wrong_probability.loc[0, 'True_PROB'] = float('nan')
        duplicate = frame.copy()
        duplicate.loc[3, 'ENTITY'] = 2
        for invalid in (frame.iloc[:-1], wrong_label, wrong_probability,
                        duplicate):
            with self.assertRaises(AssertionError):
                score(invalid, expected)


if __name__ == '__main__':
    unittest.main()
