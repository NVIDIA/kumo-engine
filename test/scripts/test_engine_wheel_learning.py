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
          'packages/kumo-engine/test/wheel_learning.py')
LIMITS = dict(learnable_min=.75, independent_distance=.05, max_mean_drop=.02,
              max_seed_drop=.04)


class WheelLearningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = runpy.run_path(str(WORKER))

    def rows(self):
        return [
            dict(seed=0, case='learnable', auroc=.9),
            dict(seed=0, case='independent', auroc=.48),
            dict(seed=1, case='learnable', auroc=.8),
            dict(seed=1, case='independent', auroc=.52),
        ]

    def comparison(self):
        learning = dict(
            status='PASS', source_sha256='a' * 64, settings={'threads': 4},
            configuration={'python': '3.12'}, dependencies_sha256='b' * 64,
            seeds=[0, 1], runs=[
                dict(row, training_seed=row['seed'],
                     data_sha256=str(row['seed']), plan_sha256=str(row['seed']))
                for row in self.rows()
            ])
        for row in learning['runs']:
            if row['case'] == 'independent':
                self.set_control(row, .5)
        reference = dict(status='PASS', wheel=dict(version='1.0.0',
                                                   sha256='c' * 64),
                         learning=learning)
        baseline = {
            key: learning[key]
            for key in ('source_sha256', 'settings', 'configuration',
                        'dependencies_sha256', 'seeds')
        }
        baseline.update(reference_wheel=reference['wheel'], limits=LIMITS,
                        validation={'status': 'PASS'})
        return reference, copy.deepcopy(reference), baseline

    def measured_pair(self, index):
        a, b, _ = self.comparison()
        for side, report in [('a', a), ('b', b)]:
            report['status'] = report['learning']['status'] = 'MEASURED'
            report['learning']['run_id'] = f'run-{index}-{side}'
        return a, b

    def set_control(self, row, score):
        count = round(score * 200)
        row['auroc'] = score
        row['control'] = dict(
            targets=[False] * 200 + [True] * 200, probabilities=[1] *
            (200 - count) + [0] * count + [1] * count + [0] * (200 - count))

    def test_compare_accepts_matched_runs_in_any_order(self):
        compare = self.worker['compare_results']
        reference, candidate, baseline = self.comparison()
        candidate['wheel'] = dict(version='1.0.1', sha256='d' * 64)
        candidate['learning']['runs'].reverse()
        result = compare(reference, candidate, baseline)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['mean_auroc_drop'], 0)

    def test_compare_rejects_quality_drops_even_above_the_absolute_floor(self):
        compare = self.worker['compare_results']
        for scores in ([.86, .76], [.85, .85]):
            reference, candidate, baseline = self.comparison()
            for row, score in zip(candidate['learning']['runs'][::2], scores):
                row['auroc'] = score
            with self.subTest(scores=scores), self.assertRaisesRegex(
                    ValueError, 'drop'):
                compare(reference, candidate, baseline)

    def reporting(self):
        reporter = WORKER.with_name('wheel_learning_report.py')
        self.assertTrue(reporter.is_file(), 'The learning reporter is missing')
        return runpy.run_path(str(reporter))['report_comparison']

    def test_reporting_lists_all_missed_limits_without_approving_release(self):
        reference, candidate, baseline = self.comparison()
        baseline = copy.deepcopy(baseline)
        baseline['release_approved'] = True
        baseline['limits']['independent_distance'] = .01
        for result, values in ((reference, [.7, .6]), (candidate, [.5, .4])):
            result['status'] = result['learning']['status'] = 'MEASURED'
            for row, score in zip(result['learning']['runs'][::2], values):
                row['auroc'] = score
        for row in candidate['learning']['runs']:
            if row['case'] == 'independent':
                self.set_control(row, .53)
        candidate['learning']['runs'].reverse()
        result = self.reporting()(reference, candidate, baseline)
        self.assertEqual(result['status'], 'MEASURED')
        self.assertEqual(result['quality_status'], 'FAIL')
        self.assertFalse(result['release_approved'])
        self.assertAlmostEqual(result['mean_auroc']['reference']['learnable'],
                               .65)
        self.assertAlmostEqual(result['mean_auroc']['candidate']['learnable'],
                               .45)
        self.assertAlmostEqual(result['mean_auroc_drop'], .2)
        self.assertEqual(
            {(row['arm'], row['check'], row.get('seed'))
             for row in result['failed_limits']}, {
                 ('reference', 'learnable_min', None),
                 ('candidate', 'learnable_min', None),
                 ('candidate', 'independent_distance', None),
                 ('candidate', 'max_mean_drop', None),
                 ('candidate', 'max_seed_drop', 0),
                 ('candidate', 'max_seed_drop', 1),
             })
        self.assertEqual(result['limits'], baseline['limits'])

    def test_reporting_keeps_evidence_and_configuration_failures_strict(self):
        report = self.reporting()
        for field in ('source_sha256', 'dependencies_sha256', 'configuration',
                      'settings', 'seeds', 'status'):
            reference, candidate, baseline = self.comparison()
            for result in (reference, candidate):
                result['status'] = result['learning']['status'] = 'MEASURED'
            candidate['learning'][field] = None
            with self.subTest(field=field), self.assertRaises(
                (ValueError, TypeError)):
                report(reference, candidate, baseline)
        for field in ('training_seed', 'data_sha256', 'plan_sha256'):
            reference, candidate, baseline = self.comparison()
            for result in (reference, candidate):
                result['status'] = result['learning']['status'] = 'MEASURED'
            candidate['learning']['runs'][0][field] = 'different'
            with self.subTest(field=field), self.assertRaises(ValueError):
                report(reference, candidate, baseline)
        for value in (float('nan'), float('inf'), -1, 2):
            reference, candidate, baseline = self.comparison()
            for result in (reference, candidate):
                result['status'] = result['learning']['status'] = 'MEASURED'
            candidate['learning']['runs'][0]['auroc'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                report(reference, candidate, baseline)

    def test_reporting_does_not_bypass_the_chance_control(self):
        reference, candidate, baseline = self.comparison()
        for result in (reference, candidate):
            result['status'] = result['learning']['status'] = 'MEASURED'
        for row in candidate['learning']['runs']:
            if row['case'] == 'independent':
                self.set_control(row, .8 if row['seed'] == 0 else .2)
        with self.assertRaisesRegex(ValueError, 'independent'):
            self.reporting()(reference, candidate, baseline)

    def test_compare_rejects_unmatched_or_unfinished_evidence(self):
        compare = self.worker['compare_results']
        for field in ('source_sha256', 'configuration', 'dependencies_sha256',
                      'status', 'seeds'):
            reference, candidate, baseline = self.comparison()
            candidate['learning'][field] = None
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare(reference, candidate, baseline)
        for field in ('training_seed', 'data_sha256', 'plan_sha256'):
            reference, candidate, baseline = self.comparison()
            candidate['learning']['runs'][0][field] = 'different'
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare(reference, candidate, baseline)
        reference, candidate, baseline = self.comparison()
        reference['wheel'] = dict(version='other', sha256='e' * 64)
        with self.assertRaises(ValueError):
            compare(reference, candidate, baseline)
        reference, candidate, baseline = self.comparison()
        baseline['validation']['status'] = 'PENDING'
        with self.assertRaises(ValueError):
            compare(reference, candidate, baseline)

    def test_comparison_rejects_opposite_independent_associations(self):
        reference, candidate, baseline = self.comparison()
        for row in candidate['learning']['runs']:
            if row['case'] == 'independent':
                self.set_control(row, .8 if row['seed'] == 0 else .2)
        with self.assertRaisesRegex(ValueError, 'independent'):
            self.worker['compare_results'](reference, candidate, baseline)

    def test_calibration_uses_observed_repeats_instead_of_proposed_limits(self):
        calibrate = self.worker['calibrate']
        pairs = [self.measured_pair(0), self.measured_pair(1)]
        for _, repeat in pairs:
            for row in repeat['learning']['runs']:
                if row['case'] == 'learnable':
                    row['auroc'] -= .01
        baseline = calibrate(pairs)
        self.assertAlmostEqual(baseline['limits']['learnable_min'], .84)
        self.assertEqual(baseline['limits']['independent_distance'], 0)
        self.assertAlmostEqual(baseline['limits']['max_mean_drop'], .01)
        self.assertAlmostEqual(baseline['limits']['max_seed_drop'], .01)
        self.assertFalse(baseline['release_approved'])
        self.assertEqual(baseline['validation']['status'], 'PENDING')
        invalid = copy.deepcopy(pairs)
        invalid[1][0]['learning']['runs'].pop()
        with self.assertRaises(ValueError):
            calibrate(invalid)
        invalid = copy.deepcopy(pairs)
        invalid[1][0]['status'] = 'FAIL'
        with self.assertRaises(ValueError):
            calibrate(invalid)

    def test_calibration_rejects_copied_repeat_evidence(self):
        pair = self.measured_pair(0)
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.worker['calibrate']([pair, copy.deepcopy(pair)])

    def test_calibration_cannot_bless_predictable_independent_labels(self):
        pairs = [self.measured_pair(i) for i in range(10)]
        for scores in ((.8, .8), (.2, .2), (.8, .2)):
            for pair in pairs:
                for report in pair:
                    for row in report['learning']['runs']:
                        if row['case'] == 'independent':
                            self.set_control(row, scores[row['seed']])
            with self.subTest(scores=scores), self.assertRaisesRegex(
                    ValueError, 'independent'):
                self.worker['calibrate'](pairs)
        for pair in pairs:
            for report in pair:
                for row in report['learning']['runs']:
                    if row['case'] == 'independent':
                        self.set_control(row, .53)
        small = self.worker['calibrate'](pairs[:2])['chance_sanity']
        repeated = self.worker['calibrate'](pairs)['chance_sanity']
        self.assertEqual(small['p_value'], repeated['p_value'])
        self.assertGreater(small['p_value'], small['alpha'])

    def test_unlearned_reference_cannot_define_a_passing_baseline(self):
        pairs = [self.measured_pair(0), self.measured_pair(1)]
        for pair in pairs:
            for result in pair:
                for row in result['learning']['runs']:
                    row['auroc'] = .5
        with self.assertRaisesRegex(ValueError, 'learning'):
            self.worker['calibrate'](pairs)

    def test_calibration_rejects_inconsistent_control_evidence(self):
        for field in ('targets', 'probabilities', 'auroc'):
            pairs = [self.measured_pair(i) for i in range(2)]
            row = pairs[0][0]['learning']['runs'][1]
            if field == 'auroc':
                row['auroc'] = .7
            else:
                row['control'][field][0] = (True if field == 'targets' else
                                            float('nan'))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.worker['calibrate'](pairs)

    def test_invalid_limits_cannot_disable_the_quality_checks(self):
        reference, candidate, baseline = self.comparison()
        for value in (float('nan'), float('inf'), -1, 2):
            changed = copy.deepcopy(baseline)
            changed['limits']['max_mean_drop'] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, 'limit'):
                self.worker['compare_results'](reference, candidate, changed)

    def test_calibration_command_rejects_a_failed_reserved_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for group in ('calibration', 'validation'):
                for index in range(2):
                    for side, report in zip(
                        ('a', 'b'), self.measured_pair(f'{group}-{index}')):
                        folder = root / group / f'pair-{index}' / side
                        folder.mkdir(parents=True)
                        (folder / 'summary.json').write_text(json.dumps(report))
            for scenario in ('valid', 'overlap', 'failed'):
                failed = scenario != 'valid'
                if scenario == 'failed':
                    bad = root / 'validation/pair-0/b/summary.json'
                    report = json.loads(bad.read_text())
                    report['learning']['runs'][0]['auroc'] = .5
                    bad.write_text(json.dumps(report))
                output = root / f'baseline-{scenario}'
                completed = subprocess.run([
                    sys.executable,
                    str(WORKER), '--calibrate',
                    str(root / 'calibration'), '--validate-with',
                    str(root / ('calibration' if scenario == 'overlap' else
                                'validation')), '--output-dir',
                    str(output)
                ], capture_output=True, text=True)
                self.assertEqual(completed.returncode, 1 if failed else 0,
                                 completed.stderr)
                baseline = json.loads(
                    (output / 'learning-baseline.json').read_text())
                self.assertEqual(baseline['validation']['status'],
                                 'FAIL' if failed else 'PASS')
                self.assertFalse(baseline['release_approved'])

    def test_model_recipe_does_not_sample_training_choices(self):
        make_plan = self.worker['make_plan']
        plan = make_plan(17)
        self.assertEqual(plan.training_job.manual_seed, 17)
        sections = [
            plan.optimization, plan.neighbor_sampling, plan.model_architecture,
            plan.model_architecture.model[0]
        ]
        for section in sections:
            for name, value, metadata in section.items():
                if metadata.tunable and isinstance(value, list):
                    self.assertEqual(len(value), 1, name)
        for name in ('base_lr', 'weight_decay', 'lr_scheduler', 'optimizer'):
            self.assertIsInstance(getattr(plan.optimization, name), list)

    def test_selected_training_parameters_cannot_silently_change(self):
        verify = self.worker['verify_plan']
        requested = dict(
            optimization=dict(batch_size=[64], base_lr=[.01], unused='???'))
        actual = dict(
            optimization=dict(batch_size=[64], base_lr=[.01], unused=None))
        verify(requested, actual)
        for key, value in [('batch_size', [32]), ('base_lr', [.001])]:
            changed = copy.deepcopy(actual)
            changed['optimization'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                    ValueError, 'plan'):
                verify(requested, changed)

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
                    self.worker['summarize'](rows, [0, 1], LIMITS)

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
        # Calibration can round distinct scores into ties. Use saved values.
        tied = frame.copy()
        tied['True_PROB'] = [
            2.06115369216775e-9, 4.539787187241018e-5, 4.5397911890177056e-5, 1.
        ]
        tied['False_PROB'] = 1 - tied['True_PROB']
        self.assertEqual(score(tied, expected), 1.)
        tied['True_PROB'] = [0., 0., 0., 1.]
        tied['False_PROB'] = 1 - tied['True_PROB']
        self.assertEqual(score(tied, expected), .75)
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
