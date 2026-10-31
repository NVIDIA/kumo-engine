# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small CPU checks for failures that would invalidate a nightly result."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('nightly_check', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)
APPROVED = '1bc15ecb7696bce216b05aa883d6897d2b8ce4342c6c2f9d62b7cc28b314539e'


class NightlyRuntimeTest(unittest.TestCase):
    def rows(self):
        return [
            dict(task=task, arm=arm, seed=0, status='MEASURED', physical_fits=1,
                 fit_id=f'{task}-{arm}', checkpoint_sha256='a' * 64,
                 prediction_batches={stage: 1
                                     for stage in nightly.PREDICTIONS})
            for task in nightly.smoke.TASKS for arm in ('normal', 'control')
        ]

    def test_one_report_requires_all_sixteen_cases_and_rejects_two_reports(
            self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = dict(format='kumo-engine-nightly-result-v1',
                          status='MEASURED', input_manifest_sha256='f' * 64,
                          expected_runs=16, completed_runs=16, runs=self.rows())
            path = root / 'nightly-result.json'
            nightly.save(path, report)
            self.assertEqual(nightly.load_report(root, 'f' * 64),
                             (path, report))
            for bad in [
                    dict(report, completed_runs=15),
                    dict(report, runs=report['runs'][:-1]),
                    dict(report, shard='node'),
                    dict(report, status='FAIL'),
                    dict(
                        report, runs=[
                            dict(row, fit_id='reused') for row in report['runs']
                        ]),
                    dict(
                        report, runs=[
                            dict(row, status='FAIL') if i == 3 else row
                            for i, row in enumerate(report['runs'])
                        ])
            ]:
                nightly.save(path, bad)
                with self.assertRaises(ValueError):
                    nightly.load_report(root, 'f' * 64)
            nightly.save(path, report)
            nightly.save(root / 'old/nightly-result.json', report)
            with self.assertRaises(ValueError):
                nightly.load_report(root, 'f' * 64)
            path.unlink()
            (root / 'old/nightly-result.json').unlink()
            with self.assertRaises(ValueError):
                nightly.load_report(root, 'f' * 64)

    def test_frozen_files_reject_escape_symlink_and_modified_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'rows').write_bytes(b'original')
            files = {'rows': nightly.digest(root / 'rows')}
            self.assertEqual(nightly.safe_file(root, 'rows', files),
                             root / 'rows')
            for name in ('../rows', '/rows', './rows'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    nightly.safe_file(root, name, files)
            (root / 'link').symlink_to(root / 'rows')
            with self.assertRaises(ValueError):
                nightly.safe_file(root, 'link', {'link': files['rows']})
            (root / 'rows').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                nightly.safe_file(root, 'rows', files)

    def test_ranked_boundary_and_exclusions_across_batches(self):
        frame = pd.DataFrame(
            dict(ENTITY=[1] * 4 + [2] * 4, CLASS=[10, 11, 12, 13] * 2,
                 SCORE=[.9, .8, .7, .1, .9, .8, .7, .1]))
        result = nightly.ranked_rows(
            [frame.iloc[:3], frame.iloc[3:6], frame.iloc[6:]], [1, 2],
            [10, 11, 12, 13, 14], [[10], []], 2, 4)
        self.assertEqual(result['ranked_ids'], [[11, 12], [10, 11]])
        self.assertEqual(result['raw_rows'], 8)
        for bad in (frame.iloc[:-1], frame.assign(SCORE=.5),
                    frame.assign(CLASS=[10] * 8),
                    frame.assign(ENTITY=frame.ENTITY.astype(float))):
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                nightly.ranked_rows([bad], [1, 2], [10, 11, 12, 13, 14],
                                    [[10], []], 2, 4)

    def test_no_full_catalog_retry_when_five_hundred_is_insufficient(self):
        frame = pd.DataFrame(
            dict(ENTITY=[1] * 500, CLASS=list(range(500)),
                 SCORE=list(range(500, 0, -1))))
        with self.assertRaises(ValueError):
            nightly.ranked_rows([frame], [1], list(range(600)),
                                [list(range(499))], 5, 500)

    def test_reported_recall_uses_all_positives_not_only_k(self):
        truth = pd.DataFrame({'TARGET1': [[1, 2, 3, 4, 5, 6]]})
        score = nightly.quality('static_link', truth,
                                {'ranked_ids': [[1, 2, 3, 4, 5]]}, [])
        self.assertAlmostEqual(score, 5 / 6)

    def test_partial_inputs_are_the_same_rows_and_a_proper_subset(self):
        truth = pd.DataFrame(
            dict(ENTITY1=[2, 1],
                 TIMESTAMP=pd.to_datetime(['2024-01-01',
                                           '2024-01-01']), TARGET1=[1, 0]))
        inputs = truth[['ENTITY1', 'TIMESTAMP']]
        self.assertEqual(
            nightly.check_inputs(inputs.iloc[:1], truth, partial=True), [2])
        for bad in [
                inputs,
                inputs.assign(ENTITY1=[3, 4]),
                inputs.assign(TIMESTAMP=pd.Timestamp('2024-01-02'))
        ]:
            with self.assertRaises(ValueError):
                nightly.check_inputs(bad, truth, partial=True)
        with self.assertRaises(ValueError):
            nightly.check_inputs(inputs[['ENTITY1']], truth)

    def test_missing_or_changed_retained_artifact_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'raw.parquet').write_bytes(b'raw')
            artifacts = {'raw.parquet': nightly.digest(root / 'raw.parquet')}
            nightly.verify_artifacts(root, artifacts)
            with self.assertRaises(ValueError):
                nightly.verify_artifacts(root, artifacts,
                                         required=['missing-output.parquet'])
            (root / 'raw.parquet').unlink()
            with self.assertRaises(ValueError):
                nightly.verify_artifacts(root, artifacts)

    def test_static_inputs_accept_frozen_timestamps_and_reject_changes(self):
        truth = pd.DataFrame(dict(
            ENTITY1=[2, 1], TIMESTAMP=pd.to_datetime(['1970-01-01'] * 2),
            TARGET1=[[10], [11]]))
        inputs = truth[['ENTITY1', 'TIMESTAMP']]
        self.assertEqual(nightly.check_inputs(inputs, truth, static=True),
                         [2, 1])
        self.assertEqual(nightly.check_inputs(
            inputs.iloc[1:], truth, static=True, partial=True), [1])
        self.assertEqual(nightly.check_inputs(
            inputs[['ENTITY1']], truth, static=True), [2, 1])
        for bad in (inputs.assign(TIMESTAMP=pd.Timestamp('1970-01-02')),
                    truth):
            with self.subTest(columns=list(bad)), self.assertRaises(ValueError):
                nightly.check_inputs(bad, truth, static=True)

    def test_complete_roster_rejects_reused_fit_and_partial_result(self):
        rows = self.rows()
        nightly.check_results(rows)
        invalid = [
            rows[:-1], rows + [rows[0]],
            [dict(row, seed=False) for row in rows],
            [dict(row, fit_id=rows[0]['fit_id']) for row in rows],
            [
                dict(row, status='FAIL') if i == 3 else row
                for i, row in enumerate(rows)
            ]
        ]
        for bad in invalid:
            with self.assertRaises(ValueError):
                nightly.check_results(bad)


if __name__ == '__main__':
    unittest.main()
