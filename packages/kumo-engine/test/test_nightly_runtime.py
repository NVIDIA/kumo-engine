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
    def test_resource_capture_keeps_cpu_tree_and_gpu_measures_separate(self):
        import torch
        from unittest.mock import patch

        with patch.object(torch.cuda, 'mem_get_info', return_value=(1000, 9000)), \
                patch.object(torch.cuda, 'memory_allocated', return_value=2000), \
                patch.object(torch.cuda, 'memory_reserved', return_value=3000):
            values, finish = nightly.start_measurement('training')
            finish()
        nightly.validate_measurement(values)
        self.assertGreaterEqual(values['samples'], 2)
        self.assertGreater(values['cpu_tree_pss_sampled_peak_bytes'], 0)
        self.assertGreaterEqual(values['cpu_tree_rss_sampled_peak_bytes'],
                                values['cpu_tree_pss_sampled_peak_bytes'])
        self.assertEqual(values['cuda_device_used_sampled_peak_bytes'], 8000)
        self.assertEqual(values['torch_allocated_sampled_peak_bytes'], 2000)
        self.assertEqual(values['torch_reserved_sampled_peak_bytes'], 3000)
        self.assertGreater(values['elapsed_seconds'], 0)

    def test_missing_gpu_measurement_invalidates_the_measurement(self):
        import torch
        from unittest.mock import patch

        with patch.object(torch.cuda, 'mem_get_info',
                          side_effect=RuntimeError('GPU query failed')):
            values, finish = nightly.start_measurement('training')
            finish()
        self.assertIsNone(values['cuda_device_used_sampled_peak_bytes'])
        with self.assertRaisesRegex(ValueError, 'measurement'):
            nightly.validate_measurement(values)

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

    def test_ranked_sort_spills_without_losing_or_reordering_rows(self):
        import numpy as np
        import pytest
        from unittest.mock import patch

        duckdb = pytest.importorskip('duckdb')
        connect = duckdb.connect
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'ranked.parquet'
            with connect(config={'threads': 4}) as db:
                db.sql("SELECT i AS ROW_ID, "
                       "(hash(i)%100000000000)::BIGINT AS ENTITY, "
                       "(i%500)::BIGINT AS CLASS, "
                       "(hash(i+2)%1000000)::DOUBLE / 1000000 AS SCORE "
                       "FROM range(8000000) t(i)").write_parquet(str(source))
            seen = np.zeros(8000000, dtype=bool)
            count, previous = 0, None
            # Reproduce the worker's 48-core default with a real DuckDB query.
            with patch.object(duckdb, 'connect', side_effect=lambda *, config:
                              connect(config={'threads': 48, **config})):
                for frame in nightly.ordered_batches([str(source)],
                                                     root / 'spill'):
                    pd.testing.assert_frame_equal(
                        frame, frame.sort_values(
                            ['ENTITY', 'SCORE', 'CLASS'],
                            ascending=[True, False, True]))
                    first = (int(frame.ENTITY.iloc[0]),
                             -float(frame.SCORE.iloc[0]),
                             int(frame.CLASS.iloc[0]))
                    if previous is not None:
                        self.assertLessEqual(previous, first)
                    previous = (int(frame.ENTITY.iloc[-1]),
                                -float(frame.SCORE.iloc[-1]),
                                int(frame.CLASS.iloc[-1]))
                    positions = frame.ROW_ID.to_numpy()
                    self.assertTrue(((positions >= 0) &
                                     (positions < len(seen))).all())
                    seen[positions] = True
                    count += len(frame)
            self.assertEqual(count, 8000000)
            self.assertTrue(seen.all())

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

    def test_nightly_numeric_reporting_keeps_the_default_pr_bound_strict(self):
        with self.assertRaises(ValueError):
            nightly.smoke.compare_predictions([[.5]], [[.501]], atol=1e-4)
        compare = nightly.compare_predictions
        result = compare([[.5]], [[.501]], atol=0,
                         before_decisions=[True], after_decisions=[True])
        self.assertFalse(result['within_tolerance'])
        self.assertAlmostEqual(result['max_abs_difference'], .001)
        for after, decisions in (([[.501]], [False]),
                                 ([[float('nan')]], [True])):
            with self.assertRaises(ValueError):
                compare([[.5]], after,
                        before_decisions=[True], after_decisions=decisions)

    def test_nightly_public_callback_keeps_exact_restored_state_proof(self):
        import pytorch_lightning as pl
        import torch

        model = pl.LightningModule()
        model.layer = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.ckpt'
            torch.save({'state_dict': model.state_dict()}, checkpoint)
            state = dict(stage='reload', reload_checks=0, restored_tensors=0,
                         checkpoint_sha256=nightly.digest(checkpoint))
            with nightly.observe_cuda(state):
                trainer = pl.Trainer(accelerator='cpu', devices=1,
                                     logger=False, enable_checkpointing=False,
                                     enable_model_summary=False)
                trainer.ckpt_path = str(checkpoint)
                [proof] = [cb for cb in trainer.callbacks
                           if type(cb).__module__ == nightly.__name__]
                proof.on_predict_start(trainer, model)
                self.assertEqual(state['reload_checks'], 1)
                self.assertEqual(state['restored_tensors'], 2)
                with torch.no_grad():
                    model.layer.weight.add_(1)
                with self.assertRaisesRegex(ValueError, 'tensor'):
                    proof.on_predict_start(trainer, model)


if __name__ == '__main__':
    unittest.main()
