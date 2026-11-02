# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small CPU checks for failures that would invalidate a nightly result."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('nightly_check', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)
APPROVED = '1bc15ecb7696bce216b05aa883d6897d2b8ce4342c6c2f9d62b7cc28b314539e'


class NightlyRuntimeTest(unittest.TestCase):
    def test_observer_keeps_default_kernels_after_cuda_initialization(self):
        import pytorch_lightning as pl
        import torch

        previous = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, previous)
        torch.use_deterministic_algorithms(False)
        state = dict(stage='predict')
        with patch.object(torch.cuda, 'is_initialized', return_value=True), \
                nightly.observe_cuda(state):
            pl.Trainer(accelerator='cpu', devices=1, logger=False,
                       enable_checkpointing=False)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())

    def rows(self):
        return [
            dict(task=task, arm=arm, seed=0, status='MEASURED', physical_fits=1,
                 fit_id=f'{task}-{arm}', checkpoint_sha256='a' * 64,
                 prediction_batches={stage: 1
                                     for stage in nightly.PREDICTIONS})
            for task in nightly.smoke.TASKS for arm in ('normal', 'control')
        ]

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
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'ranked.parquet'
            schema = pa.schema([('ROW_ID', pa.int64()), ('ENTITY', pa.int64()),
                                ('CLASS', pa.int64()), ('SCORE', pa.float64())])
            with pq.ParquetWriter(source, schema) as writer:
                for start in range(0, 8000000, 65536):
                    rows = np.arange(start, min(start + 65536, 8000000))
                    writer.write_table(
                        pa.Table.from_arrays([
                            rows, (rows * 1103515245 + 12345) % 100000000000,
                            rows % 500, (rows * 48271 % 1000000) / 1000000
                        ], schema=schema))
            seen = np.zeros(8000000, dtype=bool)
            count, previous = 0, None
            for frame in nightly.ordered_batches([str(source)], root / 'spill'):
                pd.testing.assert_frame_equal(
                    frame,
                    frame.sort_values(['ENTITY', 'SCORE', 'CLASS'],
                                      ascending=[True, False, True]))
                first = (int(frame.ENTITY.iloc[0]), -float(frame.SCORE.iloc[0]),
                         int(frame.CLASS.iloc[0]))
                if previous is not None:
                    self.assertLessEqual(previous, first)
                previous = (int(frame.ENTITY.iloc[-1]),
                            -float(frame.SCORE.iloc[-1]),
                            int(frame.CLASS.iloc[-1]))
                positions = frame.ROW_ID.to_numpy()
                self.assertTrue(
                    ((positions >= 0) & (positions < len(seen))).all())
                seen[positions] = True
                count += len(frame)
            self.assertEqual(count, 8000000)
            self.assertTrue(seen.all())
            self.assertEqual(list((root / 'spill').iterdir()), [])

    def test_parquet_merge_keeps_large_keys_and_exact_anchors_and_closes_work(
            self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame(
                dict(
                    ENTITY=pd.Series(
                        [2**63 + 3, 2**63 + 1, 2**63 + 3, 2**63 + 1],
                        dtype='uint64'), CLASS=[4, 9, 2,
                                                3], SCORE=[.4, .9, .4, .8],
                    ROW_ID=[0, 1, 2, 3], ANCHOR_TIMESTAMP=pd.to_datetime([
                        '2024-01-01 00:00:00.000000001',
                        '2024-01-01 00:00:00.000000002',
                        '2024-01-01 00:00:00.000000003',
                        '2024-01-01 00:00:00.000000004'
                    ])))
            paths = [root / 'a.parquet', root / 'b.parquet']
            frame.iloc[:2].to_parquet(paths[0], index=False)
            frame.iloc[2:].to_parquet(paths[1], index=False)
            batches = nightly.ordered_batches([p.as_uri() for p in paths],
                                              root / 'sort')
            try:
                actual = next(batches)
                self.assertEqual(actual.ROW_ID.tolist(), [1, 3, 2, 0])
                self.assertEqual(actual.ENTITY.tolist(),
                                 [2**63 + 1, 2**63 + 1, 2**63 + 3, 2**63 + 3])
                self.assertEqual(actual.ENTITY.dtype, frame.ENTITY.dtype)
                self.assertEqual(
                    actual.ANCHOR_TIMESTAMP.tolist(),
                    frame.ANCHOR_TIMESTAMP.iloc[[1, 3, 2, 0]].tolist())
            finally:
                batches.close()
            self.assertEqual(list((root / 'sort').iterdir()), [])

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

    def test_static_inputs_accept_frozen_timestamps_and_reject_changes(self):
        truth = pd.DataFrame(
            dict(ENTITY1=[2, 1], TIMESTAMP=pd.to_datetime(['1970-01-01'] * 2),
                 TARGET1=[[10], [11]]))
        inputs = truth[['ENTITY1', 'TIMESTAMP']]
        self.assertEqual(nightly.check_inputs(inputs, truth, static=True),
                         [2, 1])
        self.assertEqual(
            nightly.check_inputs(inputs.iloc[1:], truth, static=True,
                                 partial=True), [1])
        self.assertEqual(
            nightly.check_inputs(inputs[['ENTITY1']], truth, static=True),
            [2, 1])
        for bad in (inputs.assign(TIMESTAMP=pd.Timestamp('1970-01-02')), truth):
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

    def test_nightly_public_callback_keeps_exact_restored_state_proof(self):
        import pytorch_lightning as pl
        import torch

        self.addCleanup(
            torch.use_deterministic_algorithms,
            torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        model = pl.LightningModule()
        model.layer = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.ckpt'
            torch.save({'state_dict': model.state_dict()}, checkpoint)
            state = dict(stage='reload', reload_checks=0, restored_tensors=0,
                         checkpoint_sha256=nightly.digest(checkpoint))
            with nightly.observe_cuda(state):
                trainer = pl.Trainer(accelerator='cpu', devices=1, logger=False,
                                     enable_checkpointing=False,
                                     enable_model_summary=False)
                trainer.ckpt_path = str(checkpoint)
                [proof] = [
                    cb for cb in trainer.callbacks
                    if type(cb).__module__ == nightly.__name__
                ]
                proof.on_predict_start(trainer, model)
                self.assertEqual(state['reload_checks'], 1)
                self.assertEqual(state['restored_tensors'], 2)
                with torch.no_grad():
                    model.layer.weight.add_(1)
                with self.assertRaisesRegex(ValueError, 'tensor'):
                    proof.on_predict_start(trainer, model)


if __name__ == '__main__':
    unittest.main()
