# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small CPU checks for failures that would invalidate a nightly result."""

import functools
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

SOURCE = Path(
    __file__).parents[2] / 'packages/kumo-engine/scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('nightly_check', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)


class NightlyRuntimeTest(unittest.TestCase):
    def test_declared_runtime_profile_does_not_require_pytest(self):
        profile = {'torch': '2.14.1', 'pytorch-lightning': '2.6.6'}
        nightly.check_dependencies(profile, profile)
        with self.assertRaises(ValueError):
            nightly.check_dependencies(dict(profile, pytest='8.0'), profile)

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

    def test_training_requires_real_fixed_work_and_valid_early_stop(self):
        state = dict(
            fits=1, completed_fits=1, core_fit_attempts=1, training_batches=80,
            optimizer_steps=80, training_epochs=[
                dict(epoch=i, batches=20, batch_sizes=[64] * 20)
                for i in range(4)
            ], early_stopping=dict(stopped_epoch=3, wait_count=3, patience=3))
        nightly.check_training(state, 40)
        nightly.check_training(dict(state, optimizer_steps=160), 120)
        nightly.check_training(
            dict(
                state, training_batches=200, optimizer_steps=400,
                training_epochs=[
                    dict(epoch=i, batches=20, batch_sizes=[64] * 20)
                    for i in range(10)
                ]), 400)
        for key, value in [('core_fit_attempts', 2), ('training_batches', 79),
                           ('early_stopping', {})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                nightly.check_training(dict(state, **{key: value}), 40)
        state['training_epochs'][0]['batch_sizes'][0] = 32
        with self.assertRaises(ValueError):
            nightly.check_training(state, 40)

    def test_effective_recipe_changes_are_not_hidden_by_requested_plan(self):
        config = dict(
            root_dir='/old', version=0, kumo_version='1.0.0',
            refit_cfg=dict(best_val_epoch=3, refitted_full=False,
                           refitted_trainval=False),
            loader_kwargs={'batch_size':
                           64}, sampler_kwargs={'adaptive_sampling':
                                                False}, loss='mae')
        self.assertEqual(
            nightly.stable_trainer(config),
            dict(loader_kwargs={'batch_size': 64},
                 sampler_kwargs={'adaptive_sampling': False}, loss='mae'))
        changed = dict(config, loss='mse')
        self.assertNotEqual(nightly.signature(nightly.stable_trainer(config)),
                            nightly.signature(nightly.stable_trainer(changed)))

    def test_core_retry_is_blocked_before_a_second_fit_and_eval_does_not_retry(
            self):
        calls = []

        def retry(function):
            @functools.wraps(function)
            def wrapped(*args):
                try:
                    return function(*args)
                except RuntimeError:
                    return function(*args)

            return wrapped

        class Core:
            def fit(self):
                calls.append('fit')

            @retry
            def predict_v2(self):
                calls.append('predict')
                raise RuntimeError('out of memory')

            test = predict_v2

        class Data:
            def _pseudo_time_options(self, stage):
                return {}

        modules = {
            'diskgraph':
            SimpleNamespace(execute_tt_ingest=lambda: None),
            'kumo_core.train.trainer':
            SimpleNamespace(Trainer=Core),
            'kumo_core.diskgraph.lightning_data':
            SimpleNamespace(DiskgraphLightningDataModule=Data)
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, modules):
            state = dict(core_fit_attempts=0, training_batches=20,
                         optimizer_steps=40,
                         ledger_path=str(Path(directory) / 'core.json'))
            with nightly.native_calls(state):
                core = Core()
                core.fit()
                with self.assertRaises(ValueError):
                    core.fit()
                with self.assertRaises(RuntimeError):
                    core.predict_v2()
            self.assertEqual(calls, ['fit', 'predict'])
            self.assertEqual(len(state['core_fit_ledger']), 2)

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

    def test_behavior_needs_actual_child_work_positive_fits_and_retained_logs(
            self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            behavior = root / 'behavior'
            child_path = behavior / 'pytest-temp/case/child-result.json'
            nightly.save(
                child_path,
                dict(fits=0, fit_attempts=0, state_checks=2, restored_tensors=3,
                     prediction_batches={'reuse': 1}, rows=1071))
            for name in ('child.stdout', 'child.stderr', 'request.json'):
                child_path.with_name(name).write_text('{}')
            (behavior / 'junit.xml').write_text(
                '<testsuites><testsuite><testcase name="a"/></testsuite></testsuites>'
            )
            artifacts = {
                str(p.relative_to(behavior)): nightly.digest(p)
                for p in behavior.rglob('*') if p.is_file()
            }
            result = dict(
                status='PASS', coverage={
                    k: True
                    for k in ('I2', 'B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B8')
                }, pytest_exit_code=0, bad=[], deselected=0,
                collection_errors=[], collected=['test.py::a'],
                calls={'test.py::a': 'passed'}, physical_fits=1,
                fit_ledger=[dict(status='PASS', steps=2,
                                 cuda_batches=1)], artifacts=artifacts,
                b7_receipt=dict(path=str(child_path.relative_to(behavior)),
                                sha256=nightly.digest(child_path)))
            nightly.save(behavior / 'result.json', result)
            nightly.check_behavior(root)
            nightly.save(behavior / 'result.json',
                         dict(result, physical_fits=0, fit_ledger=[]))
            with self.assertRaises(ValueError):
                nightly.check_behavior(root)
            nightly.save(behavior / 'result.json', result)
            child = json.loads(child_path.read_text())
            nightly.save(child_path, dict(child, restored_tensors=0))
            result['artifacts'][str(
                child_path.relative_to(behavior))] = nightly.digest(child_path)
            result['b7_receipt']['sha256'] = nightly.digest(child_path)
            nightly.save(behavior / 'result.json', result)
            with self.assertRaises(ValueError):
                nightly.check_behavior(root)
            child_path.with_name('child.stdout').unlink()
            with self.assertRaises(ValueError):
                nightly.check_behavior(root)

    def test_complete_roster_rejects_reused_fit_and_partial_result(self):
        rows = [
            dict(
                task=task, arm=arm, seed=0, status='MEASURED', physical_fits=1,
                fit_id=f'{task}-{arm}', checkpoint_sha256='a' * 64,
                prediction_batches={
                    s: 1
                    for s in ('predict', 'repeat', 'partial', 'reload')
                }) for task in nightly.smoke.TASKS
            for arm in ('normal', 'control')
        ]
        nightly.check_results(rows)
        invalid = [
            rows[:-1], rows + [rows[0]],
            [dict(row, fit_id=rows[0]['fit_id']) for row in rows]
        ]
        for bad in invalid:
            with self.assertRaises(ValueError):
                nightly.check_results(bad)


if __name__ == '__main__':
    unittest.main()
