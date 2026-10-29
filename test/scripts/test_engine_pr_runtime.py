# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'packages/kumo-engine/scripts/pr_check.py'


def module():
    spec = importlib.util.spec_from_file_location('pr_check', SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class RuntimeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'),
                         'Requires the checkpoint runtime')
    def test_install_workflow_checks_the_actual_restored_checkpoint(self):
        import torch

        workflow = runpy.run_path(
            str(ROOT / 'packages/kumo-engine/test/wheel_workflow.py'))
        self.assertIn('attach_reload_identity', workflow)
        for mutation in (None, 'restored_weight', 'checkpoint_bytes'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
            ) as directory:
                model = torch.nn.Linear(2, 1)
                checkpoint = Path(directory) / 'model.ckpt'
                torch.save({'state_dict': model.state_dict()}, checkpoint)
                result = dict(expected_reload_checkpoint_sha256=hashlib.sha256(
                    checkpoint.read_bytes()).hexdigest())

                def restore(payload):
                    model.load_state_dict(payload['state_dict'])
                    if mutation == 'restored_weight':
                        with torch.no_grad():
                            model.weight.add_(1)
                    return 'restored'

                strategy = SimpleNamespace(load_model_state_dict=restore,
                                           lightning_module=model)
                trainer = SimpleNamespace(pl_trainer=SimpleNamespace(
                    strategy=strategy, ckpt_path=str(checkpoint)))
                workflow['attach_reload_identity'](trainer, result)
                payload = {'state_dict': model.state_dict()}
                if mutation == 'checkpoint_bytes':
                    checkpoint.write_bytes(b'different-checkpoint')
                if mutation:
                    with self.assertRaisesRegex(ValueError,
                                                'tensor|fingerprint'):
                        strategy.load_model_state_dict(payload)
                    self.assertNotIn('reload_identity_checks', result)
                else:
                    self.assertEqual(strategy.load_model_state_dict(payload),
                                     'restored')
                    [receipt] = result['reload_identity_checks']
                    self.assertEqual(receipt['status'], 'PASS')
                    self.assertEqual(receipt['tensor_count'], 2)
                    self.assertEqual(
                        receipt['checkpoint_sha256'],
                        result['expected_reload_checkpoint_sha256'])
                    self.assertEqual(
                        receipt['restore_path'],
                        'lightning_strategy.load_model_state_dict')

    @pytest.mark.agent_local
    def test_build_receipt_rejects_wrong_revision_and_dirty_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)

            def git(*args):
                return subprocess.check_output(
                    ['git', '-C', str(repo), *args], text=True).strip()

            git('init', '-q')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.invalid')
            (repo / 'VERSION').write_text('1.0.0')
            git('add', 'VERSION')
            git('commit', '-qm', 'fixture')
            revision = git('rev-parse', 'HEAD')
            self.assertEqual(
                module().source_identity(repo, revision)['revision'], revision)
            with self.assertRaises(ValueError):
                module().source_identity(repo, '0' * 40)
            (repo / 'VERSION').write_text('2.0.0')
            with self.assertRaises(ValueError):
                module().source_identity(repo, revision)

    @pytest.mark.agent_local
    def test_installed_archive_members_must_match_even_when_version_matches(
            self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / 'candidate.whl'
            prefix = root / 'installed'
            prefix.mkdir()
            files = {
                'kumo/engine/__init__.py': b'__version__="1.0.0"\n',
                'diskgraph/native.so': b'exact-native-payload'
            }
            with ZipFile(wheel, 'w') as archive:
                for name, value in files.items():
                    archive.writestr(name, value)
                    destination = prefix / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(value)
                archive.writestr('kumo_engine-1.0.0.dist-info/METADATA',
                                 'Name: kumo-engine\nVersion: 1.0.0\n')
            self.assertEqual(module().verify_installed_files(wheel, prefix), 2)
            (prefix / 'diskgraph/native.so').write_bytes(b'substituted-wheel')
            with self.assertRaises(ValueError):
                module().verify_installed_files(wheel, prefix)

    @pytest.mark.agent_local
    def test_no_completed_gpu_fits_or_failed_fit_cannot_pass(self):
        checked = module().verify_fits
        with self.assertRaises(ValueError):
            checked([])
        for row in [
                dict(status='PASS', device='cpu', steps=2),
                dict(status='PASS', device='cuda:0', steps=0),
                dict(status='FAIL', device='cuda:0', steps=2),
                dict(status='PASS', device='cuda:0', steps=2, cuda_batches=0)
        ]:
            with self.assertRaises(ValueError):
                checked([row])
        checked([dict(status='PASS', device='cuda:0', steps=2, cuda_batches=1)])

    @pytest.mark.agent_local
    def test_only_eight_normal_seed_zero_primary_fits_pass(self):
        tasks = [
            'binary', 'multiclass', 'multilabel', 'multilabel_ranking',
            'regression', 'forecasting', 'static_link', 'temporal_link'
        ]
        report = dict(
            status='MEASURED', learning_quality_policy='report_only',
            runs=[dict(task=t, seed=0, case='learnable') for t in tasks])
        module().verify_roster(report)
        for key, value in [('seed', 1), ('case', 'independent'),
                           ('task', 'binary')]:
            broken = json.loads(json.dumps(report))
            broken['runs'][-1][key] = value
            with self.assertRaises(ValueError):
                module().verify_roster(broken)

    @pytest.mark.agent_local
    def test_release_wheel_cannot_be_recorded_as_this_pr_build(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / 'released.whl'
            with ZipFile(wheel, 'w') as archive:
                archive.writestr('kumo_engine-1.0.0.dist-info/METADATA',
                                 'Name: kumo-engine\nVersion: 1.0.0\n')
            source = dict(revision='a' * 40, tree='b' * 40,
                          bundle_version='1.0.0+aaaaaaaa')
            with self.assertRaises(ValueError):
                module().build_receipt(source, wheel)

    @pytest.mark.agent_local
    def test_quickstart_rejects_duplicate_missing_or_nonfinite_predictions(
            self):
        import pandas as pd

        frame = pd.DataFrame({
            'ENTITY': [1, 2],
            'TARGET_PRED': [False, True],
            'False_PROB': [.9, .2],
            'True_PROB': [.1, .8]
        })
        module().verify_quickstart_predictions(frame, [1, 2])
        for bad in (frame.iloc[:1], frame.assign(ENTITY=[1, 1]),
                    frame.assign(True_PROB=[.1, float('nan')]),
                    frame.assign(False_PROB=[-.1, .2])):
            with self.assertRaises(ValueError):
                module().verify_quickstart_predictions(bad, [1, 2])

    @pytest.mark.agent_local
    def test_prediction_work_observer_accepts_link_and_forecast_output_shapes(
            self):
        score, horizon_a, horizon_b = object(), object(), object()
        extract = module().prediction_tensors
        self.assertEqual(extract({'score': score}), [score])
        self.assertEqual(extract({'logits': {
            'a': horizon_a,
            'b': horizon_b
        }}), [horizon_a, horizon_b])
        self.assertEqual(extract({'logits': score}), [score])

    @pytest.mark.agent_local
    def test_quickstart_rejects_unnormalized_or_wrong_class_outputs(self):
        import pandas as pd

        frame = pd.DataFrame({
            'ENTITY': [1],
            'TARGET_PRED': [True],
            'False_PROB': [.1],
            'True_PROB': [.9]
        })
        for bad in (frame.assign(False_PROB=.9),
                    frame.assign(TARGET_PRED=False)):
            with self.assertRaises(ValueError):
                module().verify_quickstart_predictions(bad, [1])


if __name__ == '__main__':
    unittest.main()
