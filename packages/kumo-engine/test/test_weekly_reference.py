# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Migrate saved reference bytes without a worker, model call, or new fit."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / 'scripts/weekly_reference.py'


def module():
    spec = importlib.util.spec_from_file_location('weekly_reference', SOURCE)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def legacy_fixture(api, root):
    scores = {
        'Q13': (.6, .01),
        'Q14': (2., 10.),
        'Q15': (.533, .531),
        'Q16': (.9, .5),
        'Q17': (.742, .004)
    }
    specs, cases = [], []
    current = dict(reference=dict(version='1.0.0', wheel_sha256='a' * 64),
                   cases=[])
    for task, (dataset, metric) in api.weekly.PROFILES.items():
        for arm in ('fixed', ) if task == 'P2' else ('normal', 'control'):
            name = task + '-' + arm
            identity = dict(data_sha256='b' * 64)
            spec = dict(
                id=name, workload=task, dataset=dataset, metric=metric, seed=0,
                identity=identity,
                condition='scrambled_answer_control' if arm == 'control' else
                'fixed_scale_workload' if arm == 'fixed' else 'normal')
            specs.append(spec)
            cases.append(name)
            directory = root / 'cases' / name
            model = directory / ('resume-worker/saved-model'
                                 if task == 'P2' else 'worker/saved-model')
            model.mkdir(parents=True)
            (model / 'model').mkdir()
            (model / 'model/checkpoint').write_bytes(
                ('checkpoint-' + name).encode())
            api.weekly.save(model / 'training-spec.json',
                            dict(predictive_query='query-' + task))
            result = dict(
                status='MEASURED', fit_id='fit-' + name, seed=0,
                training_seed=0, identity=identity, query='query-' + task,
                metric=metric, entity_ids=[11, 12],
                score=None if task == 'P2' else scores[task][arm == 'control'],
                expected_reload_checkpoint_sha256=api.weekly.digest(
                    model / 'model/checkpoint'), checks={
                        key: 'PASS'
                        for key in ('fit', 'prediction', 'repeat_prediction',
                                    'partial_prediction', 'reload',
                                    'hidden_answers', 'chronology')
                    })
            if metric.startswith('recall@'):
                result.update(ranked_ids=[[3, 4], [4, 3]],
                              ranked_scores=[[.7, .3], [.6, .4]])
            else:
                result['predictions'] = [[.2], [.8]]
            api.weekly.save(directory / 'result.json', result)
            api.weekly.save(directory / 'cuda-batches.json',
                            dict(training=3, prediction=4))
            api.weekly.save(
                directory / 'weekly-case.json',
                dict(schema=1, status='MEASURED', mode='reference',
                     case_id=name, case_spec=spec, identity=identity,
                     physical_fits=1, result=result))
            current['cases'].append(
                dict(id=name, dataset=dataset, metric=metric, seed=0, arm=arm,
                     query='query-' + task, legacy_identity=identity,
                     tolerance=dict(atol=1e-5, rtol=1e-5)))
    # This file is retained historical code. Import must only hash its bytes.
    (root / 'host_worker.py'
     ).write_text("raise RuntimeError('legacy worker must never run')\n")
    api.weekly.save(
        root / 'manifest.json',
        dict(schema=1, mode='reference', status='MEASURED', physical_fits=11,
             version='1.0.0', wheel_sha256='a' * 64, cases=cases,
             case_specs=specs, files=api.weekly.inventory(root)))
    return current, api.weekly.digest(root / 'manifest.json')


class ReferenceImport(unittest.TestCase):
    def test_import_preserves_all_saved_models_without_retraining(self):
        self.assertTrue(SOURCE.is_file())
        api = module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / 'legacy'
            legacy.mkdir()
            manifest, checksum = legacy_fixture(api, legacy)
            before = api.weekly.inventory(legacy)
            report = api.import_reference(legacy, checksum, manifest, 'd' * 64,
                                          root / 'converted')
            self.assertEqual(report['new_physical_fits'], 0)
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(set(report['cases']), api.weekly.ROSTER)
            loaded = api.weekly.load_reference(
                root / 'converted',
                api.weekly.digest(root / 'converted/reference.json'), manifest,
                'd' * 64)
            self.assertEqual(loaded['cases']['Q17-normal']['fit_id'],
                             'fit-Q17-normal')
            self.assertEqual(loaded['cases']['P2-fixed']['source_model_path'],
                             'resume-worker/saved-model')
            self.assertEqual(
                api.weekly.read(root /
                                'converted/cases/Q15-normal/predictions.json'),
                dict(entity_ids=[11, 12], values=[[.2], [.8]],
                     decisions=[False, True]))
            self.assertEqual(
                (root / 'converted/cases/Q17-normal/model/model/checkpoint'
                 ).read_bytes(), b'checkpoint-Q17-normal')
            self.assertEqual(api.weekly.inventory(legacy), before)

    def test_import_rejects_changed_source_and_different_workload(self):
        self.assertTrue(SOURCE.is_file())
        api = module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / 'legacy'
            legacy.mkdir()
            manifest, checksum = legacy_fixture(api, legacy)
            changed = copy.deepcopy(manifest)
            changed['cases'][0]['legacy_identity']['data_sha256'] = 'f' * 64
            with self.assertRaises(ValueError):
                api.import_reference(legacy, checksum, changed, 'd' * 64,
                                     root / 'wrong-input')
            (legacy / 'cases/Q17-normal/worker/saved-model/model/checkpoint'
             ).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                api.import_reference(legacy, checksum, manifest, 'd' * 64,
                                     root / 'changed-model')


if __name__ == '__main__':
    unittest.main()
