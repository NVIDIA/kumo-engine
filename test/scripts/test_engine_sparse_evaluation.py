# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check sparse binary case budgets without importing Engine or training."""
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

CASES = Path(__file__).resolve().parents[2] / 'packages/kumo-engine/test/cases'
COMMON = ModuleType('common')
COMMON.__dict__.update(runpy.run_path(str(CASES / 'common.py')))


def case_options(name, scale):
    api = ModuleType('kumoapi.model_plan')
    api.NumNeighborsConfig = lambda **values: values
    api.TrainingTableGenerationPlan = lambda **values: values
    api.WeightMode = SimpleNamespace(SAMPLE='sample')
    captured = []
    with patch.dict(
            'sys.modules', {
                'common': COMMON,
                'kumoapi': ModuleType('kumoapi'),
                'kumoapi.model_plan': api
            }):
        main = runpy.run_path(str(CASES / (name + '.py')))['main']
        with patch.dict(
                main.__globals__, {
                    'parse_args': lambda *a, **kw: SimpleNamespace(scale=scale),
                    'sources': lambda args: None,
                    'build_graph': lambda connector: None,
                    'run': lambda *a, **kw: captured.append((a[4], kw))
                }):
            main()
    assert len(captured) == 1
    return captured[0]


class SparseEvaluationTest(unittest.TestCase):
    def test_weekly_cases_use_one_balanced_trial_and_shared_evaluation_budget(
            self):
        for name in ('payments_binary', 'fraud_binary'):
            for scale in ('10B', '25B'):
                with self.subTest(case=name, scale=scale):
                    overrides, options = case_options(name, scale)
                    self.assertEqual(
                        overrides.get('training_job.num_experiments'), 1)
                    self.assertEqual(overrides.get('optimization.max_epochs'),
                                     1)
                    self.assertEqual(overrides.get('optimization.batch_size'),
                                     [64])
                    self.assertEqual(
                        overrides.get('optimization.majority_sampling_ratio'),
                        [1.0])
                    self.assertEqual(overrides.get('optimization.weight_mode'),
                                     ['sample'])
                    self.assertNotIn('min_eval_class_rows', options)
                    self.assertNotIn('optimization.max_val_steps', overrides)
                    self.assertNotIn('optimization.max_test_steps', overrides)

    def test_small_cases_keep_training_budgets_and_use_shared_evaluation_budget(
            self):
        weekly_fields = ('training_job.num_experiments',
                         'optimization.max_epochs',
                         'optimization.majority_sampling_ratio',
                         'optimization.weight_mode')
        for name in ('payments_binary', 'fraud_binary'):
            for scale in ('100M', '1B'):
                with self.subTest(case=name, scale=scale):
                    overrides, options = case_options(name, scale)
                    self.assertNotIn('min_eval_class_rows', options)
                    self.assertTrue(
                        all(key not in overrides for key in weekly_fields))
                    self.assertNotIn('optimization.max_val_steps', overrides)
                    self.assertNotIn('optimization.max_test_steps', overrides)

    def test_native_counts_include_positives_for_each_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'manifest.json'
            manifest.write_text(
                json.dumps({'split_stats': {
                    'path': 'stats.json'
                }}))
            splits = [('train', 1000, 7), ('val', 200, 2), ('test', 300, 3)]
            (root / 'stats.json').write_text(
                json.dumps({
                    name + '_table_stats': {
                        'num_train_rows': rows,
                        'target_distribution': {
                            'count': {
                                'true': positives,
                                'false': rows - positives
                            }
                        }
                    }
                    for name, rows, positives in splits
                }))
            counts = COMMON.table_counts(
                SimpleNamespace(manifest_path=manifest))
            self.assertEqual(
                counts, {
                    'rows': 1500,
                    'splits': {
                        'train': {
                            'rows': 1000,
                            'positive_rows': 7
                        },
                        'val': {
                            'rows': 200,
                            'positive_rows': 2
                        },
                        'test': {
                            'rows': 300,
                            'positive_rows': 3
                        }
                    }
                })
