# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import copy
import importlib.util
import inspect
import runpy
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / 'packages/kumo-engine/test'
sys.path.insert(0, str(ROOT))


class WheelScaleLearningTest(unittest.TestCase):
    def worker(self):
        path = ROOT / 'wheel_scale_learning.py'
        self.assertTrue(path.is_file(), 'Prepared-profile execution is missing')
        return runpy.run_path(str(path))

    def exercise_prediction_sequence(self, directory, *, missing_repeat=False,
                                     ranked=False):
        worker = self.worker()
        tree = ast.parse(inspect.getsource(worker['run_case']))
        self.assertEqual(
            sum(
                isinstance(node, ast.Call) and isinstance(
                    node.func, ast.Attribute) and node.func.attr == 'fit'
                for node in ast.walk(tree)), 1)
        body = next(
            node.body for node in ast.walk(tree)
            if isinstance(node, ast.With) and any(
                isinstance(item, ast.FunctionDef) and item.name == 'predict'
                for item in node.body))
        start = next(
            i for i, item in enumerate(body)
            if isinstance(item, ast.FunctionDef) and item.name == 'predict')
        actual = ast.Module(body=body[start:], type_ignores=[])

        class IndexedIds(list):
            def index(self, value):
                raise AssertionError(
                    'Prediction must not scan entity positions')

            def __contains__(self, value):
                raise AssertionError(
                    'Prediction must not scan entity membership')

        anchor, ids = datetime(2025, 1, 1), IndexedIds([10, 20])
        truth = dict(entity_ids=ids, targets=[True, False], anchor=anchor)
        if ranked:
            truth = dict(entity_ids=ids, target_format='ranked_link_v1',
                         target_ids=[[8], [7]], excluded_ids=[[0], [0]],
                         catalog=list(range(9)), classes=[], metric='recall@5',
                         anchor=None)
        inputs = [dict(ENTITY1=entity, TIMESTAMP=anchor) for entity in ids]
        prepared = dict(truth=truth, prediction_rows=inputs,
                        rows=[dict(row, SPLIT=2) for row in inputs])
        events, seeds, evidence = [], [], dict(consistency_measurements=[])
        gpu = runpy.run_path(str(ROOT / 'wheel_gpu_learning.py'))
        import pyarrow.parquet as pq

        class Connector:
            def __init__(self, root_dir):
                self.root = Path(root_dir)

            def __getitem__(self, name):
                return self.root / name

        class Model:
            training_table_generation_plan = 'fixed'

            def predict(model, table, **kwargs):
                population = pq.read_table(table.table_path).to_pandas()
                label = table.table_path.stem
                events.append((model, population.ENTITY1.tolist(), label))
                if ranked:
                    frame = pd.DataFrame([
                        dict(ENTITY=entity, CLASS=item,
                             SCORE=[10., 9., 8., 7., 6., 6., 6., 6., 6.][item])
                        for entity in population.ENTITY1 for item in range(9)
                    ]).sort_values(['ENTITY', 'SCORE', 'CLASS'],
                                   ascending=[True, False, False])
                    frame = frame.groupby('ENTITY', sort=False).head(
                        kwargs['num_classes_to_return'])
                    native = directory / f'{label}-native.parquet'
                    frame.to_parquet(native, index=False)
                    return SimpleNamespace(files=[str(native)])
                if missing_repeat and label == 'repeat-predictions-inputs':
                    population = population.iloc[:1]
                probability = population.ENTITY1.map({10: .8, 20: .2})
                if label == 'repeat-predictions-inputs':
                    probability += .000005
                return SimpleNamespace(frame=pd.DataFrame(
                    dict(ENTITY=population.ENTITY1, True_PROB=probability,
                         False_PROB=1 - probability, TARGET_PRED=probability >
                         .5, TIMESTAMP=anchor, ANCHOR_TIMESTAMP=anchor)))

        trainer, loaded = Model(), Model()

        def compare(expected, result):
            measured = gpu['measure_consistency'](expected, result)
            evidence['consistency_measurements'].append(
                dict(
                    stage='reload_and_predict'
                    if events[-1][0] is loaded else 'prediction', **measured))
            return measured['max_abs_difference']

        namespace = dict(
            worker, prepared=prepared,
            case='static_link' if ranked else 'binary', ids=ids, truth=truth,
            ranked_links=ranked, positions={
                entity: index
                for index, entity in enumerate(ids)
            }, directory=directory, input_dir=directory / 'inputs',
            output_dir=directory / 'outputs', seed=3, time_column='signup_time',
            trainer=trainer, fitted=SimpleNamespace(bundle_dir=directory /
                                                    'bundle'), query=object(),
            snapshot=object(), evidence=evidence, pq=pq,
            measurements=worker['Measurements'](directory),
            torch=SimpleNamespace(manual_seed=seeds.append),
            checks=SimpleNamespace(verify_reload=compare),
            ordered_batches=lambda files, directory: (pd.read_parquet(path)
                                                      for path in files),
            MetadataField=SimpleNamespace(ANCHOR_TIMESTAMP='anchor'),
            kumo=SimpleNamespace(
                FileConnector=Connector, PredictionTable=SimpleNamespace(
                    from_source_table=lambda source, **kwargs: SimpleNamespace(
                        table_path=source)),
                PredictionOutputConfig=lambda **kwargs: kwargs,
                Trainer=SimpleNamespace(load=lambda path: loaded)))
        namespace['input_dir'].mkdir()
        namespace['output_dir'].mkdir()
        exec(
            compile(ast.fix_missing_locations(actual),
                    str(ROOT / 'wheel_scale_learning.py'), 'exec'), namespace)
        return namespace, events, seeds

    def test_full_repeat_uses_same_model_population_seed_and_saves_raw_output(
            self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            result, events, seeds = self.exercise_prediction_sequence(directory)
            self.assertEqual([label for model, ids, label in events], [
                'predictions-inputs', 'repeat-predictions-inputs',
                'partial-predictions-inputs', 'reloaded-predictions-inputs'
            ])
            self.assertIs(events[0][0], events[1][0])
            self.assertIs(events[0][0], events[2][0])
            self.assertEqual(events[0][1], events[1][1])
            self.assertEqual(seeds, [3] * 4)
            self.assertAlmostEqual(result['repeat_delta'], .000005)
            self.assertEqual(
                pd.read_parquet(directory /
                                'repeat-predictions.parquet').ENTITY.tolist(),
                [10, 20])
            self.assertEqual([
                item['scope']
                for item in result['evidence']['consistency_measurements']
            ], ['repeat_full', 'partial', 'reload'])
            self.assertTrue(
                all(item['qualified'] is False and item['gpu_tolerance'] is None
                    for item in result['evidence']['consistency_measurements']))

    def test_full_repeat_rejects_missing_predictions_and_preserves_failed_raw_rows(
            self):
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            with self.assertRaisesRegex(ValueError, 'missing or repeated'):
                self.exercise_prediction_sequence(directory,
                                                  missing_repeat=True)
            self.assertEqual(
                len(pd.read_parquet(directory / 'repeat-predictions.parquet')),
                1)

    def test_ranked_sequence_streams_files_and_replays_only_ambiguous_ids(self):
        with tempfile.TemporaryDirectory() as path:
            result, events, seeds = self.exercise_prediction_sequence(
                Path(path), ranked=True)
            self.assertEqual(len(events), 8)
            self.assertEqual(result['predictions']['ranked_ids'],
                             [[1, 2, 3, 4, 5]] * 2)
            calls = result['evidence']['ranked_prediction_calls']
            self.assertEqual([row['full_catalog'] for row in calls],
                             [False, True] * 4)
            self.assertEqual([row['entities'] for row in calls],
                             [2, 2, 2, 2, 1, 1, 2, 2])
            self.assertEqual([row['raw_limit'] for row in calls], [7, 9] * 4)
            self.assertTrue(all(row['seconds'] >= 0 for row in calls))
            self.assertTrue(
                all(row['score_coordinates'] == 'entity_class'
                    and row['added_recommendations'] ==
                    row['removed_recommendations'] == 0
                    and row['qualified'] is False
                    for row in result['evidence']['consistency_measurements']))

    @unittest.skipUnless(importlib.util.find_spec('torch'),
                         'Requires the checkpoint runtime')
    def test_selected_checkpoint_requires_positive_completed_training_steps(
            self):
        import torch

        read = self.worker()['selected_training_step']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'step=0.ckpt'
            for value in (None, 0, -1, False, 1.5):
                torch.save({'global_step': value}, path)
                with self.subTest(step=value), self.assertRaisesRegex(
                        ValueError, 'completed training steps'):
                    read(path)
            for value in (1, 80, 320):
                torch.save({'global_step': value}, path)
                self.assertEqual(read(path), value)

    def test_binary_predictions_keep_ids_probabilities_and_anchor(self):
        align = self.worker()['prediction_matrix']
        truth = dict(entity_ids=[20, 10], targets=[False, True],
                     anchor=datetime(2025, 1, 1))
        frame = pd.DataFrame(
            dict(ENTITY=[10, 20], True_PROB=[.8, .1], False_PROB=[.2, .9],
                 TARGET_PRED=[True, False], TIMESTAMP=[truth['anchor']] * 2))
        np.testing.assert_array_equal(align('binary', frame, truth),
                                      [[.1], [.8]])
        broken = [
            frame.iloc[:1],
            pd.concat([frame, frame.iloc[:1]]),
            frame.assign(False_PROB=[.9, .9]),
            frame.assign(False_PROB=[.200005, .9]),
            frame.assign(TARGET_PRED=False),
            frame.drop(columns='TARGET_PRED'),
            frame.assign(TARGET_PRED=pd.Series([True, pd.NA], dtype='boolean')),
            frame.assign(TARGET=[False, False]),
            frame.drop(columns='TIMESTAMP'),
            frame.assign(TIMESTAMP=truth['anchor'] + timedelta(days=1))
        ]
        for value in broken:
            with self.subTest(value=value.to_dict()), self.assertRaises(
                (ValueError, AssertionError)):
                align('binary', value, truth)

    def test_forecast_requires_all_twelve_coordinates(self):
        align = self.worker()['prediction_matrix']
        horizons = [
            datetime(2025, 1, 1) + timedelta(days=7 * i) for i in range(12)
        ]
        truth = dict(entity_ids=[9], horizons=horizons)
        frame = pd.DataFrame(
            dict(ENTITY=[9] * 12, TIMESTAMP=horizons,
                 TARGET_PRED=list(range(12))))
        np.testing.assert_array_equal(align('forecasting', frame, truth),
                                      [list(range(12))])
        for value in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]])):
            with self.assertRaises(ValueError):
                align('forecasting', value, truth)

    def test_only_links_can_omit_native_scalar_holdout(self):
        verify = self.worker()['verify_holdout']
        for case in ('static_link', 'temporal_link'):
            verify(case, None, {})
        for case in ('binary', 'regression', 'forecasting'):
            with self.subTest(case=case), self.assertRaises(ValueError):
                verify(case, None, {})
        truth = dict(
            entity_ids=[9], horizons=[
                datetime(2025, 1, 1) + timedelta(days=7 * i) for i in range(12)
            ], targets=[list(range(12))])
        frame = pd.DataFrame(
            dict(ENTITY=[9] * 12, TIMESTAMP=truth['horizons'],
                 TARGET_PRED=list(range(12)), TARGET=list(range(12))))
        verify('forecasting', frame, truth)
        with self.assertRaises(ValueError):
            verify('forecasting', frame.assign(TARGET=99), truth)

    def test_binary_partial_population_does_not_need_both_answer_classes(self):
        align = self.worker()['prediction_matrix']
        truth = dict(entity_ids=[20, 10], targets=[False, False],
                     anchor=datetime(2025, 1, 1))
        frame = pd.DataFrame(
            dict(ENTITY=[10, 20], True_PROB=[.2, .1], False_PROB=[.8, .9],
                 TARGET_PRED=False, TIMESTAMP=truth['anchor']))
        np.testing.assert_array_equal(align('binary', frame, truth),
                                      [[.1], [.2]])

    def test_native_forecast_time_alias_keeps_horizons_and_anchor(self):
        worker = self.worker()
        anchor = datetime(2025, 1, 1)
        horizons = [anchor + timedelta(days=7 * i) for i in range(12)]
        truth = dict(entity_ids=[9], anchor=anchor, horizons=horizons)
        frame = pd.DataFrame(
            dict(ENTITY=[9] * 12, signup_time=horizons, ANCHOR_TIMESTAMP=anchor,
                 TARGET_PRED=list(range(12))))
        normalized = worker['normalize_prediction_time']('forecasting', frame,
                                                         truth, 'signup_time',
                                                         require_anchor=True)
        self.assertEqual(normalized['signup_time'].tolist(), horizons)
        self.assertEqual(normalized['TIMESTAMP'].tolist(), horizons)
        np.testing.assert_array_equal(
            worker['prediction_matrix']('forecasting', normalized, truth),
            [list(range(12))])
        for broken in (frame.drop(columns='signup_time'),
                       frame.drop(columns='ANCHOR_TIMESTAMP'),
                       frame.assign(ANCHOR_TIMESTAMP=anchor +
                                    timedelta(days=1)),
                       frame.assign(TIMESTAMP=anchor + timedelta(days=1))):
            with self.subTest(
                    columns=broken.columns), self.assertRaises(ValueError):
                worker['normalize_prediction_time']('forecasting', broken,
                                                    truth, 'signup_time',
                                                    require_anchor=True)

    def test_temporal_anchor_metadata_is_required_and_aliases_cannot_conflict(
            self):
        normalize = self.worker()['normalize_prediction_time']
        truth = dict(anchor=datetime(2025, 1, 1))
        frame = pd.DataFrame(dict(ENTITY=[9], ANCHOR_TIMESTAMP=truth['anchor']))
        for case in ('binary', 'multiclass', 'multilabel', 'multilabel_ranking',
                     'regression', 'temporal_link'):
            with self.subTest(case=case):
                normalized = normalize(case, frame, truth, 'signup_time',
                                       require_anchor=True)
                self.assertEqual(normalized['TIMESTAMP'].tolist(),
                                 [truth['anchor']])
                for broken in (frame.drop(columns='ANCHOR_TIMESTAMP'),
                               frame.assign(TIMESTAMP=truth['anchor'] +
                                            timedelta(days=1)),
                               frame.assign(signup_time=truth['anchor'] +
                                            timedelta(days=1))):
                    with self.assertRaises(ValueError):
                        normalize(case, broken, truth, 'signup_time',
                                  require_anchor=True)
        for field in ('ENTITY', 'TARGET_PRED', 'CLASS', 'ANCHOR_TIMESTAMP'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize('forecasting', frame, truth, field)

    def test_static_scores_cover_full_catalog_and_mask_known_links(self):
        align = self.worker()['prediction_matrix']
        truth = dict(entity_ids=[7], catalog=list(range(12)), known={'7': [0]})
        frame = pd.DataFrame(
            dict(ENTITY=[7] * 12, CLASS=list(range(12)),
                 SCORE=list(range(12, 0, -1))))
        values = align('static_link', frame, truth)
        self.assertLess(values[0, 0], values[0, 11])
        for value in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]])):
            with self.assertRaises(ValueError):
                align('static_link', value, truth)

    def test_ranked_static_stream_requires_tie_fallback_and_complete_coordinates(
            self):
        select = self.worker()['ranked_from_batches']
        truth = dict(entity_ids=[9], catalog=list(range(7)), excluded_ids=[[]],
                     metric='recall@5')
        first = pd.DataFrame(
            dict(ENTITY=[9] * 5, CLASS=[0, 1, 2, 3, 6],
                 SCORE=[10., 9., 8., 7., 6.]))
        self.assertEqual(select([first], truth, 5)['fallback_ids'], [9])
        full = pd.concat([
            first,
            pd.DataFrame(dict(ENTITY=[9, 9], CLASS=[4, 5], SCORE=[6., 6.]))
        ])
        result = select([full.iloc[:3], full.iloc[3:]], truth, 7)
        self.assertEqual(result['fallback_ids'], [])
        self.assertEqual(result['ranked_ids'], [[0, 1, 2, 3, 4]])
        self.assertEqual(result['raw_rows'], 7)
        for broken in (full.iloc[:-1], full.assign(CLASS=0),
                       full.assign(ENTITY=8), full.assign(SCORE=float('nan'))):
            with self.subTest(rows=len(broken)), self.assertRaises(ValueError):
                select([broken], truth, 7)

    def test_ranked_popularity_keeps_the_existing_known_item_mask_and_tie_order(
            self):
        baseline = self.worker()['ranked_baseline']
        truth = dict(entity_ids=[9], catalog=list(range(8)), metric='recall@5',
                     excluded_ids=[[0, 1]],
                     training_popularity=[9, 8, 7, 7, 7, 7, 6, 5])
        result = baseline(truth)
        self.assertEqual(result['ranked_ids'], [[2, 3, 4, 5, 6]])
        self.assertEqual(result['ranked_scores'], [[7, 7, 7, 7, 6]])

    def test_ranked_scores_use_public_mean_recall_for_model_and_baseline(self):
        score = self.worker()['ranked_task_score']
        with patch.dict(sys.modules, {'scalebench': None,
                                      'scalebench.metrics': None}):
            for task, k, expected, baseline_expected in (
                    ('static_link', 5, 5 / 22, 2 / 11),
                    ('temporal_link', 10, 31 / 44, 9 / 22)):
                with self.subTest(task=task):
                    row = dict(
                        task=task, metric=f'recall@{k}',
                        target_format='ranked_link_v1', entity_ids=[10, 20],
                        catalog=list(range(12)),
                        target_ids=[list(range(11)), [10, 11]],
                        excluded_ids=[[], []],
                        ranked_ids=[list(range(k)), list(range(1, k + 1))],
                        ranked_scores=[list(range(k, 0, -1))] * 2,
                        baseline_ranked_ids=[list(range(12 - k, 12)),
                                             list(range(k))],
                        baseline_ranked_scores=[list(range(k, 0, -1))] * 2)
                    self.assertAlmostEqual(score(row), expected)
                    self.assertAlmostEqual(score(row, baseline=True),
                                           baseline_expected)

    def test_public_ranked_scoring_keeps_coordinate_and_exclusion_checks(self):
        score = self.worker()['ranked_task_score']
        row = dict(task='static_link', metric='recall@5',
                   target_format='ranked_link_v1', entity_ids=[7],
                   catalog=list(range(7)), target_ids=[[0, 6]],
                   excluded_ids=[[]], ranked_ids=[[0, 1, 2, 3, 4]],
                   ranked_scores=[[5, 4, 3, 2, 1]])
        for values in (
                dict(ranked_ids=[]),
                dict(ranked_ids=[[0, 1, 2, 3, 3]]),
                dict(ranked_scores=[[float('nan')] * 5]),
                dict(excluded_ids=[[0]])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                score(dict(row, **values))
        with self.assertRaises(ValueError):
            score(dict(row, baseline_ranked_ids=[[0, 1, 2, 3]],
                       baseline_ranked_scores=[[4, 3, 2, 1]]), baseline=True)

    def test_recall_ten_counts_ranks_eight_and_ten_and_needs_full_catalog(self):
        checks = runpy.run_path(str(ROOT / 'wheel_task_checks.py'))
        targets = [[int(i in (7, 9)) for i in range(12)]]
        predictions = [list(range(12, 0, -1))]
        self.assertEqual(checks['score']('recall@5', targets, predictions), 0)
        self.assertEqual(checks['score']('recall@10', targets, predictions), 1)
        with self.assertRaises(ValueError):
            checks['score']('recall@10', [[1] + [0] * 8], [[1] * 9])
        row = dict(task='temporal_link', seed=0, case='independent',
                   metric='recall@10', targets=targets, predictions=[[1] * 12],
                   control_eligible=[[True] * 12])
        self.assertGreater(
            checks['independent_controls']([[row]])['temporal_link']['p_value'],
            .01 / 7)
        row['control_eligible'][0][7] = False
        with self.assertRaises(ValueError):
            checks['independent_controls']([[row]])

    def test_prediction_inputs_hide_answers_and_keep_only_past_forecast(self):
        verify = self.worker()['prediction_inputs']
        anchor = datetime(2025, 1, 1)
        history = list(range(12))
        prepared = dict(
            truth={
                'entity_ids': [9],
                'anchor': anchor
            }, rows=[
                dict(ENTITY1=9, TIMESTAMP=anchor, SPLIT=2,
                     autoregression_labels=history, future_labels=[99] * 12)
            ], prediction_rows=[
                dict(ENTITY1=9, TIMESTAMP=anchor, autoregression_labels=history)
            ])
        self.assertEqual(verify(prepared, 'forecasting', [9]),
                         prepared['prediction_rows'])
        for mutation in ('answers', 'history', 'anchor', 'ids'):
            changed = copy.deepcopy(prepared)
            row = changed['prediction_rows'][0]
            if mutation == 'answers':
                row['future_labels'] = [99] * 12
            elif mutation == 'history':
                row['autoregression_labels'] = [99] * 12
            elif mutation == 'anchor':
                row['TIMESTAMP'] += timedelta(days=1)
            else:
                row['ENTITY1'] = 10
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify(changed, 'forecasting', [9])

    def test_profile_admission_uses_five_run_mean_not_individual_floor(self):
        verify = self.worker()['validate_profile_learning']
        rows = [
            dict(task='binary', seed=seed, case='learnable', score=value,
                 baseline_score=.5, metric='auroc')
            for seed, value in enumerate([.4, .8, .8, .8, .8])
        ]
        result = verify(rows)
        self.assertAlmostEqual(result['binary']['mean_score'], .72)
        self.assertFalse(result['binary']['release_approved'])
        result = verify(rows[:3], seeds=[0, 1, 2])
        self.assertAlmostEqual(result['binary']['mean_score'], 2 / 3)
        with self.assertRaises(ValueError):
            verify(rows[:3])
        with self.assertRaises(ValueError):
            verify(rows, seeds=[0, 1, 2])
        for values in ([.5] * 5, [.4] * 5):
            broken = [
                dict(row, score=value) for row, value in zip(rows, values)
            ]
            with self.assertRaisesRegex(ValueError, 'baseline'):
                verify(broken)
        for broken in (rows[:-1], rows + [rows[0]],
                       [dict(row, score=float('nan')) for row in rows]):
            with self.assertRaises(ValueError):
                verify(broken)
        weak = [dict(row, score=.4) for row in rows]
        result = verify(weak, calibration=True)
        self.assertFalse(result['binary']['useful_learning_passed'])
        self.assertFalse(result['binary']['release_approved'])
        with self.assertRaisesRegex(ValueError, 'baseline'):
            verify(weak)
        with self.assertRaises(ValueError):
            verify(weak[:-1], calibration=True)
        regression = [
            dict(row, task='regression', metric='mae', score=1,
                 baseline_score=2) for row in rows
        ]
        self.assertEqual(verify(regression)['regression']['mean_score'], 1)

    def test_binary_quality_uses_positive_scores_and_rejects_extra_columns(
            self):
        score = self.worker()['task_score']
        self.assertEqual(score('binary', [False, True], [[.1], [.9]]), 1)
        self.assertEqual(score('binary', [False, True], [[.5], [.5]]), .5)
        for values in ([[.1, .9], [.9, .1]], [[float('nan')], [.9]], [[.1]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                score('binary', [False, True], values)

    def test_plans_keep_task_recipe_and_bound_real_graph_sampling(self):
        from kumoapi.model_plan import MissingType

        make = self.worker()['make_plan']
        for case in ('binary', 'multiclass', 'multilabel', 'multilabel_ranking',
                     'regression', 'forecasting', 'static_link',
                     'temporal_link'):
            with self.subTest(case=case):
                plan = make(case, 3)
                values = plan.as_json()
                self.assertEqual(values['training_job']['manual_seed'], 3)
                neighbors = values['neighbor_sampling']['num_neighbors']
                self.assertEqual(neighbors[0]['hop1']['default'], 16)
                self.assertEqual(neighbors[0]['hop2']['default'], 16)
                self.assertEqual(
                    plan.neighbor_sampling.sample_from_entity_table,
                    MissingType.VALUE)
                self.assertEqual(
                    values['neighbor_sampling']['sample_from_entity_table'],
                    MissingType.VALUE)
                self.assertFalse(
                    any(
                        key.startswith(('groups.', 'ents.', 'events.'))
                        for key in plan.column_processing.encoder_overrides))
                metric = 'recall@10' if case == 'temporal_link' else None
                if metric:
                    self.assertEqual(values['training_job']['metrics'],
                                     [metric])

    def test_runtime_requires_task_type_and_actual_temporal_cutoff(self):
        verify = self.worker()['verify_runtime']
        runtime = dict(
            task=dict(task_type='binary_classification', is_temporal=True),
            table=dict(columns=dict(timestamp='TIMESTAMP')))
        schema = dict(node_types=dict(conversions=dict(
            time_column='conversion_time', feature_columns=['campaign_id'])))
        config = dict(sampler_kwargs=dict(temporal_strategy='last'))
        verify('binary', runtime, schema, config)
        for mutation in ('task', 'temporal', 'stamp', 'cutoff', 'leak'):
            r, s, c = copy.deepcopy((runtime, schema, config))
            if mutation == 'task':
                r['task']['task_type'] = 'regression'
            elif mutation == 'temporal':
                r['task']['is_temporal'] = False
            elif mutation == 'stamp':
                s['node_types']['conversions']['time_column'] = None
            elif mutation == 'cutoff':
                c['sampler_kwargs']['temporal_strategy'] = 'uniform'
            else:
                s['node_types']['conversions']['feature_columns'].append(
                    'TARGET1')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify('binary', r, s, c)

    def test_forecast_training_schema_can_store_targets_but_graph_cannot(self):
        verify = self.worker()['verify_runtime']
        runtime = dict(task=dict(task_type='forecasting', is_temporal=True),
                       table=dict(columns=dict(timestamp='TIMESTAMP')))
        schema = dict(node_types=dict(
            orders=dict(time_column='order_time', feature_columns=['daypart']),
            train_table=dict(
                feature_columns=['future_labels', 'autoregression_labels'])))
        config = dict(sampler_kwargs=dict(temporal_strategy='last'))
        verify('forecasting', runtime, schema, config)
        schema['node_types']['orders']['feature_columns'].append(
            'future_labels')
        with self.assertRaises(ValueError):
            verify('forecasting', runtime, schema, config)

    @unittest.skipUnless(importlib.util.find_spec('duckdb'),
                         'Requires the pinned DuckDB runtime')
    def test_ranked_reader_accepts_public_prediction_file_tuples(self):
        read = self.worker()['ordered_batches']
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            file = directory / 'predictions.parquet'
            pd.DataFrame(dict(ENTITY=[2, 1], CLASS=[0, 0],
                              SCORE=[.2, .1])).to_parquet(file)
            frames = list(read((str(file), ), directory / 'sort'))
            self.assertEqual(pd.concat(frames).ENTITY.tolist(), [1, 2])


if __name__ == '__main__':
    unittest.main()
