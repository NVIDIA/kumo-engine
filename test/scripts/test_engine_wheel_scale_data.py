# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check source-derived targets and safe control preparation."""

import copy
import hashlib
import json
import os
import runpy
import tempfile
import unittest
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

WORKER = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/wheel_scale_data.py')
ANCHOR = datetime(2024, 1, 1)


@dataclass(frozen=True)
class FixturePlan:
    split: str
    timeframe_step: int = 14
    train_start_offset: str = '???'
    train_end_offset: str = '???'
    forecast_length: object = '???'
    lag_timesteps: object = '???'
    year_over_year: object = '???'


class ScaleDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)

    def worker(self):
        self.assertTrue(WORKER.is_file(),
                        'Prepared PR 172 data support is missing')
        return runpy.run_path(str(WORKER))

    def table(self, name, values):
        directory = self.data / name
        directory.mkdir()
        pq.write_table(pa.table(values), directory / 'part.parquet')

    def test_capacity_counts_native_work_without_dense_score_arrays(self):
        worker = self.worker()
        self.table('merchants', dict(merchant_id=list(range(100))))
        rows = [
            dict(SPLIT=stage, ENTITY1=stage * 10 + i) for stage in range(3)
            for i in range(6)
        ]
        measured = worker['evaluation_capacity']('static_link', rows, self.data)
        self.assertEqual(measured['native_rows'], {
            'train': 6,
            'val': 6,
            'test': 6
        })
        self.assertEqual(measured['evaluation_entities'], 6)
        self.assertEqual(measured['score_columns'], 100)
        self.assertEqual(measured['score_cells'], 600)
        self.assertEqual(measured['three_dense_arrays_lower_bound_bytes'],
                         14400)
        self.assertIsNone(measured['within_budget'])
        limited = worker['evaluation_capacity']('static_link', rows, self.data,
                                                max_dense_bytes=14000)
        self.assertFalse(limited['within_budget'])
        self.assertEqual(
            worker['evaluation_capacity']('forecasting', rows,
                                          self.data)['score_columns'], 12)
        self.assertEqual(
            worker['evaluation_capacity']('binary', rows,
                                          self.data)['score_columns'], 1)

    def test_ranked_link_truth_keeps_one_popularity_vector_without_dense_rows(
            self):
        worker = self.worker()
        self.table('merchants', dict(merchant_id=[10, 20, 30, 40]))
        rows = [
            dict(
                ENTITY1=stage * 2 + i, TIMESTAMP=ANCHOR, SPLIT=stage,
                TARGET1=([10, 10] if stage == 0 else [30]),
                __kumo_internal_frequency_column__=[7, 2]
                if stage == 0 else [1], exclude_target_keys=[20, 40])
            for stage in range(3) for i in range(2)
        ]
        for case in ('static_link', 'temporal_link'):
            with self.subTest(case=case):
                truth, inputs = worker['prediction_context'](case, rows,
                                                             self.data,
                                                             ranked_links=True)
                self.assertEqual(truth['target_format'], 'ranked_link_v1')
                self.assertEqual(truth['target_ids'], [[30], [30]])
                self.assertEqual(truth['training_popularity'], [2, 0, 0, 0])
                self.assertNotIn('targets', truth)
                self.assertNotIn('baseline_predictions', truth)
                self.assertNotIn('control_eligible', truth)
                self.assertTrue(
                    all(set(row) == {'ENTITY1', 'TIMESTAMP'} for row in inputs))
                if case == 'static_link':
                    self.assertEqual(truth['excluded_ids'],
                                     [[20, 40], [20, 40]])
                else:
                    self.assertNotIn('excluded_ids', truth)
        changed = copy.deepcopy(rows)
        changed[-1]['TARGET1'] = [999]
        with self.assertRaisesRegex(ValueError, 'catalog'):
            worker['prediction_context']('temporal_link', changed, self.data,
                                         ranked_links=True)

    def test_ranked_and_dense_preparation_never_share_cached_truth(self):
        worker = self.worker()
        (self.data / 'manifest.json').write_text('{}')
        source = self.data / 'kumo/build_graph.py'
        source.parent.mkdir()
        source.write_text('')
        source_hash = hashlib.sha256(b'kumo/build_graph.py').hexdigest()
        manifest_hash = hashlib.sha256(b'{}').hexdigest()
        key = (str(self.data.resolve()), manifest_hash, source_hash,
               'static_link', None)
        worker['prepare_profile'].__globals__['read_profile'] = lambda *a: {}
        worker['_PREPARED'][(*key, False)] = dict(mode='dense')
        worker['_PREPARED'][(*key, True)] = dict(mode='ranked')
        for ranked, expected in ((False, 'dense'), (True, 'ranked')):
            self.assertEqual(
                worker['prepare_profile'](self.data, self.data, 'static_link',
                                          self.data / 'unused',
                                          ranked_links=ranked),
                dict(mode=expected))

    def test_ranked_link_isolation_retains_sparse_normal_and_control_truth(
            self):
        worker = self.worker()
        normal = self.isolation_context(worker, 'static_link')
        pq.write_table(pa.table(dict(merchant_id=list(range(10, 110, 10)))),
                       self.data / 'merchants/part.parquet')
        normal['identity']['ranked_links'] = True
        normal['truth'], normal['prediction_rows'] = worker[
            'prediction_context']('static_link', normal['rows'], self.data,
                                  ranked_links=True)
        worker['isolate_final'].__globals__[
            '_reimport_rows'] = self.fake_reimport
        calibrated = worker['isolate_final'](normal, self.data / 'calibration',
                                             self.data / 'sealed')
        self.assertEqual(calibrated['truth']['target_format'], 'ranked_link_v1')
        self.assertNotIn('targets', calibrated['truth'])
        calibrated['graph'] = SimpleNamespace(_named_tables={})
        calibrated['graph_counts'] = dict(merchants=10)
        control = worker['control_table'](calibrated, 0, self.data / 'control')
        self.assertEqual(control['truth']['target_format'], 'ranked_link_v1')
        self.assertNotIn('targets', control['truth'])
        worker['verify_final_isolation'](control)

    def test_static_native_routing_does_not_parse_unsupported_public_constructor(
            self):
        worker = self.worker()
        normal = self.isolation_context(worker, 'static_link')
        table = normal['normal_table']
        table.generation_plan = replace(table.generation_plan,
                                        timeframe_step='???')
        spec = table.staging_dir / 'tt_ingest_spec.json'
        metadata = json.loads(spec.read_text())
        metadata.update(
            entity_table='customers', entity_column='customer_id',
            fact_pseudo_time=dict(table='purchases', entity_col='customer_id',
                                  target_col='merchant_id'),
            lp_rhs_is_temporal=False, lp_resolution=[
                dict(src_col='TARGET1', kind='list', dst_table='merchants')
            ])
        spec.write_text(json.dumps(metadata))
        normal.update(
            graph=SimpleNamespace(
                _named_tables={
                    'merchants':
                    SimpleNamespace(definition=SimpleNamespace(
                        pkey='merchant_id'))
                }), snapshot=SimpleNamespace(index_path=self.data / 'index'),
            query=SimpleNamespace(_parsed=SimpleNamespace(
                target_stype='categorical')))
        native = self.data / 'native'
        native.mkdir()
        (native / 'manifest.json').write_text(
            json.dumps(dict(split_kind='CustomLinkSplit')))
        calls, native_calls = [], []

        def build(config, family, cls):
            calls.append((config, family))
            return config

        def unsupported(value):
            raise ValueError('Public Core cannot parse CustomLinkSplit')

        def native_build(**kwargs):
            native_calls.append(kwargs)
            return dict(staging_dir=str(native),
                        tt_parquet_path=str(native / 'tt.parquet'))

        modules = {
            'diskgraph':
            SimpleNamespace(
                SplitSpec=object,
                build_train_table_artifacts_for_existing_table=native_build),
            'kumo_core.training.table':
            SimpleNamespace(build_split_spec=build,
                            split_config_from_repr=unsupported)
        }
        with self.fixture_imports(), patch.dict('sys.modules', modules):
            worker['_link_control'].__globals__[
                '_verify_link_resolution'] = lambda *a: None
            result = worker['_link_control'](normal, self.data / 'rows.parquet',
                                             normal['rows'],
                                             self.data / 'replay')
            self.assertEqual(native_calls[0].get('fact_pseudo_time'),
                             metadata['fact_pseudo_time'])
            del metadata['fact_pseudo_time']
            spec.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'fact'):
                worker['_link_control'](normal, self.data / 'rows.parquet',
                                        normal['rows'], self.data / 'replay')
            self.assertEqual(len(native_calls), 1)
        self.assertEqual(
            calls, [(dict(split_column='LINK_STAGE', manual_train_values=['0'],
                          manual_val_values=['1'],
                          manual_test_values=['2']), 'CustomLinkSplit')])
        self.assertEqual(result.generation_plan, table.generation_plan)

    def test_static_native_resolution_has_no_temporal_seed_timestamps(self):
        worker = self.worker()
        options, mask = [], [False]
        rows = [
            dict(SPLIT=stage, ENTITY1=50, TIMESTAMP=ANCHOR,
                 TARGET1=[10 * (stage + 1)],
                 exclude_target_keys=[10 * (i + 1) for i in range(stage)])
            for stage in range(3)
        ]

        def features(requests):
            request, = requests
            key = request['feature_names'][0]
            values = [50] if request['node_type'] == 'customers' else [
                10, 20, 30
            ]
            return [dict(type='feature', batch=pa.record_batch({key: values}))]

        def native_iter(**kwargs):
            options.append(kwargs.get('core_stage_receipt'))
            stage = ('train', 'val', 'test').index(kwargs['stage'])
            task = dict(
                seed_node_ids=pa.array([0]),
                lp_pos_rhs_global=pa.array([stage]),
                lp_pos_counts=pa.array([1]), lp_pos_mask=pa.array(mask),
                lp_exclude_rhs_global=pa.array(list(range(stage)),
                                               type=pa.int64()),
                lp_exclude_counts=pa.array([stage]))
            return iter([dict(task=task)]), None

        def temporal_only(*args, **kwargs):
            self.fail(
                'Static native rows do not expose temporal seed timestamps')

        def ingest(**kwargs):
            return dict(
                key_failures=dict(parse_failures=0, lookup_failures=0,
                                  membership_failures=0),
                task_dataset=dict(stages=[]))

        modules = {
            'diskgraph':
            SimpleNamespace(
                execute_tt_ingest=ingest, SamplerEngine=lambda *a, **k:
                SimpleNamespace(fetch_features=features),
                RuntimeIndex=lambda *a, **k: SimpleNamespace(iter=native_iter)),
            'kumo_core.diskgraph.lightning_data':
            SimpleNamespace(
                _index_np_time_unit=temporal_only,
                DiskgraphLightningDataModule=SimpleNamespace(
                    _pseudo_time_options=lambda _, stage: dict(
                        core_stage_receipt=stage))),
            'kumo_core.diskgraph.runtime_adapter':
            SimpleNamespace(engine_int64_to_ns=temporal_only)
        }
        prepared = dict(
            case='static_link',
            snapshot=SimpleNamespace(index_path=self.data / 'index'),
            graph_counts=dict(customers=1, merchants=3))
        with patch.dict('sys.modules', modules):
            worker['_verify_link_resolution'](
                prepared, SimpleNamespace(staging_dir=self.data), rows,
                'native-proof', self.data, 'customers', 'customer_id',
                'merchants', 'merchant_id')
            self.assertEqual(options, ['train', 'val', 'test'])
            for invalid in ([True], []):
                mask[:] = invalid
                with self.subTest(mask=invalid), self.assertRaisesRegex(
                        ValueError, 'positive'):
                    worker['_verify_link_resolution'](
                        prepared, SimpleNamespace(staging_dir=self.data), rows,
                        'native-proof', self.data, 'customers', 'customer_id',
                        'merchants', 'merchant_id')

    def isolation_context(self, worker, case='regression'):
        normal, _ = self.frozen_context(
            worker, 'regression' if case == 'static_link' else case)
        if case == 'static_link':
            normal['case'] = case
            normal['query_text'] = normal['normal_table'].query = worker[
                'CASES'][case]['query']
            normal['identity']['case'] = case
            normal['generation_plan'] = replace(
                normal['generation_plan'],
                split='RandomLinkSplit([0.6, 0.2, 0.2])')
            normal['normal_table'].generation_plan = normal['generation_plan']
            normal['normal_table'].required_stages = ('train', 'val')
            self.table('merchants', dict(merchant_id=[10, 20, 30, 40]))
        ranges = worker['generation_ranges'](
            normal['manifest'], case) if case != 'static_link' else None
        rows = [
            dict(
                ENTITY1=i, TIMESTAMP=(datetime.fromisoformat(ranges[stage][0])
                                      if ranges else ANCHOR), SPLIT=stage,
                TARGET1=float(i)) for stage in range(3)
            for i in range(12 if stage == 2 else 4)
        ]
        if case == 'forecasting':
            for row in rows:
                row.update(future_labels=[row['TARGET1']] * 12,
                           autoregression_labels=[row['TARGET1'] + 1] * 12)
                del row['TARGET1']
        elif case == 'static_link':
            for row in rows:
                row.update(
                    TARGET1=[10 * (row['SPLIT'] + 1)],
                    __kumo_internal_frequency_column__=[2],
                    exclude_target_keys=[
                        10 * (i + 1) for i in range(row['SPLIT'])
                    ])
        elif case == 'binary':
            for row in rows:
                row['TARGET1'] = bool(row['ENTITY1'] % 2)
        elif case in ('multilabel', 'multilabel_ranking'):
            for row in rows:
                row['TARGET1'] = 'breakfast' if row['ENTITY1'] % 2 else ''
        normal['rows'] = rows
        normal['truth'], normal['prediction_rows'] = worker[
            'prediction_context'](case, rows, self.data)
        if case == 'static_link':
            normal['truth'], normal['prediction_rows'] = worker[
                'prediction_context'](case, rows, self.data, source_known={
                    i: {40}
                    for i in range(12)
                })
        return normal

    def fake_reimport(self, prepared, rows, directory, *, filename):
        import shutil
        directory = Path(directory)
        staging = directory / 'staging'
        shutil.copytree(prepared['normal_table'].staging_dir, staging)
        for path in staging.rglob('*.parquet'):
            if not path.is_file():
                continue
            name = path.parent.name
            stage = {
                'train_seeds': 0,
                'val_seeds': 1,
                'test_seeds': 2
            }.get(name)
            selected = rows if stage is None else [
                r for r in rows if r['SPLIT'] == stage
            ]
            pq.write_table(pa.Table.from_pylist(selected), path)
        self.fit_rows = copy.deepcopy(rows)
        table = copy.copy(prepared['normal_table'])
        table.staging_dir = staging
        table.train_table_path = staging / 'tt.parquet'
        table.manifest_path = staging / 'manifest.json'
        table.required_stages = (prepared['normal_table'].required_stages
                                 if prepared['case'] in ('forecasting',
                                                         'static_link',
                                                         'temporal_link') else
                                 ('train', 'val'))
        if prepared['case'] == 'static_link':
            manifest = json.loads(table.manifest_path.read_text())
            manifest['split_kind'] = 'CustomLinkSplit'
            table.manifest_path.write_text(json.dumps(manifest))
        if prepared['case'] not in ('forecasting', 'static_link',
                                    'temporal_link'):
            table.generation_plan = replace(
                prepared['generation_plan'],
                split=("ManualSplit(col='SPLIT', train_value=0, "
                       "val_value=1, test_value=2)"))
        return table, list(reversed(rows))

    def test_final_answers_never_enter_fit_or_calibration(self):
        worker = self.worker()
        normal = self.isolation_context(worker)
        before = copy.deepcopy(normal['rows'])
        worker['isolate_final'].__globals__[
            '_reimport_rows'] = self.fake_reimport
        calibrated = worker['isolate_final'](normal, self.data / 'calibration',
                                             self.data / 'sealed')
        final = pq.read_table(self.data / 'sealed/rows.parquet').to_pylist()
        final_ids = {r['ENTITY1'] for r in final}
        self.assertEqual(len(final_ids), 6)
        self.assertFalse(
            final_ids
            & {r['ENTITY1']
               for r in self.fit_rows if r['SPLIT'] == 2})
        self.assertEqual([r for r in self.fit_rows if r['SPLIT'] != 2],
                         [r for r in before if r['SPLIT'] != 2])
        self.assertFalse(final_ids & set(calibrated['truth']['entity_ids']))
        self.assertEqual(normal['rows'], before)
        worker['verify_final_isolation'](calibrated)
        altered = copy.deepcopy(normal)
        for row in altered['rows']:
            if row['SPLIT'] == 2:
                row['TARGET1'] += 1000
        second = worker['isolate_final'](altered, self.data / 'calibration-2',
                                         self.data / 'sealed-2')
        self.assertEqual(set(second['truth']['entity_ids']),
                         set(calibrated['truth']['entity_ids']))

    def test_final_isolation_rejects_tampered_fit_and_reserved_payloads(self):
        worker = self.worker()
        normal = self.isolation_context(worker)
        worker['isolate_final'].__globals__[
            '_reimport_rows'] = self.fake_reimport
        calibrated = worker['isolate_final'](normal, self.data / 'calibration',
                                             self.data / 'sealed')
        for path in (self.data / 'sealed/rows.parquet',
                     calibrated['normal_table'].train_table_path /
                     'part.parquet'):
            before = path.read_bytes()
            path.write_bytes(before + b'tampered')
            with self.subTest(path=path), self.assertRaises(ValueError):
                worker['verify_final_isolation'](calibrated)
            path.write_bytes(before)
        calibrated['rows'][0]['TARGET1'] += 1
        with self.assertRaises(ValueError):
            worker['verify_final_isolation'](calibrated)

    def test_isolation_preserves_complete_forecasts_and_static_projection(self):
        for case in ('forecasting', 'static_link'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as path:
                saved = self.data
                self.data = Path(path)
                worker = self.worker()
                normal = self.isolation_context(worker, case)
                normal['graph'], normal['snapshot'] = object(), object()
                worker['isolate_final'].__globals__[
                    '_reimport_rows'] = self.fake_reimport
                calibrated = worker['isolate_final'](normal,
                                                     self.data / 'calibration',
                                                     self.data / 'sealed')
                final = pq.read_table(self.data /
                                      'sealed/rows.parquet').to_pylist()
                metadata = json.loads(
                    (self.data / 'sealed/descriptor.json').read_text())
                self.assertIs(calibrated['graph'], normal['graph'])
                self.assertIs(calibrated['snapshot'], normal['snapshot'])
                if case == 'forecasting':
                    self.assertTrue(
                        all(
                            len(row['future_labels']) == len(
                                row['autoregression_labels']) == 12
                            for row in final))
                    self.assertEqual(len(metadata['context']['horizons']), 12)
                else:
                    self.assertTrue(
                        all(row['exclude_target_keys'] == [10, 20]
                            for row in final))
                    self.assertTrue(
                        all(40 in values for values in metadata['context']
                            ['known'].values()))
                    self.assertTrue(
                        all(row == [False, False, True, False]
                            for row in calibrated['truth']['control_eligible']))
                    self.assertEqual(
                        metadata['partition']['native_import']['routing'],
                        'CustomLinkSplit')
                    self.assertEqual(
                        metadata['partition']['native_import']
                        ['source_generation_split'],
                        normal['normal_table'].generation_plan.split)
                    self.assertIs(calibrated['normal_table'].generation_plan,
                                  normal['normal_table'].generation_plan)
                self.data = saved

    def test_isolation_never_rerolls_for_invalid_binary_class_coverage(self):
        worker = self.worker()
        normal = self.isolation_context(worker, 'binary')
        final = worker['_final_ids']('binary', normal['rows'])
        for row in normal['rows']:
            if row['SPLIT'] == 2:
                row['TARGET1'] = row['ENTITY1'] in final
        worker['isolate_final'].__globals__[
            '_reimport_rows'] = self.fake_reimport
        with self.assertRaisesRegex(ValueError, 'both answer classes'):
            worker['isolate_final'](normal, self.data / 'calibration',
                                    self.data / 'sealed')
        self.assertFalse((self.data / 'sealed/descriptor.json').exists())

    def test_isolation_rejects_overlapping_seal_and_fit_directories(self):
        worker = self.worker()
        normal = self.isolation_context(worker)
        with self.assertRaisesRegex(ValueError, 'separate'):
            worker['isolate_final'](normal, self.data / 'fit',
                                    self.data / 'fit/final')

    def test_isolation_rejects_invalid_reserved_binary_coverage(self):
        worker = self.worker()
        normal = self.isolation_context(worker, 'binary')
        final = worker['_final_ids']('binary', normal['rows'])
        for row in normal['rows']:
            if row['SPLIT'] == 2 and row['ENTITY1'] in final:
                row['TARGET1'] = True
        worker['isolate_final'].__globals__[
            '_reimport_rows'] = self.fake_reimport
        with self.assertRaisesRegex(ValueError, 'both answer classes'):
            worker['isolate_final'](normal, self.data / 'calibration',
                                    self.data / 'sealed')

    def test_isolated_controls_rebind_native_files_and_keep_the_final_seal(
            self):
        worker = self.worker()
        normal = self.isolation_context(worker)
        worker['isolate_final'].__globals__[
            '_reimport_rows'] = self.fake_reimport
        calibrated = worker['isolate_final'](normal, self.data / 'calibration',
                                             self.data / 'sealed')
        original = (self.data / 'sealed/descriptor.json').read_bytes()
        control = worker['control_table'](calibrated, 0, self.data / 'control')
        worker['verify_final_isolation'](control)
        self.assertNotEqual(control['final_isolation'],
                            calibrated['final_isolation'])
        self.assertEqual((self.data / 'sealed/descriptor.json').read_bytes(),
                         original)
        metadata = json.loads((Path(control['final_isolation']['directory']) /
                               'descriptor.json').read_text())
        self.assertIs(metadata['final_control_answers_created'], False)
        self.assertEqual(metadata['parent'], calibrated['final_isolation'])
        final_ids = {
            r['ENTITY1']
            for r in pq.read_table(self.data /
                                   'sealed/rows.parquet').to_pylist()
        }
        self.assertFalse(
            final_ids
            & {r['ENTITY1']
               for r in self.fit_rows if r['SPLIT'] == 2})
        wrong = dict(control, final_isolation=calibrated['final_isolation'])
        with self.assertRaises(ValueError):
            worker['verify_final_isolation'](wrong)

    def test_static_calibration_controls_do_not_repeat_prior_targets(self):
        rows = [
            dict(ENTITY1=1, SPLIT=stage, TIMESTAMP=ANCHOR, TARGET1=[10],
                 exclude_target_keys=[10],
                 __kumo_internal_frequency_column__=[2]) for stage in range(3)
        ]
        drawn = self.worker()['static_control_targets'](rows,
                                                        list(range(10, 110,
                                                                   10)),
                                                        {
                                                            1: {10, 20, 30, 40}
                                                        }, 0)
        self.assertEqual(len({r['TARGET1'][0] for r in drawn}), 3)
        for index, row in enumerate(drawn):
            self.assertEqual(set(row['exclude_target_keys']),
                             {r['TARGET1'][0]
                              for r in drawn[:index]})
            self.assertEqual(row['__kumo_internal_frequency_column__'], [2])

    def test_isolated_static_controls_use_absent_pools_and_record_the_projection(
            self):
        worker = self.worker()
        normal = self.isolation_context(worker, 'static_link')
        pq.write_table(pa.table(dict(merchant_id=list(range(10, 110, 10)))),
                       self.data / 'merchants/part.parquet')
        normal['truth'], normal['prediction_rows'] = worker[
            'prediction_context']('static_link', normal['rows'], self.data)
        self.table('visible-purchases',
                   dict(customer_id=list(range(12)), merchant_id=[40] * 12))
        self.table('purchases',
                   dict(customer_id=list(range(12)), merchant_id=[70] * 12))
        visible = self.data / 'visible-purchases'
        normal['graph'] = SimpleNamespace(
            _named_tables={
                'purchases':
                SimpleNamespace(
                    table_name='visible', connector=SimpleNamespace(
                        source_table_for=lambda name: SimpleNamespace(path=str(
                            visible))))
            })
        normal['graph_counts'] = {'purchases': 24}
        worker['isolate_final'].__globals__[
            '_reimport_rows'] = self.fake_reimport
        calibrated = worker['isolate_final'](normal, self.data / 'calibration',
                                             self.data / 'sealed')
        control = worker['control_table'](calibrated, 0, self.data / 'control')
        worker['verify_final_isolation'](control)
        known = {i: {40} for i in range(12)}
        for row in calibrated['rows']:
            known[row['ENTITY1']].update(row['TARGET1'])
        self.assertTrue(
            all(
                set(row['TARGET1']).isdisjoint(known[row['ENTITY1']])
                for row in control['rows']))
        self.assertEqual(control['identity']['control_pool']['method'],
                         'absent_from_visible_calibration_relationships')
        pq.write_table(
            pa.table(dict(customer_id=list(range(12)), merchant_id=[80] * 12)),
            self.data / 'purchases/part.parquet')
        repeated = worker['control_table'](calibrated, 0,
                                           self.data / 'repeat-control')
        self.assertEqual(control['rows'], repeated['rows'])
        self.assertEqual(control['identity'], repeated['identity'])
        self.assertEqual(
            control['identity']['control_pool']['projected_graph']
            ['node_counts'], {'purchases': 24})
        self.assertEqual(control['identity']['control_pool']['null_method'],
                         'uniform_eligible_items')
        with self.fixture_imports(), patch('importlib.metadata.version',
                                           return_value='corrected-producer'):
            for seed, prepared in ((None, calibrated), (0, control)):
                fixture = self.data / f'frozen-static-{seed}'
                worker['freeze_control'](prepared, fixture)
                worker['load_control'].__globals__['_rows'] = (
                    lambda table, directory: pq.read_table(
                        table.train_table_path).to_pylist())
                loaded = worker['load_control'](
                    normal if seed is None else calibrated, seed, fixture,
                    self.data / f'loaded-static-{seed}')
                self.assertEqual(loaded['identity'], prepared['identity'])
                for name in ('targets', 'baseline_predictions',
                             'control_eligible'):
                    self.assertEqual(
                        dict(
                            zip(loaded['truth']['entity_ids'],
                                loaded['truth'][name])),
                        dict(
                            zip(prepared['truth']['entity_ids'],
                                prepared['truth'][name])))
                self.assertEqual(loaded['truth']['known'],
                                 prepared['truth']['known'])
                worker['verify_final_isolation'](loaded)
                path = fixture / 'descriptor.json'
                original = path.read_bytes()
                changed = json.loads(original)
                changed['native_split_kind'] = 'ManualSplit'
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    worker['load_control'](
                        normal if seed is None else calibrated, seed, fixture,
                        self.data / f'wrong-routing-{seed}')
                path.write_bytes(original)
                if seed is not None:
                    changed = json.loads(original)
                    changed['identity']['control_pool'][
                        'method'] = 'uniform_absent_source_pairs'
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        worker['load_control'](calibrated, seed, fixture,
                                               self.data /
                                               'legacy-final-aware-pool')
                    path.write_bytes(original)

    def test_isolated_multilabel_control_is_frozen_and_loaded_without_reimport(
            self):
        worker = self.worker()
        normal = self.isolation_context(worker, 'multilabel')
        worker['isolate_final'].__globals__[
            '_reimport_rows'] = self.fake_reimport
        calibrated = worker['isolate_final'](normal, self.data / 'calibration',
                                             self.data / 'sealed')
        control = worker['control_table'](calibrated, 0, self.data / 'control')
        fixture = self.data / 'frozen'
        with self.fixture_imports(), patch('importlib.metadata.version',
                                           return_value='corrected-producer'):
            worker['freeze_control'](control, fixture)
            before = {
                str(p.relative_to(fixture)): p.read_bytes()
                for p in fixture.rglob('*') if p.is_file()
            }
            worker['load_control'].__globals__[
                '_rows'] = lambda table, directory: pq.read_table(
                    table.train_table_path).to_pylist()
            loaded = worker['load_control'](calibrated, 0, fixture,
                                            self.data / 'reference-export')
        worker['verify_final_isolation'](loaded)
        self.assertEqual(loaded['identity'], control['identity'])
        self.assertTrue(any(row['TARGET1'] == '' for row in loaded['rows']))
        self.assertEqual(
            before, {
                str(p.relative_to(fixture)): p.read_bytes()
                for p in fixture.rglob('*') if p.is_file()
            })

    def test_normal_multilabel_calibration_replays_without_reference_reimport(
            self):
        worker = self.worker()
        normal = self.isolation_context(worker, 'multilabel')
        worker['isolate_final'].__globals__[
            '_reimport_rows'] = self.fake_reimport
        calibrated = worker['isolate_final'](normal, self.data / 'calibration',
                                             self.data / 'sealed')
        fixture = self.data / 'frozen-normal'
        with self.fixture_imports(), patch('importlib.metadata.version',
                                           return_value='corrected-producer'):
            descriptor = worker['freeze_control'](calibrated, fixture)
            self.assertEqual(descriptor['arm'], 'normal')
            self.assertIsNone(descriptor['seed'])
            before = {
                str(p.relative_to(fixture)): p.read_bytes()
                for p in fixture.rglob('*') if p.is_file()
            }
            worker['load_control'].__globals__[
                '_rows'] = lambda table, directory: pq.read_table(
                    table.train_table_path).to_pylist()
            worker['load_control'].__globals__[
                '_reimport_rows'] = lambda *a, **k: self.fail(
                    'Reference cannot reimport normal multilabel rows')
            loaded = worker['load_control'](normal, None, fixture,
                                            self.data / 'consumer-export')
            self.assertEqual(loaded['identity'], calibrated['identity'])
            self.assertEqual(len(loaded['truth']['entity_ids']), 6)
            self.assertTrue(any(r['TARGET1'] == '' for r in loaded['rows']))
            worker['verify_final_isolation'](loaded)
            self.assertEqual(
                before, {
                    str(p.relative_to(fixture)): p.read_bytes()
                    for p in fixture.rglob('*') if p.is_file()
                })
            changed = dict(normal, rows=copy.deepcopy(normal['rows']))
            changed['rows'][0]['TARGET1'] = 'changed'
            with self.assertRaises(ValueError):
                worker['load_control'](changed, None, fixture,
                                       self.data / 'wrong-consumer')

    def test_nonlink_calibration_replays_exact_native_statistics(self):
        for case in ('binary', 'multiclass', 'regression', 'forecasting'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as path:
                saved, self.data = self.data, Path(path)
                self.addCleanup(setattr, self, 'data', saved)
                worker = self.worker()
                normal = self.isolation_context(worker, case)
                if case == 'regression':
                    normal['normal_table'].generation_plan = replace(
                        normal['normal_table'].generation_plan,
                        forecast_length=1, lag_timesteps=0,
                        year_over_year=False)
                if case == 'multiclass':
                    for row in normal['rows']:
                        row['TARGET1'] = row['ENTITY1'] % 2
                    normal['truth'], normal['prediction_rows'] = worker[
                        'prediction_context'](case, normal['rows'], self.data)
                worker['isolate_final'].__globals__[
                    '_reimport_rows'] = self.fake_reimport
                calibrated = worker['isolate_final'](normal,
                                                     self.data / 'calibration',
                                                     self.data / 'sealed')
                control = worker['control_table'](calibrated, 0,
                                                  self.data / 'control')
                with self.fixture_imports(), patch(
                        'importlib.metadata.version',
                        return_value='corrected-producer'):
                    for seed, prepared in ((None, calibrated), (0, control)):
                        fixture = self.data / f'frozen-{seed}'
                        worker['freeze_control'](prepared, fixture)
                        before = {
                            str(p.relative_to(fixture)): p.read_bytes()
                            for p in fixture.rglob('*') if p.is_file()
                        }
                        worker['load_control'].__globals__['_rows'] = (
                            lambda table, directory: pq.read_table(
                                table.train_table_path).to_pylist())
                        worker['load_control'].__globals__['_reimport_rows'] = (
                            lambda *a, **k: self.fail(
                                'Frozen native replay must not reimport'))
                        loaded = worker['load_control'](
                            normal if seed is None else calibrated, seed,
                            fixture, self.data / f'export-{seed}')
                        self.assertEqual(loaded['identity'],
                                         prepared['identity'])
                        worker['verify_final_isolation'](loaded)
                        if case == 'regression' and seed is None:
                            for name, value in (('forecast_length',
                                                 2), ('lag_timesteps', 1),
                                                ('year_over_year', True)):
                                changed = dict(
                                    normal, normal_table=copy.copy(
                                        normal['normal_table']))
                                changed[
                                    'normal_table'].generation_plan = replace(
                                        normal['normal_table'].generation_plan,
                                        **{name: value})
                                with self.subTest(
                                        field=name), self.assertRaises(
                                            ValueError):
                                    worker['load_control'](changed, None,
                                                           fixture, self.data /
                                                           f'wrong-{name}')
                        self.assertEqual(
                            before, {
                                str(p.relative_to(fixture)): p.read_bytes()
                                for p in fixture.rglob('*') if p.is_file()
                            })
                        if case == 'forecasting':
                            self.assertTrue(
                                all(
                                    len(row['future_labels']) == 12
                                    for row in loaded['rows']))

    def test_rehashed_native_test_parts_cannot_add_final_ids(self):
        worker = self.worker()
        normal = self.isolation_context(worker)
        worker['isolate_final'].__globals__[
            '_reimport_rows'] = self.fake_reimport
        calibrated = worker['isolate_final'](normal, self.data / 'calibration',
                                             self.data / 'sealed')
        staging = calibrated['normal_table'].staging_dir
        path = staging / 'test_seeds/part.parquet'
        final = pq.read_table(self.data / 'sealed/rows.parquet').to_pylist()[0]
        pq.write_table(
            pa.Table.from_pylist(pq.read_table(path).to_pylist() + [final]),
            path)
        descriptor = self.data / 'sealed/descriptor.json'
        metadata = json.loads(descriptor.read_text())
        metadata['fit_files'] = worker['_control_staging_hashes'](staging)
        content = json.dumps(metadata, indent=2, sort_keys=True).encode()
        descriptor.write_bytes(content)
        calibrated['final_isolation']['sha256'] = hashlib.sha256(
            content).hexdigest()
        with self.assertRaisesRegex(ValueError, 'native test'):
            worker['verify_final_isolation'](calibrated)

    def test_forecast_fingerprints_accept_only_equal_numeric_placeholder_casts(
            self):
        worker = self.worker()
        rows = [
            dict(ENTITY1=1, TIMESTAMP=ANCHOR, SPLIT=stage, TARGET1=1,
                 future_labels=[1] * 12, autoregression_labels=[0] * 12)
            for stage in range(3)
        ]
        imported = copy.deepcopy(rows)
        for row in imported:
            row['TARGET1'] = 1.0
        signature = worker['_stage_fingerprints']
        self.assertEqual(signature('forecasting', rows),
                         signature('forecasting', imported))
        for field in ('TARGET1', 'future_labels', 'autoregression_labels'):
            changed = copy.deepcopy(imported)
            if field == 'TARGET1':
                changed[0][field] = 1.0000000000000002
            else:
                changed[0][field][0] += 1
            self.assertNotEqual(signature('forecasting', rows),
                                signature('forecasting', changed))

    def test_link_staging_requires_declared_files_without_scalar_class_stats(
            self):
        worker = self.worker()
        normal, _ = self.frozen_context(worker)
        root = normal['normal_table'].staging_dir
        path = root / 'manifest.json'
        manifest = json.loads(path.read_text())
        del manifest['train_table_col_stats']
        path.write_text(json.dumps(manifest))
        (root / 'train_table_col_stats.json').unlink()
        with self.assertRaises(ValueError):
            worker['_control_staging_hashes'](root)
        hashes = worker['_control_staging_hashes'](root, link=True)
        self.assertNotIn('train_table_col_stats.json', hashes)
        manifest['train_table_col_stats'] = {
            'path': 'train_table_col_stats.json'
        }
        path.write_text(json.dumps(manifest))
        with self.assertRaises((ValueError, OSError)):
            worker['_control_staging_hashes'](root, link=True)

    def test_static_native_payload_keeps_zipped_targets_frequencies_and_stages(
            self):
        worker = self.worker()
        rows = [
            dict(ENTITY1=1, TIMESTAMP=ANCHOR, SPLIT=stage, TARGET1=targets,
                 __kumo_internal_frequency_column__=frequencies,
                 exclude_target_keys=prior)
            for stage, targets, frequencies, prior in ((0, [20, 10], [2, 1],
                                                        []), (1, [30], [7],
                                                              [10, 20]),
                                                       (2, [40], [9],
                                                        [10, 20, 30]))
        ]
        [payload] = worker['_static_link_payload'](rows[::-1])
        self.assertNotIn('exclude_target_keys', payload)
        self.assertNotIn('SPLIT', payload)
        triples = list(
            zip(payload['TARGET1'],
                payload['__kumo_internal_frequency_column__'],
                payload['LINK_STAGE']))
        self.assertEqual(triples, [(10, 1, '0'), (20, 2, '0'), (30, 7, '1'),
                                   (40, 9, '2')])
        for mutation in ('duplicate', 'frequency', 'exclusion'):
            changed = copy.deepcopy(rows)
            if mutation == 'duplicate':
                changed[1]['TARGET1'] = [10]
            elif mutation == 'frequency':
                changed[0]['__kumo_internal_frequency_column__'].pop()
            else:
                changed[1]['exclude_target_keys'] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                worker['_static_link_payload'](changed)

    def test_binary_target_audit_rejects_wrong_window_answers(self):
        worker = self.worker()
        self.table(
            'conversions',
            dict(
                user_id=[1, 2], conversion_time=[
                    ANCHOR + timedelta(days=1), ANCHOR + timedelta(days=8)
                ]))
        rows = [
            dict(ENTITY1=1, TIMESTAMP=ANCHOR, TARGET1=True),
            dict(ENTITY1=2, TIMESTAMP=ANCHOR, TARGET1=False)
        ]
        worker['audit_targets']('binary', self.data, rows)
        rows[0]['TARGET1'] = False
        with self.assertRaisesRegex(ValueError, 'source events'):
            worker['audit_targets']('binary', self.data, rows)

    def test_delivery_label_sets_come_from_the_requested_window(self):
        worker = self.worker()
        self.table(
            'orders',
            dict(user_id=[1, 1, 1], order_id=[1, 2, 3],
                 order_time=[ANCHOR + timedelta(days=i) for i in [1, 5, 15]],
                 daypart=['breakfast', 'lunch', 'late']))
        rows = [dict(ENTITY1=1, TIMESTAMP=ANCHOR, TARGET1='breakfast|lunch')]
        rows.append(dict(ENTITY1=2, TIMESTAMP=ANCHOR, TARGET1=''))
        for task in ('multilabel', 'multilabel_ranking'):
            worker['audit_targets'](task, self.data, rows)
        rows[0]['TARGET1'] = 'breakfast|late'
        with self.assertRaisesRegex(ValueError, 'source events'):
            worker['audit_targets']('multilabel', self.data, rows)

    def test_weekly_forecast_audits_future_and_observed_history(self):
        worker = self.worker()
        self.table(
            'orders',
            dict(user_id=[1] * 4, order_id=[1, 2, 3, 4],
                 order_time=[ANCHOR + timedelta(days=i) for i in [1, 8, 9, 80]],
                 daypart=['lunch'] * 4))
        future = [1, 2] + [0] * 9 + [1]
        rows = [
            dict(ENTITY1=1, TIMESTAMP=ANCHOR, TARGET1=1,
                 autoregression_labels=[0] * 12, future_labels=future)
        ]
        worker['audit_targets']('forecasting', self.data, rows)
        rows[0]['autoregression_labels'][0] = 1
        with self.assertRaisesRegex(ValueError, 'history'):
            worker['audit_targets']('forecasting', self.data, rows)

    def test_forecast_control_moves_whole_answers_without_moving_history(self):
        worker = self.worker()
        rows = [
            dict(ENTITY1=i, TIMESTAMP=ANCHOR, SPLIT=0, TARGET1=i,
                 autoregression_labels=[i + 10] * 12, future_labels=[i] * 12)
            for i in (1, 2)
        ]
        changed = worker['shuffle_targets']('forecasting', rows, 1)
        self.assertEqual([r['TARGET1'] for r in changed], [2, 1])
        self.assertEqual([r['future_labels'] for r in changed],
                         [[2] * 12, [1] * 12])
        for before, after in zip(rows, changed):
            for field in ('ENTITY1', 'TIMESTAMP', 'autoregression_labels'):
                self.assertEqual(before[field], after[field])
        self.assertEqual(rows[0]['TARGET1'], 1)

    def test_link_control_keeps_frequency_lists_with_their_targets(self):
        worker = self.worker()
        rows = [
            dict(ENTITY1=1, SPLIT=0, TIMESTAMP=ANCHOR, TARGET1=[10, 20],
                 __kumo_internal_frequency_column__=[2, 1]),
            dict(ENTITY1=2, SPLIT=0, TIMESTAMP=ANCHOR, TARGET1=[30],
                 __kumo_internal_frequency_column__=[4])
        ]
        changed = worker['shuffle_targets']('temporal_link', rows, 1)
        self.assertEqual([r['TARGET1'] for r in changed], [[30], [10, 20]])
        self.assertEqual(
            [r['__kumo_internal_frequency_column__'] for r in changed],
            [[4], [2, 1]])
        self.assertEqual([r['ENTITY1'] for r in changed], [1, 2])

    def test_temporal_link_control_keeps_targets_at_their_original_anchor(self):
        rows = [
            dict(ENTITY1=i, SPLIT=0, TIMESTAMP=ANCHOR + timedelta(days=i // 2),
                 TARGET1=[10 + i], __kumo_internal_frequency_column__=[i + 1])
            for i in range(4)
        ]
        changed = self.worker()['shuffle_targets']('temporal_link', rows, 1)
        for day in (0, 1):
            self.assertEqual(
                {(tuple(row['TARGET1']),
                  tuple(row['__kumo_internal_frequency_column__']))
                 for row in changed
                 if row['TIMESTAMP'] == ANCHOR + timedelta(days=day)},
                {((10 + i, ), (i + 1, ))
                 for i in (day * 2, day * 2 + 1)})
        self.assertEqual([row['ENTITY1'] for row in changed], [0, 1, 2, 3])
        self.assertNotEqual([row['TARGET1'] for row in changed],
                            [row['TARGET1'] for row in rows])

    def test_temporal_control_audit_accepts_window_births_and_rejects_invalid_rhs(
            self):
        self.table(
            'merchants',
            dict(
                merchant_id=[10, 20, 30, 40], created_time=[
                    ANCHOR + timedelta(days=i) for i in (1, 7, 8, -1)
                ], expired_time=[None, None, None, ANCHOR]))
        definition = SimpleNamespace(pkey='merchant_id',
                                     time_col='created_time',
                                     end_time_col='expired_time')
        audit = self.worker()['audit_temporal_link_targets']
        rows = [dict(ENTITY1=1, TIMESTAMP=ANCHOR, TARGET1=[10, 20])]
        audit(self.data, rows, 'merchants', definition)
        for invalid in (30, 40, 99):
            with self.subTest(target=invalid), self.assertRaisesRegex(
                    ValueError, 'target.*window'):
                audit(self.data, [dict(rows[0], TARGET1=[10, invalid])],
                      'merchants', definition)

    def test_static_control_requires_exclusion_preserving_native_proof(self):
        worker = self.worker()
        with self.assertRaisesRegex(NotImplementedError, 'static-link'):
            worker['shuffle_targets']('static_link', [], 0)

    def test_controls_do_not_mix_split_populations(self):
        worker = self.worker()
        rows = [dict(ENTITY1=i, SPLIT=i // 2, TARGET1=i) for i in range(6)]
        before = copy.deepcopy(rows)
        changed = worker['shuffle_targets']('regression', rows, 1)
        for stage in range(3):
            self.assertEqual(
                {r['TARGET1']
                 for r in changed if r['SPLIT'] == stage},
                {stage * 2, stage * 2 + 1})
        self.assertEqual(rows, before)

    def test_control_assignments_ignore_physical_row_order(self):
        worker = self.worker()
        for case in ('regression', 'temporal_link'):
            rows = [
                dict(ENTITY1=i // 2, SPLIT=i // 4,
                     TIMESTAMP=ANCHOR + timedelta(days=i % 2),
                     TARGET1=[i] if case == 'temporal_link' else i)
                for i in range(12)
            ]

            def assignments(values):
                return {
                    (r['SPLIT'], r['ENTITY1'], r['TIMESTAMP']): r['TARGET1']
                    for r in values
                }

            forward = worker['shuffle_targets'](case, rows, 0)
            reverse = worker['shuffle_targets'](case, rows[::-1], 0)
            with self.subTest(case=case):
                self.assertEqual(assignments(forward), assignments(reverse))

    def test_profile_rejects_changed_data_after_validation(self):
        worker = self.worker()
        path = self.data / 'adtech/pr'
        path.mkdir(parents=True)
        file = path / 'users.parquet'
        file.write_bytes(b'original bytes')
        manifest = dict(
            graph='adtech', tables=dict(users=dict(files=[
                dict(path='adtech/pr/users.parquet', bytes=file.stat().st_size,
                     sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            ])))
        text = json.dumps(manifest)
        (path / 'manifest.json').write_text(text)
        (path / '_VALIDATED.json').write_text(
            json.dumps(
                dict(ok=True, checksums_verified=True,
                     manifest_sha256=hashlib.sha256(
                         text.encode()).hexdigest())))
        worker['read_profile'](path, 'binary')
        file.write_bytes(b'changed bytes!')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            worker['read_profile'](path, 'binary')
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            worker['prepare_profile'](path, self.data, 'binary',
                                      self.data / 'out')

    def test_ranges_reuse_training_history_without_moving_heldout_windows(self):
        worker = self.worker()
        manifest = dict(time_window=dict(start='2024-01-01T00:00:00Z',
                                         end='2025-01-01T00:00:00Z'))
        self.assertEqual(worker['generation_ranges'](manifest, 'binary'),
                         [('2024-01-01T00:00:00', '2024-12-18T00:00:00'),
                          ('2024-12-18T00:00:00', '2024-12-25T00:00:00'),
                          ('2024-12-25T00:00:00', '2025-01-01T00:00:00')])

    def test_forecast_ranges_preserve_complete_twelve_week_heldout_windows(
            self):
        worker = self.worker()
        manifest = dict(time_window=dict(start='2024-01-01T00:00:00Z',
                                         end='2025-01-01T00:00:00Z'))
        ranges = worker['generation_ranges'](manifest, 'forecasting')
        self.assertEqual(ranges[0][0], '2024-01-01T00:00:00')
        for start, end in ranges[1:]:
            self.assertEqual(
                datetime.fromisoformat(end) - datetime.fromisoformat(start),
                timedelta(days=84))
        self.assertEqual(ranges[0][1], ranges[1][0])
        self.assertEqual(ranges[1][1], ranges[2][0])

    def test_chronology_accepts_complete_windows_at_native_boundaries(self):
        worker = self.worker()
        manifest = dict(time_window=dict(start='2024-01-01T00:00:00Z',
                                         end='2025-01-01T00:00:00Z'))
        for case in set(worker['CASES']) - {'static_link'}:
            ranges = worker['generation_ranges'](manifest, case)
            horizon = timedelta(days=worker['CASES'][case]['days'])
            rows = [
                dict(SPLIT=stage, TIMESTAMP=anchor)
                for stage, (start, end) in enumerate(ranges)
                for anchor in (datetime.fromisoformat(start),
                               datetime.fromisoformat(end) - horizon)
            ]
            with self.subTest(case=case):
                worker['validate_chronology'](case, manifest, rows)
        worker['validate_chronology']('static_link', {}, [])

    def test_chronology_rejects_answer_windows_crossing_every_stage_boundary(
            self):
        worker = self.worker()
        manifest = dict(time_window=dict(start='2024-01-01T00:00:00Z',
                                         end='2025-01-01T00:00:00Z'))
        for case in ('binary', 'forecasting'):
            ranges = worker['generation_ranges'](manifest, case)
            horizon = timedelta(days=worker['CASES'][case]['days'])
            rows = [
                dict(SPLIT=stage, TIMESTAMP=datetime.fromisoformat(start))
                for stage, (start, _) in enumerate(ranges)
            ]
            for stage, (_, end) in enumerate(ranges):
                changed = copy.deepcopy(rows)
                changed[stage]['TIMESTAMP'] = (datetime.fromisoformat(end) -
                                               horizon + timedelta(seconds=1))
                with self.subTest(case=case,
                                  stage=stage), self.assertRaisesRegex(
                                      ValueError, 'answer window.*stage'):
                    worker['validate_chronology'](case, manifest, changed)

    def test_chronology_rejects_wrong_stage_missing_stage_and_early_anchor(
            self):
        worker = self.worker()
        manifest = dict(time_window=dict(start='2024-01-01T00:00:00Z',
                                         end='2025-01-01T00:00:00Z'))
        ranges = worker['generation_ranges'](manifest, 'binary')
        rows = [
            dict(SPLIT=stage, TIMESTAMP=datetime.fromisoformat(start))
            for stage, (start, _) in enumerate(ranges)
        ]
        bad_stage = [dict(row) for row in rows]
        bad_stage[0]['SPLIT'] = 1
        bad_anchor = [dict(row) for row in rows]
        bad_anchor[1]['TIMESTAMP'] -= timedelta(seconds=1)
        for changed in (bad_stage, bad_anchor, rows[:2],
                        [dict(rows[0], SPLIT=3), *rows[1:]]):
            with self.subTest(rows=changed), self.assertRaisesRegex(
                    ValueError, 'stage'):
                worker['validate_chronology']('binary', manifest, changed)

    def test_context_accepts_native_training_history_with_fixed_evaluation(
            self):
        rows = [
            dict(ENTITY1=1, TIMESTAMP=ANCHOR + timedelta(days=7 * i),
                 TARGET1=bool(i == 2), SPLIT=0) for i in range(3)
        ] + [
            dict(ENTITY1=10 + label, TIMESTAMP=ANCHOR +
                 timedelta(days=21 + label), TARGET1=bool(label), SPLIT=1)
            for label in (0, 1)
        ] + [
            dict(ENTITY1=20 + label, TIMESTAMP=ANCHOR + timedelta(days=35),
                 TARGET1=bool(label), SPLIT=2) for label in (0, 1)
        ]
        truth, prediction = self.worker()['prediction_context']('binary', rows,
                                                                self.data)
        self.assertEqual(truth['entity_ids'], [20, 21])
        self.assertEqual(truth['targets'], [False, True])
        self.assertEqual(truth['baseline_predictions'], [1 / 3, 1 / 3])
        self.assertEqual(prediction, [
            dict(ENTITY1=i, TIMESTAMP=ANCHOR + timedelta(days=35))
            for i in (20, 21)
        ])

    def test_binary_context_has_hidden_answers_and_training_only_baseline(self):
        rows = [
            dict(ENTITY1=stage * 2 + label, TIMESTAMP=ANCHOR,
                 TARGET1=bool(label), SPLIT=stage) for stage in range(3)
            for label in (0, 1)
        ]
        truth, prediction = self.worker()['prediction_context']('binary', rows,
                                                                self.data)
        self.assertEqual(truth['entity_ids'], [4, 5])
        self.assertEqual(truth['targets'], [False, True])
        self.assertEqual(truth['baseline_predictions'], [.5, .5])
        self.assertEqual(prediction,
                         [dict(ENTITY1=i, TIMESTAMP=ANCHOR) for i in (4, 5)])

    def test_context_rejects_multiple_evaluation_anchors(self):
        rows = [
            dict(ENTITY1=stage * 2 + label, TIMESTAMP=ANCHOR,
                 TARGET1=bool(label), SPLIT=stage) for stage in range(3)
            for label in (0, 1)
        ]
        rows[-1]['TIMESTAMP'] += timedelta(days=1)
        with self.assertRaisesRegex(ValueError, 'one anchor'):
            self.worker()['prediction_context']('binary', rows, self.data)

    def test_forecast_prediction_only_carries_observed_history(self):
        rows = [
            dict(ENTITY1=stage, SPLIT=stage, TIMESTAMP=ANCHOR, TARGET1=1,
                 future_labels=[1] * 12, autoregression_labels=[2] * 12)
            for stage in range(3)
        ]
        truth, prediction = self.worker()['prediction_context']('forecasting',
                                                                rows, self.data)
        self.assertEqual(truth['targets'], [[1] * 12])
        self.assertEqual(truth['baseline_predictions'], [[2] * 12])
        self.assertEqual(truth['horizons'],
                         [ANCHOR + timedelta(days=7 * i) for i in range(12)])
        self.assertEqual(set(prediction[0]),
                         {'ENTITY1', 'TIMESTAMP', 'autoregression_labels'})

    @unittest.skipUnless(os.environ.get('KUMO_SCALE_NATIVE_BUNDLE'),
                         'Requires the prepared PR 172 native bundle')
    def test_temporal_controls_preserve_window_targets_and_native_ids(self):
        worker = self.worker()
        bundle = Path(os.environ['KUMO_SCALE_NATIVE_BUNDLE'])
        normal = worker['prepare_profile'](bundle / 'data/commerce/pr',
                                           bundle / 'scale-bench',
                                           'temporal_link',
                                           self.data / 'normal')
        plan = normal['normal_table'].generation_plan
        self.assertTrue(plan.split.startswith('TimeRangeSplit('))
        for seed in range(5):
            control = worker['control_table'](normal, seed,
                                              self.data / f'control-{seed}')
            self.assertEqual(control['normal_table'].generation_plan, plan)
            worker['validate_reimport']('temporal_link',
                                        worker['shuffle_targets'](
                                            'temporal_link', normal['rows'],
                                            seed), control['rows'])
            self.assertEqual(control['identity']['temporal_control_import'],
                             'audited_window_targets_and_native_ids')
            self.assertNotIn(
                'SPLIT',
                pq.read_schema(self.data /
                               f'control-{seed}/data/control.parquet').names)

    @unittest.skipUnless(os.environ.get('KUMO_SCALE_NATIVE_BUNDLE'),
                         'Requires the prepared PR 172 native bundle')
    def test_forecast_controls_reuse_native_time_ranges_and_exact_payloads(
            self):
        worker = self.worker()
        bundle = Path(os.environ['KUMO_SCALE_NATIVE_BUNDLE'])
        normal = worker['prepare_profile'](bundle / 'data/delivery/pr',
                                           bundle / 'scale-bench',
                                           'forecasting', self.data / 'normal')
        plan = normal['normal_table'].generation_plan
        self.assertTrue(plan.split.startswith('TimeRangeSplit('))
        for seed in range(5):
            control = worker['control_table'](normal, seed,
                                              self.data / f'control-{seed}')
            actual = control['normal_table'].generation_plan
            self.assertEqual(actual.split, plan.split)
            self.assertEqual(actual.timeframe_step, 7)
            self.assertEqual(actual.forecast_length, 12)
            self.assertEqual(actual.lag_timesteps, 12)
            self.assertIs(actual.year_over_year, False)
            self.assertIs(control['generation_plan'], actual)
            expected = worker['shuffle_targets']('forecasting', normal['rows'],
                                                 seed)
            worker['validate_reimport']('forecasting', expected,
                                        control['rows'])
            self.assertEqual([
                sum(row['SPLIT'] == stage for row in control['rows'])
                for stage in range(3)
            ], [
                sum(row['SPLIT'] == stage for row in normal['rows'])
                for stage in range(3)
            ])
            self.assertNotIn(
                'SPLIT',
                pq.read_schema(self.data /
                               f'control-{seed}/data/control.parquet').names)

        changed = dict(
            normal, generation_plan=replace(normal['generation_plan'],
                                            train_start_offset=0))
        with self.assertRaisesRegex(ValueError, 'generation plan'):
            worker['control_table'](changed, 0, self.data / 'changed-offset')

    def test_static_projection_removes_heldout_pairs_from_every_direct_relation(
            self):
        for name in ('purchases', 'reviews', 'views', 'favs'):
            self.table(
                name,
                dict(customer_id=[1, 1, 2], merchant_id=[10, 20, 20],
                     other=[1, 2, 3]))
        self.table('merchants', dict(merchant_id=[10, 20]))
        rows = [
            dict(ENTITY1=1, TARGET1=[10], SPLIT=0),
            dict(ENTITY1=1, TARGET1=[20], SPLIT=2)
        ]
        before = (self.data / 'purchases/part.parquet').read_bytes()
        result = self.worker()['project_static_sources'](
            self.data, ['purchases', 'reviews', 'views', 'favs', 'merchants'],
            rows, self.data / 'projected')
        self.assertEqual(set(result), {'purchases', 'reviews', 'views', 'favs'})
        for path in result.values():
            table = pq.read_table(path)
            self.assertEqual(table.column_names,
                             ['customer_id', 'merchant_id', 'other'])
            self.assertEqual(table.to_pylist(), [
                dict(customer_id=1, merchant_id=10, other=1),
                dict(customer_id=2, merchant_id=20, other=3)
            ])
        self.assertEqual((self.data / 'purchases/part.parquet').read_bytes(),
                         before)

    def test_static_payload_retains_native_anchors_and_observed_pair_frequencies(
            self):
        self.table('purchases',
                   dict(customer_id=[1, 1, 1, 2], merchant_id=[10, 10, 20, 20]))
        rows = self.worker()['_static_payload'](self.data, None, {
            1: ANCHOR,
            2: ANCHOR
        })
        self.assertEqual(
            rows[0],
            dict(ENTITY1=1, TIMESTAMP=ANCHOR, TARGET1=[10, 20],
                 __kumo_internal_frequency_column__=[2, 1]))

    def test_static_prediction_retains_its_native_anchor(self):
        self.table('merchants', dict(merchant_id=[10, 20, 30]))
        rows = [
            dict(
                ENTITY1=1, SPLIT=stage, TIMESTAMP=ANCHOR,
                TARGET1=[10 * (stage + 1)], exclude_target_keys=[
                    10 * (index + 1) for index in range(stage)
                ]) for stage in range(3)
        ]
        _, prediction = self.worker()['prediction_context']('static_link', rows,
                                                            self.data)
        self.assertEqual(prediction, [dict(ENTITY1=1, TIMESTAMP=ANCHOR)])

    def test_static_control_uses_only_absent_pairs_and_keeps_source_payload_size(
            self):
        rows = [
            dict(ENTITY1=1, TIMESTAMP=ANCHOR, TARGET1=[10, 20],
                 __kumo_internal_frequency_column__=[9, 3])
        ]
        known = {1: {10, 20, 30}}
        catalog = list(range(10, 100, 10))
        draw = self.worker()['static_control_targets']
        values = [draw(rows, catalog, known, seed)[0] for seed in range(5)]
        self.assertGreater(len({tuple(row['TARGET1']) for row in values}), 1)
        for row in values:
            self.assertEqual(len(set(row['TARGET1'])), 2)
            self.assertFalse(set(row['TARGET1']) & known[1])
            self.assertEqual(row['__kumo_internal_frequency_column__'], [9, 3])
            self.assertEqual(row['TIMESTAMP'], ANCHOR)
        self.assertEqual(rows[0]['TARGET1'], [10, 20])
        with self.assertRaisesRegex(ValueError, 'absent'):
            draw(rows, [10, 20, 30, 40], known, 0)

    def test_static_control_masks_source_neighbors_and_native_prior_targets(
            self):
        self.table('merchants', dict(merchant_id=[10, 20, 30, 40, 50]))
        rows = [
            dict(
                ENTITY1=1, SPLIT=stage, TIMESTAMP=ANCHOR,
                TARGET1=[10 * (stage + 3)], exclude_target_keys=[
                    10 * (index + 3) for index in range(stage)
                ]) for stage in range(3)
        ]
        truth, _ = self.worker()['prediction_context']('static_link', rows,
                                                       self.data, source_known={
                                                           1: {10, 20}
                                                       })
        self.assertEqual(truth['known'], {'1': [10, 20, 30, 40]})
        self.assertEqual(truth['control_eligible'], [[False] * 4 + [True]])

    def test_snapshot_counts_reject_native_data_loss_and_use_projected_rows(
            self):
        snapshot = self.data / 'snapshot'
        snapshot.mkdir()
        report = snapshot / 'ingest_stats.json'
        report.write_text(
            json.dumps(dict(stats=dict(node_counts=dict(purchases=2)))))
        manifest = dict(tables=dict(purchases=dict(rows=3)))
        check = self.worker()['validate_snapshot_counts']
        with self.assertRaisesRegex(ValueError, 'populations'):
            check(snapshot, ['purchases'], manifest)
        self.table('projected', dict(customer_id=[1, 2], merchant_id=[10, 20]))
        self.assertEqual(
            check(snapshot, ['purchases'], manifest,
                  {'purchases': self.data / 'projected/part.parquet'}),
            dict(purchases=2))

    def test_reimport_ignores_row_order_but_rejects_changed_payloads(self):
        rows = [
            dict(ENTITY1=i, TIMESTAMP=ANCHOR, SPLIT=0, TARGET1=[10, 20],
                 __kumo_internal_frequency_column__=[2, 1]) for i in (1, 2)
        ]
        check = self.worker()['validate_reimport']
        check('temporal_link', rows, rows[::-1])
        changed = copy.deepcopy(rows)
        changed[0]['__kumo_internal_frequency_column__'][0] = 3
        with self.assertRaisesRegex(ValueError, 'payloads'):
            check('temporal_link', rows, changed)

    def frozen_context(self, worker, case='multilabel'):
        staging = self.data / 'source'
        staging.mkdir()
        manifest = dict(time_window=dict(start='2024-01-01T00:00:00Z',
                                         end='2025-01-01T00:00:00Z'))
        ranges = worker['generation_ranges'](manifest, case)
        rows = [
            dict(ENTITY1=stage * 2 + label,
                 TIMESTAMP=datetime.fromisoformat(ranges[stage][0]),
                 SPLIT=stage, TARGET1='breakfast' if label else '')
            for stage in range(3) for label in (0, 1)
        ]
        control = worker['shuffle_targets'](case, rows, 0)
        table_dir = staging / 'tt.parquet'
        table_dir.mkdir()
        pq.write_table(pa.Table.from_pylist(control),
                       table_dir / 'part.parquet')
        stages = []
        for split, name in enumerate(('train', 'val', 'test')):
            directory = staging / f'{name}_seeds'
            directory.mkdir()
            pq.write_table(
                pa.Table.from_pylist(
                    [row for row in control if row['SPLIT'] == split]),
                directory / 'part.parquet')
            stages.append(dict(key=name, parts=[f'{name}_seeds/part.parquet']))
        (staging / 'tt_ingest_spec.json').write_text(
            json.dumps(
                dict(schema_version='diskgraph.tt_ingest_spec.v3',
                     split_stages=stages)))
        (staging / 'manifest.json').write_text(
            json.dumps(
                dict(
                    version=3, train_table=dict(
                        path='tt.parquet',
                        parts=[dict(path='tt.parquet/part.parquet')]),
                    stats=dict(path='stats.json'),
                    split_stats=dict(path='split_stats.json'),
                    train_table_col_stats=dict(
                        path='train_table_col_stats.json'))))
        for name in ('stats.json', 'split_stats.json',
                     'train_table_col_stats.json'):
            (staging / name).write_text('{}')
        plan = FixturePlan('TimeRangeSplit([])')
        table = SimpleNamespace(
            query=worker['CASES'][case]['query'], staging_dir=staging,
            train_table_path=table_dir, manifest_path=staging / 'manifest.json',
            required_stages=('train', 'val', 'test'),
            generation_plan=replace(plan, train_start_offset='inferred',
                                    train_end_offset='inferred'))
        normal = dict(
            case=case, rows=rows, profile_dir=self.data, query_text=table.query,
            normal_table=table, generation_plan=plan, manifest=manifest,
            identity=dict(case=case, manifest_sha256='profile',
                          scale_source_sha256='source', control_seed=None))
        changed = copy.copy(table)
        changed.required_stages = ('train', 'val')
        changed.generation_plan = replace(
            plan, split=("ManualSplit(col='SPLIT', train_value=0, "
                         'val_value=1, test_value=2)'))
        prepared = dict(normal, rows=control, normal_table=changed,
                        generation_plan=changed.generation_plan,
                        identity=dict(normal['identity'], control_seed=0))
        return normal, prepared

    def fixture_imports(self):
        engine = SimpleNamespace(
            TrainingTable=lambda **values: SimpleNamespace(**values))
        return patch.dict(
            'sys.modules', {
                'kumoapi.json_serde':
                SimpleNamespace(to_json_dict=asdict),
                'kumo_core.io.json':
                SimpleNamespace(from_json_dict=lambda value, cls: cls(**value)),
                'kumo_core.model_plan':
                SimpleNamespace(TrainingTableGenerationPlan=FixturePlan),
                'kumo':
                SimpleNamespace(engine=engine),
                'kumo.engine':
                engine
            })

    def test_frozen_control_preserves_bytes_identity_and_hidden_answers(self):
        for case in ('multilabel', 'multilabel_ranking'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as path:
                saved = self.data
                self.data = Path(path)
                worker = self.worker()
                normal, control = self.frozen_context(worker, case)
                fixture = self.data / 'fixture'
                with self.fixture_imports(), patch('importlib.metadata.version',
                                                   return_value='corrected'):
                    worker['freeze_control'](control, fixture)
                    original = {
                        str(path.relative_to(fixture)): path.read_bytes()
                        for path in fixture.rglob('*') if path.is_file()
                    }
                    worker['load_control'].__globals__['_rows'] = (
                        lambda table, directory: pq.read_table(
                            table.train_table_path).to_pylist())
                    loaded = worker['load_control'](normal, 0, fixture,
                                                    self.data / 'export')
                self.assertEqual(loaded['identity'], control['identity'])
                self.assertEqual(loaded['generation_plan'],
                                 control['generation_plan'])
                self.assertEqual(loaded['rows'], control['rows'])
                self.assertEqual(loaded['truth']['classes'], ['breakfast'])
                self.assertTrue(
                    all(
                        set(row) == {'ENTITY1', 'TIMESTAMP'}
                        for row in loaded['prediction_rows']))
                self.assertEqual(
                    original, {
                        str(path.relative_to(fixture)): path.read_bytes()
                        for path in fixture.rglob('*') if path.is_file()
                    })
                self.assertNotIn('frozen_control', loaded['identity'])
                self.assertEqual(loaded['frozen_control']['producer_version'],
                                 'corrected')
                self.data = saved

    def test_frozen_control_rejects_matching_payloads_with_leaked_answer_windows(
            self):
        worker = self.worker()
        normal, control = self.frozen_context(worker)
        fixture = self.data / 'fixture'
        with self.fixture_imports(), patch('importlib.metadata.version',
                                           return_value='corrected'):
            worker['freeze_control'](control, fixture)
            validation_anchor = normal['rows'][2]['TIMESTAMP']
            for context in (normal, control):
                for row in context['rows'][:2]:
                    row['TIMESTAMP'] = validation_anchor
            worker['load_control'].__globals__['_rows'] = (
                lambda table, directory: control['rows'])
            with self.assertRaisesRegex(ValueError, 'answer window.*stage'):
                worker['load_control'](normal, 0, fixture, self.data / 'export')

    def test_frozen_control_rejects_changed_files_paths_and_assignment(self):
        worker = self.worker()
        normal, control = self.frozen_context(worker)
        fixture = self.data / 'fixture'
        with self.fixture_imports(), patch('importlib.metadata.version',
                                           return_value='corrected'):
            with self.assertRaisesRegex(ValueError, 'overlap'):
                worker['freeze_control'](
                    control, control['normal_table'].staging_dir / 'frozen')
            worker['freeze_control'](control, fixture)
            load = worker['load_control']
            load.__globals__['_rows'] = lambda table, directory: pq.read_table(
                table.train_table_path).to_pylist()
            for changed in (
                    dict(normal, query_text='different query'),
                    dict(
                        normal, identity=dict(normal['identity'],
                                              manifest_sha256='changed')),
                    dict(
                        normal, normal_table=SimpleNamespace(
                            **{
                                **vars(normal['normal_table']), 'generation_plan':
                                replace(normal['generation_plan'],
                                        timeframe_step=7)
                            }))):
                with self.subTest(
                        changed=changed), self.assertRaises(ValueError):
                    load(changed, 0, fixture, self.data / 'export')
            with self.assertRaisesRegex(ValueError, 'outside'):
                load(normal, 0, fixture, fixture / 'staging/export')
            with self.assertRaises(ValueError):
                load(normal, 1, fixture, self.data / 'export')
            target = fixture / 'staging/train_table_col_stats.json'
            target.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                load(normal, 0, fixture, self.data / 'export')
            target.write_text('{}')
            link = fixture / 'staging/original-producer'
            link.symlink_to(control['normal_table'].staging_dir,
                            target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'links'):
                load(normal, 0, fixture, self.data / 'export')
            link.unlink()
            spec = fixture / 'staging/tt_ingest_spec.json'
            original = spec.read_bytes()
            changed = json.loads(original)
            changed['split_stages'][0]['parts'] = ['/original/producer.parquet']
            spec.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'path'):
                load(normal, 0, fixture, self.data / 'export')
            spec.write_bytes(original)
            worker['load_control'].__globals__['_rows'] = (
                lambda table, directory:
                [dict(row, TARGET1='wrong') for row in control['rows']])
            with self.assertRaisesRegex(ValueError, 'payloads'):
                load(normal, 0, fixture, self.data / 'export')

    def test_frozen_staging_rejects_a_link_to_the_original_producer(self):
        worker = self.worker()
        _, control = self.frozen_context(worker)
        link = self.data / 'frozen-staging'
        link.symlink_to(control['normal_table'].staging_dir,
                        target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'links'):
            worker['_control_staging_hashes'](link)

    def test_frozen_control_rejects_explicit_offset_or_missing_stage(self):
        worker = self.worker()
        normal, control = self.frozen_context(worker)
        fixture = self.data / 'fixture'
        with self.fixture_imports(), patch('importlib.metadata.version',
                                           return_value='corrected'):
            worker['freeze_control'](control, fixture)
            descriptor = fixture / 'descriptor.json'
            original = json.loads(descriptor.read_text())
            for value in (dict(
                    original, generation_plan=dict(original['generation_plan'],
                                                   train_start_offset=3)),
                          dict(original, required_stages=['train'])):
                descriptor.write_text(json.dumps(value))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    worker['load_control'](normal, 0, fixture,
                                           self.data / 'export')

    def test_frozen_control_rejects_missing_or_unrelated_fixtures(self):
        worker = self.worker()
        normal, control = self.frozen_context(worker)
        with self.assertRaises(FileNotFoundError):
            worker['load_control'](normal, 0, self.data / 'missing', self.data)
        with self.fixture_imports():
            for invalid in (dict(control, case='binary'),
                            dict(control, identity=normal['identity'])):
                with self.assertRaises(ValueError):
                    worker['freeze_control'](invalid, self.data / 'invalid')


if __name__ == '__main__':
    unittest.main()
