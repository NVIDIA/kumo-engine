#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the audited Adoption samples with one installed GPU wheel."""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from wheel_gpu_learning import measure_consistency, observe_gpu, verify_devices
from wheel_learning import score_predictions, verify_plan
from wheel_measurements import (
    Measurements,
    TrainingWindow,
    binary_split_counts,
    fixed_loaders,
    measure_recommendations,
    verify_training_work,
)

RECIPES = {
    'affirm': dict(
        query=('PREDICT COUNT(transaction_events.* WHERE '
               'transaction_events.target_label = 1, 0, 30, days) > 0 '
               'FOR EACH account_profile.account_id'),
        prediction_time='2026-08-30', metric='test_auroc'),
    'sothebys': dict(
        query=('PREDICT LIST_DISTINCT(transaction_events.item_id, 0, 30, days) '
               'RANK TOP 10 FOR EACH entity_profile.entity_id'),
        prediction_time='2026-06-15', metric='test_recall@10'),
}
for size in ('small', 'medium'):
    RECIPES['doordash_' + size] = dict(
        query=('PREDICT LIST_DISTINCT(orders.BUSINESS_ID, 0, 7, days) '
               'RANK TOP 12 FOR EACH users.CONSUMER_ID'),
        prediction_time='2026-05-11', metric='test_recall@10', top_k=12)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str,
                                     allow_nan=False) + '\n')


def local_graph_definition(payload):
    definition = copy.deepcopy(payload['train_pquery']['graph'])
    for name, table in definition['tables'].items():
        table['source_table'] = {'path': name}
    return definition


def target_time_ranges(target_end):
    end = datetime.fromisoformat(target_end)
    test_start = end - timedelta(days=7)
    validation_start = test_start - timedelta(days=7)
    train_start = validation_start - timedelta(days=30)
    return [[start.date().isoformat(), stop.date().isoformat()]
            for start, stop in ((train_start, validation_start),
                                (validation_start, test_start), (test_start, end))]


def verify_population(frame, expected_ids, *, top_k=None, class_ids=None):
    expected = {str(value) for value in expected_ids}
    if not expected or len(expected) != len(expected_ids):
        raise ValueError('Prediction inputs are empty or duplicated')
    if {str(value) for value in frame['ENTITY'].unique()} != expected:
        raise ValueError('Prediction entity IDs differ from the requested population')
    keys = ['ENTITY'] if top_k is None else ['ENTITY', 'CLASS']
    if frame.duplicated(keys).any() or len(frame) != len(expected) * (top_k or 1):
        raise ValueError('Prediction counts or unique recommendations are incomplete')
    if top_k is not None and not frame.groupby('ENTITY').size().eq(top_k).all():
        raise ValueError('An entity has an incomplete recommendation count')
    if class_ids is not None and not {str(value) for value in frame['CLASS'].unique()} <= {
            str(value) for value in class_ids}:
        raise ValueError('Recommendations contain unknown class IDs')


def score_customer(dataset, frame, answers):
    import numpy as np
    import pandas as pd

    def key(entity, stamp):
        return str(entity), pd.Timestamp(stamp).isoformat()

    truth = {key(row.ENTITY1, row.TIMESTAMP): row.TARGET1
             for row in answers.itertuples()}
    if not truth or len(truth) != len(answers):
        raise ValueError('Evaluation answers are empty or duplicated')
    frame = frame.copy()
    if 'TIMESTAMP' not in frame:
        entities = answers['ENTITY1'].astype(str)
        if entities.duplicated().any() or set(frame['ENTITY'].astype(str)) != set(entities):
            raise ValueError('Missing timestamps make evaluation anchors ambiguous')
        anchors = dict(zip(entities, answers['TIMESTAMP']))
        frame['TIMESTAMP'] = frame['ENTITY'].astype(str).map(anchors)
    pairs = zip(frame['ENTITY'], frame['TIMESTAMP'])
    frame['evaluation_key'] = [key(entity, stamp) for entity, stamp in pairs]
    if set(frame['evaluation_key']) != set(truth):
        raise ValueError('Evaluation predictions omit or add entity/time keys')
    if dataset == 'affirm':
        if frame['evaluation_key'].duplicated().any():
            raise ValueError('Duplicate binary predictions')
        labels = [truth[item] for item in truth]
        binary_split_counts(labels)
        scored = frame.assign(ENTITY=frame['evaluation_key'],
                              TARGET=[truth[item] for item in frame['evaluation_key']])
        return score_predictions(scored, truth)
    if (frame.duplicated(['evaluation_key', 'CLASS']).any()
            or not np.isfinite(frame['SCORE']).all()):
        raise ValueError('Duplicate or nonfinite recommendation predictions')
    expected = {item: {str(value) for value in values}
                for item, values in truth.items()}
    top_k = RECIPES[dataset].get('top_k', 10)
    if (any(not values for values in expected.values())
            or frame['evaluation_key'].value_counts().max() > top_k):
        raise ValueError('Invalid top-K recommendation result')
    hits = dict.fromkeys(expected, 0)
    for item, label in zip(frame['evaluation_key'], frame['CLASS'].astype(str)):
        hits[item] += label in expected[item]
    return float(np.mean([hits[item] / len(values)
                          for item, values in expected.items()]))


def build_graph(kumo, dataset, data, payload=None):
    from kumoapi.graph import Edge
    from kumoapi.typing import Stype

    connector = kumo.FileConnector(root_dir=str(data))
    if dataset.startswith('doordash_'):
        from kumostress.cases.direct_local import _graph

        return _graph(local_graph_definition(payload), {'s3': connector})
    if dataset == 'affirm':
        parents = dict(account_profile='account_id', network_profile='network_id',
                       device_profile='device_id', payment_profile='payment_id',
                       location_profile='location_id', contact_profile='contact_id')
        specs = {name: (key, 'created_date' if name == 'account_profile' else None)
                 for name, key in parents.items()}
        specs['transaction_events'] = ('event_id', 'event_date')
        links = [('transaction_events', key, name)
                 for name, key in parents.items()]
    else:
        specs = dict(entity_profile=('entity_id', None),
                     item_profile=('item_id', 'start_date'),
                     browsing_events=(None, 'event_date'),
                     transaction_events=(None, 'event_date'))
        links = [(event, key, parent)
                 for event in ('browsing_events', 'transaction_events')
                 for key, parent in (('entity_id', 'entity_profile'),
                                     ('item_id', 'item_profile'))]
    tables = {}
    for name, (key, timestamp) in specs.items():
        extra = {'end_time_column': 'end_date'} if name == 'item_profile' else {}
        table = kumo.Table.from_source_table(
            connector[name + '.parquet'], primary_key=key,
            time_column=timestamp, **extra).infer_metadata()
        for child, foreign_key, _ in links:
            if child == name:
                table = table.with_column(foreign_key, stype=Stype.ID)
        tables[name] = table
    graph = kumo.Graph(tables=tables, edges=[Edge(*link) for link in links])
    graph.validate()
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=RECIPES, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--frozen-plan', type=Path, required=True)
    parser.add_argument('--resource-helper', type=Path, required=True)
    parser.add_argument('--graph-recipe', type=Path)
    parser.add_argument('--prepared-manifest', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--loader-workers', type=int, default=8)
    parser.add_argument('--training-probe', action='store_true',
                        help='Train and score validation answers, then stop before operational prediction')
    args = parser.parse_args()
    if args.seed < 0 or args.loader_workers < 1:
        parser.error('Seed must be nonnegative and loader workers must be positive')
    doordash = args.dataset.startswith('doordash_')
    if args.training_probe and not doordash:
        parser.error('Training-profile probes currently require a DoorDash recipe')
    if doordash and (args.graph_recipe is None or args.prepared_manifest is None
                     or not args.frozen_plan.is_file()):
        parser.error('DoorDash needs a graph recipe, prepared manifest, and existing frozen plan')
    args.output.mkdir(parents=True, exist_ok=False)
    args.scratch.mkdir(parents=True, exist_ok=False)
    measurements = Measurements(args.output)
    recipe = dict(RECIPES[args.dataset])
    report = dict(status='RUNNING', dataset=args.dataset, seed=args.seed,
                  version=args.version, recipe=recipe, release_approved=False,
                  quality_limit=None, performance_limit=None,
                  performance_profile_qualified=False,
                  training_recipe='Adoption notebook: one experiment, no refit, three epochs')
    report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if doordash:
        report['training_recipe'] = 'Fixed benchmark profile: 20 warmup and 200 timed batches, one experiment, no refit'
    report['quality_metric'] = ('AUROC' if args.dataset == 'affirm'
                                else 'Recall@' + str(recipe.get('top_k', 10)))
    report['prediction_scope'] = 'All eligible entities at the frozen prediction time'
    report['reload_scope'] = 'The complete reserved quality population'
    started = time.monotonic()
    device_evidence = {}
    try:
        measurements.start_monitor(args.resource_helper, args.scratch)
        with measurements.stage('import', gpu=False):
            import numpy as np
            import pandas as pd
            import pyarrow.parquet as pq
            import torch
            from kumoapi.json_serde import from_json, to_json_dict
            from kumoapi.model_plan import ModelPlan, RunMode

            import kumo.engine as kumo
        if (importlib.metadata.version('kumo-engine') != args.version
                or not Path(kumo.__file__).resolve().is_relative_to(Path(sys.prefix))
                or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise ValueError('Use the requested isolated archive with one CUDA GPU')
        torch.set_num_threads(4)
        if doordash:
            torch.set_float32_matmul_precision('medium')
        report['runtime'] = dict(python=sys.version, torch=torch.__version__,
                                 cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name(0),
                                 import_path=kumo.__file__)
        report['runtime']['resource_limits'] = {
            name: Path('/sys/fs/cgroup', name).read_text().strip()
            for name in ('memory.max', 'cpu.max')
            if Path('/sys/fs/cgroup', name).exists()}
        if doordash:
            prepared = json.loads(args.prepared_manifest.read_text())
            audit = json.loads((args.data / 'copy-verification.json').read_text())
            digest = hashlib.sha256(args.prepared_manifest.read_bytes()).hexdigest()
            if audit.get('status') != 'PASS' or audit.get('manifest_sha256') != digest:
                raise ValueError('The prepared data copy lacks matching file fingerprints')
            report['inputs'] = {}
            for name, spec in prepared['tables'].items():
                for file in spec['files']:
                    path = args.data / name / file['name']
                    actual = path.stat()
                    record = audit['files'][name + '/' + file['name']]
                    if (actual.st_size != file['bytes'] or record['sha256'] != file['sha256']
                            or actual.st_mtime_ns != record['mtime_ns']):
                        raise ValueError('Prepared input changed after its fingerprint check')
                    report['inputs'][name + '/' + file['name']] = file['sha256']
            payload = json.loads(args.graph_recipe.read_text())
            if payload['train_pquery']['query_string'] != recipe['query']:
                raise ValueError('DoorDash query differs from the reused recipe')
            report['graph_recipe_sha256'] = hashlib.sha256(args.graph_recipe.read_bytes()).hexdigest()
        else:
            payload = None
            report['inputs'] = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(args.data.glob('*.parquet'))}
        kumo.configure(scratch_dir=args.scratch)
        class_ids = None
        if args.dataset != 'affirm':
            source, column = (('businesses', 'BUSINESS_ID') if doordash
                              else ('item_profile.parquet', 'item_id'))
            class_ids = pq.read_table(args.data / source, columns=[column])[column].to_pylist()
        with measurements.stage('graph_definition'):
            graph = build_graph(kumo, args.dataset, args.data, payload)
            query = kumo.PredictiveQuery(graph=graph, query=recipe['query'])
            query.validate()
        with measurements.stage('graph_snapshot'):
            snapshot = graph.snapshot()
        with measurements.stage('training_table'):
            if doordash:
                from kumoapi.model_plan import TrainingTableGenerationPlan

                ranges = target_time_ranges(prepared['tables']['orders']['time_range']['max'])
                generation = TrainingTableGenerationPlan(
                    split=f'TimeRangeSplit({ranges!r}, timeframe=7, flexible_boundaries=False)',
                    timeframe_step=7)
                write_json(args.output / 'generation-plan.json', to_json_dict(generation))
                table = query.generate_training_table(snapshot, plan=generation)
            else:
                table = query.generate_training_table(snapshot)
        report['splits'] = {}
        for stage in ('train', 'val', 'test'):
            frame = pq.read_table(Path(table.staging_dir) / (stage + '_seeds'))
            from wheel_temporal_tasks import canonical_stage

            canonical = canonical_stage('temporal_link', json.loads(
                json.dumps(frame.to_pylist(), default=str)))
            pq.write_table(frame, args.output / (stage + '-seeds.parquet'))
            values = frame['TARGET1'].to_pylist()
            summary = dict(rows=len(values), columns=frame.column_names)
            summary['sha256'] = hashlib.sha256(
                json.dumps(canonical, sort_keys=True).encode()).hexdigest()
            if args.dataset == 'affirm':
                summary.update(binary_split_counts(values))
            elif not values or not any(values):
                raise ValueError('Recommendation split contains no answers')
            if 'TIMESTAMP' in frame.column_names:
                times = frame['TIMESTAMP'].to_pylist()
                summary.update(first_anchor=str(min(times)), last_anchor=str(max(times)))
            report['splits'][stage] = summary
        if doordash:
            previous_end = None
            for stage in ('train', 'val', 'test'):
                bounds = report['splits'][stage]
                first = datetime.fromisoformat(bounds['first_anchor'])
                end = datetime.fromisoformat(bounds['last_anchor']) + timedelta(days=7)
                if previous_end is not None and first < previous_end:
                    raise ValueError('Temporal target windows overlap between stages')
                previous_end = end
            if previous_end > datetime.fromisoformat(prepared['tables']['orders']['time_range']['max']):
                raise ValueError('Evaluation target windows exceed observed order history')
            if report['splits']['test']['first_anchor'] != report['splits']['test']['last_anchor']:
                raise ValueError('This profile requires one fixed test prediction time')
            recipe['prediction_time'] = report['splits']['test']['first_anchor']
            report['chronology'] = 'PASS'
        with measurements.stage('model_plan'):
            if args.frozen_plan.exists():
                plan = from_json(args.frozen_plan.read_text(), ModelPlan)
            else:
                plan = query.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
                plan.training_job.num_experiments = 1
                plan.training_job.refit_trainval = False
                plan.training_job.refit_full = False
                plan.training_job.disable_explain = True
                plan.optimization.max_epochs = 3
                args.frozen_plan.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.frozen_plan, to_json_dict(plan))
            plan.training_job.manual_seed = args.seed
            write_json(args.output / 'requested-plan.json', to_json_dict(plan))
        report['frozen_plan_sha256'] = hashlib.sha256(args.frozen_plan.read_bytes()).hexdigest()
        with measurements.stage('prediction_table'):
            prediction_table = query.generate_prediction_table(
                snapshot, prediction_time=datetime.fromisoformat(recipe['prediction_time']))
        population = pq.read_table(prediction_table.table_path, columns=['ENTITY1'])
        pq.write_table(population, args.output / 'prediction-inputs.parquet')
        expected_ids = population['ENTITY1'].to_pylist()
        if args.preflight_only:
            measurements.finish_monitor()
            measurements.monitor = None
            report.update(status='PREFLIGHT', eligible_prediction_rows=prediction_table.row_count)
            return 0
        device_evidence = {}
        loader_profile = nullcontext()
        if doordash:
            loader_profile = fixed_loaders(
                measurements, report, batch_size=plan.optimization.batch_size[0],
                workers=args.loader_workers)
        with observe_gpu(device_evidence), loader_profile:
            import pytorch_lightning as pl
            from kumo_core.train.trainer import Trainer as CoreTrainer

            work = report['training_work'] = []

            class WorkCounter(pl.Callback):
                def __init__(self):
                    self.window = TrainingWindow(torch.cuda.synchronize, timing=doordash)

                def on_train_batch_start(self, trainer, module, batch, batch_idx):
                    if doordash:
                        self.window.batch_start()

                def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                    self.window.batch_end(optimizer_steps=int(trainer.global_step))
                    if doordash:
                        report['training_window'] = self.window.summary()

                def on_train_end(self, trainer, module):
                    stop = trainer.early_stopping_callback
                    work.append(dict(completed_batches=self.window.batches,
                                     optimizer_steps=int(trainer.global_step),
                                     current_epoch=int(trainer.current_epoch),
                                     early_stopped=bool(stop and stop.stopped_epoch)))

            original_create = CoreTrainer.create_pl_trainer

            def create(core_trainer):
                created = original_create(core_trainer)
                core_trainer.pl_trainer.callbacks.append(WorkCounter())
                return created

            trainer = kumo.Trainer(model_plan=plan)
            with (patch.object(CoreTrainer, 'create_pl_trainer', create),
                  measurements.stage('training')):
                result = trainer.fit(table, snapshot=snapshot)
            report['new_training_fits'] = 1
            if len(work) != 1 or work[0].get('completed_batches', work[0].get('completed_steps', 0)) <= 0:
                raise ValueError('Missing single-trial completed training work')
            if doordash:
                optimization = plan.optimization
                if optimization.min_steps_per_epoch != optimization.max_steps_per_epoch:
                    raise ValueError('The fixed profile needs equal minimum and maximum batches')
                expected = optimization.max_epochs * optimization.max_steps_per_epoch
                verify_training_work(work, report.get('training_window', {}),
                                     expected_batches=expected)
                report['required_training_batches'] = expected
            shutil.copytree(result.bundle_dir, args.output / 'bundle')
            for name in ('resolved_model_plan', 'trainer_config', 'encoders',
                         'target_encoder'):
                shutil.copyfile(result.artifact_paths[name], args.output / (name + '.json'))
            if doordash:
                [metadata] = trainer.scratch_dir.glob('runs/*-fit-*/task_dataset/metadata.json')
                runtime = json.loads(metadata.with_name('runtime_manifest.json').read_text())
                schema = json.loads(metadata.with_name('feature_schema.json').read_text())
                config = json.loads((args.output / 'trainer_config.json').read_text())
                if (runtime.get('task', {}).get('task_type') != 'temporal_link_prediction'
                        or runtime.get('task', {}).get('is_temporal') is not True
                        or runtime.get('table', {}).get('columns', {}).get('timestamp') != 'TIMESTAMP'
                        or schema.get('node_types', {}).get('orders', {}).get('time_column') != 'ORDER_TIME'
                        or config.get('sampler_kwargs', {}).get('temporal_strategy') != 'last'):
                    raise ValueError('Missing native event or prediction-time cutoff evidence')
                for name in ('runtime_manifest.json', 'feature_schema.json'):
                    shutil.copyfile(metadata.with_name(name), args.output / name)
                report['temporal_sampling'] = 'PASS: native seed timestamps and event-time cutoffs'
            if result.holdout_path is not None:
                if result.holdout_truncated is not False:
                    raise ValueError('Held-out predictions are truncated or incomplete')
                result.holdout_df().to_parquet(args.output / 'holdout.parquet', index=False)
            for config in trainer.scratch_dir.glob('**/automl/version_*/config.json'):
                shutil.copyfile(config, args.output / 'selected-trainer-config.json')
            for selected in trainer.scratch_dir.glob('**/automl/version_*/model_plan.json'):
                shutil.copyfile(selected, args.output / 'selected-model-plan.json')
            selected_path = args.output / 'selected-model-plan.json'
            verify_plan(to_json_dict(plan), json.loads(selected_path.read_text()))
            report['selected_plan'] = 'PASS: fixed requested fields match the actual selected plan'
            report['metrics'] = {key: float(value) for key, value in result.metrics.items()
                                 if isinstance(value, (int, float)) and math.isfinite(value)}
            metric = 'val_recall@10' if args.training_probe else recipe['metric']
            if metric not in report['metrics']:
                raise ValueError('Missing held-out quality metric')
            report['quality_score'] = report['metrics'][metric]
            write_json(args.output / 'progress.json', dict(report, **device_evidence, stage='training_complete'))
            options = ({'binary_classification_threshold': 0.5} if args.dataset == 'affirm'
                       else {'num_classes_to_return': recipe.get('top_k', 10)})
            evaluated_stage = 'val' if args.training_probe else 'test'
            reserved = pq.read_table(args.output / (evaluated_stage + '-seeds.parquet'))
            pq.write_table(reserved.select(['ENTITY1', 'TIMESTAMP']),
                           args.output / 'evaluation-inputs.parquet')
            evaluation_connector = kumo.FileConnector(root_dir=str(args.output))
            evaluation_table = kumo.PredictionTable.from_source_table(
                evaluation_connector['evaluation-inputs.parquet'], pq=query,
                snapshot=snapshot, plan=trainer.training_table_generation_plan)
            with measurements.stage('validation_prediction' if args.training_probe else 'heldout_prediction'):
                evaluated = trainer.predict(evaluation_table, snapshot=snapshot, **options)
            verify_population(evaluated.frame, reserved['ENTITY1'].to_pylist(),
                              top_k=None if args.dataset == 'affirm' else recipe.get('top_k', 10),
                              class_ids=class_ids)
            evaluated.frame.to_parquet(args.output / 'evaluation-predictions.parquet', index=False)
            report['quality_score'] = score_customer(args.dataset, evaluated.frame, reserved.to_pandas())
            write_json(args.output / 'progress.json', dict(report, **device_evidence, stage='heldout_scored'))
            if args.training_probe:
                verify_devices(device_evidence, ('training', 'prediction'))
                measurements.finish_monitor()
                measurements.monitor = None
                report.update(status='TRAINED_PROFILE_PROBE', quality_metric='validation_Recall@12',
                              prediction_scope='Complete validation cohort only',
                              evaluation_scope='Validation selects the profile. SDK auxiliary test metrics are not reserved validation. No independent test or operational prediction.')
                return 0
            del evaluated, reserved
            gc.collect()
            with measurements.stage('prediction'):
                prediction = trainer.predict(prediction_table, snapshot=snapshot, **options)
            frame = prediction.frame
            verify_population(frame, expected_ids,
                              top_k=None if args.dataset == 'affirm' else recipe.get('top_k', 10),
                              class_ids=class_ids)
            frame.to_parquet(args.output / 'predictions.parquet', index=False)
            del frame
            gc.collect()
            loaded = kumo.Trainer.load(result.bundle_dir)
            with measurements.stage('reload_prediction'):
                reloaded = loaded.predict(evaluation_table, snapshot=snapshot, **options)
            after = reloaded.frame
            before = pd.read_parquet(args.output / 'evaluation-predictions.parquet')
            verify_population(after, before['ENTITY'].unique(),
                              top_k=None if args.dataset == 'affirm' else recipe.get('top_k', 10),
                              class_ids=class_ids)
            after.to_parquet(args.output / 'reloaded-predictions.parquet', index=False)
            frame = pd.read_parquet(args.output / 'predictions.parquet')
            keys = ['ENTITY'] if args.dataset == 'affirm' else ['ENTITY', 'CLASS']
            score = ['False_PROB', 'True_PROB'] if args.dataset == 'affirm' else ['SCORE']
            repeated = frame[frame['ENTITY'].isin(before['ENTITY'])]
            left = before.sort_values(keys)
            for name, compared in (('reload', after), ('repeated_prediction', repeated)):
                if args.dataset != 'affirm':
                    report[name] = measure_recommendations(before, compared)
                    continue
                right = compared.sort_values(keys)
                if (left.duplicated(keys).any() or not left[keys].reset_index(drop=True).equals(
                        right[keys].reset_index(drop=True))):
                    raise ValueError(f'Prediction identities changed in {name}')
                report[name] = measure_consistency(left[score].to_numpy(), right[score].to_numpy())
            if not np.isfinite(frame[score].to_numpy()).all():
                raise ValueError('Nonfinite predictions')
            if args.dataset != 'affirm' and frame.groupby('ENTITY').size().max() > recipe.get('top_k', 10):
                raise ValueError('Top-ten query returned more than ten recommendations')
            report['prediction_rows'] = len(frame)
            report['prediction_entities'] = int(frame['ENTITY'].nunique())
            report['bundle_dir'] = str(args.output / 'bundle')
        verify_devices(device_evidence, ('training', 'prediction', 'reload_and_predict'))
        report.update(device_evidence)
        report['status'] = 'MEASURED'
        if args.dataset == 'sothebys' and report['quality_score'] == 0:
            report['quality_warning'] = 'Zero held-out Recall@10 cannot qualify learning-loss detection'
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    finally:
        report.update(device_evidence)
        try:
            measurements.finish_monitor()
        except Exception as error:
            report.update(status='FAIL', memory_error=f'{type(error).__name__}: {error}')
            traceback.print_exc()
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(args.output / 'result.json', report)
    return 0 if report['status'] == 'MEASURED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
