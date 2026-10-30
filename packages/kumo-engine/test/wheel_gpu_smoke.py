#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise eight task types against one installed candidate wheel on CUDA."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import random
import shutil
import sys
import tempfile
import time
import traceback
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

DELIVERY_FILTER = ' WHERE COUNT(orders.*, -90, 0, days) > 0'
TASKS = {
    'binary':
    dict(
        graph='adtech', task_type='binary_classification', days=7,
        metric='auroc', loss='binary_cross_entropy',
        query=('PREDICT COUNT(conversions.*, 0, 7, days) > 0 '
               'FOR EACH users.user_id '
               'WHERE COUNT(impressions.*, -14, 0, days) > 0')),
    'multiclass':
    dict(
        graph='delivery', task_type='multiclass_classification', days=14,
        metric='acc', loss='cross_entropy',
        query=('PREDICT FIRST(orders.daypart, 0, 14, days) '
               'FOR EACH users.user_id' + DELIVERY_FILTER)),
    'multilabel':
    dict(
        graph='delivery', task_type='multilabel_classification', days=14,
        metric='ap_macro', loss='binary_cross_entropy',
        query=('PREDICT LIST_DISTINCT(orders.daypart, 0, 14, days) '
               'CLASSIFY TOP 6 FOR EACH users.user_id' + DELIVERY_FILTER)),
    'multilabel_ranking':
    dict(
        graph='delivery', task_type='multilabel_ranking', days=14,
        metric='recall@5', loss='cross_entropy',
        query=('PREDICT LIST_DISTINCT(orders.daypart, 0, 14, days) '
               'RANK TOP 5 FOR EACH users.user_id' + DELIVERY_FILTER)),
    'regression':
    dict(
        graph='delivery', task_type='regression', days=30, metric='mae',
        loss='huber', query=('PREDICT COUNT(orders.*, 0, 30, days) '
                             'FOR EACH users.user_id' + DELIVERY_FILTER)),
    'forecasting':
    dict(
        graph='delivery', task_type='forecasting', days=84, metric='mae',
        loss='mae', query=('PREDICT COUNT(orders.*, 0, 7, days) '
                           'FORECAST 12 TIMEFRAMES FOR EACH users.user_id')),
    'static_link':
    dict(
        graph='commerce', task_type='static_link_prediction', days=0,
        metric='recall@5', loss='cross_entropy',
        query=('PREDICT LIST_DISTINCT(purchases.merchant_id) '
               'RANK TOP 5 FOR EACH customers.customer_id')),
    'temporal_link':
    dict(
        graph='commerce', task_type='temporal_link_prediction', days=7,
        metric='recall@10', loss='cross_entropy',
        query=('PREDICT LIST_DISTINCT(purchases.merchant_id, 0, 7, days) '
               'RANK TOP 10 FOR EACH customers.customer_id')),
}


def verify_installed_files(wheel, distribution):
    """Match installed files to the finished archive, except pip's rewritten RECORD."""
    from zipfile import ZipFile

    location = Path(distribution.locate_file('')).resolve()
    checked = 0
    with ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith('/') or name.endswith('.dist-info/RECORD'):
                continue
            destination = Path(distribution.locate_file(name)).resolve()
            if (not destination.is_relative_to(location)
                    or not destination.is_file()
                    or hashlib.sha256(destination.read_bytes()).digest() !=
                    hashlib.sha256(archive.read(name)).digest()):
                raise ValueError(f'Installed wheel member differs: {name}')
            checked += 1
    if not checked:
        raise ValueError('Wheel contains no installed files')
    return checked


def prediction_values(task, frame, ids, *, classes=(), anchor=None,
                      time_column='TIMESTAMP'):
    """Require complete, unique output coordinates before aligning values."""
    import numpy as np
    import pandas as pd

    if (not ids or len(set(ids)) != len(ids) or frame.empty
            or frame['ENTITY'].isna().any()
            or set(frame['ENTITY']) != set(ids)):
        raise ValueError('Prediction entities are missing, repeated or foreign')
    if task != 'static_link':
        stamp = pd.Timestamp(anchor)
        if (pd.isna(stamp) or 'ANCHOR_TIMESTAMP' not in frame or
                not pd.to_datetime(frame['ANCHOR_TIMESTAMP']).eq(stamp).all()):
            raise ValueError('Prediction anchors differ from the inputs')
    if task in ('binary', 'regression'):
        if not frame['ENTITY'].is_unique:
            raise ValueError('Scalar prediction entities are duplicated')
        ordered = frame.set_index('ENTITY').loc[ids]
        if task == 'binary':
            values = ordered[['False_PROB', 'True_PROB']].to_numpy(dtype=float)
            if (not ((values >= 0) & (values <= 1)).all() or
                    not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=1e-7)
                    or not np.array_equal(ordered['TARGET_PRED'],
                                          values[:, 1] > .5)):
                raise ValueError(
                    'Binary probabilities or chosen labels are invalid')
        else:
            values = ordered[['TARGET_PRED']].to_numpy(dtype=float)
    else:
        key, column = ('CLASS', 'SCORE')
        coordinates = list(classes)
        if task == 'forecasting':
            frame = frame.copy()
            key, column = time_column, 'TARGET_PRED'
            frame[key] = pd.to_datetime(frame[key])
            coordinates = [
                pd.Timestamp(anchor) + timedelta(days=7 * i) for i in range(12)
            ]
        if (not coordinates or len(set(coordinates)) != len(coordinates)
                or frame[key].isna().any()
                or frame.duplicated(['ENTITY', key]).any()
                or set(zip(frame['ENTITY'], frame[key])) != {
                    (entity, coordinate)
                    for entity in ids for coordinate in coordinates
                }):
            raise ValueError(
                'Prediction output coordinates are incomplete or repeated')
        values = frame.pivot(
            index='ENTITY', columns=key,
            values=column).loc[ids, coordinates].to_numpy(dtype=float)
        if task in ('multiclass', 'multilabel', 'multilabel_ranking'):
            if not ((values >= 0) & (values <= 1)).all():
                raise ValueError('Class probabilities are invalid')
            if task != 'multilabel' and not np.allclose(values.sum(axis=1), 1,
                                                        atol=1e-6, rtol=1e-6):
                raise ValueError('Class probabilities do not sum to one')
            if task == 'multiclass':
                chosen = frame.loc[frame['PREDICTED'], ['ENTITY', 'SCORE']]
                if (not chosen['ENTITY'].is_unique
                        or set(chosen['ENTITY']) != set(ids)
                        or not np.array_equal(
                            chosen['SCORE'],
                            frame.groupby('ENTITY')['SCORE'].max().loc[
                                chosen['ENTITY']])):
                    raise ValueError(
                        'Multiclass output does not choose its highest score')
            elif task == 'multilabel' and not np.array_equal(
                    frame['PREDICTED'], frame['SCORE'] > .5):
                raise ValueError('Multilabel output differs from its threshold')
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Prediction values must be nonempty and finite')
    return values


def prediction_decisions(task, frame, ids, *, classes=()):
    """Retain labels and published ranks independently of numeric tolerance."""
    if task in ('regression', 'forecasting'):
        return None
    if task == 'binary':
        return frame.set_index('ENTITY').loc[ids, 'TARGET_PRED'].to_numpy()
    column = 'PREDICTED'
    if task not in ('multiclass', 'multilabel'):
        column = 'RANK'
        if column not in frame:
            frame = frame.assign(
                RANK=frame.groupby('ENTITY', sort=False).cumcount())
    return frame.pivot(index='ENTITY', columns='CLASS',
                       values=column).loc[ids, list(classes)].to_numpy()


def compare_predictions(before, after, *, atol, rtol, qualify=False,
                        before_decisions=None, after_decisions=None,
                        report_numeric_drift=False):
    """Measure numeric drift; qualification never supplies a normal CI pass."""
    import numpy as np

    left, right = np.asarray(before,
                             dtype=float), np.asarray(after, dtype=float)
    if (left.ndim != 2 or not left.size or left.shape != right.shape
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or not all(math.isfinite(v) and v >= 0 for v in (atol, rtol))):
        raise ValueError(
            'Reload comparison requires equal finite prediction shapes')
    difference = np.abs(left - right)
    result = dict(
        decisions_equal=bool(np.array_equal(before_decisions, after_decisions)),
        max_abs_difference=float(difference.max()),
        max_relative_difference=float(
            (difference / np.maximum(np.abs(left), 1e-12)).max()),
        within_tolerance=bool(np.allclose(left, right, atol=atol, rtol=rtol)))
    if not qualify and not result['decisions_equal']:
        raise ValueError('Prediction decisions changed after repeat or reload')
    if not qualify and not report_numeric_drift and not result[
            'within_tolerance']:
        raise ValueError(
            f'Prediction difference exceeds atol={atol}, rtol={rtol}: {result}')
    return result


def make_plan(task):
    """Keep the existing normal seed-zero recipe, without quality controls."""
    from kumoapi.encoder import Index, MultiCategorical, Numerical
    from kumoapi.model_plan import (
        GNNModelPlan,
        MissingType,
        ModelPlan,
        NumNeighborsConfig,
    )

    spec = TASKS[task]
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.pruning = None
    plan.training_job.metrics = [spec['metric']]
    plan.training_job.tune_metric = spec['metric']
    plan.training_job.disable_compilation = True
    plan.training_job.disable_explain = True
    plan.training_job.manual_seed = 0
    plan.training_job.refit_trainval = False
    plan.training_job.refit_full = False
    plan.optimization.max_epochs = 10
    plan.optimization.min_steps_per_epoch = 20
    plan.optimization.max_steps_per_epoch = 20
    plan.optimization.batch_size = [64]
    plan.optimization.loss = [spec['loss']]
    plan.optimization.optimizer = ['Adam']
    plan.optimization.base_lr = [.01]
    plan.optimization.weight_decay = [0.0]
    plan.optimization.majority_sampling_ratio = ([None] if task == 'binary' else
                                                 MissingType.VALUE)
    plan.optimization.weight_mode = ([None]
                                     if task == 'binary' else MissingType.VALUE)
    plan.optimization.max_val_steps = 2000
    plan.optimization.max_test_steps = 4000
    plan.optimization.early_stopping = [{'min_delta': 0.0, 'patience': 3}]
    plan.optimization.lr_scheduler = [
        dict(name='constant_with_warmup', interval='step',
             kwargs={'warmup_ratio_or_steps': .1})
    ]
    plan.model_architecture.model = [
        GNNModelPlan(channels=[16], aggregation=[['sum', 'mean', 'max']],
                     dropout=[0.0])
    ]
    plan.model_architecture.num_pre_message_passing_layers = [1]
    plan.model_architecture.num_post_message_passing_layers = [2]
    plan.model_architecture.activation = ['gelu']
    plan.model_architecture.normalization = ['layer_norm']
    plan.model_architecture.use_seq_id = [False]
    plan.neighbor_sampling.num_neighbors = [
        NumNeighborsConfig(hop1={'default': 16}, hop2={'default': 16})
    ]
    plan.neighbor_sampling.sample_from_entity_table = MissingType.VALUE
    plan.neighbor_sampling.adaptive_sampling = False
    if task in ('binary', 'multiclass'):
        target = Index(min_occ=1, na_strategy='separate')
    elif task in ('multilabel', 'multilabel_ranking'):
        target = MultiCategorical(min_occ=1)
    else:
        target = Numerical(scaler=None, na_strategy='mean')
    plan.column_processing.encoder_overrides = ({}
                                                if task.endswith('link') else {
                                                    'train_table.TARGET': target
                                                })
    if task == 'forecasting':
        plan.model_architecture.past_encoder = ['decomposed']
        plan.model_architecture.handle_new_entities = True
    elif task.endswith('link'):
        plan.model_architecture.module = 'shallow_rhs'
        plan.model_architecture.target_embedding_mode = ['lookup']
        plan.model_architecture.output_embedding_dim = [16]
        plan.model_architecture.handle_new_target_entities = False
    return plan


def target_classes(task, targets):
    """Decode exported multicategorical tokens before collecting class names."""
    if task == 'multiclass':
        return sorted(set(targets))
    members = (filter(None, value.split('|'))
               if isinstance(value, str) else value for value in targets)
    return sorted({category for value in members for category in value})


def build_graph(spec, directory, *, static=False, sources=None):
    from kumoapi.graph import Edge
    from kumoapi.table import Column

    import kumo.engine as kumo

    files = kumo.FileConnector(root_dir=str(directory))
    tables = {}
    for name, definition in spec['tables'].items():
        columns = [Column(**column) for column in definition['cols']]
        table = kumo.Table.from_source_table(
            files[(sources
                   or {}).get(name, name)], [column.name for column in columns],
            primary_key=definition.get('pkey'),
            time_column=None if static else definition.get('time_col'),
            end_time_column=None if static else definition.get('end_time_col'),
            dtypes={column.name: column.dtype
                    for column in columns})
        tables[name] = replace(
            table, definition=replace(
                table.definition, cols=columns,
                time_col=None if static else definition.get('time_col'),
                end_time_col=None
                if static else definition.get('end_time_col')))
    graph = kumo.Graph(tables, [Edge(**edge) for edge in spec['edges']])
    graph.validate()
    return graph, graph.snapshot()


def prepare_task(task, root, metadata, directory, graphs):
    import pyarrow.parquet as pq
    from kumoapi.model_plan import TrainingTableGenerationPlan

    import kumo.engine as kumo

    spec = TASKS[task]
    source = root / spec['graph'] / 'pr'
    graph_key = (spec['graph'], task == 'static_link')
    if graph_key not in graphs:
        graphs[graph_key] = build_graph(metadata[spec['graph']], source,
                                        static=task == 'static_link')
    graph, snapshot = graphs[graph_key]
    query = kumo.PredictiveQuery(graph, spec['query'])
    query.validate()
    if query.get_task_type().value != spec['task_type']:
        raise ValueError(f'{task} resolved to a different task type')
    if task == 'static_link':
        # Native random link splitting replaces the former quality/control payload.
        generation = TrainingTableGenerationPlan(
            split='RandomLinkSplit([0.6, 0.2, 0.2])')
    else:
        manifest = json.loads((source / 'manifest.json').read_text())
        start, end = [
            datetime.fromisoformat(manifest['time_window'][key].replace(
                'Z', '+00:00')).astimezone(timezone.utc).replace(tzinfo=None)
            for key in ('start', 'end')
        ]
        horizon = timedelta(days=spec['days'])
        if end - 4 * horizon < start:
            raise ValueError(
                'Dataset cannot hold three complete temporal stages')
        ranges = [
            ((start if offset == 3 else end - offset * horizon).isoformat(),
             (end - (offset - 1) * horizon).isoformat()) for offset in (3, 2, 1)
        ]
        split = (f'TimeRangeSplit({ranges!r}, timeframe={spec["days"]}, '
                 'flexible_boundaries=False)')
        generation = TrainingTableGenerationPlan(
            split=split,
            timeframe_step=7 if task == 'forecasting' else spec['days'],
            **(dict(lag_timesteps=12, year_over_year=False)
               if task == 'forecasting' else {}))
    table = query.generate_training_table(snapshot, plan=generation)
    directory.mkdir(parents=True)
    table.export(
        kumo.TrainingTableOutputConfig(
            kumo.FileConnector(root_dir=str(directory)), 'rows'))
    rows = pq.read_table(directory / 'rows').to_pylist()
    if {row['SPLIT'] for row in rows} != {0, 1, 2}:
        raise ValueError('Training, validation or test stage is empty')
    test = [row for row in rows if row['SPLIT'] == 2]
    ids = [row['ENTITY1'] for row in test]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Test-stage entities are empty or duplicated')
    anchor = None
    if task != 'static_link':
        anchors = {row['TIMESTAMP'] for row in test}
        if len(anchors) != 1:
            raise ValueError('Prediction inputs require one test-stage anchor')
        anchor = anchors.pop()
    classes = []
    if task.endswith('link'):
        classes = sorted(
            pq.read_table(source / 'merchants',
                          columns=['merchant_id'])['merchant_id'].to_pylist())
    elif task in ('multiclass', 'multilabel', 'multilabel_ranking'):
        classes = target_classes(task, [row['TARGET1'] for row in rows])
    columns = ['ENTITY1'] if task == 'static_link' else ['ENTITY1', 'TIMESTAMP']
    if task == 'forecasting':
        columns.append('autoregression_labels')
        if any(len(row['autoregression_labels']) != 12 for row in test):
            raise ValueError(
                'Forecast prediction inputs need twelve history values')
    inputs = directory / 'inputs.parquet'
    import pyarrow as pa
    pq.write_table(
        pa.Table.from_pylist([{key: row[key]
                               for key in columns} for row in test]), inputs)
    return dict(query=query, snapshot=snapshot, training_table=table,
                input_file=inputs, ids=ids, classes=classes, anchor=anchor)


def verify_checkpoint(model, path, expected_digest, *, exclude=()):
    """Compare the weights that Lightning actually restored, with no tolerance."""
    import numpy as np
    import pandas as pd
    import torch
    from kumo_core.io.filesystem import local_path
    from kumo_core.train.checkpoint import load_kumo_checkpoint

    if path is None:
        raise ValueError('Reload did not select a checkpoint')
    path = local_path(str(path))
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError('Reload selected a different checkpoint')
    expected = load_kumo_checkpoint(str(path), map_location='cpu')['state_dict']
    actual = model.state_dict()
    tensors = 0

    def equal(left, right):
        nonlocal tensors
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if (left.dtype != right.dtype or left.shape != right.shape
                    or not torch.equal(left.detach().cpu(),
                                       right.detach().cpu())):
                raise ValueError(
                    'Restored model tensor differs from its checkpoint')
            tensors += 1
        elif type(left) is not type(right):
            raise ValueError('Restored model state types differ')
        elif isinstance(left, dict):
            if set(left) != set(right):
                raise ValueError('Restored model state keys differ')
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            if len(left) != len(right):
                raise ValueError('Restored model state lengths differ')
            for old, new in zip(left, right):
                equal(old, new)
        elif isinstance(left, pd.DataFrame):
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif left != right:
            raise ValueError('Restored model state values differ')

    if set(expected) != set(actual) or not set(exclude) <= set(expected):
        raise ValueError('Restored model state keys differ')
    equal({key: value
           for key, value in expected.items() if key not in exclude},
          {key: value
           for key, value in actual.items() if key not in exclude})
    if not tensors:
        raise ValueError('Reload checked no model tensors')
    return tensors


@contextmanager
def observe_cuda(state):
    from numbers import Integral

    import torch
    from kumo_core.train.trainer import Trainer as CoreTrainer
    from pytorch_lightning import Callback
    from pytorch_lightning.callbacks import EarlyStopping

    original_create = CoreTrainer.create_pl_trainer

    class Proof(Callback):
        def on_fit_start(self, trainer, module):
            state['fits'] += 1
            if 'fit_ledger' in state:
                state['fit_ledger'].append(
                    dict(status='RUNNING', steps=0, cuda_batches=0))

        def check(self, trainer, values):
            if (str(trainer.strategy.root_device) != 'cuda:0' or not values
                    or any(not isinstance(value, torch.Tensor) or
                           str(value.device) != 'cuda:0' or not value.numel()
                           for value in values)):
                raise ValueError('Completed work did not use CUDA tensors')
            torch.cuda.synchronize()

        def on_train_batch_end(self, trainer, module, values, batch, batch_idx):
            self.check(
                trainer,
                [values.get('loss') if isinstance(values, dict) else values])
            state['training_batches'] += 1
            if 'fit_ledger' in state:
                state['fit_ledger'][-1]['cuda_batches'] += 1
            if 'training_epochs' in state:
                seed = (getattr(module, 'lhs_name', None)
                        or getattr(module, 'entity_table_name', None)
                        or getattr(batch, 'input_type', None))
                size = batch[seed].batch_size
                if isinstance(
                        size,
                        bool) or not isinstance(size, Integral) or size < 1:
                    raise ValueError(
                        'Training batch has no valid seed-row count')
                state['training_epochs'][-1]['batch_sizes'].append(int(size))
                state['training_epochs'][-1]['batches'] += 1

        def on_train_epoch_start(self, trainer, module):
            if 'training_epochs' in state:
                state['training_epochs'].append(
                    dict(epoch=int(trainer.current_epoch), batches=0,
                         batch_sizes=[]))

        def on_fit_end(self, trainer, module):
            state['completed_fits'] += 1
            state['optimizer_steps'] += int(trainer.global_step)
            if 'fit_ledger' in state:
                state['fit_ledger'][-1].update(status='PASS',
                                               steps=int(trainer.global_step))
            if 'training_epochs' in state:
                [stop] = [
                    callback for callback in trainer.callbacks
                    if isinstance(callback, EarlyStopping)
                ]
                state['early_stopping'] = dict(
                    stopped_epoch=int(stop.stopped_epoch),
                    wait_count=int(stop.wait_count),
                    patience=int(stop.patience))

        def on_predict_batch_end(self, trainer, module, values, batch,
                                 batch_idx, dataloader_idx=0):
            values = values.get('logits', values.get('score')) if isinstance(
                values, dict) else values
            self.check(
                trainer,
                list(values.values()) if isinstance(values, dict) else [values])
            state['prediction_batches'][state['stage']] += 1

    with ExitStack() as stack:

        def create(trainer):
            result = original_create(trainer)
            pl_trainer = trainer.pl_trainer
            if str(pl_trainer.strategy.root_device) != 'cuda:0':
                raise ValueError('Trainer fell back from CUDA')
            pl_trainer.callbacks.append(Proof())
            if state['stage'] == 'reload':
                strategy = pl_trainer.strategy
                restore = strategy.load_model_state_dict

                def load(*args, **kwargs):
                    result = restore(*args, **kwargs)
                    state['restored_tensors'] = verify_checkpoint(
                        strategy.lightning_module, pl_trainer.ckpt_path,
                        state['checkpoint_sha256'])
                    state['reload_checks'] += 1
                    return result

                stack.enter_context(
                    patch.object(strategy, 'load_model_state_dict', load))
            return result

        stack.enter_context(
            patch.object(CoreTrainer, 'create_pl_trainer', create))
        yield


def run_task(task, prepared, directory, args):
    import numpy as np
    import torch
    from kumo_core.io.filesystem import local_path
    from kumo_core.train.checkpoint import load_kumo_checkpoint
    from kumoapi.jobs import MetadataField

    import kumo.engine as kumo

    started = time.monotonic()
    kumo.configure(scratch_dir=directory / 'scratch')
    state = dict(
        stage='fit', fits=0, training_batches=0, completed_fits=0,
        optimizer_steps=0, prediction_batches={
            stage: 0
            for stage in ('predict', 'repeat', 'reload')
        }, reload_checks=0, restored_tensors=0)
    seed = lambda: (random.seed(0), np.random.seed(0), torch.manual_seed(0))
    seed()
    with observe_cuda(state):
        trainer = kumo.Trainer(make_plan(task), accelerator='gpu')
        fitted = trainer.fit(prepared['training_table'],
                             snapshot=prepared['snapshot'])
        checkpoint = local_path(
            str(fitted.artifact_paths['selected_checkpoint']))
        steps = load_kumo_checkpoint(str(checkpoint),
                                     map_location='cpu').get('global_step')
        if (type(steps) is not int or steps <= 0
                or not state['training_batches'] or state['fits'] != 1
                or state['completed_fits'] != 1
                or state['optimizer_steps'] < steps):
            raise ValueError(
                'Training completed no optimizer updates or CUDA batches')
        state['checkpoint_sha256'] = hashlib.sha256(
            checkpoint.read_bytes()).hexdigest()
        [runtime_file] = trainer.scratch_dir.glob(
            'runs/*-fit-*/task_dataset/runtime_manifest.json')
        runtime = json.loads(runtime_file.read_text())
        if runtime['task']['task_type'] != TASKS[task]['task_type']:
            raise ValueError('Fitted model resolved to a different task type')
        schema = json.loads(
            runtime_file.with_name('feature_schema.json').read_text())
        entity = runtime['table']['seed_entity_table']
        time_column = schema['node_types'][entity]['time_column']
        saved = directory / 'saved-model'
        shutil.copytree(fitted.bundle_dir, saved)

        def predict(model, stage, cycle=0):
            state['stage'] = stage
            previous_batches = state['prediction_batches'][stage]
            seed()
            inputs = prepared['input_file']
            table = kumo.PredictionTable.from_source_table(
                kumo.FileConnector(root_dir=str(inputs.parent))[inputs.name],
                pq=prepared['query'], snapshot=prepared['snapshot'],
                plan=model.training_table_generation_plan)
            options = {}
            if prepared['classes']:
                options['num_classes_to_return'] = len(prepared['classes'])
            if task == 'binary':
                options['binary_classification_threshold'] = .5
            if task != 'static_link':
                options['output_config'] = kumo.PredictionOutputConfig(
                    connector=kumo.FileConnector(root_dir=str(directory /
                                                              'predictions')),
                    output_paths={'predictions': f'{stage}-{cycle}'},
                    extra_fields=[MetadataField.ANCHOR_TIMESTAMP])
            frame = model.predict(table, snapshot=prepared['snapshot'],
                                  **options).frame
            if state['prediction_batches'][stage] <= previous_batches:
                raise ValueError('Prediction call completed no CUDA batches')
            values = prediction_values(task, frame, prepared['ids'],
                                       classes=prepared['classes'],
                                       anchor=prepared['anchor'],
                                       time_column=time_column or 'TIMESTAMP')
            decisions = prediction_decisions(task, frame, prepared['ids'],
                                             classes=prepared['classes'])
            return values, decisions

        before, before_decisions = predict(trainer, 'predict')
        comparisons = []
        for cycle in range(1, args.repetitions + 1):
            previous_reload_checks = state['reload_checks']
            previous_batches = state['prediction_batches'].copy()
            repeated, repeat_decisions = predict(trainer, 'repeat', cycle)
            repeat_delta = compare_predictions(
                before, repeated, atol=args.reload_atol, rtol=args.reload_rtol,
                qualify=args.qualify, before_decisions=before_decisions,
                after_decisions=repeat_decisions,
                report_numeric_drift=args.report_numeric_drift)
            del trainer
            gc.collect()
            trainer = kumo.Trainer.load(saved, accelerator='gpu')
            after, after_decisions = predict(trainer, 'reload', cycle)
            reload_delta = compare_predictions(
                before, after, atol=args.reload_atol, rtol=args.reload_rtol,
                qualify=args.qualify, before_decisions=before_decisions,
                after_decisions=after_decisions,
                report_numeric_drift=args.report_numeric_drift)
            if (state['reload_checks'] <= previous_reload_checks
                    or not state['restored_tensors']):
                raise ValueError(
                    'Prediction cycle lacks freshly restored model state')
            comparisons.append(
                dict(
                    cycle=cycle, repeat=repeat_delta, reload=reload_delta,
                    prediction_batches={
                        stage: state['prediction_batches'][stage] -
                        previous_batches[stage]
                        for stage in ('repeat', 'reload')
                    }, restored_tensors=state['restored_tensors']))
    if (state['fits'] != 1 or state['completed_fits'] != 1
            or len(comparisons) != args.repetitions
            or state['reload_checks'] < args.repetitions
            or not state['restored_tensors']
            or not all(state['prediction_batches'].values())):
        raise ValueError(
            'Reloaded state or completed CUDA prediction work is missing')
    return dict(
        task=task, task_type=TASKS[task]['task_type'], seed=0,
        status='MEASURED' if args.qualify else 'PASS', metrics={
            name: float(value) if math.isfinite(value) else None
            for name, value in fitted.metrics.items()
        }, entities=len(prepared['ids']), output_shape=list(before.shape),
        physical_fits=state['fits'], completed_fits=state['completed_fits'],
        optimizer_steps=state['optimizer_steps'],
        selected_checkpoint_steps=steps,
        training_batches=state['training_batches'],
        prediction_batches=state['prediction_batches'],
        checkpoint_sha256=state['checkpoint_sha256'],
        restored_tensors=state['restored_tensors'],
        reload_checks=state['reload_checks'], comparisons=comparisons,
        seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--graph-metadata', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-version', required=True)
    parser.add_argument(
        '--repetitions', type=int, default=1,
        help='Prediction/reload cycles per model; each task trains once')
    parser.add_argument(
        '--qualify', action='store_true',
        help='Measure numeric differences; does not produce a normal CI pass')
    parser.add_argument(
        '--report-numeric-drift', action='store_true',
        help='Report numeric drift; keep labels, ranks and execution strict')
    # The T4 qualification measured at most 1.91e-6 across 48 comparisons.
    parser.add_argument('--reload-atol', type=float, default=1e-5)
    parser.add_argument('--reload-rtol', type=float, default=0.)
    args = parser.parse_args()
    if args.repetitions < 1 or any(
            not math.isfinite(value) or value < 0
            for value in (args.reload_atol, args.reload_rtol)):
        parser.error(
            'Repetitions must be positive and tolerances finite and nonnegative'
        )
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(
        status='RUNNING', repetitions=args.repetitions, tasks=[],
        qualification=args.qualify, reload_atol=args.reload_atol,
        reload_rtol=args.reload_rtol,
        report_numeric_drift=args.report_numeric_drift,
        numeric_drift_policy=('qualification' if args.qualify else 'report_only'
                              if args.report_numeric_drift else 'enforced'))
    started = time.monotonic()
    try:
        import numpy as np
        import torch

        import kumo.engine as kumo

        version = importlib.metadata.version('kumo-engine')
        if (version != args.expected_version or kumo.__version__ != version
                or not Path(kumo.__file__).resolve().is_relative_to(
                    Path(sys.prefix)) or not torch.cuda.is_available()
                or torch.cuda.device_count() != 1):
            raise ValueError(
                'Require the selected installed wheel and exactly one CUDA GPU')
        torch.set_num_threads(4)
        report.update(version=version, import_path=kumo.__file__,
                      python=sys.version, gpu=torch.cuda.get_device_name(0),
                      torch=torch.__version__, cuda=torch.version.cuda)
        metadata = json.loads(args.graph_metadata.read_text())
        with tempfile.TemporaryDirectory(prefix='gpu-smoke-',
                                         dir=args.output_dir) as temporary:
            work = Path(temporary)
            prepared, graphs = {}, {}
            kumo.configure(scratch_dir=work / 'prepared-scratch')
            for task in TASKS:
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                prepared[task] = prepare_task(task, args.data_dir, metadata,
                                              work / 'inputs' / task, graphs)
            for task in TASKS:
                report['current_task'] = task
                print(
                    f'{task}: one fit, {args.repetitions} prediction/reload cycles',
                    flush=True)
                directory = work / 'models' / task
                directory.mkdir(parents=True)
                row = run_task(task, prepared[task], directory, args)
                report['tasks'].append(row)
                print(json.dumps(row), flush=True)
                (args.output_dir / 'summary.json').write_text(
                    json.dumps(report, indent=2, allow_nan=False) + '\n')
                shutil.rmtree(directory)
                gc.collect()
            report['status'] = 'MEASURED' if args.qualify else 'PASS'
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    finally:
        report['seconds'] = time.monotonic() - started
        (args.output_dir / 'summary.json'
         ).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0 if report['status'] in ('PASS', 'MEASURED') else 1


if __name__ == '__main__':
    sys.exit(main())
