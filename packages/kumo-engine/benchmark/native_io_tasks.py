# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproducible real training and prediction under each native I/O backend."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from native_io_isolation import (
    file_hash,
    file_tree,
    validate_clone,
    verify_baseline,
)
from native_io_resources import cgroup_path, group_snapshot

from io_utils import fsutil

# Shared correctness helpers do not install callbacks or modify trainer behavior.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'test'))
from wheel_gpu_smoke import (  # noqa: E402
    compare_predictions,
    make_plan,
    prediction_decisions,
    prediction_values,
)

TASKS = ('binary', 'regression', 'temporal_link')
QUERIES = {
    'binary': 'PREDICT COUNT(events.*, 0, 7, days) > 0 FOR EACH users.id',
    'regression': 'PREDICT COUNT(events.*, 0, 30, days) FOR EACH users.id',
    'temporal_link': ('PREDICT LIST_DISTINCT(events.item_id, 0, 7, days) '
                      'RANK TOP 10 FOR EACH users.id'),
}
HORIZONS = {'binary': 7, 'regression': 30, 'temporal_link': 7}


def json_write(path: Path, value) -> None:
    fsutil.write_text(str(path), json.dumps(value, indent=2, allow_nan=False)
                      + '\n')


def write_table(path: Path, table: pa.Table) -> None:
    with fsutil.get_binary_handle(str(path), 'wb') as stream:
        pq.write_table(table, stream, compression='snappy')


def read_table(path: str) -> pa.Table:
    with fsutil.get_binary_handle(path, 'rb') as stream:
        return pq.read_table(stream)


def feature_array(rows: int, width: int, seed: int) -> pa.Array:
    values = np.random.default_rng(seed).standard_normal(
        (rows, width), dtype=np.float32)
    return pa.ListArray.from_arrays(
        pa.array(np.arange(rows + 1, dtype=np.int32) * width),
        pa.array(values.ravel()))


def generate_data(root: Path, task: str, scale: int, width: int,
                  seed: int) -> dict:
    """Three deterministic topologies; every dense feature is a model input."""
    users, items, weeks = 128 * scale, 256 * scale, 72
    row_counts = {}
    for table, count in (('users', users), ('items', items)):
        directory = root / table
        fsutil.makedirs(str(directory), exist_ok=True)
        for part, start in enumerate(range(0, count, 8192)):
            stop = min(start + 8192, count)
            ids = np.arange(start, stop, dtype=np.int64)
            write_table(directory / f'part-{part:06}.parquet', pa.table({
                'id': ids, 'value': (ids % 17).astype(np.float32),
                'features': feature_array(len(ids), width,
                                          seed + start + (table == 'items')),
            }))
        row_counts[table] = count
    directory = root / 'events'
    fsutil.makedirs(str(directory), exist_ok=True)
    count = 0
    for part, start in enumerate(range(0, users, 64)):
        user = np.repeat(np.arange(start, min(start + 64, users)), weeks * 2)
        week = np.tile(np.repeat(np.arange(weeks), 2), min(64, users - start))
        occurrence = np.tile(np.arange(2), len(user) // 2)
        keep = (user + week) % 3 != 0
        user, week, occurrence = user[keep], week[keep], occurrence[keep]
        rng = np.random.default_rng(seed + 100000 + part)
        if task == 'binary':
            item = rng.integers(0, items, len(user))
        elif task == 'regression':
            item = rng.integers(0, items, len(user))
            hot = rng.random(len(user)) < .8
            item[hot] %= max(1, items // 100)
        else:
            # Community neighborhoods rotate with time, unlike uniform/hotspot.
            item = (user * 17 + week * 3 + occurrence) % items
        dates = (np.datetime64('2020-01-01', 'ms')
                 + (week * 7 + user % 3).astype('timedelta64[D]'))
        write_table(directory / f'part-{part:06}.parquet', pa.table({
            'id': np.arange(count, count + len(user), dtype=np.int64),
            'user_id': user, 'item_id': item, 'time': pa.array(dates),
            'value': (week + occurrence).astype(np.float32),
            'features': feature_array(len(user), width, seed + 200000 + part),
        }))
        count += len(user)
    row_counts['events'] = count
    return row_counts


def source_graph(source: Path):
    from kumoapi.graph import Edge
    from kumoapi.typing import Stype

    import kumo.engine as kumo

    connector = kumo.FileConnector(root_dir=str(source))
    tables = {
        name: kumo.Table.from_source_table(
            connector[name], primary_key='id',
            time_column='time' if name == 'events' else None).with_column(
                'features', stype=Stype.sequence)
        for name in ('users', 'items', 'events')
    }
    tables['events'] = tables['events'].with_column(
        'user_id', stype=Stype.ID).with_column('item_id', stype=Stype.ID)
    graph = kumo.Graph(tables, [Edge('events', 'user_id', 'users'),
                                Edge('events', 'item_id', 'items')])
    graph.validate()
    return graph


def prepare(args) -> None:
    from kumoapi.model_plan import TrainingTableGenerationPlan

    import kumo.engine as kumo

    directory = args.prepared_dir.resolve()
    if fsutil.exists(str(directory / 'manifest.json')):
        raise ValueError('Prepared directory already has an immutable manifest')
    fsutil.makedirs(str(directory), exist_ok=True)
    source = args.data_root.resolve() / args.task
    if fsutil.exists(str(source)):
        raise ValueError('Generate into a new data root to avoid stale shards')
    started = time.monotonic()
    counts = generate_data(source, args.task, args.scale, args.feature_width,
                           args.seed)
    generated = time.monotonic()
    kumo.configure(scratch_dir=directory / 'scratch')
    graph = source_graph(source)
    snapshot = graph.snapshot()
    snapshot_done = time.monotonic()
    query = kumo.PredictiveQuery(graph, QUERIES[args.task])
    query.validate()
    start, end = datetime(2020, 1, 1), datetime(2020, 1, 1) + timedelta(days=504)
    horizon = timedelta(days=HORIZONS[args.task])
    ranges = [((start if offset == 3 else end - offset * horizon).isoformat(),
               (end - (offset - 1) * horizon).isoformat())
              for offset in (3, 2, 1)]
    generation = TrainingTableGenerationPlan(
        split=f'TimeRangeSplit({ranges!r}, timeframe={horizon.days}, '
              'flexible_boundaries=False)', timeframe_step=horizon.days)
    training = query.generate_training_table(snapshot, plan=generation)
    training.export(kumo.TrainingTableOutputConfig(
        kumo.FileConnector(root_dir=str(directory)), 'rows'))
    rows = pa.concat_tables([read_table(path) for path in fsutil.listdir(
        str(directory / 'rows'), suffix='.parquet')])
    if set(rows['SPLIT'].to_pylist()) != {0, 1, 2}:
        raise ValueError('One of the three task stages is empty')
    selected = rows.filter(pc.equal(rows['SPLIT'], 2)).sort_by(
        [('ENTITY1', 'ascending'), ('TIMESTAMP', 'ascending')])
    ids = selected['ENTITY1'].to_pylist()
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Prediction population must be unique and nonempty')
    heldout_entities = len(ids)
    if len(ids) > args.prediction_entities:
        selected = selected.take(np.linspace(
            0, len(ids) - 1, args.prediction_entities, dtype=np.int64))
        ids = selected['ENTITY1'].to_pylist()
    anchor = selected['TIMESTAMP'][0].as_py()
    if any(stamp != anchor for stamp in selected['TIMESTAMP'].to_pylist()):
        raise ValueError('Expected one held-out prediction anchor')
    inputs = directory / 'prediction-inputs.parquet'
    write_table(inputs, selected.select(['ENTITY1', 'TIMESTAMP']))
    index_files = file_tree(str(snapshot.index_path))
    all_files = sorted(set(index_files + file_tree(str(training.staging_dir))
                           + file_tree(str(directory / 'rows')) + [str(inputs)]))
    config = dict(task=args.task, scale=args.scale, width=args.feature_width,
                  seed=args.seed, row_counts=counts, generator_version=1,
                  prediction_entities=args.prediction_entities)
    finished = time.monotonic()
    json_write(directory / 'manifest.json', dict(
        version=1, task=args.task, source=str(source), config=config,
        config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()
                                     ).hexdigest(),
        source_sha256={path: file_hash(path) for path in file_tree(str(source))},
        snapshot={'id': snapshot.id, 'index_path': str(snapshot.index_path),
                  'graph_config': snapshot.graph_config},
        training={'query': training.query,
                  'train_table_path': str(training.train_table_path),
                  'manifest_path': str(training.manifest_path),
                  'staging_dir': str(training.staging_dir),
                  'required_stages': list(training.required_stages),
                  'generation_plan': asdict(training.generation_plan)},
        inputs=str(inputs), ids=ids, anchor=anchor.isoformat(),
        heldout_entities=heldout_entities,
        pristine_owned_files=[{
            'path': str(Path(path).relative_to(directory)),
            'size': fsutil.filesize(path), 'sha256': file_hash(path),
        } for path in all_files],
        index_files=index_files, owned_files=all_files,
        index_bytes=sum(fsutil.filesize(path) for path in index_files),
        training_rows=rows.num_rows,
        preparation_seconds={'generate': generated - started,
                             'snapshot': snapshot_done - generated,
                             'training_table': finished - snapshot_done},
    ))


def controlled_plan(task: str, steps: int, batch_size: int):
    from kumoapi.encoder import NumericalList
    from kumoapi.model_plan import GNNModelPlan

    plan = make_plan(task)
    plan.optimization.max_epochs = 1
    plan.optimization.min_steps_per_epoch = steps
    plan.optimization.max_steps_per_epoch = steps
    plan.optimization.max_val_steps = 4
    plan.optimization.max_test_steps = 4
    plan.optimization.batch_size = [batch_size]
    plan.optimization.early_stopping = [None]
    plan.model_architecture.model = [GNNModelPlan(
        channels=[16], aggregation=[['mean']], dropout=[0.0])]
    plan.column_processing.encoder_overrides.update({
        f'{name}.features': NumericalList(scaler=None)
        for name in ('users', 'items', 'events')
    })
    return plan


@contextmanager
def observe_work(state: dict, workers: int, threads: int, mode: str,
                 namespace: str):
    """Count completed batches without synchronizing individual CUDA batches."""
    import diskgraph
    import pytorch_lightning as pl
    from kumo_core.prediction import materialize_prediction_task
    from kumo_core.train.diskgraph_nested_trainer import (
        _ttng_diskgraph_runtime_config,
    )
    from kumo_core.training.workflow import materialize_generated_training_task

    class Proof(pl.Callback):
        def on_fit_start(self, trainer, module):
            state['fits'] += 1

        def on_train_start(self, trainer, module):
            state['optimizer_count'] = len(trainer.optimizers)
            state['training_interval'] = {
                'started': time.perf_counter(),
                'native_before': diskgraph.page_io_stats().to_dict(),
                'resources_before': group_snapshot(cgroup_path()),
            }

        def on_train_end(self, trainer, module):
            interval = state['training_interval']
            interval.update(
                seconds=time.perf_counter() - interval['started'],
                native_after=diskgraph.page_io_stats().to_dict(),
                resources_after=group_snapshot(cgroup_path()))

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            state['train_batches'] += 1

        def on_fit_end(self, trainer, module):
            state['optimizer_steps'] += int(trainer.global_step)
            state['completed_fits'] += 1

        def on_predict_batch_end(self, trainer, module, outputs, batch,
                                 batch_idx, dataloader_idx=0):
            state['predict_batches'][state['phase']] += 1

    class ObservedTrainer(pl.Trainer):
        def __init__(self, *args, **kwargs):
            kwargs['callbacks'] = [*(kwargs.get('callbacks') or []), Proof()]
            super().__init__(*args, **kwargs)

    def configured(*args, **kwargs):
        config = _ttng_diskgraph_runtime_config(*args, **kwargs)
        if config is None:
            raise ValueError('Real benchmark did not select a native runtime')
        automatic = dict(config)
        config.update(num_workers=workers, rayon_threads=threads)
        if mode == 'standalone':
            config['mode'] = 'standalone'
        state['loaders'].append({'automatic': automatic, 'effective': config})
        return config

    def materialize(request, task_dataset, generated, *, tt_id, runtime=None):
        actual = f'{namespace}-fit'
        if fsutil.exists(f'{request.prepared_graph.index_path}/train_tables/{actual}'):
            raise ValueError('Controlled training task namespace already exists')
        state['task_namespaces'].append({'phase': 'fit', 'namespace': actual})
        return materialize_generated_training_task(
            request, task_dataset, generated, tt_id=actual, runtime=runtime)

    def materialize_prediction(request, generated, *, graph_index_id,
                               runtime=None):
        phase = state['phase']
        if phase not in {'predict', 'repeat', 'reload'}:
            raise ValueError(f'Unexpected prediction materialization phase: {phase}')
        actual = f'{namespace}-{phase}'
        if fsutil.exists(f'{request.index_path}/train_tables/{actual}'):
            raise ValueError('Controlled prediction task namespace already exists')
        state['task_namespaces'].append({'phase': phase, 'namespace': actual})
        return materialize_prediction_task(
            replace(request, tt_id=actual), replace(generated, tt_id=actual),
            graph_index_id=graph_index_id, runtime=runtime)

    with patch.object(pl, 'Trainer', ObservedTrainer), patch(
            'kumo_core.train.diskgraph_nested_trainer._ttng_diskgraph_runtime_config',
            configured), patch(
                'kumo_core.training.workflow.materialize_generated_training_task',
                materialize), patch(
                    'kumo_core.prediction.materialize_prediction_task',
                    materialize_prediction):
        yield


def check_work(state: dict, steps: int) -> None:
    if (state['fits'] != 1 or state['completed_fits'] != 1
            or state['optimizer_steps'] != steps * state.get('optimizer_count', 1)
            or state['train_batches'] != steps
            or set(state['predict_batches']) != {'predict', 'repeat', 'reload'}
            or not all(state['predict_batches'].values())):
        raise ValueError(f'Completed work differs from fixed plan: {state}')


def run(args) -> dict:
    import diskgraph
    import diskgraph._c as native
    import torch
    from kumo_core.io.filesystem import local_path
    from kumo_core.io.json import from_json_dict
    from kumo_core.model_plan import TrainingTableGenerationPlan
    from kumo_core.train.checkpoint import load_kumo_checkpoint
    from kumoapi.jobs import MetadataField

    import kumo.engine as kumo
    from kumo.engine.graph import GraphSnapshot
    from kumo.engine.pquery.training_table import TrainingTable

    manifest = json.loads(fsutil.read_text(str(args.prepared_dir / 'manifest.json')))
    if manifest['task'] != args.task:
        raise ValueError('Prepared task differs from selected task')
    validate_clone(manifest, str(args.prepared_dir))
    build = (None if args.build_provenance is None else json.loads(
        fsutil.read_text(str(args.build_provenance))))
    native_sha256 = file_hash(native.__file__)
    if build is not None and native_sha256 != build['native']['sha256']:
        raise ValueError('Native module differs from supplied build provenance')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    output = args.output.resolve()
    fsutil.makedirs(str(output.parent), exist_ok=True)
    work = output.parent / (output.stem + '-work')
    if fsutil.exists(str(work)):
        raise ValueError('Use a new output path for each fresh-process run')
    fsutil.makedirs(str(work))
    kumo.configure(scratch_dir=work / 'scratch')
    graph = source_graph(Path(manifest['source']))
    query = kumo.PredictiveQuery(graph, QUERIES[args.task])
    snapshot = GraphSnapshot(**{
        **manifest['snapshot'],
        'index_path': Path(manifest['snapshot']['index_path']),
    })
    saved = manifest['training']
    training = TrainingTable(
        query=saved['query'], train_table_path=Path(saved['train_table_path']),
        manifest_path=Path(saved['manifest_path']),
        staging_dir=Path(saved['staging_dir']),
        required_stages=tuple(saved['required_stages']),
        generation_plan=from_json_dict(saved['generation_plan'],
                                       TrainingTableGenerationPlan))
    namespace = f'tt-native-io-benchmark-{args.task}-v1'
    state = dict(fits=0, completed_fits=0, optimizer_steps=0, train_batches=0,
                 phase='fit', predict_batches={name: 0 for name in (
                     'predict', 'repeat', 'reload')}, loaders=[], task_namespaces=[])
    seconds, counters, resources_per_phase = {}, {}, {}
    group = cgroup_path()

    def measure(phase, function):
        state['phase'] = phase
        before = diskgraph.page_io_stats().to_dict()
        resource_before = group_snapshot(group)
        started = time.perf_counter()
        value = function()
        seconds[phase] = time.perf_counter() - started
        after = diskgraph.page_io_stats().to_dict()
        resource_after = group_snapshot(group)
        resources_per_phase[phase] = dict(before=resource_before,
                                          after=resource_after)
        counters[phase] = {
            'before': before, 'after': after,
            'delta': {key: value - before[key] for key, value in after.items()
                      if type(value) is int and key not in {
                          'pid', 'runtime_rings', 'cache_entries',
                          'cache_capacity_pages', 'max_batch_pages'}},
            'scope': 'process_only',
        }
        return value

    def predict(model, stage):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        inputs = Path(manifest['inputs'])
        table = kumo.PredictionTable.from_source_table(
            kumo.FileConnector(root_dir=str(inputs.parent))[inputs.name], pq=query,
            snapshot=snapshot, plan=model.training_table_generation_plan)
        options = dict(output_config=kumo.PredictionOutputConfig(
            connector=kumo.FileConnector(root_dir=str(work)),
            output_paths={'predictions': f'predictions/{stage}'},
            extra_fields=[MetadataField.ANCHOR_TIMESTAMP]))
        if args.task == 'temporal_link':
            options['num_classes_to_return'] = 10
        if args.task == 'binary':
            options['binary_classification_threshold'] = .5
        frame = model.predict(table, snapshot=snapshot, num_workers=1,
                              **options).frame
        if args.task == 'temporal_link':
            if 'RANK' not in frame:
                frame = frame.assign(
                    RANK=frame.groupby('ENTITY', sort=False).cumcount())
            if (set(frame['ENTITY']) != set(manifest['ids'])
                    or frame.duplicated(['ENTITY', 'CLASS']).any()
                    or not frame.groupby('ENTITY').size().eq(10).all()):
                raise ValueError('Link predictions are incomplete or repeated')
            ordered = frame.sort_values(['ENTITY', 'RANK'])
            values = ordered['SCORE'].to_numpy().reshape(-1, 10)
            ranks = ordered['RANK'].to_numpy().reshape(-1, 10)
            classes = ordered['CLASS'].to_numpy().reshape(-1, 10)
            return values, ranks, classes
        values = prediction_values(
            args.task, frame, manifest['ids'],
            anchor=datetime.fromisoformat(manifest['anchor']))
        return values, prediction_decisions(
            args.task, frame, manifest['ids']), None

    with observe_work(state, args.workers, args.rayon_threads, args.loader_mode,
                      namespace):
        plan = controlled_plan(args.task, args.steps, args.batch_size)
        plan.training_job.manual_seed = args.seed
        requested_model_plan = asdict(plan)
        requested_plan_sha = hashlib.sha256(json.dumps(
            requested_model_plan, sort_keys=True, separators=(',', ':'),
            allow_nan=False).encode()).hexdigest()
        trainer = kumo.Trainer(plan, accelerator='cpu')
        started = time.perf_counter()
        fitted = measure('fit', lambda: trainer.fit(training, snapshot=snapshot))
        before, decisions, classes = measure(
            'predict', lambda: predict(trainer, 'first'))
        seconds['e2e'] = time.perf_counter() - started
        repeated, repeat_decisions, repeat_classes = measure(
            'repeat', lambda: predict(trainer, 'repeat'))
        restored = kumo.Trainer.load(fitted.bundle_dir, accelerator='cpu')
        after, after_decisions, after_classes = measure(
            'reload', lambda: predict(restored, 'reload'))
    check_work(state, args.steps)
    expected_namespaces = [{'phase': phase, 'namespace': f'{namespace}-{phase}'}
                           for phase in ('fit', 'predict', 'repeat', 'reload')]
    if state['task_namespaces'] != expected_namespaces:
        raise ValueError('Trial did not ingest four distinct controlled task namespaces')
    for record in expected_namespaces:
        path = snapshot.index_path / 'train_tables' / record['namespace']
        if not fsutil.exists(str(path / 'manifest.json')):
            raise ValueError('Trial did not publish its task inside the private index')
    checkpoint = str(local_path(str(fitted.artifact_paths['selected_checkpoint'])))
    checkpoint_steps = load_kumo_checkpoint(
        checkpoint, map_location='cpu')['global_step']
    if checkpoint_steps != state['optimizer_steps']:
        raise ValueError('Selected checkpoint does not contain the trained steps')
    if args.backend != 'mmap' and not counters['fit']['after']['submitted']:
        raise ValueError('Requested backend completed no native I/O')
    if (args.backend == 'direct'
            and int(os.environ.get('DG_DIRECT_CACHE_GB', '0')) > 0
            and not counters['fit']['after']['cache_initialized']):
        raise ValueError('Requested native cache was not initialized')
    for values, labels in ((repeated, repeat_decisions), (after, after_decisions)):
        compare_predictions(before, values, task=args.task, atol=1e-6,
                            before_decisions=decisions, after_decisions=labels)
    if not (np.array_equal(classes, repeat_classes)
            and np.array_equal(classes, after_classes)):
        raise ValueError('Reload/repeat changed the recommended item identities')
    work_files = file_tree(str(work))
    runtime_paths = set()
    for path in work_files:
        if path.endswith('/runtime_manifest.json'):
            runtime = json.loads(fsutil.read_text(path))
            for stage in runtime['stages'].values():
                physical = Path(stage['index_path']).resolve()
                if not (physical.is_relative_to(args.prepared_dir.resolve())
                        or physical.is_relative_to(work)):
                    raise ValueError('Runtime reads a stage outside the private trial')
                runtime_paths.add(str(physical))
    if not runtime_paths:
        raise ValueError('Trial produced no runtime stage provenance')
    schemas = [path for path in work_files
               if path.endswith('/feature_schema.json')]
    if not schemas:
        raise ValueError('Training produced no materialized feature schema')
    for path in schemas:
        schema = json.loads(fsutil.read_text(path))['node_types']
        for name in ('users', 'items', 'events'):
            columns = schema[name]['column_specs']
            if not any(column['column_name'] == 'features'
                       and column['encoder']['name'] == 'NumericalList'
                       for column in columns):
                raise ValueError(f'Dense features were not used for {name}')
    config = dict(prepared=manifest['config_sha256'], steps=args.steps,
                  batch_size=args.batch_size, workers=args.workers,
                  threads=args.threads, rayon_threads=args.rayon_threads,
                  seed=args.seed, loader_mode=args.loader_mode,
                  task_namespace=namespace,
                  requested_model_plan_sha256=requested_plan_sha)
    result = dict(
        version=1, status='PASS', task=args.task, backend=args.backend,
        requested_model_plan=requested_model_plan,
        regime=args.regime, repeat=args.repeat, phase_seconds=seconds,
        config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()
                                     ).hexdigest(),
        config=config, native_io=counters,
        resources_per_phase=resources_per_phase,
        work={'optimizer_steps': state['optimizer_steps'],
              'optimizer_count': state['optimizer_count'],
              'selected_checkpoint_steps': checkpoint_steps,
              'selected_checkpoint_sha256': file_hash(checkpoint),
              'verified_feature_tables': ['users', 'items', 'events'],
              'runtime_stage_paths': sorted(runtime_paths),
              'train_batches': state['train_batches'],
              'predict_rows': len(manifest['ids']),
              'predict_batches': state['predict_batches'],
              'loaders': state['loaders'],
              'materialization_calls': state['task_namespaces'],
              'training_interval': state['training_interval']},
        predictions={'values': before.tolist(),
                     'decisions': None if decisions is None else decisions.tolist(),
                     'classes': None if classes is None else classes.tolist()},
        provenance={'engine': kumo.__file__, 'native': native.__file__,
                    'native_sha256': native_sha256,
                    'engine_version': kumo.__version__,
                    'diskgraph_version': importlib.metadata.version('diskgraph'),
                    'python': sys.version, 'torch': torch.__version__,
                    'build': build,
                    'isolation': manifest['isolation'],
                    'prepared_manifest': str(args.prepared_dir / 'manifest.json'),
                    'index_bytes': manifest['index_bytes'],
                    'environment': {key: os.environ.get(key) for key in (
                        'DG_EDGE_BACKEND', 'DG_DIRECT_CACHE_GB', 'DG_IO_RINGS',
                        'DG_IO_QD', 'RAYON_NUM_THREADS', 'PYTHONHASHSEED')}},
    )
    verify_baseline(manifest)
    json_write(output, result)
    if args.cleanup_work:
        fsutil.rm(str(work))
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--backend', choices=('mmap', 'buffered', 'direct'),
                        default='mmap')
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--build-provenance', type=Path)
    parser.add_argument('--cleanup-work', action='store_true')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--feature-width', type=int, default=256)
    parser.add_argument('--prediction-entities', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--loader-mode', choices=('auto', 'standalone'),
                        default='standalone')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--rayon-threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--regime', default='uncontrolled')
    args = parser.parse_args(argv)
    for name in ('scale', 'feature_width', 'steps', 'batch_size', 'threads',
                 'rayon_threads', 'prediction_entities'):
        if vars(args)[name] < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.workers < 0 or args.repeat < 0:
        parser.error('--workers and --repeat must be nonnegative')
    if args.prepare and args.data_root is None:
        parser.error('--prepare requires --data-root')
    if not args.prepare and args.output is None:
        parser.error('A measured run requires --output')
    return args


def main(argv=None) -> int:
    args = arguments(argv)
    os.environ['DG_EDGE_BACKEND'] = args.backend
    os.environ['RAYON_NUM_THREADS'] = str(args.rayon_threads)
    cache = int(os.environ.get('DG_DIRECT_CACHE_GB', '0'))
    if cache < 0:
        raise ValueError('DG_DIRECT_CACHE_GB must be a nonnegative integer')
    if args.prepare:
        prepare(args)
    else:
        run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
