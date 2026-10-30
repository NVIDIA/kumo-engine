#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run frozen 100M nightly cases. Input artifacts contain data, never code."""

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import random
import resource
import shutil
import struct
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

HELPER = Path(__file__).with_name('wheel_gpu_smoke.py')
if not HELPER.is_file():
    HELPER = Path(__file__).resolve().parents[1] / 'test/wheel_gpu_smoke.py'
_SPEC = importlib.util.spec_from_file_location('nightly_smoke', HELPER)
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)
STAGES = ('train', 'val', 'test')
PREDICTIONS = ('predict', 'repeat', 'partial', 'reload')
KEY_FAILURES = ('key_parse_failures', 'key_lookup_failures',
                'key_membership_failures', 'pk_duplicate_total')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True,
                   allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def verify_artifacts(root, files, required=()):
    if not files or not set(required) <= set(files):
        raise ValueError(
            'Required retained artifacts are missing from the hash map')
    for name in files:
        safe_file(root, name, files)


def safe_file(root, name, files):
    relative = PurePosixPath(name)
    if (not isinstance(name, str) or relative.is_absolute() or '\\' in name
            or name != relative.as_posix()
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError(f'Invalid input path: {name}')
    path = Path(root)
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f'Input paths must be regular: {name}')
    if (not path.is_file() or name not in files or digest(path) != files[name]):
        raise ValueError(f'Missing or changed frozen input: {name}')
    return path


def load_inputs(root, expected):
    root = Path(root).resolve()
    manifest = root / 'manifest.json'
    if manifest.is_symlink() or digest(manifest) != expected:
        raise ValueError('Input manifest checksum differs')
    spec = read(manifest)
    if spec.get('format') != 'kumo-engine-nightly-inputs-v1':
        raise ValueError('Use the admitted nightly data manifest')
    if not isinstance(spec.get('files'), dict) or not spec['files']:
        raise ValueError('Input manifest needs exact regular-file hashes')
    for name in spec['files']:
        if Path(name).suffix not in ('.json', '.parquet', '.bin', ''):
            raise ValueError('Input artifacts contain data and metadata only')
        safe_file(root, name, spec['files'])
    identities = [(c['task'], c['arm'], c['seed']) for c in spec['cases']]
    wanted = {(t, a, 0) for t in smoke.TASKS for a in ('normal', 'control')}
    if (len(identities) != 16 or set(identities) != wanted
            or any(type(c['seed']) is not int for c in spec['cases'])):
        raise ValueError(
            'Input manifest requires exactly sixteen seed-zero cases')
    for case in spec['cases']:
        graph = spec['graphs'][case['graph']]
        if (graph['profile'] != smoke.TASKS[case['task']]['graph']
                or graph['static'] is not (case['task'] == 'static_link')):
            raise ValueError('Task graph profile or static visibility differs')
        failures = case['native_expectations']['key_failures']
        if set(failures) != set(KEY_FAILURES) or any(
                type(v) is not int or v for v in failures.values()):
            raise ValueError(
                'Admitted native key failures must be exactly four integer zeros'
            )
    safe_file(root, spec['candidate_dependencies'], spec['files'])
    for graph in spec['graphs'].values():
        safe_file(root, graph['schema'], spec['files'])
        for source in graph['sources'].values():
            path = root / source
            if source.startswith('/') or '..' in PurePosixPath(source).parts:
                raise ValueError('Graph sources must stay under the input root')
            paths = list(path.rglob('*')) if path.is_dir() else [path]
            if not paths or any(p.is_symlink() for p in paths):
                raise ValueError('Graph sources must be regular frozen files')
            for item in paths:
                if item.is_file() and str(
                        item.relative_to(root)) not in spec['files']:
                    raise ValueError('Graph contains an undeclared source file')
    return spec


def check_dependencies(packages, profile):
    normalize = lambda name: name.lower().replace('_', '-').replace('.', '-')
    if {
            normalize(k): v
            for k, v in packages.items()
    } != {
            normalize(k): v
            for k, v in profile.items()
    }:
        raise ValueError(
            'Installed non-Engine packages differ from the frozen profile')


def verify_runtime(version, profile, allowed_uuids):
    import torch

    import kumo.engine as kumo

    normalize = lambda name: name.lower().replace('_', '-').replace('.', '-')
    packages = {
        normalize(d.metadata['Name']): d.version
        for d in importlib.metadata.distributions()
        if normalize(d.metadata['Name']) != 'kumo-engine'
    }
    check_dependencies(packages, profile)
    if (sys.platform != 'linux' or sys.version_info[:2] != (3, 12)
            or importlib.metadata.version('kumo-engine') != version
            or kumo.__version__ != version or
            not Path(kumo.__file__).resolve().is_relative_to(Path(sys.prefix))
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError(
            'Use the installed Linux3.12 wheel and one actual CUDA GPU')
    torch.set_num_threads(4)
    properties = torch.cuda.get_device_properties(0)
    identifier = str(properties.uuid)
    uuid.UUID(identifier.removeprefix('GPU-'))
    if ('RTX PRO 6000' not in properties.name or not allowed_uuids
            or identifier.removeprefix('GPU-')
            not in {v.removeprefix('GPU-')
                    for v in allowed_uuids}):
        raise ValueError('CUDA work requires an admitted retained RTX GPU')
    direct = json.loads(
        importlib.metadata.distribution('kumo-engine').read_text(
            'direct_url.json'))
    wheel_hash = direct['archive_info']['hashes']['sha256']
    if len(wheel_hash) != 64:
        raise ValueError('Installed wheel archive identity is missing')
    return dict(version=version, dependencies=packages, gpu=properties.name,
                gpu_uuid=identifier,
                gpu_memory_bytes=int(properties.total_memory),
                torch=torch.__version__, cuda=torch.version.cuda, threads=4,
                import_path=kumo.__file__, wheel_sha256=wheel_hash)


def load_table(root, files, case):
    import pyarrow.parquet as pq
    from kumo_core.io.json import from_json_dict
    from kumo_core.model_plan import TrainingTableGenerationPlan
    from kumoapi.json_serde import to_json_dict

    import kumo.engine as kumo

    descriptor = safe_file(root, case['descriptor'], files)
    value = read(descriptor)
    task, arm = case['task'], case['arm']
    if (value['format'] != 'wheel-scale-control-v1' or value['case'] != task
            or value['arm'] != ('normal' if arm == 'normal' else 'control')
            or value['seed'] != (None if arm == 'normal' else 0)
            or value['query'] != smoke.TASKS[task]['query']
            or value['identity']['control_seed'] != value['seed']
            or value['train_table_path'] != 'tt.parquet'
            or value['manifest_path'] != 'manifest.json'):
        raise ValueError('Frozen task, arm, query or native paths differ')
    staging = descriptor.parent / 'staging'
    declared = {
        str(p.relative_to(staging))
        for p in staging.rglob('*') if p.is_file()
    }
    if declared != set(value['files']) or any(p.is_symlink()
                                              for p in staging.rglob('*')):
        raise ValueError('Frozen staging has missing, extra or linked files')
    for name, expected in value['files'].items():
        path = safe_file(root, str((staging / name).relative_to(root)), files)
        if digest(path) != expected:
            raise ValueError('Descriptor and artifact hashes differ')
    spec, manifest = read(staging / 'tt_ingest_spec.json'), read(
        staging / 'manifest.json')
    if spec['schema_version'] != 'diskgraph.tt_ingest_spec.v3':
        raise ValueError('Unexpected frozen native schema')
    if task == 'static_link':
        fact = spec.get('fact_pseudo_time')
        if (manifest['version'] != 4
                or value['native_split_kind'] != 'CustomLinkSplit'
                or manifest['split_kind'] != 'CustomLinkSplit'
                or fact != dict(table='purchases', entity_col='customer_id',
                                target_col='merchant_id')
                or manifest['fact_pseudo_time'] != dict(
                    path='fact_pseudo_time.bin',
                    format='tt_fact_pseudo_time_v1', node_type='purchases')
                or manifest['checksums']['fact_pseudo_time_sha256']
                != value['files']['fact_pseudo_time.bin']):
            raise ValueError('Static native routing or fact visibility differs')
    elif manifest['version'] != 3:
        raise ValueError('Unexpected frozen native version')
    stages = {v['key']: v for v in spec['split_stages']}
    if set(stages) != set(STAGES):
        raise ValueError('Frozen table needs all three native stages')
    for stage, entry in stages.items():
        rows = sum(
            pq.ParquetFile(staging / p).metadata.num_rows
            for p in entry['parts'])
        if type(case['stage_counts'][stage]) is not int or not rows or (
                rows != entry['row_count']
                or rows != case['stage_counts'][stage]):
            raise ValueError('Frozen native stage count differs')
    table = kumo.TrainingTable(
        query=value['query'], staging_dir=staging,
        train_table_path=staging / 'tt.parquet',
        manifest_path=staging / 'manifest.json',
        required_stages=tuple(value['required_stages']),
        generation_plan=from_json_dict(value['generation_plan'],
                                       TrainingTableGenerationPlan))
    if to_json_dict(table.generation_plan) != value['generation_plan']:
        raise ValueError('Installed wheel changed the frozen generation plan')
    return table, stages, value


def stage_frame(table, entry):
    import pandas as pd
    return pd.concat(
        [pd.read_parquet(table.staging_dir / p) for p in entry['parts']],
        ignore_index=True)


def check_inputs(frame, truth, *, partial=False, static=False):
    required = {'ENTITY1'} if static else {'ENTITY1', 'TIMESTAMP'}
    if 'autoregression_labels' in truth:
        required.add('autoregression_labels')
    if set(frame) != required:
        raise ValueError(
            'Prediction inputs omit required anchors/history or include labels')
    if (frame.empty or frame.ENTITY1.isna().any() or not frame.ENTITY1.is_unique
            or not truth.ENTITY1.is_unique):
        raise ValueError('Prediction populations must be nonempty and unique')
    ids, expected = frame.ENTITY1.tolist(), truth.set_index('ENTITY1')
    if (not set(ids) <= set(expected.index)
            or not partial and set(ids) != set(expected.index)
            or partial and len(ids) >= len(expected)):
        raise ValueError('Full or partial prediction population differs')
    for column in frame.columns:
        if column != 'ENTITY1':
            if column not in expected or signature(
                    frame[column].astype(str).tolist()) != signature(
                        expected.loc[ids, column].astype(str).tolist()):
                raise ValueError(
                    'Prediction anchors or history differ from frozen rows')
    return ids


def check_training(state, selected_steps):
    epochs = state['training_epochs']
    if (state['core_fit_attempts'] != 1 or state['fits'] != 1
            or state['completed_fits'] != 1 or not 1 <= len(epochs) <= 10
            or any(e['batches'] != 20 or len(e['batch_sizes']) != 20
                   or set(e['batch_sizes']) != {64} for e in epochs)
            or state['training_batches'] != 20 * len(epochs)
            or type(state['optimizer_steps']) is not int
            or state['optimizer_steps'] <= 0 or type(selected_steps) is not int
            or not 0 < selected_steps <= state['optimizer_steps']):
        raise ValueError(
            'Fit count, fixed training work or optimizer updates differ')
    if len(epochs) < 10:
        stop = state.get('early_stopping', {})
        if (type(stop.get('stopped_epoch')) is not int
                or stop['stopped_epoch'] <= 0 or stop.get('patience') != 3
                or stop.get('wait_count', 0) < 3):
            raise ValueError(
                'Short training lacks actual early-stopping evidence')


def stable_trainer(value):
    return {
        k: v
        for k, v in value.items()
        if k not in ('root_dir', 'version', 'kumo_version', 'refit_cfg')
    }


def check_recipe(case, directory):
    config = read(directory / 'trainer_config.json')
    if (signature(read(directory / 'resolved_model_plan.json'))
            != case['resolved_plan_sha256']
            or signature(read(directory / 'selected-model-plan.json'))
            != case['selected_plan_sha256'] or signature(
                stable_trainer(config)) != case['trainer_config_sha256']
            or any(config['refit_cfg'][k] is not False
                   for k in ('refitted_full', 'refitted_trainval'))
            or config['loader_kwargs']['batch_size'] != 64
            or config['sampler_kwargs']['adaptive_sampling'] is not False):
        raise ValueError(
            'Effective resolved/selected recipe, loader or trainer settings differ'
        )


def input_signature(manifest, case):
    return dict(descriptor=manifest['files'][case['descriptor']],
                prediction=manifest['files'][case['prediction_input']],
                partial=manifest['files'][case['partial_input']],
                requested_plan=case['requested_plan_sha256'])


def ranked_difference(before, after):
    added, removed, reordered, differences = 0, 0, 0, []
    for left, right, a, b in zip(before['ranked_ids'], after['ranked_ids'],
                                 before['ranked_scores'],
                                 after['ranked_scores'], strict=True):
        old, new = dict(zip(left, a,
                            strict=True)), dict(zip(right, b, strict=True))
        added += len(new.keys() - old.keys())
        removed += len(old.keys() - new.keys())
        reordered += int(left != right)
        differences.extend(
            abs(old[k] - new[k]) for k in old.keys() & new.keys())
    return dict(added_items=added, removed_items=removed,
                reordered_entities=reordered,
                max_abs_aligned_score_difference=max(differences, default=None),
                policy='report_only')


def ranked_rows(batches, ids, catalog, exclusions, k, raw_limit, *,
                anchor=None):
    """Certify top-K from a bounded, entity/score/key-ordered raw stream."""
    import pandas as pd
    if (not ids or len(set(ids)) != len(ids)
            or len(set(catalog)) != len(catalog) or len(exclusions) != len(ids)
            or type(raw_limit) is not int
            or not 0 < raw_limit <= min(500, len(catalog))):
        raise ValueError('Invalid ranked population or bounded output request')
    positions, members = {v: i for i, v in enumerate(ids)}, set(catalog)
    items, scores = [None] * len(ids), [None] * len(ids)
    current, pairs, count = None, [], 0

    def finish(entity):
        index = positions.get(entity)
        if index is None or items[index] is not None:
            raise ValueError('Repeated or foreign ranked entity')
        blocked = set(exclusions[index])
        if (len(blocked) != len(exclusions[index]) or not blocked <= members
                or len(pairs) != raw_limit or len({p[0]
                                                   for p in pairs}) != raw_limit
                or any(p[0] not in members or not math.isfinite(p[1])
                       for p in pairs)):
            raise ValueError(
                'Incomplete, duplicate or invalid ranked coordinates')
        wanted = min(k, len(catalog) - len(blocked))
        eligible = sorted((p for p in pairs if p[0] not in blocked),
                          key=lambda p: (-p[1], p[0]))[:wanted]
        if (not wanted or len(eligible) != wanted or raw_limit < len(catalog)
                and min(p[1] for p in pairs) >= eligible[-1][1]):
            raise ValueError(
                'Capped ranked output cannot certify the eligible cutoff')
        items[index], scores[index] = [p[0] for p in eligible
                                       ], [p[1] for p in eligible]

    for frame in batches:
        if not {'ENTITY', 'CLASS', 'SCORE'} <= set(frame):
            raise ValueError('Ranked columns are missing')
        for col in ('ENTITY', 'CLASS'):
            if (not pd.api.types.is_integer_dtype(frame[col])
                    or pd.api.types.is_bool_dtype(frame[col])
                    or frame[col].isna().any()):
                raise ValueError(
                    'Ranked IDs must retain their integer business type')
        if anchor is not None and (
                'ANCHOR_TIMESTAMP' not in frame or not pd.to_datetime(
                    frame.ANCHOR_TIMESTAMP).eq(pd.Timestamp(anchor)).all()):
            raise ValueError('Ranked prediction anchors differ')
        for entity, item, score in frame[['ENTITY', 'CLASS', 'SCORE'
                                          ]].itertuples(index=False, name=None):
            if current is not None and entity != current:
                finish(current)
                pairs = []
            current = entity
            pairs.append((int(item), float(score)))
            count += 1
            if len(pairs) > raw_limit:
                raise ValueError('Too many ranked rows for one entity')
    if current is not None:
        finish(current)
    if any(v is None for v in items):
        raise ValueError('Ranked output is missing entities')
    return dict(ranked_ids=items, ranked_scores=scores, raw_rows=count)


def ordered_batches(files, directory):
    import duckdb
    directory.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={
            'memory_limit': '256MB',
            'temp_directory': str(directory)
    }) as db:
        for batch in db.read_parquet(list(files)).order(
                'ENTITY, SCORE DESC, CLASS ASC').fetch_record_batch(65536):
            yield batch.to_pandas()


def quality(task, truth, values, classes):
    import numpy as np
    from sklearn import metrics
    targets = truth.TARGET1.tolist()
    if task.endswith('link'):
        return float(
            np.mean([
                len(set(y) & set(p)) / len(set(y))
                for y, p in zip(targets, values['ranked_ids'], strict=True)
            ]))
    if task == 'binary':
        return float(metrics.roc_auc_score(targets, values[:, 1])) if len(
            set(targets)) == 2 else None
    if task == 'multiclass':
        return float(
            np.mean(np.asarray(classes)[values.argmax(axis=1)] == targets))
    if task in ('multilabel', 'multilabel_ranking'):
        labels = [
            set(v.split('|') if isinstance(v, str) else v) for v in targets
        ]
        y = np.asarray([[c in row for c in classes] for row in labels],
                       dtype=int)
        if task == 'multilabel':
            return float(
                metrics.average_precision_score(y, values, average='macro'))
        return float(
            np.mean([
                sum(y[i, np.lexsort((np.arange(len(classes)), -row))[:5]]) /
                sum(y[i]) if sum(y[i]) else 0 for i, row in enumerate(values)
            ]))
    target = np.asarray(truth.future_labels.tolist() if task ==
                        'forecasting' else targets)
    return float(np.mean(np.abs(target.reshape(values.shape) - values)))


@contextmanager
def native_calls(state):
    import diskgraph
    from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule
    from kumo_core.train.trainer import Trainer
    original, ingest = Trainer.fit, diskgraph.execute_tt_ingest
    pseudo = DiskgraphLightningDataModule._pseudo_time_options

    def fit(*args, **kwargs):
        state['core_fit_attempts'] += 1
        row = dict(status='RUNNING', attempt=state['core_fit_attempts'])
        state.setdefault('core_fit_ledger', []).append(row)
        save(state['ledger_path'], state['core_fit_ledger'])
        try:
            if state['core_fit_attempts'] != 1:
                raise ValueError(
                    'Automatic fit retries or refits are forbidden')
            result = original(*args, **kwargs)
            row.update(status='PASS', cuda_batches=state['training_batches'],
                       steps=state['optimizer_steps'])
            return result
        except BaseException:
            row['status'] = 'FAIL'
            raise
        finally:
            save(state['ledger_path'], state['core_fit_ledger'])

    def consume(*args, **kwargs):
        result = ingest(*args, **kwargs)
        failures = result['key_failures']
        if any(
                type(failures.get(k)) is not int or failures[k] != 0
                for k in ('parse_failures', 'lookup_failures',
                          'membership_failures')):
            raise ValueError(
                'Native training-table ingest has missing or failed keys')
        state['native_ingests'].append(
            dict(extension=result['tt_extension_path'], key_failures=failures))
        return result

    def options(module, stage):
        result = pseudo(module, stage)
        state['pseudo_options'].setdefault(stage, []).append(result)
        return result

    with patch.object(Trainer, 'fit', fit), patch.object(
            diskgraph, 'execute_tt_ingest', consume), patch.object(
                DiskgraphLightningDataModule, '_pseudo_time_options',
                options), patch.object(Trainer, 'predict_v2',
                                       inspect.unwrap(
                                           Trainer.predict_v2)), patch.object(
                                               Trainer, 'test',
                                               inspect.unwrap(Trainer.test)):
        yield


def check_snapshot(snapshot, graph_spec):
    from kumo_core.diskgraph.metadata import load_diskgraph_schema
    stats = read(Path(snapshot.index_path) / 'ingest_stats.json')['stats']
    failures = {k: stats.get(k) for k in KEY_FAILURES}
    if (any(type(v) is not int or v != 0 for v in failures.values())
            or stats['node_counts'] != graph_spec['row_counts']):
        raise ValueError('Graph row counts or strict native key checks differ')
    schema = read(Path(graph_spec['input_root']) /
                  graph_spec['schema'])[graph_spec['profile']]
    if load_diskgraph_schema(str(snapshot.index_path)).pkey_dict != {
            name: table['pkey']
            for name, table in schema['tables'].items() if table.get('pkey')
    }:
        raise ValueError('Native graph primary-key roles differ')
    return dict(node_counts=stats['node_counts'], key_failures=failures)


def check_feature_schema(runtime, features, schema, *, static=False):
    expected_nodes = set(schema['tables'])
    extra = {'train_table'
             } if runtime['task']['task_type'] == 'forecasting' else set()
    if (not expected_nodes <= set(features['node_types'])
            or not set(features['node_types']) - expected_nodes <= extra):
        raise ValueError('Fitted graph is missing source tables')
    for name, table in schema['tables'].items():
        expected = {
            c['name']: (c['dtype'], c['stype'])
            for c in table['cols'] if c['stype'] != 'ID'
        }
        actual = features['node_types'][name]
        if ({
                c['column_name']: (c['dtype'], c['stype'])
                for c in actual['column_specs']
        } != expected or actual['time_column']
                != (None if static else table.get('time_col'))):
            raise ValueError('Fitted source feature types or time roles differ')
    if {
            tuple(e[k] for k in ('src_table', 'fkey', 'dst_table'))
            for e in schema['edges']
    } != {tuple(e)
          for e in runtime['graph']['edge_types']}:
        raise ValueError('Fitted source relationship roles differ')


def audit_link_rows(runtime_path, table, stages, graph_schema, graph_counts,
                    expected, state):
    """Read the real consumed native labels at each Core cutoff, without fitting."""
    import diskgraph
    import numpy as np
    import pyarrow.parquet as pq
    from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule
    from kumo_core.diskgraph.stores import read_node_column

    runtime = read(runtime_path)
    entity = runtime['table']['seed_entity_table']
    target = read(runtime_path.with_name('metadata.json'))['target_node_type']
    index = runtime['graph']['index_path']
    keys = {
        name:
        read_node_column(index, name, graph_schema['tables'][name]['pkey'],
                         graph_counts[name]).to_pylist()
        for name in {entity, target}
    }
    native = diskgraph.RuntimeIndex(str(runtime_path), num_threads=4)
    unit = read(Path(index) / 'index_metadata.json').get(
        'timestamp_unit', 'seconds').lower()
    scale = {
        'seconds': 10**9,
        'second': 10**9,
        'secs': 10**9,
        's': 10**9,
        'milliseconds': 10**6,
        'millis': 10**6,
        'ms': 10**6,
        'microseconds': 1000,
        'micros': 1000,
        'us': 1000,
        'nanoseconds': 1,
        'nanos': 1,
        'ns': 1
    }[unit]
    result = {}

    def checksum(rows):
        count, total, xor = 0, 0, 0
        for row in rows:
            value = int(signature(row), 16)
            count, total, xor = count + 1, (total +
                                            value) % (1 << 256), xor ^ value
        return dict(rows=count, sum=f'{total:064x}', xor=f'{xor:064x}')

    for number, stage in enumerate(STAGES):
        observed_options = list(state['pseudo_options'].get(stage, []))
        options = DiskgraphLightningDataModule._pseudo_time_options(
            SimpleNamespace(_task_type=runtime['task']['task_type']), stage)
        if expected and (expected['pseudo_cutoffs'][stage]
                         != options.get('pseudo_seed_time')
                         or options.get('disable_pseudo_time') is not False
                         or stage in ('train', 'val')
                         and options not in observed_options):
            raise ValueError('Actual Core training cutoffs differ')

        def source_rows():
            for part in stages[stage]['parts']:
                for batch in pq.ParquetFile(table.staging_dir /
                                            part).iter_batches():
                    for row in batch.to_pylist():
                        stamp = row['TIMESTAMP']
                        yield [
                            row['ENTITY1'],
                            int(np.datetime64(stamp, 'ns').astype('int64')),
                            sorted(row['TARGET1']),
                            sorted(row.get('exclude_target_keys', []))
                        ]

        def consumed_rows():
            iterator, count = native.iter(
                stage=stage, sampling_plan={'layers': []}, num_workers=1,
                worker_id=0, shuffle=False, epoch=0, batch_size=16384,
                entity_node_type=entity, target_node_type=target, **options)
            if count != stages[stage]['row_count']:
                raise ValueError('Native consumer stage size differs')
            for batch in iterator:
                task = batch['task']
                seeds, stamps, counts, positives = [
                    task[k].to_pylist()
                    for k in ('seed_node_ids', 'seed_timestamps',
                              'lp_pos_counts', 'lp_pos_rhs_global')
                ]
                mask = task.get('lp_pos_mask')
                masks = mask.to_pylist(
                ) if mask is not None else [False] * len(positives)
                ec, ex = task.get('lp_exclude_counts'), task.get(
                    'lp_exclude_rhs_global')
                if ((ec is None) != (ex is None) or len(seeds) != len(counts)
                        or sum(counts) != len(positives)
                        or len(masks) != len(positives)
                        or expected and mask is None
                        or any(type(v) is not bool or v for v in masks)):
                    raise ValueError('Native positive offsets or masks differ')
                ec, ex = ([0] * len(seeds),
                          []) if ec is None else (ec.to_pylist(),
                                                  ex.to_pylist())
                if len(ec) != len(seeds) or sum(ec) != len(ex):
                    raise ValueError('Native exclusion offsets differ')
                offset, excluded_offset = 0, 0
                for seed, stamp, count, excluded_count in zip(
                        seeds, stamps, counts, ec, strict=True):
                    if (type(count) is not int or count < 0
                            or type(excluded_count) is not int
                            or excluded_count < 0
                            or not 0 <= seed < len(keys[entity])
                            or any(not 0 <= v < len(keys[target])
                                   for v in positives[offset:offset + count] +
                                   ex[excluded_offset:excluded_offset +
                                      excluded_count])):
                        raise ValueError(
                            'Native IDs or offsets are outside their business-key maps'
                        )
                    yield [
                        keys[entity][seed],
                        int(stamp) * scale,
                        sorted(keys[target][v]
                               for v in positives[offset:offset + count]),
                        sorted(keys[target][v]
                               for v in ex[excluded_offset:excluded_offset +
                                           excluded_count])
                    ]
                    offset, excluded_offset = offset + count, excluded_offset + excluded_count

        left, right = checksum(source_rows()), checksum(consumed_rows())
        if left != right:
            raise ValueError(
                'Consumed raw IDs, positives, anchors or exclusions differ')
        result[stage] = right
    if expected:
        extension = Path(runtime['stages']['train']['index_path'])
        raw = (extension / 'fact_pseudo_time.bin').read_bytes()
        version, dtype, count = struct.unpack('<IIQ', raw[8:24])
        if raw[:8] != b'DGPTIME\0' or (version,
                                       dtype) != (1,
                                                  1) or len(raw) != 24 + count:
            raise ValueError('Consumed static fact sidecar is invalid')
        fact_keys = read_node_column(index, 'purchases', 'purchase_id',
                                     graph_counts['purchases']).to_pylist()
        if count != len(fact_keys) or any(v > 3 for v in raw[24:]) or signature(
                sorted(zip(fact_keys,
                           raw[24:]))) != expected['fact_pk_tags_sha256']:
            raise ValueError(
                'Consumed static raw fact-key visibility tags differ')
    return result


def run_case(root, manifest, case, graph, snapshot, output):
    import numpy as np
    import pandas as pd
    import torch
    from kumo_core.io.filesystem import local_path
    from kumo_core.train.checkpoint import load_kumo_checkpoint
    from kumoapi.jobs import MetadataField

    import kumo.engine as kumo

    output.mkdir(parents=True)
    task = case['task']
    started = time.monotonic()
    state = dict(stage='fit', fits=0, completed_fits=0, core_fit_attempts=0,
                 training_batches=0, optimizer_steps=0, training_epochs=[],
                 prediction_batches={s: 0
                                     for s in PREDICTIONS}, reload_checks=0,
                 restored_tensors=0, native_ingests=[], pseudo_options={},
                 fit_ledger=[])
    report = dict(task=task, arm=case['arm'], seed=0, status='RUNNING',
                  work=state)
    state['ledger_path'] = str(output / 'core-fits.json')
    table, stages, descriptor = load_table(root, manifest['files'], case)
    truth = stage_frame(table, stages['test'])
    full = pd.read_parquet(
        safe_file(root, case['prediction_input'], manifest['files']))
    partial = pd.read_parquet(
        safe_file(root, case['partial_input'], manifest['files']))
    ids = check_inputs(full, truth, static=task == 'static_link')
    partial_ids = check_inputs(partial, truth, partial=True,
                               static=task == 'static_link')
    truth = truth.set_index('ENTITY1').loc[ids].reset_index()
    if task == 'forecasting' and any(
            len(v) != 12 for col in ('autoregression_labels', 'future_labels')
            for v in truth[col]):
        raise ValueError(
            'Forecast inputs/answers require all twelve coordinates')
    query = kumo.PredictiveQuery(graph, descriptor['query'])
    query.validate()
    plan = smoke.make_plan(task)
    if signature(plan.as_json()) != case['requested_plan_sha256']:
        raise ValueError('Requested full seed-zero model recipe differs')
    save(output / 'requested-plan.json', plan.as_json())
    save(output / 'generation-plan.json', descriptor['generation_plan'])
    report['input_signature'] = input_signature(manifest, case)
    classes = case.get('classes', [])
    catalog = []
    if task.endswith('link'):
        spec = case['catalog']
        catalog = pd.read_parquet(
            safe_file(root, spec['path'], manifest['files']),
            columns=[spec['column']])[spec['column']].tolist()
        if not catalog or len(set(catalog)) != len(catalog):
            raise ValueError('Link catalog keys are missing or duplicated')
    anchor = None if task == 'static_link' else full.TIMESTAMP.iloc[0]
    if anchor is not None and not full.TIMESTAMP.eq(anchor).all():
        raise ValueError('Full prediction needs the fixed frozen test anchor')
    seed = lambda: (random.seed(0), np.random.seed(0), torch.manual_seed(0))
    work = output.parent / 'work' / output.name
    kumo.configure(scratch_dir=work)
    torch.cuda.reset_peak_memory_stats()
    peak = [0]
    stopped = threading.Event()

    def memory():
        import psutil
        process = psutil.Process()
        while not stopped.wait(.2):
            try:
                peak[0] = max(
                    peak[0],
                    sum(p.memory_info().rss
                        for p in [process, *process.children(recursive=True)]
                        if p.is_running()))
            except psutil.Error:
                pass

    monitor = threading.Thread(target=memory, daemon=True)
    monitor.start()
    try:
        seed()
        with smoke.observe_cuda(state), native_calls(state):
            trainer = kumo.Trainer(plan, accelerator='gpu')
            torch.cuda.synchronize()
            train_started = time.monotonic()
            fitted = trainer.fit(table, snapshot=snapshot)
            torch.cuda.synchronize()
            report['timings'] = dict(training=time.monotonic() - train_started,
                                     preparation=train_started - started)
            checkpoint = local_path(
                str(fitted.artifact_paths['selected_checkpoint']))
            state['checkpoint_sha256'] = digest(checkpoint)
            selected_steps = load_kumo_checkpoint(
                str(checkpoint), map_location='cpu')['global_step']
            check_training(state, selected_steps)
            [runtime_path] = trainer.scratch_dir.glob(
                'runs/*-fit-*/task_dataset/runtime_manifest.json')
            runtime = read(runtime_path)
            if runtime['task']['task_type'] != smoke.TASKS[task][
                    'task_type'] or {
                        name: v['row_count']
                        for name, v in runtime['stages'].items()
                    } != case['stage_counts']:
                raise ValueError('Fitted task or consumed stage counts differ')
            for name in ('runtime_manifest', 'metadata', 'feature_schema'):
                shutil.copyfile(runtime_path.with_name(name + '.json'),
                                output / (name + '.json'))
            for name in ('resolved_model_plan', 'trainer_config', 'encoders',
                         'target_encoder'):
                shutil.copyfile(fitted.artifact_paths[name],
                                output / (name + '.json'))
            [selected] = trainer.scratch_dir.glob(
                'runs/*-fit-*/models/*/automl/version_*/model_plan.json')
            shutil.copyfile(selected, output / 'selected-model-plan.json')
            check_recipe(case, output)
            graph_spec = manifest['graphs'][case['graph']]
            graph_schema = read(root /
                                graph_spec['schema'])[graph_spec['profile']]
            check_feature_schema(runtime, read(output / 'feature_schema.json'),
                                 graph_schema, static=task == 'static_link')
            if task.endswith('link'):
                report['native_consumption'] = audit_link_rows(
                    runtime_path, table, stages, graph_schema,
                    graph_spec['row_counts'], case['native_expectations']
                    if task == 'static_link' else None, state)
            saved = output / 'published-model'
            shutil.copytree(fitted.bundle_dir, saved)
            checkpoints = [
                p for p in saved.rglob('*')
                if p.is_file() and digest(p) == state['checkpoint_sha256']
            ]
            if not checkpoints:
                raise ValueError(
                    'Published model does not retain the selected checkpoint bytes'
                )
            report.update(
                fit_id=str(uuid.uuid4()) + ':' + str(fitted.bundle_dir),
                checkpoint_sha256=digest(checkpoint),
                checkpoint_artifact=str(checkpoints[0].relative_to(output)),
                selected_checkpoint_steps=selected_steps)
            time_column = read(
                runtime_path.with_name('feature_schema.json'))['node_types'][
                    runtime['table']['seed_entity_table']]['time_column']

            def predict(model, stage, frame):
                seed()
                torch.cuda.synchronize()
                stage_started = time.monotonic()
                state['stage'] = stage
                before = state['prediction_batches'][stage]
                inputs = output / 'prediction-inputs' / (stage + '.parquet')
                inputs.parent.mkdir(exist_ok=True)
                frame.to_parquet(inputs, index=False)
                pt = kumo.PredictionTable.from_source_table(
                    kumo.FileConnector(
                        root_dir=str(inputs.parent))[inputs.name], pq=query,
                    snapshot=snapshot,
                    plan=model.training_table_generation_plan)
                selected = frame.ENTITY1.tolist()
                options = dict(output_config=kumo.PredictionOutputConfig(
                    kumo.FileConnector(root_dir=str(output /
                                                    'prediction-outputs')),
                    output_paths={'predictions': stage}, extra_fields=[] if task
                    == 'static_link' else [MetadataField.ANCHOR_TIMESTAMP]))
                if task.endswith('link'):
                    options['num_classes_to_return'] = min(500, len(catalog))
                elif classes:
                    options['num_classes_to_return'] = len(classes)
                if task == 'binary':
                    options['binary_classification_threshold'] = .5
                result = model.predict(pt, snapshot=snapshot, **options)
                torch.cuda.synchronize()
                if state['prediction_batches'][stage] <= before:
                    raise ValueError(
                        'Prediction completed no actual CUDA batches')
                raw = output / 'prediction-outputs' / stage
                raw.mkdir(parents=True, exist_ok=True)
                retained = []
                for index, source in enumerate(result.files):
                    destination = raw / f'raw-{index}.parquet'
                    if Path(source).resolve() != destination.resolve():
                        shutil.copyfile(local_path(source), destination)
                    retained.append(str(destination.relative_to(output)))
                if task.endswith('link'):
                    selected_truth = truth.set_index('ENTITY1').loc[selected]
                    excluded = selected_truth.exclude_target_keys.tolist(
                    ) if task == 'static_link' else [[] for _ in selected]
                    values = ranked_rows(
                        ordered_batches(result.files,
                                        work / 'sort' / stage), selected,
                        catalog, excluded, 5 if task == 'static_link' else 10,
                        min(500, len(catalog)), anchor=anchor)
                    save(output / (stage + '-ranked.json'), values)
                else:
                    frame = result.frame
                    if frame.ENTITY.dtype != full.ENTITY1.dtype:
                        raise ValueError(
                            'Prediction changed the original entity ID type')
                    frame.to_parquet(output / (stage + '.parquet'), index=False)
                    values = smoke.prediction_values(
                        task, frame, selected, classes=classes, anchor=anchor,
                        time_column=time_column or 'TIMESTAMP')
                report.setdefault('predictions', {})[stage] = dict(
                    entities=len(selected), rows=result.num_rows,
                    files=retained, original_files=list(result.files),
                    input_sha256=digest(inputs))
                report['timings'][stage] = time.monotonic() - stage_started
                return values

            original = predict(trainer, 'predict', full)
            repeated = predict(trainer, 'repeat', full)
            subset = predict(trainer, 'partial', partial)
            del trainer
            gc.collect()
            trainer = kumo.Trainer.load(saved, accelerator='gpu')
            reloaded = predict(trainer, 'reload', full)
            if state['reload_checks'] < 1 or state['restored_tensors'] <= 0:
                raise ValueError(
                    'Reload lacks an exact restored checkpoint check')
            lookup = {v: i for i, v in enumerate(ids)}
            positions = [lookup[v] for v in partial_ids]
            report['differences'] = {}
            for label, other, selected in [('repeat', repeated, None),
                                           ('partial', subset, positions),
                                           ('reload', reloaded, None)]:
                if task.endswith('link'):
                    before = original if selected is None else {
                        k: [original[k][i] for i in selected]
                        for k in ('ranked_ids', 'ranked_scores')
                    }
                    report['differences'][label] = ranked_difference(
                        before, other)
                else:
                    report['differences'][label] = smoke.compare_predictions(
                        original if selected is None else original[selected],
                        other, atol=0, rtol=0, qualify=True)
            report.update(status='MEASURED',
                          physical_fits=state['core_fit_attempts'],
                          prediction_batches=state['prediction_batches'],
                          score=quality(task, truth, original, classes),
                          metric=smoke.TASKS[task]['metric'],
                          policy='report_only')
            check_training(state, selected_steps)
            load_table(root, manifest['files'], case)
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc(),
                      physical_fits=state['core_fit_attempts'])
        raise
    finally:
        stopped.set()
        monitor.join()
        report.update(
            seconds=time.monotonic() - started, cpu_peak_sampled_bytes=peak[0],
            cpu_memory_sample_interval_seconds=.2,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report['process_lifetime_peak_rss_bytes'] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss * 1024
        report['artifacts'] = {
            str(p.relative_to(output)): digest(p)
            for p in output.rglob('*')
            if p.is_file() and p != output / 'result.json'
        }
        report['seconds'] = time.monotonic() - started
        save(output / 'result.json', report)
    return report


def check_results(rows):
    wanted = {(task, arm, 0)
              for task in smoke.TASKS
              for arm in ('normal', 'control')}
    if (len(rows) != 16 or {(r['task'], r['arm'], r['seed'])
                            for r in rows} != wanted
            or len({r['fit_id']
                    for r in rows}) != 16 or
            any(r['status'] != 'MEASURED' or type(r['seed']) is not int
                or type(r['physical_fits']) is not int
                or r['physical_fits'] != 1 or len(r['checkpoint_sha256']) != 64
                or not all(r['prediction_batches'].get(s, 0) > 0
                           for s in PREDICTIONS) for r in rows)):
        raise ValueError(
            'Comparison needs sixteen complete unique one-fit workflows')


def load_report(root, expected):
    paths = list(Path(root).rglob('nightly-result.json'))
    if len(paths) != 1:
        raise ValueError('Exactly one complete sixteen-case report is required')
    path = paths[0]
    report = read(path)
    if (report['format'] != 'kumo-engine-nightly-result-v1' or 'shard' in report
            or report['status'] != 'MEASURED'
            or report['input_manifest_sha256'] != expected
            or report['expected_runs'] != 16 or report['completed_runs'] != 16):
        raise ValueError(
            'The single report must complete all sixteen frozen cases')
    check_results(report['runs'])
    return path, report


def run(args):
    started = time.monotonic()
    root, output = args.input_root.resolve(), args.output_dir.resolve()
    if output.exists() and any(p.name != 'setup' for p in output.iterdir()):
        raise ValueError('Use a fresh output directory')
    manifest = load_inputs(root, args.input_sha256)
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / 'input-metadata'
    metadata.mkdir()
    shutil.copyfile(root / 'manifest.json', metadata / 'manifest.json')
    profile = manifest['candidate_dependencies']
    (metadata / profile).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / profile, metadata / profile)
    runtime = verify_runtime(args.expected_version, read(root / profile),
                             manifest['allowed_gpu_uuids'])
    check_install(output, runtime)
    report = dict(format='kumo-engine-nightly-result-v1', status='RUNNING',
                  input_manifest_sha256=args.input_sha256, runtime=runtime,
                  expected_runs=16, runs=[], helper_sha256=digest(HELPER),
                  runner_sha256=digest(__file__))
    graphs = {}
    try:
        for case in manifest['cases']:
            key = case['graph']
            if key not in graphs:
                import kumo.engine as kumo
                spec = manifest['graphs'][key]
                kumo.configure(scratch_dir=output / 'work' / 'graphs' / key)
                graphs[key] = smoke.build_graph(
                    read(root / spec['schema'])[spec['profile']], root,
                    static=spec['static'], sources=spec['sources'])
                report.setdefault('graphs', {})[key] = check_snapshot(
                    graphs[key][1], dict(spec, input_root=str(root)))
            row = run_case(root, manifest, case, *graphs[key],
                           output / (case['task'] + '-' + case['arm']))
            report['runs'].append(row)
            save(output / 'nightly-result.json', report)
        check_results(report['runs'])
        load_inputs(root, args.input_sha256)
        report['status'] = 'MEASURED'
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        report.update(seconds=time.monotonic() - started,
                      completed_runs=len(report['runs']))
        save(output / 'nightly-result.json', report)


def check_install(root, runtime):
    wheel, build = read(root / 'setup/wheel.json'), read(root /
                                                         'setup/build.json')
    if (type(wheel['installed_files']) is not int
            or wheel['installed_files'] <= 0
            or wheel['sha256'] != build['wheel_sha256']
            or wheel['sha256'] != runtime['wheel_sha256']
            or wheel['version'] != build['version']
            or wheel['version'] != runtime['version'] or any(
                len(build[k]) != 40 or any(v not in '0123456789abcdef'
                                           for v in build[k])
                for k in ('revision', 'tree'))):
        raise ValueError(
            'Selected source, installed wheel or archive identity differs')
    return build


def check_behavior(root):
    import xml.etree.ElementTree as ET
    paths = list(root.rglob('behavior/result.json'))
    if len(paths) != 1:
        raise ValueError(
            'Exactly one complete required behavior result is needed')
    path = paths[0]
    value = read(path)
    coverage = {'I2', 'B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B8'}
    if (value['status'] != 'PASS' or set(value['coverage']) != coverage
            or any(v is not True for v in value['coverage'].values())
            or value['pytest_exit_code'] != 0 or value['bad']
            or value['deselected'] != 0 or value['collection_errors']
            or not value['collected']
            or len(set(value['collected'])) != len(value['collected'])
            or set(value['calls']) != set(value['collected'])
            or set(value['calls'].values()) != {'passed'}
            or type(value['physical_fits']) is not int
            or value['physical_fits'] <= 0
            or value['physical_fits'] != len(value['fit_ledger'])
            or any(r['status'] != 'PASS' or type(r['steps']) is not int
                   or r['steps'] <= 0 or type(r['cuda_batches']) is not int
                   or r['cuda_batches'] <= 0 for r in value['fit_ledger'])):
        raise ValueError(
            'Required behavior coverage, test calls or extra-fit ledger is incomplete'
        )
    verify_artifacts(path.parent, value['artifacts'])
    receipt = value['b7_receipt']
    child_path = safe_file(path.parent, receipt['path'], value['artifacts'])
    children = [
        p for p in value['artifacts']
        if p.startswith('pytest-temp/') and p.endswith('/child-result.json')
    ]
    required = [
        str(PurePosixPath(receipt['path']).with_name(n))
        for n in ('child.stdout', 'child.stderr', 'request.json')
    ]
    if (children != [receipt['path']]
            or value['artifacts'][receipt['path']] != receipt['sha256']
            or any(p not in value['artifacts']
                   for p in [*required, 'junit.xml'])):
        raise ValueError(
            'Original B7 child receipt or retained logs are missing')
    child = read(child_path)
    if (any(
            type(child.get(k)) is not int or child[k] != 0
            for k in ('fits', 'fit_attempts')) or child.get('state_checks') != 2
            or child.get('rows') != 1071
            or type(child.get('restored_tensors')) is not int
            or child['restored_tensors'] <= 0
            or type(child.get('prediction_batches', {}).get('reuse')) is not int
            or child['prediction_batches']['reuse'] <= 0):
        raise ValueError(
            'B7 child lacks actual zero-fit CUDA reuse and restored-state proof'
        )
    junit = ET.parse(path.with_name('junit.xml')).getroot()
    tests = list(junit.iter('testcase'))
    if (len(tests) != len(value['collected']) or any(
            test.find(name) is not None for test in tests
            for name in ('failure', 'error', 'skipped'))):
        raise ValueError(
            'Behavior JUnit contradicts its complete passing inventory')
    return dict(physical_fits=value['physical_fits'],
                coverage=value['coverage'], artifacts=value['artifacts'],
                b7_receipt=receipt)


def compare(args):
    manifest_path = args.input_root / 'manifest.json'
    if manifest_path.is_symlink() or digest(manifest_path) != args.input_sha256:
        raise ValueError('Copied input manifest checksum differs')
    manifest = read(manifest_path)
    reference = read(args.reference)
    if (digest(args.reference) != args.reference_sha256
            or reference['input_manifest_sha256'] != args.input_sha256
            or reference['format'] != 'kumo-engine-nightly-reference-v1'
            or reference['version'] != '1.0.0' or reference['wheel_sha256'] !=
            '1bc15ecb7696bce216b05aa883d6897d2b8ce4342c6c2f9d62b7cc28b314539e'):
        raise ValueError('Approved reference or input-manifest binding differs')
    path, report = load_report(args.results_dir, args.input_sha256)
    rows = report['runs']
    if report['runner_sha256'] != digest(
            __file__) or report['helper_sha256'] != digest(HELPER):
        raise ValueError('Candidate report used different source helpers')
    expected = read(
        safe_file(args.input_root, manifest['candidate_dependencies'],
                  manifest['files']))
    admitted = {(v['task'], v['arm']): v for v in manifest['cases']}
    check_dependencies(report['runtime']['dependencies'], expected)
    check_install(path.parent, report['runtime'])
    if digest(path.parent / 'reference.json') != args.reference_sha256:
        raise ValueError('Run did not retain the exact approved reference copy')
    if digest(path.parent /
              'input-metadata/manifest.json') != args.input_sha256:
        raise ValueError('Run did not retain the exact admitted input manifest')
    if report['runtime']['gpu_uuid'].removeprefix('GPU-') not in {
            v.removeprefix('GPU-')
            for v in manifest['allowed_gpu_uuids']
    }:
        raise ValueError('Run used an unapproved GPU identity')
    for row in report['runs']:
        directory = path.parent / (row['task'] + '-' + row['arm'])
        if read(directory / 'result.json') != row:
            raise ValueError(
                'Aggregate case row differs from original result bytes')
        required = [
            row['checkpoint_artifact'],
            'requested-plan.json',
            'generation-plan.json',
            'selected-model-plan.json',
            'resolved_model_plan.json',
            'trainer_config.json',
            'feature_schema.json',
            'runtime_manifest.json',
            'metadata.json',
            'encoders.json',
            'target_encoder.json',
            'core-fits.json',
        ]
        for stage in PREDICTIONS:
            required.extend(row['predictions'].get(stage, {}).get('files', []))
            required.append(f'prediction-inputs/{stage}.parquet')
            required.append(stage + (
                '-ranked.json' if row['task'].endswith('link') else '.parquet'))
        verify_artifacts(directory, row['artifacts'], required=required)
        check_training(row['work'], row['selected_checkpoint_steps'])
        case = admitted[(row['task'], row['arm'])]
        if row['input_signature'] != input_signature(manifest, case):
            raise ValueError(
                'Candidate input signatures differ from the admitted manifest')
        check_recipe(case, directory)
        if row['artifacts'].get(
                row['checkpoint_artifact']) != row['checkpoint_sha256']:
            raise ValueError(
                'Published checkpoint artifact and restored checkpoint identity differ'
            )
        for key in ('core_fit_ledger', 'fit_ledger'):
            ledger = row['work'].get(key, [])
            if len(ledger) != 1 or ledger[0]['status'] != 'PASS' or ledger[0][
                    'steps'] <= 0 or ledger[0]['cuda_batches'] <= 0:
                raise ValueError(
                    'Core or Lightning physical-fit ledger is incomplete')
        if not all(row['predictions'].get(s, {}).get('files')
                   for s in PREDICTIONS):
            raise ValueError('Complete raw prediction artifacts are missing')
        if row['work']['restored_tensors'] <= 0 or row['work'][
                'reload_checks'] < 1:
            raise ValueError('Restored checkpoint state proof is missing')
    comparisons = []
    references = {(v['task'], v['arm']): v for v in reference['cases']}
    if set(references) != {(r['task'], r['arm'])
                           for r in rows} or len(reference['cases']) != 16:
        raise ValueError('Reference needs all sixteen original cases')
    for row in rows:
        previous = references[(row['task'], row['arm'])]
        if (previous['input_signature'] != row['input_signature']
                or previous['seed'] != 0 or previous['metric'] != row['metric']
                or not previous['fit_id']
                or len(previous['checkpoint_sha256']) != 64
                or any(previous[k] != admitted[(row['task'], row['arm'])][k]
                       for k in ('resolved_plan_sha256', 'selected_plan_sha256',
                                 'trainer_config_sha256'))):
            raise ValueError(
                'Reference and candidate row/plan/input signatures differ')
        comparisons.append(
            dict(
                task=row['task'], arm=row['arm'], score=row['score'],
                reference_score=previous['score'],
                baseline=previous['baseline'],
                difference=None if row['score'] is None
                or previous['score'] is None else row['score'] -
                previous['score'], candidate_fit_id=row['fit_id'],
                reference_fit_id=previous['fit_id'],
                checkpoint_sha256=row['checkpoint_sha256'],
                reference_checkpoint_sha256=previous['checkpoint_sha256'],
                seconds=row['seconds'], differences=row['differences']))
    behavior = check_behavior(args.results_dir)
    save(
        args.output_dir / 'comparison.json',
        dict(status='MEASURED', quality_policy='report_only',
             input_manifest_sha256=args.input_sha256,
             reference_sha256=args.reference_sha256, primary_physical_fits=16,
             extra_physical_fits=behavior['physical_fits'],
             coverage=dict(behavior['coverage'], I1=True,
                           B6=True), behavior=behavior, cases=comparisons))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'compare'):
        p = commands.add_parser(name)
        p.add_argument('--input-root', type=Path, required=True)
        p.add_argument('--input-sha256', required=True)
        p.add_argument('--output-dir', type=Path, required=True)
        if name == 'run':
            p.add_argument('--expected-version', required=True)
        else:
            p.add_argument('--results-dir', type=Path, required=True)
            p.add_argument('--reference', type=Path, required=True)
            p.add_argument('--reference-sha256', required=True)
    args = parser.parse_args(argv)
    (run if args.command == 'run' else compare)(args)


if __name__ == '__main__':
    main()
