#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run five CPU sampling operations or reuse one model on matching GPUs."""

import argparse
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import runpy
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

WEEKLY = runpy.run_path(str(Path(__file__).with_name('weekly_check.py')))
read, save, digest, pinned = [
    WEEKLY[name] for name in ('read', 'save', 'digest', 'pinned')
]


def ids_digest(ids):
    return hashlib.sha256(
        json.dumps(sorted(ids), separators=(',', ':')).encode()).hexdigest()


def inventory(root):
    files = WEEKLY['inventory'](root)
    files.pop('auxiliary-result.json', None)
    return files


def no_fits(row):
    if (type(row.get('physical_fits')) is not int or row['physical_fits'] != 0
            or type(row.get('fit_attempts')) is not int
            or row['fit_attempts'] != 0):
        raise ValueError('Ancillary work cannot add or attempt training')


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':'),
                   allow_nan=False).encode()).hexdigest()


def verify_p1_identity(spec):
    worker = spec['worker']
    receipt = read(
        pinned(worker['native_identity']['path'],
               worker['native_identity']['sha256']))
    if canonical(
            receipt) != spec['data_sha256'] or receipt['work'] != spec['work']:
        raise ValueError(
            'P1 data identity is not derived from its sealed native inputs and work'
        )
    profile = read(
        pinned(receipt['profile']['path'], receipt['profile']['sha256']))
    correction = read(
        pinned(receipt['correction']['path'], receipt['correction']['sha256']))
    manifest = read(
        pinned(receipt['dataset_manifest']['path'],
               receipt['dataset_manifest']['sha256']))
    rows = {name: table['rows'] for name, table in manifest['tables'].items()}
    if (manifest.get('graph') != 'commerce' or manifest.get('rung') != '100M'
            or correction.get('graph') != 'commerce'
            or correction.get('rung') != '100M' or
            correction.get('raw_profile_sha256') != receipt['profile']['sha256']
            or correction.get('dataset_manifest_sha256')
            != receipt['dataset_manifest']['sha256']
            or correction.get('admission_audit_sha256')
            != profile['data_manifest_sha256']
            or manifest.get('total_rows') != sum(rows.values())
            or manifest['total_rows'] < 100000000
            or correction.get('total_rows') != manifest['total_rows'] or any(
                type(count) is not int or count < 0
                for count in rows.values())):
        raise ValueError(
            'P1 requires the real complete commerce/100M source manifest')
    index = Path(worker['index']).resolve()
    snapshot = receipt['snapshot']
    if (str(index) != str(Path(snapshot['index_path']).resolve())
            or str(index) != str(Path(profile['index_path']).resolve())):
        raise ValueError('P1 native snapshot belongs to another graph')
    files = {}
    for path in index.rglob('*'):
        if path.is_symlink():
            raise ValueError('P1 native snapshot cannot contain symbolic links')
        if path.is_file():
            files[path.relative_to(index).as_posix()] = dict(
                sha256=digest(path), bytes=path.stat().st_size)
    if files != snapshot['files']:
        raise ValueError(
            'P1 native snapshot files differ from the complete sealed roster')
    stats = read(index / 'ingest_stats.json')['stats']
    metadata = read(index / 'index_metadata.json')
    if (stats.get('node_counts') != rows
            or stats.get('total_nodes') != sum(rows.values()) or {
                node['name']: node['max_id'] + 1
                for node in metadata['node_types']
            } != rows or files['_MANIFEST.json']['sha256']
            != profile['index_manifest_sha256']
            or files['ingest_stats.json']['sha256']
            != profile['ingest_stats_sha256']):
        raise ValueError(
            'P1 native table populations differ from the full source graph')
    for name in ('seedmap', 'seed_population'):
        pinned(receipt[name]['path'], receipt[name]['sha256'])
    if (receipt['seedmap'] != worker['seedmap']
            or profile['seedmap_sha256'] != worker['seedmap']['sha256']
            or str(Path(profile['seedmap']).resolve()) != str(
                Path(worker['seedmap']['path']).resolve())):
        raise ValueError('P1 seed map differs from the retained workload')
    work = spec['work']
    matched = dict(n='seed_evaluations', unique_customers='unique_customers',
                   plan='fanout', batch='batch_size', procs='processes',
                   threads='threads_per_process')
    if any(
            work.get(key) != profile.get(field)
            for key, field in matched.items()):
        raise ValueError(
            'P1 requested work differs from the retained frozen recipe')
    if (work.get('expected_work') != {
            key: profile['expected_work'][key]
            for key in ('batches', 'requested_seeds')
    } or work.get('historical_sampled_work') != {
            key: profile['expected_work'][key]
            for key in ('nodes', 'edges')
    }):
        raise ValueError(
            'P1 completion requirements or historical work context differ')
    if receipt['selection'].get('offset') != work['offset'] or receipt[
            'selection'].get('n') != work['n']:
        raise ValueError('P1 selected seed evaluations differ')
    return receipt


def verify_p1(spec, row):
    no_fits(row)
    operations = row.get('operations', [])
    if row.get('check') != 'P1' or row.get('status') != 'MEASURED' or len(
            operations) != 5:
        raise ValueError(
            'P1 requires exactly five completed sampling operations')
    work = spec['work']
    for index, operation in enumerate(operations):
        measured = operation.get('result', {})
        if (operation.get('operation') != index
                or operation.get('device') != 'cpu'
                or operation.get('cache_policy') != 'no_drop_uncontrolled'
                or operation.get('cache_qualified') is not False
                or operation.get('work') != work
                or operation.get('data_sha256') != spec['data_sha256']
                or measured.get('status') != 'ok'
                or measured.get('cache') != 'warm'
                or measured.get('seeds') != work['n']
                or measured.get('features') is not True or any(
                    measured.get(name) != work[name]
                    for name in ('hops', 'batch', 'procs', 'threads'))
                or 'plan' in work and measured.get('plan') != work['plan']):
            raise ValueError(
                'Sampling work is incomplete or differs from the frozen CPU profile'
            )
        workers = operation.get('workers', [])
        if (len(workers) != work['procs']
                or {item.get('slot')
                    for item in workers} != set(range(work['procs']))
                or len({item.get('pid')
                        for item in workers}) != work['procs']
                or any(
                    item.get('status') != 'PASS' or item.get('allocation', {}).
                    get('cpu_bytes') != 0 or item.get('delay_seconds_per_batch')
                    != 0 or item.get('retained_buffers') != 0
                    for item in workers)):
            raise ValueError(
                'P1 lacks complete native CPU worker receipts or contains a deliberate fault'
            )
        actual_work = {
            key: sum(item['work'][key] for item in workers)
            for key in ('batches', 'requested_seeds', 'nodes', 'edges')
        }
        if (any(actual_work[key] != work['expected_work'][key]
                for key in ('batches', 'requested_seeds'))
                or actual_work != operation.get('sampled_work')
                or actual_work['nodes'] != measured['rows']
                or actual_work['requested_seeds'] != work['n'] or any(
                    type(actual_work[key]) is not int or actual_work[key] <= 0
                    for key in ('nodes', 'edges'))):
            raise ValueError(
                'Actual P1 batches, seed evaluations, nodes, or edges differ')
        for key in ('wall_s', 'nodes_per_seed', 'rows', 'cpu_util',
                    'cores_busy', 'peak_rss_gb_sum'):
            value = measured.get(key)
            if not WEEKLY['finite'](value) or value < 0 or key in (
                    'nodes_per_seed', 'rows', 'peak_rss_gb_sum') and value == 0:
                raise ValueError(
                    'Sampling resource or completed-work measurements are missing'
                )


def ranked_coordinates(spec, values):
    ids, catalog = values['entity_ids'], values['catalog']
    members = set(catalog)
    expected = spec['population']
    if (not ids or len(ids) != len(set(ids))
            or len(ids) != expected['expected_prediction_count']
            or ids_digest(ids) != expected['prediction_ids_sha256'] or
            expected.get('prediction_anchor') != values.get('prediction_anchor')
            or not catalog or len(catalog) != len(members)
            or catalog != sorted(catalog)
            or len(values['ranked_ids']) != len(ids)
            or len(values['ranked_scores']) != len(ids)):
        raise ValueError('Retained prediction population or catalogue differs')
    excluded = values.get('excluded_ids', [[] for _ in ids])
    if len(excluded) != len(ids):
        raise ValueError('Prediction exclusions are incomplete')
    k = int(spec['metric'].split('@')[1])
    coordinates = {}
    for entity, items, scores, blocked in zip(ids, values['ranked_ids'],
                                              values['ranked_scores'],
                                              excluded):
        if (len(items) != len(set(items))
                or len(items) != min(k,
                                     len(catalog) - len(set(blocked)))
                or len(items) != len(scores)
                or not set(items + blocked) <= members
                or set(items) & set(blocked)
                or not all(map(WEEKLY['finite'], scores))):
            raise ValueError(
                'Ranked outputs are incomplete, repeated, or nonfinite')
        for item, score in zip(items, scores):
            coordinates[(entity, item)] = score
    return coordinates


def verify_b9(spec, row):
    no_fits(row)
    if (row.get('check') != 'B9' or row.get('status') != 'MEASURED'
            or row.get('retained_fit_id') != spec['retained_fit_id']
            or row.get('checkpoint_sha256') != spec['checkpoint_sha256']
            or [call.get('workers')
                for call in row.get('calls', [])] != [1, 2]):
        raise ValueError(
            'B9 needs fresh one- and two-GPU prediction from the retained model'
        )
    indexed = []
    for call in row['calls']:
        records, workers = call.get('devices', []), call['workers']
        if (len(records) != workers
                or {record.get('worker_id')
                    for record in records} != set(range(workers))
                or len({record.get('pid')
                        for record in records}) != workers
                or len({record.get('physical_gpu_uuid')
                        for record in records}) != workers):
            raise ValueError('B9 lacks distinct physical GPU workers')
        for record in records:
            worker = record['worker_id']
            if (record.get('status') != 'PASS'
                    or record.get('fit_attempts') != 0
                    or record.get('gpu_name') != spec['gpu_name']
                    or record.get('batch_retries_disabled') is not True
                    or record.get('eval_batch_size')
                    != spec['prediction_batch_size']
                    or record.get('root_device') != f'cuda:{worker}'
                    or record.get('score_device') != record['root_device']
                    or any(
                        type(record.get(key)) is not int or record[key] <= 0
                        for key in ('pid', 'completed_batches',
                                    'score_elements'))):
                raise ValueError('B9 lacks actual work on both matching GPUs')
            uuid.UUID(record['physical_gpu_uuid'].removeprefix('GPU-'))
            receipts = record.get('reload_identity_checks')
            if not receipts or any(
                    receipt.get('status') != 'PASS' or receipt.get(
                        'checkpoint_sha256') != spec['checkpoint_sha256']
                    or receipt.get('restore_path') !=
                    'lightning_strategy.load_model_state_dict'
                    or type(receipt.get('tensor_count')) is not int
                    or receipt['tensor_count'] <= 0 for receipt in receipts):
                raise ValueError(
                    'Both workers must load the exact retained checkpoint')
        indexed.append(ranked_coordinates(spec, call['predictions']))
    before, after = indexed
    if row['calls'][0]['predictions']['catalog'] != row['calls'][1][
            'predictions']['catalog']:
        raise ValueError('B9 prediction catalogues differ')
    common = before.keys() & after.keys()
    return dict(
        status='MEASURED', qualified=False, gpu_tolerance=None,
        common_recommendations=len(common),
        removed_recommendations=len(before.keys() - after.keys()),
        added_recommendations=len(after.keys() - before.keys()),
        changed_entities=len({
            entity
            for entity, _ in before.keys() ^ after.keys()
        }), max_abs_difference=max(
            (abs(before[key] - after[key]) for key in common), default=None))


def verify_b9_freshness(before, after):
    fields = ('lease', 'seal_sha256', 'full_roster_sha256', 'payload_files',
              'payload_bytes', 'frozen_native_inputs', 'catalog_files_sha256')
    for proof in (before, after):
        frozen = proof.get('frozen_native_inputs', {})
        if (proof.get('status') != 'UNCHANGED_FULL_HASHED_PAYLOAD'
                or not isinstance(proof.get('lease'), dict)
                or not proof['lease']
                or any(not re.fullmatch('[0-9a-f]{64}', str(proof.get(key)))
                       for key in ('seal_sha256', 'full_roster_sha256')) or any(
                           type(proof.get(key)) is not int or proof[key] <= 0
                           for key in ('payload_files', 'payload_bytes'))
                or frozen.get('status') != 'FULL_NATIVE_INPUT_HASHES_VERIFIED'
                or not frozen.get('training_files')
                or not frozen.get('prediction_files')
                or not frozen.get('prediction_population_sha256')
                or not proof.get('catalog_files_sha256')):
            raise ValueError(
                'B9 full native payload freshness evidence is incomplete')
    if any(before[key] != after[key] for key in fields):
        raise ValueError(
            'B9 native payload or host lease changed across prediction')


def load_evidence(root, spec, manifest_sha256):
    row = read(root / 'auxiliary-result.json')
    if row.get('schema') != 1 or row.get(
            'manifest_sha256') != manifest_sha256 or inventory(root) != row.get(
                'files'):
        raise ValueError('Ancillary identity or raw evidence files differ')
    source = pinned(root / 'input-manifest.json', manifest_sha256)
    if read(source) != spec:
        raise ValueError(
            'The ancillary manifest differs from its original frozen input')
    if spec['check'] == 'B9' and not read(root / 'memory.json').get(
            'overall', {}).get('samples'):
        raise ValueError('B9 resource samples are missing')
    if spec['check'] == 'B9':
        verify_b9_freshness(read(root / 'payload-precheck.json'),
                            read(root / 'payload-postcheck.json'))
    (verify_p1 if spec['check'] == 'P1' else verify_b9)(spec, row)
    return row


def reject_fit(path, *args, **kwargs):
    row = read(path) if path.exists() else dict(fit_attempts=0)
    row['fit_attempts'] += 1
    save(path, row)
    raise RuntimeError('Training is forbidden during retained ancillary work')


@contextmanager
def forbid_training(directory):
    from kumo_core.automl.auto_trainer import AutoTrainer
    from kumo_core.train.trainer import Trainer

    import kumo.engine as engine

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{os.getpid()}.json'
    if not path.exists():
        save(path, dict(fit_attempts=0))

    def reject(*args, **kwargs):
        return reject_fit(path, *args, **kwargs)

    with ExitStack() as stack:
        for owner, name in ((engine.Trainer, 'fit'), (Trainer, 'fit'),
                            (AutoTrainer, 'run')):
            stack.enter_context(patch.object(owner, name, reject))
        yield


def load_module(name, record):
    path = pinned(record['path'], record['sha256']).resolve()
    current = sys.modules.get(name)
    if current is not None and (not getattr(current, '__file__', None)
                                or Path(current.__file__).resolve() != path):
        raise ValueError(
            'A loaded auxiliary helper differs from its frozen source')
    if current is None:
        definition = importlib.util.spec_from_file_location(name, path)
        current = importlib.util.module_from_spec(definition)
        sys.modules[name] = current
        definition.loader.exec_module(current)
    if digest(current.__file__) != record['sha256']:
        raise ValueError('A loaded auxiliary helper fingerprint differs')
    return current


def b9_prediction_reader(spec):
    record = spec.get('worker', {}).get('prediction_reader')
    if (not isinstance(record, dict) or set(record) != {'path', 'sha256'}
            or spec.get('inputs', {}).get(record['path']) != record['sha256']):
        raise ValueError('Freeze the separate B9 prediction reader source')
    return load_module('weekly_prediction_reader', record)


def load_spec(path, expected):
    spec = read(pinned(path, expected))
    if (spec.get('schema') != 1 or spec.get('check') not in ('P1', 'B9')
            or spec.get('mode') not in ('reference', 'candidate')
            or spec.get('mode') == 'reference'
            and spec.get('wheel', {}).get('version') != '1.0.0'
            or not isinstance(spec.get('inputs'), dict) or not spec['inputs']
            or spec.get('source_sha256') != digest(__file__)):
        raise ValueError('Use a complete pinned P1 or B9 auxiliary manifest')
    for source, sha256 in spec['inputs'].items():
        pinned(source, sha256)
    for name in ('wheel', 'dependency_profile', 'capacity'):
        pinned(spec[name]['path'], spec[name]['sha256'])
    if spec['check'] == 'P1':
        work = spec['work']
        if (work.get('cache') != 'no_drop_uncontrolled'
                or work.get('features') is not True or any(
                    type(work.get(key)) is not int or work[key] <= 0
                    for key in ('n', 'hops', 'batch', 'procs', 'threads'))
                or type(work.get('offset')) is not int or work['offset'] < 0
                or not isinstance(work.get('plan'), list)
                or len(work['plan']) != work['hops']):
            raise ValueError(
                'Freeze full P1 work, CPU threads, seed map, and sampler plan')
        if spec.get('dataset') != 'commerce/100M':
            raise ValueError('P1 requires the full commerce/100M snapshot')
        worker = spec['worker']
        verify_p1_identity(spec)
        pinned(worker['probe']['path'], worker['probe']['sha256'])
        packet = Path(worker['probe']['path']).resolve().parent
        for name, record in (('dg_bench.py', worker['benchmark']),
                             ('common.py', worker['common'])):
            if digest(packet / name) != record['sha256']:
                raise ValueError(
                    'The sampling subprocess would use different source helpers'
                )
        measurement_helper = packet / 'wheel_measurements.py'
        if str(measurement_helper) not in spec['inputs']:
            raise ValueError(
                'Freeze the sampling subprocess measurement helper')
        pinned(measurement_helper, spec['inputs'][str(measurement_helper)])
        index = Path(worker['index']).resolve()
        if Path(worker['index_metadata']
                ['path']).resolve() != index / 'index_metadata.json':
            raise ValueError('P1 index metadata belongs to another snapshot')
        for source in index.rglob('*'):
            if source.is_symlink():
                raise ValueError(
                    'The frozen sampling index cannot use symbolic links')
            if source.is_file() and spec['inputs'].get(
                    str(source)) != digest(source):
                raise ValueError('A native P1 snapshot file is not frozen')
        pinned(worker['benchmark']['path'], worker['benchmark']['sha256'])
        pinned(worker['common']['path'], worker['common']['sha256'])
        pinned(worker['seedmap']['path'], worker['seedmap']['sha256'])
        pinned(worker['index_metadata']['path'],
               worker['index_metadata']['sha256'])
    else:
        if (spec.get('metric') != 'recall@10'
                or type(spec.get('prediction_batch_size')) is not int
                or spec['prediction_batch_size'] <= 0):
            raise ValueError(
                'The initial B9 route reuses the fixed P2 ranking workload')
        pinned(spec['retained_case']['path'], spec['retained_case']['sha256'])
        pinned(spec['worker']['b9_helper']['path'],
               spec['worker']['b9_helper']['sha256'])
        pinned(spec['worker']['model_helper']['path'],
               spec['worker']['model_helper']['sha256'])
        b9_prediction_reader(spec)
    return spec


def cpu_hardware():
    instances = [
        path.read_text().strip()
        for path in (Path('/sys/devices/virtual/dmi/id/board_asset_tag'),
                     Path('/var/lib/cloud/data/instance-id'))
        if path.is_file() and os.access(path, os.R_OK)
    ]
    drivers = subprocess.check_output([
        'nvidia-smi', '--query-gpu=driver_version',
        '--format=csv,noheader,nounits'
    ], text=True).splitlines()
    if len(set(drivers)) != 1:
        raise ValueError('The auxiliary host has mismatched GPU drivers')
    return dict(
        driver=drivers[0].strip(), instance_id=next(
            (value for value in instances if value.startswith('i-')), None),
        vcpus=os.cpu_count(), memory_total_bytes=next(
            int(line.split()[1]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemTotal:')))


def verify_b9_capacity(spec, capacity, primary):
    if (primary.get('workload') != 'P2'
            or capacity.get('identity') != primary.get('identity')
            or not capacity.get('identity')
            or capacity.get('population') != primary.get('population')
            or capacity.get('population') != spec['population'] or
            capacity.get('hardware_profile') != primary.get('hardware_profile')
            or not capacity.get('hardware_profile')
            or capacity.get('prediction_batch_size')
            != spec['prediction_batch_size']):
        raise ValueError(
            'B9 capacity differs from the full retained P2 identity, population, or prediction work'
        )


def verify_environment(spec, output):
    if (not sys.flags.isolated or sys.prefix == sys.base_prefix
            or sys.platform != 'linux' or sys.version_info[:2] != (3, 12)
            or platform.machine() != 'x86_64' or Path(sys.prefix).resolve()
            != Path(spec['expected_prefix']).resolve()):
        raise ValueError(
            'Use the declared isolated Linux Python 3.12 environment')
    helpers = runpy.run_path(str(Path(__file__).with_name('pr_check.py')))
    wheel = Path(spec['wheel']['path'])
    if (helpers['wheel_metadata'](wheel) != spec['wheel']['version']
            or importlib.metadata.version('kumo-engine')
            != spec['wheel']['version']):
        raise ValueError('The auxiliary installed wheel version differs')
    distribution = importlib.metadata.distribution('kumo-engine')
    helpers['verify_installed_files'](wheel, Path(distribution.locate_file('')))
    packages = {
        name: item.version
        for item in importlib.metadata.distributions()
        if (name := re.sub('[-_.]+', '-', item.metadata['Name']).lower()
            ) != 'kumo-engine'
    }
    if packages != read(spec['dependency_profile']['path']):
        raise ValueError(
            'Auxiliary dependencies differ from the frozen profile')
    with (output / 'pip-check.log').open('w') as log:
        subprocess.run(
            [sys.executable, '-I', '-m', 'pip', '--isolated', 'check'],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    capacity = read(spec['capacity']['path'])
    actual = cpu_hardware()
    if (capacity.get('status') != 'ADMITTED'
            or capacity.get('check') != spec['check']
            or capacity.get('native_capacity_checked') is not True
            or any(actual[key] != capacity.get('observed_hardware', {}).get(key)
                   for key in actual)):
        raise ValueError(
            'Auxiliary native admission belongs to different host capacity')
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'
    ], text=True)
    if any(
            int(pid.strip()) != os.getpid() for pid in processes.splitlines()
            if pid.strip()):
        raise ValueError('The auxiliary GPU host has another active GPU owner')
    import torch
    if spec['check'] == 'P1':
        if torch.cuda.device_count() != 0 or os.environ.get(
                'KUMO_ACCELERATOR') != 'cpu':
            raise ValueError('P1 must expose CPU sampling only')
    else:
        primary = read(spec['retained_case']['path'])['case_spec']
        verify_b9_capacity(spec, capacity, primary)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise ValueError('B9 requires exactly two visible matching GPUs')
        actual['gpus'] = [
            dict(
                gpu_uuid=str(torch.cuda.get_device_properties(index).uuid),
                name=torch.cuda.get_device_name(index),
                memory_bytes=torch.cuda.get_device_properties(
                    index).total_memory) for index in range(2)
        ]
        expected = capacity.get('gpus', [])
        if len(expected) != 2:
            raise ValueError('B9 capacity must bind both physical GPUs')
        for observed, frozen in zip(actual['gpus'], expected):
            if (uuid.UUID(observed['gpu_uuid'].removeprefix('GPU-'))
                    != uuid.UUID(frozen['gpu_uuid'].removeprefix('GPU-'))
                    or observed['name'] != frozen['name']
                    or observed['name'] != spec['gpu_name']
                    or observed['memory_bytes'] != frozen['memory_bytes']):
                raise ValueError(
                    'B9 physical GPUs differ from native admission')
        if len({item['gpu_uuid'] for item in actual['gpus']}) != 2:
            raise ValueError('B9 cannot alias the same physical GPU twice')
    if spec['check'] == 'P1' and (
            capacity.get('data_sha256') != spec['data_sha256']
            or capacity.get('work_sha256') != canonical(spec['work'])
            or capacity.get('seedmap_sha256')
            != spec['worker']['seedmap']['sha256']):
        raise ValueError(
            'P1 capacity is not bound to the sealed graph, seed map, and work')
    return actual


def execute_p1(spec, output):
    worker, work = spec['worker'], spec['work']
    receipt = verify_p1_identity(spec)
    import pyarrow as pa
    import pyarrow.parquet as pq
    seedmap = pq.read_table(worker['seedmap']['path'])
    if (len(seedmap) != work['unique_customers'] or
            not {'customer_id', 'anchor_s', 'gid'} <= set(seedmap.schema.names)
            or any(seedmap[name].null_count
                   for name in ('customer_id', 'anchor_s', 'gid'))
            or len(set(seedmap['customer_id'].to_pylist())) != len(seedmap)):
        raise ValueError(
            'The frozen P1 seed map differs from the unique source population')
    source_seeds = pq.read_table(receipt['seed_population']['path'],
                                 columns=['customer_id',
                                          'anchor']).slice(0, len(seedmap))
    anchors = source_seeds['anchor'].cast(pa.timestamp('s')).cast(
        pa.int64()).to_pylist()
    if (source_seeds['customer_id'].to_pylist()
            != seedmap['customer_id'].to_pylist()
            or anchors != seedmap['anchor_s'].to_pylist()):
        raise ValueError(
            'P1 customer IDs or anchors differ from the frozen published seed population'
        )
    source = Path(worker['benchmark']['path'])
    sys.path.insert(0, str(source.parent))
    load_module('common', worker['common'])
    benchmark = load_module('dg_bench', worker['benchmark'])
    if benchmark.C.plan(work['hops']) != work['plan']:
        raise ValueError(
            'The effective sampler plan differs from the frozen P1 plan')
    engine = benchmark.engine(worker['index'], work['threads'])
    mapped = benchmark.fetch(engine, 'customers', seedmap['gid'].to_numpy(),
                             ['customer_id'])
    if mapped is None or mapped['customer_id'].to_pylist(
    ) != seedmap['customer_id'].to_pylist():
        raise ValueError(
            'Native P1 global IDs do not resolve to the frozen customer IDs')
    del engine
    selected = benchmark.C.tile(seedmap, work['offset'] + work['n']).slice(
        work['offset'], work['n'])
    selection = [[row['customer_id'], row['anchor_s'], row['gid']]
                 for row in selected.to_pylist()]
    if canonical(selection) != receipt['selection']['sha256']:
        raise ValueError(
            'P1 declared repeated seed evaluations differ from the frozen ordered selection'
        )
    save(
        output / 'input-verification.json',
        dict(status='PASS', physical_fits=0, data_sha256=spec['data_sha256'],
             matched_mapping_rows=len(seedmap), seed_evaluations=len(selected),
             selection_sha256=canonical(selection),
             cache_policy='no_drop_uncontrolled', cache_qualified=False))
    operations = []
    for index in range(5):
        directory = output / f'operation-{index}'
        directory.mkdir()
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                           KUMO_ACCELERATOR='cpu',
                           KUMO_SAMPLING_WORK_DIR=str(directory / 'workers'),
                           KUMO_SAMPLING_DELAY_SECONDS='0',
                           KUMO_SAMPLING_CPU_BYTES_PER_WORKER='0')
        command = [
            sys.executable, '-I', worker['probe']['path'], 'rate', '--index',
            worker['index'], '--seedmap', worker['seedmap']['path'], '--n',
            str(work['n']), '--offset',
            str(work['offset']), '--hops',
            str(work['hops']), '--batch',
            str(work['batch']), '--procs',
            str(work['procs']), '--threads',
            str(work['threads']), '--cap-s',
            str(spec['operation_timeout_seconds']), '--results',
            str(directory / 'result.jsonl'), '--label', f'P1-{index}'
        ]
        with (directory / 'run.log').open('w') as log:
            subprocess.run(command, env=environment,
                           cwd=Path(worker['probe']['path']).parent, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        raw = [
            json.loads(line)
            for line in (directory / 'result.jsonl').read_text().splitlines()
        ]
        if len(raw) != 1:
            raise ValueError('Sampling produced missing or extra operations')
        workers = [
            read(path)
            for path in sorted((directory / 'workers').glob('worker-*.json'))
        ]
        actual = {
            key: sum(item['work'][key] for item in workers)
            for key in ('batches', 'requested_seeds', 'nodes', 'edges')
        }
        operations.append(
            dict(
                operation=index, device='cpu', work=work,
                cache_policy='no_drop_uncontrolled', cache_qualified=False,
                benchmark_cache_label=raw[0].get('cache'),
                data_sha256=spec['data_sha256'], result=raw[0], workers=workers,
                sampled_work=actual, historical_work_difference={
                    key: actual[key] - work['historical_sampled_work'][key]
                    for key in ('nodes', 'edges')
                }))
        save(output / 'sampling-operations.json', operations)
    performance_reason = 'No compatible approved-version cache-qualified performance baseline'
    row = dict(
        check='P1', status='MEASURED', physical_fits=0, fit_attempts=0,
        operations=operations,
        historical_sampled_work=work['historical_sampled_work'],
        historical_profile_sha256=receipt['profile']['sha256'],
        historical_reference=read(receipt['profile']['path']).get(
            'environments',
            {}).get('reference'), cache_policy='no_drop_uncontrolled',
        cache_qualified=False, performance_comparison_valid=False,
        performance_reason=performance_reason)
    verify_p1(spec, row)
    return row


def observed_b9_worker(*, worker_id, config, **kwargs):
    """Observe each real scorer and reject training inside spawned workers."""
    spec = load_spec(Path(os.environ['KUMO_WEEKLY_AUX_MANIFEST']),
                     os.environ['KUMO_WEEKLY_AUX_MANIFEST_SHA256'])
    helper = load_module('b9_helpers', spec['worker']['b9_helper'])
    primary = read(spec['retained_case']['path'])['case_spec']
    helpers = Path(primary['worker']['helper_root'])
    sys.path.insert(0, str(helpers))
    gpu = importlib.import_module('wheel_gpu_learning')
    if digest(gpu.__file__) != primary['inputs'].get(
            str(helpers / 'wheel_gpu_learning.py')):
        raise ValueError('The checkpoint observer source differs')
    import torch
    from kumo_core.automl.auto_trainer import AutoTrainer
    from kumo_core.train.trainer import Trainer
    from pytorch_lightning import Callback
    record = dict(status='RUNNING', worker_id=worker_id, pid=os.getpid(),
                  run_id=config.run_id, attempt_id=config.attempt_id,
                  tensor_kind='ranking_scores', completed_batches=0,
                  score_elements=0, reload_identity_checks=[])
    root = Path(os.environ['KUMO_WEEKLY_AUX_CONTEXT_DIR'])
    destination = root / f'{config.attempt_id}-worker-{worker_id}.json'
    original_create = Trainer.create_pl_trainer

    class CompletedRankingScores(Callback):
        def on_predict_batch_end(self, trainer, module, outputs, batch,
                                 batch_idx, dataloader_idx=0):
            scores = outputs.get('score') if isinstance(outputs, dict) else None
            device = trainer.strategy.root_device
            if (not isinstance(scores, torch.Tensor) or not scores.is_cuda
                    or not scores.numel() or scores.device != device
                    or str(device) != f'cuda:{worker_id}'
                    or not torch.isfinite(scores).all()):
                raise ValueError(
                    'Prediction did not complete finite ranking scores on its assigned GPU'
                )
            torch.cuda.synchronize(device)
            properties = torch.cuda.get_device_properties(device)
            record.update(
                root_device=str(device), score_device=str(scores.device),
                physical_gpu_uuid='GPU-' +
                str(properties.uuid).removeprefix('GPU-'),
                gpu_name=properties.name)
            record['completed_batches'] += 1
            record['score_elements'] += scores.numel()

    with ExitStack() as stack:
        stack.enter_context(
            forbid_training(Path(os.environ['KUMO_WEEKLY_AUX_FIT_LEDGER'])))
        stack.enter_context(gpu.without_batch_retries(Trainer, AutoTrainer))
        record['batch_retries_disabled'] = True

        def create(trainer):
            observed_batch = trainer.datamodule.eval_loader_kwargs['batch_size']
            if observed_batch != spec['prediction_batch_size']:
                raise ValueError(
                    'B9 prediction batch size differs from its frozen profile')
            record['eval_batch_size'] = observed_batch
            value = original_create(trainer)
            trainer.pl_trainer.callbacks.append(CompletedRankingScores())
            strategy = trainer.pl_trainer.strategy
            restore = strategy.load_model_state_dict

            def load_weights(*args, **kwargs):
                restored = restore(*args, **kwargs)
                receipt = gpu.verify_loaded_checkpoint(
                    strategy.lightning_module, trainer.pl_trainer.ckpt_path,
                    spec['checkpoint_sha256'])
                receipt[
                    'restore_path'] = 'lightning_strategy.load_model_state_dict'
                record['reload_identity_checks'].append(receipt)
                return restored

            stack.enter_context(
                patch.object(strategy, 'load_model_state_dict', load_weights))
            return value

        stack.enter_context(patch.object(Trainer, 'create_pl_trainer', create))
        try:
            value = helper._real_score_worker(worker_id=worker_id,
                                              config=config, **kwargs)
            if not record['reload_identity_checks'] or not record[
                    'completed_batches']:
                raise ValueError(
                    'The prediction worker did not prove loaded model state')
            record.update(status='PASS', fit_attempts=0)
            return value
        except BaseException as error:
            record.update(status='FAIL',
                          error=f'{type(error).__name__}: {error}')
            raise
        finally:
            save(destination, record)


def load_saved_model(trainer_type, bundle):
    # Released 1.0.0 reads its accelerator from KUMO_ACCELERATOR.
    return trainer_type.load(bundle)


def b9_payload_precheck(prepared, output):
    proof = prepared.get('payload_freshness', {})
    verify_b9_freshness(proof, proof)
    save(output / 'payload-precheck.json', proof)
    return proof


def b9_payload_postcheck(native, config, precheck, output):
    try:
        previous = dict(precheck['lease'])
        proof = native.verify_inputs(config, previous_lease=previous)
        verify_b9_freshness(precheck, proof)
        save(output / 'payload-postcheck.json', proof)
        return proof
    except BaseException as error:
        save(output / 'payload-postcheck.json',
             dict(status='FAIL', error=f'{type(error).__name__}: {error}'))
        raise


def execute_b9(spec, output):
    import pyarrow.parquet as pq
    import torch
    from kumo_core import prediction
    from kumoapi.jobs import MetadataField

    import kumo.engine as engine

    path = Path(spec['retained_case']['path'])
    if output.resolve().is_relative_to(path.parent.resolve()):
        raise ValueError(
            'Ancillary output must preserve the immutable primary case directory'
        )
    retained = read(path)
    primary = retained['case_spec']
    original = read(path.parent / 'input-manifest.json')
    WEEKLY['load_case'](path, primary, spec['mode'], original)
    if (primary['workload'] != 'P2'
            or retained['result']['fit_id'] != spec['retained_fit_id']
            or spec['population'] != primary['population']
            or retained['wheel_sha256'] != spec['wheel']['sha256']
            or retained['version'] != spec['wheel']['version']
            or retained['device']['gpu'] != spec['gpu_name']):
        raise ValueError(
            'The retained model differs from the approved auxiliary workload')
    WEEKLY['verify_case_inputs'](primary)
    worker = primary['worker']
    reader = b9_prediction_reader(spec)
    helper = load_module('weekly_model_helper', spec['worker']['model_helper'])
    if digest(helper.__file__) != worker['sha256']:
        raise ValueError('The retained case used a different model helper')
    native = load_module('weekly_prepared_data', worker['data_module'])
    config = read(pinned(worker['config']['path'], worker['config']['sha256']))
    helper_root = Path(worker['helper_root'])
    sys.path.insert(0, str(helper_root))
    learning = importlib.import_module('wheel_scale_learning')
    meters = importlib.import_module('wheel_measurements')
    for module in (learning, meters):
        if digest(module.__file__) != primary['inputs'].get(
                str(helper_root / Path(module.__file__).name)):
            raise ValueError(
                'A retained prediction helper differs from its frozen source')
    [saved_files] = [
        path.parent / name for name in retained['files']
        if Path(name).name == 'saved-model-files.json'
    ]
    if read(saved_files) != spec['model_files']:
        raise ValueError(
            'The auxiliary model files are not the retained primary model')
    bundle = Path(spec['model_bundle'])
    before = {
        name: sha256
        for name, sha256 in WEEKLY['inventory'](bundle).items()
    }
    if before != spec['model_files'] or digest(
            bundle / 'model/checkpoint') != spec['checkpoint_sha256']:
        raise ValueError('The saved model bytes differ from the retained fit')
    if spec['checkpoint_sha256'] != retained['result'][
            'expected_reload_checkpoint_sha256']:
        raise ValueError('B9 selected another checkpoint')
    sys.path.insert(0, str(Path(__file__).parent))
    canonical = importlib.import_module('weekly_aux')
    if digest(canonical.__file__) != digest(__file__):
        raise ValueError(
            'The spawned worker would use different observer source')
    load_module('b9_helpers', spec['worker']['b9_helper'])
    engine.configure(scratch_dir=output / 'scratch')
    measurements = meters.Measurements(output)
    calls = []
    try:
        measurements.start_monitor(worker['monitor_helper']['path'],
                                   output / 'scratch')
        with measurements.stage('verify_retained_inputs', gpu=False):
            prepared = native.load_prepared(config, primary['prepared_dir'],
                                            condition='normal')
            payload_precheck = b9_payload_precheck(prepared, output)
            truth = dict(prepared['truth'], metric=primary['metric'])
            helper.verify_population(truth, spec['population'])
            ids = truth['entity_ids']
            full_inputs = helper.prediction_inputs(prepared, ids)
            input_dir = output / 'prediction-inputs'
            input_dir.mkdir()
            pq.write_table(full_inputs, input_dir / 'full.parquet')
            snapshot = prepared['snapshot']
            query = prepared.get('prediction_query', prepared['query'])
            model = load_saved_model(engine.Trainer, bundle)
            table = engine.PredictionTable.from_source_table(
                engine.FileConnector(root_dir=str(input_dir))['full.parquet'],
                pq=query, snapshot=snapshot,
                plan=model.training_table_generation_plan)
            staged = pq.read_table(table.table_path)
            helper.prediction_inputs(dict(prepared, prediction_rows=staged),
                                     ids)
            limit = helper.prediction_limit(truth, primary['metric'],
                                            worker['prediction_policy'])
        for count, label in ((1, 'one-gpu'), (2, 'two-gpu')):
            destination = output / label
            destination.mkdir()
            for folder in ('worker-proof', 'prediction-outputs'):
                (destination / folder).mkdir()
            environment = dict(KUMO_WEEKLY_AUX_CONTEXT_DIR=str(destination /
                                                               'worker-proof'))
            torch.manual_seed(0)
            with (patch.dict(os.environ, environment),
                  patch.object(prediction, '_score_prediction_shard',
                               canonical.observed_b9_worker),
                  measurements.stage(label)):
                predictions = model.predict(
                    table, snapshot=snapshot, num_workers=count,
                    num_classes_to_return=limit,
                    output_config=engine.PredictionOutputConfig(
                        connector=engine.FileConnector(
                            root_dir=str(destination / 'prediction-outputs')),
                        output_paths={'predictions': 'predictions'},
                        extra_fields=[MetadataField.ANCHOR_TIMESTAMP]))
                batches = (learning.normalize_prediction_time(
                    'weekly', batch, truth,
                    retained['result']['entity_time_column'],
                    require_anchor=True) for batch in reader.ordered_batches(
                        predictions.files, destination / 'sort'))
                values = learning.ranked_from_batches(batches, truth, limit)
            devices = [
                read(path) for path in sorted((destination /
                                               'worker-proof').glob('*.json'))
            ]
            if len(devices) != count:
                raise ValueError(
                    'B9 has incomplete actual GPU and loaded-state worker receipts'
                )
            coordinates = dict(
                entity_ids=ids, catalog=truth['catalog'],
                ranked_ids=values['ranked_ids'],
                ranked_scores=values['ranked_scores'],
                prediction_anchor=spec['population']['prediction_anchor'])
            if 'excluded_ids' in truth:
                coordinates['excluded_ids'] = truth['excluded_ids']
            calls.append(
                dict(workers=count, predictions=coordinates, devices=devices))
            save(destination / 'call.json', calls[-1])
        if before != WEEKLY['inventory'](bundle):
            raise ValueError(
                'Retained model bytes changed during B9 prediction')
        WEEKLY['verify_case_inputs'](primary)
        b9_payload_postcheck(native, config, payload_precheck, output)
        row = dict(check='B9', status='MEASURED', physical_fits=0,
                   fit_attempts=0, retained_fit_id=spec['retained_fit_id'],
                   checkpoint_sha256=spec['checkpoint_sha256'], calls=calls)
        row['comparison'] = verify_b9(spec, row)
        save(output / 'model-files.json', before)
        return row
    finally:
        measurements.finish_monitor()


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    row = dict(schema=1, status='RUNNING', manifest_sha256=args.manifest_sha256,
               physical_fits=0, fit_attempts=0)
    save(output / 'auxiliary-result.json', row)
    lock = None
    started = time.monotonic()
    try:
        spec = load_spec(args.manifest, args.manifest_sha256)
        row['check'] = spec['check']
        shutil.copyfile(args.manifest, output / 'input-manifest.json')
        pinned(output / 'input-manifest.json', args.manifest_sha256)
        os.environ.update(
            CUDA_VISIBLE_DEVICES='' if spec['check'] == 'P1' else '0,1',
            KUMO_ACCELERATOR='cpu' if spec['check'] == 'P1' else 'gpu',
            KUMO_WEEKLY_AUX_MANIFEST=str(args.manifest.resolve()),
            KUMO_WEEKLY_AUX_MANIFEST_SHA256=args.manifest_sha256,
            KUMO_WEEKLY_AUX_FIT_LEDGER=str(output / 'fit-attempts'))
        lock = Path(spec['host_lock']).open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        row['observed_hardware'] = verify_environment(spec, output)
        with forbid_training(output / 'fit-attempts'):
            row.update(
                (execute_p1 if spec['check'] == 'P1' else execute_b9)(spec,
                                                                      output))
        if spec['check'] == 'B9':
            WEEKLY['materialize_prediction_links'](output)
        load_spec(args.manifest, args.manifest_sha256)
        row['fit_attempts'] = sum(
            read(path)['fit_attempts']
            for path in (output / 'fit-attempts').glob('*.json'))
        (verify_p1 if spec['check'] == 'P1' else verify_b9)(spec, row)
    except BaseException as error:
        row.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                   traceback=traceback.format_exc())
        raise
    finally:
        if lock is not None:
            lock.close()
        row['seconds'] = time.monotonic() - started
        if (output / 'fit-attempts').exists():
            row['fit_attempts'] = sum(
                read(path)['fit_attempts']
                for path in (output / 'fit-attempts').glob('*.json'))
        try:
            row['files'] = inventory(output)
        except BaseException as error:
            row.update(status='FAIL',
                       inventory_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            save(output / 'auxiliary-result.json', row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
