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
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import wraps
from pathlib import Path, PurePosixPath
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
REFERENCE_WHEEL = '1bc15ecb7696bce216b05aa883d6897d2b8ce4342c6c2f9d62b7cc28b314539e'


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


@contextmanager
def measure_operation(report, name):
    started = time.monotonic()
    try:
        yield
    finally:
        report.setdefault('timings', {})[name] = time.monotonic() - started


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
    safe_file(root, spec['reference_dependencies'], spec['files'])
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
    if not profile or not all(
            packages.get(normalize(k)) == v for k, v in profile.items()):
        raise ValueError(
            'Installed dependencies differ from the selected profile')


def load_reference(args):
    reference = read(args.reference)
    if (digest(args.reference) != args.reference_sha256
            or reference.get('input_manifest_sha256') != args.input_sha256
            or reference.get('format') != 'kumo-engine-nightly-reference-v1'
            or reference.get('version') != '1.0.0'
            or reference.get('wheel_sha256') != REFERENCE_WHEEL):
        raise ValueError('Approved reference or input-manifest binding differs')
    return reference


def reference_cases(reference, manifest):
    references = {(v['task'], v['arm']): v for v in reference['cases']}
    admitted = {(v['task'], v['arm']): v for v in manifest['cases']}
    if (len(reference['cases']) != 16 or set(references) != set(admitted)
            or len({v['fit_id']
                    for v in reference['cases']}) != 16 or len(admitted) != 16):
        raise ValueError('Reference requires sixteen distinct admitted cases')
    for key, previous in references.items():
        case = admitted[key]
        if (type(previous['seed']) is not int or previous['seed'] != 0
                or previous['input_signature'] != input_signature(
                    manifest, case)
                or previous['metric'] != smoke.TASKS[key[0]]['metric']
                or not previous['fit_id'] or
                not re.fullmatch('[0-9a-f]{64}', previous['checkpoint_sha256'])
                or any(
                    type(previous[k]) not in
                    (int, float) or not math.isfinite(previous[k])
                    for k in ('score', 'baseline'))):
            raise ValueError(
                'Reference input, recipe or measurement identity differs')
    return references


def verify_runtime(version, profile, *, reference_only=False):
    import torch

    import kumo.engine as kumo

    normalize = lambda name: name.lower().replace('_', '-').replace('.', '-')
    packages = {
        normalize(d.metadata['Name']): d.version
        for d in importlib.metadata.distributions()
        if normalize(d.metadata['Name']) != 'kumo-engine'
    }
    if reference_only and profile is not None:
        check_dependencies(packages, profile)
    if (sys.platform != 'linux' or platform.machine() != 'x86_64'
            or sys.version_info[:2] != (3, 12)
            or importlib.metadata.version('kumo-engine') != version
            or kumo.__version__ != version
            or not Path(kumo.__file__).resolve().is_relative_to(
                Path(sys.prefix).resolve()) or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1):
        raise ValueError(
            'Use the installed Linux3.12 wheel and one actual CUDA GPU')
    properties = torch.cuda.get_device_properties(0)
    identifier = str(properties.uuid)
    uuid.UUID(identifier.removeprefix('GPU-'))
    if 'RTX PRO 6000' not in properties.name or properties.total_memory <= 0:
        raise ValueError(
            'CUDA work requires the authorized RTX PRO 6000 profile')
    driver = subprocess.check_output([
        'nvidia-smi', '--id=GPU-' + identifier.removeprefix('GPU-'),
        '--query-gpu=driver_version', '--format=csv,noheader'
    ], text=True).strip()
    if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', driver):
        raise ValueError('CUDA driver identity is missing or ambiguous')
    direct = json.loads(
        importlib.metadata.distribution('kumo-engine').read_text(
            'direct_url.json'))
    wheel_hash = direct['archive_info']['hashes']['sha256']
    if len(wheel_hash) != 64:
        raise ValueError('Installed wheel archive identity is missing')
    return dict(
        version=version, dependencies=packages, reproducibility=dict(
            protocol='product-defaults',
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(
            ),
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
        gpu=properties.name, gpu_count=1, driver=driver,
        architecture=platform.machine(), compute_capability=[
            properties.major, properties.minor
        ], gpu_uuid=identifier, gpu_memory_bytes=int(properties.total_memory),
        torch=torch.__version__, cuda=torch.version.cuda,
        threads=torch.get_num_threads(), import_path=kumo.__file__,
        wheel_sha256=wheel_hash, cpu_count=os.cpu_count(),
        system_memory_bytes=os.sysconf('SC_PAGE_SIZE') *
        os.sysconf('SC_PHYS_PAGES'))


def load_table(root, files, case):
    import pyarrow.parquet as pq
    from kumo_core.io.json import from_json_dict
    from kumo_core.model_plan import TrainingTableGenerationPlan

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
    return table, stages, value


def stage_frame(table, entry):
    import pandas as pd
    return pd.concat(
        [pd.read_parquet(table.staging_dir / p) for p in entry['parts']],
        ignore_index=True)


def check_inputs(frame, truth, *, partial=False, static=False):
    required = {'ENTITY1'} if static else {'ENTITY1', 'TIMESTAMP'}
    if static and 'TIMESTAMP' in frame:
        required.add('TIMESTAMP')
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
    ledger = state['fit_ledger']
    completed = [fit for fit in ledger if fit['status'] == 'PASS']
    if (not completed or type(state['fits']) is not int
            or state['fits'] != len(ledger)
            or state['completed_fits'] != len(completed)
            or state['optimizer_steps'] != sum(fit['steps']
                                               for fit in completed)
            or state['training_batches'] != sum(fit['cuda_batches']
                                                for fit in ledger)
            or type(selected_steps) is not int or selected_steps <= 0):
        raise ValueError(
            'Candidate counts, CUDA work or optimizer updates differ')
    for fit in ledger:
        epochs = fit['epochs']
        if (fit['status'] not in ('PASS', 'FAIL')
                or fit['cuda_batches'] != sum(e['batches'] for e in epochs)
                or any(e['batches'] != len(e['batch_sizes']) or any(
                    type(n) is not int or n <= 0 for n in e['batch_sizes'])
                       for e in epochs) or fit['status'] == 'PASS' and
            (type(fit['steps']) is not int or fit['steps'] <= 0
             or fit['cuda_batches'] <= 0 or not fit['validation'])):
            raise ValueError('Candidate training evidence is incomplete')
    if not any(
            fit.get('checkpoint_sha256') == state['checkpoint_sha256']
            and selected_steps <= fit['steps'] for fit in completed):
        raise ValueError(
            'Selected checkpoint has no completed candidate evidence')


def input_signature(manifest, case):
    return dict(descriptor=manifest['files'][case['descriptor']],
                prediction=manifest['files'][case['prediction_input']],
                partial=manifest['files'][case['partial_input']],
                requested_plan=case['requested_plan_sha256'])


def ranked_difference(before, after, *, atol, measure_only=False):
    """Bound scores, inverted pairs, and proven changes at the top-K boundary."""
    if any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in (atol, )):
        raise ValueError('Prediction tolerances must be finite and nonnegative')
    groups = [
        value[key] for value in (before, after)
        for key in ('ranked_ids', 'ranked_scores')
    ]
    if not groups[0] or len({len(group) for group in groups}) != 1:
        raise ValueError('Ranked populations are incomplete')

    def coordinates(names, scores):
        if (not names or len(names) != len(scores)
                or len(set(names)) != len(names) or any(
                    type(v) not in (int, float) or not math.isfinite(v)
                    for v in scores)
                or any(x < y for x, y in zip(scores, scores[1:]))):
            raise ValueError('Ranked coordinates are incomplete or invalid')
        return dict(zip(names, scores, strict=True))

    maximum, maximum_gap, reordered, inversions, added, removed = 0., 0., 0, 0, 0, 0
    unproven = 0
    for index, (left, a, right, b) in enumerate(zip(*groups, strict=True)):
        old, observed = coordinates(left, a), coordinates(right, b)
        if len(left) != len(right):
            raise ValueError('Ranked output cardinality changed')
        changed = left != right
        added += len(observed.keys() - old.keys())
        removed += len(old.keys() - observed.keys())
        if set(left) != set(right):
            if any(
                    len(value.get(key, [])) != len(groups[0])
                    for value in (before, after)
                    for key in ('boundary_ids', 'boundary_scores')):
                raise ValueError(
                    'Membership changes require retained cutoff coordinates')
            union = set(left) | set(right)
            orders, values = [], []
            for value, names, scores in ((before, left, a), (after, right, b)):
                boundary, numbers = value['boundary_ids'][index], value[
                    'boundary_scores'][index]
                mapped = coordinates(boundary, numbers)
                if (boundary[:len(names)] != names
                        or numbers[:len(scores)] != scores
                        or len(boundary) < len(names) + 1):
                    raise ValueError(
                        'Cutoff coordinates do not prove the membership change')
                orders.append([name for name in boundary if name in union])
                values.append(mapped)
            if not all(union <= mapped.keys() for mapped in values):
                if not measure_only:
                    raise ValueError(
                        'Cutoff coordinates do not prove the membership change')
                unproven += 1
                common = union & values[0].keys() & values[1].keys()
                orders = [[name for name in order if name in common]
                          for order in orders]
            left, right = orders
            old, observed = values
        for name in left:
            delta = abs(old[name] - observed[name])
            if delta > atol and not measure_only:
                raise ValueError('Ranked scores exceed the reviewed tolerance')
            maximum = max(maximum, delta)
        if changed:
            reordered += 1
            positions = {name: position for position, name in enumerate(right)}
            for i, first in enumerate(left):
                for second in left[i + 1:]:
                    if positions[first] > positions[second]:
                        maximum_gap = max(maximum_gap, old[first] - old[second],
                                          observed[second] - observed[first])
                        if maximum_gap > atol and not measure_only:
                            raise ValueError(
                                'Rank change exceeds the reviewed score-gap budget'
                            )
                        inversions += 1
    return dict(
        status='PASS' if not unproven and max(maximum, maximum_gap) <= atol else
        'FAIL', policy='measurement' if measure_only else 'enforced',
        reordered_entities=reordered, rank_inversions=inversions,
        added_items=added, removed_items=removed,
        max_abs_aligned_score_difference=maximum,
        max_inverted_score_gap=maximum_gap,
        unproven_membership_changes=unproven, atol=atol, rtol=0.)


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
    boundary_ids, boundary_scores = [None] * len(ids), [None] * len(ids)
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
                          key=lambda p: (-p[1], p[0]))[:2 * wanted]
        if (not wanted or len(eligible) < wanted or raw_limit < len(catalog)
                and min(p[1] for p in pairs) >= eligible[wanted - 1][1]):
            raise ValueError(
                'Capped ranked output cannot certify the eligible cutoff')
        boundary_ids[index], boundary_scores[index] = [
            p[0] for p in eligible
        ], [p[1] for p in eligible]
        items[index], scores[index] = boundary_ids[
            index][:wanted], boundary_scores[index][:wanted]

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
    return dict(ranked_ids=items, ranked_scores=scores, raw_rows=count,
                boundary_ids=boundary_ids, boundary_scores=boundary_scores)


def ordered_batches(files, directory, *, ranked=True):
    """Merge Parquet runs with bounded batches and at most 32 open inputs."""
    import heapq
    from contextlib import ExitStack, closing
    from itertools import islice
    from tempfile import TemporaryDirectory

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory.mkdir(parents=True, exist_ok=True)
    order = [('ENTITY', 'ascending')]
    if ranked:
        order += [('SCORE', 'descending'), ('CLASS', 'ascending')]
    schema, sequence = None, 0
    with TemporaryDirectory(prefix='parquet-sort-', dir=directory) as name:
        work = Path(name)

        def spill(tables):
            nonlocal sequence
            path = work / f'{sequence}.parquet'
            sequence += 1
            with pq.ParquetWriter(path, schema) as writer:
                for table in tables:
                    writer.write_table(table, row_group_size=4096)
            return path

        def rows(path):
            with pq.ParquetFile(path) as reader:
                for batch in reader.iter_batches(batch_size=512,
                                                 use_threads=False):
                    yield from batch.to_pandas(ignore_metadata=True).itertuples(
                        index=False, name=None)

        def merge(paths):
            with ExitStack() as stack:
                combined = heapq.merge(
                    *(stack.enter_context(closing(rows(p))) for p in paths),
                    key=key)
                while chunk := list(islice(combined, 65536)):
                    yield pa.Table.from_pandas(
                        pd.DataFrame.from_records(chunk, columns=schema.names),
                        schema=schema, preserve_index=False)

        runs = []
        for path in files:
            with pq.ParquetFile(path) as reader:
                for batch in reader.iter_batches(batch_size=65536,
                                                 use_threads=False):
                    if schema is None:
                        schema = batch.schema.remove_metadata()
                    if (not schema.equals(batch.schema, check_metadata=False) or
                            not {column
                                 for column, _ in order} <= set(schema.names)):
                        raise ValueError('Prediction columns or schemas differ')
                    for column in ('ENTITY',
                                   'CLASS') if ranked else ('ENTITY', ):
                        values = batch.column(schema.get_field_index(column))
                        if not pa.types.is_integer(
                                values.type) or values.null_count:
                            raise ValueError(
                                f'Prediction changed the integer {column.lower()} key type'
                            )
                    if ranked:
                        scores = batch.column(
                            schema.get_field_index('SCORE')).to_numpy(
                                zero_copy_only=False)
                        if (not pd.api.types.is_numeric_dtype(scores.dtype)
                                or not np.isfinite(scores).all()):
                            raise ValueError(
                                'Prediction scores must be finite numbers')
                    runs.append(
                        spill([
                            pa.Table.from_batches([batch],
                                                  schema=schema).sort_by(order)
                        ]))
        if not runs:
            return
        entity = schema.get_field_index('ENTITY')
        score, item = schema.get_field_index('SCORE'), schema.get_field_index(
            'CLASS')
        key = (lambda row: (row[entity], -row[score], row[item])
               ) if ranked else lambda row: row[entity]
        while len(runs) > 32:
            following = []
            for start in range(0, len(runs), 32):
                group = runs[start:start + 32]
                if len(group) == 1:
                    following.extend(group)
                    continue
                following.append(spill(merge(group)))
                for path in group:
                    path.unlink()
            runs = following
        for table in merge(runs):
            yield table.to_pandas(ignore_metadata=True)


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
                    or hashlib.sha256(destination.read_bytes()).digest()
                    != hashlib.sha256(archive.read(name)).digest()):
                raise ValueError(f'Installed wheel member differs: {name}')
            checked += 1
    if not checked:
        raise ValueError('Wheel contains no installed files')
    return checked


def verify_checkpoint(model, path, expected_digest):
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

    equal(expected, actual)
    if not tensors:
        raise ValueError('Reload checked no model tensors')
    return tensors


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


@contextmanager
def observe_cuda(state):
    """Extend the shared public CUDA observer with nightly-only work proof."""
    from numbers import Integral

    import pytorch_lightning as pl
    import torch
    from kumo_core.io.filesystem import local_path
    from kumo_core.train.trainer import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.trainer.states import TrainerFn

    recoveries = state.setdefault('oom_recoveries', [])

    def observe_recovery(method, original):
        @wraps(original)
        def observed(*args, **kwargs):
            before = torch.cuda.memory_stats().get('num_ooms')
            result = original(*args, **kwargs)
            after = torch.cuda.memory_stats().get('num_ooms')
            if before is not None and after is not None and after > before:
                recoveries.append(
                    dict(stage=state['stage'], method=method,
                         allocator_ooms=after - before))
            return result

        return observed

    class Proof(pl.Callback):
        def on_fit_start(self, trainer, module):
            if 'fit_ledger' in state:
                self.candidate = dict(status='RUNNING', steps=0, cuda_batches=0,
                                      epochs=[], validation=[])
                state['fit_ledger'].append(self.candidate)

        def on_train_epoch_start(self, trainer, module):
            if 'training_epochs' in state:
                epoch = dict(epoch=int(trainer.current_epoch), batches=0,
                             batch_sizes=[])
                state['training_epochs'].append(epoch)
                self.candidate['epochs'].append(epoch)

        def on_train_batch_end(self, trainer, module, values, batch, batch_idx):
            if 'fit_ledger' in state:
                self.candidate['cuda_batches'] += 1
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

        def on_validation_end(self, trainer, module):
            if (not getattr(self, 'candidate', None) or trainer.sanity_checking
                    or state['stage'] != 'fit'):
                return
            metrics = {}
            for name, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() != 1:
                        continue
                    value = value.detach().cpu().item()
                if type(value) in (int, float):
                    metrics[name] = value if math.isfinite(value) else None
            fitting = trainer.state.fn == TrainerFn.FITTING
            self.candidate['validation'].append(
                dict(epoch=int(trainer.current_epoch), metrics=metrics,
                     phase='fit' if fitting else 'candidate-validation'))
            if not fitting and trainer.ckpt_path is not None:
                self.candidate['checkpoint_sha256'] = digest(
                    local_path(str(trainer.ckpt_path)))

        def on_fit_end(self, trainer, module):
            if 'fit_ledger' in state:
                self.candidate.update(status='PASS',
                                      steps=int(trainer.global_step))
                checkpoint = trainer.checkpoint_callback.best_model_path
                checkpoint_digest = digest(checkpoint)
                self.candidate.update(
                    checkpoint_sha256=checkpoint_digest,
                    checkpoint_before_validation_sha256=checkpoint_digest,
                    early_stopping=[
                        dict(callback=type(stop).__name__, monitor=stop.monitor,
                             stopped_epoch=int(stop.stopped_epoch),
                             wait_count=int(stop.wait_count),
                             patience=int(stop.patience))
                        for stop in trainer.callbacks
                        if isinstance(stop, EarlyStopping)
                    ])

        def on_exception(self, trainer, module, error):
            if state['stage'] == 'fit' and getattr(self, 'candidate', None):
                self.candidate.update(error=f'{type(error).__name__}: {error}')
                if self.candidate['status'] == 'RUNNING':
                    self.candidate.update(status='FAIL',
                                          steps=int(trainer.global_step))

        def on_predict_start(self, trainer, module):
            if state['stage'] == 'reload' and 'reload_checks' in state:
                state['restored_tensors'] = verify_checkpoint(
                    module, trainer.ckpt_path, state['checkpoint_sha256'])
                state['reload_checks'] += 1

    with smoke.observe_cuda(state):
        base = pl.Trainer

        class ObservedTrainer(base):
            def __init__(self, *args, **kwargs):
                kwargs['callbacks'] = [
                    *(kwargs.get('callbacks') or []),
                    Proof()
                ]
                super().__init__(*args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(pl, 'Trainer', ObservedTrainer))
            for method in ('predict_v2', 'test'):
                stack.enter_context(
                    patch.object(
                        Trainer, method,
                        observe_recovery(method, getattr(Trainer, method))))
            yield


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


def check_learning(rows, bounds):
    """Require normal gain and a control near its independent baseline."""
    tasks = {}
    if not rows or set(bounds) != {row['task'] for row in rows}:
        raise ValueError('Learning bounds must cover every measured task')
    for task, limits in bounds.items():
        if (set(limits) != {'min_normal_gain', 'max_control_drift'} or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in limits.values()) or limits['min_normal_gain'] <= 0
                or limits['max_control_drift'] < 0):
            raise ValueError(
                'Learning bounds need a positive gain and finite control tolerance'
            )
        pair = [row for row in rows if row['task'] == task]
        if len(pair) != 2 or {row['arm']
                              for row in pair} != {'normal', 'control'}:
            raise ValueError(
                'Exactly one normal and control measurement required')
        normal, control = [
            next(row for row in pair if row['arm'] == arm)
            for arm in ('normal', 'control')
        ]
        for row in pair:
            validate_quality_measurement(task, row.get('score'))
            validate_quality_measurement(task, row.get('baseline'))
        direction = -1 if task in ('regression', 'forecasting') else 1
        gain = direction * (normal['score'] - normal['baseline'])
        paired_gain = direction * (normal['score'] - control['score'])
        drift = abs(control['score'] - control['baseline'])
        passed = (gain >= limits['min_normal_gain']
                  and paired_gain >= limits['min_normal_gain']
                  and drift <= limits['max_control_drift'])
        tasks[task] = dict(
            status='PASS' if passed else 'FAIL', normal_score=normal['score'],
            control_score=control['score'], normal_baseline=normal['baseline'],
            control_baseline=control['baseline'], normal_gain=gain,
            normal_control_gain=paired_gain, control_drift=drift, **limits)
    return dict(
        status='PASS' if all(v['status'] == 'PASS'
                             for v in tasks.values()) else 'FAIL',
        policy='enforced', tasks=tasks)


def validate_quality_measurement(task, score):
    if (type(score) not in (int, float) or not math.isfinite(score) or score < 0
            or task not in ('regression', 'forecasting') and score > 1):
        raise ValueError('Quality measurement is missing or invalid')


def load_policy(path, expected):
    if path.is_symlink() or digest(path) != expected:
        raise ValueError('Reviewed policy checksum differs')
    value = read(path)
    if (set(value) != {'format', 'learning', 'predictions'}
            or value['format'] != 'kumo-engine-nightly-policy-v1'
            or set(value['learning']) != set(smoke.TASKS)
            or set(value['predictions']) != set(smoke.TASKS)):
        raise ValueError('The policy must cover all eight tasks')
    for task in smoke.TASKS:
        learning, prediction = value['learning'][task], value['predictions'][
            task]
        if (set(learning) != {'min_normal_gain', 'max_control_drift'}
                or set(prediction) != {'atol'} or any(
                    type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in (*learning.values(), *prediction.values()))
                or learning['min_normal_gain'] <= 0 or prediction['atol'] <= 0):
            raise ValueError(
                'Use explicit finite learning and prediction bounds')
    return value


def summarize(rows, reference, policy):
    check_results(rows)
    previous = {(row['task'], row['arm']): row for row in reference['cases']}
    if (len(previous) != 16 or len(reference['cases']) != 16
            or set(previous) != {(row['task'], row['arm'])
                                 for row in rows}):
        raise ValueError(
            'Reference and candidate must cover the same sixteen cases')
    scores = [
        dict(
            task=row['task'], arm=row['arm'], score=row['score'],
            baseline=previous[(row['task'], row['arm'])]['baseline'],
            historical_baseline=previous[(row['task'], row['arm'])].get(
                'historical_baseline',
                previous[(row['task'], row['arm'])]['baseline']))
        for row in rows
    ]
    learning = check_learning(scores, policy['learning'])
    return dict(
        status='PASS' if learning['status'] == 'PASS' and all(
            row.get('prediction_checks', {}).get('status') == 'PASS'
            for row in rows) else 'FAIL', learning=learning,
        numerical_difference_policy='enforced', cases=[
            dict(score,
                 reference_score=previous[(row['task'], row['arm'])]['score'],
                 seconds=row['seconds'], fit_id=row['fit_id'])
            for row, score in zip(rows, scores, strict=True)
        ])


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


def run_case(root, manifest, case, graph, snapshot, output, scratch, tolerance,
             *, measure_only=False):
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
    state = dict(stage='fit', fits=0, completed_fits=0, training_batches=0,
                 optimizer_steps=0, training_epochs=[],
                 prediction_batches={s: 0
                                     for s in PREDICTIONS}, reload_checks=0,
                 restored_tensors=0, fit_ledger=[])
    report = dict(task=task, arm=case['arm'], seed=0, status='RUNNING',
                  work=state)
    try:
        with measure_operation(report, 'preparation'):
            table, stages, descriptor = load_table(root, manifest['files'],
                                                   case)
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
                    len(v) != 12
                    for col in ('autoregression_labels', 'future_labels')
                    for v in truth[col]):
                raise ValueError(
                    'Forecast inputs/answers require all twelve coordinates')
            query = kumo.PredictiveQuery(graph, descriptor['query'])
            query.validate()
            plan = query.suggest_model_plan(snapshot)
            save(output / 'requested-plan.json', plan.as_json())
            state['requested_batch_sizes'] = plan.as_json(
            )['optimization']['batch_size']
            report['training_recipe'] = 'engine-suggested-fast'
            report['requested_plan_sha256'] = digest(output /
                                                     'requested-plan.json')
            report['selection_metric'] = plan.as_json(
            )['training_job']['tune_metric']
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
                    raise ValueError(
                        'Link catalog keys are missing or duplicated')
            anchor = None if task == 'static_link' else full.TIMESTAMP.iloc[0]
            if anchor is not None and not full.TIMESTAMP.eq(anchor).all():
                raise ValueError(
                    'Full prediction needs the fixed frozen test anchor')
            seed = lambda: (random.seed(0), np.random.seed(0),
                            torch.manual_seed(0))
            work = scratch / output.name
            kumo.configure(scratch_dir=work)
            seed()
        with observe_cuda(state):
            trainer = kumo.Trainer(plan)
            torch.cuda.synchronize()
            with measure_operation(report, 'training'):
                fitted = trainer.fit(table, snapshot=snapshot)
                torch.cuda.synchronize()
            save(output / 'candidate-validation.json', state['fit_ledger'])
            save(output / 'fit-metrics.json', fitted.metrics)
            checkpoint = local_path(
                str(fitted.artifact_paths['selected_checkpoint']))
            state['checkpoint_sha256'] = digest(checkpoint)
            selected_steps = load_kumo_checkpoint(
                str(checkpoint), map_location='cpu')['global_step']
            if query.get_task_type().value != smoke.TASKS[task]['task_type']:
                raise ValueError(
                    'The public query resolves to a different task')
            for name in ('resolved_model_plan', 'trainer_config', 'encoders',
                         'target_encoder'):
                if name in fitted.artifact_paths:
                    shutil.copyfile(local_path(fitted.artifact_paths[name]),
                                    output / (name + '.json'))
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
            check_training(state, selected_steps)
            time_column = case.get('prediction_time_column', 'TIMESTAMP')

            def predict(model, stage, frame):
                label = 'reloaded_prediction' if stage == 'reload' else stage
                with measure_operation(report, label):
                    torch.cuda.synchronize()
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
                        output_paths={'predictions': stage
                                      }, extra_fields=[] if task ==
                        'static_link' else [MetadataField.ANCHOR_TIMESTAMP]))
                    if task.endswith('link'):
                        options['num_classes_to_return'] = min(
                            500, len(catalog))
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
                        selected_truth = truth.set_index(
                            'ENTITY1').loc[selected]
                        excluded = selected_truth.exclude_target_keys.tolist(
                        ) if task == 'static_link' else [[] for _ in selected]
                        values = ranked_rows(
                            ordered_batches(result.files,
                                            work / 'sort' / stage), selected,
                            catalog, excluded,
                            5 if task == 'static_link' else 10,
                            min(500, len(catalog)), anchor=anchor)
                        save(output / (stage + '-ranked.json'), values)
                    else:
                        frame = result.frame
                        if frame.ENTITY.dtype != full.ENTITY1.dtype:
                            raise ValueError(
                                'Prediction changed the original entity ID type'
                            )
                        frame.to_parquet(output / (stage + '.parquet'),
                                         index=False)
                        values = smoke.prediction_values(
                            task, frame, selected, classes=classes,
                            anchor=anchor, time_column=time_column
                            or 'TIMESTAMP')
                    decisions = (values['ranked_ids'] if task.endswith('link')
                                 else smoke.prediction_decisions(
                                     task, frame, selected, classes=classes))
                    report.setdefault('predictions', {})[stage] = dict(
                        entities=len(selected), rows=result.num_rows,
                        files=retained, original_files=list(result.files),
                        input_sha256=digest(inputs))
                    return values, decisions

            original, original_decisions = predict(trainer, 'predict', full)
            report.update(score=quality(task, truth, original, classes),
                          metric=smoke.TASKS[task]['metric'])
            save(output / 'result.json', report)
            repeated, repeat_decisions = predict(trainer, 'repeat', full)
            subset, subset_decisions = predict(trainer, 'partial', partial)
            del trainer
            gc.collect()
            with measure_operation(report, 'model_load'):
                trainer = kumo.Trainer.load(saved)
                torch.cuda.synchronize()
            reloaded, reload_decisions = predict(trainer, 'reload', full)
            if state['reload_checks'] < 1 or state['restored_tensors'] <= 0:
                raise ValueError(
                    'Reload lacks an exact restored checkpoint check')
            lookup = {v: i for i, v in enumerate(ids)}
            positions = [lookup[v] for v in partial_ids]
            report['differences'] = {}
            for label, other, decisions, selected in [
                ('repeat', repeated, repeat_decisions, None),
                ('partial', subset, subset_decisions, positions),
                ('reload', reloaded, reload_decisions, None)
            ]:
                if task.endswith('link'):
                    before = original if selected is None else {
                        k: [original[k][i] for i in selected]
                        for k in ('ranked_ids', 'ranked_scores', 'boundary_ids',
                                  'boundary_scores')
                    }
                    report['differences'][label] = ranked_difference(
                        before, other, **tolerance, measure_only=measure_only)
                else:
                    report['differences'][label] = smoke.compare_predictions(
                        original if selected is None else original[selected],
                        other, task=task, **tolerance,
                        before_decisions=original_decisions
                        if selected is None or original_decisions is None else
                        original_decisions[selected], after_decisions=decisions,
                        measure_only=measure_only)
            report['prediction_checks'] = dict(status='PASS' if all(
                value['status'] == 'PASS'
                for value in report['differences'].values()) else 'FAIL')
            report.update(status='MEASURED', physical_fits=state['fits'],
                          prediction_batches=state['prediction_batches'],
                          policy='measurement' if measure_only else 'enforced')
            check_training(state, selected_steps)
            load_table(root, manifest['files'], case)
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc(),
                      physical_fits=state['fits'])
        raise
    finally:
        try:
            report['artifacts'] = {}
            for path in output.rglob('*'):
                if path.is_file() and path != output / 'result.json':
                    report['artifacts'][str(
                        path.relative_to(output))] = digest(path)
        except BaseException as error:
            report[
                'artifact_preservation_error'] = f'{type(error).__name__}: {error}'
            if report['status'] != 'FAIL':
                report.update(status='FAIL',
                              error=report['artifact_preservation_error'],
                              traceback=traceback.format_exc())
                raise
        finally:
            report['seconds'] = time.monotonic() - started
            save(output / 'result.json', report)

    return report


def check_results(rows, cases=None):
    wanted = ({(task, arm, 0)
               for task in smoke.TASKS
               for arm in ('normal', 'control')}
              if cases is None else {(case['task'], case['arm'], case['seed'])
                                     for case in cases})
    if (len(rows) != len(wanted) or {(r['task'], r['arm'], r['seed'])
                                     for r in rows} != wanted
            or len({r['fit_id']
                    for r in rows}) != len(wanted)
            or any(r['status'] != 'MEASURED' or type(r['seed']) is not int or
                   type(r['physical_fits']) is not int or r['physical_fits'] < 1
                   or len(r['checkpoint_sha256']) != 64 or not all(
                       r['prediction_batches'].get(s, 0) > 0
                       for s in PREDICTIONS) for r in rows)):
        raise ValueError('Comparison needs complete unique candidate workflows')


def case_learning(row, reference, bounds):
    task, score, baseline = row['task'], row['score'], reference['baseline']
    validate_quality_measurement(task, score)
    validate_quality_measurement(task, baseline)
    if row['arm'] == 'normal':
        gain = (-1 if task in ('regression',
                               'forecasting') else 1) * (score - baseline)
        passed = gain >= bounds['min_normal_gain']
    else:
        passed = abs(score - baseline) <= bounds['max_control_drift']
    return dict(
        status='PASS' if passed else 'FAIL', score=score, baseline=baseline,
        historical_baseline=reference.get('historical_baseline',
                                          baseline), **bounds)


def resource_requirements(measurement, headroom=1.25):
    """Propose admission minima from one complete case's measured peaks."""
    peaks = measurement.get('peaks', {})
    if (measurement.get('format') != 'kumo-engine-nightly-resources-v1'
            or measurement.get('status') != 'MEASURED'
            or type(headroom) not in (int, float) or not math.isfinite(headroom)
            or headroom <= 1
            or not isinstance(measurement.get('gpu_class'), str)
            or not measurement['gpu_class']
            or type(measurement.get('gpus')) is not int
            or measurement['gpus'] < 1
            or any(not re.fullmatch('[0-9a-f]{64}', measurement.get(name, ''))
                   for name in ('wheel_sha256', 'harness_sha256',
                                'input_manifest_sha256')) or set(peaks)
            != {'cpu_cores', 'ram_bytes', 'scratch_bytes', 'gpu_memory_bytes'}
            or any(
                type(value) not in
                (int, float) or not math.isfinite(value) or value <= 0
                for value in (*peaks.values(), measurement.get('seconds')))):
        raise ValueError(
            'Resource requirements need complete measured evidence')
    minimum = dict(
        cpus=math.ceil(peaks['cpu_cores'] * headroom),
        ram_bytes=math.ceil(peaks['ram_bytes'] * headroom),
        scratch_bytes=math.ceil(peaks['scratch_bytes'] * headroom),
        gpu_memory_bytes=math.ceil(peaks['gpu_memory_bytes'] * headroom),
        gpu_class=measurement['gpu_class'], gpus=measurement['gpus'],
        time_seconds=math.ceil(measurement['seconds'] * headroom))
    return dict(minimum=minimum, measurement=measurement, headroom=headroom)


def admit_resources(profile, capacity):
    if not isinstance(profile, dict) or set(profile) != {
            'minimum', 'measurement', 'headroom'
    }:
        raise ValueError(
            'The case needs reviewed measured resource requirements')
    if profile != resource_requirements(profile['measurement'],
                                        profile['headroom']):
        raise ValueError('Resource minima differ from their measured evidence')
    for name, required in profile['minimum'].items():
        actual = capacity.get(name)
        if ((actual != required)
                if name == 'gpu_class' else type(actual) not in (int, float) or
            (not math.isfinite(actual)
             and not (name == 'time_seconds' and actual == math.inf))
                or actual < required):
            raise ValueError(
                f'Insufficient {name}: {actual!r}, requires {required!r}')
    return profile['minimum']


def available_capacity(scratch, runtime, time_seconds):
    import psutil
    import torch

    cpus = len(os.sched_getaffinity(0))
    memory = psutil.virtual_memory().available
    try:
        relative = next(
            line.split('::', 1)[1]
            for line in Path('/proc/self/cgroup').read_text().splitlines()
            if line.startswith('0::'))
        group = Path('/sys/fs/cgroup') / relative.lstrip('/')
        quota, period = (group / 'cpu.max').read_text().split()
        if quota != 'max':
            cpus = min(cpus, int(quota) / int(period))
        limit = (group / 'memory.max').read_text().strip()
        if limit != 'max':
            memory = min(
                memory,
                int(limit) - int((group / 'memory.current').read_text()))
    except (OSError, ValueError, StopIteration):
        pass
    return dict(cpus=cpus, ram_bytes=memory,
                scratch_bytes=shutil.disk_usage(scratch).free,
                gpu_class=runtime['gpu'], gpus=runtime['gpu_count'],
                gpu_memory_bytes=torch.cuda.mem_get_info()[0],
                time_seconds=time_seconds)


@contextmanager
def track_resources(output, scratch, runtime):
    """Sample one process tree and its owned files, once per second."""
    import threading

    import psutil
    import torch

    process = psutil.Process()
    peaks = dict(cpu_cores=0., ram_bytes=0, scratch_bytes=0, gpu_memory_bytes=0)
    result = dict(format='kumo-engine-nightly-resources-v1',
                  status='INCOMPLETE', peaks=peaks, errors=[],
                  sample_interval_seconds=1, gpu_class=runtime['gpu'],
                  gpus=runtime['gpu_count'])
    done = threading.Event()
    started = previous_time = time.monotonic()
    previous_cpu = {}

    def sample():
        nonlocal previous_cpu, previous_time
        now = time.monotonic()
        cpu, rss, current = 0., 0, {}
        for child in [process, *process.children(recursive=True)]:
            try:
                times = child.cpu_times()
                current[child.pid] = times.user + times.system
                cpu += max(
                    0., current[child.pid] -
                    previous_cpu.get(child.pid, current[child.pid]))
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peaks['cpu_cores'] = max(peaks['cpu_cores'],
                                 cpu / max(now - previous_time, 1e-6))
        peaks['ram_bytes'] = max(peaks['ram_bytes'], rss)
        previous_cpu, previous_time = current, now
        size = 0
        for directory in (output, scratch):
            for path in directory.rglob('*'):
                try:
                    if path.is_file():
                        size += path.stat().st_blocks * 512
                except FileNotFoundError:
                    pass
        peaks['scratch_bytes'] = max(peaks['scratch_bytes'], size)
        free, total = torch.cuda.mem_get_info()
        peaks['gpu_memory_bytes'] = max(peaks['gpu_memory_bytes'], total - free)

    def sample_safely():
        try:
            sample()
        except Exception as error:
            message = f'{type(error).__name__}: {error}'
            if message not in result['errors']:
                result['errors'].append(message)

    def observe():
        while not done.wait(1):
            sample_safely()

    sample_safely()
    worker = threading.Thread(target=observe, daemon=True)
    worker.start()
    try:
        yield result
    finally:
        done.set()
        worker.join()
        sample_safely()
        result['seconds'] = time.monotonic() - started


@contextmanager
def case_deadline(seconds):
    import signal

    def expired(signum, frame):
        raise TimeoutError(f'Case exceeded its {seconds}-second time budget')

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run(args):
    started = time.monotonic()
    root, output, scratch = args.input_root.resolve(), args.output_dir.resolve(
    ), args.scratch.resolve()
    if output.exists() and any(p.name != 'setup' for p in output.iterdir()):
        raise ValueError('Use a fresh output directory')
    if any(
            scratch.is_relative_to(p) or p.is_relative_to(scratch)
            for p in (root, output)):
        raise ValueError('Keep scratch outside the input and evidence trees')
    manifest = load_inputs(root, args.input_sha256)
    reference = load_reference(args)
    reference_cases(reference, manifest)
    policy = load_policy(args.policy, args.policy_sha256)
    selected = [
        case for case in manifest['cases']
        if getattr(args, 'case', None) in (None,
                                           case['task'] + '-' + case['arm'])
    ]
    if not selected:
        raise ValueError('Select a frozen task and answer arm')
    requirements = getattr(args, 'requirements', None)
    if requirements is not None:
        expected = getattr(args, 'requirements_sha256', '')
        if (len(selected) != 1 or requirements.is_symlink()
                or not re.fullmatch('[0-9a-f]{64}', expected)
                or digest(requirements) != expected):
            raise ValueError('Case requirements checksum or selection differs')
        selected = [dict(selected[0], requirements=read(requirements))]
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=False)
    runtime = verify_runtime(
        args.expected_version,
        read(root / manifest['reference_dependencies']) if args.reference_only
        and not getattr(args, 'measure_resources', False) else None,
        reference_only=args.reference_only)
    runtime['installed_files'] = verify_installed_files(
        args.wheel, importlib.metadata.distribution('kumo-engine'))
    if digest(args.wheel) != runtime['wheel_sha256']:
        raise ValueError('Installed archive and selected wheel differ')
    check_install(output, runtime, reference_only=args.reference_only,
                  build_receipt=getattr(args, 'build_receipt', None))
    report = dict(
        format='kumo-engine-nightly-result-v1', status='RUNNING',
        input_manifest_sha256=args.input_sha256, runtime=runtime,
        reference_sha256=args.reference_sha256,
        policy_sha256=args.policy_sha256, reference_only=args.reference_only,
        measurement_only=getattr(args, 'measure_resources', False),
        training_recipe='engine-suggested-fast',
        requirements_sha256=getattr(args, 'requirements_sha256',
                                    None), expected_runs=len(selected), runs=[],
        helper_sha256=digest(HELPER), runner_sha256=digest(__file__))
    try:
        for case in selected:
            directory = output / (case['task'] + '-' + case['arm'])
            graph = snapshot = None
            resources = None
            graph_timing = {}
            row = dict(task=case['task'], arm=case['arm'], seed=case['seed'],
                       status='RUNNING')
            try:
                import kumo.engine as kumo
                profile = case.get('requirements')
                budget = getattr(args, 'time_budget_seconds', None)
                if budget is None:
                    declared = profile.get('minimum', {}) if isinstance(
                        profile, dict) else {}
                    budget = declared.get('time_seconds', 0) if isinstance(
                        declared, dict) else 0
                capacity = available_capacity(scratch, runtime, budget)
                if report['measurement_only']:
                    budget = getattr(args, 'time_budget_seconds', None)
                    if type(budget) is not int or budget <= 0:
                        raise ValueError(
                            'Resource measurement needs an explicit time budget'
                        )
                    minimum = dict(time_seconds=budget)
                else:
                    minimum = admit_resources(profile, capacity)
                    measured = profile['measurement']
                    if ((measured.get('task'),
                         measured.get('arm')) != (case['task'], case['arm'])
                            or measured.get('input_manifest_sha256')
                            != args.input_sha256
                            or measured.get('input_signature')
                            != input_signature(manifest, case)):
                        raise ValueError(
                            'Resource evidence belongs to another case or input'
                        )
                key = case['graph']
                spec = manifest['graphs'][key]
                with track_resources(directory, scratch / directory.name,
                                     runtime) as resources:
                    with case_deadline(minimum['time_seconds']):
                        with measure_operation(graph_timing, 'graph'):
                            kumo.configure(scratch_dir=scratch /
                                           directory.name / 'graph')
                            graph, snapshot = build_graph(
                                read(root / spec['schema'])[spec['profile']],
                                root, static=spec['static'],
                                sources=spec['sources'])
                            check_snapshot(snapshot,
                                           dict(spec, input_root=str(root)))
                        row = run_case(root, manifest, case, graph, snapshot,
                                       directory, scratch,
                                       policy['predictions'][case['task']],
                                       measure_only=report['measurement_only'])
                        check_results([row], [case])
                if resources.get('errors'):
                    raise ValueError('Resource observation failed')
                previous = next(value for value in reference['cases']
                                if (value['task'],
                                    value['arm']) == (case['task'],
                                                      case['arm']))
                row['learning'] = case_learning(
                    row, previous, policy['learning'][case['task']])
                row['admission'] = dict(capacity=capacity, requirements=profile)
            except Exception as error:
                try:
                    row = read(directory / 'result.json')
                except (OSError, ValueError):
                    pass
                row.update(status='FAIL',
                           error=f'{type(error).__name__}: {error}',
                           traceback=traceback.format_exc())
            finally:
                del graph, snapshot
                gc.collect()
            row.setdefault('timings',
                           {}).update(graph_timing.get('timings', {}))
            if resources is not None:
                resources.update(
                    status='MEASURED' if row['status'] == 'MEASURED' else
                    'INCOMPLETE', task=case['task'], arm=case['arm'],
                    input_signature=input_signature(manifest, case),
                    input_manifest_sha256=args.input_sha256,
                    wheel_sha256=runtime['wheel_sha256'],
                    harness_sha256=digest(__file__))
                row['resources'] = resources
                proposal = None
                if resources['status'] == 'MEASURED':
                    try:
                        proposal = resource_requirements(resources)
                    except (KeyError, TypeError, ValueError) as error:
                        resources['status'] = 'INCOMPLETE'
                        row.update(status='FAIL',
                                   error=f'{type(error).__name__}: {error}',
                                   traceback=traceback.format_exc())
                save(directory / 'resources.json', resources)
                if report['measurement_only'] and proposal is not None:
                    save(directory / 'requirements-proposal.json', proposal)
            save(directory / 'result.json', row)
            report['runs'].append(row)
            save(output / 'nightly-result.json', report)
        if any(row['status'] != 'MEASURED' for row in report['runs']):
            summary = dict(status='FAIL', cases=report['runs'])
        elif len(selected) == 16:
            summary = summarize(report['runs'], reference, policy)
        else:
            summary = dict(
                status='PASS' if all(
                    row['learning']['status'] == 'PASS'
                    and row.get('prediction_checks', {}).get('status') == 'PASS'
                    for row in report['runs']) else 'FAIL',
                cases=report['runs'])
        save(output / 'comparison.json', summary)
        load_inputs(root, args.input_sha256)
        if report['measurement_only'] and all(row['status'] == 'MEASURED'
                                              for row in report['runs']):
            report['status'] = summary['status'] = 'MEASURED'
            save(output / 'comparison.json', summary)
        else:
            report['status'] = 'PASS' if summary['status'] == 'PASS' else 'FAIL'
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        report.update(seconds=time.monotonic() - started,
                      completed_runs=len(report['runs']))
        save(output / 'nightly-result.json', report)
    return report['status'] in ('PASS', 'MEASURED')


def check_install(root, runtime, *, reference_only=False, build_receipt=None):
    build = read(build_receipt or root / 'setup/build.json')
    if (type(runtime.get('installed_files')) is not int
            or runtime['installed_files'] <= 0
            or build['wheel_sha256'] != runtime['wheel_sha256']
            or build['version'] != runtime['version']):
        raise ValueError(
            'Selected source, installed wheel or archive identity differs')
    if reference_only and (runtime['version'] != '1.0.0'
                           or runtime['wheel_sha256'] != REFERENCE_WHEEL):
        raise ValueError(
            'Reference-only work requires the exact approved wheel')
    if build.get('format') == 'kumo-engine-published-reference-v1':
        harness = os.environ.get('GITHUB_SHA', '')
        if (not reference_only or not re.fullmatch('[0-9a-f]{40}', harness)
                or build.get('harness') != harness):
            raise ValueError(
                'Published reference receipt has no matching harness')
    elif any(not re.fullmatch('[0-9a-f]{40}', build.get(k, ''))
             for k in ('revision', 'tree')):
        raise ValueError('Selected source revision or tree identity differs')
    return build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--policy-sha256', required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--build-receipt', type=Path, required=True)
    parser.add_argument(
        '--case', required=True, choices=[
            task + '-' + arm for task in smoke.TASKS
            for arm in ('normal', 'control')
        ])
    parser.add_argument('--expected-version', required=True)
    parser.add_argument('--time-budget-seconds', type=int)
    parser.add_argument('--measure-resources', action='store_true')
    parser.add_argument('--requirements', type=Path)
    parser.add_argument('--requirements-sha256')
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args(argv)
    if args.time_budget_seconds is not None and args.time_budget_seconds <= 0:
        parser.error('--time-budget-seconds must be positive')
    if args.measure_resources and args.time_budget_seconds is None:
        parser.error('--measure-resources requires --time-budget-seconds')
    if bool(args.requirements) != bool(args.requirements_sha256):
        parser.error('Provide both --requirements and --requirements-sha256')
    if not args.measure_resources and args.requirements is None:
        parser.error(
            'Provide reviewed requirements or select --measure-resources')
    return 0 if run(args) else 1


if __name__ == '__main__':
    raise SystemExit(main())
