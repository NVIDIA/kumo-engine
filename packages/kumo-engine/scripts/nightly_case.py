# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute one scale-suite case with the installed Engine wheel."""

import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from datetime import date, datetime
from functools import wraps
from itertools import islice, zip_longest
from numbers import Integral, Real
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

PREDICTIONS = ('predict', 'repeat', 'partial', 'reload')
STAGES = {0: 'train', 1: 'val', 2: 'test'}
ANSWER_FIELDS = {
    'TARGET1', 'future_labels', '__kumo_internal_frequency_column__'
}
CAPS = dict(num_experiments=2, max_epochs=3, max_steps_per_epoch=200,
            max_val_steps=100, max_test_steps=200, sample_entities=1000,
            sample_seed=0)
_LABEL_PARALLEL_ROWS = 1_000_000
_LABEL_WORKERS = 8
_LABEL_BLOCK_BYTES = 64 * 1024**2


def neighbor_depths(plan):
    depths = []
    options = plan.neighbor_sampling.num_neighbors
    if not isinstance(options, list) or not options:
        raise ValueError('Deep graph cases need explicit public hop settings')
    for option in options:
        if hasattr(option, 'num_hops'):
            option.num_hops()
        hops = list(option.to_dict().values()) if hasattr(option,
                                                          'to_dict') else option
        if not isinstance(hops, list) or not 1 <= len(hops) <= 6:
            raise ValueError(
                'Graph hop settings must define one to six consecutive hops')
        for hop in hops:
            values = ([hop.default, *hop.__pydantic_extra__.values()]
                      if hasattr(hop, 'default') else [hop])
            if not any(
                    type(value) is int and (value == -1 or value > 0)
                    for value in values):
                raise ValueError(
                    'Each configured graph hop needs an active fanout')
        depths.append(len(hops))
    return depths


def capped_plan(query, snapshot, *, min_hops=None, max_hops=None):
    """Keep the public suggested behavior except for the reviewed work caps."""
    plan = copy.deepcopy(query.suggest_model_plan(snapshot))
    plan.training_job.num_experiments = CAPS['num_experiments']
    for name in ('max_epochs', 'max_steps_per_epoch', 'max_val_steps',
                 'max_test_steps'):
        setattr(plan.optimization, name, CAPS[name])
    if min_hops is not None or max_hops is not None:
        if (type(min_hops) is not int or type(max_hops) is not int
                or not 1 <= min_hops <= max_hops <= 6):
            raise ValueError(
                'Graph hop bounds must be integers from one to six')
        depths = neighbor_depths(plan)
        if max(depths) > max_hops:
            raise ValueError(
                'Suggested graph hops exceed this case depth limit')
        for option, depth in zip(plan.neighbor_sampling.num_neighbors, depths,
                                 strict=True):
            for index in range(depth, min_hops):
                if hasattr(option, 'to_dict'):
                    setattr(option, f'hop{index + 1}',
                            copy.deepcopy(getattr(option, f'hop{depth}')))
                else:
                    option.append(copy.deepcopy(option[-1]))
        neighbor_depths(plan)
    return plan


def sample_populations(frame, seed=0, size=1000, *, ordered=False):
    if frame.empty or 'ENTITY1' not in frame or frame.ENTITY1.duplicated().any(
    ):
        raise ValueError('Prediction population must contain unique entities')
    population = frame if ordered else frame.sort_values('ENTITY1').reset_index(
        drop=True)
    positions = list(range(len(population)))
    random.Random(seed).shuffle(positions)
    consistency = population.iloc[positions[:min(size, len(population))]].copy()
    if len(consistency) < len(population):
        partial = consistency.copy()
    else:
        partial = population.iloc[positions[:len(population) // 2]].copy()
    return dict(consistency=consistency, partial=partial,
                proper_subset=0 < len(partial) < len(population))


def check_training_coverage(ledger):
    trials = {}
    for attempt in ledger:
        identity = attempt.get('trial_id')
        if (not identity or attempt.get('kind') != 'candidate'
                or attempt.get('steps', 0) <= 0
                or attempt.get('cuda_batches', 0) <= 0):
            continue
        trials.setdefault(identity, set()).update(
            epoch['epoch'] for epoch in attempt.get('epochs', [])
            if epoch.get('completed') is True and epoch.get('batches', 0) > 0)
    epochs = max(map(len, trials.values()), default=0)
    return dict(status='PASS' if len(trials) >= 2 and epochs >= 2 else 'FAIL',
                distinct_trials=len(trials), max_completed_epochs=epochs)


def with_reference_replay(replay, fresh, *, evidence=None):
    evidence = {} if evidence is None else evidence
    evidence.update(status='SKIPPED')
    if replay is not None:
        try:
            result = replay()
            evidence.clear()
            evidence.update(result)
        except Exception as error:
            evidence.update(status='FAIL',
                            error=f'{type(error).__name__}: {error}',
                            traceback=traceback.format_exc())
    return fresh(), evidence


def _value(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (datetime, date)) or (hasattr(value, 'dtype')
                                               and value.dtype.kind == 'M'):
        import pandas as pd
        return {'timestamp_ns': int(pd.Timestamp(value).value)}
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Nonfinite training label')
        return int(number) if number.is_integer() else number
    if hasattr(value, 'tolist'):
        return _value(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    raise ValueError(f'Unsupported training value: {type(value).__name__}')


def _row(row, task):
    result = {key: _value(value) for key, value in row.items()}
    if task in ('multilabel', 'multilabel_ranking', 'static_link',
                'temporal_link'):
        targets = result['TARGET1']
        targets = targets.split('|') if isinstance(targets, str) else targets
        frequency = result.get('__kumo_internal_frequency_column__')
        if frequency is None:
            result['TARGET1'] = sorted(targets)
        else:
            pairs = sorted(zip(targets, frequency, strict=True))
            result['TARGET1'] = [item for item, _ in pairs]
            result['__kumo_internal_frequency_column__'] = [n for _, n in pairs]
    if 'exclude_target_keys' in result:
        result['exclude_target_keys'] = sorted(result['exclude_target_keys'])
    return result


def _training_batches(files, scratch, helpers, *, ordered=False):
    import pyarrow.parquet as pq

    names = pq.read_schema(files[0]).names
    if not {'ENTITY1', 'TARGET1', 'SPLIT'} <= set(names):
        raise ValueError('Exported training rows lack keys, labels, or stages')
    if ordered:
        for path in files:
            with pq.ParquetFile(path) as reader:
                for batch in reader.iter_batches(batch_size=65536,
                                                 use_threads=False):
                    yield batch.to_pandas(ignore_metadata=True)
        return
    keys = [('ENTITY1', 'ascending')]
    if 'TIMESTAMP' in names:
        keys.append(('TIMESTAMP', 'ascending'))
    keys.append(('SPLIT', 'ascending'))
    yield from helpers.ordered_batches(files, scratch, ranked=False,
                                       sort_keys=keys)


def _encoded_labels(batches, task):
    import pandas as pd

    previous = {}
    for frame in batches:
        if (not pd.api.types.is_integer_dtype(frame.ENTITY1)
                or pd.api.types.is_bool_dtype(frame.ENTITY1)):
            raise ValueError('Training entities must retain integer IDs')
        for row in frame.to_dict('records'):
            split = row['SPLIT']
            if type(split) is not int or split not in STAGES:
                raise ValueError('Training row has an invalid split')
            stage = STAGES[split]
            key = (row['ENTITY1'], int(pd.Timestamp(row['TIMESTAMP']).value)
                   if 'TIMESTAMP' in row else 0)
            if stage in previous and previous[stage] >= key:
                raise ValueError('Duplicate or unordered training stage keys')
            previous[stage] = key
            yield stage, key, (
                json.dumps(_row(row, task), sort_keys=True,
                           separators=(',', ':'), allow_nan=False).encode() +
                b'\n')


def _encode_label_block(source, task, destination):
    """Run in a fresh CPU process without importing the model runtime."""
    import pyarrow as pa

    destination.mkdir()
    with source.open('rb') as stream:
        frame = pa.ipc.open_file(stream).read_all().to_pandas(
            ignore_metadata=True, use_threads=False)
    metadata = {stage: dict(rows=0) for stage in STAGES.values()}
    with ExitStack() as stack:
        streams = {
            stage: stack.enter_context(
                (destination / (stage + '.bin')).open('wb'))
            for stage in STAGES.values()
        }
        for stage, key, payload in _encoded_labels([frame], task):
            item = metadata[stage]
            item.setdefault('first', key)
            item['last'] = key
            item['rows'] += 1
            streams[stage].write(payload)
    (destination / 'metadata.json').write_text(json.dumps(metadata))


def _label_block(source, task):
    destination = source.parent / 'encoded'
    process = subprocess.run([
        sys.executable,
        str(Path(__file__).resolve()), '--encode-label-block',
        str(source), task,
        str(destination)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    if process.returncode:
        raise ValueError('Label encoder failed: ' + process.stderr[-4096:])
    return destination, json.loads((destination / 'metadata.json').read_text())


def label_identities(batches, task, scratch, *, workers=1):
    """Keep the serial hash bytes with bounded, ordered CPU encoding."""
    if type(workers) is not int or not 1 <= workers <= _LABEL_WORKERS:
        raise ValueError('Label encoding needs one to eight CPU workers')
    hashes = {
        stage: hashlib.sha256(b'nightly-logical-labels-v1\n')
        for stage in STAGES.values()
    }
    counts = dict.fromkeys(STAGES.values(), 0)
    if workers == 1:
        for stage, _, payload in _encoded_labels(batches, task):
            hashes[stage].update(payload)
            counts[stage] += 1
    else:
        import pandas as pd
        import pyarrow as pa

        previous = {}

        def consume(future):
            destination, metadata = future.result()
            for stage, item in metadata.items():
                if not item['rows']:
                    continue
                first, last = tuple(item['first']), tuple(item['last'])
                if stage in previous and previous[stage] >= first:
                    raise ValueError(
                        'Duplicate or unordered training stage keys')
                previous[stage] = last
                with (destination / (stage + '.bin')).open('rb') as stream:
                    while block := stream.read(1024**2):
                        hashes[stage].update(block)
                counts[stage] += item['rows']
            shutil.rmtree(destination.parent)

        scratch = Path(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='label-encoding-', dir=scratch) as name, \
                ThreadPoolExecutor(max_workers=workers) as pool:
            pending, index = deque(), 0
            for frame in batches:
                if (not pd.api.types.is_integer_dtype(frame.ENTITY1)
                        or pd.api.types.is_bool_dtype(frame.ENTITY1)):
                    raise ValueError(
                        'Training entities must retain integer IDs')
                table = pa.Table.from_pandas(frame, preserve_index=False)
                offset, size = 0, min(65536, len(table))
                while offset < len(table):
                    block = table.slice(offset, size)
                    while block.nbytes > _LABEL_BLOCK_BYTES:
                        if size == 1:
                            raise ValueError(
                                'One training row exceeds label encoding capacity'
                            )
                        size = max(1, size // 2)
                        block = table.slice(offset, size)
                    directory = Path(name) / str(index)
                    directory.mkdir()
                    source = directory / 'rows.arrow'
                    with source.open('wb') as stream, pa.ipc.new_file(
                            stream, block.schema) as writer:
                        writer.write_table(block)
                    pending.append(pool.submit(_label_block, source, task))
                    if len(pending) >= 2 * workers:
                        consume(pending.popleft())
                    offset += len(block)
                    index += 1
            while pending:
                consume(pending.popleft())
    if not all(counts.values()):
        raise ValueError('Training table needs all three nonempty stages')
    return {
        stage: dict(rows=counts[stage], sha256=hashes[stage].hexdigest())
        for stage in STAGES.values()
    }


def stage_identities(files, task, scratch, *, helpers, ordered=False):
    """Hash every logical row, preserving the original label-hash protocol."""
    import pyarrow.parquet as pq

    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    cpus = len(os.sched_getaffinity(0)) if hasattr(
        os, 'sched_getaffinity') else (os.cpu_count() or 1)
    workers = min(_LABEL_WORKERS, cpus) if rows >= _LABEL_PARALLEL_ROWS else 1
    return label_identities(
        _training_batches(files, scratch, helpers, ordered=ordered), task,
        Path(scratch) / 'labels', workers=workers)


def _ordered_training_files(files, directory, helpers):
    """Retain one case-local sort for hashing and control construction."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory.mkdir(parents=True, exist_ok=False)
    schema = pq.read_schema(files[0]).remove_metadata()
    path = directory / 'rows.parquet'
    with pq.ParquetWriter(path, schema) as writer:
        for frame in _training_batches(files, directory / 'sort', helpers):
            writer.write_table(
                pa.Table.from_pandas(
                    frame, schema=schema,
                    preserve_index=False).replace_schema_metadata(None))
    return [str(path)]


def control_payload(normal, control, task):
    """Apply frozen control answers only after generated input keys match."""
    keys = ['SPLIT', 'ENTITY1'
            ] + (['TIMESTAMP'] if 'TIMESTAMP' in normal else [])
    if set(normal) != set(control):
        raise ValueError('Control columns differ from generated rows')
    left, right = normal.set_index(keys), control.set_index(keys)
    if (left.index.has_duplicates or right.index.has_duplicates
            or set(left.index) != set(right.index)):
        raise ValueError('Control entity, time, or split keys differ')
    aligned = right.loc[left.index].reset_index()
    original = normal.reset_index(drop=True)
    for name in set(normal) - ANSWER_FIELDS:
        if [_value(v)
                for v in original[name]] != [_value(v) for v in aligned[name]]:
            raise ValueError(f'Control changed observed input {name}')
    result = original.copy()
    for name in ANSWER_FIELDS & set(normal):
        result[name] = aligned[name].tolist()
    return result


def prediction_top_k(task, case):
    wanted = case.get('top_k', 10 if task == 'temporal_link' else 5)
    if type(wanted) is not int or wanted < 1:
        raise ValueError('Ranked prediction needs a positive top K')
    return wanted


def prediction_limit(task, case, catalog):
    wanted = prediction_top_k(task, case)
    maximum = max(21, wanted + 1) if task == 'temporal_link' else 500
    requested = case.get('raw_limit', maximum)
    if type(requested) is not int or requested < 1 or not catalog:
        raise ValueError('Ranked prediction needs a positive output budget')
    limit = min(requested, maximum, len(catalog))
    if limit < min(wanted, len(catalog)):
        raise ValueError(
            'Ranked output budget cannot contain the requested top K')
    return limit


def prediction_population(truth, task):
    """Derive prediction inputs from the current case without including answers."""
    columns = ['ENTITY1']
    if task != 'static_link' or 'TIMESTAMP' in truth:
        columns.append('TIMESTAMP')
    if 'autoregression_labels' in truth:
        columns.append('autoregression_labels')
    return truth[columns].sort_values('ENTITY1').reset_index(drop=True)


def static_exclusions(table, stages, truth):
    import pyarrow.parquet as pq

    known = {entity: set() for entity in truth.ENTITY1}
    for stage in ('train', 'val'):
        for name in stages[stage]['parts']:
            with pq.ParquetFile(table.staging_dir / name) as source:
                for batch in source.iter_batches(batch_size=65536,
                                                 columns=['ENTITY1',
                                                          'TARGET1']):
                    for row in batch.to_pylist():
                        if row['ENTITY1'] in known:
                            known[row['ENTITY1']].update(row['TARGET1'])
    result = truth.copy()
    expected = [sorted(known[entity]) for entity in truth.ENTITY1]
    if 'exclude_target_keys' in truth:
        actual = [list(value) for value in truth.exclude_target_keys]
        if any(
                len(value) != len(set(value)) or sorted(value) != wanted
                for value, wanted in zip(actual, expected, strict=True)):
            raise ValueError(
                'Native static exclusions differ from prior-stage targets')
    for targets, excluded in zip(truth.TARGET1, expected, strict=True):
        if set(targets) & set(excluded):
            raise ValueError(
                'Static held-out targets overlap prior-stage targets')
    result['exclude_target_keys'] = expected
    return result


def _export(table, directory):
    import kumo.engine as kumo

    directory.mkdir(parents=True, exist_ok=False)
    table.export(
        kumo.TrainingTableOutputConfig(
            kumo.FileConnector(root_dir=str(directory)), 'rows'))
    files = sorted(
        str(path) for path in (directory / 'rows').rglob('*.parquet'))
    if not files:
        raise ValueError('Training-table export produced no rows')
    return files


def _sorted_rows(files, scratch, helpers, *, ordered=False):
    for frame in _training_batches(files, scratch, helpers, ordered=ordered):
        yield from frame.to_dict('records')


def _controlled_files(normal, control, task, directory, helpers, *,
                      ordered=False):
    import pandas as pd

    directory.mkdir(parents=True, exist_ok=False)
    rows = directory / 'rows'
    rows.mkdir()
    left = _sorted_rows(normal, directory / 'sort-normal', helpers,
                        ordered=ordered)
    right = _sorted_rows(control, directory / 'sort-control', helpers,
                         ordered=ordered)
    paired = zip_longest(left, right)
    index = 0
    while block := list(islice(paired, 65536)):
        if any(a is None or b is None for a, b in block):
            raise ValueError('Control row population differs')
        values = control_payload(pd.DataFrame([a for a, _ in block]),
                                 pd.DataFrame([b for _, b in block]), task)
        if task in ('forecasting', 'temporal_link'):
            values = values.drop(columns='SPLIT')
        values.to_parquet(rows / f'part-{index:05d}.parquet', index=False)
        index += 1
    if not index:
        raise ValueError('Control answer payload is empty')


def normal_case(manifest, case):
    identity = case.get('variant', case['task'])
    normal = next((
        value for value in manifest['cases']
        if value['task'] == case['task'] and value.get('variant', value['task'])
        == identity and value.get('scale', '100M') == case.get('scale', '100M')
        and value['arm'] == 'normal'), None)
    if normal is None:
        raise ValueError('The case needs its matching normal input definition')
    return normal


def generate_training_table(root, manifest, case, query, snapshot, scratch, *,
                            helpers, bootstrap=False, reference_output=None):
    """Record reference answers or compare candidate answers with expectations."""
    if reference_output is not None and not bootstrap:
        raise ValueError('Only reference mode records new reference data')
    from kumo_core.io.json import from_json_dict
    from kumoapi.json_serde import to_json_dict
    from kumoapi.model_plan import TrainingTableGenerationPlan
    from kumoapi.train import TrainingTableSpec

    import kumo.engine as kumo

    normal = normal_case(manifest, case)
    descriptor = helpers.read(
        helpers.safe_file(root, normal['descriptor'], manifest['files']))
    definition = (descriptor['query_generation_plan'] if 'query_generation_plan'
                  in descriptor else descriptor['generation_plan'])
    generation = from_json_dict(definition, TrainingTableGenerationPlan)
    table = query.generate_training_table(snapshot, plan=generation)
    export_root = Path(
        reference_output) if reference_output is not None else scratch
    files = _export(table, export_root / 'generated-normal')
    retained = list(files)
    inline_control = False
    if case['arm'] == 'control':
        files = _ordered_training_files(files, scratch / 'ordered-normal',
                                        helpers)
    normal_identity = stage_identities(files, case['task'],
                                       scratch / 'identity-normal',
                                       helpers=helpers,
                                       ordered=case['arm'] == 'control')
    if not bootstrap:
        expected = normal.get('stage_identity')
        if expected is None and normal.get('normal_rows'):
            names = normal['normal_rows']
            names = [names] if isinstance(names, str) else names
            frozen_files = [
                str(helpers.safe_file(root, name, manifest['files']))
                for name in names
            ]
            expected = stage_identities(frozen_files, case['task'],
                                        scratch / 'identity-frozen-normal',
                                        helpers=helpers)
        if expected is None and (Path(root) / normal['descriptor']
                                 ).parent.joinpath('staging').is_dir():
            frozen, _, _ = helpers.load_table(root, manifest['files'], normal)
            frozen_files = _export(frozen, scratch / 'frozen-normal')
            expected = stage_identities(frozen_files, case['task'],
                                        scratch / 'identity-frozen-normal',
                                        helpers=helpers)
        if expected is None:
            raise ValueError(
                'Candidate generation needs reviewed normal-stage hashes')
        if normal_identity != expected:
            raise ValueError(
                'Candidate-generated normal stage counts or labels differ')
    identity = normal_identity
    if case['arm'] == 'control':
        inline_control = bootstrap or (not case.get('control_rows') and
                                       case.get('control_recipe') is not None)
        if inline_control:
            recipe = case.get('control_recipe')
            if recipe is None:
                raise ValueError(
                    'Reference control needs a pinned control recipe')
            if not bootstrap and case.get('stage_identity') is None:
                raise ValueError(
                    'Candidate control needs reviewed stage hashes')
            controlled = helpers.sibling('nightly_controls').generate(
                files, case['task'], scratch / 'fresh-control', recipe,
                helpers=helpers, ordered=True)
        elif case.get('control_rows'):
            names = case['control_rows']
            names = [names] if isinstance(names, str) else names
            controlled = [
                str(helpers.safe_file(root, name, manifest['files']))
                for name in names
            ]
        else:
            try:
                frozen, _, _ = helpers.load_table(root, manifest['files'], case)
                controlled = _export(frozen, scratch / 'frozen-control')
            except (OSError, KeyError, ValueError) as error:
                raise ValueError(
                    'Control requires a frozen scrambled answer payload'
                ) from error
        if not inline_control:
            controlled = _ordered_training_files(controlled,
                                                 scratch / 'ordered-control',
                                                 helpers)
        expected = None if bootstrap else case.get('stage_identity')
        if expected is None:
            # Reference mode still checks that import preserves its input rows.
            expected = stage_identities(controlled, case['task'],
                                        scratch / 'identity-frozen-control',
                                        helpers=helpers, ordered=True)
        source = scratch / 'controlled-source'
        _controlled_files(files, controlled, case['task'], source, helpers,
                          ordered=True)
        options = dict(allow_new_target_entities=True
                       ) if case['task'] == 'temporal_link' else {}
        table = kumo.TrainingTable.from_source_table(
            kumo.FileConnector(root_dir=str(source))['rows'],
            TrainingTableSpec(split_col=None if case['task'] in (
                'forecasting', 'temporal_link') else 'SPLIT'), pq=query,
            snapshot=snapshot, plan=generation, **options)
        final_files = _export(table, export_root / 'generated-control')
        retained.extend(final_files)
        identity = stage_identities(final_files, case['task'],
                                    scratch / 'identity-control',
                                    helpers=helpers)
        if identity != expected:
            label = ('Reference control import'
                     if bootstrap else 'Candidate control')
            raise ValueError(f'{label} stage counts or labels differ')
    spec = helpers.read(table.staging_dir / 'tt_ingest_spec.json')
    stages = {entry['key']: entry for entry in spec['split_stages']}
    if set(stages) != set(STAGES.values()):
        raise ValueError('Generated table lacks required stages')
    proof = dict(
        normal_stages=normal_identity, stages=identity,
        answer_policy='record_reference' if bootstrap else 'compare_approved',
        query_generation_plan=definition,
        generation_plan=to_json_dict(table.generation_plan))
    if inline_control:
        proof['control_recipe'] = dict(case['control_recipe'])
    if reference_output is not None:
        receipt = dict(
            status='PREPARED', **proof,
            descriptor_sha256=manifest['files'][normal['descriptor']], files={
                str(Path(name).relative_to(export_root)):
                dict(sha256=helpers.digest(Path(name)),
                     bytes=Path(name).stat().st_size)
                for name in retained
            })
        helpers.save(export_root / 'reference-data.json', receipt)
        proof['reference_data'] = 'reference-data/reference-data.json'
    return table, stages, proof


def baseline_score(task, table, stages, truth, classes, catalog, *, helpers,
                   top_k=None):
    """Fit a constant/popularity predictor on training answers only."""
    import numpy as np
    import pyarrow.parquet as pq

    counts = Counter()
    horizons = None
    samples = 0
    for name in stages['train']['parts']:
        with pq.ParquetFile(table.staging_dir / name) as source:
            for batch in source.iter_batches(batch_size=65536):
                frame = batch.to_pandas()
                samples += len(frame)
                if task == 'forecasting':
                    values = np.asarray(frame.future_labels.tolist(),
                                        dtype=float)
                    horizons = horizons or [
                        Counter() for _ in range(values.shape[1])
                    ]
                    for index, counter in enumerate(horizons):
                        keys, number = np.unique(values[:, index],
                                                 return_counts=True)
                        counter.update(
                            dict(zip(keys, map(int, number), strict=True)))
                elif task in ('multilabel', 'multilabel_ranking', 'static_link',
                              'temporal_link'):
                    for value in frame.TARGET1:
                        counts.update(
                            set(
                                value.split('|') if isinstance(value, str
                                                               ) else value))
                else:
                    counts.update(frame.TARGET1.value_counts().to_dict())
    if not samples:
        raise ValueError('Baseline needs nonempty training answers')

    def median(counter):
        total, seen = sum(counter.values()), 0
        middle = [(total - 1) // 2, total // 2]
        selected = []
        for value, number in sorted(counter.items()):
            selected.extend(value for index in middle
                            if seen <= index < seen + number)
            seen += number
        return float(sum(selected) / 2)

    detail = dict(training_rows=samples)
    if task == 'binary':
        constant = counts[True] / samples
        values = np.tile([1 - constant, constant], (len(truth), 1))
        detail.update(method='training-positive-frequency', constant=constant)
    elif task == 'multiclass':
        winner = min(counts, key=lambda key: (-counts[key], key))
        values = np.zeros((len(truth), len(classes)))
        values[:, classes.index(winner)] = 1
        detail.update(method='training-majority-class', constant=_value(winner))
    elif task == 'multilabel':
        values = np.tile([counts[key] / samples for key in classes],
                         (len(truth), 1))
        detail.update(method='training-label-frequencies')
    elif task in ('multilabel_ranking', 'static_link', 'temporal_link'):
        choices = catalog if task.endswith('link') else classes
        ordered = sorted(choices, key=lambda key: (-counts[key], key))
        k = prediction_top_k(task, {} if top_k is None else dict(top_k=top_k))
        excluded = truth.exclude_target_keys if task == 'static_link' else [
            ()
        ] * len(truth)

        def eligible(blocked):
            blocked = set(blocked)
            return list(
                islice((key for key in ordered if key not in blocked), k))

        values = dict(ranked_ids=[eligible(blocked) for blocked in excluded])
        if task == 'multilabel_ranking':
            values = np.tile([counts[key] for key in classes], (len(truth), 1))
        detail.update(method='training-popularity')
    elif task == 'forecasting':
        constant = [median(counter) for counter in horizons]
        values = np.tile(constant, (len(truth), 1))
        detail.update(method='training-median-per-horizon', constant=constant)
    else:
        constant = median(counts)
        values = np.full((len(truth), 1), constant)
        detail.update(method='training-median', constant=constant)
    score = helpers.quality(task, truth, values, classes)
    helpers.validate_quality_measurement(task, score)
    return dict(score=score, **detail)


def write_prediction(path, population, values, decisions, task):
    """Keep aligned replay values in bounded, data-only Parquet batches."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        pa.field('ENTITY1',
                 pa.array(population.ENTITY1).type),
        pa.field('payload', pa.string())
    ])
    with pq.ParquetWriter(path, schema) as writer:
        for start in range(0, len(population), 65536):
            end = min(start + 65536, len(population))
            payload = []
            for index in range(start, end):
                if task.endswith('link'):
                    row = {
                        key: values[key][index]
                        for key in ('ranked_ids', 'ranked_scores',
                                    'boundary_ids', 'boundary_scores')
                    }
                else:
                    row = dict(
                        values=values[index], decisions=None
                        if decisions is None else decisions[index])
                payload.append(
                    json.dumps(_value(row), sort_keys=True,
                               separators=(',', ':'), allow_nan=False))
            writer.write_table(
                pa.Table.from_arrays([
                    pa.array(population.ENTITY1.iloc[start:end]),
                    pa.array(payload)
                ], schema=schema))


def read_prediction(path, task):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    ids, rows = [], []
    with pq.ParquetFile(path) as source:
        if source.schema_arrow.names != ['ENTITY1', 'payload']:
            raise ValueError('Canonical prediction schema differs')
        for batch in source.iter_batches(batch_size=65536):
            frame = batch.to_pandas()
            if (not pd.api.types.is_integer_dtype(frame.ENTITY1)
                    or pd.api.types.is_bool_dtype(frame.ENTITY1)):
                raise ValueError('Replay entity IDs changed their type')
            ids.extend(frame.ENTITY1.tolist())
            rows.extend(json.loads(value) for value in frame.payload)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Canonical predictions need unique entities')
    if task.endswith('link'):
        values = {
            key: [row[key] for row in rows]
            for key in ('ranked_ids', 'ranked_scores', 'boundary_ids',
                        'boundary_scores')
        }
        return values, values['ranked_ids'], ids
    values = np.asarray([row['values'] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Canonical predictions contain nonfinite scores')
    decisions = None if rows[0]['decisions'] is None else np.asarray(
        [row['decisions'] for row in rows])
    return values, decisions, ids


def compare_values(task, before, after, tolerance, *, helpers,
                   before_decisions=None, after_decisions=None, positions=None,
                   measure_only=False):
    import numpy as np

    if tolerance.get('rtol', 0) != 0:
        raise ValueError(
            'Prediction comparisons require zero relative tolerance')
    atol = tolerance.get('atol')
    pending = atol is None and measure_only
    if pending:
        atol = 0.
    if type(atol) not in (int, float) or not math.isfinite(atol) or atol < 0:
        raise ValueError(
            'Candidate predictions need a reviewed finite tolerance')
    if task.endswith('link'):
        if positions is not None:
            before = {
                key: [before[key][index] for index in positions]
                for key in ('ranked_ids', 'ranked_scores', 'boundary_ids',
                            'boundary_scores')
            }
        result = helpers.ranked_difference(before, after, atol=atol,
                                           measure_only=measure_only)
    else:
        before = np.asarray(before)
        if positions is not None:
            before = before[positions]
            if before_decisions is not None:
                before_decisions = np.asarray(before_decisions)[positions]
        result = helpers.smoke.compare_predictions(
            before, after, task=task, atol=atol,
            before_decisions=before_decisions, after_decisions=after_decisions,
            measure_only=measure_only)
    if pending:
        result.update(atol=None, within_tolerance=None,
                      calibration='PENDING_REVIEW')
        if not result.get('unproven_membership_changes', 0):
            result['status'] = 'MEASURED'
    return result


def replay_reference(replay, root, predict, state, tolerance, *, task, classes,
                     helpers, output, population):
    import pandas as pd

    import kumo.engine as kumo

    if root is None:
        raise ValueError(
            'Reference replay requires the verified evidence archive')
    root = Path(root)
    model = replay['model']
    paths = {
        str(Path(model['path']) / name): expected
        for name, expected in model['files'].items()
    }
    for name in paths:
        helpers.safe_file(root, name, paths)
    directory = root / model['path']
    actual = {
        str(path.relative_to(root))
        for path in directory.rglob('*') if path.is_file()
    }
    if actual != set(paths) or any(path.is_symlink()
                                   for path in directory.rglob('*')):
        raise ValueError('Reference model contains unbound or linked files')

    def artifact(record):
        return helpers.safe_file(root, record['path'],
                                 {record['path']: record['sha256']})

    inputs = pd.read_parquet(artifact(replay['population']))
    pd.testing.assert_frame_equal(
        inputs.set_index('ENTITY1').sort_index(),
        population.set_index('ENTITY1').sort_index())
    before, decisions, ids = read_prediction(artifact(replay['predictions']),
                                             task)
    if inputs.ENTITY1.tolist() != ids:
        raise ValueError(
            'Reference replay population differs from stored predictions')
    state.update(replay_checkpoint_sha256=replay['checkpoint_sha256'],
                 replay_checks=0, replay_restored_tensors=0)
    state['prediction_batches']['replay'] = 0
    model = kumo.Trainer.load(directory)
    after, after_decisions = predict(model, 'replay', inputs)
    if state['replay_checks'] < 1 or state['replay_restored_tensors'] <= 0:
        raise ValueError(
            'Reference replay lacks an exact checkpoint restoration check')
    result = compare_values(task, before, after, tolerance, helpers=helpers,
                            before_decisions=decisions,
                            after_decisions=after_decisions)
    helpers.save(output / 'reference-replay.json', result)
    return result


@contextmanager
def observe_cuda(state, *, helpers):
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
                checkpoint = getattr(trainer, 'checkpoint_callback', None)
                directory = getattr(checkpoint, 'dirpath', None)
                refit = (getattr(checkpoint, 'save_last', False)
                         and not getattr(checkpoint, 'monitor', None))
                self.candidate = dict(
                    trial_id=str(directory) if directory else None,
                    kind='refit' if refit else 'candidate', status='RUNNING',
                    steps=0, cuda_batches=0, epochs=[], validation=[])
                state['fit_ledger'].append(self.candidate)

        def on_train_epoch_start(self, trainer, module):
            if 'training_epochs' in state:
                epoch = dict(epoch=int(trainer.current_epoch), batches=0,
                             completed=False, batch_sizes=[])
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

        def on_train_epoch_end(self, trainer, module):
            if getattr(self, 'candidate', None) and self.candidate['epochs']:
                self.candidate['epochs'][-1]['completed'] = True

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
                self.candidate['checkpoint_sha256'] = helpers.digest(
                    local_path(str(trainer.ckpt_path)))

        def on_fit_end(self, trainer, module):
            if 'fit_ledger' in state:
                self.candidate.update(status='PASS',
                                      steps=int(trainer.global_step))
                checkpoint = trainer.checkpoint_callback.best_model_path
                checkpoint_digest = helpers.digest(checkpoint)
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
            if state['stage'] == 'replay':
                state['replay_restored_tensors'] = helpers.verify_checkpoint(
                    module, trainer.ckpt_path,
                    state['replay_checkpoint_sha256'])
                state['replay_checks'] += 1
            if state['stage'] == 'reload' and 'reload_checks' in state:
                state['restored_tensors'] = helpers.verify_checkpoint(
                    module, trainer.ckpt_path, state['checkpoint_sha256'])
                state['reload_checks'] += 1

    with helpers.smoke.observe_cuda(state):
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


def run_case(root, manifest, case, graph, snapshot, output, scratch, tolerance,
             *, helpers, measure_only=False, reference=None,
             reference_root=None, reference_destination=None,
             reference_context=None):
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
    report = dict(task=task, scale=case.get('scale', '100M'), arm=case['arm'],
                  seed=case.get('seed', 0), status='RUNNING', work=state)
    if case.get('variant'):
        report['variant'] = case['variant']
    try:
        if not measure_only and (type(tolerance.get('atol')) not in (int, float)
                                 or not math.isfinite(tolerance['atol'])
                                 or tolerance['atol'] < 0
                                 or tolerance.get('rtol', 0) != 0):
            raise ValueError(
                'Candidate predictions need a reviewed finite tolerance')
        with helpers.measure_operation(report, 'preparation'):
            work = scratch / output.name
            kumo.configure(scratch_dir=work)
            random.seed(case.get('seed', 0))
            np.random.seed(case.get('seed', 0))
            torch.manual_seed(case.get('seed', 0))
            descriptor = helpers.read(
                helpers.safe_file(root, case['descriptor'], manifest['files']))
            query = kumo.PredictiveQuery(graph, descriptor['query'])
            query.validate()
            if query.get_task_type(
            ).value != helpers.smoke.TASKS[task]['task_type']:
                raise ValueError(
                    'The public query resolves to a different task')
            expected_case = copy.deepcopy(case)
            if not measure_only and reference and reference.get(
                    'training_table'):
                expected_case['stage_identity'] = reference['training_table'][
                    'stages']
                normal_case(manifest, case)['stage_identity'] = reference[
                    'training_table']['normal_stages']
            table, stages, proof = generate_training_table(
                root, manifest, expected_case, query, snapshot,
                work / 'training-table', helpers=helpers,
                bootstrap=measure_only, reference_output=output /
                'reference-data' if measure_only else None)
            report['training_table'] = proof
            helpers.save(output / 'training-table-identity.json', proof)
            truth = helpers.stage_frame(table, stages['test'])
            if task == 'static_link':
                truth = static_exclusions(table, stages, truth)
            generated_population = measure_only or 'prediction_input' not in case
            if generated_population:
                full = prediction_population(truth, task)
                population_path = output / 'current-population.parquet'
                full.to_parquet(population_path, index=False)
            else:
                full = pd.read_parquet(
                    helpers.safe_file(root, case['prediction_input'],
                                      manifest['files']))
            ids = helpers.check_inputs(full, truth,
                                       static=task == 'static_link')
            populations = sample_populations(full, CAPS['sample_seed'],
                                             CAPS['sample_entities'],
                                             ordered=generated_population)
            consistency, partial = populations['consistency'], populations[
                'partial']
            if populations['proper_subset']:
                helpers.check_inputs(partial, truth, partial=True,
                                     static=task == 'static_link')
            truth = truth.set_index('ENTITY1').loc[ids].reset_index()
            if task == 'forecasting' and any(
                    len(v) != 12
                    for col in ('autoregression_labels', 'future_labels')
                    for v in truth[col]):
                raise ValueError(
                    'Forecast inputs/answers require all twelve coordinates')
            plan = capped_plan(query, snapshot, min_hops=case.get('min_hops'),
                               max_hops=case.get('max_hops'))
            if case.get('min_hops') is not None:
                report['graph_depth'] = dict(
                    min_hops=case['min_hops'], max_hops=case['max_hops'],
                    configured_hops=neighbor_depths(plan))
            helpers.save(output / 'requested-plan.json', plan.as_json())
            state['requested_batch_sizes'] = plan.as_json(
            )['optimization']['batch_size']
            report['training_recipe'] = 'capped-nightly-v1'
            report['requested_plan_sha256'] = helpers.digest(
                output / 'requested-plan.json')
            report['plan_sha256'] = report['requested_plan_sha256']
            report['selection_metric'] = plan.as_json(
            )['training_job']['tune_metric']
            helpers.save(output / 'generation-plan.json',
                         proof['generation_plan'])
            report['input_signature'] = helpers.input_signature(
                manifest, case, prediction=helpers.digest(population_path)
                if generated_population else None)
            classes, catalog = case.get('classes', []), []
            if task.endswith('link'):
                spec = case['catalog']
                catalog = pd.read_parquet(
                    helpers.safe_file(root, spec['path'], manifest['files']),
                    columns=[spec['column']])[spec['column']].tolist()
                if not catalog or len(set(catalog)) != len(catalog):
                    raise ValueError(
                        'Link catalog keys are missing or duplicated')
                report['top_k'] = prediction_top_k(task, case)
                report['raw_limit'] = prediction_limit(task, case, catalog)
            anchor = None if task == 'static_link' else full.TIMESTAMP.iloc[0]
            if anchor is not None and not full.TIMESTAMP.eq(anchor).all():
                raise ValueError(
                    'Full prediction needs the fixed frozen test anchor')
            if measure_only or reference is None:
                baseline = baseline_score(task, table, stages, truth, classes,
                                          catalog, helpers=helpers,
                                          top_k=case.get('top_k'))
                report['baseline'], report['baseline_evidence'] = baseline[
                    'score'], baseline
            else:
                report['baseline'] = reference['baseline']
            helpers.save(
                output / 'sample-entities.json',
                dict(ids=consistency.ENTITY1.tolist(), seed=CAPS['sample_seed'],
                     proper_subset=populations['proper_subset']))
            report['sample_sha256'] = helpers.digest(output /
                                                     'sample-entities.json')
        if measure_only and reference_destination is not None:
            helpers.save(
                output / 'reference-context.json',
                dict(case=case, bindings=reference_context or {},
                     input_signature=report['input_signature']))
            with helpers.measure_operation(report, 'reference_publication'):
                report['reference_publication'] = helpers.sibling(
                    'nightly_evidence').publish(output, reference_destination)
                helpers.save(output / 'reference-publication.json',
                             report['reference_publication'])
        with observe_cuda(state, helpers=helpers):
            time_column = case.get('prediction_time_column', 'TIMESTAMP')

            def prediction_output(model, stage, frame, label, limit=None):
                torch.cuda.synchronize()
                state['stage'] = stage
                before = state['prediction_batches'][stage]
                inputs = output / 'prediction-inputs' / (label + '.parquet')
                inputs.parent.mkdir(exist_ok=True)
                frame.to_parquet(inputs, index=False)
                pt = kumo.PredictionTable.from_source_table(
                    kumo.FileConnector(
                        root_dir=str(inputs.parent))[inputs.name], pq=query,
                    snapshot=snapshot,
                    plan=model.training_table_generation_plan)
                options = dict(output_config=kumo.PredictionOutputConfig(
                    kumo.FileConnector(root_dir=str(output /
                                                    'prediction-outputs')),
                    output_paths={'predictions': label}, extra_fields=[] if task
                    == 'static_link' else [MetadataField.ANCHOR_TIMESTAMP]))
                if task.endswith('link'):
                    options['num_classes_to_return'] = limit
                elif classes:
                    options['num_classes_to_return'] = len(classes)
                if task == 'binary':
                    options['binary_classification_threshold'] = .5
                result = model.predict(pt, snapshot=snapshot, **options)
                torch.cuda.synchronize()
                if state['prediction_batches'][stage] <= before:
                    raise ValueError(
                        'Prediction completed no actual CUDA batches')
                raw = output / 'prediction-outputs' / label
                raw.mkdir(parents=True, exist_ok=True)
                retained = []
                for index, source in enumerate(result.files):
                    source = local_path(source)
                    destination = (Path(source)
                                   if Path(source).resolve().is_relative_to(
                                       raw.resolve()) else raw /
                                   f'raw-{index}.parquet')
                    if Path(source).resolve() != destination.resolve():
                        shutil.copyfile(local_path(source), destination)
                    retained.append(str(destination.relative_to(output)))
                return result, inputs, retained

            def predict(model, stage, frame, *, name=None):
                label = name or ('reloaded_prediction'
                                 if stage == 'reload' else stage)
                with helpers.measure_operation(report, label):
                    frame_input = frame
                    selected = frame.ENTITY1.tolist()
                    result, inputs, retained = prediction_output(
                        model, stage, frame, label, report.get('raw_limit'))
                    original_files = list(result.files)
                    retries = []
                    if task.endswith('link'):
                        selected_truth = truth.set_index(
                            'ENTITY1').loc[selected]
                        excluded = selected_truth.exclude_target_keys.tolist(
                        ) if task == 'static_link' else [[] for _ in selected]
                        retry_index = 0

                        def retry_ranked(ids, limit):
                            nonlocal retry_index
                            retry_index += 1
                            retry_label = f'{label}-cutoff-{retry_index}'
                            subset = frame_input.set_index(
                                'ENTITY1').loc[ids].reset_index()
                            extra, _, files = prediction_output(
                                model, stage, subset, retry_label, limit)
                            retained.extend(files)
                            original_files.extend(extra.files)
                            return helpers.ordered_batches(
                                extra.files, work / 'sort' / retry_label)

                        values = helpers.ranked_rows(
                            helpers.ordered_batches(result.files,
                                                    work / 'sort' / label),
                            selected, catalog, excluded, report['top_k'],
                            report['raw_limit'], anchor=anchor,
                            retry=retry_ranked)
                        retries = values['cutoff_retries']
                        helpers.save(output / (label + '-ranked.json'), values)
                    else:
                        frame = result.frame
                        if frame.ENTITY.dtype != full.ENTITY1.dtype:
                            raise ValueError(
                                'Prediction changed the original entity ID type'
                            )
                        frame.to_parquet(output / (label + '.parquet'),
                                         index=False)
                        values = helpers.smoke.prediction_values(
                            task, frame, selected, classes=classes,
                            anchor=anchor, time_column=time_column
                            or 'TIMESTAMP')
                    decisions = (values['ranked_ids'] if task.endswith('link')
                                 else helpers.smoke.prediction_decisions(
                                     task, frame, selected, classes=classes))
                    report.setdefault('predictions', {})[label] = dict(
                        stage=stage, entities=len(selected),
                        rows=values['raw_rows']
                        if task.endswith('link') else result.num_rows,
                        files=retained, original_files=original_files,
                        input_sha256=helpers.digest(inputs),
                        cutoff_retries=retries)
                    if stage == 'predict':
                        write_prediction(output / 'full-predictions.parquet',
                                         frame_input, values, decisions, task)
                    return values, decisions

            replay = case.get('replay')
            report['reference_replay'] = {}

            def reference_work():
                if case.get('replay_error'):
                    return dict(status='FAIL', error=case['replay_error'])
                return replay_reference(replay, reference_root, predict, state,
                                        tolerance, task=task, classes=classes,
                                        helpers=helpers, output=output,
                                        population=full)

            def fresh_fit():
                state['stage'] = 'fit'
                model = kumo.Trainer(plan)
                torch.cuda.synchronize()
                with helpers.measure_operation(report, 'training'):
                    fitted = model.fit(table, snapshot=snapshot)
                    torch.cuda.synchronize()
                return model, fitted

            (trainer, fitted), _ = with_reference_replay(
                reference_work if replay is not None or case.get('replay_error')
                else None, fresh_fit, evidence=report['reference_replay'])
            helpers.save(output / 'candidate-validation.json',
                         state['fit_ledger'])
            helpers.save(output / 'fit-metrics.json', fitted.metrics)
            checkpoint = local_path(
                str(fitted.artifact_paths['selected_checkpoint']))
            state['checkpoint_sha256'] = helpers.digest(checkpoint)
            selected_steps = load_kumo_checkpoint(
                str(checkpoint), map_location='cpu')['global_step']
            if query.get_task_type(
            ).value != helpers.smoke.TASKS[task]['task_type']:
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
                p for p in saved.rglob('*') if p.is_file()
                and helpers.digest(p) == state['checkpoint_sha256']
            ]
            if not checkpoints:
                raise ValueError(
                    'Published model does not retain the selected checkpoint bytes'
                )
            report.update(
                fit_id=str(uuid.uuid4()) + ':' + str(fitted.bundle_dir),
                checkpoint_sha256=helpers.digest(checkpoint),
                checkpoint_artifact=str(checkpoints[0].relative_to(output)),
                selected_checkpoint_steps=selected_steps)
            if 'graph_depth' in report:
                from kumo_core.io.json import load_json
                from kumo_core.model_plan import ModelPlan

                resolved_path = output / 'resolved_model_plan.json'
                if not resolved_path.is_file():
                    raise ValueError(
                        'Deep cases need the resolved graph hop artifact')
                depths = neighbor_depths(load_json(resolved_path, ModelPlan))
                report['graph_depth']['resolved_hops'] = depths
                if any(not case['min_hops'] <= depth <= case['max_hops']
                       for depth in depths):
                    raise ValueError(
                        'Selected resolved graph hops violate case bounds')
            helpers.check_training(state, selected_steps)
            original, original_decisions = predict(trainer, 'predict', full)
            report.update(
                score=helpers.quality(task, truth, original, classes),
                metric=case.get('metric', helpers.smoke.TASKS[task]['metric']))
            helpers.save(output / 'result.json', report)
            sample, sample_decisions = predict(trainer, 'partial', consistency,
                                               name='sample')
            repeated, repeat_decisions = predict(trainer, 'repeat', consistency)
            if len(consistency) == len(full) and populations['proper_subset']:
                subset, subset_decisions = predict(trainer, 'partial', partial,
                                                   name='proper-subset')
            else:
                subset, subset_decisions = sample, sample_decisions
            del trainer
            gc.collect()
            with helpers.measure_operation(report, 'model_load'):
                trainer = kumo.Trainer.load(saved)
                torch.cuda.synchronize()
            reloaded, reload_decisions = predict(trainer, 'reload', consistency)
            if state['reload_checks'] < 1 or state['restored_tensors'] <= 0:
                raise ValueError(
                    'Reload lacks an exact restored checkpoint check')
            lookup = {v: i for i, v in enumerate(ids)}
            sample_positions = [lookup[v] for v in consistency.ENTITY1]
            positions = [lookup[v] for v in partial.ENTITY1]
            report['differences'] = {}
            for label, before, other, before_decisions, decisions, selected in [
                ('sample', original, sample, original_decisions,
                 sample_decisions, sample_positions),
                ('repeat', sample, repeated, sample_decisions, repeat_decisions,
                 None),
                ('partial', original, subset, original_decisions,
                 subset_decisions, positions),
                ('reload', sample, reloaded, sample_decisions, reload_decisions,
                 None)
            ]:
                if label == 'partial' and not populations['proper_subset']:
                    report['differences'][label] = dict(
                        status='FAIL',
                        reason='A nonempty proper subset cannot be exercised')
                    continue
                report['differences'][label] = compare_values(
                    task, before, other, tolerance, helpers=helpers,
                    before_decisions=before_decisions,
                    after_decisions=decisions, positions=selected,
                    measure_only=measure_only)
            outcomes = {
                value['status']
                for value in report['differences'].values()
            }
            report['prediction_checks'] = dict(
                status='FAIL' if 'FAIL' in
                outcomes else 'MEASURED' if 'MEASURED' in outcomes else 'PASS')
            report['training_coverage'] = check_training_coverage(
                state['fit_ledger'])
            report['model'] = dict(
                path='published-model', files={
                    str(path.relative_to(saved)): helpers.digest(path)
                    for path in saved.rglob('*') if path.is_file()
                })
            report['population'] = dict(
                path='prediction-inputs/predict.parquet', sha256=helpers.digest(
                    output / 'prediction-inputs/predict.parquet'))
            report['canonical_predictions'] = dict(
                path='full-predictions.parquet',
                sha256=helpers.digest(output / 'full-predictions.parquet'))
            report.update(status='MEASURED', physical_fits=state['fits'],
                          prediction_batches=state['prediction_batches'],
                          policy='measurement' if measure_only else 'enforced')
            helpers.check_training(state, selected_steps)
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
                        path.relative_to(output))] = helpers.digest(path)
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
            helpers.save(output / 'result.json', report)

    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Encode one bounded CPU label block')
    parser.add_argument('--encode-label-block', type=Path, required=True)
    parser.add_argument('task')
    parser.add_argument('destination', type=Path)
    arguments = parser.parse_args()
    _encode_label_block(arguments.encode_label_block, arguments.task,
                        arguments.destination)
