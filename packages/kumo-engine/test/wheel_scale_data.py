# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare existing scale-bench graphs and audit their real event targets."""

import copy
import hashlib
import importlib.util
import json
import random
import runpy
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

DELIVERY_FILTER = ' WHERE COUNT(orders.*, -90, 0, days) > 0'
CASES = {
    'binary':
    dict(
        graph='adtech', source_task='user_converts',
        query=('PREDICT COUNT(conversions.*, 0, 7, days) > 0 '
               'FOR EACH users.user_id '
               'WHERE COUNT(impressions.*, -14, 0, days) > 0'),
        task_type='binary_classification', metric='auroc', days=7),
    'multiclass':
    dict(
        graph='delivery', source_task='daypart',
        query=('PREDICT FIRST(orders.daypart, 0, 14, days) '
               'FOR EACH users.user_id' + DELIVERY_FILTER),
        task_type='multiclass_classification', metric='accuracy', days=14),
    'multilabel':
    dict(
        graph='delivery', source_task='daypart',
        query=('PREDICT LIST_DISTINCT(orders.daypart, 0, 14, days) '
               'CLASSIFY TOP 6 FOR EACH users.user_id' + DELIVERY_FILTER),
        task_type='multilabel_classification', metric='ap_macro', days=14),
    'multilabel_ranking':
    dict(
        graph='delivery', source_task='daypart',
        query=('PREDICT LIST_DISTINCT(orders.daypart, 0, 14, days) '
               'RANK TOP 5 FOR EACH users.user_id' + DELIVERY_FILTER),
        task_type='multilabel_ranking', metric='recall@5', days=14, top_k=5),
    'regression':
    dict(
        graph='delivery', source_task='orders_30d',
        query=('PREDICT COUNT(orders.*, 0, 30, days) '
               'FOR EACH users.user_id' + DELIVERY_FILTER),
        task_type='regression', metric='mae', days=30),
    'forecasting':
    dict(
        graph='delivery', source_task='orders_30d',
        query=('PREDICT COUNT(orders.*, 0, 7, days) '
               'FORECAST 12 TIMEFRAMES FOR EACH users.user_id'),
        task_type='forecasting', metric='mae', days=84, step_days=7),
    'static_link':
    dict(
        graph='commerce', source_task='next_merchants',
        query=('PREDICT LIST_DISTINCT(purchases.merchant_id) '
               'RANK TOP 5 FOR EACH customers.customer_id'),
        task_type='static_link_prediction', metric='recall@5', days=0, top_k=5),
    'temporal_link':
    dict(
        graph='commerce', source_task='next_merchants',
        query=('PREDICT LIST_DISTINCT(purchases.merchant_id, 0, 7, days) '
               'RANK TOP 10 FOR EACH customers.customer_id'),
        task_type='temporal_link_prediction', metric='recall@10', days=7,
        top_k=10),
}
_GRAPHS = {}
_PREPARED = {}


def read_profile(profile_dir, case):
    """Require the validated manifest and its unchanged source files."""
    profile_dir = Path(profile_dir).resolve()
    content = (profile_dir / 'manifest.json').read_bytes()
    manifest = json.loads(content)
    validation = json.loads((profile_dir / '_VALIDATED.json').read_text())
    digest = hashlib.sha256(content).hexdigest()
    if (manifest['graph'] != CASES[case]['graph']
            or validation.get('ok') is not True
            or validation.get('checksums_verified') is not True
            or validation.get('manifest_sha256') != digest):
        raise ValueError('Profile validation or manifest fingerprint differs')
    root = profile_dir.parent.parent
    for table in manifest['tables'].values():
        for entry in table['files']:
            path = (root / entry['path']).resolve()
            if not path.is_relative_to(profile_dir):
                raise ValueError('Profile file escapes the dataset directory')
            if not path.is_file() or path.stat().st_size != entry['bytes']:
                raise ValueError(
                    'Source file fingerprint differs from manifest')
            with path.open('rb') as stream:
                if hashlib.file_digest(stream,
                                       'sha256').hexdigest() != entry['sha256']:
                    raise ValueError(
                        'Source file fingerprint differs from manifest')
    return manifest


def generation_ranges(manifest, case):
    """Reuse native training history and reserve complete held-out windows."""
    window = manifest['time_window']
    start, end = [
        datetime.fromisoformat(window[name].replace('Z', '+00:00')).astimezone(
            timezone.utc).replace(tzinfo=None) for name in ('start', 'end')
    ]
    horizon = timedelta(days=CASES[case]['days'])
    if not horizon or end - 4 * horizon < start:
        raise ValueError(
            'Dataset time window cannot hold complete stage windows')
    return [((start if offset == 3 else end - offset * horizon).isoformat(),
             (end - (offset - 1) * horizon).isoformat())
            for offset in (3, 2, 1)]


def validate_chronology(case, manifest, rows):
    """Require each temporal answer window to stay inside its staged period."""
    if case == 'static_link':
        return
    ranges = [
        tuple(map(datetime.fromisoformat, bounds))
        for bounds in generation_ranges(manifest, case)
    ]
    horizon = timedelta(days=CASES[case]['days'])
    seen = set()
    for row in rows:
        stage, anchor = row.get('SPLIT'), row.get('TIMESTAMP')
        if (type(stage) is not int or stage not in range(3)
                or not isinstance(anchor, datetime)):
            raise ValueError('Temporal row lacks a valid stage or anchor')
        start, end = ranges[stage]
        if not start <= anchor or anchor + horizon > end:
            raise ValueError('Temporal answer window crosses its stage bounds')
        seen.add(stage)
    if seen != {0, 1, 2}:
        raise ValueError(
            'Temporal training, validation or evaluation stage is empty')


def _members(value):
    return set(filter(None, value.split('|'))) if isinstance(
        value, str) else set(value)


def audit_targets(case, source_dir, rows):
    """Compare native targets with independent SQL over source Parquet."""
    import duckdb
    import pyarrow as pa

    if not rows:
        raise ValueError('No staged rows to audit')
    connection = duckdb.connect()
    table, key, stamp = (('conversions', 'user_id',
                          'conversion_time') if case == 'binary' else
                         ('purchases', 'customer_id',
                          'purchase_time') if case.endswith('link') else
                         ('orders', 'user_id', 'order_time'))
    path = str(Path(source_dir) / table / '*.parquet').replace("'", "''")
    seeds = pa.table(
        dict(row_id=list(range(len(rows))),
             ENTITY1=[r['ENTITY1'] for r in rows],
             TIMESTAMP=[r.get('TIMESTAMP', datetime(1970, 1, 1))
                        for r in rows]))
    connection.register('seeds', seeds)
    connection.execute(
        f"CREATE VIEW events AS SELECT * FROM read_parquet('{path}')")
    try:
        if case == 'static_link':
            expected = {}
            for entity, item in connection.execute(
                    'SELECT DISTINCT customer_id, merchant_id FROM events'
            ).fetchall():
                expected.setdefault(entity, set()).add(item)
            supplied = {}
            for row in rows:
                targets = _members(row['TARGET1'])
                if not targets <= expected.get(row['ENTITY1'], set()):
                    raise ValueError('Static targets differ from source events')
                supplied.setdefault(row['ENTITY1'], set()).update(targets)
            if supplied != expected:
                raise ValueError(
                    'Static target population differs from source events')
            return
        if case == 'forecasting':

            def counts(first):
                query = (
                    'SELECT row_id, list(n ORDER BY period) FROM ('
                    'SELECT s.row_id, p.range AS period, count(e.user_id) AS n '
                    'FROM seeds s CROSS JOIN range(12) p LEFT JOIN events e '
                    'ON e.user_id = s.ENTITY1 AND e.order_time > '
                    f"s.TIMESTAMP + (p.range + {first}) * INTERVAL 7 DAY "
                    'AND e.order_time <= s.TIMESTAMP + '
                    f"(p.range + {first + 1}) * INTERVAL 7 DAY "
                    'GROUP BY s.row_id, p.range) GROUP BY row_id')
                return dict(connection.execute(query).fetchall())

            future, history = counts(0), counts(-12)
            for index, row in enumerate(rows):
                if row.get('autoregression_labels') != history[index]:
                    raise ValueError(
                        'Forecast history differs from source events')
                if (row.get('future_labels') != future[index]
                        or row['TARGET1'] != future[index][0]):
                    raise ValueError(
                        'Forecast targets differ from source events')
            return
        aggregate = (
            'count(e.user_id) > 0' if case == 'binary' else
            'first(e.daypart ORDER BY e.order_time, e.order_id)' if case
            == 'multiclass' else 'count(e.user_id)' if case == 'regression' else
            'list(DISTINCT e.merchant_id) FILTER (WHERE e.merchant_id IS NOT NULL)'
            if case == 'temporal_link' else
            'list(DISTINCT e.daypart) FILTER (WHERE e.daypart IS NOT NULL)')
        days = CASES[case]['days']
        expected = dict(
            connection.execute(
                f'SELECT s.row_id, {aggregate} FROM seeds s LEFT JOIN events e '
                f'ON e.{key} = s.ENTITY1 AND e.{stamp} > s.TIMESTAMP '
                f'AND e.{stamp} <= s.TIMESTAMP + INTERVAL {days} DAY '
                'GROUP BY s.row_id').fetchall())
        for index, row in enumerate(rows):
            value, target = expected[index], row['TARGET1']
            equal = (_members(value or []) == _members(target)
                     if case in ('multilabel', 'multilabel_ranking',
                                 'temporal_link') else value == target)
            if not equal:
                raise ValueError('Targets differ from source events')
    finally:
        connection.close()


def shuffle_targets(case, rows, seed):
    """Keep observed inputs fixed while permuting complete answer payloads."""
    if case == 'static_link':
        raise NotImplementedError(
            'A static-link control needs native exclusion-preserving proof')
    result = copy.deepcopy(rows)
    randomizer = random.Random(seed)
    fields = ('TARGET1', 'future_labels', '__kumo_internal_frequency_column__')
    groups = {}
    for index, row in enumerate(rows):
        anchor = row['TIMESTAMP'] if case == 'temporal_link' else None
        groups.setdefault((row['SPLIT'], anchor), []).append(index)
    for key in sorted(groups):
        selected = sorted(
            groups[key],
            key=lambda index: (rows[index]['ENTITY1'], rows[index].get(
                'TIMESTAMP', datetime.min)))
        shuffled = selected[:]
        randomizer.shuffle(shuffled)
        for destination, source in zip(selected, shuffled):
            for name in fields:
                if name in rows[source]:
                    result[destination][name] = copy.deepcopy(
                        rows[source][name])
    return result


def audit_temporal_link_targets(source_dir, rows, target_table, definition):
    """Require every RHS key to exist and be valid in the registered window."""
    import duckdb
    import pyarrow as pa

    answers = [
        dict(anchor=row['TIMESTAMP'], target=target) for row in rows
        for target in row['TARGET1']
    ]
    if not answers:
        raise ValueError('No temporal targets to audit against their window')

    def quote(name):
        return 'm."' + name.replace('"', '""') + '"'

    invalid = [quote(definition.pkey) + ' IS NULL']
    if definition.time_col:
        invalid.append(
            quote(definition.time_col) +
            f' > a.anchor + INTERVAL {CASES["temporal_link"]["days"]} DAY')
    if definition.end_time_col:
        invalid.append(quote(definition.end_time_col) + ' <= a.anchor')
    path = str(Path(source_dir) / target_table / '*.parquet').replace("'", "''")
    with duckdb.connect() as connection:
        connection.register('answers', pa.Table.from_pylist(answers))
        failures = connection.execute(
            f"SELECT count(*) FROM answers a LEFT JOIN read_parquet('{path}') m "
            f'ON a.target = {quote(definition.pkey)} WHERE ' +
            ' OR '.join(invalid)).fetchone()[0]
    if failures:
        raise ValueError(
            'Temporal target keys or validity differ from the query window')


def _verify_link_resolution(prepared, table, rows, tt_id, output_dir,
                            entity_table, entity_key, target_table, target_key):
    import diskgraph
    import pyarrow as pa
    from kumo_core.diskgraph.lightning_data import _index_np_time_unit
    from kumo_core.diskgraph.runtime_adapter import engine_int64_to_ns

    case = prepared['case']
    static = case == 'static_link'
    index = str(prepared['snapshot'].index_path)
    result = diskgraph.execute_tt_ingest(staging_dir=str(table.staging_dir),
                                         index_path=index, tt_id=tt_id)
    if any(result['key_failures'][name] != 0
           for name in ('parse_failures', 'lookup_failures',
                        'membership_failures')):
        raise ValueError('Temporal control ingest has unresolved native keys')
    sampler = diskgraph.SamplerEngine(index, num_threads=1)
    mappings = {}
    for name, key in ((entity_table, entity_key), (target_table, target_key)):
        ids = list(range(prepared['graph_counts'][name]))
        batches = [
            event['batch'] for event in sampler.fetch_features(
                [dict(node_type=name, node_ids=ids, feature_names=[key])])
            if event['type'] == 'feature'
        ]
        values = pa.Table.from_batches(batches)[key].to_pylist()
        mappings[name] = dict(zip(ids, values, strict=True))
    manifest = Path(output_dir) / 'control-runtime.json'
    manifest.write_text(
        json.dumps(
            dict(
                kind='diskgraph_task_dataset', graph=dict(index_path=index),
                stages=result['task_dataset']['stages'],
                task=dict(is_temporal=not static,
                          task_type=CASES[case]['task_type']))))
    runtime = diskgraph.RuntimeIndex(str(manifest), num_threads=1)
    unit = None if static else _index_np_time_unit({'graph_index_path': index})
    expected = {
        ((row['SPLIT'], row['ENTITY1']) if static else (row['SPLIT'], row['ENTITY1'], row['TIMESTAMP'])):
        sorted(row['TARGET1'])
        for row in rows
    }
    if len(expected) != len(rows):
        raise ValueError(
            'Temporal control source has duplicate native row keys')
    actual = {}
    actual_exclusions = {}
    expected_exclusions = {
        (row['SPLIT'], row['ENTITY1']): sorted(row['exclude_target_keys'])
        for row in rows
    } if static else {}
    for split, stage in enumerate(('train', 'val', 'test')):
        iterator, _ = runtime.iter(stage=stage, sampling_plan={'layers': []},
                                   num_workers=1, worker_id=0, shuffle=False,
                                   epoch=0, batch_size=512,
                                   entity_node_type=entity_table,
                                   target_node_type=target_table)
        for batch in iterator:
            task = batch['task']
            anchors = ([None] * len(task['seed_node_ids']) if static else
                       engine_int64_to_ns(task['seed_timestamps'], unit).cast(
                           pa.timestamp('ns')).to_pylist())
            targets = task['lp_pos_rhs_global'].to_pylist()
            offset = 0
            excluded = task.get('lp_exclude_rhs_global')
            excluded = [] if excluded is None else excluded.to_pylist()
            excluded_counts = task.get('lp_exclude_counts')
            excluded_counts = ([0] *
                               len(task['seed_node_ids']) if excluded_counts
                               is None else excluded_counts.to_pylist())
            excluded_offset = 0
            for entity, anchor, count, excluded_count in zip(
                    task['seed_node_ids'].to_pylist(), anchors,
                    task['lp_pos_counts'].to_pylist(), excluded_counts,
                    strict=True):
                key = ((split, mappings[entity_table][entity]) if static else
                       (split, mappings[entity_table][entity], anchor))
                if key in actual:
                    raise ValueError('Temporal control repeated a native row')
                actual[key] = sorted(mappings[target_table][target]
                                     for target in targets[offset:offset +
                                                           count])
                offset += count
                if static:
                    actual_exclusions[key] = sorted(
                        mappings[target_table][target]
                        for target in excluded[excluded_offset:excluded_offset +
                                               excluded_count])
                    if set(actual[key]) & set(actual_exclusions[key]):
                        raise ValueError(
                            'Static native positives overlap prior-stage exclusions'
                        )
                    excluded_offset += excluded_count
            if offset != len(targets):
                raise ValueError(
                    'Temporal control native target lengths differ')
            if static and excluded_offset != len(excluded):
                raise ValueError('Static native exclusion lengths differ')
    if actual != expected:
        raise ValueError('Temporal control native target IDs changed')
    if static and actual_exclusions != expected_exclusions:
        raise ValueError('Static native exclusion IDs changed')


def _link_control(prepared, rows_path, rows, output_dir):
    """Use native link routing with audited targets and prior-stage exclusions."""
    import diskgraph
    from kumo_core.training.table import (
        build_split_spec,
        split_config_from_repr,
    )

    import kumo.engine as kumo

    normal = prepared['normal_table']
    case = prepared['case']
    metadata = json.loads(
        (normal.staging_dir / 'tt_ingest_spec.json').read_text())
    rhs = [
        entry for entry in metadata['lp_resolution']
        if entry['src_col'] == 'TARGET1' and entry['kind'] == 'list'
    ]
    if (case not in ('static_link', 'temporal_link') or len(rhs) != 1
            or normal.query != CASES[case]['query']):
        raise ValueError('Audited future targets require a temporal-link table')
    entity, target = metadata['entity_table'], rhs[0]['dst_table']
    definition = prepared['graph']._named_tables[target].definition
    if case == 'temporal_link':
        for payload in (prepared['rows'], rows):
            audit_temporal_link_targets(prepared['profile_dir'], payload,
                                        target, definition)
    generation = normal.generation_plan
    if case == 'static_link':
        # Core accepts the source RandomLinkSplit recipe, but not the native
        # CustomLinkSplit constructor. Replay its exact retained assignments.
        config, family = dict(split_column='LINK_STAGE',
                              manual_train_values=['0'],
                              manual_val_values=['1'],
                              manual_test_values=['2']), 'CustomLinkSplit'
    else:
        config, family = split_config_from_repr(generation.split)
    if case == 'temporal_link' and family != 'TimeRangeSplit':
        raise ValueError(
            'Temporal control requires the original time-range split')
    if case != 'static_link' or generation.timeframe_step != '???':
        config['timeframe_step'] = generation.timeframe_step
    split = build_split_spec(config, family, diskgraph.SplitSpec)
    tt_id = 'control-' + hashlib.sha256(str(
        output_dir.resolve()).encode()).hexdigest()[:16]
    stype = prepared['query']._parsed.target_stype
    result = diskgraph.build_train_table_artifacts_for_existing_table(
        output_path=str(rows_path),
        index_path=str(prepared['snapshot'].index_path), split_config=split,
        split_family_hint=family, input_timestamp_unit='nanoseconds',
        tt_id=tt_id,
        split_stats_config=dict(entity_table=entity,
                                entity_pk_column=metadata['entity_column']),
        train_table_col_stats_config=dict(
            target_stype=getattr(stype, 'value', stype)),
        target_entity_table=target, task_type=CASES[case]['task_type'],
        lp_rhs_is_temporal=metadata['lp_rhs_is_temporal'],
        allow_new_target_entities=True, ingest_on_error='abort')
    native_manifest = json.loads(
        (Path(result['staging_dir']) / 'manifest.json').read_text())
    if native_manifest.get('split_kind') != family:
        raise ValueError('Native link replay used the wrong split routing')
    table = kumo.TrainingTable(
        query=normal.query, train_table_path=Path(result['tt_parquet_path']),
        manifest_path=Path(result['staging_dir']) / 'manifest.json',
        staging_dir=Path(result['staging_dir']),
        required_stages=normal.required_stages, generation_plan=generation)
    _verify_link_resolution(prepared, table, rows, tt_id, output_dir, entity,
                            metadata['entity_column'], target, definition.pkey)
    return table


def _static_link_payload(rows):
    """Group zipped link/frequency/stage values for native per-link routing."""
    groups = {}
    for row in rows:
        groups.setdefault((row['ENTITY1'], row['TIMESTAMP']), []).append(row)
    result = []
    for (entity, anchor), group in sorted(groups.items()):
        triples, prior = [], set()
        for row in sorted(group, key=lambda row: row['SPLIT']):
            targets, frequencies = row['TARGET1'], row[
                '__kumo_internal_frequency_column__']
            if (type(row['SPLIT']) is not int or row['SPLIT'] not in (0, 1, 2)
                    or len(targets) != len(frequencies) or not targets or any(
                        type(value) is not int or value <= 0
                        for value in frequencies)
                    or len(set(targets)) != len(targets) or set(targets) & prior
                    or set(row['exclude_target_keys']) != prior):
                raise ValueError(
                    'Static target/frequency/stage associations or exclusions differ'
                )
            triples.extend(
                (target, weight, str(row['SPLIT']))
                for target, weight in zip(targets, frequencies, strict=True))
            prior.update(targets)
        triples.sort(key=lambda value: (value[2], value[0]))
        result.append(
            dict(
                ENTITY1=entity, TIMESTAMP=anchor,
                TARGET1=[value[0] for value in triples],
                __kumo_internal_frequency_column__=[
                    value[1] for value in triples
                ], LINK_STAGE=[value[2] for value in triples]))
    return result


def validate_reimport(case, before, after):
    canonical = runpy.run_path(
        str(Path(__file__).with_name(
            'wheel_temporal_tasks.py')))['canonical_stage']
    if canonical(case, before) != canonical(case, after):
        raise ValueError('Reimport changed inputs, target payloads or stages')


def prediction_context(case, rows, source_dir, *, source_known=None,
                       ranked_links=False):
    """Build scoring answers and prediction inputs from the native stages."""
    import pyarrow.parquet as pq

    if type(ranked_links
            ) is not bool or ranked_links and not case.endswith('link'):
        raise ValueError('Ranked links require a link task and a boolean flag')

    stages = [[row for row in rows if row['SPLIT'] == stage]
              for stage in range(3)]
    if any(not stage for stage in stages):
        raise ValueError(
            'Every native training, validation and test stage is required')
    test = stages[2]
    if case != 'static_link' and len({row['TIMESTAMP'] for row in test}) != 1:
        raise ValueError('Evaluation needs one anchor for prediction alignment')
    ids = [row['ENTITY1'] for row in test]
    if len(set(ids)) != len(ids):
        raise ValueError('Evaluation entities are duplicated')
    truth = dict(entity_ids=ids, classes=[], catalog=[], horizons=[], known={},
                 anchor=None if case == 'static_link' else test[0]['TIMESTAMP'])
    columns = ['ENTITY1', 'TIMESTAMP']
    if case == 'binary':
        if any({row['TARGET1']
                for row in stage} != {False, True} for stage in stages):
            raise ValueError('Every binary stage needs both answer classes')
        truth['targets'] = [row['TARGET1'] for row in test]
        rate = sum(row['TARGET1'] for row in stages[0]) / len(stages[0])
        truth['baseline_predictions'] = [rate] * len(test)
    elif case == 'forecasting':
        if any(
                len(row.get(name, [])) != 12 for row in rows
                for name in ('future_labels', 'autoregression_labels')):
            raise ValueError('Forecast rows need twelve past and future values')
        columns.append('autoregression_labels')
        truth.update(
            targets=[row['future_labels'] for row in test],
            baseline_predictions=[[row['autoregression_labels'][-1]] * 12
                                  for row in test],
            horizons=[
                truth['anchor'] + timedelta(days=7 * i) for i in range(12)
            ])
    elif case.endswith('link'):
        catalog = sorted(
            pq.read_table(
                Path(source_dir) / 'merchants',
                columns=['merchant_id'])['merchant_id'].to_pylist())
        truth['catalog'] = catalog
        if ranked_links:
            from collections import Counter
            catalog_set = set(catalog)
            counts = Counter(item for row in stages[0]
                             for item in _members(row['TARGET1']))
            if any(_members(row['TARGET1']) - catalog_set for row in rows):
                raise ValueError(
                    'Link targets are absent from the fixed catalog')
            truth.update(
                target_format='ranked_link_v1',
                target_ids=[sorted(_members(row['TARGET1'])) for row in test],
                training_popularity=[counts[item] for item in catalog])
        else:
            truth['targets'] = [[
                int(item in _members(row['TARGET1'])) for item in catalog
            ] for row in test]
            popularity = [
                sum(item in _members(row['TARGET1']) for row in stages[0])
                for item in catalog
            ]
            truth['baseline_predictions'] = [popularity[:] for _ in test]
        if case == 'static_link':
            for index, row in enumerate(test):
                known = row.get('exclude_target_keys')
                if not isinstance(
                        known, list) or set(known) & _members(row['TARGET1']):
                    raise ValueError(
                        'Static prediction needs disjoint native exclusions')
                known = sorted(
                    set(known)
                    | (source_known or {}).get(row['ENTITY1'], set()))
                if ranked_links and set(known) - catalog_set:
                    raise ValueError(
                        'Link exclusions are absent from the fixed catalog')
                if set(known) & _members(row['TARGET1']):
                    raise ValueError(
                        'Static control answers appear in source relations')
                truth['known'][str(row['ENTITY1'])] = known
                if not ranked_links:
                    for item in known:
                        truth['baseline_predictions'][index][catalog.index(
                            item)] = -1
            if ranked_links:
                truth['excluded_ids'] = [
                    truth['known'][str(entity)] for entity in ids
                ]
            else:
                truth['control_eligible'] = [[
                    item not in truth['known'][str(entity)] for item in catalog
                ] for entity in ids]
    else:
        node = runpy.run_path(
            str(Path(__file__).with_name('wheel_node_tasks.py')))
        targets = [row['TARGET1'] for row in rows]
        if case in ('multilabel', 'multilabel_ranking'):
            targets = [sorted(_members(value)) for value in targets]
        classes = ([] if case == 'regression' else sorted({
            category
            for value in targets
            for category in ([value] if case == 'multiclass' else value)
        }))
        encoded = node['encode_targets'](case, targets, classes)
        split = [row['SPLIT'] for row in rows]
        truth.update(
            classes=classes, targets=[
                value for value, stage in zip(encoded, split) if stage == 2
            ], baseline_predictions=node['baseline_predictions'](case,
                                                                 encoded, split,
                                                                 len(test)))
    return truth, [{name: row[name] for name in columns} for row in test]


def _pair_tables(source_dir, table_names, *, sources=None):
    import pyarrow.parquet as pq

    paths = ({
        name: Path(source_dir) / name
        for name in table_names
    } if sources is None else sources)
    return [
        name for name, path in paths.items()
        if {'customer_id', 'merchant_id'} <= set(
            pq.read_schema(path if Path(path).is_file(
            ) else next(Path(path).glob('*.parquet'))).names)
    ]


def _source_neighbors(source_dir, table_names, *, sources=None):
    import duckdb

    paths = ({
        name: Path(source_dir) / name
        for name in table_names
    } if sources is None else sources)
    tables = _pair_tables(source_dir, table_names, sources=paths)
    queries = [
        "SELECT DISTINCT customer_id, merchant_id FROM read_parquet('" +
        str(paths[name] if Path(paths[name]).is_file() else Path(paths[name]) /
            '*.parquet').replace("'", "''") + "')" for name in tables
    ]
    known = {}
    with duckdb.connect() as connection:
        if not queries:
            return known, tables
        for entity, item in connection.execute(
                ' UNION '.join(queries)).fetchall():
            known.setdefault(entity, set()).add(item)
    return known, tables


def _static_controls(prepared, seed):
    """Use visible graph relationships and retained labels, never final answers."""
    tables = prepared['graph']._named_tables
    sources = {
        name: Path(table.connector.source_table_for(table.table_name).path)
        for name, table in tables.items()
    }
    known, names = _source_neighbors(prepared['profile_dir'], tables,
                                     sources=sources)
    for row in prepared['rows']:
        known.setdefault(row['ENTITY1'], set()).update(_members(row['TARGET1']))
    rows = static_control_targets(prepared['rows'],
                                  prepared['truth']['catalog'], known, seed)
    pool = dict(
        method='absent_from_visible_calibration_relationships',
        draw_policy='without_replacement_across_stages',
        source_pair_tables=names, source_neighbors_sha256=hashlib.sha256(
            json.dumps(
                {
                    str(entity): sorted(values)
                    for entity, values in known.items()
                }, sort_keys=True).encode()).hexdigest(),
        null_method='uniform_eligible_items', projected_graph=dict(
            scope='unchanged_isolated_normal_projection',
            node_counts=prepared['graph_counts'],
            original_stages=prepared['identity']['evaluation_isolation']
            ['source_stages']))
    return rows, known, pool


def static_control_targets(rows, catalog, source_known, seed):
    """Draw uniform answers from nonedges without changing graph inputs."""
    result = copy.deepcopy(rows)
    randomizer = random.Random(seed)
    drawn, catalog = {}, set(catalog)
    order = sorted(
        range(len(result)), key=lambda i:
        (result[i].get('SPLIT', 0), result[i]['ENTITY1']))
    for index in order:
        row = result[index]
        prior = drawn.setdefault(row['ENTITY1'], set())
        available = sorted(catalog - source_known.get(row['ENTITY1'], set()) -
                           prior)
        count = len(row['TARGET1'])
        if count > len(available):
            raise ValueError('Not enough absent merchants for a static control')
        row['TARGET1'] = randomizer.sample(available, count)
        if 'exclude_target_keys' in row:
            row['exclude_target_keys'] = sorted(prior)
        prior.update(row['TARGET1'])
    return result


def validate_snapshot_counts(snapshot_dir, table_names, manifest,
                             projections=None):
    import pyarrow.parquet as pq

    projections = projections or {}
    expected = {
        name:
        pq.ParquetFile(projections[name]).metadata.num_rows
        if name in projections else manifest['tables'][name]['rows']
        for name in table_names
    }
    actual = json.loads(
        (Path(snapshot_dir) /
         'ingest_stats.json').read_text())['stats']['node_counts']
    if actual != expected:
        raise ValueError('Native graph populations differ from source rows')
    return actual


def project_static_sources(source_dir, table_names, rows, output_dir):
    """Hide held-out pairs in every direct customer-to-merchant relation."""
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    heldout = {(row['ENTITY1'], item)
               for row in rows if row['SPLIT'] != 0
               for item in _members(row['TARGET1'])}
    connection = duckdb.connect()
    connection.register(
        'heldout',
        pa.table(
            dict(customer_id=[pair[0] for pair in sorted(heldout)],
                 merchant_id=[pair[1] for pair in sorted(heldout)])))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    try:
        for name in _pair_tables(source_dir, table_names):
            source = Path(source_dir) / name
            path = str(source / '*.parquet').replace("'", "''")
            kept = connection.execute(
                f"SELECT e.* FROM read_parquet('{path}') e ANTI JOIN heldout h "
                'USING(customer_id, merchant_id)').to_arrow_table()
            destination = output_dir / f'{name}.parquet'
            pq.write_table(kept, destination)
            result[name] = destination
    finally:
        connection.close()
    return result


def _rows(table, directory):
    import pyarrow.parquet as pq

    from kumo.engine import FileConnector, TrainingTableOutputConfig

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table.export(
        TrainingTableOutputConfig(FileConnector(root_dir=str(directory)),
                                  'rows'))
    rows = pq.read_table(directory / 'rows').to_pylist()
    if table.generation_plan.split.startswith(
        ('RandomLinkSplit', 'CustomLinkSplit')):
        spec = json.loads(
            (table.staging_dir / 'tt_ingest_spec.json').read_text())
        exclusions = {}
        for stage in spec['split_stages']:
            split = {'train': 0, 'val': 1, 'test': 2}[stage['key']]
            for part in stage['parts']:
                for row in pq.read_table(table.staging_dir / part).to_pylist():
                    exclusions[split,
                               row['ENTITY1']] = row['exclude_target_keys']
        for row in rows:
            row['exclude_target_keys'] = exclusions[row['SPLIT'],
                                                    row['ENTITY1']]
    return rows


def _static_payload(source_dir, control_seed, anchors, source_known=None):
    import duckdb
    import pyarrow.parquet as pq

    path = str(Path(source_dir) / 'purchases/*.parquet').replace("'", "''")
    with duckdb.connect() as connection:
        values = connection.execute(
            'SELECT customer_id, list(merchant_id ORDER BY merchant_id), '
            'list(n ORDER BY merchant_id) FROM ('
            f"SELECT customer_id, merchant_id, count(*) n FROM read_parquet('{path}') "
            'GROUP BY customer_id, merchant_id) GROUP BY customer_id ORDER BY customer_id'
        ).fetchall()
    rows = [
        dict(ENTITY1=entity, TIMESTAMP=anchors[entity], TARGET1=targets,
             __kumo_internal_frequency_column__=counts, SPLIT=0)
        for entity, targets, counts in values
    ]
    if control_seed is not None:
        catalog = pq.read_table(
            Path(source_dir) / 'merchants',
            columns=['merchant_id'])['merchant_id'].to_pylist()
        rows = static_control_targets(rows, catalog, source_known, control_seed)
    return [{
        name: value
        for name, value in row.items() if name != 'SPLIT'
    } for row in rows]


def _static_graph(graph):
    # Static links use the same columns and edges without temporal table roles.
    return replace(
        graph, tables={
            name:
            replace(
                table, definition=replace(table.definition, time_col=None,
                                          end_time_col=None))
            for name, table in graph._named_tables.items()
        })


def evaluation_capacity(case, rows, source_dir, *, max_dense_bytes=None):
    """Count native work without allocating an evaluation-by-catalog matrix.

    Three eight-byte arrays are a lower bound. Python objects, graph storage,
    repeat predictions and other work need additional memory.
    """
    import pyarrow.parquet as pq

    if case not in CASES or (max_dense_bytes is not None and
                             (type(max_dense_bytes) is not int
                              or max_dense_bytes <= 0)):
        raise ValueError(
            'Use a known task and a positive explicit memory budget')
    counts = {
        name: sum(row['SPLIT'] == stage for row in rows)
        for stage, name in enumerate(('train', 'val', 'test'))
    }
    ids = [row['ENTITY1'] for row in rows if row['SPLIT'] == 2]
    if not all(counts.values()) or len(set(ids)) != len(ids):
        raise ValueError(
            'Capacity needs complete native stages and unique evaluation IDs')
    if case.endswith('link'):
        width = sum(
            pq.ParquetFile(path).metadata.num_rows
            for path in (Path(source_dir) / 'merchants').rglob('*.parquet'))
    elif case == 'forecasting':
        width = 12
    elif case in ('binary', 'regression'):
        width = 1
    else:
        labels = (row['TARGET1'] for row in rows if row['SPLIT'] in (0, 1))
        width = len({
            label
            for value in labels
            for label in ([value] if case == 'multiclass' else _members(value))
        })
    if width < 1:
        raise ValueError('Capacity needs the complete target coordinate set')
    cells = len(ids) * width
    lower = cells * 3 * 8
    return dict(
        case=case, native_rows=counts, evaluation_entities=len(ids),
        score_columns=width, score_cells=cells,
        three_dense_arrays_lower_bound_bytes=lower,
        budget_bytes=max_dense_bytes, within_budget=None if max_dense_bytes
        is None else lower <= max_dense_bytes, capacity_qualified=False)


def prepare_profile(profile_dir, scale_bench_dir, case, output_dir, *,
                    control_seed=None, preflight_only=False,
                    ranked_links=False):
    """Prepare immutable PR 172 inputs once, before fitting fresh models."""
    manifest = read_profile(profile_dir, case)
    profile_dir, scale_bench_dir = Path(profile_dir).resolve(), Path(
        scale_bench_dir).resolve()
    source_files = [scale_bench_dir / 'kumo/build_graph.py']
    public_graphs = scale_bench_dir / 'graphs.json'
    if public_graphs.is_file():
        source_files.append(public_graphs)
    else:
        source_files.extend(sorted((scale_bench_dir / 'scalebench').glob('*.py')))
    source_hash = hashlib.sha256(b''.join(
        str(path.relative_to(scale_bench_dir)).encode() + path.read_bytes()
        for path in source_files)).hexdigest()
    manifest_hash = hashlib.sha256(
        (profile_dir / 'manifest.json').read_bytes()).hexdigest()
    key = (str(profile_dir), manifest_hash, source_hash)
    if type(ranked_links
            ) is not bool or ranked_links and not case.endswith('link'):
        raise ValueError('Ranked links require a link task and a boolean flag')
    prepared_key = (*key, case, control_seed, ranked_links)
    if prepared_key in _PREPARED and not preflight_only:
        return _PREPARED[prepared_key]
    if control_seed is not None and case != 'static_link':
        raise ValueError('Use control_table for nonstatic task controls')

    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumoapi.model_plan import TrainingTableGenerationPlan
    from kumoapi.train import TrainingTableSpec

    import kumo.engine as kumo

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    kumo.configure(scratch_dir=output_dir / 'scratch')
    sys.path.insert(0, str(scale_bench_dir))
    loader = importlib.util.spec_from_file_location('wheel_scale_graph',
                                                    source_files[0])
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    spec = CASES[case]
    files = kumo.FileConnector(root_dir=str(profile_dir))
    graph_key = (*key, case == 'static_link')
    if graph_key not in _GRAPHS:
        graph, _ = module.kumo_graph(spec['graph'], files,
                                     task=spec['source_task'])
        if case == 'static_link':
            graph = _static_graph(graph)
        graph.validate()
        _GRAPHS[graph_key] = graph, graph.snapshot()
    graph, snapshot = _GRAPHS[graph_key]
    graph_counts = validate_snapshot_counts(snapshot.index_path,
                                            graph._named_tables, manifest)
    query = kumo.PredictiveQuery(graph, spec['query'])
    query.validate()
    if query.get_task_type().value != spec['task_type']:
        raise ValueError('Prepared query resolved to the wrong task type')
    source_known, pair_tables = None, []
    if case == 'static_link':
        generation = TrainingTableGenerationPlan(
            split='RandomLinkSplit([0.6, 0.2, 0.2])')
        generated = query.generate_training_table(snapshot, plan=generation)
        native_rows = _rows(generated, output_dir / 'native')
        anchors = {row['ENTITY1']: row['TIMESTAMP'] for row in native_rows}
        if any(row['TIMESTAMP'] != anchors[row['ENTITY1']]
               for row in native_rows):
            raise ValueError('Static native stages disagree on entity anchors')
        if control_seed is not None:
            source_known, pair_tables = _source_neighbors(
                profile_dir, graph._named_tables)
        payload = _static_payload(profile_dir, control_seed, anchors,
                                  source_known)
        data = output_dir / 'data'
        data.mkdir()
        pq.write_table(pa.Table.from_pylist(payload), data / 'payload.parquet')
        connector = kumo.FileConnector(root_dir=str(data))

        def materialize(pquery, graph_snapshot):
            return kumo.TrainingTable.from_source_table(
                connector['payload.parquet'], TrainingTableSpec(), pq=pquery,
                snapshot=graph_snapshot, plan=generation)

        full = materialize(query, snapshot)
        original = _rows(full, output_dir / 'unprojected')
        projections = project_static_sources(profile_dir, graph._named_tables,
                                             original, data / 'projection')
        projected_files = {
            name:
            connector[str(projections[name].relative_to(data))]
            if name in projections else files[name]
            for name in graph._named_tables
        }
        graph, _ = module.kumo_graph(spec['graph'], projected_files,
                                     task=spec['source_task'])
        graph = _static_graph(graph)
        snapshot = graph.snapshot()
        graph_counts = validate_snapshot_counts(
            snapshot.index_path, graph._named_tables, manifest,
            projections if control_seed is None else None)
        query = kumo.PredictiveQuery(graph, spec['query'])
        table = materialize(query, snapshot)
        rows = _rows(table, output_dir / 'projected')
        validate_reimport(case, original, rows)
    else:
        ranges = generation_ranges(manifest, case)
        generation = TrainingTableGenerationPlan(
            split=(f'TimeRangeSplit({ranges!r}, timeframe={spec["days"]}, '
                   'flexible_boundaries=False)'),
            timeframe_step=spec.get('step_days', spec['days']),
            **({
                'lag_timesteps': 12,
                'year_over_year': False
            } if case == 'forecasting' else {}))
        table = query.generate_training_table(snapshot, plan=generation)
        rows = _rows(table, output_dir)
    validate_chronology(case, manifest, rows)
    if control_seed is None:
        audit_targets(case, profile_dir, rows)
    if preflight_only:
        capacity = evaluation_capacity(case, rows, profile_dir)
        (output_dir / 'evaluation-capacity.json').write_text(
            json.dumps(capacity, indent=2))
        return capacity
    truth, prediction_rows = prediction_context(case, rows, profile_dir,
                                                source_known=source_known,
                                                ranked_links=ranked_links)
    identity = dict(
        manifest_sha256=manifest_hash, scale_source_sha256=source_hash,
        case=case, recipe=spec, control_seed=control_seed,
        forecast_profile_qualified=False if case == 'forecasting' else None)
    if ranked_links:
        identity['ranked_links'] = True
    if source_known is not None:
        identity['control_pool'] = dict(
            method='uniform_absent_source_pairs',
            source_pair_tables=pair_tables,
            source_neighbors_sha256=hashlib.sha256(
                json.dumps(
                    {
                        str(entity): sorted(items)
                        for entity, items in source_known.items()
                    }, sort_keys=True).encode()).hexdigest(),
            null_method='uniform_eligible_items', projected_rows_removed=0)
    prepared = dict(graph=graph, snapshot=snapshot, query=query,
                    query_text=spec['query'], normal_table=table, rows=rows,
                    prediction_rows=prediction_rows, truth=truth,
                    generation_plan=generation, manifest=manifest,
                    identity=identity, profile_dir=profile_dir, case=case,
                    graph_counts=graph_counts)
    _PREPARED[prepared_key] = prepared
    return prepared


def _reimport_rows(prepared, rows, output_dir, *, filename):
    """Reuse native manual, forecast and future-link imports without new data."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumoapi.json_serde import to_json_dict
    from kumoapi.train import TrainingTableSpec

    import kumo.engine as kumo

    case = prepared['case']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = output_dir / 'data'
    data.mkdir()
    kumo.configure(scratch_dir=output_dir / 'scratch')
    source = pa.Table.from_pylist(
        _static_link_payload(rows) if case == 'static_link' else rows)
    forecasting = case == 'forecasting'
    if forecasting or case == 'temporal_link':
        source = source.drop(['SPLIT'])
    pq.write_table(source, data / filename)
    if case in ('static_link', 'temporal_link'):
        table = _link_control(prepared, data / filename, rows, output_dir)
    else:
        table = kumo.TrainingTable.from_source_table(
            kumo.FileConnector(root_dir=str(data))[filename],
            TrainingTableSpec(split_col=None if forecasting else 'SPLIT'),
            pq=prepared['query'], snapshot=prepared['snapshot'],
            plan=prepared['generation_plan'])
    if forecasting:
        expected = to_json_dict(prepared['normal_table'].generation_plan)
        actual = to_json_dict(table.generation_plan)
        if expected['split'].startswith('TimeRangeSplit('):
            for name in ('train_start_offset', 'train_end_offset'):
                if (expected[name] in ('inferred', '???')
                        and actual[name] in ('inferred', '???')):
                    expected.pop(name)
                    actual.pop(name)
        if actual != expected:
            raise ValueError(
                'Forecast control changed its time-range generation plan')
    imported = _rows(table, output_dir / 'imported')
    validate_reimport(case, rows, imported)
    validate_chronology(case, prepared['manifest'], imported)
    return table, imported


def control_table(prepared, seed, output_dir):
    """Reimport permuted answers while retaining native input rows and stages."""
    case = prepared['case']
    known, pool = None, {}
    if case == 'static_link' and 'final_isolation' in prepared:
        verify_final_isolation(prepared)
        if not prepared.get('graph_counts'):
            raise ValueError(
                'Isolated static controls require observed projected graph counts'
            )
        control_rows, known, pool = _static_controls(prepared, seed)
    else:
        control_rows = shuffle_targets(case, prepared['rows'], seed)
    table, imported = _reimport_rows(prepared, control_rows, output_dir,
                                     filename='control.parquet')
    truth, prediction_rows = prediction_context(
        case, imported, prepared['profile_dir'], source_known=known,
        ranked_links=prepared['identity'].get('ranked_links', False))
    result = {
        **prepared, 'normal_table': table,
        'generation_plan': table.generation_plan,
        'rows': imported,
        'truth': truth,
        'prediction_rows': prediction_rows,
        'identity': {
            **prepared['identity'], 'control_seed':
            seed,
            **({
                'control_pool': pool
            } if pool else {}),
            **({
                'temporal_control_import':
                'audited_window_targets_and_native_ids'
            } if case == 'temporal_link' else {})
        }
    }
    return (_bind_control_final(prepared, result,
                                Path(output_dir) / 'final-isolation')
            if 'final_isolation' in prepared else result)


def _stage_fingerprints(case, rows):
    canonical = runpy.run_path(
        str(Path(__file__).with_name(
            'wheel_temporal_tasks.py')))['canonical_stage']
    fingerprint = runpy.run_path(
        str(Path(__file__).with_name('wheel_task_checks.py')))['fingerprint']
    if case == 'forecasting':
        # Native import may cast the optional scalar placeholder int to float.
        # Compare its exact value; the real horizon/history arrays stay strict.
        rows = [
            dict(row, TARGET1=str(Fraction(str(row['TARGET1']))))
            if 'TARGET1' in row else row for row in rows
        ]
    return {
        name:
        fingerprint(
            canonical(case, [row for row in rows if row['SPLIT'] == stage]))
        for stage, name in enumerate(('train', 'val', 'test'))
    }


def _final_ids(case, rows):
    ids = [row['ENTITY1'] for row in rows if row['SPLIT'] == 2]
    if (len(ids) < 4 or any(type(value) not in (str, int) for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Isolation needs at least four unique evaluation IDs')
    ordered = sorted(
        ids, key=lambda value: hashlib.sha256(
            json.dumps(
                ['wheel-final-isolation-v1', case,
                 type(value).__name__, value]).encode()).hexdigest())
    return set(ordered[:len(ordered) // 2])


def isolate_final(prepared, calibration_dir, final_dir):
    """Opt in before fitting: reserve half of stage 2 by task/entity hash.

    Selection never reads answer values or rerolls for label coverage. Preserve
    complete native rows and earlier history. The final directory stays sealed
    until recipes and initial limits are frozen; this function scores no model.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    calibration_dir, final_dir = Path(calibration_dir).resolve(), Path(
        final_dir).resolve()
    if (calibration_dir.is_relative_to(final_dir)
            or final_dir.is_relative_to(calibration_dir)
            or 'final_isolation' in prepared):
        raise ValueError(
            'Final and calibration directories must be separate and new')
    case, rows = prepared['case'], prepared['rows']
    selected = _final_ids(case, rows)
    final = [
        row for row in rows if row['SPLIT'] == 2 and row['ENTITY1'] in selected
    ]
    calibration = [
        row for row in rows
        if row['SPLIT'] != 2 or row['ENTITY1'] not in selected
    ]
    # Check admission after the label-blind choice. Never select again on failure.
    for cohort in (final, [row for row in calibration if row['SPLIT'] == 2]):
        labels = ([row['TARGET1']
                   for row in cohort] if case in ('binary', 'multiclass',
                                                  'multilabel',
                                                  'multilabel_ranking') else [])
        if case == 'binary' and set(labels) != {False, True}:
            raise ValueError('Both partitions need both answer classes')
        if case == 'multiclass' and set(labels) != set(
                prepared['truth']['classes']):
            raise ValueError('Both partitions need the fixed class coverage')
        if case in ('multilabel', 'multilabel_ranking'):
            counts = [
                sum(label in _members(value) for value in labels)
                for label in prepared['truth']['classes']
            ]
            if any(not 0 < count < len(labels) for count in counts):
                raise ValueError(
                    'Both partitions need positive and negative class coverage')
    table, imported = _reimport_rows(prepared, calibration, calibration_dir,
                                     filename='calibration.parquet')
    fit_root = Path(table.staging_dir).resolve()
    if final_dir.is_relative_to(fit_root) or fit_root.is_relative_to(final_dir):
        raise ValueError('Final answers must stay outside fit staging')
    validate_reimport(case, calibration, imported)
    known = {
        row['ENTITY1']:
        set(prepared['truth']['known'].get(str(row['ENTITY1']), []))
        for row in calibration if row['SPLIT'] == 2
    }
    truth, prediction_rows = prediction_context(
        case, imported, prepared['profile_dir'], source_known=known,
        ranked_links=prepared['identity'].get('ranked_links', False))
    if truth['classes'] != prepared['truth']['classes']:
        raise ValueError(
            'Reserved splitting changed the fixed target coordinate set')
    context = {
        key: prepared['truth'][key]
        for key in ('classes', 'catalog', 'horizons', 'anchor')
    }
    context['known'] = {
        str(entity): prepared['truth']['known'][str(entity)]
        for entity in selected if str(entity) in prepared['truth']['known']
    }
    context = json.loads(
        json.dumps(context, default=lambda value: value.isoformat(),
                   allow_nan=False))
    fingerprint = runpy.run_path(
        str(Path(__file__).with_name('wheel_task_checks.py')))['fingerprint']
    partition = dict(format='wheel-final-isolation-v1',
                     rule='sha256-task-entity-half',
                     final_context_sha256=fingerprint(context),
                     source_stages=_stage_fingerprints(case, rows),
                     calibration_stages=_stage_fingerprints(case, imported),
                     final_stage=_stage_fingerprints(case, final)['test'])
    if case == 'static_link':
        partition['native_import'] = dict(
            routing='CustomLinkSplit',
            source_generation_split=prepared['normal_table'].generation_plan.
            split, stage_assignment='replay_exact_retained_source_stages')
    final_dir.mkdir(parents=True, exist_ok=False)
    final_path = final_dir / 'rows.parquet'
    pq.write_table(pa.Table.from_pylist(final), final_path)
    with final_path.open('rb') as stream:
        final_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata = dict(
        partition=partition, context=context, profile=prepared['identity'],
        query=prepared['query_text'], final_rows_sha256=final_hash,
        fit_files=_control_staging_hashes(table.staging_dir,
                                          link=case.endswith('link')))
    content = json.dumps(metadata, indent=2, sort_keys=True,
                         allow_nan=False).encode()
    (final_dir / 'descriptor.json').write_bytes(content)
    result = dict(
        prepared, normal_table=table, generation_plan=table.generation_plan,
        rows=imported, truth=truth, prediction_rows=prediction_rows,
        identity=dict(prepared['identity'], evaluation_isolation=partition),
        final_isolation=dict(directory=str(final_dir),
                             sha256=hashlib.sha256(content).hexdigest()))
    verify_final_isolation(result)
    return result


def verify_final_isolation(prepared):
    """Reject changed fit files, final answers, populations or split metadata."""
    import pyarrow.parquet as pq

    descriptor = prepared['final_isolation']
    root = Path(descriptor['directory']).resolve()
    fit_root = Path(prepared['normal_table'].staging_dir).resolve()
    if root.is_relative_to(fit_root) or fit_root.is_relative_to(root):
        raise ValueError('Final answers must stay outside fit staging')
    content = (root / 'descriptor.json').read_bytes()
    if hashlib.sha256(content).hexdigest() != descriptor['sha256']:
        raise ValueError('Reserved final descriptor changed')
    metadata = json.loads(content)
    partition = prepared['identity']['evaluation_isolation']
    profile = {
        key: value
        for key, value in prepared['identity'].items()
        if key != 'evaluation_isolation'
    }
    fingerprint = runpy.run_path(
        str(Path(__file__).with_name('wheel_task_checks.py')))['fingerprint']
    if (metadata['partition'] != partition or metadata['profile'] != profile
            or fingerprint(
                metadata['context']) != partition['final_context_sha256']
            or metadata['query'] != prepared['query_text']
            or prepared['normal_table'].query != prepared['query_text']
            or _control_staging_hashes(prepared['normal_table'].staging_dir,
                                       link=prepared['case'].endswith('link'))
            != metadata['fit_files']):
        raise ValueError('Reserved final profile or fit staging changed')
    control = prepared['identity'].get('control_seed') is not None
    rows_root = root
    if control:
        parent = metadata.get('parent', {})
        rows_root = Path(parent.get('directory', '')).resolve()
        parent_content = (rows_root / 'descriptor.json').read_bytes()
        parent_metadata = json.loads(parent_content)
        parent_profile = dict(profile, control_seed=None)
        for field in ('control_pool', 'temporal_control_import'):
            parent_profile.pop(field, None)
        if (hashlib.sha256(parent_content).hexdigest() != parent.get('sha256')
                or parent_metadata['partition'] != partition
                or parent_metadata['profile'] != parent_profile
                or metadata.get('final_control_answers_created') is not False
                or parent_metadata['final_rows_sha256']
                != metadata['final_rows_sha256']):
            raise ValueError(
                'Calibration control requires its unchanged ordinary final seal'
            )
    final_path = rows_root / 'rows.parquet'
    with final_path.open('rb') as stream:
        if hashlib.file_digest(
                stream, 'sha256').hexdigest() != metadata['final_rows_sha256']:
            raise ValueError('Reserved final rows changed')
    final = pq.read_table(final_path).to_pylist()
    case, calibration = prepared['case'], prepared['rows']
    combined = calibration + final
    selected = _final_ids(case, combined)
    calibration_ids = [
        row['ENTITY1'] for row in calibration if row['SPLIT'] == 2
    ]
    spec = json.loads((fit_root / 'tt_ingest_spec.json').read_text())
    [test] = [stage for stage in spec['split_stages'] if stage['key'] == 'test']
    native_ids = [
        entity for part in test['parts'] for entity in pq.read_table(
            fit_root / part, columns=['ENTITY1'])['ENTITY1'].to_pylist()
    ]
    if (len(native_ids) != len(calibration_ids)
            or set(native_ids) != set(calibration_ids)
            or set(native_ids) & selected):
        raise ValueError(
            'Reserved IDs or wrong counts reached native test parts')
    if (any(row['SPLIT'] != 2
            for row in final) or {row['ENTITY1']
                                  for row in final} != selected
            or any(row['SPLIT'] == 2 and row['ENTITY1'] in selected
                   for row in calibration)
            or set(prepared['truth']['entity_ids']) != set(calibration_ids)
            or len(prepared['truth']['entity_ids']) != len(calibration_ids)
            or {row['ENTITY1']
                for row in prepared['prediction_rows']} != set(calibration_ids)
            or not control and _stage_fingerprints(case, combined)
            != partition['source_stages'] or _stage_fingerprints(
                case, calibration) != (metadata.get('control_stages') if control
                                       else partition['calibration_stages'])
            or _stage_fingerprints(case,
                                   final)['test'] != partition['final_stage']):
        raise ValueError(
            'Reserved final populations or answer associations changed')


def _bind_control_final(normal, control, directory):
    """Share the ordinary final seal; final control answers do not exist yet."""
    verify_final_isolation(normal)
    if type(control['identity'].get(
            'control_seed')) is not int or control['case'] != normal['case']:
        raise ValueError(
            'An isolated calibration control needs a fixed seed and task')
    parent = normal['final_isolation']
    metadata = json.loads(
        (Path(parent['directory']) / 'descriptor.json').read_text())
    metadata.update(
        profile={
            key: value
            for key, value in control['identity'].items()
            if key != 'evaluation_isolation'
        }, parent=parent, final_control_answers_created=False,
        control_stages=_stage_fingerprints(control['case'], control['rows']),
        fit_files=_control_staging_hashes(
            control['normal_table'].staging_dir,
            link=control['case'].endswith('link')))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    content = json.dumps(metadata, indent=2, sort_keys=True,
                         allow_nan=False).encode()
    (directory / 'descriptor.json').write_bytes(content)
    result = dict(
        control,
        final_isolation=dict(directory=str(directory),
                             sha256=hashlib.sha256(content).hexdigest()))
    verify_final_isolation(result)
    return result


def _control_staging_hashes(root, *, link=False):
    """Cover all files and reject artifact paths outside the frozen staging."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Frozen control paths must not contain links')
    root = root.resolve()
    files = sorted(root.rglob('*'))
    if any(path.is_symlink() for path in files):
        raise ValueError('Frozen control paths must not contain links')
    hashes = {}
    for path in files:
        if path.is_file():
            with path.open('rb') as stream:
                hashes[str(path.relative_to(root))] = hashlib.file_digest(
                    stream, 'sha256').hexdigest()
    required = ('manifest.json', 'tt_ingest_spec.json', 'stats.json',
                'split_stats.json')
    if not link:
        required += ('train_table_col_stats.json', )
    if any(name not in hashes for name in required):
        raise ValueError('Frozen control lacks required producer files')
    manifest, spec = [
        json.loads((root / name).read_text()) for name in required[:2]
    ]
    if (manifest['version'] != 3
            or manifest['train_table']['path'] != 'tt.parquet'
            or spec['schema_version'] != 'diskgraph.tt_ingest_spec.v3'
            or {stage['key']
                for stage in spec['split_stages']} != {'train', 'val', 'test'}):
        raise ValueError('Unexpected frozen control artifact contract')

    def paths(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'path':
                    yield child
                elif key == 'parts' and isinstance(child, list):
                    for part in child:
                        yield from ([part]
                                    if isinstance(part, str) else paths(part))
                else:
                    yield from paths(child)
        elif isinstance(value, list):
            for child in value:
                yield from paths(child)

    for value in paths([manifest, spec]):
        if (not isinstance(value, str) or '\\' in value
                or any(part in ('', '.', '..') for part in value.split('/'))):
            raise ValueError(f'Unsafe frozen control path: {value!r}')
        path = (root / value).resolve(strict=True)
        if not path.is_relative_to(root) or not (path.is_file()
                                                 or path.is_dir()):
            raise ValueError('Frozen control path escapes its staging')
    return hashes


def freeze_control(prepared, destination):
    """Freeze native files; seed None is an isolated normal."""
    from importlib.metadata import version

    from kumoapi.json_serde import to_json_dict

    case, seed = prepared['case'], prepared['identity']['control_seed']
    if 'final_isolation' in prepared:
        verify_final_isolation(prepared)
    table = prepared['normal_table']
    if (case not in CASES or (seed is not None and
                              (type(seed) is not int or seed not in range(5)))
            or seed is None and 'final_isolation' not in prepared
            or table.query != CASES[case]['query']
            or table.query != prepared['query_text']):
        raise ValueError('Only prepared native tables can be frozen')
    source = table.staging_dir.resolve()
    hashes = _control_staging_hashes(source, link=case.endswith('link'))
    native_family = 'CustomLinkSplit' if case == 'static_link' else 'TimeRangeSplit'
    if case.endswith('link') and json.loads(
        (source /
         'manifest.json').read_text()).get('split_kind') != native_family:
        raise ValueError('Frozen link table used the wrong native routing')
    destination = Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Frozen control source and destination overlap')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, destination / 'staging')
    if hashes != _control_staging_hashes(destination / 'staging',
                                         link=case.endswith('link')):
        raise ValueError('Frozen control fingerprint differs after copying')
    descriptor = dict(
        format='wheel-scale-control-v1', case=case, seed=seed,
        arm='normal' if seed is None else 'control',
        identity=prepared['identity'], query=table.query,
        generation_plan=to_json_dict(table.generation_plan),
        required_stages=list(table.required_stages), train_table_path=str(
            table.train_table_path.resolve().relative_to(source)),
        manifest_path=str(table.manifest_path.resolve().relative_to(source)),
        producer_version=version('kumo-engine'), files=hashes)
    if case.endswith('link'):
        descriptor['native_split_kind'] = native_family
    if seed is None:
        seal = prepared['final_isolation']
        (destination / 'final').mkdir()
        for name in ('rows.parquet', 'descriptor.json'):
            shutil.copyfile(
                Path(seal['directory']) / name, destination / 'final' / name)
        descriptor['final_isolation'] = dict(sha256=seal['sha256'])
    (destination / 'descriptor.json').write_text(
        json.dumps(descriptor, indent=2, allow_nan=False))
    return descriptor


def load_control(normal, seed, source, output_dir):
    """Use the same frozen control with either wheel's own graph snapshot."""
    source = Path(source)
    descriptor_path = source / 'descriptor.json'
    descriptor = json.loads(descriptor_path.read_text())
    if Path(output_dir).resolve().is_relative_to(source.resolve()):
        raise ValueError('Frozen control export must stay outside the fixture')
    from kumo_core.io.json import from_json_dict
    from kumo_core.model_plan import TrainingTableGenerationPlan
    from kumoapi.json_serde import to_json_dict

    import kumo.engine as kumo

    case = normal['case']
    identity = dict(normal['identity'], control_seed=seed)
    expected_rows = None
    known = None
    if seed is not None and case == 'static_link':
        if 'final_isolation' not in normal:
            raise ValueError(
                'Static frozen controls require isolated calibration')
        expected_rows, known, pool = _static_controls(normal, seed)
        identity['control_pool'] = pool
    elif seed is not None and case == 'temporal_link':
        identity[
            'temporal_control_import'] = 'audited_window_targets_and_native_ids'
    if seed is None:
        partition = descriptor.get('identity', {}).get('evaluation_isolation',
                                                       {})
        if (descriptor.get('arm') != 'normal'
                or not descriptor.get('final_isolation')
                or 'final_isolation' in normal
                or partition.get('source_stages') != _stage_fingerprints(
                    case, normal['rows'])):
            raise ValueError(
                'A frozen normal calibration needs its matching full native source and seal'
            )
        selected = _final_ids(case, normal['rows'])
        expected_rows = [
            row for row in normal['rows']
            if row['SPLIT'] != 2 or row['ENTITY1'] not in selected
        ]
        identity['evaluation_isolation'] = partition
    expected_plan = normal['normal_table'].generation_plan
    if case not in ('forecasting', 'static_link', 'temporal_link'):
        expected_plan = replace(
            expected_plan, split=("ManualSplit(col='SPLIT', train_value=0, "
                                  'val_value=1, test_value=2)'))
    expected_plan = to_json_dict(expected_plan)
    # Manual imports retain unset offsets rather than the inferred marker.
    for name in ('train_start_offset', 'train_end_offset'):
        if (expected_plan.get(name) in ('inferred', '???') and
                descriptor['generation_plan'].get(name) in ('inferred', '???')):
            expected_plan[name] = descriptor['generation_plan'][name]
    # Generated scalar regression resolves forecast defaults; manual imports
    # retain the unset values from the same requested generation recipe.
    requested = to_json_dict(normal['generation_plan'])
    if case == 'regression':
        for name, value in (('forecast_length', 1), ('lag_timesteps', 0),
                            ('year_over_year', False)):
            if (requested.get(name) == '???'
                    and descriptor['generation_plan'].get(name) == '???'
                    and expected_plan.get(name) == value):
                expected_plan[name] = '???'
    if (case not in CASES or (seed is not None and
                              (type(seed) is not int or seed not in range(5)))
            or descriptor.get(
                'arm', 'control') != ('normal' if seed is None else 'control')
            or normal['identity']['control_seed'] is not None
            or descriptor['format'] != 'wheel-scale-control-v1'
            or descriptor['case'] != case or descriptor['seed'] != seed
            or descriptor['identity'] != identity
            or descriptor['query'] != normal['query_text']
            or descriptor['query'] != normal['normal_table'].query
            or descriptor['generation_plan'] != expected_plan
            or descriptor['required_stages']
            != (list(normal['normal_table'].required_stages) if case
                == 'forecasting' or case.endswith('link') else ['train', 'val'])
            or descriptor['train_table_path'] != 'tt.parquet'
            or descriptor['manifest_path'] != 'manifest.json'):
        raise ValueError('Frozen control query, profile, seed or plan differs')
    hashes = _control_staging_hashes(source / 'staging',
                                     link=case.endswith('link'))
    if hashes != descriptor['files']:
        raise ValueError('Frozen control fingerprint differs')
    if case.endswith('link'):
        native_family = 'CustomLinkSplit' if case == 'static_link' else 'TimeRangeSplit'
        if (descriptor.get('native_split_kind') != native_family or json.loads(
            (source / 'staging/manifest.json').read_text()).get('split_kind')
                != native_family):
            raise ValueError('Frozen link table used the wrong native routing')
    table = kumo.TrainingTable(
        query=descriptor['query'], staging_dir=source / 'staging',
        train_table_path=source / 'staging/tt.parquet',
        manifest_path=source / 'staging/manifest.json',
        required_stages=tuple(descriptor['required_stages']),
        generation_plan=from_json_dict(descriptor['generation_plan'],
                                       TrainingTableGenerationPlan))
    if to_json_dict(table.generation_plan) != descriptor['generation_plan']:
        raise ValueError('Wheel changed the frozen control generation plan')
    rows = _rows(table, output_dir)
    validate_reimport(
        case, expected_rows if expected_rows is not None else shuffle_targets(
            case, normal['rows'], seed), rows)
    validate_chronology(case, normal['manifest'], rows)
    if hashes != _control_staging_hashes(source / 'staging',
                                         link=case.endswith('link')):
        raise ValueError('Export changed the frozen control fingerprint')
    truth, prediction_rows = prediction_context(
        case, rows, normal['profile_dir'], source_known=known,
        ranked_links=identity.get('ranked_links', False))
    result = dict(
        normal, normal_table=table, generation_plan=table.generation_plan,
        rows=rows, truth=truth, prediction_rows=prediction_rows,
        identity=identity, frozen_control=dict(
            producer_version=descriptor['producer_version'],
            descriptor_sha256=hashlib.sha256(
                descriptor_path.read_bytes()).hexdigest()))
    if seed is None:
        result['final_isolation'] = dict(
            directory=str((source / 'final').resolve()),
            sha256=descriptor['final_isolation']['sha256'])
        verify_final_isolation(result)
        return result
    return (_bind_control_final(normal, result,
                                Path(output_dir) / 'final-isolation')
            if 'final_isolation' in normal else result)
