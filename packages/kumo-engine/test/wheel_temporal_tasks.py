#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure forecasting and link learning with external reserved answers."""

import json
import math
import random
import runpy
import shutil
import statistics
from datetime import datetime, timedelta
from pathlib import Path

START = datetime(2024, 1, 1)
TASK_TYPES = {
    'forecasting': 'forecasting',
    'static_link': 'static_link_prediction',
    'temporal_link': 'temporal_link_prediction'
}


def date(day):
    return (START + timedelta(days=day)).isoformat()


def make_fixture(task, independent, seed=0):
    """Use matching fixed data for each seed and both measured wheels."""
    if task not in TASK_TYPES:
        raise ValueError(f'Unknown temporal task: {task}')
    rng = random.Random(73 + seed)
    ids = list(range(60 if task == 'forecasting' else 100))
    catalog = list(range(100))
    item_orders = ([rng.sample(range(10), 10) for _ in range(10)]
                   if task != 'forecasting' and not independent else None)
    anchor = 96 if task == 'forecasting' else 48
    events, targets, known = [], [], {}
    last = []
    for entity in ids:
        if task == 'forecasting':
            phases = [
                rng.randrange(96) if independent else (day + entity) % 96
                for day in range(anchor + 12)
            ]
            values = [.25 * min(phase, 96 - phase) for phase in phases]
            events.extend(
                dict(ENTITY_ID=entity, DAY=day, VALUE=value)
                for day, value in enumerate(values[:anchor]))
            targets.append(values[anchor:])
            last.append([values[anchor - 1]] * 12)
            continue
        pool = catalog if independent else list(
            range((entity % 10) * 10, (entity % 10 + 1) * 10))
        if task == 'static_link':
            # Give each item equal observed history and reserved coverage.
            chosen = (rng.sample(pool, 9) if item_orders is None else [
                pool[item_orders[entity % 10][(entity // 10 + offset) % 10]]
                for offset in range(9)
            ])
            observed, reserved = chosen[:6], chosen[6:]
            known[str(entity)] = observed
            events.extend(
                dict(ENTITY_ID=entity, ITEM_ID=item) for item in observed)
        else:
            for day in range(anchor):
                events.extend(
                    dict(ENTITY_ID=entity, DAY=day, ITEM_ID=item)
                    for item in rng.sample(pool, 3))
            # Equal item counts prevent arbitrary within-group rankings from
            # changing the cohort's score. Control answers stay random.
            reserved = (rng.sample(pool, 3) if item_orders is None else [
                pool[item_orders[entity % 10][(entity // 10 + offset) % 10]]
                for offset in range(3)
            ])
        targets.append([int(item in reserved) for item in catalog])
    fixture = dict(entity_ids=ids, catalog=catalog, anchor=anchor,
                   events=events, targets=targets, known=known)
    if task == 'forecasting':
        fixture.update(horizons=[date(anchor + offset) for offset in range(12)],
                       baseline_predictions=last)
    else:
        popularity = [0] * len(catalog)
        for event in events:
            popularity[event['ITEM_ID']] += 1
        fixture['baseline_predictions'] = [list(popularity) for _ in ids]
        if task == 'static_link':
            fixture['control_eligible'] = [[
                item not in known[str(entity)] for item in catalog
            ] for entity in ids]
            for entity in ids:
                for item in known[str(entity)]:
                    fixture['baseline_predictions'][entity][item] = -1
    return fixture


def constant_median_diagnostic(training_values, targets):
    median = float(statistics.median(training_values))
    error = statistics.mean(
        abs(value - median) for row in targets for value in row)
    return dict(training_median=median, training_median_mae=error)


def forecast_history(fixture, entity_ids):
    anchor = fixture['anchor']
    values = {}
    if len(set(entity_ids)) != len(entity_ids):
        raise ValueError('Forecast history has duplicate requested entities')
    for row in fixture['events']:
        key = row['ENTITY_ID'], row['DAY']
        if (key in values or row['ENTITY_ID'] not in fixture['entity_ids']
                or not 0 <= row['DAY'] < anchor
                or not math.isfinite(row['VALUE'])):
            raise ValueError('Forecast history contains invalid observed data')
        values[key] = row['VALUE']
    try:
        return [[values[entity, day] for day in range(anchor - 12, anchor)]
                for entity in entity_ids]
    except KeyError as error:
        raise ValueError(
            'Forecast history is missing observed values') from error


def validate_forecast_inputs(fixture, entity_ids, rows):
    history = dict(zip(entity_ids, forecast_history(fixture, entity_ids)))
    if (len(rows) != len(entity_ids) or {row.get('ENTITY1')
                                         for row in rows} != set(entity_ids)
            or any(
                set(row) != {'ENTITY1', 'TIMESTAMP', 'autoregression_labels'}
                or row['TIMESTAMP'] != START + timedelta(days=fixture['anchor'])
                or row['autoregression_labels'] != history[row['ENTITY1']]
                for row in rows)):
        raise ValueError(
            'Forecast prediction inputs differ from observed history')
    return 'PASS'


def canonical_stage(task, rows):
    rows = json.loads(
        json.dumps(rows, default=lambda value: value.isoformat(),
                   allow_nan=False))
    if task != 'forecasting':
        for row in rows:
            targets = row.get('TARGET1')
            if isinstance(targets, list):
                order = sorted(range(len(targets)), key=targets.__getitem__)
                row['TARGET1'] = [targets[index] for index in order]
                frequencies = row.get('__kumo_internal_frequency_column__')
                if isinstance(frequencies, list):
                    if len(frequencies) != len(targets):
                        raise ValueError(
                            'Target frequencies have a different size')
                    row['__kumo_internal_frequency_column__'] = [
                        frequencies[index] for index in order
                    ]
            if isinstance(row.get('exclude_target_keys'), list):
                row['exclude_target_keys'].sort()
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def save_stages(task, staging_dir, directory):
    import pyarrow.parquet as pq

    fingerprint = runpy.run_path(
        str(Path(__file__).with_name('wheel_task_checks.py')))['fingerprint']
    destination = Path(directory) / 'stage-seeds'
    destination.mkdir(parents=True)
    stages, hashes = {}, {}
    for stage in ('train', 'val', 'test'):
        frame = pq.read_table(Path(staging_dir) / f'{stage}_seeds')
        pq.write_table(frame, destination / f'{stage}.parquet')
        stages[stage] = frame.to_pylist()
        if not stages[stage]:
            raise ValueError(f'Missing generated stage: {stage}')
        hashes[stage] = fingerprint(canonical_stage(task, stages[stage]))
    return stages, hashes


def validate_chronology(stages, horizon, anchor):
    """Reject complete target windows that cross later stages or answers."""
    previous_end = None
    for name in ('train', 'val', 'test'):
        rows = stages.get(name, [])
        if not rows:
            raise ValueError(f'Missing chronological stage: {name}')
        first = min(row['DAY'] for row in rows)
        end = max(row['DAY'] + horizon for row in rows)
        if previous_end is not None and previous_end > first:
            raise ValueError(f'Target windows overlap the {name} stage')
        if end > anchor:
            raise ValueError('Target windows include reserved answers')
        previous_end = end
    return 'PASS'


def validate_temporal_sampling(task, runtime, trainer, schema):
    """Make sure that saved inputs enable the native seed-time cutoff."""
    metadata = runtime.get('task', {})
    columns = runtime.get('table', {}).get('columns', {})
    if (metadata.get('task_type') != TASK_TYPES[task]
            or metadata.get('is_temporal') is not True
            or columns.get('timestamp') != 'TIMESTAMP' or schema.get(
                'node_types', {}).get('events', {}).get('time_column') != 'TS'):
        raise ValueError('The temporal sampler lacks event or seed timestamps')
    strategy = trainer.get('sampler_kwargs', {}).get('temporal_strategy')
    if strategy != 'last':
        raise ValueError('The sampler does not select the temporal cutoff')
    return dict(use_seed_timestamps=True, temporal_strategy=strategy,
                seed_timestamp_column='TIMESTAMP', event_time_column='TS',
                evidence='saved_runtime_manifest_and_trainer_configuration')


def align_predictions(task, frame, fixture):
    """Require every entity and output coordinate before metric calculation."""
    import numpy as np
    import pandas as pd

    if task == 'forecasting':
        key, value = 'TIMESTAMP', 'TARGET_PRED'
        coordinates = pd.to_datetime(fixture['horizons'])
        frame = frame.copy()
        frame[key] = pd.to_datetime(frame[key])
    else:
        key, value = 'CLASS', 'SCORE'
        coordinates = fixture['catalog']
    if frame.duplicated(['ENTITY', key]).any():
        raise ValueError('Prediction contains duplicate output coordinates')
    expected = {(entity, coordinate)
                for entity in fixture['entity_ids']
                for coordinate in coordinates}
    if set(zip(frame['ENTITY'], frame[key])) != expected:
        raise ValueError('Prediction population differs from reserved inputs')
    matrix = frame.pivot(index='ENTITY', columns=key, values=value).reindex(
        index=fixture['entity_ids'], columns=coordinates).to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError('Predictions must be finite')
    if task == 'static_link':
        for index, entity in enumerate(fixture['entity_ids']):
            for item in fixture['known'].get(str(entity), []):
                matrix[index,
                       fixture['catalog'].index(item)] = matrix[index].min() - 1
    return matrix


def make_plan(task, seed):
    from kumoapi.encoder import Index, Numerical
    from kumoapi.model_plan import MissingType, NumNeighborsConfig

    frozen = runpy.run_path(str(Path(__file__).with_name('wheel_learning.py')))
    plan = frozen['make_plan'](seed)
    metric = 'mae' if task == 'forecasting' else 'recall@5'
    plan.training_job.metrics = [metric]
    plan.training_job.tune_metric = metric
    plan.optimization.loss = [
        'mae' if task == 'forecasting' else 'cross_entropy'
    ]
    plan.optimization.majority_sampling_ratio = MissingType.VALUE
    plan.optimization.weight_mode = MissingType.VALUE
    plan.neighbor_sampling.sample_from_entity_table = MissingType.VALUE
    plan.column_processing.encoder_overrides = {
        'ents.GROUP': Index(min_occ=1, na_strategy='separate')
    }
    # Exhaust these small graphs so batching cannot change sampled neighbors.
    plan.neighbor_sampling.num_neighbors = [
        NumNeighborsConfig(hop1={'default': -1}, hop2={'default': -1})
    ]
    if task == 'forecasting':
        plan.model_architecture.past_encoder = ['decomposed']
        plan.model_architecture.handle_new_entities = True
        plan.column_processing.encoder_overrides.update({
            'events.VALUE':
            Numerical(scaler=None, na_strategy='mean'),
            'train_table.TARGET':
            Numerical(scaler=None, na_strategy='mean')
        })
    else:
        plan.model_architecture.module = 'shallow_rhs'
        plan.model_architecture.target_embedding_mode = ['lookup']
        plan.model_architecture.output_embedding_dim = [16]
        plan.model_architecture.handle_new_target_entities = False
    return plan


def run_case(task, seed, independent, directory, *, calibration_budget=None):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    from kumoapi.graph import Edge
    from kumoapi.json_serde import to_json_dict
    from kumoapi.model_plan import TrainingTableGenerationPlan
    from kumoapi.typing import Stype

    import kumo.engine as kumo

    checks = runpy.run_path(
        str(Path(__file__).with_name('wheel_task_checks.py')))
    frozen = runpy.run_path(str(Path(__file__).with_name('wheel_learning.py')))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    data = directory / 'data'
    data.mkdir()
    fixture = make_fixture(task, independent, seed)
    (directory / 'truth.json').write_text(json.dumps(fixture, indent=2))
    ids = fixture['entity_ids']
    pd.DataFrame({
        'ID': ids,
        'GROUP': [entity % 10 for entity in ids]
    }).to_parquet(data / 'ents.parquet', index=False)
    events = pd.DataFrame(fixture['events'])
    events['ID'] = np.arange(len(events))
    temporal = task != 'static_link'
    if temporal:
        if events['DAY'].max() >= fixture['anchor']:
            raise ValueError('Reserved answers reached the observed graph')
        events['TS'] = pd.to_datetime([date(day) for day in events.pop('DAY')
                                       ]) + pd.Timedelta(hours=12)
    else:
        observed = {(row['ENTITY_ID'], row['ITEM_ID'])
                    for row in fixture['events']}
        if any(target and (entity, item) in observed
               for entity, row in enumerate(fixture['targets'])
               for item, target in enumerate(row)):
            raise ValueError('Reserved links reached the observed graph')
    events.to_parquet(data / 'events.parquet', index=False)
    connector = kumo.FileConnector(root_dir=str(data))
    kumo.configure(scratch_dir=directory / 'scratch')
    entities = kumo.Table.from_source_table(
        connector['ents.parquet'],
        primary_key='ID').infer_metadata().with_column('GROUP',
                                                       stype=Stype.categorical)
    event_table = kumo.Table.from_source_table(
        connector['events.parquet'], primary_key='ID',
        **({
            'time_column': 'TS'
        } if temporal else {})).infer_metadata().with_column(
            'ENTITY_ID', stype=Stype.ID)
    tables = {'ents': entities, 'events': event_table}
    edges = [Edge('events', 'ENTITY_ID', 'ents')]
    if task != 'forecasting':
        pd.DataFrame({
            'ID': fixture['catalog']
        }).to_parquet(data / 'items.parquet', index=False)
        tables['items'] = kumo.Table.from_source_table(
            connector['items.parquet'], primary_key='ID').infer_metadata()
        tables['events'] = event_table.with_column('ITEM_ID', stype=Stype.ID)
        edges.append(Edge('events', 'ITEM_ID', 'items'))
    graph = kumo.Graph(tables, edges)
    snapshot = graph.snapshot()
    query_text = (
        'PREDICT SUM(events.VALUE, 0, 1, days) FORECAST 12 TIMEFRAMES '
        'FOR EACH ents.ID'
        if task == 'forecasting' else 'PREDICT LIST_DISTINCT(events.ITEM_ID' +
        (', 0, 1, days)' if temporal else ')') + ' RANK TOP 5 FOR EACH ents.ID')
    query = kumo.PredictiveQuery(graph, query_text)
    if temporal:
        boundaries = [12, 60, 84, 96
                      ] if task == 'forecasting' else [1, 24, 36, 48]
        ranges = [[date(start), date(end)]
                  for start, end in zip(boundaries, boundaries[1:])]
        horizon = 12 if task == 'forecasting' else 1
        generation = TrainingTableGenerationPlan(
            split=(f'TimeRangeSplit({ranges!r}, timeframe={horizon}, '
                   'flexible_boundaries=False)'), timeframe_step=1,
            **({
                'lag_timesteps': 12,
                'year_over_year': False
            } if task == 'forecasting' else {}))
    else:
        generation = TrainingTableGenerationPlan(
            split='RandomLinkSplit([0.6, 0.2, 0.2])')
    table = query.generate_training_table(snapshot, plan=generation)
    stages, stage_hashes = save_stages(task, table.staging_dir, directory)
    chronology = 'PASS'
    training_values = []
    if temporal:
        actual = {
            (row['ENTITY_ID'], row['DAY']): row['VALUE']
            for row in fixture['events']
        } if task == 'forecasting' else {}
        for stage, rows in stages.items():
            for row in rows:
                row['DAY'] = int(
                    (pd.Timestamp(row['TIMESTAMP']) - pd.Timestamp(START)).days)
                if task == 'forecasting':
                    expected = [
                        actual.get((row['ENTITY1'], row['DAY'] + offset))
                        for offset in range(12)
                    ]
                    if list(row.get('future_labels',
                                    [])) != expected or None in expected:
                        raise ValueError(
                            'Forecast targets differ from observed history')
                    if stage == 'train':
                        training_values.extend(expected)
        chronology = validate_chronology(stages, horizon, fixture['anchor'])
    plan = make_plan(task, seed)
    fault = checks['apply_training_budget'](plan, calibration_budget)
    (directory / 'requested-plan.json').write_text(
        json.dumps(plan.as_json(), indent=2))
    (directory / 'generation-plan.json').write_text(
        json.dumps(to_json_dict(generation), indent=2))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    trainer = kumo.Trainer(plan)
    fitted = trainer.fit(table, snapshot=snapshot)
    [metadata
     ] = trainer.scratch_dir.glob('runs/*-fit-*/task_dataset/metadata.json')
    task_metadata = json.loads(metadata.read_text())
    if task_metadata['task_type'] != TASK_TYPES[task]:
        raise ValueError('The fitted task type differs from the requested task')
    shutil.copyfile(metadata, directory / 'task-metadata.json')
    for name in ('resolved_model_plan', 'trainer_config', 'encoders',
                 'target_encoder'):
        shutil.copyfile(fitted.artifact_paths[name], directory / f'{name}.json')
    if temporal:
        runtime_path = metadata.with_name('runtime_manifest.json')
        schema_path = metadata.with_name('feature_schema.json')
        sampling = validate_temporal_sampling(
            task, json.loads(runtime_path.read_text()),
            json.loads(
                Path(fitted.artifact_paths['trainer_config']).read_text()),
            json.loads(schema_path.read_text()))
        shutil.copyfile(runtime_path, directory / 'runtime-manifest.json')
        shutil.copyfile(schema_path, directory / 'feature-schema.json')
        (directory / 'temporal-sampling.json').write_text(
            json.dumps(sampling, indent=2))
    [selected] = trainer.scratch_dir.glob(
        'runs/*-fit-*/models/*/automl/version_*/model_plan.json')
    selected_plan = json.loads(selected.read_text())
    frozen['verify_plan'](plan.as_json(), selected_plan)
    shutil.copyfile(selected, directory / 'selected-plan.json')
    (directory / 'fit-metrics.json').write_text(
        json.dumps(fitted.metrics, indent=2))

    def predict(model, selected_ids, label):
        if task == 'forecasting':
            selected_query = query
            if selected_ids != ids:
                members = ', '.join(str(entity) for entity in selected_ids)
                selected_query = kumo.PredictiveQuery(
                    graph, f'{query_text} WHERE ents.ID IS IN ({members})')
            population_table = selected_query.generate_prediction_table(
                snapshot,
                prediction_time=START + timedelta(days=fixture['anchor']),
                plan=model.training_table_generation_plan)
        else:
            population = pd.DataFrame({'ENTITY1': selected_ids})
            if temporal:
                population['TIMESTAMP'] = pd.Timestamp(date(fixture['anchor']))
            path = data / f'{label}.parquet'
            population.to_parquet(path, index=False)
            population_table = kumo.PredictionTable.from_source_table(
                connector[path.name], pq=query, snapshot=snapshot,
                plan=model.training_table_generation_plan)
        (directory / f'{label}-query.txt').write_text(population_table.query)
        staged = pq.read_table(population_table.table_path)
        pq.write_table(staged, directory / f'{label}-inputs.parquet')
        if task == 'forecasting':
            validate_forecast_inputs(fixture, selected_ids, staged.to_pylist())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        frame = model.predict(
            population_table, snapshot=snapshot,
            **({} if task == 'forecasting' else {
                'num_classes_to_return': len(fixture['catalog'])
            })).frame
        frame.to_parquet(directory / f'{label}.parquet', index=False)
        selected_fixture = dict(fixture, entity_ids=selected_ids)
        matrix = align_predictions(task, frame, selected_fixture)
        return checks['verify_predictions'](
            selected_ids, sorted(frame['ENTITY'].unique().tolist()), matrix)

    before = predict(trainer, ids, 'predictions')
    partial = predict(trainer, ids[:len(ids) // 2], 'partial-predictions')
    partial_delta = checks['verify_reload'](before[:len(partial)], partial)
    restored = kumo.Trainer.load(fitted.bundle_dir)
    after = predict(restored, ids, 'reloaded-predictions')
    reload_delta = checks['verify_reload'](before, after)
    row = dict(
        metric='mae' if task == 'forecasting' else 'recall@5',
        targets=fixture['targets'], predictions=before.tolist(),
        baseline_predictions=fixture['baseline_predictions'], entity_ids=ids,
        data_sha256=checks['fingerprint'](dict(
            fixture=fixture, query=query_text,
            generation=to_json_dict(generation),
            stages=stage_hashes)), stage_sha256=stage_hashes,
        plan_sha256=checks['fingerprint'](selected_plan), training_seed=seed,
        fit_id=f'{trainer.scratch_dir.name}/{metadata.parent.parent.name}',
        prediction_input_mode=('generated_forecast_history'
                               if task == 'forecasting' else 'custom_entities'),
        task_type=TASK_TYPES[task],
        reload_max_abs=reload_delta, partial_max_abs=partial_delta, checks=dict(
            fit='PASS', prediction='PASS', partial_prediction='PASS',
            reload='PASS', hidden_answers='PASS', chronology=chronology))
    if task == 'static_link':
        row['control_eligible'] = fixture['control_eligible']
    if task == 'forecasting':
        row['diagnostics'] = constant_median_diagnostic(training_values,
                                                        fixture['targets'])
        (directory / 'diagnostics.json').write_text(
            json.dumps(row['diagnostics'], indent=2))
    if fault is not None:
        row['calibration_fault'] = fault
    return row
