# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure four node-learning tasks without setting quality limits."""

import gc
import json
import random
import runpy
import statistics
import struct
import time
from pathlib import Path

_BINARY = runpy.run_path(str(Path(__file__).with_name('wheel_learning.py')))
TASKS = {
    'multiclass':
    ('multiclass_classification', 'accuracy', 'acc', 'cross_entropy'),
    'multilabel': ('multilabel_classification', 'ap_macro', 'ap_macro',
                   'binary_cross_entropy'),
    'multilabel_ranking':
    ('multilabel_ranking', 'recall@5', 'recall@5', 'cross_entropy'),
    'regression': ('regression', 'mae', 'mae', 'huber'),
}


def make_data(task, seed, independent):
    if task not in TASKS:
        raise ValueError(f'Unknown node task: {task}')
    values, split = _BINARY['create_data'](seed, False)
    labels = random.Random(1000 + seed)
    if task == 'multiclass':
        classes = ['high', 'low', 'middle']
    elif task == 'regression':
        classes = []
    else:
        classes = [f'label_{i:02}' for i in range(12)]
    targets = []
    for group in values['GROUP_ID']:
        if task == 'multiclass':
            target = ['low', 'middle', 'high'][min(group * 3 // 20, 2)]
        elif task == 'regression':
            target = 2 * group / 19 + labels.gauss(0, .02)
            target = struct.unpack('f', struct.pack('f', target))[0]
        else:
            bucket = group * 6 // 20
            target = [
                classes[bucket], classes[(bucket + 1) % 6], classes[bucket + 6]
            ]
        targets.append(target)
    if independent:
        permutation = random.Random(2000 + seed)
        for stage in (0, 1, 2):
            indices = [i for i, value in enumerate(split) if value == stage]
            shuffled = [targets[i] for i in indices]
            permutation.shuffle(shuffled)
            for index, target in zip(indices, shuffled):
                targets[index] = target
    values['LABEL'] = targets
    return values, split, classes


def encode_targets(task, targets, classes):
    if task == 'regression':
        return [[float(value)] for value in targets]
    return [[
        int(category == value if task == 'multiclass' else category in value)
        for category in classes
    ] for value in targets]


def training_targets(task, targets):
    # Custom node tables store multicategorical sets as pipe-separated strings.
    return ['|'.join(value)
            for value in targets] if task in ('multilabel',
                                              'multilabel_ranking') else targets


def baseline_predictions(task, targets, split, count):
    training = [target for target, stage in zip(targets, split) if stage == 0]
    if task == 'regression':
        row = [statistics.median(target[0] for target in training)]
    else:
        row = [statistics.mean(column) for column in zip(*training)]
        if task == 'multiclass':
            majority = max(range(len(row)), key=row.__getitem__)
            row = [int(index == majority) for index in range(len(row))]
    return [row[:] for _ in range(count)]


def prediction_matrix(task, frame, ids, classes):
    import numpy as np
    from wheel_task_checks import verify_predictions

    if set(frame['ENTITY']) != set(ids):
        raise ValueError('Prediction entities differ from the evaluation IDs')
    if task == 'regression':
        if not frame['ENTITY'].is_unique:
            raise ValueError('Duplicate regression prediction entities')
        matrix = frame.set_index('ENTITY').loc[ids, ['TARGET_PRED']].to_numpy()
    else:
        if (set(frame['CLASS']) != set(classes)
                or len(frame) != len(ids) * len(classes)
                or frame.duplicated(['ENTITY', 'CLASS']).any()):
            raise ValueError('Prediction classes are missing or duplicated')
        if task == 'multiclass':
            if not (frame.groupby('ENTITY')['PREDICTED'].sum() == 1).all():
                raise ValueError('Multiclass output needs one chosen class')
            chosen = frame.loc[frame['PREDICTED'], ['ENTITY', 'SCORE']]
            maxima = frame.groupby('ENTITY')['SCORE'].max()
            if not np.array_equal(chosen['SCORE'],
                                  maxima.loc[chosen['ENTITY']]):
                raise ValueError('Chosen class must have the highest score')
        elif task == 'multilabel':
            if not (frame['PREDICTED'] == (frame['SCORE'] > .5)).all():
                raise ValueError('Multilabel output differs from threshold')
        matrix = frame.pivot(index='ENTITY', columns='CLASS',
                             values='SCORE').loc[ids, classes].to_numpy()
        if not ((matrix >= 0) & (matrix <= 1)).all():
            raise ValueError('Invalid class probabilities')
        if task in ('multiclass', 'multilabel_ranking'):
            if not np.allclose(matrix.sum(axis=1), 1, atol=1e-6, rtol=1e-6):
                raise ValueError('Class probabilities do not sum to one')
    return verify_predictions(ids, ids, matrix)


def verify_hidden_answers(schema):
    if any('LABEL' in node['feature_columns']
           for node in schema['node_types'].values()):
        raise ValueError('Evaluation answers appear in model features')


def verify_holdout_answers(task, frame, ids, classes, expected):
    import numpy as np

    if task in ('multiclass', 'regression'):
        if not (frame.groupby('ENTITY')['TARGET'].nunique() == 1).all():
            raise ValueError('Repeated held-out answers disagree')
        values = frame.groupby('ENTITY')['TARGET'].first().loc[ids].tolist()
        recorded = encode_targets(task, values, classes)
    else:
        recorded = frame.pivot(index='ENTITY', columns='CLASS',
                               values='LABEL').loc[ids, classes].to_numpy()
    if not np.array_equal(recorded, expected):
        raise ValueError('Held-out answers differ from generated targets')


def make_plan(task, seed):
    from kumoapi.encoder import Index, MultiCategorical, Numerical
    from kumoapi.model_plan import MissingType, NumNeighborsConfig

    _, _, metric, loss = TASKS[task]
    plan = _BINARY['make_plan'](seed)
    plan.training_job.metrics = [metric]
    plan.training_job.tune_metric = metric
    plan.optimization.loss = [loss]
    plan.optimization.majority_sampling_ratio = MissingType.VALUE
    plan.optimization.weight_mode = MissingType.VALUE
    plan.neighbor_sampling.num_neighbors = [
        NumNeighborsConfig(hop1={'default': -1}, hop2={'default': -1})
    ]
    if task == 'regression':
        encoder = Numerical(scaler=None, na_strategy='mean')
    elif task == 'multiclass':
        encoder = Index(min_occ=1, na_strategy='separate')
    else:
        encoder = MultiCategorical(min_occ=1)
    plan.column_processing.encoder_overrides['train_table.TARGET'] = encoder
    return plan


def run_case(task, seed, independent, directory):
    import numpy as np
    import pandas as pd
    import torch
    from kumoapi.graph import Edge
    from kumoapi.train import TrainingTableSpec
    from kumoapi.typing import Stype
    from wheel_task_checks import fingerprint, score, verify_reload

    import kumo.engine as kumo

    started = time.monotonic()
    directory = Path(directory)
    directory.mkdir(parents=True)
    data = directory / 'data'
    data.mkdir()
    values, split, classes = make_data(task, seed, independent)
    targets = encode_targets(task, values['LABEL'], classes)
    ids = [entity for entity, stage in zip(values['ID'], split) if stage == 2]
    expected = [target for target, stage in zip(targets, split) if stage == 2]
    pd.DataFrame(values).to_parquet(data / 'entities.parquet', index=False)
    pd.DataFrame({
        'ID': range(20),
        'RISK': [i / 19 for i in range(20)],
    }).to_parquet(data / 'groups.parquet', index=False)
    pd.DataFrame({
        'ENTITY1': values['ID'],
        'TARGET1': training_targets(task, values['LABEL']),
        'SPLIT': split
    }).to_parquet(data / 'seeds.parquet', index=False)
    pd.DataFrame({
        'ENTITY': values['ID'],
        'TARGET': values['LABEL'],
        'SPLIT': split
    }).to_parquet(directory / 'truth.parquet', index=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    kumo.configure(scratch_dir=directory / 'scratch')
    connector = kumo.FileConnector(root_dir=str(data))
    stype = (Stype.numerical if task == 'regression' else Stype.categorical
             if task == 'multiclass' else Stype.multicategorical)
    graph = kumo.Graph(
        {
            'entities':
            kumo.Table.from_source_table(
                connector['entities'],
                primary_key='ID').infer_metadata().with_column(
                    'LABEL', stype=stype),
            'groups':
            kumo.Table.from_source_table(
                connector['groups'],
                primary_key='ID').infer_metadata().with_column(
                    'RISK', stype=Stype.numerical),
        }, [Edge('entities', 'GROUP_ID', 'groups')])
    snapshot = graph.snapshot()
    stats = json.loads(
        (snapshot.index_path / 'ingest_stats.json').read_text())['stats']
    if stats['total_nodes'] != 2020 or stats['total_edges'] != 4000:
        raise ValueError('Generated graph counts differ from the fixture')
    (directory / 'graph-stats.json').write_text(json.dumps(stats, indent=2))
    modifier = (' CLASSIFY TOP 12' if task == 'multilabel' else
                ' RANK TOP 5' if task == 'multilabel_ranking' else '')
    query = kumo.PredictiveQuery(
        graph, f'PREDICT entities.LABEL{modifier} FOR EACH entities.ID')
    task_type, metric, _, _ = TASKS[task]
    if query.get_task_type().value != task_type:
        raise ValueError('The query resolved to the wrong learning task')
    table = kumo.TrainingTable.from_source_table(
        connector['seeds'], TrainingTableSpec(split_col='SPLIT'), pq=query,
        snapshot=snapshot)
    plan = make_plan(task, seed)
    (directory / 'requested-plan.json').write_text(
        json.dumps(plan.as_json(), indent=2))
    trainer = kumo.Trainer(plan)
    result = trainer.fit(table, snapshot=snapshot)
    [metadata
     ] = trainer.scratch_dir.glob('runs/*-fit-*/task_dataset/metadata.json')
    actual_metadata = json.loads(metadata.read_text())
    if actual_metadata['task_type'] != task_type:
        raise ValueError('The fitted task type differs from the requested task')
    (directory / 'task-metadata.json').write_text(metadata.read_text())
    for name in ('resolved_model_plan', 'trainer_config', 'encoders',
                 'target_encoder'):
        (directory / f'{name}.json').write_text(
            Path(result.artifact_paths[name]).read_text())
    [selected] = trainer.scratch_dir.glob(
        'runs/*-fit-*/models/*/automl/version_*/model_plan.json')
    selected_plan = json.loads(selected.read_text())
    _BINARY['verify_plan'](plan.as_json(), selected_plan)
    (directory / 'selected-model-plan.json').write_text(selected.read_text())
    [schema_path] = trainer.scratch_dir.glob(
        'runs/*-fit-*/task_dataset/feature_schema.json')
    schema = json.loads(schema_path.read_text())
    verify_hidden_answers(schema)
    (directory / 'feature-schema.json').write_text(schema_path.read_text())
    if result.holdout_truncated is not False:
        raise ValueError('The evaluation output is missing or truncated')
    holdout = result.holdout_df()
    holdout.to_parquet(directory / 'holdout.parquet', index=False)
    prediction_matrix(task, holdout, ids, classes)
    verify_holdout_answers(task, holdout, ids, classes, expected)

    def predict(model, selected_ids, name):
        pd.DataFrame({
            'ENTITY1': selected_ids
        }).to_parquet(data / f'{name}.parquet', index=False)
        prediction_table = kumo.PredictionTable.from_source_table(
            connector[name], pq=query, snapshot=snapshot)
        options = {'num_classes_to_return': len(classes)} if classes else {}
        frame = model.predict(prediction_table, snapshot=snapshot,
                              **options).frame
        frame.to_parquet(directory / f'{name}.parquet', index=False)
        return prediction_matrix(task, frame, selected_ids, classes)

    predictions = predict(trainer, ids, 'predictions')
    partial = predict(trainer, ids[:37], 'partial-predictions')
    partial_difference = verify_reload(predictions[:37], partial)
    loaded = kumo.Trainer.load(result.bundle_dir)
    reloaded = predict(loaded, ids, 'reloaded-predictions')
    reload_difference = verify_reload(predictions, reloaded)
    baseline = baseline_predictions(task, targets, split, len(ids))
    row = dict(
        fit_id=f'{trainer.scratch_dir.name}/{metadata.parent.parent.name}',
        metric=metric, targets=expected, predictions=predictions.tolist(),
        baseline_predictions=baseline, entity_ids=ids, training_seed=seed,
        data_sha256=fingerprint([values, split, classes
                                 ]), plan_sha256=fingerprint(selected_plan),
        task_type=task_type, score=score(metric, expected, predictions),
        baseline_score=score(metric, expected, baseline), classes=classes,
        query=query.query, metrics=result.metrics, split_counts=[
            split.count(stage) for stage in (0, 1, 2)
        ], seconds=round(time.monotonic() - started,
                         3), checks=dict(fit='PASS', prediction='PASS',
                                         partial_prediction='PASS',
                                         reload='PASS', hidden_answers='PASS'),
        partial_max_abs_difference=partial_difference,
        reload_max_abs_difference=reload_difference)
    (directory / 'result.json').write_text(
        json.dumps(row, indent=2, allow_nan=False))
    del trainer, result, loaded
    gc.collect()
    return row
