# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure candidate learning on prepared scale-bench profiles."""

import gc
import json
import math
import random
import shutil
import statistics
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import wheel_learning as binary
import wheel_node_tasks as node
import wheel_temporal_tasks as temporal
from wheel_gpu_learning import (
    checkpoint_sha256,
    observe_gpu,
    verify_devices,
    verify_reload_identity,
)
from wheel_measurements import (
    Measurements,
    measure_recommendations,
    observe_training_work,
    verify_training_counts,
)
from wheel_scale_data import CASES
from wheel_task_checks import (
    _integer_ranked_ids,
    fingerprint,
    score,
    validate_ranked_link,
    verify_predictions,
)


def make_plan(case, seed):
    from kumoapi.model_plan import MissingType, NumNeighborsConfig

    plan = (binary.make_plan(seed) if case == 'binary' else node.make_plan(
        case, seed) if case in node.TASKS else temporal.make_plan(case, seed))
    plan.neighbor_sampling.num_neighbors = [
        NumNeighborsConfig(hop1={'default': 16}, hop2={'default': 16})
    ]
    plan.neighbor_sampling.sample_from_entity_table = MissingType.VALUE
    plan.column_processing.encoder_overrides = {
        key: value
        for key, value in plan.column_processing.encoder_overrides.items()
        if key.startswith('train_table.')
    }
    if case == 'temporal_link':
        plan.training_job.metrics = ['recall@10']
        plan.training_job.tune_metric = 'recall@10'
    return plan


def selected_training_step(path):
    from kumo_core.train.checkpoint import load_kumo_checkpoint

    step = load_kumo_checkpoint(path, map_location='cpu').get('global_step')
    if type(step) is not int or step <= 0:
        raise ValueError('Selected checkpoint has no completed training steps')
    return step


def normalize_prediction_time(case, frame, truth, time_column, *,
                              require_anchor=False):
    if case == 'static_link':
        return frame
    time_column = 'TIMESTAMP' if time_column is None else time_column
    reserved = {
        'ENTITY', 'TARGET', 'TARGET_PRED', 'CLASS', 'SCORE', 'PREDICTED'
    }
    if (not isinstance(time_column, str) or not time_column
            or time_column in reserved
            or case == 'forecasting' and time_column == 'ANCHOR_TIMESTAMP'):
        raise ValueError('Entity time role conflicts with prediction fields')
    if require_anchor and 'ANCHOR_TIMESTAMP' not in frame:
        raise ValueError('Prediction anchor metadata is missing')
    anchor = pd.Timestamp(truth['anchor'])
    if pd.isna(anchor) or ('ANCHOR_TIMESTAMP' in frame and not (pd.to_datetime(
            frame['ANCHOR_TIMESTAMP']) == anchor).all()):
        raise ValueError('Prediction anchor differs from evaluation inputs')
    names = ['TIMESTAMP', time_column]
    if case != 'forecasting':
        names.append('ANCHOR_TIMESTAMP')
    values = [
        pd.to_datetime(frame[name]) for name in dict.fromkeys(names)
        if name in frame
    ]
    if not values or any(not value.equals(values[0]) for value in values[1:]):
        raise ValueError('Prediction times are missing or aliases conflict')
    if case != 'forecasting' and not (values[0] == anchor).all():
        raise ValueError('Prediction anchor differs from evaluation inputs')
    result = frame.copy()
    result['TIMESTAMP'] = values[0]
    return result


def prediction_matrix(case, frame, truth):
    ids = truth['entity_ids']
    if (case not in ('forecasting', 'static_link')
            and ('TIMESTAMP' not in frame or not (pd.to_datetime(
                frame['TIMESTAMP']) == pd.Timestamp(truth['anchor'])).all())):
        raise ValueError('Prediction anchors differ from evaluation inputs')
    if case == 'binary':
        if (not frame['ENTITY'].is_unique or set(frame['ENTITY']) != set(ids)):
            raise ValueError('Binary prediction entities are missing or '
                             'repeated')
        ordered = frame.set_index('ENTITY').loc[ids].reset_index()
        if ('TARGET' in ordered
                and ordered['TARGET'].tolist() != truth['targets']):
            raise ValueError('Held-out answers differ from the native targets')
        probabilities = verify_predictions(
            ids, ids, ordered[['False_PROB', 'True_PROB']].to_numpy())
        if (not ((probabilities >= 0) & (probabilities <= 1)).all()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6,
                                   rtol=1e-7)):
            raise ValueError('Invalid binary probabilities')
        if ('TARGET_PRED' not in ordered or ordered['TARGET_PRED'].isna().any()
                or not (ordered['TARGET_PRED'] == (ordered['True_PROB']
                                                   > .5)).all()):
            raise ValueError('Binary predictions differ from their threshold')
        values = ordered[['True_PROB']].to_numpy()
    elif case in node.TASKS:
        values = node.prediction_matrix(case, frame, ids, truth['classes'])
    else:
        values = temporal.align_predictions(case, frame, truth)
    return verify_predictions(ids, ids, values)


def prediction_inputs(prepared, case, ids):
    """Select held-out inputs without future answers or target columns."""
    allowed = {'ENTITY1', 'TIMESTAMP'}
    if case == 'forecasting':
        allowed.add('autoregression_labels')
    expected = {
        row['ENTITY1']: row
        for row in prepared['rows'] if row['SPLIT'] == 2
    }
    rows = prepared['prediction_rows']
    if (len(rows) != len(expected)
            or len(expected) != len(prepared['truth']['entity_ids'])
            or {row['ENTITY1']
                for row in rows} != set(expected) or len(set(ids)) != len(ids)
            or not set(ids) <= set(expected)):
        raise ValueError('Prediction inputs differ from the evaluation '
                         'population')
    for row in rows:
        required = allowed if case != 'static_link' else {'ENTITY1'}
        if (not required <= set(row) or not set(row) <= allowed
                or any(value != expected[row['ENTITY1']].get(name)
                       for name, value in row.items())):
            raise ValueError('Prediction inputs expose answers or changed '
                             'history')
        if case == 'forecasting' and len(row['autoregression_labels']) != 12:
            raise ValueError('Forecast prediction needs twelve past values')
    indexed = {row['ENTITY1']: row for row in rows}
    return [indexed[entity] for entity in ids]


def verify_holdout(case, frame, truth):
    if case.endswith('link'):
        # Native link fits have no scalar holdout; score explicit predictions.
        return
    if frame is None:
        raise ValueError('Native scalar holdout is missing')
    prediction_matrix(case, frame, truth)
    if case in node.TASKS:
        node.verify_holdout_answers(case, frame, truth['entity_ids'],
                                    truth['classes'], truth['targets'])
    elif case == 'forecasting' and not np.array_equal(
            prediction_matrix(case, frame.assign(TARGET_PRED=frame['TARGET']),
                              truth), truth['targets']):
        raise ValueError('Forecast holdout answers differ from native targets')


def verify_runtime(case, runtime, schema, config):
    if runtime.get('task', {}).get('task_type') != CASES[case]['task_type']:
        raise ValueError('Fitted task type differs from the requested task')
    hidden = {'TARGET', 'TARGET1', 'LABEL', 'future_labels'}
    if not schema.get('node_types') or any(
        (hidden -
         ({'future_labels'} if case == 'forecasting' and name == 'train_table'
          else set())).intersection(item.get('feature_columns', []))
            for name, item in schema['node_types'].items()):
        raise ValueError('Held-out answers appear in model features')
    if case == 'static_link':
        return
    source, stamp = (('conversions', 'conversion_time') if case == 'binary' else
                     ('purchases',
                      'purchase_time') if case.endswith('link') else
                     ('orders', 'order_time'))
    if (runtime['task'].get('is_temporal') is not True or runtime.get(
            'table', {}).get('columns', {}).get('timestamp') != 'TIMESTAMP'
            or schema.get('node_types', {}).get(
                source, {}).get('time_column') != stamp or config.get(
                    'sampler_kwargs', {}).get('temporal_strategy') != 'last'):
        raise ValueError('Missing native seed timestamps or temporal cutoff')


def task_score(case, targets, predictions):
    if case == 'binary':
        values = verify_predictions(list(range(len(targets))),
                                    list(range(len(targets))), predictions)
        if values.shape[1] != 1:
            raise ValueError('Binary quality needs one positive score per '
                             'entity')
        frame = pd.DataFrame(
            dict(ENTITY=range(len(targets)), TARGET=targets,
                 True_PROB=values[:, 0], False_PROB=1 - values[:, 0]))
        return binary.score_predictions(frame, dict(enumerate(targets)))
    return score(CASES[case]['metric'], targets, predictions)


def ranked_baseline(truth):
    catalog = truth['catalog']
    popularity = np.asarray(truth['training_popularity'])
    if popularity.shape != (
            len(catalog), ) or not np.isfinite(popularity).all():
        raise ValueError('Training popularity differs from the catalogue')
    order = np.lexsort((catalog, -popularity))
    k = int(truth['metric'].split('@')[1])
    items, scores = [], []
    for excluded in truth.get('excluded_ids',
                              [[] for _ in truth['entity_ids']]):
        blocked = set(excluded)
        selected = list(
            islice((index for index in order if catalog[index] not in blocked),
                   k))
        items.append([catalog[index] for index in selected])
        scores.append(popularity[selected].tolist())
    return dict(ranked_ids=items, ranked_scores=scores)


def ranked_task_score(row, *, baseline=False):
    from scalebench.metrics import ranking

    if baseline:
        row = dict(row, ranked_ids=row['baseline_ranked_ids'],
                   ranked_scores=row['baseline_ranked_scores'])
    validate_ranked_link(row)
    lengths = [len(items) for items in row['ranked_ids']]
    result = ranking(row['entity_ids'], row['target_ids'],
                     np.repeat(row['entity_ids'], lengths),
                     [item for items in row['ranked_ids'] for item in items],
                     [rank for length in lengths for rank in range(length)],
                     k=int(row['metric'].split('@')[1]))
    if result['n_with_predictions'] != len(row['entity_ids']):
        raise ValueError('Ranked scoring has missing prediction entities')
    return result[row['metric']]


def subset_truth(truth, selected_ids, positions):
    names = (('target_ids', 'excluded_ids') if truth.get('target_format')
             == 'ranked_link_v1' else ('targets', ))
    return dict(
        truth, entity_ids=selected_ids, **{
            name: [truth[name][positions[entity]] for entity in selected_ids]
            for name in names if name in truth
        })


def validate_profile_learning(rows, *, calibration=False,
                              seeds=tuple(range(5))):
    """Check useful learning for profile admission, without release approval."""
    normal = [row for row in rows if row['case'] == 'learnable']
    seeds = tuple(seeds)
    if seeds not in ((0, 1, 2), (0, 1, 2, 3, 4)):
        raise ValueError('Only the fixed PR or full seed roster is supported')
    if not normal:
        raise ValueError('Missing normal profile measurements')
    result = {}
    for case in sorted({row['task'] for row in normal}):
        selected = [row for row in normal if row['task'] == case]
        if (len(selected) != len(seeds) or {row['seed']
                                            for row in selected} != set(seeds)
                or any(
                    type(row['seed']) is not int or row['metric'] != CASES[case]
                    ['metric'] or not math.isfinite(row[key])
                    for row in selected
                    for key in ('score', 'baseline_score'))):
            raise ValueError('Profile admission needs all selected seed '
                             'scores')
        quality = statistics.mean(row['score'] for row in selected)
        baseline = statistics.mean(row['baseline_score'] for row in selected)
        useful = (baseline -
                  quality if CASES[case]['metric'] == 'mae' else quality -
                  baseline) > 0
        if not useful and not calibration:
            raise ValueError(
                f'{case}: mean learning does not beat its baseline')
        result[case] = dict(mean_score=quality, mean_baseline_score=baseline,
                            release_approved=False)
        if calibration:
            result[case]['useful_learning_passed'] = useful
    return result


def verify_calibration_plan(row, directory, fault):
    from wheel_task_checks import weakened_training_plan

    detail = row.get('calibration_fault')
    original = make_plan(row['task'], row['seed']).as_json()
    if (not isinstance(detail, dict)
            or set(detail) != {'kind', 'budget', 'original_plan'}
            or any(detail[key] != fault[key] for key in ('kind', 'budget'))
            or detail['original_plan'] != original):
        raise ValueError('Missing or mismatched calibration fault provenance')
    expected = weakened_training_plan(original, fault['budget'])
    requested = json.loads((directory / 'requested-plan.json').read_text())
    selected = json.loads((directory / 'selected-model-plan.json').read_text())
    if requested != expected or fingerprint(selected) != row['plan_sha256']:
        raise ValueError('Actual calibration plan differs from declared fault')
    binary.verify_plan(requested, selected)


def run_case(prepared, case, seed, independent, directory, *,
             calibration_budget=None):
    """Fit one fresh model with a prepared normal or control table."""
    import pyarrow.parquet as pq
    import torch
    import wheel_task_checks as checks
    from kumoapi.jobs import MetadataField
    from kumoapi.json_serde import to_json_dict

    import kumo.engine as kumo

    if (case not in CASES or type(seed) is not int or seed not in range(5)
            or prepared['case'] != case
            or prepared['identity'].get('control_seed')
            != (seed if independent else None)):
        raise ValueError('Prepared profile, arm, or training seed differs')
    started = time.monotonic()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    input_dir = directory / 'prediction-inputs'
    input_dir.mkdir()
    output_dir = directory / 'prediction-outputs'
    output_dir.mkdir()
    truth = prepared['truth']
    ranked_links = truth.get('target_format') == 'ranked_link_v1'
    if ranked_links:
        if not case.endswith('link'):
            raise ValueError('Ranked output requires a link task')
        truth = dict(truth, metric=CASES[case]['metric'])
    ids = truth['entity_ids']
    positions = {entity: index for index, entity in enumerate(ids)}
    if len(ids) < 2:
        raise ValueError('Full and partial predictions need distinct '
                         'populations')
    prediction_inputs(prepared, case, ids)
    snapshot, query = prepared['snapshot'], prepared['query']
    if query.get_task_type().value != CASES[case]['task_type']:
        raise ValueError('Prepared query resolved to the wrong task type')
    plan = make_plan(case, seed)
    fault = checks.apply_training_budget(plan, calibration_budget)
    requested = plan.as_json()
    (directory / 'requested-plan.json').write_text(
        json.dumps(requested, indent=2))
    (directory / 'generation-plan.json').write_text(
        json.dumps(to_json_dict(prepared['generation_plan']), indent=2))
    stage_hashes = {
        name:
        fingerprint(
            temporal.canonical_stage(
                case,
                [row for row in prepared['rows'] if row['SPLIT'] == split]))
        for split, name in enumerate(('train', 'val', 'test'))
    }
    kumo.configure(scratch_dir=directory / 'scratch')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    evidence = {}
    measurements = Measurements(directory)
    try:
        with observe_gpu(evidence, strict_batches=True):
            trainer = kumo.Trainer(plan)
            with observe_training_work(evidence), measurements.stage(
                    'training'):
                fitted = trainer.fit(prepared['normal_table'],
                                     snapshot=snapshot)
            verify_training_counts(evidence, budget=calibration_budget)
            selected_step = selected_training_step(
                fitted.artifact_paths['selected_checkpoint'])
            evidence['expected_reload_checkpoint_sha256'] = checkpoint_sha256(
                fitted.artifact_paths['selected_checkpoint'])
            [metadata] = trainer.scratch_dir.glob(
                'runs/*-fit-*/task_dataset/metadata.json')
            dataset = metadata.parent
            task_metadata = json.loads(metadata.read_text())
            if task_metadata['task_type'] != CASES[case]['task_type']:
                raise ValueError('The fitted task type differs from the '
                                 'requested task')
            for name in ('metadata', 'runtime_manifest', 'feature_schema'):
                shutil.copyfile(dataset / f'{name}.json',
                                directory / f'{name}.json')
            for name in ('resolved_model_plan', 'trainer_config', 'encoders',
                         'target_encoder'):
                shutil.copyfile(fitted.artifact_paths[name],
                                directory / f'{name}.json')
            runtime = json.loads(
                (dataset / 'runtime_manifest.json').read_text())
            schema = json.loads((dataset / 'feature_schema.json').read_text())
            verify_runtime(
                case, runtime, schema,
                json.loads((directory / 'trainer_config.json').read_text()))
            entity = runtime['table']['seed_entity_table']
            if entity != task_metadata['entity_table_name']:
                raise ValueError('Fitted entity table differs from runtime')
            time_column = schema['node_types'][entity]['time_column']
            [selected] = trainer.scratch_dir.glob(
                'runs/*-fit-*/models/*/automl/version_*/model_plan.json')
            selected_plan = json.loads(selected.read_text())
            binary.verify_plan(requested, selected_plan)
            shutil.copyfile(selected, directory / 'selected-model-plan.json')
            progress = selected.with_name('train_progress.json')
            if progress.is_file():
                shutil.copyfile(progress, directory / progress.name)
            holdout = None
            if (fitted.holdout_path is not None
                    and fitted.holdout_truncated is not False):
                raise ValueError('Held-out predictions are missing or '
                                 'truncated')
            if fitted.holdout_path is not None:
                holdout = normalize_prediction_time(case, fitted.holdout_df(),
                                                    truth, time_column)
                holdout.to_parquet(directory / 'holdout.parquet', index=False)
            verify_holdout(case, holdout, truth)

            def predict(model, selected_ids, label, *, full_catalog=False,
                        measure=True):
                source = input_dir / f'{label}-inputs.parquet'
                population = prediction_inputs(prepared, case, selected_ids)
                selected_set = set(selected_ids)
                pd.DataFrame(population).to_parquet(source, index=False)
                connector = kumo.FileConnector(root_dir=str(source.parent))
                table = kumo.PredictionTable.from_source_table(
                    connector[source.name], pq=query, snapshot=snapshot,
                    plan=model.training_table_generation_plan)
                staged = pq.read_table(table.table_path).to_pylist()
                # Native prediction staging must retain the supplied population.
                selected_context = dict(
                    prepared, prediction_rows=staged, rows=[
                        row for row in prepared['rows']
                        if row['SPLIT'] != 2 or row['ENTITY1'] in selected_set
                    ], truth=dict(truth, entity_ids=selected_ids))
                prediction_inputs(selected_context, case, selected_ids)
                options = {}
                selected_truth = subset_truth(truth, selected_ids, positions)
                if case == 'binary':
                    options['binary_classification_threshold'] = .5
                elif case in node.TASKS and truth['classes']:
                    options['num_classes_to_return'] = len(truth['classes'])
                elif case.endswith('link'):
                    limit = len(truth['catalog'])
                    if ranked_links and not full_catalog:
                        limit = min(
                            limit, 500,
                            int(truth['metric'].split('@')[1]) + max(
                                map(len, selected_truth.get(
                                    'excluded_ids', [[]])), default=0) + 1)
                    options['num_classes_to_return'] = limit
                if case != 'static_link':
                    options['output_config'] = kumo.PredictionOutputConfig(
                        connector=kumo.FileConnector(root_dir=str(output_dir)),
                        output_paths={'predictions': label},
                        extra_fields=[MetadataField.ANCHOR_TIMESTAMP])
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                with measurements.stage(label) if measure else nullcontext():
                    started_prediction = time.monotonic()
                    result = model.predict(table, snapshot=snapshot, **options)
                    if ranked_links:
                        files = result.files
                        batches = (normalize_prediction_time(
                            case, batch, selected_truth, time_column,
                            require_anchor=True) for batch in ordered_batches(
                                files, directory / f'{label}-sort'))
                        values = ranked_from_batches(batches, selected_truth,
                                                     limit)
                        evidence.setdefault(
                            'ranked_prediction_calls', []).append(
                                dict(
                                    label=label, entities=len(selected_ids),
                                    raw_limit=limit,
                                    raw_rows=values['raw_rows'],
                                    files=list(files),
                                    full_catalog=full_catalog,
                                    seconds=time.monotonic() -
                                    started_prediction,
                                    fallback_ids=values['fallback_ids']))
                        if values['fallback_ids']:
                            if len(truth['catalog']) > 500:
                                raise ValueError(
                                    'Eligible topK remains ambiguous at the '
                                    'public 500-result limit')
                            fallback = predict(model, values['fallback_ids'],
                                               f'{label}-fallback',
                                               full_catalog=True, measure=False)
                            selected_positions = {
                                entity: index
                                for index, entity in enumerate(selected_ids)
                            }
                            for index, entity in enumerate(
                                    values['fallback_ids']):
                                position = selected_positions[entity]
                                for field in ('ranked_ids', 'ranked_scores'):
                                    values[field][position] = fallback[field][
                                        index]
                        return values
                    frame = result.frame
                frame = normalize_prediction_time(case, frame, truth,
                                                  time_column,
                                                  require_anchor=True)
                frame.to_parquet(directory / f'{label}.parquet', index=False)
                return prediction_matrix(case, frame, selected_truth)

            def compare(expected, actual, scope, population):
                if ranked_links:

                    def frame(values):
                        return pd.DataFrame(
                            dict(
                                ENTITY=np.repeat(population, [
                                    len(items) for items in values['ranked_ids']
                                ]), CLASS=[
                                    item for items in values['ranked_ids']
                                    for item in items
                                ], SCORE=[
                                    score for scores in values['ranked_scores']
                                    for score in scores
                                ]))

                    measured = measure_recommendations(frame(expected),
                                                       frame(actual))
                    evidence['consistency_measurements'].append(
                        dict(
                            stage='reload_and_predict'
                            if scope == 'reload' else 'prediction', scope=scope,
                            score_coordinates='entity_class', **measured))
                    return measured['max_abs_difference']
                else:
                    delta = checks.verify_reload(expected, actual)
                evidence['consistency_measurements'][-1]['scope'] = scope
                return delta

            predictions = predict(trainer, ids, 'predictions')
            repeated = predict(trainer, ids, 'repeat-predictions')
            repeat_delta = compare(predictions, repeated, 'repeat_full', ids)
            count = max(1, len(ids) // 2)
            partial = predict(trainer, ids[:count], 'partial-predictions')
            expected_partial = ({
                name: predictions[name][:count]
                for name in ('ranked_ids', 'ranked_scores')
            } if ranked_links else predictions[:count])
            partial_delta = compare(expected_partial, partial, 'partial',
                                    ids[:count])
            loaded = kumo.Trainer.load(fitted.bundle_dir)
            reloaded = predict(loaded, ids, 'reloaded-predictions')
            reload_delta = compare(predictions, reloaded, 'reload', ids)
        verify_devices(evidence,
                       ('training', 'prediction', 'reload_and_predict'))
        verify_reload_identity(evidence)
        if ranked_links:
            baseline = ranked_baseline(truth)
            scoring = dict(target_format='ranked_link_v1',
                           target_ids=truth['target_ids'],
                           catalog=truth['catalog'],
                           ranked_ids=predictions['ranked_ids'],
                           ranked_scores=predictions['ranked_scores'],
                           baseline_ranked_ids=baseline['ranked_ids'],
                           baseline_ranked_scores=baseline['ranked_scores'])
            if 'excluded_ids' in truth:
                scoring['excluded_ids'] = truth['excluded_ids']
            scored_row = dict(scoring, task=case, metric=CASES[case]['metric'],
                              entity_ids=ids)
            scoring.update(
                score=ranked_task_score(scored_row),
                baseline_score=ranked_task_score(scored_row, baseline=True))
        else:
            baseline = ([[value] for value in truth['baseline_predictions']]
                        if case == 'binary' else truth['baseline_predictions'])
            scoring = dict(
                targets=truth['targets'], predictions=predictions.tolist(),
                baseline_predictions=baseline,
                score=task_score(case, truth['targets'], predictions),
                baseline_score=task_score(case, truth['targets'], baseline))
        row = dict(
            task=case, seed=seed,
            case='independent' if independent else 'learnable',
            status='MEASURED', training_seed=seed,
            fit_id=f'{trainer.scratch_dir.name}/{metadata.parent.parent.name}',
            selected_checkpoint_global_step=selected_step,
            task_type=CASES[case]['task_type'], metric=CASES[case]['metric'],
            entity_ids=ids, **scoring, data_sha256=fingerprint(
                dict(identity=prepared['identity'], stages=stage_hashes)),
            plan_sha256=fingerprint(selected_plan), stage_sha256=stage_hashes,
            query=prepared['query_text'], classes=truth['classes'],
            entity_table=entity, entity_time_column=time_column, checks=dict(
                fit='PASS', prediction='PASS', repeat_prediction='PASS',
                partial_prediction='PASS', reload='PASS', hidden_answers='PASS',
                chronology='PASS'), partial_max_abs_difference=partial_delta,
            repeat_max_abs_difference=repeat_delta,
            reload_max_abs_difference=reload_delta, split_counts=[
                sum(row['SPLIT'] == stage for row in prepared['rows'])
                for stage in range(3)
            ], seconds=time.monotonic() - started,
            operation_measurements=measurements.rows, **evidence)
        if 'control_eligible' in truth:
            row['control_eligible'] = truth['control_eligible']
        if fault is not None:
            row['calibration_fault'] = fault
        if case == 'binary':
            row['auroc'] = row['score']
            if independent:
                row['control'] = dict(targets=truth['targets'],
                                      probabilities=predictions[:, 0].tolist())
        (directory / 'result.json').write_text(
            json.dumps(row, indent=2, allow_nan=False))
        return row
    finally:
        (directory / 'device.json').write_text(json.dumps(evidence, indent=2))
        gc.collect()


def ordered_batches(files, scratch, *, batch_rows=65536):
    """Spill-sort public prediction files without collecting the full result."""
    import duckdb

    scratch.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={
            'memory_limit': '128MB',
            'temp_directory': str(scratch)
    }) as connection:
        reader = connection.read_parquet(list(files)).order(
            'ENTITY, SCORE DESC, CLASS ASC').fetch_record_batch(batch_rows)
        for batch in reader:
            yield batch.to_pandas()


def ranked_from_batches(batches, truth, raw_limit):
    """Retain only K eligible items per entity from an entity-ordered stream."""
    ids, catalog = truth['entity_ids'], truth['catalog']
    k = int(truth['metric'].split('@')[1])
    if (not ids or len(set(ids)) != len(ids) or type(raw_limit) is not int
            or not 0 < raw_limit <= len(catalog)):
        raise ValueError('Invalid ranked prediction population or limit')
    positions, members = {
        entity: i
        for i, entity in enumerate(ids)
    }, set(catalog)
    blocked = truth.get('excluded_ids', [[] for _ in ids])
    items, scores = [None] * len(ids), [None] * len(ids)
    fallback, raw_rows = [], 0
    current, item_parts, score_parts = None, [], []

    def finish(entity):
        index = positions.get(entity)
        if index is None or items[index] is not None:
            raise ValueError('Repeated or unknown prediction entity')
        values, weights = np.concatenate(item_parts), np.concatenate(
            score_parts)
        excluded = set(blocked[index])
        if (len(values) != raw_limit or len(set(values)) != raw_limit
                or not set(values) <= members or not np.isfinite(weights).all()
                or len(excluded) != len(blocked[index])
                or not excluded <= members):
            raise ValueError('Invalid or incomplete ranked output coordinates')
        wanted = min(k, len(catalog) - len(excluded))
        eligible = np.asarray(
            [i for i, item in enumerate(values) if item not in excluded],
            dtype=np.int64)
        if not wanted:
            raise ValueError('No eligible prediction items')
        order = eligible[np.lexsort(
            (values[eligible], -weights[eligible]))][:wanted]
        items[index], scores[index] = values[order].tolist(
        ), weights[order].tolist()
        if (len(order) != wanted or raw_limit < len(catalog)
                and weights.min() == weights[order[-1]]):
            fallback.append(entity)

    for frame in batches:
        if not {'ENTITY', 'CLASS', 'SCORE'} <= set(frame):
            raise ValueError('Ranked prediction columns are missing')
        entities, values, weights = (frame[name].to_numpy()
                                     for name in ('ENTITY', 'CLASS', 'SCORE'))
        if not _integer_ranked_ids(entities) or not _integer_ranked_ids(values):
            raise ValueError('Prediction entity and item IDs must be integers')
        raw_rows += len(frame)
        starts = np.r_[0,
                       np.flatnonzero(entities[1:] != entities[:-1]) + 1,
                       len(frame)]
        for first, stop in zip(starts[:-1], starts[1:]):
            if first == stop:
                continue
            entity = entities[first]
            if current is not None and entity != current:
                finish(current)
                item_parts, score_parts = [], []
            current = entity
            item_parts.append(values[first:stop])
            score_parts.append(weights[first:stop])
            if sum(len(part) for part in item_parts) > raw_limit:
                raise ValueError('Too many prediction rows for an entity')
    if current is not None:
        finish(current)
    if any(value is None for value in items):
        raise ValueError('Prediction entities are missing')
    return dict(ranked_ids=items, ranked_scores=scores, fallback_ids=fallback,
                raw_rows=raw_rows)
