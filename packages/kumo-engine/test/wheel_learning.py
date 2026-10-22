#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure learning on generated data. Limits remain provisional."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import statistics
import sys
import time
import traceback
import uuid
from pathlib import Path

SETTINGS = {
    'entities': 2000,
    'groups': 20,
    'epochs': 10,
    'steps_per_epoch': 20,
    'threads': 4,
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,
                                     sort_keys=True).encode()).hexdigest()


def summarize(rows, seeds, limits=None):
    expected = {(seed, case)
                for seed in seeds
                for case in ('learnable', 'independent')}
    if not seeds or len(rows) != len(expected) or {(row['seed'], row['case'])
                                                   for row in rows} != expected:
        raise ValueError('Every seed needs one result for each case')
    if any(not math.isfinite(row['auroc']) or not 0 <= row['auroc'] <= 1
           for row in rows):
        raise ValueError('Invalid AUROC score')
    means = {
        case: statistics.mean(row['auroc'] for row in rows
                              if row['case'] == case)
        for case in ('learnable', 'independent')
    }
    if limits is not None:
        if means['learnable'] < limits['learnable_min'] - 1e-12:
            raise ValueError(f'learnable AUROC is too low: {means}')
        if abs(means['independent'] -
               .5) > limits['independent_distance'] + 1e-12:
            raise ValueError(f'independent AUROC is outside its band: {means}')
    return means


def validate_learning(report, baseline):
    limits = baseline['limits']
    if any(not math.isfinite(limits[key]) or not 0 <= limits[key] <= 1
           for key in ('learnable_min', 'independent_distance', 'max_mean_drop',
                       'max_seed_drop')):
        raise ValueError('Invalid learning limit')
    if limits['learnable_min'] <= .5 + limits['independent_distance']:
        raise ValueError(
            'Learning limits must separate signal from independent labels')
    for field in ('source_sha256', 'settings', 'configuration',
                  'dependencies_sha256', 'seeds'):
        if report.get(field) != baseline[field]:
            raise ValueError(
                f'Learning evidence differs from baseline: {field}')
    return summarize(report['runs'], report['seeds'], baseline['limits'])


def compare_results(reference, candidate, baseline, measurement=False):
    if not measurement and baseline.get('validation',
                                        {}).get('status') != 'PASS':
        raise ValueError('Reference calibration needs passing reserved runs')
    if reference['wheel'] != baseline['reference_wheel']:
        raise ValueError('Reference wheel differs from the quality baseline')
    status = 'MEASURED' if measurement else 'PASS'
    for result in (reference, candidate):
        if result.get('status') != status or result['learning'].get(
                'status') != status:
            raise ValueError('Both wheels need passing learning evidence')
        validate_learning(result['learning'], baseline)
    before, after = reference['learning'], candidate['learning']
    for field in ('python', 'platform'):
        if before.get(field) != after.get(field):
            raise ValueError(f'Wheel environments differ: {field}')
    old = {(row['seed'], row['case']): row for row in before['runs']}
    drops = []
    for row in after['runs']:
        previous = old[row['seed'], row['case']]
        for field in ('training_seed', 'data_sha256', 'plan_sha256'):
            if previous[field] != row[field]:
                raise ValueError(f'Unmatched learning runs: {field}')
        if row['case'] == 'learnable':
            drops.append(previous['auroc'] - row['auroc'])
    mean_drop = statistics.mean(drops)
    if mean_drop > baseline['limits']['max_mean_drop'] + 1e-12 or max(
            drops) > baseline['limits']['max_seed_drop'] + 1e-12:
        raise ValueError(
            f'AUROC drop exceeds measured reference variation: {drops}')
    return dict(status=status, mean_auroc_drop=mean_drop,
                per_seed_auroc_drop=drops, limits=baseline['limits'],
                chance_sanity=chance_sanity([before, after], baseline['seeds']),
                release_approved=baseline.get('release_approved', False))


def verify_unique_evidence(result, seen):
    run_id = result['learning'].get('run_id')
    if not isinstance(run_id, str) or not run_id:
        raise ValueError('Reference evidence needs a run identity')
    keys = {run_id, fingerprint(result)}
    if keys & seen:
        raise ValueError('Reference evidence was reused')
    seen.update(keys)


def chance_sanity(arms, seeds):
    import numpy as np
    from scipy.stats import rankdata

    observed = np.zeros(len(arms))
    null = np.zeros((999, len(arms)))
    rng = np.random.default_rng(0)
    for seed in sorted(seeds):
        rows = [
            next(row for row in arm['runs']
                 if row['seed'] == seed and row['case'] == 'independent')
            for arm in arms
        ]
        targets = rows[0]['control']['targets']
        if any(row['control']['targets'] != targets for row in rows):
            raise ValueError('Independent control labels differ between arms')
        labels = np.asarray(targets, dtype=bool)
        probabilities = np.asarray(
            [row['control']['probabilities'] for row in rows])
        positives = int(labels.sum())
        negatives = len(labels) - positives
        if (not positives or not negatives or probabilities.ndim != 2
                or probabilities.shape[1] != len(labels)
                or not np.isfinite(probabilities).all()
                or not ((probabilities >= 0) & (probabilities <= 1)).all()):
            raise ValueError('Invalid independent control probabilities')
        ranks = rankdata(probabilities, axis=1)
        offset = positives * (positives + 1) / 2
        denominator = positives * negatives
        scores = (ranks @ labels - offset) / denominator
        if not np.allclose(scores, [row['auroc']
                                    for row in rows], rtol=0, atol=1e-12):
            raise ValueError('Independent control scores differ from evidence')
        observed += (scores - .5)**2 / len(seeds)
        # Share each label shuffle across arms that repeat the same fixture.
        shuffled = np.asarray([rng.permutation(labels) for _ in range(999)])
        shuffled_scores = (shuffled @ ranks.T - offset) / denominator
        null += (shuffled_scores - .5)**2 / len(seeds)
    statistic = float(np.max(observed))
    p_value = (1 + int(
        (np.max(null, axis=1) >= statistic - 1e-12).sum())) / 1000
    if p_value <= .01:
        raise ValueError('Reference independent labels differ from chance')
    return dict(method='joint_monte_carlo_label_permutation', draws=999, seed=0,
                alpha=.01, statistic=statistic, p_value=p_value,
                statistic_name='max_arm_mean_seed_squared_auroc_distance')


def calibrate(pairs):
    if len(pairs) < 2:
        raise ValueError('Calibration needs repeated reference pairs')
    first = pairs[0][0]
    baseline = {
        field: first['learning'][field]
        for field in ('source_sha256', 'settings', 'configuration',
                      'dependencies_sha256', 'seeds')
    }
    means, mean_drops, seed_drops, records = [], [], [], []
    seen = set()
    for before, after in pairs:
        for result in (before, after):
            verify_unique_evidence(result, seen)
            if result['wheel'] != first['wheel'] or result.get(
                    'status'
            ) != 'MEASURED' or result['learning']['status'] != 'MEASURED':
                raise ValueError(
                    'Calibration requires measured reference repeats')
            for field, value in baseline.items():
                if result['learning'].get(field) != value:
                    raise ValueError(f'Calibration inputs differ: {field}')
            means.append(
                summarize(result['learning']['runs'], baseline['seeds']))
        if any(before['learning'].get(key) != after['learning'].get(key)
               for key in ('python', 'platform')):
            raise ValueError('Calibration environments differ')
        old = {(r['seed'], r['case']): r for r in before['learning']['runs']}
        drops = []
        for row in after['learning']['runs']:
            previous = old[row['seed'], row['case']]
            if any(previous[k] != row[k]
                   for k in ('training_seed', 'data_sha256', 'plan_sha256')):
                raise ValueError(
                    'Calibration pairs must use matching data and plans')
            if row['case'] == 'learnable':
                drops.append(previous['auroc'] - row['auroc'])
        mean_drops.append(abs(statistics.mean(drops)))
        seed_drops.extend(map(abs, drops))
        records.append({
            side:
            dict(
                evidence_sha256=fingerprint(result),
                run_id=result['learning']['run_id'], scores={
                    case: [
                        r['auroc'] for r in sorted(result['learning']['runs'],
                                                   key=lambda r: r['seed'])
                        if r['case'] == case
                    ]
                    for case in ('learnable', 'independent')
                })
            for side, result in zip(('a', 'b'), (before, after))
        })
    baseline.update(
        reference_wheel=first['wheel'],
        method='observed_reference_repeat_envelope', release_approved=False,
        chance_sanity=chance_sanity(
            [result['learning'] for pair in pairs for result in pair],
            baseline['seeds']), calibration=records,
        validation={'status': 'PENDING'}, limits=dict(
            learnable_min=min(m['learnable'] for m in means),
            independent_distance=max(abs(m['independent'] - .5) for m in means),
            max_mean_drop=max(mean_drops), max_seed_drop=max(seed_drops)))
    if baseline['limits']['learnable_min'] <= .5 + baseline['limits'][
            'independent_distance'] or any(row['auroc'] <= .5 for pair in pairs
                                           for result in pair
                                           for row in result['learning']['runs']
                                           if row['case'] == 'learnable'):
        raise ValueError('Reference does not show consistent learning')
    return baseline


def create_data(seed, independent):
    features = random.Random(seed)
    labels = random.Random((2000 if independent else 1000) + seed)
    groups = [
        features.randrange(SETTINGS['groups'])
        for _ in range(SETTINGS['entities'])
    ]
    split = [0] * 1200 + [1] * 400 + [2] * 400
    random.Random(3000 + seed).shuffle(split)
    return {
        'ID':
        list(range(SETTINGS['entities'])),
        'GROUP_ID':
        groups,
        'LABEL': [
            labels.random() < (.5 if independent else .9 if group < 10 else .1)
            for group in groups
        ],
    }, split


def score_predictions(frame, expected):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    frame = frame.sort_values('ENTITY')
    assert frame['ENTITY'].is_unique
    assert frame['ENTITY'].tolist() == sorted(expected)
    assert frame['TARGET'].tolist() == [
        expected[key] for key in sorted(expected)
    ]
    probabilities = frame[['False_PROB', 'True_PROB']].to_numpy()
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert set(frame['TARGET']) == {False, True}
    return float(roc_auc_score(frame['TARGET'], frame['True_PROB']))


def make_plan(seed):
    from kumoapi.encoder import Index, Numerical
    from kumoapi.model_plan import GNNModelPlan, ModelPlan, NumNeighborsConfig

    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.pruning = None
    plan.training_job.metrics = ['auroc']
    plan.training_job.tune_metric = 'auroc'
    plan.training_job.disable_compilation = True
    plan.training_job.disable_explain = True
    plan.training_job.manual_seed = seed
    plan.training_job.refit_trainval = False
    plan.training_job.refit_full = False
    plan.optimization.max_epochs = SETTINGS['epochs']
    plan.optimization.min_steps_per_epoch = SETTINGS['steps_per_epoch']
    plan.optimization.max_steps_per_epoch = SETTINGS['steps_per_epoch']
    plan.optimization.batch_size = [64]
    plan.optimization.loss = ['binary_cross_entropy']
    plan.optimization.optimizer = ['Adam']
    plan.optimization.base_lr = [.01]
    plan.optimization.weight_decay = [0.0]
    plan.optimization.majority_sampling_ratio = [None]
    plan.optimization.weight_mode = [None]
    plan.optimization.max_val_steps = 2000
    plan.optimization.max_test_steps = 4000
    plan.optimization.early_stopping = [{'min_delta': 0.0, 'patience': 3}]
    plan.optimization.lr_scheduler = [{
        'name': 'constant_with_warmup',
        'interval': 'step',
        'kwargs': {
            'warmup_ratio_or_steps': .1
        }
    }]
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
    plan.neighbor_sampling.sample_from_entity_table = True
    plan.neighbor_sampling.adaptive_sampling = False
    plan.column_processing.encoder_overrides = {
        'groups.RISK': Numerical(scaler=None, na_strategy='mean'),
        'train_table.TARGET': Index(min_occ=1, na_strategy='separate'),
    }
    return plan


def verify_plan(requested, selected):
    if isinstance(requested, dict):
        for key, value in requested.items():
            verify_plan(value, selected.get(key, '???'))
    elif requested != '???' and requested != selected:
        raise ValueError(
            f'Selected model plan differs: {requested!r} != {selected!r}')


def run_case(root, evidence, seed, independent):
    import numpy as np
    import pandas as pd
    import torch
    from kumoapi.graph import Edge
    from kumoapi.train import TrainingTableSpec
    from kumoapi.typing import Stype

    import kumo.engine as kumo

    root.mkdir(parents=True)
    evidence.mkdir()
    data = root / 'data'
    data.mkdir()
    values, split = create_data(seed, independent)
    pd.DataFrame({
        'ENTITY': values['ID'],
        'TARGET': values['LABEL'],
        'SPLIT': split
    }).to_parquet(evidence / 'split.parquet', index=False)
    pd.DataFrame(values).to_parquet(data / 'entities.parquet')
    pd.DataFrame({
        'ID': range(20),
        'RISK': [.9] * 10 + [.1] * 10
    }).to_parquet(data / 'groups.parquet')
    pd.DataFrame({
        'ENTITY1': values['ID'],
        'TARGET1': values['LABEL'],
        'SPLIT': split
    }).to_parquet(data / 'seeds.parquet')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    kumo.configure(scratch_dir=root / 'scratch')
    connector = kumo.FileConnector(root_dir=str(data))
    graph = kumo.Graph(
        {
            'entities':
            kumo.Table.from_source_table(connector['entities'],
                                         primary_key='ID').infer_metadata(),
            'groups':
            kumo.Table.from_source_table(
                connector['groups'],
                primary_key='ID').infer_metadata().with_column(
                    'RISK', stype=Stype.numerical),
        }, [Edge('entities', 'GROUP_ID', 'groups')])
    started = time.monotonic()
    snapshot = graph.snapshot()
    stats = json.loads(
        (snapshot.index_path / 'ingest_stats.json').read_text())['stats']
    assert stats['total_nodes'] == 2020 and stats['total_edges'] == 4000, stats
    query = kumo.PredictiveQuery(graph,
                                 'PREDICT entities.LABEL FOR EACH entities.ID')
    table = kumo.TrainingTable.from_source_table(
        connector['seeds'], TrainingTableSpec(split_col='SPLIT'), pq=query,
        snapshot=snapshot)
    plan = make_plan(seed)
    trainer = kumo.Trainer(plan)
    result = trainer.fit(table, snapshot=snapshot)
    (evidence / 'reported-metrics.json').write_text(
        json.dumps(result.metrics, indent=2))
    (evidence / 'model-plan.json').write_text(
        Path(result.artifact_paths['resolved_model_plan']).read_text())
    (evidence / 'trainer-config.json').write_text(
        Path(result.artifact_paths['trainer_config']).read_text())
    for name in ('encoders', 'target_encoder'):
        (evidence / f'{name}.json').write_text(
            Path(result.artifact_paths[name]).read_text())
    [trial] = trainer.scratch_dir.glob(
        'runs/*-fit-*/models/*/automl/version_*/model_plan.json')
    (evidence / 'selected-model-plan.json').write_text(trial.read_text())
    verify_plan(plan.as_json(), json.loads(trial.read_text()))
    frame = result.holdout_df()
    frame.to_parquet(evidence / 'holdout.parquet', index=False)
    assert result.holdout_truncated is False
    expected = {
        key: label
        for key, label, stage in zip(values['ID'], values['LABEL'], split)
        if stage == 2
    }
    # Reported metrics precede calibration. Score the saved probabilities.
    score = score_predictions(frame, expected)
    [schema_path] = trainer.scratch_dir.glob(
        'runs/*-fit-*/task_dataset/feature_schema.json')
    schema = json.loads(schema_path.read_text())
    (evidence / 'feature-schema.json').write_text(json.dumps(schema, indent=2))
    assert 'LABEL' not in schema['node_types']['entities']['feature_columns']
    # Exercise the actual scorer with deliberately useless and leaked outputs.
    control = frame.copy()
    control['True_PROB'] = control['False_PROB'] = .5
    assert score_predictions(control, expected) == .5
    control['True_PROB'] = control['TARGET'].astype(float)
    control['False_PROB'] = 1 - control['True_PROB']
    assert score_predictions(control, expected) == 1.0
    row = {
        'seed': seed,
        'training_seed': seed,
        'data_sha256': fingerprint([values, split]),
        'plan_sha256': fingerprint(plan.as_json()),
        'case': 'independent' if independent else 'learnable',
        'auroc': score,
        'evaluation_rows': len(frame),
        'seconds': round(time.monotonic() - started, 3),
        'metrics': result.metrics
    }
    if independent:
        ordered = frame.sort_values('ENTITY')
        row['control'] = dict(targets=ordered['TARGET'].tolist(),
                              probabilities=ordered['True_PROB'].tolist())
    (evidence / 'result.json').write_text(json.dumps(row, indent=2))
    del trainer, result
    gc.collect()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version')
    parser.add_argument(
        '--calibrate', type=Path,
        help='Directory of pair-*/a and pair-*/b measurement summaries')
    parser.add_argument('--validate-with', type=Path,
                        help='Separate directory of reserved measurement pairs')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--measure', action='store_true',
                        help='Collect reference scores without quality limits')
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique nonnegative integers')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.calibrate:

        def read_pairs(root):
            return [
                tuple(
                    json.loads((pair / side / 'summary.json').read_text())
                    for side in ('a', 'b'))
                for pair in sorted(root.glob('pair-*'))
            ]

        baseline = calibrate(read_pairs(args.calibrate))
        try:
            if args.validate_with:
                reserved = read_pairs(args.validate_with)
                if len(reserved) < 2:
                    raise ValueError('At least two reserved pairs are required')
                checks = []
                seen = {
                    value
                    for record in baseline['calibration']
                    for side in ('a', 'b')
                    for value in (record[side]['evidence_sha256'],
                                  record[side]['run_id'])
                }
                for a, b in reserved:
                    verify_unique_evidence(a, seen)
                    verify_unique_evidence(b, seen)
                    for before, after in ((a, b), (b, a)):
                        compare_results(before, after, baseline,
                                        measurement=True)
                    checks.append(dict(a=fingerprint(a), b=fingerprint(b)))
                baseline['validation'] = dict(status='PASS', pairs=checks)
        except Exception as exc:
            baseline['validation'] = dict(status='FAIL', error=str(exc))
        (output / 'learning-baseline.json').write_text(
            json.dumps(baseline, indent=2))
        return 1 if baseline['validation']['status'] == 'FAIL' else 0
    if not args.version or args.validate_with:
        parser.error(
            'Training requires --version; --validate-with requires --calibrate')
    report = {
        'status': 'RUNNING',
        'run_id': uuid.uuid4().hex,
        'settings': SETTINGS,
        'seeds': args.seeds,
        'python': sys.version,
        'platform': platform.platform(),
        'runs': []
    }
    report['source_sha256'] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    started = time.monotonic()
    try:
        import torch

        import kumo.engine as kumo

        assert sys.platform == 'linux' and platform.machine() == 'x86_64'
        assert sys.version_info[:2] == (3, 12)
        assert importlib.metadata.version(
            'kumo-engine') == kumo.__version__ == args.version
        assert Path(kumo.__file__).resolve().is_relative_to(Path(sys.prefix))
        assert not torch.cuda.is_available()
        torch.set_num_threads(SETTINGS['threads'])
        report['version'] = kumo.__version__
        report['import_path'] = kumo.__file__
        packages = {
            re.sub(r'[-_.]+', '-', distribution.metadata['Name']).lower():
            distribution.version
            for distribution in importlib.metadata.distributions()
        }
        (output / 'packages.json').write_text(
            json.dumps(packages, indent=2, sort_keys=True))
        report['dependencies_sha256'] = fingerprint({
            name: version
            for name, version in packages.items() if name != 'kumo-engine'
        })
        report['configuration'] = dict(os=sys.platform,
                                       machine=platform.machine(),
                                       python='3.12', device='cpu',
                                       threads=SETTINGS['threads'])
        for seed in args.seeds:
            for independent in (False, True):
                name = f'seed-{seed}-' + ('independent'
                                          if independent else 'learnable')
                print(name, flush=True)
                report['stage'] = name
                report['runs'].append(
                    run_case(Path.cwd() / 'learning-work' / name, output / name,
                             seed, independent))
                (output / 'learning-result.json').write_text(
                    json.dumps(report, indent=2))
        report['chance_sanity'] = chance_sanity([report], args.seeds)
        report['mean_auroc'] = summarize(report['runs'], args.seeds)
        if not args.measure:
            baseline = json.loads(
                Path(__file__).with_name(
                    'wheel_workflow.json').read_text())['learning']
            if baseline.get('validation', {}).get('status') != 'PASS':
                raise ValueError(
                    'Reference calibration needs passing reserved runs')
            validate_learning(report, baseline)
        assert time.monotonic(
        ) - started < 1200, 'Learning job exceeded 20 minutes'
        report['status'] = 'MEASURED' if args.measure else 'PASS'
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        (output / 'learning-result.json').write_text(
            json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] in ('PASS', 'MEASURED') else 1


if __name__ == '__main__':
    sys.exit(main())
