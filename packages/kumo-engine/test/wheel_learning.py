#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure learning on generated data; limits remain provisional."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

SETTINGS = {
    'entities': 2000,
    'groups': 20,
    'epochs': 10,
    'steps_per_epoch': 20,
    'threads': 4,
    'min_auroc': .75,
    'independent_band': [.45, .55],
    'limits_approved': False,
}


def summarize(rows, seeds):
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
    if means['learnable'] < SETTINGS['min_auroc']:
        raise ValueError(f'learnable AUROC is too low: {means}')
    lower, upper = SETTINGS['independent_band']
    if not lower <= means['independent'] <= upper:
        raise ValueError(f'independent AUROC is outside its band: {means}')
    return means


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


def run_case(root, evidence, seed, independent):
    import numpy as np
    import pandas as pd
    import torch
    from kumoapi.graph import Edge
    from kumoapi.model_plan import ModelPlan
    from kumoapi.train import TrainingTableSpec
    from kumoapi.typing import Stype

    import kumo.engine as kumo

    root.mkdir(parents=True)
    evidence.mkdir()
    data = root / 'data'
    data.mkdir()
    values, split = create_data(seed, independent)
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
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.training_job.manual_seed = seed
    plan.training_job.refit_trainval = False
    plan.training_job.refit_full = False
    plan.optimization.max_epochs = SETTINGS['epochs']
    plan.optimization.max_steps_per_epoch = SETTINGS['steps_per_epoch']
    plan.optimization.batch_size = [64]
    plan.model_architecture.channels = [16]
    trainer = kumo.Trainer(plan)
    result = trainer.fit(table, snapshot=snapshot)
    assert result.holdout_truncated is False
    frame = result.holdout_df()
    expected = {
        key: label
        for key, label, stage in zip(values['ID'], values['LABEL'], split)
        if stage == 2
    }
    score = score_predictions(frame, expected)
    assert math.isclose(score, result.metrics['test_auroc'], abs_tol=1e-6)
    [schema_path] = trainer.scratch_dir.glob(
        'runs/*-fit-*/task_dataset/feature_schema.json')
    schema = json.loads(schema_path.read_text())
    assert 'LABEL' not in schema['node_types']['entities']['feature_columns']
    (evidence / 'feature-schema.json').write_text(json.dumps(schema, indent=2))
    (evidence / 'model-plan.json').write_text(
        Path(result.artifact_paths['resolved_model_plan']).read_text())
    frame.to_parquet(evidence / 'holdout.parquet', index=False)
    pd.DataFrame({
        'ENTITY': values['ID'],
        'TARGET': values['LABEL'],
        'SPLIT': split
    }).to_parquet(evidence / 'split.parquet', index=False)
    # Exercise the actual scorer with deliberately useless and leaked outputs.
    control = frame.copy()
    control['True_PROB'] = control['False_PROB'] = .5
    assert score_predictions(control, expected) == .5
    control['True_PROB'] = control['TARGET'].astype(float)
    control['False_PROB'] = 1 - control['True_PROB']
    assert score_predictions(control, expected) == 1.0
    row = {
        'seed': seed,
        'case': 'independent' if independent else 'learnable',
        'auroc': score,
        'evaluation_rows': len(frame),
        'seconds': round(time.monotonic() - started, 3),
        'metrics': result.metrics
    }
    (evidence / 'result.json').write_text(json.dumps(row, indent=2))
    del trainer, result
    gc.collect()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique nonnegative integers')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        'status': 'RUNNING',
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
        (output / 'packages.json').write_text(
            json.dumps(
                {
                    distribution.metadata['Name']: distribution.version
                    for distribution in importlib.metadata.distributions()
                }, indent=2, sort_keys=True))
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
        report['mean_auroc'] = summarize(report['runs'], args.seeds)
        assert time.monotonic(
        ) - started < 1200, 'Learning job exceeded 20 minutes'
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        (output / 'learning-result.json').write_text(
            json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
