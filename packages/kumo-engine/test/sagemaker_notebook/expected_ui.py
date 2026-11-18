# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive browser expectations from the collected lifecycle execution receipts."""
import json
import math
import re
from pathlib import Path


def expectations(directory):
    directory = Path(directory)
    phases = {}
    for name in ('fit', 'score', 'evaluate'):
        phase = json.loads((directory / (name + '.json')).read_text())
        if phase.get('status') != 'passed' or phase.get('returncode') != 0:
            raise ValueError('UI requires a successfully executed lifecycle: ' +
                             name)
        phases[name] = phase['result']
    fit, score, evaluate = (phases[name]
                            for name in ('fit', 'score', 'evaluate'))
    if (len(fit['versions']) != 2 or len(score['explicit']) != 2
            or len(score['aliases']) != 2 or len(evaluate['monitoring']) != 2):
        raise ValueError(
            'UI requires two versions and the complete scoring/monitoring cycle'
        )
    runs = []
    for version in fit['versions']:
        runs.append(dict(roles=['fit'], reference=version['reference']))
    for prediction in score['explicit'] + score['aliases']:
        runs.append(dict(roles=['score'], reference=prediction['reference']))
    if len({item['reference']['run'] for item in runs}) != 6:
        raise ValueError('UI requires six distinct fit and prediction runs')
    by_run = {item['reference']['run']: item for item in runs}
    for monitoring, alias in zip(evaluate['monitoring'], score['aliases']):
        if monitoring['prediction'] != alias['prediction']:
            raise ValueError('Monitoring identifies another alias prediction')
        run = alias['reference']['run']
        item = by_run[run]
        for kind, record in [('drift', monitoring['drift_reference']),
                             ('evaluate', monitoring['reference'])]:
            previous = item['reference']
            if any(
                    record.get(key) != previous.get(key)
                    for key in ('run', 'experiment_id', 'run_name', 'status')):
                raise ValueError('Monitoring changed the tracked run identity')
            for key in ('kumo.bundle_id', 'kumo.prediction_id',
                        'kumo.model_name', 'kumo.model_version',
                        'kumo.fit_run_id'):
                if record['params'].get(key) != previous['params'].get(key):
                    raise ValueError(
                        'Monitoring changed the prediction lineage')
            if any(record['metrics'].get(key) != value
                   for key, value in previous['metrics'].items()):
                raise ValueError('Monitoring changed or lost an earlier metric')
            item.update(reference=record)
            item['roles'].append(kind)
    seen = set()
    for item in runs:
        record = item['reference']
        run = record['run']
        if (not re.fullmatch(r'[a-f0-9]{32}', run) or run in seen
                or record.get('status') != 'FINISHED'
                or not record.get('run_name')
                or not record.get('experiment_id')):
            raise ValueError(
                'UI requires distinct finished runs with recorded identities')
        seen.add(run)
        metrics = record['metrics']
        if not metrics or any(not math.isfinite(value)
                              for value in metrics.values()):
            raise ValueError('UI requires recorded finite metric values')
        if 'fit' in item['roles']:
            for name in ('train/loss', 'val/auroc'):
                points = record['history'].get(name, [])
                if (len({step
                         for step, _ in points}) < 2
                        or any(not math.isfinite(value)
                               for _, value in points)):
                    raise ValueError('UI requires multi-epoch training history')
    if len({v['version'] for v in fit['versions']}) != 2:
        raise ValueError('UI requires two distinct registered model versions')
    if any(not re.fullmatch(r'[1-9][0-9]*', str(v['version']))
           for v in fit['versions']):
        raise ValueError('UI requires numeric managed MLflow model versions')
    return dict(runs=runs, model=fit['model'], versions=fit['versions'])
