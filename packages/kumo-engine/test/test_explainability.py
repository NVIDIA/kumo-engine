# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data side of the post-training explainability widget.

The format, coverage and missing-file tests use small ``tmp_path`` runs. The
answer-key tests load retained H&M training runs and compare every computed
number with independently computed evidence; set ``KUMO_EXPLAINABILITY_RUNS``
(``<task>/scratch/trainers/*/runs/*-fit-*`` beneath it) and
``KUMO_EXPLAINABILITY_EVIDENCE`` (``values.json``, ``fits.json``,
``cutoffs.json``) to run them.
"""

import base64
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pandas as pd
import pytest

from kumo.engine._explainability import (
    _cards,
    _classification,
    _columns,
    _coverage,
    _files,
    _forecasting,
)
from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability import (
    ExplainabilityView,
    _holdout,
    _metrics,
    _regression,
    _render,
    _run,
    _subgraphs,
    open_run,
)
from kumo.engine.trainer.trainer import TrainingResult

MINUS = '−'

# ------------------------------------------------------------------ formats


@pytest.mark.parametrize('value,text', [
    (136.59754, '136.6'),
    (110334.3125, '110,334.3'),
    (0.7207305, '0.721'),
    (0.0442945, '0.0443'),
    (0.000297592, '0.000298'),
    (0.5, '0.500'),
    (9.9996, '10.0'),
    (0.0999996, '0.100'),
    (-6.8, f'{MINUS}6.80'),
    (0.0, '0'),
    (float('nan'), 'N/A'),
])
def test_number(value, text):
    assert fmt.number(value) == text


@pytest.mark.parametrize('value,text', [
    (0.2207305, '+0.221'),
    (-0.0562022, f'{MINUS}0.0562'),
    (50.835, '+50.8'),
    (-12.42, f'{MINUS}12.4'),
    (0.0, '0'),
])
def test_signed(value, text):
    assert fmt.signed(value) == text


def test_shared_scale_and_other_formats():
    assert fmt.numbers([-6.8, 21.6, 577.0]) == [f'{MINUS}6.8', '21.6', '577.0']
    assert fmt.numbers([0.761, 0.0759]) == ['0.761', '0.0759']
    assert fmt.count(4932) == '4,932'
    assert fmt.percent(0.79298) == '79.3%'
    assert fmt.multiple(55.71) == '55.7×'
    assert fmt.cutoff(0.5) == '0.50'
    assert fmt.date('2020-05-23T00:00:00') == '2020-05-23'


# ----------------------------------------------------------- small runs


def _metrics_card(run: _run.Run) -> dict[str, Any]:
    card = _metrics.card(run)
    assert card is not None
    return card


def _importance(run: _run.Run) -> dict[str, Any]:
    card = _columns.importance(run)
    assert card is not None
    return card


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _run_dir(tmp_path: Path, *, test_rows: int, batch: int, max_test_steps: int,
             task_type: str = 'regression', holdout_rows: int | None = None,
             refit: bool = True, explain: bool = True) -> Path:
    """A minimal retained run with the files the widget reads."""
    run = tmp_path / 'run'
    bundle = run / 'published-model'
    trial = run / 'models' / 'task' / 'automl' / 'version_0'
    published = Path(f'{trial}_refit_trainval') if refit else trial
    table = tmp_path / 'snap' / 'train_tables' / 'tt-run'
    _write_json(bundle / 'prediction-schema.json', {
        'task_type': task_type,
        'target_column': 'TARGET'
    })
    _write_json(
        bundle / 'training-spec.json', {
            'predictive_query':
            'PREDICT SUM(orders.value, 0, 30, days) FOR EACH users.user_id'
        })
    _write_json(
        bundle / 'model' / 'resolved-model-plan.json', {
            'training_job': {
                'metrics': ['mae', 'mse', 'rmse'],
                'tune_metric': 'mae',
                'refit_trainval': refit
            },
            'optimization': {
                'max_test_steps': max_test_steps
            },
        })
    _write_json(bundle / 'model' / 'config.json', {
        'loader_kwargs': {
            'batch_size': batch
        },
        'eval_loader_kwargs': None
    })
    metrics = {'test_mae': 2.0, 'test_mse': 9.0, 'test_rmse': 3.0}
    if refit:
        metrics.update(test_refit_mae=1.5, test_refit_mse=4.0,
                       test_refit_rmse=2.0)
    _write_json(bundle / 'metrics.json', metrics)
    rows = test_rows if holdout_rows is None else holdout_rows
    pd.DataFrame({
        'ENTITY': range(rows),
        'TARGET': [float(i % 3) for i in range(rows)],
        'TARGET_PRED': [1.0] * rows,
    }).to_parquet(bundle / 'holdout.parquet')
    _write_json(trial.parent / 'best_trainer.json',
                {'best_trainer': str(trial)})
    _write_json(trial / 'config.json',
                {'refit_cfg': {
                    'refitted_trainval': refit
                }})
    published.mkdir(parents=True, exist_ok=True)
    if explain:
        (published / 'explain_test').mkdir(parents=True, exist_ok=True)
        (published / 'explain_test' / 'scores.csv'
         ).write_text('table,column,score\nusers,age,0.3\norders,value,0.1\n')
    split = {
        'num_train_rows': test_rows,
        'anchor_date': '2024-01-01T00:00:00',
        'min_date': '2024-01-01T00:00:00',
        'max_date': '2024-01-01T00:00:00'
    }
    _write_json(
        table / 'split_stats.json', {
            'test_table_stats': split,
            'val_table_stats': {
                **split, 'num_train_rows': 40
            },
            'train_table_stats': {
                **split, 'num_train_rows': 400
            },
        })
    _write_json(
        run / 'task_dataset' / 'runtime_manifest.json', {
            'graph': {
                'index_path': str(tmp_path / 'snap')
            },
            'stages': {
                'test': {
                    'index_path': str(table)
                }
            },
        })
    _write_json(
        bundle / 'publication-manifest.json', {
            'artifacts': [{
                'role': 'holdout',
                'relative_path': 'holdout.parquet'
            }, {
                'role': 'holdout_meta',
                'relative_path': 'holdout-meta.json'
            }]
        })
    return run


def test_coverage_line_only_when_a_cap_applies(tmp_path):
    files = _files.open_run(
        str(_run_dir(tmp_path, test_rows=100, batch=8, max_test_steps=10)))
    loaded = _run.load(files)
    # 10 steps of 8 rows evaluate 80 of 100 test rows; the explanation pass
    # replays 50 batches of 8 // 4 = 2 rows.
    assert loaded.coverage.metrics == 80
    assert loaded.coverage.explanation == 100
    assert (_metrics_card(loaded)['coverage'] ==
            'based on the first 80 of 100 test rows')
    assert _importance(loaded)['coverage'] is None


def test_coverage_absent_without_cap(tmp_path):
    files = _files.open_run(
        str(_run_dir(tmp_path, test_rows=100, batch=64, max_test_steps=2000)))
    loaded = _run.load(files)
    assert loaded.coverage.line(loaded.coverage.metrics) is None
    assert loaded.coverage.line(loaded.coverage.explanation) is None
    assert _metrics_card(loaded)['coverage'] is None


def test_explanation_cap_matches_kumo_replay(tmp_path):
    files = _files.open_run(
        str(_run_dir(tmp_path, test_rows=33_603, batch=512,
                     max_test_steps=2000)))
    loaded = _run.load(files)
    assert loaded.coverage.explanation == 50 * 128
    assert (_importance(loaded)['coverage'] ==
            'based on the first 6,400 of 33,603 test rows')


def test_holdout_seeds_counts_forecast_anchors():
    frame = pd.DataFrame({
        'ENTITY': [1, 1, 2, 2],
        'ANCHOR_TIMESTAMP': [0, 0, 0, 0],
        'TIMESTAMP': [0, 1, 0, 1]
    })
    assert _coverage.holdout_seeds(frame) == 2


def test_metrics_card_uses_ui_last_epoch_training(tmp_path):
    loaded = _run.load(
        _files.open_run(
            str(_run_dir(tmp_path, test_rows=30, batch=64,
                         max_test_steps=2000))))
    loaded.metrics.update(train_mae=0.8, val_mae=1.7)
    card = _metrics_card(loaded)
    tiles = card['views'][0]['tiles']
    assert [t['name'] for t in tiles] == ['MAE', 'RMSE']
    mae = tiles[0]
    assert mae['tune'] and mae['holdout'] == 1.5 and mae['training'] == 0.8
    assert mae['gap_text'] == 'holdout minus training: +0.700'
    table = card['views'][1]['table']
    assert [g['title'] for g in table['groups']
            ] == ['Published model', 'Training run · last epoch']
    assert table['columns'][1]['label'] == 'Validation'
    assert table['rows'][0]['values'] == [1.5, 1.7, 0.8]
    assert 'Last training epoch' in mae['training_scope']


@pytest.mark.parametrize('bundled', [True, False, 'unlisted'])
def test_epoch_history_follows_search_experiments(tmp_path, bundled):
    run = _run_dir(tmp_path, test_rows=3, batch=64, max_test_steps=2000)
    bundle = run / 'published-model'
    automl = run / 'models' / 'task' / 'automl'
    files = _files.locate(str(bundle), None if bundled else str(run))
    loaded = _run.load(files)
    assert 'history' not in [
        view['id'] for view in _metrics_card(loaded)['views']
    ]
    path = bundle / 'metric-history.json' if bundled else automl / 'auto_train_progress.json'
    first = {
        'train_mae': 0.5,
        'train_loss': float('nan'),
        'train_f1_per_label_0': 0,
        'test_mae': 0
    }
    epochs = {
        '3': {
            'train_metrics': {
                'train_mae': 0.4
            },
            'validation_metrics': {
                'val_mae': 0.8
            }
        },
        '0': {
            'train_metrics': first
        },
        'bad': [],
        '2': None,
    }
    trials = {
        '10': {
            'metrics': {
                '2': {
                    'train_metrics': {
                        'train_loss': 1
                    }
                }
            }
        },
        '2': {
            'metrics': epochs
        },
        'empty': {
            'metrics': {}
        },
    }
    _write_json(path, {'trial_progress': trials})
    if bundled == 'unlisted':
        assert _metrics._history(loaded) is None
        assert 'Epoch history' not in ExplainabilityView(files)._repr_svg_()
        return
    if bundled:
        from kumo_core.artifacts import add_bundle_file
        add_bundle_file(bundle, 'metric_history', 'metric-history.json')
    history = _metrics_card(loaded)['views'][-1]
    assert history['metrics'] == ['mae', 'loss']
    assert history['experiments'] == [
        {
            'train_mae': [(1, 0.5), (4, 0.4)],
            'val_mae': [(3, 0.8)],
            'train_loss': [(1, None)]
        },
        {
            'train_loss': [(2, 1)]
        },
    ]
    json.dumps(history, allow_nan=False)
    # History alone is sufficient, including in a bundle without holdout scores.
    (bundle / 'metrics.json').unlink()
    html = ExplainabilityView(files).html()
    assert '"id":"history"' in html.replace(' ', '')
    assert 'Epoch history' in html


@pytest.mark.parametrize('truncated', [False, True])
@pytest.mark.parametrize('full_refit', [None, 'provenance', 'config'])
def test_bundle_alone_keeps_published_holdout(tmp_path, truncated, full_refit):
    run = _run_dir(tmp_path, test_rows=30, batch=64, max_test_steps=2000)
    bundle = run / 'published-model'
    _write_json(bundle / 'holdout-meta.json', {'truncated': truncated})
    if full_refit == 'provenance':
        _write_json(bundle / '_provenance.json', {'fit_mode': 'refit_full'})
    elif full_refit == 'config':
        _write_json(bundle / 'model' / 'config.json',
                    {'refit_cfg': {
                        'refitted_full': True
                    }})
    files = _files.locate(str(bundle))
    loaded = _run.load(files)
    card = _metrics_card(loaded)
    mae = card['views'][0]['tiles'][0]
    assert mae['holdout'] == 1.5 and mae['training'] is None
    assert mae['callout'] is not None  # baselines need only the holdout
    table = card['views'][1]['table']
    assert [c['label'] for c in table['columns']] == ['Holdout']
    assert [g['span'] for g in table['groups']] == [1]
    assert _columns.importance(loaded) is None
    assert bool(card['coverage']) == truncated
    view = ExplainabilityView(files)
    assert ('truncated retained holdout' in view._repr_svg_()) == truncated
    assert ('before full refit' in view._repr_svg_()) == bool(full_refit)
    assert table['groups'][0]['title'] == ('Model before full refit'
                                           if full_refit else 'Published model')


@pytest.mark.parametrize('missing', [
    'holdout.parquet',
    'explain_test',
    'split_stats.json',
    'metrics-array',
    'metrics-string',
    'split-stats-array',
    'split-stats-string',
    'runtime-stages',
    'runtime-test',
    'runtime-graph',
    'runtime-path',
    'scores-nonfinite',
    'scores-duplicate',
])
def test_missing_optional_files_omit_views(tmp_path, missing):
    run = _run_dir(tmp_path, test_rows=30, batch=64, max_test_steps=2000)
    targets = {
        'holdout.parquet':
        run / 'published-model' / 'holdout.parquet',
        'explain_test':
        next(run.glob('models/*/automl/*_refit_trainval')) / 'explain_test' /
        'scores.csv',
        'split_stats.json':
        tmp_path / 'snap' / 'train_tables' / 'tt-run' / 'split_stats.json',
    }
    if missing.startswith('metrics-'):
        _write_json(run / 'published-model' / 'metrics.json',
                    [1, 2] if missing == 'metrics-array' else 'invalid')
    elif missing.startswith('split-stats-'):
        _write_json(
            targets['split_stats.json'], {
                'test_table_stats':
                [1, 2] if missing == 'split-stats-array' else 'invalid'
            })
    elif missing.startswith('runtime-'):
        malformed = {
            'runtime-stages': {
                'stages': [1]
            },
            'runtime-test': {
                'stages': {
                    'test': [1]
                }
            },
            'runtime-graph': {
                'graph': [1]
            },
            'runtime-path': {
                'stages': {
                    'test': {
                        'index_path': [1]
                    }
                }
            }
        }
        _write_json(run / 'task_dataset/runtime_manifest.json',
                    malformed[missing])
    elif missing.startswith('scores-'):
        targets['explain_test'].write_text(
            'table,column,score\nusers,age,inf\n' if missing ==
            'scores-nonfinite' else
            'table,column,score\nusers,age,0.3\nusers,age,0.2\n')
    else:
        targets[missing].unlink()
    if missing == 'explain_test' or missing.startswith('scores-'):
        _write_json(
            targets['explain_test'].parent.parent / 'column_analysis' / 'test' /
            'column_analysis.json', {
                'task_type':
                'regression',
                'cohorts': [{
                    'stype': None,
                    'target_score': 0.0,
                    'table_name': 'users',
                    'col_name': column,
                    'hop': 0,
                    'cohorts': ['a', 'b'],
                    'populations': [0.5, 0.5],
                    'predictions': [1, 2],
                    'targets': [1.5, 1.5],
                    'prediction_score': 1.0
                } for column in ['age', 'region']]
            })
    loaded = _run.load(_files.open_run(str(run)))
    if missing.startswith('metrics-'):
        assert loaded.metrics == {}
        assert loaded.holdout is not None
        view = ExplainabilityView(loaded.files)
        assert view.data['tabs'] and 'Training diagnostics' in view._repr_svg_()
        return
    card = _metrics_card(loaded)
    assert card is not None
    tile = card['views'][0]['tiles'][0]
    if missing == 'holdout.parquet':
        assert loaded.holdout is None and tile['baselines'] == []
        assert tile['callout'] is None
    if missing == 'explain_test' or missing.startswith('scores-'):
        assert _columns.importance(loaded) is None
        groups = _columns.groups(loaded)
        assert groups is not None and groups['selectable']
        assert len(groups['columns']) == 2
    if missing == 'split_stats.json' or missing.startswith(
        ('split-stats-', 'runtime-')):
        assert card['coverage'] is None
        assert card['views'][1]['table']['columns'][0]['detail'] is None
        view = ExplainabilityView(loaded.files)
        assert view.data['tabs'] and 'Training diagnostics' in view._repr_svg_()


def test_unsupported_task_type(tmp_path):
    run = _run_dir(tmp_path, test_rows=30, batch=64, max_test_steps=2000,
                   task_type='unknown')
    with pytest.raises(_run.UnsupportedModelError):
        _run.load(_files.open_run(str(run)))


@pytest.mark.parametrize(
    'task', [*_run.CLASS_LABELED, 'regression', 'binary_classification'])
def test_labeled_classification_cohorts(tmp_path, task):
    path = _run_dir(tmp_path, test_rows=3, batch=4, max_test_steps=20,
                    task_type=task)
    run = _run.load(_files.open_run(str(path)))
    assert run.task_type == task
    if task != 'binary_classification':
        assert run.holdout is not None and len(run.holdout) == 3
    assert run.files.trial is not None
    column_path = Path(
        run.files.trial) / 'column_analysis/test/column_analysis.json'
    entry = dict(table_name='users', hop=0, col_name='age', stype=None,
                 populations=[0.25,
                              0.75], cohorts=['younger', 'older'], predictions={
                                  'Blue': [0.1, 0.2],
                                  'Red': [0.9, 0.8]
                              }, targets={
                                  'Red': [1, 0],
                                  'Blue': [0, 1]
                              }, prediction_score=0.3, target_score=0.4)
    if task not in _run.CLASS_LABELED:
        entry.update(predictions=[0.1, 0.2], targets=[0, 1])
    _write_json(column_path, {'task_type': task, 'cohorts': [entry]})
    groups = _columns.groups(run)
    assert groups is not None
    if task in _run.CLASS_LABELED:
        classes = groups['columns'][groups['default']]['by_class']
        assert [v['label'] for v in classes] == ['Blue', 'Red']
        assert classes[0]['axis'] == 'share Blue'
        assert [g['actual'] for g in classes[1]['groups']] == [1, 0]
        assert classes[0]['average_actual'] == 0.75
        del entry['targets']['Red']
    else:
        assert groups['columns'][groups['default']]['average_actual'] == 0.75
        entry['targets'][0] = float('nan')
    assert len(_importance(run)['views'][1]['bars']) == 1
    _write_json(column_path, {'task_type': task, 'cohorts': [entry]})
    assert _columns.groups(run) is None
    assert len(_importance(run)['views']) == 1


# ------------------------------------------------------ H&M answer key

RUNS = os.environ.get('KUMO_EXPLAINABILITY_RUNS')
EVIDENCE = os.environ.get('KUMO_EXPLAINABILITY_EVIDENCE')
TASKS = ['classification', 'regression', 'forecasting', 'link_prediction']
answer_key = pytest.mark.skipif(
    not RUNS or not EVIDENCE,
    reason='set KUMO_EXPLAINABILITY_RUNS and KUMO_EXPLAINABILITY_EVIDENCE')


def _evidence(name: str):
    assert EVIDENCE is not None
    return json.loads((Path(EVIDENCE) / name).read_text())


def _load(task: str) -> _run.Run:
    assert RUNS is not None
    roots = glob.glob(f'{RUNS}/{task}/scratch/trainers/*/runs/*-fit-*')
    assert len(roots) == 1
    return _run.load(_files.open_run(roots[0]))


@answer_key
@pytest.mark.parametrize('task', TASKS)
def test_fits_match_evidence(task):
    expected = _evidence('fits.json')[task]
    fit = _metrics.fits(_load(task))
    assert fit.refit
    for metric, row in expected['rows'].items():
        assert fit.holdout[metric] == pytest.approx(row['refit_holdout'])


@answer_key
@pytest.mark.parametrize('task', TASKS)
def test_metric_table_matches_evidence(task):
    expected = _evidence('values.json')[task]
    loaded = _load(task)
    card = _metrics_card(loaded)
    table = card['views'][1]['table']
    assert {row['metric'] for row in table['rows']} >= set(expected['metrics'])
    splits = expected['splits']
    test = splits['test_table_stats']
    assert table['columns'][0]['detail'] == (
        f"{test['num_train_rows']:,} rows · {test['anchor_date'][:10]}")
    rows = splits['val_table_stats']['num_train_rows']
    assert table['columns'][1]['detail'].startswith(f'{rows:,} rows')
    for tile in card['views'][0]['tiles']:
        assert tile['training'] == loaded.metrics.get('train_' + tile['metric'])
    assert card['coverage'] is None
    assert [t['tune'] for t in card['views'][0]['tiles']][0]
    assert card['views'][0]['tiles'][0]['metric'] == expected['tune']


@answer_key
def test_classification_baselines_match_evidence():
    expected = _evidence('values.json')['classification']
    tiles = _metrics_card(_load('classification'))['views'][0]['tiles']
    assert [t['metric'] for t in tiles] == ['auroc', 'ap']
    auroc, ap = tiles
    assert auroc['baselines'][0]['value'] == expected['baselines']['auroc']
    assert ap['baselines'][0]['value'] == pytest.approx(
        expected['baselines']['ap'], abs=5e-5)
    assert auroc['callout'] == '+0.221 vs chance'
    assert ap['callout'] == '+0.0993 vs base rate'
    assert auroc['gap_text'] == f'holdout minus training: {MINUS}0.121'


@answer_key
@pytest.mark.parametrize('task', ['regression', 'forecasting'])
def test_regression_baselines_match_evidence(task):
    expected = _evidence('values.json')[task]
    tiles = _metrics_card(_load(task))['views'][0]['tiles']
    assert [t['metric'] for t in tiles] == ['mae', 'rmse']
    digits = 1 if task == 'regression' else 4
    for tile in tiles:
        values = {b['name']: b['value'] for b in tile['baselines']}
        for key, name in (('zero', 'predict zero'), ('mean',
                                                     'predict the mean')):
            assert round(values[name], digits) == pytest.approx(
                expected['baselines'][key][tile['metric']])
    callouts = {t['metric']: t['callout'] for t in tiles}
    if task == 'regression':
        assert callouts == {
            'mae': '+23.6 vs predict zero',
            'rmse': f'{MINUS}12.4 vs predict the mean'
        }
    else:
        assert callouts == {
            'mae': '+0.00471 vs predict zero',
            'rmse': '+0.00157 vs predict the mean'
        }


@answer_key
def test_link_prediction_random_baselines_match_evidence():
    expected = _evidence('values.json')['link_prediction']
    loaded = _load('link_prediction')
    cat = _metrics.catalog(loaded)
    assert cat is not None and cat.size == expected['catalog']
    assert round(cat.mean_targets, 2) == expected['mean_targets']
    tiles = _metrics_card(loaded)['views'][0]['tiles']
    assert [t['metric']
            for t in tiles] == ['map@10', 'recall@10', 'precision@10']
    assert tiles[0]['baselines'] == [] and tiles[0]['callout'] is None
    for tile, key in zip(tiles[1:], ['recall@10', 'precision@10']):
        assert round(tile['baselines'][0]['value'],
                     6) == pytest.approx(expected['random'][key])
    assert [t['callout'] for t in tiles[1:]] == ['55.7× random', '44.1× random']
    rows = _metrics_card(loaded)['views'][1]['table']['rows']
    assert rows[-1]['name'] == 'Locality score'
    assert rows[-1]['values'][0] == pytest.approx(
        expected['extra']['test_refit_locality_score'])


@answer_key
@pytest.mark.parametrize('task', TASKS)
def test_importance_matches_evidence(task):
    expected = _evidence('values.json')[task]['explain']
    loaded = _load(task)
    card = _importance(loaded)
    sensitivity = card['views'][0]
    top = [(bar['label'], round(bar['share'] * 100, 1))
           for bar in sensitivity['bars'][:10]]
    assert top == [tuple(pair) for pair in expected['top10']]
    assert round(sensitivity['remainder']['share'] * 100,
                 1) == pytest.approx(expected['other'])
    assert sensitivity['remainder']['label'] == (
        f"{expected['other_n']} other columns")
    assert {
        t['table']: round(t['share'] * 100, 1)
        for t in sensitivity['tables']
    } == expected['by_table']
    assert sensitivity['top_callout'] == (
        f"{expected['top10'][0][1]:.1f}% of total")
    if 'variation_top10' in expected:
        variation = card['views'][1]
        got = [(bar['label'], round(bar['share'] * 100, 2))
               for bar in variation['bars'][:10]]
        assert got == [(name, share)
                       for name, _, share in expected['variation_top10']]
        assert round(variation['remainder']['share'] * 100,
                     1) == (expected['variation_other'])
    else:
        assert [v['id'] for v in card['views']] == ['sensitivity']
    # Forecasting replays 50 batches of 128 of its 33,603 test rows.
    if task == 'forecasting':
        assert card['coverage'] == ('based on the first 6,400 of 33,603 test '
                                    'rows')
    else:
        assert card['coverage'] is None


@answer_key
def test_dominant_group_callouts_match_selection():
    variation = _importance(_load('classification'))['views'][1]
    callouts = {
        bar['label']: bar['callout']
        for bar in variation['bars'][:10] if bar['callout']
    }
    assert callouts == {
        'transactions.postal_code': '94.9% in OTHER(*)',
        'products.product_code': '98.1% in OTHER(*)',
        'transactions.club_member_status': '94.3% in ACTIVE',
    }


@answer_key
@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_groups_match_evidence(task):
    expected = _evidence('values.json')[task]['explain']
    card = _columns.groups(_load(task))
    assert card is not None
    for view, key in ((card['columns'][card['default']], 'group_default'),
                      (card['columns']['transactions|1|COUNT(*)'],
                       'group_history')):
        want = expected[key]
        assert [(g['label'], round(g['share'] * 100,
                                   1), round(g['predicted'],
                                             3), round(g['actual'], 3))
                for g in view['groups']] == [tuple(g) for g in want['groups']]
        assert round(view['average_actual'], 3) == want['pop_avg_target']
        assert round(view['average_predicted'], 3) == want['pop_avg_pred']
        gap = want['largest_gap_group']
        assert view['groups'][view['gap_callout']['index']]['label'] == gap[0]
    default = card['columns'][card['default']]
    if task == 'classification':
        assert default['header'] == ('transactions.transaction_time · hop '
                                     '1 · timestamp · variation '
                                     '6.0%')
        assert default['average_callout'] == ('average actual 0.761 · '
                                              'average predicted 0.759')
        assert default['gap_callout']['text'] == ('predicted 0.675 · '
                                                  'actual 0.763')
    else:
        assert default['average_callout'] == ('average actual 131.9 · '
                                              'average predicted 87.8')
        assert default['gap_callout']['text'] == ('predicted 75.9 · '
                                                  'actual 198.4')
    history = card['columns']['transactions|1|COUNT(*)']
    assert ' · count · variation ' in history['header']
    assert history['groups'][0]['share_text'].endswith('of customers')
    assert default['groups'][0]['share_text'].endswith(
        'of sampled transactions rows')


@answer_key
@pytest.mark.parametrize('task', ['forecasting', 'link_prediction'])
def test_no_group_card_without_column_analysis(task):
    assert _columns.groups(_load(task)) is None


def test_holdout_coverage_counts_saved_rows(tmp_path):
    files = _files.open_run(
        str(
            _run_dir(tmp_path, test_rows=100, batch=64, max_test_steps=2000,
                     holdout_rows=60)))
    loaded = _run.load(files)
    assert loaded.coverage.line(
        loaded.coverage.holdout) == ('based on the first 60 of 100 test rows')


# ------------------------------------------------- per-model-type cards


def test_equal_count_groups_break_ties_by_entity():
    values = pd.Series([1.0, 1.0, 1.0, 2.0, 0.5]).to_numpy()
    entities = pd.Series([5, 3, 4, 1, 2]).to_numpy()
    groups = _holdout.equal_count_groups(values, entities, 5)
    # Sorted order: 0.5 (entity 2), 1.0 by entity 3, 4, 5, then 2.0.
    assert list(groups) == [3, 1, 2, 4, 0]


@pytest.mark.parametrize('task',
                         ['binary_classification', 'regression', 'forecasting'])
def test_tenths_only_include_observed_groups(tmp_path, task):
    run = _run_dir(tmp_path, task_type=task, test_rows=3, batch=4,
                   max_test_steps=20)
    frame = pd.DataFrame({
        'ENTITY': [1, 2, 3],
        'TARGET': [0, 1, 1],
        'TARGET_PRED': [0.1, 0.5, 0.9],
        'True_PROB': [0.1, 0.5, 0.9],
        'False_PROB': [0.9, 0.5, 0.1],
        'TIMESTAMP': pd.to_datetime(['2024-01-02'] * 3),
        'ANCHOR_TIMESTAMP': pd.to_datetime(['2024-01-01'] * 3)
    })
    _write_holdout(run, frame)
    data = open_run(str(run)).data
    json.dumps(data, allow_nan=False)
    charts = [
        v for tab in data['tabs'] for card in tab['cards']
        for v in card.get('views', []) if v.get('id') == 'tenths'
    ]
    assert len(charts) == 1
    rows = charts[0]['tenths']
    assert [r['label'] for r in rows] == ['1', '4', '7']
    if task == 'forecasting':
        assert charts[0]['top_callout'].startswith(rows[-1]['share_text'])
    else:
        assert charts[0]['gap_callout'].startswith(
            'tenth ' + rows[charts[0]['highlight']]['label'] + ':')


@pytest.mark.parametrize('days, word', [(1, 'day'), (7, 'week'), (31, 'month'),
                                        (3, 'step')])
def test_forecast_period_word(days, word):
    steps = pd.Series(pd.date_range('2024-01-01', periods=4, freq=f'{days}D'))
    assert _forecasting.period(steps) == word


def _write_holdout(run: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(run / 'published-model' / 'holdout.parquet')


def _classification_run(tmp_path: Path) -> _run.Run:
    run = _run_dir(tmp_path, test_rows=6, batch=64, max_test_steps=2000,
                   task_type='binary_classification')
    scores = [0.9, 0.8, 0.6, 0.5, 0.3, 0.1]
    _write_holdout(
        run,
        pd.DataFrame({
            'ENTITY': range(6),
            'TARGET': [True, True, False, True, False, False],
            'True_PROB': scores,
            'False_PROB': [1 - s for s in scores],
        }))
    return _run.load(_files.open_run(str(run)))


def test_cutoff_card_counts_scores_above_the_cutoff(tmp_path):
    card = _classification.cutoff_card(_classification_run(tmp_path))
    assert card is not None and card['default'] == 50
    true, false = card['classes']
    # 0.5 is not flagged at cutoff 0.50: TP 2, FP 1, FN 1, TN 2.
    assert true['cutoffs'][50]['counts'] == [2, 1, 1, 2]
    assert true['cutoffs'][50]['matrix_callout'] == (
        '50.0% flagged at cutoff 0.50')
    # UI switches the original confusion counts, including threshold ties.
    assert false['cutoffs'][50]['counts'] == [2, 1, 1, 2]
    nothing = true['cutoffs'][100]
    assert [i['text'] for i in nothing['strip']
            ] == ['1.00', '0.00', '0.500', '0.0%', '0.00']
    assert nothing['precision'] == 1.0
    # The score histogram's axis is True_PROB, so the False class's cutoff
    # line is labelled with its value on that axis (W5).
    assert true['cutoffs'][57]['line_callout'] == 'cutoff 0.57'
    assert false['cutoffs'][57]['line_callout'] == 'cutoff 0.43'
    # Three of six are flagged from 0.50 to 0.59; the lowest such cutoff wins.
    assert true['preset'] == 50
    assert [v['label'] for v in card['views']] == [
        'Confusion matrix', 'ROC', 'Precision-recall', 'Cumulative gain'
    ]


@pytest.mark.parametrize('outcome', [True, False])
def test_cutoff_card_retains_single_outcome_diagnostics(tmp_path, outcome):
    run = _run_dir(tmp_path, test_rows=3, batch=64, max_test_steps=2000,
                   task_type='binary_classification')
    _write_holdout(
        run,
        pd.DataFrame({
            'ENTITY': range(3),
            'TARGET': [outcome] * 3,
            'True_PROB': [0.9, 0.8, 0.7],
            'False_PROB': [0.1, 0.2, 0.3],
        }))
    loaded = _run.load(_files.open_run(str(run)))
    cutoff = _classification.cutoff_card(loaded)
    scores = _classification.scores_card(loaded)
    assert cutoff is not None and scores is not None
    point = cutoff['classes'][0]['cutoffs'][50]
    assert point['counts'] == ([3, 0, 0, 0] if outcome else [0, 3, 0, 0])
    assert point['fpr'] == 1.0
    assert point['recall'] == (1.0 if outcome else 0.0)
    json.dumps([cutoff, scores], allow_nan=False)


@pytest.mark.parametrize('distribution',
                         [None, ['invalid'], {
                             'histogram': ['invalid']
                         }])
def test_regression_without_bin_width_omits_binned_views(
        tmp_path, distribution):
    run = _run_dir(tmp_path, test_rows=30, batch=64, max_test_steps=2000)
    stats = tmp_path / 'snap/train_tables/tt-run/split_stats.json'
    value = json.loads(stats.read_text())
    value['test_table_stats']['target_distribution'] = distribution
    _write_json(stats, value)
    loaded = _run.load(_files.open_run(str(run)))
    card = _regression.predicted_card(loaded)
    assert card is not None
    assert [v['id'] for v in card['views']] == ['tenths', 'bands']
    assert 'Predicted against actual' in open_run(str(run))._repr_svg_()


@pytest.mark.parametrize('distant', [False, True, 'folded'])
def test_regression_distributions_keep_offset_and_sparse_bins(
        tmp_path, distant):
    run = _run_dir(tmp_path, test_rows=30, batch=64, max_test_steps=2000)
    loaded = _run.load(_files.open_run(str(run)))
    actual = pd.Series([10000.0] * 15 + [10001.0] * 15).to_numpy()
    pred = pd.Series([20000.0 if distant else 10001.0] * 30).to_numpy()
    if distant == 'folded':
        pred[-1] = 30000.0
    view = _regression._distributions(loaded, actual, pred, 0.02)
    assert view['start'] == 10000.0 and view['width'] == 0.02
    assert [sum(s['counts']) for s in view['series']] == [30, 30]
    if distant:
        folded = distant == 'folded'
        assert view['bin_count'] == 500001 + folded
        assert view['indices'] == [0, 50, 500000] + ([500001] if folded else [])
        assert len(view['bins']) == 3 + folded
        if folded:
            assert view['bins'][-1].startswith('≥ ')
            assert view['series'][1]['counts'][-1] == 1
    else:
        assert len(view['bins']) == 51
        assert 'indices' not in view


@pytest.mark.parametrize(
    'saved',
    ['valid', 'invalid', 'absent', 'merged', 'edge', 'separated', 'constant'])
def test_regression_heatmap_prefers_saved_artifact(tmp_path, saved):
    run = _run_dir(tmp_path, test_rows=20, batch=64, max_test_steps=2000)
    loaded = _run.load(_files.open_run(str(run)))
    if saved != 'absent':
        _write_json(
            Path(loaded.files.trial) / 'test_target_prediction_heatmap' /
            'heatmap.json', {
                'cell_width': 1.0 if saved == 'valid' else 0.0,
                'label_bin_start': -1.0,
                'label_bin_end': 0.0,
                'prediction_bin_start': -1.0,
                'prediction_bin_end': 0.0,
                'cells': [[-1.0, -1.0], [0.0, 0.0]],
                'counts': [11, 1],
                'home_dir': '',
            })
    if saved in ('merged', 'edge'):
        end_x, end_y = (200, 200) if saved == 'merged' else (110, 366)
        _write_json(
            Path(loaded.files.trial) / 'test_target_prediction_heatmap' /
            'heatmap.json', {
                'cell_width': 1.0,
                'label_bin_start': 0.0,
                'label_bin_end': end_x,
                'prediction_bin_start': 0.0,
                'prediction_bin_end': end_y,
                'cells': [[0, 0], [1 if saved == 'merged' else end_x, 0]],
                'counts': [12, 7],
                'home_dir': '',
            })
    if saved in ('separated', 'constant'):
        end_x, start_y, end_y = ((1.0, 10000.0,
                                  10001.0) if saved == 'separated' else
                                 (0.0, 0.0, 1_000_000.0))
        _write_json(
            Path(loaded.files.trial) / 'test_target_prediction_heatmap' /
            'heatmap.json', {
                'cell_width': 0.02,
                'label_bin_start': 0.0,
                'label_bin_end': end_x,
                'prediction_bin_start': start_y,
                'prediction_bin_end': end_y,
                'cells': [[0.0, start_y], [end_x, end_y]],
                'counts': [12, 7],
                'home_dir': '',
            })
    heatmap = _regression._heatmap(loaded, 1.0)
    assert heatmap is not None
    json.dumps(heatmap, allow_nan=False)
    if saved in ('merged', 'edge', 'separated', 'constant'):
        assert len(heatmap['rows']) * len(heatmap['columns']) <= 10_000
        assert heatmap['x_edges'][0] == heatmap['y_edges'][0]
        assert sum(map(sum, heatmap['counts'])) == 19
        assert heatmap['coverage'] is None
    elif saved == 'valid':
        assert heatmap['counts'] == [[11, 0], [0, 1]]
        assert heatmap['trimmed']['counts'] == [[11]]
        assert heatmap['coverage'] is None
        (run / 'published-model' / 'holdout.parquet').unlink()
        without_holdout = _run.load(_files.open_run(str(run)))
        card = _regression.predicted_card(without_holdout)
        assert card is not None
        assert [v['id'] for v in card['views']] == ['heatmap']
        image = ElementTree.fromstring(open_run(str(run))._repr_svg_())
        assert len(image.findall('.//{*}rect[@opacity]')) == 2
        labels = list(image.itertext())
        assert 'actual value' in labels and 'predicted value' in labels
        assert any('Densest cell' in label for label in labels)
    else:
        assert sum(map(sum, heatmap['counts'])) == 20
        assert heatmap['coverage'] == 'Based on 20 of 20 retained holdout rows.'


def test_regression_bands_without_a_zero_band(tmp_path):
    run = _run_dir(tmp_path, test_rows=20, batch=64, max_test_steps=2000)
    _write_holdout(
        run,
        pd.DataFrame({
            'ENTITY': range(20),
            'TARGET': [float(i + 1) for i in range(20)],
            'TARGET_PRED': [10.0] * 20,
        }))
    card = _regression.predicted_card(_run.load(_files.open_run(str(run))))
    assert card is not None
    bands = card['views'][1]['bands']
    assert [b['rows'] for b in bands] == [4] * 5
    assert bands[0]['label'] == '1.00 to 4.00'


def test_missing_holdout_keeps_the_other_cards(tmp_path):
    run = _run_dir(tmp_path, test_rows=30, batch=64, max_test_steps=2000)
    (run / 'published-model' / 'holdout.parquet').unlink()
    data = _cards.build(_run.load(_files.open_run(str(run))))
    assert [[c['id'] for c in tab['cards']]
            for tab in data['tabs']] == [['metrics'], ['importance']]


def test_build_lists_regression_cards(tmp_path):
    run = _run_dir(tmp_path, test_rows=30, batch=64, max_test_steps=2000)
    data = _cards.build(_run.load(_files.open_run(str(run))))
    assert data['model_type'] == 'regression'
    assert [
        (tab['label'], [c['id'] for c in tab['cards']]) for tab in data['tabs']
    ] == [('Evaluation', ['metrics', 'predicted']),
          ('Explanation', ['importance'])]


@answer_key
@pytest.mark.parametrize('task, cards', [
    ('classification', [['metrics', 'cutoff', 'scores'],
                        ['subgraphs', 'importance', 'groups']]),
    ('regression', [['metrics', 'predicted'],
                    ['subgraphs', 'importance', 'groups']]),
    ('forecasting', [['metrics', 'forecast'], ['importance']]),
    ('link_prediction', [['metrics'], ['subgraphs', 'importance']]),
])
def test_cards_per_model_type(task, cards):
    data = _cards.build(_load(task))
    assert [[c['id'] for c in tab['cards']] for tab in data['tabs']] == cards
    json.dumps(data)  # the payload must serialize for the front end


@answer_key
def test_cutoff_counts_match_evidence():
    expected = _evidence('cutoffs.json')
    card = _classification.cutoff_card(_load('classification'))
    assert card is not None
    for cls in card['classes']:
        assert [c['counts'] for c in cls['cutoffs']] == expected[cls['name']]


@answer_key
def test_cutoff_texts_match_selection():
    card = _classification.cutoff_card(_load('classification'))
    assert card is not None
    true, false = card['classes']
    at = true['cutoffs'][50]
    assert at['cells'] == [
        '3,862 · 78.3%', '890 · 18.0%', '49 · 1.0%', '131 · 2.7%'
    ]
    assert [i['text'] for i in at['strip']
            ] == ['0.813', '0.987', '0.810', '96.4%', '0.892']
    assert at['matrix_callout'] == '96.4% flagged at cutoff 0.50'
    assert at['roc_callout'] == 'cutoff 0.50 · TPR 0.987 · FPR 0.872'
    assert at['gain_callout'] == '96.4% flagged · 98.7% of True captured'
    assert (true['auroc_callout'], true['ap_callout'],
            true['base_rate_callout']) == ('AUROC 0.721', 'AP 0.892',
                                           'base rate 0.793')
    assert true['preset'] == 65
    preset = true['cutoffs'][65]
    assert preset['counts'] == [3360, 551, 551, 470]
    assert [i['text'] for i in preset['strip']
            ] == ['0.859', '0.859', '0.777', '79.3%', '0.859']
    f = false['cutoffs'][50]
    assert f['counts'] == [131, 49, 890, 3862]
    assert [i['text'] for i in f['strip']
            ] == ['0.728', '0.128', '0.810', '3.6%', '0.218']
    assert f['gain_callout'] == '3.6% flagged · 12.8% of False captured'
    assert (false['ap_callout'], false['base_rate_callout']) == ('AP 0.466',
                                                                 'base rate '
                                                                 '0.207')
    assert false['preset'] == 35
    assert false['cutoffs'][35]['roc_callout'] == (
        'cutoff 0.35 · TPR 0.460 · FPR 0.141')


@answer_key
def test_scores_card_matches_evidence():
    expected = _evidence('values.json')['classification']
    card = _classification.scores_card(_load('classification'))
    assert card is not None
    tenths, outcome = card['views']
    assert [(t['rows'], round(t['score'], 3), round(t['observed'], 3))
            for t in tenths['tenths']
            ] == [(n, p, o) for _, n, p, o, _, _ in expected['tenths']]
    assert tenths['reference_callout'] == 'all customers: 0.793'
    assert tenths['gap_callout'] == 'tenth 4: predicted 0.767 · observed 0.870'
    # The evidence script builds its edges with a floating-point arange that
    # drops the last edge, so it counts 25 bins over 0.44 to 0.96; the
    # selection's rule (minimum rounded down to maximum rounded up) gives 26.
    low, high, _ = expected['dist_bins']
    assert (outcome['edges'][0], outcome['edges'][-1],
            len(outcome['edges']) - 1) == (low, high, 26)
    assert sum(sum(s['counts']) for s in outcome['series']) == 4932
    assert outcome['tallest_callout'] == ('1,386 customers (28.1%) score '
                                          '0.86 to 0.88')
    true = outcome['series'][0]['counts'][outcome['tallest']]
    assert true == expected['dist_tallest'][4]
    assert [s['median_callout']
            for s in outcome['series']] == ['median 0.874', 'median 0.667']


@answer_key
def test_regression_card_matches_selection():
    expected = _evidence('values.json')['regression']
    card = _regression.predicted_card(_load('regression'))
    assert card is not None
    tenths, bands, heatmap, dist = card['views']
    assert [(round(t['predicted'], 1), round(t['actual'], 1))
            for t in tenths['tenths']] == [(p, a)
                                           for _, p, a in expected['tenths']]
    assert tenths['reference_callout'] == 'all customers: 113.0'
    assert tenths['gap_callout'] == 'tenth 10: predicted 214.9 · actual 410.5'
    assert [
        (b['rows'], round(b['median'], 1), round(b['error_share'] * 100, 1))
        for b in bands['bands']
    ] == [(n, med, se) for _, n, _, _, _, _, med, se, _ in expected['bands']]
    assert [b['label'] for b in bands['bands']] == [
        '0', '56.0 to 192.2', '192.2 to 363.8', '364.4 to 683.8',
        '687.2 to 5,989.4'
    ]
    assert bands['share_callout'] == '86.2% of squared error'
    # UI defaults to full signed bins; Hide Outliers removes sparse tails.
    assert (len(heatmap['columns']), len(heatmap['rows'])) == (35, 6)
    assert heatmap['x_edges'][0] < 0 and heatmap['y_edges'][0] < 0
    assert sum(map(sum, heatmap['counts'])) == 4932
    trimmed = heatmap['trimmed']
    assert (len(trimmed['columns']), len(trimmed['rows'])) == (11, 4)
    assert sum(map(sum, trimmed['counts'])) == 4906
    assert heatmap['densest'] == [1, 1]
    assert heatmap[
        'coverage'] == 'Based on 4,932 of 4,932 retained holdout rows.'
    assert dist['start'] < 0
    assert dist['series'][1]['counts'][0] == 3
    actual, predicted = dist['series']
    assert [i['text'] for i in actual['stats']['items']
            ] == ['0.0', '0.0', '0.0', '113.0', '0.0', '5,989.4']
    assert [i['text'] for i in predicted['stats']['items']
            ] == ['−6.8', '21.6', '47.8', '68.0', '90.7', '577.0']
    assert (actual['max_callout'],
            predicted['max_callout']) == ('highest actual 5,989.4',
                                          'highest prediction 577.0')


@answer_key
def test_forecast_card_matches_evidence():
    expected = _evidence('values.json')['forecasting']
    card = _forecasting.forecast_card(_load('forecasting'))
    assert card is not None
    horizon, totals, tenths = card['views']
    assert horizon['label'] == 'By horizon week'
    assert totals['label'] == 'Weekly totals'
    assert [(s['label'], round(s['model'], 4), round(s['zero'], 4))
            for s in horizon['steps']
            ] == [(d, m, z) for d, m, z, _, _, _ in expected['weeks']]
    assert [s['bias_callout'] for s in horizon['steps']] == [
        'bias −0.0373', 'bias −0.0253', 'bias −0.0257', 'bias −0.0488'
    ]
    assert [(s['actual_text'], s['forecast_text'])
            for s in totals['steps']] == [('1,428', '173'), ('1,026', '175'),
                                          ('1,045', '180'), ('1,822', '184')]
    assert totals['total_callout'] == (
        'forecast total 13.4% of actual over 4 weeks')
    assert [
        (t['entities'], round(t['share'] * 100, 1)) for t in tenths['tenths']
    ] == [(n, s) for _, n, s, _, _ in expected['tenths']]
    assert tenths['top_callout'] == '42.8% of actual units · 3,360 products'


# ------------------------------------------------------------- rendering

# Words and fragments that must never reach the rendered widget: scaffolding,
# internal names and file names (the captain's decision 21).
FORBIDDEN = [
    'DEFAULT', 'reads:', 'refit', 'first fit', 'test_refit', 'Insight', '.json',
    '.parquet', '.csv', 'TARGET_PRED', 'True_PROB', 'U08', 'N04'
]


def _texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _texts(v)]
    if isinstance(value, list):
        return [t for v in value for t in _texts(v)]
    return []


def _visible_texts(data: dict[str, Any]) -> list[str]:
    # Cohort keys link importance bars to column groups; they are never shown.
    out = []
    for tab in data['tabs']:
        for card in tab['cards']:
            for key, value in card.items():
                if key not in ('columns', 'default', 'id'):
                    out += _texts(value)
            if card['id'] == 'groups':
                for col in card['columns'].values():
                    out += _texts({k: v for k, v in col.items() if k != 'key'})
    return out + _texts(data['header'])


def test_render_html_is_self_contained(tmp_path):
    run = _run_dir(tmp_path, test_rows=30, batch=64, max_test_steps=2000)
    view = open_run(str(run))
    html = view.html()
    assert view.model_type == 'regression'
    assert html.startswith('<div id="kumo-')
    assert 'KX.mount(host, data, signal, onDispose);' in html
    assert 'KX.cards.predicted' in html  # the regression file
    assert 'KX.cards.cutoff' not in html  # not the classification file
    assert html.count('</script>') == 2  # the data and the code, nothing else
    if shutil.which('node'):
        # Validate the transported source, including script-tag escaping.
        source = html.split('<script>//<![CDATA[', 1)[1].split('</script>',
                                                               1)[0]
        subprocess.run(['node', '--check'], input=source, text=True,
                       capture_output=True, check=True)

    assert view._repr_html_().startswith('<div id="kumo-')
    note = 'To activate the widget, rerun this cell.'
    image = html.split('src="data:image/svg+xml;base64,', 1)[1].split('"', 1)[0]
    snapshot = ElementTree.fromstring(base64.b64decode(image))
    labels = list(snapshot.itertext())
    assert 'Metrics' in labels and 'Holdout' in labels
    assert 'Training · last epoch' in labels
    assert 'Predicted against actual' in labels
    assert 'Column importance' not in labels
    assert f'<p>{note}</p>' in html
    assert note in list(ElementTree.fromstring(view._repr_svg_()).itertext())


def test_anywidget_keeps_static_output_with_widget_model(tmp_path):
    pytest.importorskip('anywidget')
    run = _run_dir(tmp_path, test_rows=3, batch=64, max_test_steps=2000)
    view = open_run(str(run))
    widget = view.as_widget()
    try:
        bundle, _ = widget._repr_mimebundle_()
        assert bundle['application/vnd.jupyter.widget-view+json'][
            'model_id'] == widget.model_id
        assert '<script' not in bundle['text/html']
        assert 'To activate the widget, rerun this cell.' in bundle[
            'text/plain']
        assert bundle['image/svg+xml'] == view._repr_svg_()
    finally:
        widget.close()


@pytest.mark.parametrize(('kind', 'own_card'), [
    ('classification', 'cutoff'),
    ('regression', 'predicted'),
    ('forecasting', 'forecast'),
    ('link_prediction', None),
])
def test_render_scripts_follow_model_type(kind, own_card):
    script = _render.script(kind)
    assert 'KX.cards.metrics' in script and 'KX.cards.importance' in script
    assert ('KX.cards.subgraphs =' in script) == (kind != 'forecasting')
    for name in ('cutoff', 'predicted', 'forecast'):
        assert (f'KX.cards.{name} =' in script) == (name == own_card)


@pytest.mark.parametrize(
    'source', ['bundle', 'original', 'stale', 'unlisted', 'selection'])
def test_training_result_visualize_opens_its_run(tmp_path, source):
    from kumo.engine.explain import TrainingExplanations
    from kumo.engine.trainer.trainer import _listed_holdout

    run = _run_dir(tmp_path, test_rows=30, batch=64, max_test_steps=2000)
    bundle = run / 'published-model'
    holdout = bundle / 'holdout.parquet'
    if source == 'original':
        holdout = holdout.rename(run / 'holdout.parquet')
    elif source == 'stale':
        holdout = run / 'removed.parquet'
    elif source == 'unlisted':
        _write_json(bundle / 'publication-manifest.json', {'artifacts': []})
        holdout, _ = _listed_holdout(bundle)
        assert holdout is None
    result = TrainingResult(metrics={}, artifact_paths={}, bundle_dir=bundle,
                            holdout_path=holdout, holdout_truncated=True,
                            explanations=TrainingExplanations.load(str(run)))
    if source == 'selection':
        next(run.glob('models/*/automl/best_trainer.json')).unlink()
        assert result.explanations.gradient_scores().tables
    if source in ('original', 'bundle'):
        assert len(result.holdout_df()) == 30
    view = result.visualize()
    assert view.data['tabs'][0]['cards'][0]['id'] == 'metrics'
    assert any(card['id'] == 'importance' for tab in view.data['tabs']
               for card in tab['cards'])
    if source == 'unlisted':
        assert view.data['header']['holdout'] is None
        assert 'predicted' not in [
            c['id'] for c in view.data['tabs'][0]['cards']
        ]
    else:
        assert view.data['header']['holdout'].startswith('Holdout: 30 users')
        assert 'predicted' in [c['id'] for c in view.data['tabs'][0]['cards']]
        assert 'truncated' in view._repr_svg_()


@answer_key
@pytest.mark.parametrize('task', TASKS)
def test_rendered_text_has_no_scaffolding(task):
    roots = glob.glob(f'{RUNS}/{task}/scratch/trainers/*/runs/*-fit-*')
    data = open_run(roots[0]).data
    assert data['model_type'] == task
    for text in _visible_texts(data):
        for word in FORBIDDEN:
            assert word not in text, (word, text)


@answer_key
def test_header_matches_selection():
    assert [
        open_run(
            glob.glob(f'{RUNS}/{task}/scratch/trainers/*/runs/*-fit-*')
            [0]).data['header']['holdout'] for task in TASKS
    ] == [
        'Holdout: 4,932 customers · anchor 2020-05-23 · 79.3% True',
        'Holdout: 4,932 customers · anchor 2020-05-23 · 79.3% with actual 0',
        'Holdout: 33,603 products × 4 weeks = 134,412 rows · anchor '
        '2020-05-25',
        'Holdout: 1,021 customers · anchor 2020-05-23 · 4.86 items each · '
        'catalog 33,603 products',
    ]


@answer_key
def test_axis_titles_name_the_query_target():
    card = _regression.predicted_card(_load('regression'))
    assert card is not None
    views = {v['id']: v for v in card['views']}
    assert views['tenths']['y'] == 'average SUM(transactions.price)'
    assert views['distributions']['x'] == 'SUM(transactions.price, 0, 30, days)'
    for task, axis in (('classification', 'share True'),
                       ('regression',
                        'average SUM(transactions.price, 0, 30, days)')):
        card = _columns.groups(_load(task))
        assert card is not None
        assert {c['axis'] for c in card['columns'].values()} == {axis}


@pytest.mark.parametrize('failure', [None, 'missing', 'dangling', 'malformed'])
def test_subgraphs_use_typed_artifacts_and_keep_missing_distinct(
        tmp_path, failure):
    run_path = _run_dir(tmp_path, test_rows=3, batch=64, max_test_steps=2000)
    run = _run.load(_files.open_run(str(run_path)))
    root = Path(run.files.trial) / 'subgraph_collector' / 'test'
    _write_json(
        root / 'strategy_mappings.json', {
            'mappings': {
                'random': ['0', '1']
            },
            'orig_mappings': {
                'random': ['account-x', 'account-y']
            },
        })
    graph = {
        'subgraph_store': {
            'root_id': 0,
            'task_type': 'regression',
            'nodes': {
                '0': {
                    'id': 0,
                    'pkey_name': 'account_key',
                    'pkey': 9007199254740993
                }
            },
            'edges': {},
            'tables': {
                'accounts': [0]
            },
        },
        'feature_store': {
            'features': {
                '0': {
                    'balance': {
                        'name': 'balance',
                        'value': 0,
                        'score': 0.0
                    },
                    'note': {
                        'name': 'note',
                        'value': ['</script>', {
                            'missing': float('nan')
                        }],
                        'score': None
                    },
                }
            }
        },
    }
    _write_json(root / 'subgraph_json' / '0.json', graph)
    if failure != 'missing':
        other = json.loads(json.dumps(graph))
        if failure == 'dangling':
            other['subgraph_store']['edges'] = {'0': {'related': [99]}}
        _write_json(root / 'subgraph_json' / '1.json', other)
        if failure == 'malformed':
            (root / 'subgraph_json' / '1.json').write_text('not json')
    result = _subgraphs.card(run)
    assert result is not None
    assert len(result['samples']) == (2 if failure is None else 1)
    assert result['unavailable_count'] == (0 if failure is None else 1)
    sample = result['samples'][0]
    assert sample['categories'] == ['random']
    assert sample['original_ids'] == {'random': 'account-x'}
    assert sample['prediction'] is None and sample['target'] is None
    assert sample['nodes'][0]['value'] == '9007199254740993'
    assert [f['score'] for f in sample['nodes'][0]['features']] == [0.0, None]
    assert result['entity_label'] == 'users'
    json.dumps(result, allow_nan=False)
    assert sample['nodes'][0]['features'][1]['value'][1] == {'missing': None}
    assert _subgraphs.card(_run.load(_files.locate(run.files.bundle))) is None
