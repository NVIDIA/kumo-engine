# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data side of the post-training explainability widget, on small runs."""

import base64
import json
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest
from kumo_core.artifacts import add_bundle_file
from kumo_core.io import filesystem
from kumo_core.io.json import to_json_dict
from kumoapi.explain.subgraph import Feature
from kumoapi.typing import Stype
from test_explanation_artifacts import _gradient, _graph
from test_explanation_artifacts import _run as _retained_run

from kumo.engine import _explainability
from kumo.engine._explainability import _classification, _columns, _forecasting
from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability import (
    _metrics,
    _ranking,
    _regression,
    _run,
    _subgraphs,
)
from kumo.engine.explain import TrainingExplanations
from kumo.engine.trainer.trainer import TrainingResult

MINUS = '−'


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


def test_shared_scale_and_percent():
    assert fmt.numbers([-6.8, 21.6, 577.0]) == [f'{MINUS}6.8', '21.6', '577.0']
    assert fmt.numbers([0.761, 0.0759]) == ['0.761', '0.0759']
    assert fmt.percent(0.79298) == '79.3%'


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _result(tmp_path: Path, task_type: str = 'regression', *,
            test_rows: int = 30, batch: int = 64, max_test_steps: int = 2000,
            holdout: pd.DataFrame | None = None,
            scores: dict[str, float] | None = None, tune: str = 'mae',
            refit: bool = True) -> TrainingResult:
    """A fitted result with the files the widget reads."""
    run, trial = _retained_run(tmp_path, refit=refit)
    metrics = ['mae', 'mse', 'rmse'] if scores is None else sorted(
        {k.split('_', 1)[1]
         for k in scores if k.startswith('test_')})
    bundle = run / 'published-model'
    table = tmp_path / 'snap' / 'train_tables' / 'tt-run'
    files = dict(
        prediction_schema=('prediction-schema.json', dict(task_type=task_type)),
        resolved_model_plan=('model/resolved-model-plan.json',
                             dict(
                                 training_job=dict(metrics=metrics,
                                                   tune_metric=tune),
                                 optimization=dict(
                                     max_test_steps=max_test_steps))),
        trainer_config=('model/config.json',
                        dict(loader_kwargs=dict(batch_size=batch),
                             eval_loader_kwargs=None)),
        training_spec=('training-spec.json',
                       dict(
                           spec_version=1, model_plan={}, snapshot_id='s',
                           training_table_generation={},
                           predictive_query='PREDICT SUM(orders.value, 0, '
                           '30, days) FOR EACH users.user_id')),
    )
    for name, value in files.values():
        _write_json(bundle / name, value)
    _write_json(bundle / 'publication-manifest.json', {'artifacts': []})
    if holdout is None:
        holdout = pd.DataFrame(
            dict(ENTITY=range(test_rows), TARGET_PRED=1.0,
                 TARGET=[float(i % 3) for i in range(test_rows)]))
    holdout.to_parquet(bundle / 'holdout.parquet')
    add_bundle_file(bundle, 'holdout', 'holdout.parquet')
    _gradient(trial)
    split = dict(num_train_rows=test_rows, anchor_date='2024-01-01T00:00:00',
                 min_date='2024-01-01T00:00:00', max_date='2024-01-01T00:00:00')
    _write_json(
        table / 'split_stats.json',
        dict(test_table_stats=split, val_table_stats={
            **split, 'num_train_rows': 40
        }, train_table_stats={
            **split, 'num_train_rows': 400
        }))
    _write_json(
        run / 'task_dataset' / 'runtime_manifest.json',
        dict(graph=dict(index_path=str(tmp_path / 'snap')),
             stages=dict(test=dict(index_path=str(table)))))
    return TrainingResult(
        metrics=scores
        or dict(test_mae=2.0, test_mse=9.0, test_rmse=3.0, test_refit_mae=1.5,
                test_refit_mse=4.0, test_refit_rmse=2.0), artifact_paths={
                    k: str(bundle / v[0])
                    for k, v in files.items()
                }, bundle_dir=bundle, holdout_path=bundle / 'holdout.parquet',
        explanations=TrainingExplanations.load(run))


def _card(cards: Any, result: TrainingResult) -> dict[str, Any]:
    card = cards(_run.load(result))
    assert card is not None
    return card


def _binary(tmp_path: Path, scores: list[float],
            target: list[bool]) -> TrainingResult:
    return _result(
        tmp_path, 'binary_classification', test_rows=len(scores),
        holdout=pd.DataFrame(
            dict(ENTITY=range(len(scores)), TARGET=target, True_PROB=scores,
                 False_PROB=[1 - s for s in scores])))


@pytest.mark.parametrize(
    'test_rows,batch,steps,holdout_rows,metrics,explanation,holdout', [
        (100, 8, 10, 100, 'the first 80 of 100', None, None),
        (100, 64, 2000, 100, None, None, None),
        (33_603, 512, 2000, 33_603, None, 'the first 6,400 of 33,603', None),
        (100, 64, 2000, 60, None, None, 'the first 60 of 100'),
    ])
def test_coverage_lines_name_only_caps_that_apply(tmp_path, test_rows, batch,
                                                  steps, holdout_rows, metrics,
                                                  explanation, holdout):
    # The explanation pass replays 50 batches of a quarter batch.
    result = _result(
        tmp_path, test_rows=test_rows, batch=batch, max_test_steps=steps,
        holdout=pd.DataFrame(
            dict(ENTITY=range(holdout_rows), TARGET=1.0, TARGET_PRED=1.0)))

    def line(text):
        return None if text is None else f'based on {text} test rows'

    assert _card(_metrics.card, result)['coverage'] == line(metrics)
    assert _card(_columns.importance, result)['coverage'] == line(explanation)
    assert _run.load(result).holdout_coverage() == line(holdout)


def test_forecast_coverage_counts_entity_anchors(tmp_path):
    run = _run.load(_result(tmp_path))
    run.holdout = pd.DataFrame(
        dict(ENTITY=[1, 1, 2, 2], ANCHOR_TIMESTAMP=0, TIMESTAMP=[0, 1, 0, 1]))
    assert run.holdout_coverage() == 'based on the first 2 of 30 test rows'


def _scored(run: _run.Run, model_type: str, tune: str, names: list[str]):
    run.model_type = model_type
    run.plan = dict(training_job=dict(metrics=names, tune_metric=tune))
    run.metrics = {f'test_refit_{name}': 0.1 for name in names}
    return _metrics.card(run)


def test_metrics_card_is_one_view_of_pills_and_a_table(tmp_path):
    result = _result(tmp_path)
    result.metrics.update(train_mae=0.8, val_mae=1.7)
    card = _card(_metrics.card, result)
    [view] = card['views']
    assert view['pills'] == [
        dict(name='MAE', holdout_text='1.50'),
        dict(name='RMSE', holdout_text='2.00')
    ]
    table = view['table']
    assert [c['label']
            for c in table['columns']] == ['Holdout', 'Validation', 'Training']
    assert table['rows'][0] == dict(name='MAE', tune=True, baseline=False,
                                    texts=['1.50', '1.70', '0.800'])
    # Baselines follow the model rows and are scored on the holdout alone.
    baseline = [r for r in table['rows'] if r['baseline']]
    assert table['rows'][-len(baseline):] == baseline
    assert [r['name'] for r in baseline] == [
        'Predict zero · MAE', 'Predict zero · RMSE', 'Predict the mean · MAE',
        'Predict the mean · RMSE'
    ]
    assert all(r['texts'][1:] == [None, None] for r in baseline)
    assert 'groups' not in table and 'detail' not in table['columns'][0]


def _cards(result: TrainingResult) -> dict[str, dict[str, Any]]:
    data = result.visualize().data
    return {c['id']: c for tab in data['tabs'] for c in tab['cards']}


@pytest.mark.parametrize('recorded', ['bundle', 'progress', None])
def test_epoch_history_follows_search_experiments(tmp_path, recorded):
    result = _result(tmp_path)
    epochs = {
        '3':
        dict(train_metrics=dict(train_mae=0.4),
             validation_metrics=dict(val_mae=0.8)),
        '0':
        dict(train_metrics=dict(train_mae=0.5, train_loss=float('nan'),
                                train_f1_per_label_0=0, test_mae=0)),
        'bad': [],
        '2':
        None,
    }
    trials = {
        '2': dict(metrics=epochs),
        '10': dict(metrics={'2': dict(train_metrics=dict(train_loss=1))}),
        'empty': dict(metrics={}),
    }
    if recorded == 'bundle':
        _write_json(tmp_path / 'history.json', dict(trial_progress=trials))
        result.artifact_paths['metric_history'] = str(tmp_path / 'history.json')
    elif recorded == 'progress':
        # Without a bundle record, the retained run's search progress.
        _write_json(
            Path(result.explanations.trial_root).parent /
            'auto_train_progress.json', dict(trial_progress=trials))
    tabs = result.visualize().data['tabs']
    if not recorded:
        assert 'history' not in _cards(result)
        return
    assert [tab['id']
            for tab in tabs] == ['evaluation', 'training', 'explanation']
    assert [tab['label'] for tab in tabs][:2] == ['Evaluation', 'Training']
    assert 'history' not in [c['id'] for c in tabs[0]['cards']]
    assert all(view['id'] != 'history'
               for view in _cards(result)['metrics']['views'])
    history = _cards(result)['history']
    assert history['title'] == 'Learning Curves'
    assert history['metrics'] == ['mae', 'loss']
    # The retained run selected version_2, the first of these experiments.
    assert history['experiments'] == [
        dict(
            label='Experiment 3', best=True,
            series=dict(train_mae=[(1, 0.5), (4, 0.4)], val_mae=[(3, 0.8)],
                        train_loss=[(1, None)])),
        dict(label='Experiment 11', best=False,
             series=dict(train_loss=[(2, 1)])),
    ]


def test_binary_metrics_have_no_baseline_rows(tmp_path):
    run = _run.load(_binary(tmp_path, [.9, .2, .3, .1], [True, False] * 2))
    card = _scored(run, _run.CLASSIFICATION, 'auroc', ['auroc', 'ap', 'f1'])
    assert not any(r['baseline'] for r in card['views'][0]['table']['rows'])


def test_top_k_metrics_order_by_family_then_k(tmp_path):
    names = [
        'map@10', 'recall@20', 'precision@10', 'f1@10', 'recall@10',
        'precision@20', 'map@20'
    ]
    card = _scored(_run.load(_result(tmp_path)), _run.LINK_PREDICTION, 'map@10',
                   names)
    view = card['views'][0]
    assert [p['name']
            for p in view['pills']] == ['MAP@10', 'Recall@10', 'Precision@10']
    assert [r['name'] for r in view['table']['rows']] == [
        'F1@10', 'MAP@10', 'MAP@20', 'Precision@10', 'Precision@20',
        'Recall@10', 'Recall@20'
    ]


def test_header_keeps_the_query_and_notes_a_full_refit(tmp_path):
    run = _run.load(_result(tmp_path))
    assert list(_metrics.header(run)) == ['query']
    run.full_refit = True
    assert _metrics.header(run)['note'].endswith('before full refit.')


@pytest.mark.parametrize(
    'scores, retained, best',
    [
        # The retained run names version_2, whatever the scores say.
        ([0.7, 1.7, 0.2], True, [False, False, True]),
        # Without it, the one experiment that recorded the published score.
        ([0.7, 1.7, 0.2], False, [False, True, False]),
        ([1.7, 1.7, 0.2], False, [False, False, False]),
        ([0.7, 0.2], False, [False, False]),
        ([1.7], True, [False]),
    ])
def test_epoch_history_marks_the_selected_experiment(tmp_path, scores, retained,
                                                     best):
    result = _result(tmp_path)
    trials = {
        str(i): dict(metrics={'1': dict(validation_metrics=dict(val_mae=v))})
        for i, v in enumerate(scores)
    }
    _write_json(tmp_path / 'history.json', dict(trial_progress=trials))
    result.artifact_paths['metric_history'] = str(tmp_path / 'history.json')
    result = replace(
        result, metrics={
            **result.metrics, 'val_mae': 1.7
        }, explanations=result.explanations
        if retained else TrainingExplanations.unavailable())
    history = _cards(result)['history']
    assert [e['best'] for e in history['experiments']] == best
    assert [e['label'] for e in history['experiments']
            ] == [f'Experiment {i + 1}' for i in range(len(scores))]


def _split_stats(result: TrainingResult, **splits: dict[str, Any]) -> None:
    path = (Path(result.explanations.run_root).parent / 'snap' /
            'train_tables' / 'tt-run' / 'split_stats.json')
    _write_json(path, {f'{k}_table_stats': v for k, v in splits.items()})


def test_training_table_of_a_temporal_regression(tmp_path):
    result = _result(tmp_path)

    def day(month):
        return f'2024-0{month}-01T00:00:00'

    _split_stats(
        result, train=dict(num_train_rows=960, num_entities=240,
                           min_date=day(4), max_date=day(5),
                           target_distribution=dict(target_mean=2.925)),
        val=dict(num_train_rows=480, num_entities=900, min_date=day(6),
                 max_date=day(6), target_distribution=dict(target_mean=12.5)),
        test=dict(num_train_rows=30, num_entities=30, min_date=day(7),
                  max_date=day(7),
                  target_distribution=dict(target_mean=float('nan'))))
    card = _cards(result)['training_table']
    assert card['title'] == 'Training Table'
    assert card['columns'] == [
        'Rows', 'Distinct users (approx.)', 'Target mean', 'Anchor times'
    ]
    # Entities never exceed rows, and the mean column shares one scale.
    assert card['rows'] == [
        dict(name='Training',
             texts=['960', '240', '2.9', '2024-04-01 to 2024-05-01']),
        dict(name='Validation', texts=['480', '480', '12.5', '2024-06-01']),
        dict(name='Holdout', texts=['30', '30', '', '2024-07-01']),
    ]


def test_training_table_of_a_static_binary_task(tmp_path):
    result = _result(tmp_path, 'binary_classification')
    epoch = '1970-01-01T00:00:00'

    def split(rows, true):
        return dict(
            num_train_rows=rows, num_entities=rows, min_date=epoch,
            max_date=epoch,
            target_distribution=dict(count=dict(true=true, false=rows - true)))

    _split_stats(result, train=split(148, 74), val=split(51, 28),
                 test=split(41, 15))
    card = _cards(result)['training_table']
    # Static tasks carry a placeholder anchor, which is not a time range.
    assert card['columns'] == [
        'Rows', 'Distinct users (approx.)', 'Positive rate'
    ]
    assert [r['texts'] for r in card['rows']] == [['148', '148', '50.0%'],
                                                  ['51', '51', '54.9%'],
                                                  ['41', '41', '36.6%']]


def test_training_table_without_class_balance_or_counts(tmp_path):
    result = _result(tmp_path, 'multiclass_classification')
    _split_stats(
        result, train=dict(num_train_rows=5,
                           target_distribution={}), val=dict(num_entities=3),
        test=dict(num_train_rows=2,
                  target_distribution=dict(count=dict(a=1, b=1))))
    card = _cards(result)['training_table']
    assert card['columns'] == ['Rows']
    assert card['rows'] == [
        dict(name='Training', texts=['5']),
        dict(name='Holdout', texts=['2'])
    ]
    _split_stats(result, val=dict(num_entities=3))
    assert 'training_table' not in _cards(result)


def test_training_table_from_the_bundle_alone(tmp_path):
    result = _result(tmp_path)

    def ns(month):
        return int(pd.Timestamp(f'2024-0{month}-01').value)

    manifest = tmp_path / 'input.json'
    _write_json(
        manifest,
        dict(split=dict(
            stage_counts=dict(train=400, val=40, test=30),
            stage_seed_times=dict(train=[ns(1), ns(2)], val=[ns(
                3), ns(3)], test=[2**63, 2**63]))))
    result.artifact_paths['input_manifest'] = str(manifest)
    result = replace(result, explanations=TrainingExplanations.unavailable())
    card = _cards(result)['training_table']
    assert card['columns'] == ['Rows', 'Anchor times']
    assert [r['texts']
            for r in card['rows']] == [['400', '2024-01-01 to 2024-02-01'],
                                       ['40', '2024-03-01'], ['30', '']]


@pytest.mark.parametrize('truncated', [False, True])
@pytest.mark.parametrize('full_refit', [None, 'provenance', 'config'])
def test_bundle_alone_keeps_published_holdout(tmp_path, truncated, full_refit):
    result = _result(tmp_path)
    bundle = result.bundle_dir
    if full_refit == 'provenance':
        from kumo.engine._manifest import ModelManifest, stamp, write_manifest
        write_manifest(bundle / '_provenance.json',
                       ModelManifest(id='m', fit_mode='refit_full', **stamp()))
        add_bundle_file(bundle, 'provenance', '_provenance.json')
    elif full_refit == 'config':
        _write_json(bundle / 'model' / 'config.json',
                    dict(refit_cfg=dict(refitted_full=True)))
    result = replace(result, holdout_truncated=truncated,
                     explanations=TrainingExplanations.unavailable())
    card = _card(_metrics.card, result)
    view = card['views'][0]
    assert view['pills'][0]['holdout_text'] == '1.50'
    # Baselines need only the holdout.
    assert any(row['baseline'] for row in view['table']['rows'])
    assert [c['label'] for c in view['table']['columns']] == ['Holdout']
    assert _columns.importance(_run.load(result)) is None
    # Holdout scores come from the whole test split, not the retained copy;
    # the baselines come from the retained copy.
    assert card['coverage'] is None
    assert (view['coverage'] == 'Baselines: based on a truncated '
            'retained holdout; full coverage is unavailable.'
            if truncated else view['coverage'] is None)
    visual = result.visualize()
    assert ('truncated retained holdout' in visual._repr_svg_()) == truncated
    assert ('note' in visual.data['header']) == bool(full_refit)
    assert ('before full refit' in visual._repr_svg_()) == bool(full_refit)


MALFORMED = {
    'split-stats-nan': dict(test_table_stats=dict(num_train_rows=float('nan'))),
    'split-stats-inf': dict(test_table_stats=dict(num_train_rows=float('inf'))),
    'holdout-nan': float('nan'),
    'holdout-inf': float('inf'),
    'holdout-range': 1.1,
    'holdout-text': 'invalid',
    'split-stats-array': dict(test_table_stats=[1, 2]),
    'split-stats-string': dict(test_table_stats='invalid'),
    'runtime-stages': dict(stages=[1]),
    'runtime-test': dict(stages=dict(test=[1])),
    'runtime-graph': dict(graph=[1]),
    'runtime-path': dict(stages=dict(test=dict(index_path=[1]))),
    'scores-nonfinite': 'table,column,score\nusers,age,inf\n',
    'scores-duplicate': 'table,column,score\nusers,age,0.3\nusers,age,0.2\n',
}


@pytest.mark.parametrize('missing',
                         ['holdout', 'explain_test', 'split_stats', *MALFORMED])
def test_missing_optional_files_omit_views(tmp_path, missing):
    result = (_binary(tmp_path, [.2, .8], [False, True])
              if missing.startswith('holdout-') else _result(tmp_path))
    trial = Path(result.explanations.trial_root)
    stats = tmp_path / 'snap/train_tables/tt-run/split_stats.json'
    scores = trial / 'explain_test' / 'scores.csv'
    target = dict(
        holdout=result.bundle_dir / 'holdout.parquet', explain_test=scores,
        split_stats=stats, scores=scores,
        runtime=Path(result.explanations.run_root) /
        'task_dataset/runtime_manifest.json')
    target['split-stats'] = stats
    if missing not in MALFORMED:
        target[missing].unlink()
    elif missing.startswith('holdout-'):
        frame = pd.read_parquet(target['holdout'])
        frame['True_PROB'] = MALFORMED[missing]
        frame.to_parquet(target['holdout'])
    elif missing.startswith('scores-'):
        scores.write_text(MALFORMED[missing])
    else:
        _write_json(target[missing.rsplit('-', 1)[0]], MALFORMED[missing])
    explained = missing in ('explain_test', 'scores-nonfinite',
                            'scores-duplicate')
    if explained:
        _write_json(
            trial / 'column_analysis' / 'test' / 'column_analysis.json',
            dict(
                task_type='regression', cohorts=[
                    dict(stype=None, target_score=0.0, table_name='users',
                         col_name=column, hop=0, cohorts=['a', 'b'],
                         populations=[0.5, 0.5], predictions=[1, 2],
                         targets=[1.5, 1.5], prediction_score=1.0)
                    for column in ['age', 'region']
                ]))
    run = _run.load(result)
    card = _card(_metrics.card, result)
    if missing.startswith('holdout'):
        assert run.holdout is None
        assert not any(r['baseline'] for r in card['views'][0]['table']['rows'])
    if explained:
        assert _columns.importance(run) is None
        groups = _card(_columns.groups, result)
        assert groups['selectable'] and len(groups['columns']) == 2
    if missing.startswith(('split', 'runtime-')):
        assert run.covered(1) is None
    view = result.visualize()
    assert view.data['tabs'] and 'Training diagnostics' in view._repr_svg_()
    assert '<!doctype html>' in view.page()
    if missing.startswith('holdout-'):
        assert not any(c['id'] in ('cutoff', 'scores')
                       for tab in view.data['tabs'] for c in tab['cards'])


def test_unsupported_task_type(tmp_path):
    with pytest.raises(_run.UnsupportedModelError):
        _run.load(_result(tmp_path, 'unknown'))


@pytest.mark.parametrize(
    'task', [*_run.CLASS_LABELED, 'regression', 'binary_classification'])
def test_labeled_classification_cohorts(tmp_path, task):
    result = _result(
        tmp_path, task, test_rows=3, holdout=pd.DataFrame(
            dict(ENTITY=range(3), TARGET=[True, False, True], TARGET_PRED=1.0,
                 True_PROB=0.5, False_PROB=0.5)))
    assert _run.load(result).holdout is not None
    path = (Path(result.explanations.trial_root) /
            'column_analysis/test/column_analysis.json')
    entry = dict(table_name='users', hop=0, col_name='age', stype=None,
                 populations=[0.25, 0.75], cohorts=['younger', 'older'],
                 predictions=dict(Blue=[0.1, 0.2], Red=[0.9, 0.8]),
                 targets=dict(Red=[1, 0],
                              Blue=[0,
                                    1]), prediction_score=0.3, target_score=0.4)
    if task not in _run.CLASS_LABELED:
        entry.update(predictions=[0.1, 0.2], targets=[0, 1])
    _write_json(path, dict(task_type=task, cohorts=[entry]))
    groups = _card(_columns.groups, result)
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
    assert len(_card(_columns.importance, result)['views'][1]['bars']) == 1
    _write_json(path, dict(task_type=task, cohorts=[entry]))
    assert _columns.groups(_run.load(result)) is None
    assert len(_card(_columns.importance, result)['views']) == 1


def _model_type_result(tmp_path: Path, task: str) -> TrainingResult:
    """A run of one task type with every input any card reads."""
    rng = np.random.default_rng(0)
    n = 40
    holdout = pd.DataFrame(
        dict(ENTITY=np.arange(n), TARGET=rng.random(n) < 0.5,
             True_PROB=rng.random(n), TIMESTAMP=pd.Timestamp('2024-01-08'),
             ANCHOR_TIMESTAMP=pd.Timestamp('2024-01-01')))
    holdout = holdout.assign(False_PROB=1 - holdout['True_PROB'],
                             TARGET_PRED=holdout['True_PROB'] * 3)
    if task in ('regression', 'forecasting'):
        holdout['TARGET'] = rng.random(n) * 3
    result = _result(tmp_path, task, test_rows=n, holdout=holdout)
    trial = Path(result.explanations.trial_root)
    graph = to_json_dict(_graph())
    graph['subgraph_store']['task_type'] = task
    _write_json(trial / 'subgraph_collector/test/subgraph_json/0.json', graph)
    labeled = task in _run.CLASS_LABELED
    _write_json(
        trial / 'column_analysis/test/column_analysis.json',
        dict(
            task_type=task, cohorts=[
                dict(stype=None, table_name='users', col_name='age', hop=0,
                     cohorts=['a', 'b'], populations=[0.5, 0.5],
                     predictions=dict(x=[0.2, 0.4]) if labeled else [0.2, 0.4],
                     targets=dict(x=[0.3, 0.3]) if labeled else [0.3, 0.3],
                     prediction_score=1.0, target_score=None)
            ]))
    return result


@pytest.mark.parametrize('task, cards', [
    ('binary_classification', [['metrics', 'cutoff', 'scores'],
                               ['training_table'],
                               ['subgraphs', 'importance', 'groups']]),
    ('multiclass_classification', [['metrics'], ['training_table'],
                                   ['subgraphs', 'importance', 'groups']]),
    ('multilabel_classification', [['metrics'], ['training_table'],
                                   ['subgraphs', 'importance', 'groups']]),
    ('regression', [['metrics', 'predicted'], ['training_table'],
                    ['subgraphs', 'importance', 'groups']]),
    ('forecasting', [['metrics', 'forecast'], ['training_table'],
                     ['importance']]),
    ('temporal_link_prediction', [['metrics'], ['training_table'],
                                  ['subgraphs', 'importance']]),
    ('multilabel_ranking', [['metrics'], ['training_table'],
                            ['subgraphs', 'importance']]),
])
def test_cards_per_model_type(tmp_path, task, cards):
    result = _model_type_result(tmp_path, task)

    def listed():
        data = result.visualize().data
        json.dumps(data, allow_nan=False)  # the payload the front end reads
        return [[card['id'] for card in tab['cards']] for tab in data['tabs']]

    assert listed() == cards
    (result.bundle_dir / 'holdout.parquet').unlink()
    assert listed()[2] == cards[2]


def _ranking_scores(refit: bool = False) -> dict[str, float]:
    scores = {'test_loss': 5.0}
    for family, base in (('recall', 0.2), ('precision', 0.7), ('map', 0.5),
                         ('f1', 0.3), ('ndcg', 0.4)):
        for i, k in enumerate((1, 10, 5)):
            for split, shift in (('test', 0.0), ('val', 0.01), ('train', 0.02)):
                scores[f'{split}_{family}@{k}'] = base + 0.1 * i + shift
                if refit and split == 'test':
                    scores[f'test_refit_{family}@{k}'] = base + 0.1 * i + 0.05
    return scores


@pytest.mark.parametrize('task', [
    'multilabel_ranking', 'static_link_prediction', 'temporal_link_prediction'
])
@pytest.mark.parametrize('refit', [False, True])
def test_ranking_card_reports_metrics_by_k(tmp_path, task, refit):
    scores = _ranking_scores(refit)
    result = _result(tmp_path, task, scores=scores, tune='f1@5', refit=refit)
    card = _card(_ranking.card, result)
    assert [view['id'] for view in card['views']
            ] == ['f1', 'recall', 'precision', 'map', 'ndcg']
    holdout = 'test_refit_' if refit else 'test_'
    for view in card['views']:
        assert view['ks'] == [1, 5, 10]
        assert [s['id'] for s in view['series']] == ['holdout', 'val', 'train']
        for series, prefix in zip(view['series'], (holdout, 'val_', 'train_')):
            expected = [scores[f'{prefix}{view["id"]}@{k}'] for k in view['ks']]
            assert series['values'] == expected
            assert series['texts'] == [fmt.number(v) for v in expected]
    assert card['views'][1]['label'] == 'Recall'
    assert json.dumps(card, allow_nan=False)


def test_ranking_card_omits_missing_splits_and_single_cutoffs(tmp_path):
    scores = {
        'test_recall@5': 0.5,
        'test_recall@10': 0.7,
        'test_map@5': 0.4,
        'val_recall@5': 0.4
    }
    card = _card(
        _ranking.card,
        _result(tmp_path, 'multilabel_ranking', scores=scores, tune='recall@5',
                refit=False))
    assert [view['id'] for view in card['views']] == ['recall']
    recall = card['views'][0]
    assert [s['id'] for s in recall['series']] == ['holdout', 'val']
    assert recall['series'][1]['values'] == [0.4, None]
    assert recall['series'][1]['texts'] == ['0.400', None]
    only_one = {'test_recall@5': 0.5, 'test_precision@5': 0.4}
    assert _ranking.card(
        _run.load(
            _result(tmp_path / 'b', 'multilabel_ranking', scores=only_one,
                    tune='recall@5'))) is None


def test_ranking_card_applies_to_ranking_tasks_only(tmp_path):
    result = _result(tmp_path, 'regression', scores=_ranking_scores())
    assert _ranking.card(_run.load(result)) is None


@pytest.mark.parametrize(
    'cutoffs',
    [(1, 5, 10), tuple(range(1, 11)),
     tuple(range(1, 51))])
def test_ranking_snapshot_lists_the_tuned_metric_by_k(tmp_path, cutoffs):
    scores = _ranking_scores()
    scores.update({
        f'{split}_recall@{k}': k / 100
        for split in ('test', 'val', 'train')
        for k in cutoffs
    })
    result = _result(tmp_path, 'multilabel_ranking', scores=scores,
                     tune='recall@5', refit=False)
    view = result.visualize()
    assert [card['id'] for card in view.data['tabs'][0]['cards']
            ] == ['metrics', 'ranking']
    image = ElementTree.fromstring(view._repr_svg_())
    text = ' '.join(image.itertext())
    assert 'Multilabel Ranking evaluation' in text
    assert 'Recall by K' in text
    labels = image.findall('.//{http://www.w3.org/2000/svg}text')
    assert {f'K={k}' for k in cutoffs} <= {label.text for label in labels}
    assert all(0 <= float(label.attrib['x']) <= 448 for label in labels)


def test_ranking_preserves_zero_padded_metric_keys(tmp_path):
    result = _result(tmp_path, 'multilabel_ranking', scores={
        'test_recall@05': 0.4,
        'test_recall@10': 0.6
    }, tune='recall@05', refit=False)
    card = _ranking.card(_run.load(result))
    view = card['views'][0]
    assert view['ks'] == [5, 10]
    assert view['series'][0]['values'] == [0.4, 0.6]


def test_tenths_only_include_observed_groups():
    view = _run.tenths(np.arange(3), np.array([0.1, 0.5, 0.9]),
                       np.array([0, 1, 1]), 'observed', 'Overall positive rate',
                       percentage=True)
    assert [r['label'] for r in view['tenths']] == ['1', '4', '7']
    assert view['callout'] == 'Group 4 of 10: predicted 50.0% · observed 100.0%'
    assert view['reference_callout'] == 'Overall positive rate: 66.7%'


def test_equal_count_groups_break_ties_by_entity():
    groups = _run.equal_count_groups(np.array([1.0, 1.0, 1.0, 2.0, 0.5]),
                                     np.array([5, 3, 4, 1, 2]), 5)
    # Sorted order: 0.5 (entity 2), 1.0 by entity 3, 4, 5, then 2.0.
    assert list(groups) == [3, 1, 2, 4, 0]


@pytest.mark.parametrize('days, word', [(1, 'day'), (7, 'week'), (31, 'month'),
                                        (3, 'step')])
def test_forecast_period_word(days, word):
    steps = pd.Series(pd.date_range('2024-01-01', periods=4, freq=f'{days}D'))
    assert _forecasting.period(steps) == word


def test_cutoff_card_counts_scores_above_the_cutoff(tmp_path):
    card = _card(
        _classification.cutoff_card,
        _binary(tmp_path, [0.9, 0.8, 0.6, 0.5, 0.3, 0.1],
                [True, True, False, True, False, False]))
    true, false = card['classes']
    # 0.5 is not flagged at cutoff 0.50: TP 2, FP 1, FN 1, TN 2.
    assert true['counts'][50] == [2, 1, 1, 2]
    # The UI swaps the original confusion counts, including threshold ties.
    assert false['counts'][50] == [2, 1, 1, 2]
    assert true['counts'][100] == [0, 0, 3, 3]


@pytest.mark.parametrize('outcome', [True, False])
def test_cutoff_card_retains_single_outcome_diagnostics(tmp_path, outcome):
    result = _binary(tmp_path, [0.9, 0.8, 0.7], [outcome] * 3)
    cutoff = _card(_classification.cutoff_card, result)
    assert cutoff['classes'][0]['counts'][50] == ([3, 0, 0, 0]
                                                  if outcome else [0, 3, 0, 0])
    json.dumps([cutoff, _card(_classification.scores_card, result)],
               allow_nan=False)


@pytest.mark.parametrize(
    'distribution',
    [None, ['invalid'], dict(histogram=['invalid'])])
def test_regression_without_bin_width_omits_binned_views(
        tmp_path, distribution):
    result = _result(tmp_path)
    stats = tmp_path / 'snap/train_tables/tt-run/split_stats.json'
    value = json.loads(stats.read_text())
    value['test_table_stats']['target_distribution'] = distribution
    _write_json(stats, value)
    card = _card(_regression.predicted_card, result)
    assert [v['id'] for v in card['views']] == ['tenths', 'bands']
    assert 'Predicted against actual' in result.visualize()._repr_svg_()


@pytest.mark.parametrize('case', ['near', 'distant', 'folded', 'even'])
def test_regression_distributions_keep_offset_and_sparse_bins(tmp_path, case):
    actual = np.array([10000.0] * 15 + [10001.0] * 15)
    pred = np.full(30, 10001.0 if case == 'near' else 20000.0)
    if case == 'folded':
        pred[-1] = 30000.0
    if case == 'even':
        # No bin holds more than ten rows, so none folds.
        actual = pred = np.arange(40) + 0.5
    view = _regression._distributions(_run.load(_result(tmp_path)), actual,
                                      pred, 1.0 if case == 'even' else 0.02)
    assert [sum(s['counts']) for s in view['series']] == [len(actual)] * 2
    folded = case == 'folded'
    assert 'bins' not in view and view['folded'] == folded
    if case == 'even':
        assert view['start'] == 0.0 and view['bin_count'] == 40
    elif case == 'near':
        assert view['start'] == 10000.0 and view['bin_count'] == 51
        assert 'indices' not in view
    else:
        assert view['bin_count'] == 500001 + folded
        assert view['indices'] == [0, 50, 500000] + ([500001] if folded else [])
        if folded:
            assert view['series'][1]['counts'][-1] == 1


@pytest.mark.parametrize('saved', [
    'valid', 'malformed', 'absent', 'unbounded', 'inverted', 'unpaired',
    'merged', 'edge', 'separated', 'constant'
])
def test_regression_heatmap_prefers_saved_artifact(tmp_path, saved):
    result = _result(tmp_path, test_rows=20)
    width, (x0, x1), (y0, y1), cells, counts = {
        'valid': (1.0, (-1, 0), (-1, 0), [[-1, -1], [0, 0]], [11, 1]),
        'merged': (1.0, (0, 200), (0, 200), [[0, 0], [1, 0]], [12, 7]),
        'edge': (1.0, (0, 110), (0, 366), [[0, 0], [110, 0]], [12, 7]),
        'separated': (0.02, (0, 1), (1e4, 10001), [[0, 1e4], [1,
                                                              10001]], [12, 7]),
        'constant': (0.02, (0, 0), (0, 1e6), [[0, 0], [0, 1e6]], [12, 7]),
        'unbounded': (1.0, (-1e308, 1e308), (0, 0), [[0, 0]], [5]),
        'inverted': (1.0, (5, 0), (5, 0), [[5, 5]], [5]),
        'unpaired': (1.0, (0, 1), (0, 1), [[0, 0]], [5, 5]),
    }.get(saved, (1.0, (0, 0), (0, 0), [], []))
    if saved != 'absent':
        _write_json(
            Path(result.explanations.trial_root) /
            'test_target_prediction_heatmap' / 'heatmap.json',
            dict(cell_width=width, label_bin_start=x0, label_bin_end=x1,
                 prediction_bin_start=y0, prediction_bin_end=y1, cells=cells,
                 counts=counts,
                 **({} if saved == 'malformed' else dict(home_dir=''))))
    heatmap = _regression._heatmap(_run.load(result), 1.0)
    assert heatmap is not None
    json.dumps(heatmap, allow_nan=False)
    if saved in ('absent', 'malformed', 'unbounded', 'inverted', 'unpaired'):
        assert sum(map(sum, heatmap['counts'])) == 20
        assert heatmap['coverage'] == 'Based on 20 of 20 retained holdout rows.'
    elif saved == 'valid':
        assert heatmap['counts'] == [[11, 0], [0, 1]]
        assert heatmap['trimmed']['counts'] == [[11]]
        assert heatmap['coverage'] is None
        (result.bundle_dir / 'holdout.parquet').unlink()
        card = _card(_regression.predicted_card, result)
        assert [v['id'] for v in card['views']] == ['heatmap']
        image = ElementTree.fromstring(result.visualize()._repr_svg_())
        assert len(image.findall('.//{*}rect[@opacity]')) == 2
        labels = list(image.itertext())
        assert 'actual value' in labels and 'predicted value' in labels
        assert any('Densest cell' in label for label in labels)
    else:
        # Sized before allocation, with both axes on one scale.
        assert len(heatmap['rows']) * len(heatmap['columns']) <= 10_000
        assert heatmap['x_edges'][0] == heatmap['y_edges'][0]
        assert sum(map(sum, heatmap['counts'])) == 19


def test_regression_bands_without_a_zero_band(tmp_path):
    card = _card(
        _regression.predicted_card,
        _result(
            tmp_path, test_rows=20, holdout=pd.DataFrame(
                dict(ENTITY=range(20), TARGET=np.arange(20) + 1.0,
                     TARGET_PRED=10.0))))
    assert [b['label'] for b in card['views'][1]['bands']] == [
        '1.00 to 4.00', '5.00 to 8.00', '9.0 to 12.0', '13.0 to 16.0',
        '17.0 to 20.0'
    ]


def test_render_html_is_self_contained(tmp_path):
    view = _result(tmp_path).visualize()
    html = view.html()
    assert view.model_type == 'regression'
    assert html.startswith('<div id="kumo-')
    assert html.count('</script>') == 2  # the data and the code, nothing else
    assert view._repr_html_().startswith('<div id="kumo-')
    note = 'To activate the widget, rerun this cell.'
    image = html.split('src="data:image/svg+xml;base64,', 1)[1].split('"', 1)[0]
    labels = list(ElementTree.fromstring(base64.b64decode(image)).itertext())
    assert 'Metrics' in labels and 'Holdout' in labels
    assert 'Metric' in labels and 'MAE' in labels
    assert 'Predicted against actual' in labels
    assert 'Training Table' in labels
    assert 'Column importance' not in labels
    assert f'<p>{note}</p>' in html
    assert note in list(ElementTree.fromstring(view._repr_svg_()).itertext())


@pytest.mark.skipif(not shutil.which('node'), reason='needs node')
@pytest.mark.parametrize('task', [
    'binary_classification', 'regression', 'forecasting',
    'temporal_link_prediction', 'multilabel_ranking'
])
def test_rendered_script_parses_for_each_model_type(tmp_path, task):
    html = _model_type_result(tmp_path, task).visualize().html()
    # Validate the transported source, including script-tag escaping.
    source = html.split('<script>//<![CDATA[', 1)[1].split('</script>', 1)[0]
    subprocess.run(['node', '--check'], input=source, text=True,
                   capture_output=True, check=True)


def test_anywidget_keeps_static_output_with_widget_model(tmp_path):
    pytest.importorskip('anywidget')
    view = _result(tmp_path, test_rows=3).visualize()
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


@pytest.mark.parametrize('source',
                         ['bundle', 'original', 'stale', 'unlisted', 'remote'])
def test_training_result_visualize_opens_its_run(tmp_path, source):
    result = _result(tmp_path)
    holdout = result.bundle_dir / 'holdout.parquet'
    if source == 'original':
        holdout = holdout.rename(tmp_path / 'holdout.parquet')
    elif source in ('stale', 'remote'):
        holdout = tmp_path / 'removed.parquet'
    elif source == 'unlisted':
        _write_json(result.bundle_dir / 'publication-manifest.json',
                    {'artifacts': []})
        holdout = None
    explanations = result.explanations
    if source == 'remote':
        # A retained run on object storage beside a local bundle.
        remote = f'memory://{tmp_path}/run'
        filesystem.copy(explanations.run_root, remote)
        for local in ('models', 'task_dataset'):
            shutil.rmtree(Path(explanations.run_root) / local)
        explanations = TrainingExplanations.load(remote)
        history = {
            '0': dict(metrics={'1': dict(train_metrics=dict(train_mae=1))})
        }
        _write_json(tmp_path / 'history.json', dict(trial_progress=history))
        result.artifact_paths['metric_history'] = str(tmp_path / 'history.json')
    data = replace(result, holdout_path=holdout, holdout_truncated=True,
                   explanations=explanations).visualize().data
    cards = {c['id']: c for tab in data['tabs'] for c in tab['cards']}
    assert 'importance' in cards
    if source == 'unlisted':
        assert 'predicted' not in cards
    else:
        assert 'truncated' in cards['predicted']['coverage']
    assert 'holdout' not in data['header']
    if source == 'remote':
        assert [e['label']
                for e in cards['history']['experiments']] == ['Experiment 1']


@pytest.mark.parametrize('remote', [False, True])
def test_open_run_reads_a_retained_run_with_its_bundle(tmp_path, remote):
    result = _result(tmp_path)
    bundle = result.bundle_dir
    history = {'0': dict(metrics={'1': dict(train_metrics=dict(train_mae=1))})}
    _write_json(bundle / 'metric-history.json', dict(trial_progress=history))
    _write_json(bundle / 'metrics.json', result.metrics)
    for role, path in result.artifact_paths.items():
        add_bundle_file(bundle, role, str(Path(path).relative_to(bundle)))
    for role, name in (('metric_history', 'metric-history.json'),
                       ('metrics', 'metrics.json')):
        add_bundle_file(bundle, role, name)
    run = result.explanations.run_root
    if remote:
        # The whole retained run, bundle included, on object storage.
        run = f'memory://{tmp_path}/run'
        filesystem.copy(result.explanations.run_root, run)
        shutil.rmtree(result.explanations.run_root)
    data = _explainability.open_run(run).data
    cards = {c['id']: c for tab in data['tabs'] for c in tab['cards']}
    assert 'holdout' not in data['header']
    assert {'predicted', 'importance', 'history'} <= set(cards)


@pytest.mark.parametrize('failure', [None, 'missing', 'dangling', 'malformed'])
def test_subgraphs_use_typed_artifacts_and_keep_missing_distinct(
        tmp_path, failure):
    result = _result(tmp_path, test_rows=3)
    root = Path(result.explanations.trial_root) / 'subgraph_collector' / 'test'
    _write_json(
        root / 'strategy_mappings.json',
        dict(
            mappings=dict(random=['0', '1'], unsure=['1', '0']),
            orig_mappings=dict(random=['account-x', 'account-y'],
                               unsure=['account-y', 'account-x'])))
    subgraph = _graph()
    subgraph.subgraph_store.nodes[0].pkey = 2**53 + 1
    subgraph.feature_store.features[0]['note'] = Feature(
        'users', 'note', ['</script>', {}], Stype.text)
    graph = to_json_dict(subgraph)
    graph['feature_store']['features']['0']['note']['value'][1] = dict(
        missing=float('nan'))
    _write_json(root / 'subgraph_json' / '0.json', graph)
    _write_json(root / 'output' / '0.json',
                dict(prediction=2.5, target=float('nan')))
    if failure == 'dangling':
        graph['subgraph_store']['edges'] = {'0': {'related': [99]}}
    if failure != 'missing':
        _write_json(root / 'subgraph_json' / '1.json', graph)
    if failure == 'malformed':
        (root / 'subgraph_json' / '1.json').write_text('not json')
    card = _card(_subgraphs.card, result)
    assert len(card['samples']) == (2 if failure is None else 1)
    assert card['unavailable_count'] == (0 if failure is None else 1)
    # Each category keeps the collector's selection order.
    assert card['category_samples'] == dict(random=['0', '1'],
                                            unsure=['1', '0'])
    sample = card['samples'][0]
    assert sample['categories'] == ['random', 'unsure']
    assert sample['original_ids'] == dict(random='account-x',
                                          unsure='account-x')
    assert sample['prediction'] == 2.5 and sample['target'] is None
    assert sample['nodes'][0]['value'] == '9007199254740993'
    assert [f['score'] for f in sample['nodes'][0]['features']] == [0.0, None]
    assert sample['nodes'][0]['features'][1]['value'][1] == {'missing': None}
    json.dumps(card, allow_nan=False)
    model_only = replace(result,
                         explanations=TrainingExplanations.unavailable())
    assert _subgraphs.card(_run.load(model_only)) is None


def _payload(html: str) -> Any:
    return json.loads(
        html.split('class="kumo-widget-data">', 1)[1].split('</script>', 1)[0])


def test_page_is_a_standalone_document(tmp_path):
    view = _result(tmp_path).visualize()
    path = view.save(tmp_path / 'report.html', 'a<b training report')

    page = path.read_text(encoding='utf-8')
    head, body = page.split('<body>', 1)
    assert head.startswith('<!doctype html><html lang="en"><head>'
                           '<meta charset="utf-8">')
    assert '<title>a&lt;b training report</title>' in head
    assert body.startswith('<div id="kumo-')
    assert body.endswith('</div></body></html>')
    widget_id = re.compile(r'kumo-[0-9a-f]{32}')
    assert widget_id.sub('', body[:-len('</body></html>')]) == widget_id.sub(
        '', view.html())
    # Nothing is fetched from outside the page.
    assert not re.search(r'<link|@import|(src|href)="(?!data:)|url\((?!#)',
                         page)
