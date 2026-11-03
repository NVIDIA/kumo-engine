# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Widget adapter boundaries, reusing the explanation artifact fixtures."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from kumo_core.io.json import save_json
from kumoapi.explain.common import EntityMappings
from test_explanation_artifacts import (
    _cohort,
    _gradient,
    _graph,
    _run,
    _save_cohorts,
    _text,
)

from kumo.engine.explain._diagnostics import LIMITS, build_training_payload
from kumo.engine.explain.results import TrainingExplanations


def test_snapshot_preserves_refit_metrics_epochs_and_missing_values(tmp_path):
    run, trial = _run(tmp_path, refit=True)
    selected = Path(trial.removesuffix('_refit_trainval'))
    _gradient(trial)
    _save_cohorts(trial, [_cohort(classification=True)], classification=True)
    save_json({
        'val_mae': None,
        'test_mae': 0,
        'test_refit_mae': 2
    }, str(run / 'published-model/metrics.json'))
    save_json({'predictive_query': 'PREDICT users.label FOR EACH users.id'},
              str(run / 'published-model/training-spec.json'))
    progress = dict(
        num_epochs_completed=2, metrics={
            '0': dict(train_metrics={'train_mae': 1}, validation_metrics={}),
            '7': dict(train_metrics={}, validation_metrics={'val_mae': None})
        })
    save_json(progress, str(selected / 'train_progress.json'))
    save_json(
        dict(num_epochs_completed=1,
             metrics={'4': dict(train_metrics={'train_mae': 0})}),
        str(Path(trial) / 'train_progress.json'))
    save_json(
        dict(total_trials=1, completed_trials=0, trial_progress={'2':
                                                                 progress}),
        str(selected.parent / 'auto_train_progress.json'))
    result = build_training_payload(TrainingExplanations.load(run))
    assert result['provenance']['selected_trial'] == 'version_2'
    assert result['provenance'][
        'explanation_trial'] == 'version_2_refit_trainval'
    assert result['provenance']['selection_verified']
    assert result['context']['run'] == 'run'
    assert result['context']['query'].startswith('PREDICT users.label')
    assert [row['epoch']
            for row in result['history']['rows']] == ['0', '7', '4']
    assert result['history']['rows'][1]['validation_metrics'] == {
        'val_mae': None
    }
    assert result['history']['rows'][2]['fit'] == 'refit_trainval'
    assert result['metrics']['rows'] == [
        dict(name='mae', recorded_name='val_mae', value=None, split='val',
             fit='original'),
        dict(name='mae', recorded_name='test_mae', value=0, split='test',
             fit='original'),
        dict(name='mae', recorded_name='test_refit_mae', value=2, split='test',
             fit='refit_trainval')
    ]
    assert result['trials']['completed'] == 0
    assert result['trials']['rows'][0]['selected']
    assert result['trials']['rows'][0][
        'comparison_source'] == 'last_recorded_epoch'
    assert result['trials']['rows'][0]['comparison_epoch'] == '7'
    assert result['trials']['rows'][0]['comparison_metrics'] == {
        'val_mae': None
    }
    assert result['cohorts']['rows'][0]['predictions'] == {
        'no': [0.8, 0.4],
        'yes': [0.2, 0.6]
    }
    assert result['gradient']['reason'] is None
    assert str(tmp_path) not in json.dumps(result)
    json.dumps(result, allow_nan=False)


def test_sample_coverage_deduplication_and_complete_graph_budget(tmp_path):
    run, trial = _run(tmp_path)
    root = f'{trial}/subgraph_collector/test'
    for artifact in ('0', '2', '3'):
        save_json(_graph(), f'{root}/subgraph_json/{artifact}.json')
    save_json(
        EntityMappings(
            mappings={
                'random': ['0', '1'],
                'large_error': ['0']
            }, orig_mappings={
                'random': ['user-7', 'missing'],
                'large_error': ['user-7']
            }), f'{root}/strategy_mappings.json')
    result = build_training_payload(TrainingExplanations.load(run),
                                    max_samples=1)
    section = result['samples']
    assert section['coverage'] == dict(references_total=4, references_listed=4,
                                       available=3, loaded=1, missing=1,
                                       malformed=0, excluded=2,
                                       catalog_excluded=0)
    assert section['references'][0]['categories'] == ['random', 'large_error']
    assert section['references'][0]['original_ids']['random'] == 'user-7'
    assert section['references'][1]['status'] == 'missing'
    assert section['references'][2]['status'] == 'limited'
    graph = section['bodies']['0']
    assert graph['subgraph_store']['edges']['0']['events'] == [-1]
    assert graph['subgraph_store']['nodes']['-1']['count'] == 12
    assert graph['feature_store']['features']['0']['age']['score'] == 0


@pytest.mark.parametrize('kind', ['gradient', 'sample'])
def test_oversized_artifacts_are_not_opened_by_facade(tmp_path, monkeypatch,
                                                      kind):
    run, trial = _run(tmp_path)
    suffix = ('explain_test/scores.csv' if kind == 'gradient' else
              'subgraph_collector/test/subgraph_json/0.json')
    _text(Path(trial) / suffix, ' ' * (LIMITS['artifact_bytes'] + 1))
    method = 'gradient_scores' if kind == 'gradient' else 'sample'
    forbidden = Mock(side_effect=AssertionError('Oversized artifact was read'))
    monkeypatch.setattr(TrainingExplanations, method, forbidden)
    result = build_training_payload(TrainingExplanations.load(run))
    section = result['gradient' if kind == 'gradient' else 'samples']
    assert section['status'] == 'limited'
    forbidden.assert_not_called()


@pytest.mark.parametrize('kind',
                         ['trial', 'gradient', 'sample', 'mapping', 'loop'])
def test_run_containment_rejects_symlink_escape(tmp_path, kind):
    run, trial = _run(tmp_path)
    external = tmp_path / 'outside'
    external.mkdir()
    (external / 'secret').write_text('private-value')
    suffix = dict(
        trial='', gradient='explain_test/scores.csv',
        loop='explain_test/scores.csv',
        sample='subgraph_collector/test/subgraph_json/0.json',
        mapping='subgraph_collector/test/strategy_mappings.json')[kind]
    path = Path(trial) / suffix
    if kind == 'trial':
        path = run / 'models/task/automl/version_3'
        path.symlink_to(external, target_is_directory=True)
        facade = TrainingExplanations.from_trial(run, path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(path.name if kind == 'loop' else external / 'secret')
        facade = TrainingExplanations.load(run)
    result = build_training_payload(facade)
    section = result['gradient' if kind in ('trial', 'gradient',
                                            'loop') else 'samples']
    assert section['status'] == 'invalid'
    assert 'private-value' not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize('kind', ['directory', 'gradient', 'cohorts'])
def test_count_limits_precede_facade_reads(tmp_path, monkeypatch, kind):
    run, trial = _run(tmp_path)
    if kind == 'directory':
        for key in ('0', '1'):
            save_json(
                _graph(),
                f'{trial}/subgraph_collector/test/subgraph_json/{key}.json')
        bound, method, section = 'directory_entries', 'samples', 'samples'
    elif kind == 'gradient':
        _gradient(trial)
        bound, method, section = 'gradient_rows', 'gradient_scores', 'gradient'
    else:
        _save_cohorts(trial, [_cohort(), _cohort()])
        bound, method, section = 'cohorts', 'column_analysis', 'cohorts'
    monkeypatch.setitem(LIMITS, bound, 1)
    forbidden = Mock(side_effect=AssertionError('Count limit was bypassed'))
    monkeypatch.setattr(TrainingExplanations, method, forbidden)
    result = build_training_payload(TrainingExplanations.load(run))
    assert result[section]['status'] == 'limited'
    forbidden.assert_not_called()


@pytest.mark.parametrize('body',
                         ['{', '{"cohorts": null}', '{"cohorts": [NaN]}'])
def test_bad_section_preserves_valid_gradient(tmp_path, body):
    run, trial = _run(tmp_path)
    _gradient(trial)
    _text(Path(trial) / 'column_analysis/test/column_analysis.json', body)
    result = build_training_payload(TrainingExplanations.load(run))
    assert result['gradient']['status'] == 'available'
    assert result['cohorts']['status'] == 'invalid'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('path,body,section', [
    ('models/task/automl/version_2/explain_test/scores.csv',
     'table,column,score\nusers,' + 'x' * 140000 + ',1\n', 'gradient'),
    ('published-model/metrics.json', json.dumps({'test_mae': 10**400
                                                 }), 'metrics'),
    ('published-model/metrics.json', json.dumps({'test_mae': 2**53
                                                 }), 'metrics'),
])
def test_malformed_numeric_and_csv_inputs_preserve_display(
        tmp_path, path, body, section):
    run, _ = _run(tmp_path)
    _text(run / path, body)
    view = TrainingExplanations.load(run).visualize()
    assert view._payload[section]['status'] == 'invalid'
    assert 'Training results' in view._repr_html_()


def test_unverified_selection_does_not_identify_a_winner(tmp_path):
    run, trial = _run(tmp_path)
    automl = Path(trial).parent
    save_json({'best_trainer': str(automl / 'version_8')},
              str(automl / 'best_trainer.json'))
    save_json({
        'trial_progress': {
            '2': {}
        },
        'total_trials': 1
    }, str(automl / 'auto_train_progress.json'))
    view = TrainingExplanations.from_trial(run, trial).visualize()
    assert view._payload['provenance']['selection_verified'] is False
    assert view._payload['provenance']['selected_trial'] is None
    assert view._payload['provenance']['refitted_trainval'] is None
    assert view._payload['trials']['rows'][0]['selected'] is None
    assert 'Selection unverified:' in view._repr_html_()


def test_nonfinite_or_empty_sample_never_reaches_interactive_body(tmp_path):
    run, trial = _run(tmp_path)
    root = Path(trial) / 'subgraph_collector/test/subgraph_json'
    _text(root / '0.json',
          '{"feature_store":{"features":{"0":{"x":{"value":NaN}}}}}')
    _text(root / '1.json', '{}')
    save_json(_graph(), str(root / '2.json'))
    body = json.loads((root / '2.json').read_text())
    body['subgraph_store']['nodes']['0']['pkey'] = 2**63 - 1
    _text(root / '3.json', json.dumps(body))
    body['subgraph_store']['nodes']['0']['pkey'] = str(2**63 - 1)
    _text(root / '4.json', json.dumps(body))
    result = build_training_payload(TrainingExplanations.load(run))
    assert result['samples']['coverage']['malformed'] == 3
    assert list(result['samples']['bodies']) == ['2', '4']
    assert result['samples']['bodies']['4']['subgraph_store']['nodes']['0'][
        'pkey'] == str(2**63 - 1)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('reason',
                         ['disabled', 'failed', 'model_only', 'unknown'])
def test_unavailable_reason_is_explicit(reason):
    result = build_training_payload(
        TrainingExplanations.unavailable(reason=reason))
    assert result['samples']['status'] == 'unavailable'
    assert result['samples']['reason'] == reason
    assert result['samples']['coverage']['available'] is None


def test_negative_gradient_and_failed_generation_are_explicit(tmp_path):
    run, trial = _run(tmp_path)
    _text(
        Path(trial) / 'explain_test/scores.csv',
        'table,column,score\nusers,age,-0.2\n')
    result = build_training_payload(
        TrainingExplanations.from_trial(run, trial, failure='private/path'))
    assert result['gradient']['status'] == 'invalid'
    (Path(trial) / 'explain_test/scores.csv').unlink()
    _gradient(trial)
    result = build_training_payload(
        TrainingExplanations.from_trial(run, trial, failure='private/path'))
    assert result['gradient']['status'] == 'partial'
    assert result['gradient']['rows']
    assert 'private/path' not in json.dumps(result)


def test_payload_limit_discards_whole_graphs(tmp_path, monkeypatch):
    run, trial = _run(tmp_path)
    save_json(_graph(), f'{trial}/subgraph_collector/test/subgraph_json/0.json')
    original = build_training_payload(TrainingExplanations.load(run))
    bound = len(json.dumps(original, separators=(',', ':')).encode()) - 100
    monkeypatch.setitem(LIMITS, 'payload_bytes', bound)
    result = build_training_payload(TrainingExplanations.load(run))
    assert not result['samples']['bodies']
    assert all(ref['status'] != 'loaded'
               for ref in result['samples']['references'])
    assert len(json.dumps(result, separators=(',', ':')).encode()) <= bound


@pytest.mark.parametrize('max_samples', [-1, 13, True, 1.5])
def test_invalid_preload_request(max_samples):
    with pytest.raises(ValueError, match='max_samples'):
        build_training_payload(TrainingExplanations.unavailable(),
                               max_samples=max_samples)
