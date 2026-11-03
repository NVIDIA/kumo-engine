# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Widget adapter boundaries, reusing the explanation artifact fixtures."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from kumo_core.explain.schemas import EntityOutput
from kumo_core.io import filesystem
from kumo_core.io.json import save_json
from kumoapi.explain.common import EntityMappings
from kumoapi.task import TaskType
from test_explanation_artifacts import (
    _cohort,
    _gradient,
    _graph,
    _run,
    _save_cohorts,
    _text,
)

from kumo.engine.explain import _artifacts
from kumo.engine.explain._diagnostics import LIMITS, build_training_payload
from kumo.engine.explain._viewer import static_summary
from kumo.engine.explain.results import TrainingExplanations


@pytest.mark.parametrize('source', ['training', 'test', 'test_refit'])
@pytest.mark.parametrize(
    'state', ['available', 'missing', 'mismatch', 'invalid', 'boolean'])
def test_snapshot_preserves_refit_metrics_epochs_and_missing_values(
        tmp_path, source, state):
    run, trial = _run(tmp_path, refit=True)
    selected = Path(trial.removesuffix('_refit_trainval'))
    _gradient(trial)
    _save_cohorts(trial, [_cohort(classification=True)], classification=True)
    save_json(
        {
            'train_mae': 1,
            'val_loss': 0.2,
            'val_mae': None,
            'test_mae': 0,
            'test_refit_mae': 2
        }, str(run / 'published-model/metrics.json'))
    artifacts = {
        'training': selected / 'train_result.json',
        'test': selected / 'test_result.csv',
        'test_refit': Path(trial) / 'test_result.csv'
    }
    save_json({'train_mae': 1, 'val_loss': 0.2}, str(artifacts['training']))
    _text(artifacts['test'], 'test_mae\n0.000000\n')
    _text(artifacts['test_refit'], 'test_mae\n2.000000\n')
    if state == 'missing':
        artifacts[source].unlink()
    elif state == 'mismatch':
        _text(
            artifacts[source], '{"train_mae": 2, "val_loss": 0.3}'
            if source == 'training' else 'test_mae\n9.000000\n')
    elif state == 'invalid':
        _text(artifacts[source],
              '{' if source == 'training' else 'test_mae\n0\n2\n')
    elif state == 'boolean':
        _text(
            artifacts[source], '{"train_mae": true, "val_loss": false}'
            if source == 'training' else 'test_mae\ntrue\n')
    origins = {
        key: ('Unknown metric origin' if key == source and state != 'available'
              else 'Last recorded training metrics (epoch unverified)'
              if key == 'training' else 'Selected-checkpoint evaluation')
        for key in artifacts
    }
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
        dict(name='mae', recorded_name='train_mae', value=1, split='train',
             fit='original', origin=origins['training']),
        dict(name='loss', recorded_name='val_loss', value=0.2, split='val',
             fit='original', origin=origins['training']),
        dict(name='mae', recorded_name='val_mae', value=None, split='val',
             fit='original', origin='Unknown metric origin'),
        dict(name='mae', recorded_name='test_mae', value=0, split='test',
             fit='original', origin=origins['test']),
        dict(name='mae', recorded_name='test_refit_mae', value=2, split='test',
             fit='refit_trainval', origin=origins['test_refit'])
    ]
    summary = static_summary(result)
    for origin in origins.values():
        assert origin in summary
    assert 'epoch 7' not in summary
    assert result['trials']['completed'] == 0
    assert result['trials']['rows'][0]['selected']
    assert result['trials']['rows'][0][
        'comparison_source'] == 'last_recorded_epoch'
    assert result['trials']['rows'][0]['comparison_epoch'] == '7'
    assert result['trials']['rows'][0]['comparison_metrics'] == {
        'val_mae': None
    }
    progress['final_validation_metrics'] = {'val_loss': 0.3}
    save_json(
        dict(total_trials=1, completed_trials=1, trial_progress={'2':
                                                                 progress}),
        str(selected.parent / 'auto_train_progress.json'))
    final = build_training_payload(TrainingExplanations.load(run))
    assert final['trials']['rows'][0][
        'comparison_source'] == 'final_validation_metrics'
    assert final['trials']['rows'][0]['comparison_metrics'] == {'val_loss': 0.3}
    assert final['trials']['rows'][0]['comparison_epoch'] is None
    assert final['metrics'] == result['metrics']
    assert result['cohorts']['rows'][0]['predictions'] == {
        'no': [0.8, 0.4],
        'yes': [0.2, 0.6]
    }
    assert result['gradient']['reason'] is None
    assert str(tmp_path) not in json.dumps(result)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('task', [
    'regression', 'binary_classification', 'multiclass_classification',
    'multilabel_classification'
])
@pytest.mark.parametrize('output_state', [
    'available', 'missing', 'invalid', 'mismatch', 'boolean', 'string',
    'nonfinite', 'prediction_null', 'target_null', 'both_null', 'unsupported',
    'unknown', 'oversized', 'escape', 'range', 'fractional_target', 'empty',
    'list', 'label_mismatch', 'multiple_targets', 'excess_mass', 'rounding',
    'subset', 'many_labels', 'special_labels'
])
def test_sample_coverage_deduplication_and_complete_graph_budget(
        tmp_path, monkeypatch, task, output_state):
    run, trial = _run(tmp_path, refit=True)
    _gradient(trial)
    _save_cohorts(trial, [_cohort()])
    root = f'{trial}/subgraph_collector/test'
    for artifact in ('0', '2', '3'):
        graph = _graph()
        graph.subgraph_store.task_type = TaskType(task)
        save_json(graph, f'{root}/subgraph_json/{artifact}.json')
    output = dict(task_type=task, prediction=0, target=1)
    class_output = task in ('multiclass_classification',
                            'multilabel_classification')
    if class_output:
        output.update(prediction={
            'second': 0.25,
            'first': 0.75
        }, target={
            'second': 0,
            'first': 1
        })
    status = 'available'
    if output_state in ('prediction_null', 'target_null', 'both_null'):
        for key in ('prediction', 'target'):
            if output_state == 'both_null' or output_state == f'{key}_null':
                output[key] = None
        status = 'partial'
    elif output_state == 'unknown':
        output['task_type'] = None
        status = 'unknown'
    elif output_state == 'mismatch':
        output[
            'task_type'] = 'regression' if task != 'regression' else 'binary_classification'
        status = 'invalid'
    elif output_state == 'unsupported':
        output.update(task_type='multilabel_ranking',
                      prediction={'recorded class': 0.5}, target=None)
        graph.subgraph_store.task_type = TaskType.MULTILABEL_RANKING
        save_json(graph, f'{root}/subgraph_json/0.json')
        status = 'unsupported'
    elif output_state in ('boolean', 'string', 'nonfinite'):
        value = dict(boolean=False, string='0',
                     nonfinite=float('nan'))[output_state]
        output['prediction'] = {
            'second': value,
            'first': 0.75
        } if class_output else value
        status = 'invalid'
    elif output_state in ('range', 'fractional_target'):
        key = 'prediction' if output_state == 'range' else 'target'
        value = 2 if output_state == 'range' else 0.5
        output[key] = {'second': value, 'first': 0} if class_output else value
        status = 'available' if task == 'regression' else 'invalid'
    elif output_state in ('empty', 'list', 'label_mismatch', 'multiple_targets',
                          'excess_mass', 'rounding', 'subset', 'many_labels',
                          'special_labels'):
        predictions = {
            'empty': {},
            'list': [0.5],
            'label_mismatch': {
                'other': 0.5
            },
            'multiple_targets': {
                'second': 0.25,
                'first': 0.75
            },
            'excess_mass': {
                'second': 0.6,
                'first': 0.6
            },
            'rounding': {
                'second': 0.5,
                'first': 0.50000006
            },
            'subset': {
                'second': 0.25
            },
            'many_labels': {
                f'label-{i}': 0.001
                for i in range(120)
            },
            'special_labels': {
                '__proto__': 0.25,
                'constructor': 0.75
            }
        }
        output['prediction'] = predictions[output_state]
        if output_state == 'multiple_targets':
            output['target'] = {'second': 1, 'first': 1}
        elif output_state in ('subset', 'many_labels'):
            output['target'] = dict.fromkeys(output['prediction'], 0)
        elif output_state == 'special_labels':
            output['target'] = None
        status = 'available' if class_output and (
            output_state in ('rounding', 'subset', 'many_labels')
            or task == 'multilabel_classification' and output_state in
            ('multiple_targets', 'excess_mass')) else 'invalid'
        if class_output and output_state == 'special_labels':
            status = 'partial'
    path = Path(root) / 'output/0.json'
    save_json(EntityOutput(**output), str(path))
    if output_state in ('missing', 'invalid', 'oversized', 'escape'):
        status = dict(missing='missing', invalid='invalid', oversized='limited',
                      escape='invalid')[output_state]
        if output_state == 'missing':
            path.unlink()
        elif output_state == 'escape':
            path.unlink()
            path.symlink_to(tmp_path / 'outside.json')
            (tmp_path / 'outside.json').write_text('{"private":"secret"}')
        else:
            _text(
                path, '{' if output_state == 'invalid' else ' ' *
                (LIMITS['artifact_bytes'] + 1))
    # Other collectors, original entity IDs and the original fit cannot match.
    for unrelated in (
            f'{root}/output/2.json', f'{root}/output/user-7.json',
            f'{trial.removesuffix("_refit_trainval")}/subgraph_collector/test/output/0.json'
    ):
        save_json(EntityOutput(TaskType(task), 999, 999), unrelated)
    save_json(
        EntityMappings(
            mappings={
                'random': ['0', '1'],
                'large_error': ['0']
            }, orig_mappings={
                'random': ['user-7', 'missing'],
                'large_error': ['user-7']
            }), f'{root}/strategy_mappings.json')
    facade = TrainingExplanations.load(run)
    with monkeypatch.context() as snapshot:
        forbidden = Mock(side_effect=AssertionError('Artifact was reopened'))
        snapshot.setattr(_artifacts, 'load_json', forbidden)
        snapshot.setattr(filesystem, 'open_file', forbidden)
        snapshot.setattr(filesystem, 'listdir', forbidden)
        result = build_training_payload(facade, max_samples=1)
        forbidden.assert_not_called()
    assert result['gradient']['status'] == 'available'
    assert result['cohorts']['status'] == 'available'
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
    assert list(section['outputs']) == ['0']
    recorded = section['outputs']['0']
    assert recorded['status'] == status
    if status in ('available', 'partial'):
        assert recorded['prediction'] == output['prediction']
        assert recorded['target'] == output['target']
        assert recorded['task_type'] == task
        if class_output and recorded['prediction'] is not None:
            assert list(recorded['prediction']) == list(output['prediction'])
    else:
        assert recorded.get('prediction') is None
        assert recorded.get('target') is None
    assert 'secret' not in json.dumps(result)
    assert '999' not in json.dumps(section['outputs'])


@pytest.mark.parametrize('task',
                         ['static_link_prediction', 'temporal_link_prediction'])
@pytest.mark.parametrize('state', [
    'available', 'duplicates', 'prediction_null', 'target_null', 'both_null',
    'empty', 'many', 'long', 'too_long', 'unsafe_integer', 'safe_integer',
    'float', 'boolean', 'null_id', 'list_id', 'dict_id', 'nonfinite',
    'missing_id', 'missing_cols', 'invalid_cols', 'invalid_column',
    'invalid_name', 'missing_name', 'null_name', 'scalar', 'dict',
    'node_scalar', 'oversized', 'mismatch', 'unknown'
])
def test_link_output_preserves_ordered_ids_and_rejects_malformed_records(
        tmp_path, task, state):
    run, trial = _run(tmp_path)
    root = Path(trial) / 'subgraph_collector/test'
    graph = _graph()
    graph.subgraph_store.task_type = TaskType(task)
    save_json(graph, str(root / 'subgraph_json/0.json'))
    node = dict(id='item-2', id_name='item_id', cols=[])
    output = dict(task_type=task, prediction=[node,
                                              dict(node, id='item-1')],
                  target=[dict(node, id='observed')])
    status = 'available'
    if state in ('mismatch', 'unknown'):
        output['task_type'] = 'regression' if state == 'mismatch' else None
        status = 'invalid' if state == 'mismatch' else 'unknown'
    elif state.endswith('_null'):
        for key in ('prediction', 'target'):
            if state in ('both_null', f'{key}_null'):
                output[key] = None
        status = 'partial'
    elif state in ('empty', 'many', 'duplicates'):
        output['prediction'] = [] if state == 'empty' else [
            dict(node) for _ in range(120 if state == 'many' else 2)
        ]
        if state == 'empty':
            output['target'] = []
    elif state in ('scalar', 'dict', 'node_scalar'):
        output['prediction'] = dict(scalar=1, dict={}, node_scalar=[1])[state]
        status = 'invalid'
    elif state not in ('available', 'oversized'):
        changes = {
            'long': ('id', '<script>\n' + 'x' * 200),
            'too_long': ('id', 'x' * (LIMITS['string'] + 1)),
            'unsafe_integer': ('id', 2**53),
            'safe_integer': ('id', 2**53 - 1),
            'float': ('id', 1.25),
            'boolean': ('id', False),
            'null_id': ('id', None),
            'list_id': ('id', [1]),
            'dict_id': ('id', {
                'a': 1
            }),
            'nonfinite': ('id', float('inf')),
            'missing_id': ('id', None),
            'missing_cols': ('cols', None),
            'invalid_cols': ('cols', {}),
            'invalid_column': ('cols', [1]),
            'invalid_name': ('id_name', 1),
            'missing_name': ('id_name', None),
            'null_name': ('id_name', None),
        }
        key, value = changes[state]
        node[key] = value
        if state.startswith('missing_'):
            del node[key]
        status = 'available' if state in (
            'long', 'safe_integer', 'float', 'missing_name',
            'null_name') else 'limited' if state == 'too_long' else 'invalid'
    path = root / 'output/0.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(output))
    if state == 'oversized':
        path.write_text(' ' * (LIMITS['artifact_bytes'] + 1))
        status = 'limited'
    result = build_training_payload(TrainingExplanations.load(run))
    recorded = result['samples']['outputs']['0']
    assert recorded['status'] == status
    assert result['samples']['references'][0]['status'] == 'loaded'
    if status in ('available', 'partial'):
        for key in ('prediction', 'target'):
            assert recorded[key] == (None if output[key] is None else [
                dict(id=item['id'], id_name=item.get('id_name'))
                for item in output[key]
            ])
    else:
        assert 'prediction' not in recorded and 'target' not in recorded


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


@pytest.mark.parametrize(
    'kind', ['directory', 'gradient', 'cohorts', 'input_exact', 'input_short'])
def test_count_limits_precede_facade_reads(tmp_path, monkeypatch, kind):
    run, trial = _run(tmp_path)
    if kind == 'directory':
        for key in ('0', '1'):
            save_json(
                _graph(),
                f'{trial}/subgraph_collector/test/subgraph_json/{key}.json')
        bound, method, section = 'directory_entries', 'samples', 'samples'
    elif kind == 'gradient' or kind.startswith('input_'):
        _gradient(trial)
        bound, method, section = 'gradient_rows', 'gradient_scores', 'gradient'
    else:
        _save_cohorts(trial, [_cohort(), _cohort()])
        bound, method, section = 'cohorts', 'column_analysis', 'cohorts'
    limit = 1
    if kind.startswith('input_'):
        selection = sum(path.stat().st_size
                        for path in (Path(trial) / 'config.json',
                                     Path(trial).parent / 'best_trainer.json'))
        bound = 'input_bytes'
        limit = (2 * selection +
                 (Path(trial) / 'explain_test/scores.csv').stat().st_size -
                 (kind == 'input_short'))
    monkeypatch.setitem(LIMITS, bound, limit)
    forbidden = Mock(side_effect=AssertionError('Count limit was bypassed'))
    monkeypatch.setattr(TrainingExplanations, method, forbidden)
    facade = TrainingExplanations.load(run)
    result = facade.visualize()._payload
    assert result[section]['status'] == ('available' if kind == 'input_exact'
                                         else 'limited')
    if kind.startswith('input_'):
        assert facade.visualize()._payload == result
        attached = TrainingExplanations.from_trial(run, trial)
        assert attached == facade
        assert attached.visualize()._payload[section]['status'] == 'available'
        assert _artifacts.read_gradient(
            f'{trial}/explain_test/scores.csv').tables
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


@pytest.mark.parametrize(('task', 'architecture', 'module'), [
    ('static_link_prediction', 'two_tower', 'two_tower'),
    ('static_link_prediction', 'shallow_rhs', 'shallow_rhs'),
    ('static_link_prediction', 'context', 'hybrid'),
    ('temporal_link_prediction', 'two_tower', 'two_tower'),
    ('temporal_link_prediction', 'shallow_rhs', 'shallow_rhs'),
    ('temporal_link_prediction', 'context', 'hybrid'),
    ('temporal_link_prediction', 'pairwise', 'id'),
    ('regression', 'two_tower', 'two_tower'),
    ('binary_classification', 'shallow_rhs', 'shallow_rhs'),
    ('forecasting', '???', None),
])
@pytest.mark.parametrize('refit', [False, True])
@pytest.mark.parametrize('state', [
    'missing', 'retained', 'invalid', 'disabled', 'failed', 'unknown_module',
    'original_conflict', 'plan_conflict', 'missing_config', 'missing_plan',
    'malformed_config', 'oversized_plan', 'escaping_plan', 'unknown_task',
    'unverified_selection', 'cohort_conflict', 'sample_conflict',
    'missing_module_key', 'head_conflict', 'malformed_head',
    'malformed_architecture', 'plan_module_conflict', 'missing_plan_module',
    'null_module', 'list_module', 'dict_module', 'null_plan_module',
    'list_plan_module', 'dict_plan_module', 'metric_conflict', 'unknown_metric'
])
def test_producer_guidance_preserves_observation_and_abstains(
        tmp_path, task, architecture, module, refit, state):
    run, trial = _run(tmp_path, refit=refit)
    trial = Path(trial)
    original = trial.with_name('version_2')
    for path in dict.fromkeys((original, trial)):
        save_json(
            dict(
                refit_cfg=dict(refitted_trainval=refit), tune_metric='map@10',
                module_kwargs=None if task == 'forecasting' else dict(
                    module=module), head={
                        '_target_':
                        'kumo_core.modeling.heads.forecasting.ForecastingHead'
                    }), str(path / 'config.json'))
        save_json(dict(model_architecture=dict(module=architecture)),
                  str(path / 'model_plan.json'))
    save_json(dict(task_type=task),
              str(run / 'published-model/manifests/task.json'))
    save_json(
        dict(predictive_query='PREDICT LIST_DISTINCT(orders.item_id) '
             'RANK TOP 3 FOR EACH users.id'),
        str(run / 'published-model/training-spec.json'))
    if state in ('retained', 'disabled', 'failed', 'cohort_conflict',
                 'sample_conflict'):
        _gradient(str(trial))
        save_json(
            dict(task_type='regression' if state == 'cohort_conflict' else task,
                 cohorts=[_cohort()]),
            str(trial / 'column_analysis/test/column_analysis.json'))
        graph = _graph()
        graph.subgraph_store.task_type = TaskType('regression' if state ==
                                                  'sample_conflict' else task)
        save_json(graph,
                  str(trial / 'subgraph_collector/test/subgraph_json/0.json'))
    if state == 'invalid':
        _text(trial / 'explain_test/scores.csv', '{')
    elif state in ('unknown_module', 'original_conflict', 'malformed_config'):
        path = original if state == 'original_conflict' else trial
        config = json.loads((path / 'config.json').read_text())
        config['module_kwargs'] = ([] if state == 'malformed_config' else {
            'module':
            'unknown' if state == 'unknown_module' else
            'shallow_rhs' if module == 'two_tower' else 'two_tower'
        })
        save_json(config, str(path / 'config.json'))
    elif state in ('null_module', 'list_module', 'dict_module'):
        config = json.loads((trial / 'config.json').read_text())
        config['module_kwargs'] = dict(module={
            'null_module': None,
            'list_module': [],
            'dict_module': {}
        }[state])
        save_json(config, str(trial / 'config.json'))
    elif state in ('missing_module_key', 'head_conflict', 'malformed_head'):
        config = json.loads((trial / 'config.json').read_text())
        if state == 'missing_module_key':
            del config['module_kwargs']
        else:
            config['head'] = (None if state == 'malformed_head' else {
                '_target_': 'untrusted.module.Head'
            })
        save_json(config, str(trial / 'config.json'))
    elif state == 'malformed_architecture':
        save_json(dict(model_architecture=None), str(trial / 'model_plan.json'))
    elif state in ('plan_module_conflict', 'missing_plan_module',
                   'null_plan_module', 'list_plan_module', 'dict_plan_module'):
        values = dict(
            plan_module_conflict='shallow_rhs'
            if architecture == 'two_tower' else 'two_tower',
            null_plan_module=None, list_plan_module=[], dict_plan_module={})
        save_json(
            dict(model_architecture={} if state ==
                 'missing_plan_module' else dict(module=values[state])),
            str(trial / 'model_plan.json'))
    elif state in ('missing_config', 'missing_plan'):
        (trial /
         ('config.json' if state == 'missing_config' else 'model_plan.json')
         ).unlink()
    elif state in ('plan_conflict', 'oversized_plan', 'escaping_plan'):
        path = trial / 'model_plan.json'
        _text(
            path, '{' if state == 'plan_conflict' else ' ' *
            (LIMITS['artifact_bytes'] + 1))
        if state == 'escaping_plan':
            path.unlink()
            path.symlink_to(tmp_path / 'outside.json')
            _text(tmp_path / 'outside.json', '{}')
    elif state == 'unknown_task':
        save_json(dict(task_type='unknown'),
                  str(run / 'published-model/manifests/task.json'))
    elif state == 'unverified_selection':
        save_json(dict(best_trainer=str(original.with_name('version_9'))),
                  str(original.parent / 'best_trainer.json'))
    elif state in ('metric_conflict', 'unknown_metric'):
        config = json.loads((trial / 'config.json').read_text())
        config['tune_metric'] = ('map@3'
                                 if state == 'metric_conflict' else 'unknown')
        save_json(config, str(trial / 'config.json'))
    explanations = TrainingExplanations.from_trial(
        run, trial, disabled=state == 'disabled',
        failure='private/path' if state == 'failed' else None)
    availability = explanations.availability
    result = build_training_payload(explanations)
    expected = state in ('missing', 'retained', 'invalid', 'disabled', 'failed',
                         'metric_conflict', 'unknown_metric')
    expected |= task != 'forecasting' and state in ('head_conflict',
                                                    'malformed_head')
    expected |= task == 'forecasting' and state in (
        'plan_module_conflict', 'missing_plan_module', 'null_plan_module',
        'list_plan_module', 'dict_plan_module')
    is_link = task in ('static_link_prediction', 'temporal_link_prediction')
    expected &= task == 'forecasting' or (
        is_link and
        not (task == 'static_link_prediction' and module in ('hybrid', 'id')))
    for name in ('gradient', 'cohorts', 'samples'):
        note = result[name].pop('producer_note', None)
        assert bool(note) == (
            expected and
            (task == 'forecasting' or name == 'cohorts' or module == 'two_tower'
             or name == 'gradient' and module == 'shallow_rhs'))
        if note:
            assert ('Recorded forecasting.' if task == 'forecasting' else
                    'Recorded two-tower link prediction.' if module ==
                    'two_tower' else 'Recorded shallow-RHS link prediction.' if
                    name == 'gradient' else 'Recorded link prediction.') in note
            if is_link and name == 'cohorts':
                assert 'Column cohorts require a node-prediction module.' in note
            if task == 'forecasting' and name == 'gradient':
                assert 'normalized over produced columns' in note
                assert 'summed encoded model outputs across all forecast horizons' in note
                assert 'Dedicated lag/year-over-year encoders' in note
                assert 'separate forecast-head inputs, when used' in note
                assert 'No per-horizon scores are retained.' in note
            elif is_link and name == 'gradient' and module == 'shallow_rhs':
                assert 'Global and local feature sensitivities' in note
                assert 'summed retrieved dot-product scores' in note
                assert 'sampled graph feature encoders' in note
                assert 'No per-item breakdown is retained.' in note
                assert 'not probabilities or complete attributions' in note
                assert 'learned entity/item embeddings' in note
                assert 'Query TOP K does not establish' in note
                assert '3' not in note and '10' not in note
            else:
                assert 'Eligibility alone does not explain missing artifacts.' in note
            result[name]['producer_note'] = note
            assert note in static_summary(result)
            if name == 'gradient':
                assert 'Producer scope: ' + note in static_summary(result)
            del result[name]['producer_note']
    assert explanations.availability == availability
    assert 'private/path' not in json.dumps(result)
    # With eligibility evidence absent, all observations remain identical.
    for path in dict.fromkeys((original, trial)):
        plan = path / 'model_plan.json'
        plan.unlink(missing_ok=True)
    assert result == build_training_payload(explanations)


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
    assert result['provenance']['generation_status'] == (reason if reason in (
        'disabled', 'failed') else 'unknown')
    assert (f'Explanation generation: {reason}'
            in static_summary(result)) == (reason in ('disabled', 'failed'))


@pytest.mark.parametrize('count', [0, 100, 101, 450])
@pytest.mark.parametrize('context_state',
                         ['retained', 'long', 'missing', 'provenance'])
def test_static_metric_limit_is_explicit(tmp_path, count, context_state):
    run, _ = _run(tmp_path)
    save_json(
        {
            f'test_metric_{i}_<img>':
            0 if i == 0 else 1.2345678901234567 if i == 1 else None
            for i in range(count)
        }, str(run / 'published-model/metrics.json'))
    result = TrainingExplanations.load(run).visualize()._payload
    suffix = 'x' * 8192 if context_state == 'long' else ''
    result['context'] = dict(run='run-<img>' + suffix,
                             task_type='forecasting<&>',
                             query='PREDICT <script>&"quoted"' + suffix,
                             target='target<&>')
    if context_state in ('missing', 'provenance'):
        result['context'] = {}
    if context_state == 'provenance':
        result['provenance']['task_type'] = 'forecasting<&>'
    result['samples']['bodies'] = {'0': {'value': 'PRIVATE_EXAMPLE_SENTINEL'}}
    summary = static_summary(result)
    if context_state in ('retained', 'long'):
        assert 'Run: run-&lt;img&gt;' + suffix in summary
        assert 'Predictive query: PREDICT &lt;script&gt;&amp;&quot;quoted&quot;' + suffix in summary
        assert 'Target: target&lt;&amp;&gt;' in summary
    else:
        assert 'Run: Unavailable' in summary
        assert 'Predictive query: Unavailable' in summary
    assert ('Task: forecasting&lt;&amp;&gt;' in summary) == (context_state
                                                             != 'missing')
    assert ('Task: Unavailable' in summary) == (context_state == 'missing')
    assert ('Explore predictive query' in summary) == (context_state == 'long')
    assert 'PRIVATE_EXAMPLE_SENTINEL' not in summary
    assert len(result['metrics']['rows']) == count
    assert summary.count(' / test / ') == min(count, 100)
    notice = f'Showing 100 of {count} loaded metrics.'
    recovery = ('Rerun this cell in a trusted notebook to inspect all '
                'loaded metrics.')
    assert (notice in summary) == (count > 100)
    assert (recovery in summary) == (count > 100)
    if count:
        assert 'metric_0_&lt;img&gt; / Unknown metric origin: 0' in summary
        assert 'metric_1_&lt;img&gt; / Unknown metric origin: 1.2345678901234567' in summary
        assert 'metric_99_&lt;img&gt; / Unknown metric origin: Unavailable' in summary
    assert '<img>' not in summary
    assert 'metric_100_' not in summary
    if count > 100:
        assert summary.index(notice) < summary.index('<ul>')


@pytest.mark.parametrize('generation',
                         ['unknown', 'disabled', 'failed', 'both'])
@pytest.mark.parametrize(
    'state',
    ['missing', 'retained', 'invalid', 'negative', 'limited', 'invalid_root'])
def test_negative_gradient_and_failed_generation_are_explicit(
        tmp_path, generation, state):
    run, trial = _run(tmp_path)
    if state not in ('missing', 'invalid_root'):
        _gradient(trial)
        _save_cohorts(trial, [_cohort()])
        save_json(_graph(),
                  f'{trial}/subgraph_collector/test/subgraph_json/0.json')
    if state in ('invalid', 'negative', 'limited'):
        _text(
            Path(trial) / 'explain_test/scores.csv',
            '{' if state == 'invalid' else ' ' * (LIMITS['artifact_bytes'] + 1)
            if state == 'limited' else 'table,column,score\nusers,age,-0.2\n')
    failed = generation in ('failed', 'both')
    disabled = generation in ('disabled', 'both')
    expected = 'disabled' if disabled else 'failed' if failed else 'unknown'
    explanations = TrainingExplanations.from_trial(
        run, tmp_path / 'outside' if state == 'invalid_root' else trial,
        disabled=disabled, failure='private/path' if failed else None)
    result = build_training_payload(explanations)
    assert result['provenance']['generation_status'] == expected
    assert result['provenance']['generation_failure'] == (
        'Explanation generation failed' if failed else None)
    summary = static_summary(result)
    assert (f'Explanation generation: {expected}' in summary) == (expected
                                                                  != 'unknown')
    assert ('<li>Explanation generation failed</li>'
            in summary) == (generation == 'both')
    assert 'private/path' not in json.dumps(result)
    reopened = build_training_payload(TrainingExplanations.load(run))
    assert reopened['provenance']['generation_status'] == 'unknown'
    assert reopened['provenance']['generation_failure'] is None
    if state != 'invalid_root':
        for name in ('gradient', 'cohorts', 'samples'):
            original = reopened[name]
            assert result[name]['status'] == (
                'partial' if failed and original['status'] == 'available' else
                original['status'])
            for key in ('rows', 'bodies', 'outputs', 'coverage'):
                assert result[name].get(key) == original.get(key)
        if state == 'retained':
            assert any(row['score'] == 0 for row in result['gradient']['rows'])
            assert result['samples']['bodies']['0']


def test_payload_limit_discards_whole_graphs(tmp_path, monkeypatch):
    run, trial = _run(tmp_path)
    save_json(_graph(), f'{trial}/subgraph_collector/test/subgraph_json/0.json')
    save_json(EntityOutput(TaskType.REGRESSION, 0, None),
              f'{trial}/subgraph_collector/test/output/0.json')
    original = build_training_payload(TrainingExplanations.load(run))
    bound = len(json.dumps(original, separators=(',', ':')).encode()) - 100
    monkeypatch.setitem(LIMITS, 'payload_bytes', bound)
    result = build_training_payload(TrainingExplanations.load(run))
    assert not result['samples']['bodies']
    assert not result['samples']['outputs']
    assert all(ref['status'] != 'loaded'
               for ref in result['samples']['references'])
    assert len(json.dumps(result, separators=(',', ':')).encode()) <= bound


@pytest.mark.parametrize('max_samples', [-1, 13, True, 1.5])
def test_invalid_preload_request(max_samples):
    with pytest.raises(ValueError, match='max_samples'):
        build_training_payload(TrainingExplanations.unavailable(),
                               max_samples=max_samples)
