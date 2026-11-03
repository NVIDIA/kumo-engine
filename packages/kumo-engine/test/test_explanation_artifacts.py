# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small producer-schema fixtures, not artifacts from a released GPU run.

CSV uses kumo_core.io.csv.save_csv (the gradient producer's six-decimal writer).
JSON uses the shared producer schemas and kumo_core.io.json.save_json. Their
provenance is the source at 74465d01, not release-wide compatibility evidence.
"""

import json
from dataclasses import replace
from datetime import datetime

import pytest
from kumo_core.io import filesystem
from kumo_core.io.csv import save_csv
from kumo_core.io.json import save_json
from kumoapi.explain.column_analysis import ColumnAnalysisOutput, ColumnCohorts
from kumoapi.explain.common import EntityMappings
from kumoapi.explain.gradient import GraphGradientScore, TableGradientScore
from kumoapi.explain.subgraph import (
    AggregatedNodes,
    Feature,
    Subgraph,
    SubgraphFeatureStore,
    SubgraphNode,
    SubgraphStore,
)
from kumoapi.task import TaskType
from kumoapi.typing import Stype

from kumo.engine.explain._artifacts import explanation_trial
from kumo.engine.explain.errors import (
    ExplanationUnavailableError,
    InvalidExplanationError,
    InvalidRunError,
)
from kumo.engine.explain.results import TrainingExplanations


def _text(path, contents):
    filesystem.makedirs(str(path.parent), exist_ok=True)
    with filesystem.open_file(str(path), 'wt', encoding='utf-8') as destination:
        destination.write(contents)


def _run(tmp_path, *, refit=False, config=True, recorded_path=None):
    run = tmp_path / 'run'
    automl = run / 'models/task/automl'
    trial = automl / 'version_2'
    filesystem.makedirs(str(trial), exist_ok=True)
    save_json({'best_trainer': recorded_path or str(trial)},
              str(automl / 'best_trainer.json'))
    if config:
        save_json({'refit_cfg': {'refitted_trainval': refit}},
                  str(trial / 'config.json'))
    selected = explanation_trial(trial, refitted_trainval=refit)
    if refit:
        filesystem.makedirs(selected, exist_ok=True)
    return run, selected


def _gradient(trial):
    scores = GraphGradientScore(tables={
        'users': TableGradientScore('users', {'age': 1, 'zero': 0}),
        'events': TableGradientScore('events', {'age': 2}),
    }).normalize()
    save_csv(scores.to_pandas(), f'{trial}/explain_test/scores.csv')
    return scores


def _cohort(*, classification=False):
    predictions = ({'no': [0.8, 0.4], 'yes': [0.2, 0.6]}
                   if classification else [2.0, 5.0])
    return ColumnCohorts(
        stype=Stype.numerical,
        cohorts=['young', 'old'], populations=[0.25, 0.75],
        predictions=predictions, targets=None,
        prediction_score=0.25, target_score=None,
        table_name='users', hop=0, col_name='age')


def _save_cohorts(trial, cohorts, *, classification=False):
    result = ColumnAnalysisOutput(
        task_type=(TaskType.MULTICLASS_CLASSIFICATION
                   if classification else TaskType.REGRESSION),
        cohorts=cohorts)
    save_json(result, f'{trial}/column_analysis/test/column_analysis.json')


def _graph():
    return Subgraph(
        subgraph_store=SubgraphStore(
            root_id=0, task_type=TaskType.REGRESSION,
            seed_time=datetime(2024, 1, 2),
            nodes={0: SubgraphNode(0, 'user_id', 'user-7'),
                   -1: AggregatedNodes(-1, 12)},
            edges={0: {'events': [-1]}},
            tables={'users': [0], 'events': [-1]}),
        feature_store=SubgraphFeatureStore(features={
            0: {'age': Feature('users', 'age', 37, Stype.numerical, 0.0)},
            -1: {'amount': Feature('events', 'amount', 21.5, Stype.numerical)},
        }))


def test_selected_refit_reopen_and_rounding(tmp_path):
    run, trial = _run(tmp_path, refit=True)
    scores = _gradient(trial)
    # Nonselected trials must not influence deterministic selection.
    save_csv(scores.to_pandas(),
             str(run / 'models/task/automl/version_9/explain_test/scores.csv'))
    result = TrainingExplanations.load(run)
    assert result.trial_root == trial
    restored = result.gradient_scores()
    assert isinstance(restored, GraphGradientScore)
    assert restored.tables['events'].columns['age'] == 0.666667
    assert restored.tables['users'].columns['age'] == 0.333333
    assert restored.tables['users'].columns['zero'] == 0
    assert list(restored.to_pandas().columns) == ['table', 'column', 'score']
    assert result.availability['gradient'].available
    assert result.availability['cohorts'].reason == 'unknown'


def test_copied_run_rebases_old_absolute_trial(tmp_path):
    run, trial = _run(
        tmp_path, recorded_path='/old/host/run/models/task/automl/version_2')
    _gradient(trial)
    assert TrainingExplanations.load(run).trial_root == trial


def test_lazy_local_paths_survive_cwd_change(tmp_path, monkeypatch):
    run, trial = _run(tmp_path)
    _gradient(trial)
    monkeypatch.chdir(tmp_path)
    reopened = TrainingExplanations.load('run')
    attached = TrainingExplanations.from_trial(
        'run', 'run/models/task/automl/version_2')
    monkeypatch.chdir(tmp_path.parent)
    assert reopened.gradient_scores() == attached.gradient_scores()
    assert reopened.run_root == str(run)


@pytest.mark.parametrize('path', [
    '/old/models/wrong/automl/version_2',
    '/old/models/task/automl/../version_2',
    '/old/models/task/automl/version_9',
    '/old/models/task/automl/version_2_refit_trainval',
])
def test_invalid_selected_path_not_guessed(tmp_path, path):
    run, _ = _run(tmp_path, recorded_path=path)
    with pytest.raises(InvalidRunError):
        TrainingExplanations.load(run)


def test_ambiguous_run_and_model_only_rejected(tmp_path):
    run, _ = _run(tmp_path)
    save_json({'best_trainer': '/other/automl/version_0'},
              str(run / 'models/other/automl/best_trainer.json'))
    with pytest.raises(InvalidRunError, match='found 2'):
        TrainingExplanations.load(run)
    model = tmp_path / 'published-model'
    filesystem.makedirs(str(model), exist_ok=True)
    with pytest.raises(InvalidRunError, match='model bundle alone'):
        TrainingExplanations.load(model)


def test_explicit_refit_metadata_beats_stale_directory(tmp_path):
    run, trial = _run(tmp_path)
    _gradient(f'{trial}_refit_trainval')
    assert TrainingExplanations.load(run).trial_root == trial
    save_json({'refit_cfg': {'refitted_trainval': True}},
              f'{trial}/config.json')
    filesystem.rm(f'{trial}_refit_trainval')
    result = TrainingExplanations.load(run)
    assert result.trial_root == f'{trial}_refit_trainval'
    assert not result.availability['gradient'].available
    with pytest.raises(ExplanationUnavailableError):
        result.gradient_scores()


def test_missing_refit_metadata_rejected(tmp_path):
    run, _ = _run(tmp_path, refit=True, config=False)
    with pytest.raises(InvalidRunError, match='config.json'):
        TrainingExplanations.load(run)


@pytest.mark.parametrize('contents', [
    'wrong,column,score\nusers,age,1\n',
    'table,column,score\nusers,age,nan\n',
    'table,column,score\nusers,age,inf\n',
    'table,column,score\nusers,age,1\nusers,age,2\n',
    'table,column,score\n,age,1\n',
    'table,column,score\nusers,age\n',
])
def test_bad_gradient_is_not_empty_or_zero(tmp_path, contents):
    run, trial = _run(tmp_path)
    _text(tmp_path / 'run/models/task/automl/version_2/explain_test/scores.csv',
          contents)
    result = TrainingExplanations.load(run)
    assert result.availability['gradient'].available
    with pytest.raises(InvalidExplanationError):
        result.gradient_scores()


def test_zero_gradient_preserved(tmp_path):
    run, trial = _run(tmp_path)
    save_csv(GraphGradientScore(tables={
        'users': TableGradientScore('users', {'age': 0}),
    }).to_pandas(), f'{trial}/explain_test/scores.csv')
    assert TrainingExplanations.load(run).gradient_scores().tables[
        'users'].columns['age'] == 0


@pytest.mark.parametrize('classification', [False, True])
def test_cohort_identity_classes_and_optional_targets(tmp_path, classification):
    run, trial = _run(tmp_path)
    cohort = _cohort(classification=classification)
    _save_cohorts(trial, [cohort], classification=classification)
    result = TrainingExplanations.load(run)
    key = ('users', 0, 'age')
    assert result.cohort_keys() == (key,)
    if classification:
        assert result.classes(key) == ('no', 'yes')
        with pytest.raises(ValueError, match='explicitly'):
            result.cohort(key)
        with pytest.raises(KeyError):
            result.cohort(key, class_name='typo')
        selected = result.cohort(key, class_name='yes')
        assert selected.predictions == cohort.predictions
    else:
        assert result.classes(key) == ()
        with pytest.raises(ValueError, match='vectors'):
            result.cohort(key, class_name='yes')
        selected = result.cohort(key)
    assert isinstance(selected, ColumnCohorts)
    assert selected.targets is None
    assert selected.prediction_score == 0.25
    with pytest.raises(KeyError):
        result.classes(('typo', 0, 'age'))


def test_binary_cohorts_preserve_positive_class_vector(tmp_path):
    run, trial = _run(tmp_path)
    cohort = replace(_cohort(), predictions=[0.2, 0.8], targets=[0.0, 1.0])
    save_json(ColumnAnalysisOutput(TaskType.BINARY_CLASSIFICATION, [cohort]),
              f'{trial}/column_analysis/test/column_analysis.json')
    result = TrainingExplanations.load(run)
    key = ('users', 0, 'age')
    assert result.classes(key) == ()
    assert result.cohort(key).predictions == [0.2, 0.8]
    assert result.cohort(key).targets == [0.0, 1.0]


@pytest.mark.parametrize('change', [
    {'populations': [1.0]},
    {'populations': [-0.1, 1.1]},
    {'populations': [0.1, 0.1]},
    {'predictions': [float('nan'), 1]},
    {'targets': [1]},
    {'predictions': {'yes': [1, 2]}, 'targets': {'no': [0, 1]}},
    {'prediction_score': float('inf')},
    {'table_name': '???'},
    {'hop': -1},
])
def test_invalid_cohort_semantics(tmp_path, change):
    run, trial = _run(tmp_path)
    _save_cohorts(trial, [replace(_cohort(), **change)])
    with pytest.raises(InvalidExplanationError):
        TrainingExplanations.load(run).column_analysis()


def test_duplicate_cohort_keys_rejected(tmp_path):
    run, trial = _run(tmp_path)
    _save_cohorts(trial, [_cohort(), _cohort()])
    with pytest.raises(InvalidExplanationError, match='duplicate cohort'):
        TrainingExplanations.load(run).column_analysis()


def test_sample_listing_is_lazy_with_original_ids(tmp_path, monkeypatch):
    run, trial = _run(tmp_path)
    root = f'{trial}/subgraph_collector/test'
    graph = _graph()
    save_json(graph, f'{root}/subgraph_json/0.json')
    save_json(graph, f'{root}/subgraph_json/8.json')
    save_json(EntityMappings(
        mappings={'random': ['0', '1'], 'high_error': ['0']},
        orig_mappings={'random': ['user-7', 1234], 'high_error': ['user-7']}),
        f'{root}/strategy_mappings.json')
    opened = []
    real_open = filesystem.open_file

    def record_open(path, *args, **kwargs):
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(filesystem, 'open_file', record_open)
    result = TrainingExplanations.load(run)
    references = result.samples()
    assert [ref.artifact_id for ref in references] == ['0', '1', '8']
    assert references[0].categories == ('random', 'high_error')
    assert references[0].original_ids == {
        'random': 'user-7', 'high_error': 'user-7'
    }
    assert not references[1].available
    assert references[2].categories == ()
    assert result.availability['samples'].available
    assert not any('subgraph_json/' in path for path in opened)
    restored = result.sample('0')
    assert restored.subgraph_store.root_id == 0
    assert isinstance(restored.subgraph_store.nodes[-1], AggregatedNodes)
    assert restored.subgraph_store.nodes[-1].count == 12
    assert restored.subgraph_store.edges[0]['events'] == [-1]
    assert restored.feature_store.features[0]['age'].score == 0
    assert restored.feature_store.features[-1]['amount'].value == 21.5
    assert restored.subgraph_store.seed_time.year == 2024
    with pytest.raises(ExplanationUnavailableError):
        result.sample('1')


def test_graphs_without_mappings_and_partial_graphs(tmp_path):
    run, trial = _run(tmp_path)
    root = f'{trial}/subgraph_collector/test'
    save_json(Subgraph(), f'{root}/subgraph_json/3.json')
    result = TrainingExplanations.load(run)
    assert result.samples()[0].artifact_id == '3'
    assert result.sample('3').subgraph_store is None
    assert result.sample('3').feature_store is None
    # A malformed mapping must not prevent known-ID graph retrieval.
    _text(tmp_path / 'run/models/task/automl/version_2/subgraph_collector/test/'
          'strategy_mappings.json', '{')
    with pytest.raises(InvalidExplanationError):
        result.samples()
    assert result.sample('3') == Subgraph()


@pytest.mark.parametrize('artifact_id', ['../0', '/0', '1.json', '-1', '０', 0, None])
def test_sample_ids_cannot_escape_artifact_directory(tmp_path, artifact_id):
    run, _ = _run(tmp_path)
    with pytest.raises(ValueError, match='artifact ID'):
        TrainingExplanations.load(run).sample(artifact_id)


def test_broken_graph_references_rejected(tmp_path):
    run, trial = _run(tmp_path)
    graph = _graph()
    graph.subgraph_store.edges[0]['events'] = [99]
    save_json(graph, f'{trial}/subgraph_collector/test/subgraph_json/0.json')
    with pytest.raises(InvalidExplanationError, match='destination'):
        TrainingExplanations.load(run).sample('0')


@pytest.mark.parametrize('field', ['root', 'edge', 'table', 'feature'])
def test_present_graph_store_requires_valid_references(tmp_path, field):
    run, trial = _run(tmp_path)
    graph = _graph()
    if field == 'root':
        graph.subgraph_store.nodes = {}
    elif field == 'edge':
        graph.subgraph_store.edges = {99: {'events': [0]}}
    elif field == 'table':
        graph.subgraph_store.tables = {'users': [99]}
    else:
        graph.feature_store.features[99] = {}
    save_json(graph, f'{trial}/subgraph_collector/test/subgraph_json/0.json')
    with pytest.raises(InvalidExplanationError):
        TrainingExplanations.load(run).sample('0')


def test_unavailable_and_partial_generation_states(tmp_path):
    run, trial = _run(tmp_path)
    disabled = TrainingExplanations.from_trial(run, trial, disabled=True)
    assert all(state.reason == 'disabled'
               for state in disabled.availability.values())
    _gradient(trial)
    failed = TrainingExplanations.from_trial(run, trial, failure='collector failed')
    assert failed.availability['gradient'].available
    assert failed.availability['gradient'].generation_failure == 'collector failed'
    assert failed.availability['cohorts'].reason == 'failed'
    assert failed.gradient_scores().tables
    model_only = TrainingExplanations.unavailable()
    assert all(state.reason == 'model_only'
               for state in model_only.availability.values())
    with pytest.raises(ExplanationUnavailableError, match='model_only'):
        model_only.gradient_scores()


def test_read_access_does_not_create_directories(tmp_path, monkeypatch):
    run, _ = _run(tmp_path)

    def forbid_makedirs(*args, **kwargs):
        pytest.fail('Explanation read attempted to create a directory')

    monkeypatch.setattr(filesystem, 'makedirs', forbid_makedirs)
    result = TrainingExplanations.load(run)
    assert not any(state.available for state in result.availability.values())
    assert result.samples() == ()
    with pytest.raises(ExplanationUnavailableError):
        result.column_analysis()


def test_malformed_metadata_rejected(tmp_path):
    run, _ = _run(tmp_path)
    _text(run / 'models/task/automl/best_trainer.json', json.dumps({
        'best_trainer': 1234,
    }))
    with pytest.raises(InvalidRunError):
        TrainingExplanations.load(run)
