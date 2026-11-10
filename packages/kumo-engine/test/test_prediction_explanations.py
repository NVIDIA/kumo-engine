# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prediction API integration and retained PRED-stage explanation contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumo_core.io import filesystem
from kumo_core.io.json import save_json
from kumo_core.prediction_explanations import PredictionExplanationSupport
from kumo_core.types import TaskType
from kumoapi.explain.column_analysis import ColumnAnalysisOutput, ColumnCohorts
from kumoapi.explain.common import EntityMappings
from kumoapi.explain.subgraph import Subgraph, SubgraphNode, SubgraphStore
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

from kumo.engine import (
    ExplanationUnavailableError,
    InvalidExplanationError,
    InvalidRunError,
    PredictionExplanations,
    Trainer,
)
from kumo.engine.trainer._scoring import ScoringDefaults


def _cohorts(root):
    cohort = ColumnCohorts(
        table_name='users', hop=0, col_name='age', stype=Stype.numerical,
        cohorts=['young', 'old'], populations=[.25, .75],
        predictions=[.2, .8], prediction_score=.3, targets=None,
        target_score=None)
    output = ColumnAnalysisOutput(task_type=TaskType.REGRESSION,
                                  cohorts=[cohort])
    save_json(output, f'{root}/column_analysis/pred/column_analysis.json')
    return output


def _samples(root):
    collector = f'{root}/subgraph_collector/pred'
    save_json(EntityMappings(mappings={'first': ['7']},
                             orig_mappings={'first': ['current-user']}),
              f'{collector}/strategy_mappings.json')
    graph = Subgraph(subgraph_store=SubgraphStore(
        root_id=7, task_type=TaskType.REGRESSION,
        nodes={7: SubgraphNode(7, 'users', 'current-user')},
        edges={}, tables={'users': [7]}))
    save_json(graph, f'{collector}/subgraph_json/7.json')


@pytest.fixture
def prediction(tmp_path, snapshot, monkeypatch):
    """Keep scoring identity real at the API boundary, without loading a model."""
    trainer = Trainer(ModelPlan(), accelerator='cpu')
    trainer._training_result = SimpleNamespace(
        artifact_paths={}, manifest=None, scoring_defaults=ScoringDefaults())
    trainer._training_query = 'PREDICT users.label FOR EACH users.id'
    monkeypatch.setattr(Trainer, 'scratch_dir', property(lambda _: tmp_path))
    monkeypatch.setattr('kumo.engine.trainer.trainer.new_run_id', lambda: 'xai')
    monkeypatch.setattr('kumo.engine.trainer.trainer._prediction_task_type',
                        lambda *a: TaskType.REGRESSION)
    monkeypatch.setattr('kumo_core.pquery.normalize_graph', lambda *a: None)
    monkeypatch.setattr('kumo_core.pquery.parse_and_validate_query',
                        lambda *a: SimpleNamespace(
                            query=SimpleNamespace(top_k=None)))
    monkeypatch.setattr('kumo.engine.trainer.trainer.bundle_task_options',
                        lambda *a: {})
    materialized = object()
    monkeypatch.setattr('kumo.engine.trainer.trainer._materialize_prediction',
                        lambda *a: (materialized, [123]))
    monkeypatch.setattr('kumo.engine.trainer.trainer._stage_model_bundle',
                        lambda *a: tmp_path / 'bundle')
    encoders = object()
    monkeypatch.setattr(trainer, '_trained_encoders', lambda *a: encoders)
    selection = object()
    score = Mock(return_value=selection)
    monkeypatch.setattr('kumo_core.prediction.score_prediction', score)
    monkeypatch.setattr('kumo.engine.trainer.trainer._run_manifest.start',
                        lambda **kw: None)
    marker = object()
    monkeypatch.setattr('kumo.engine.trainer.trainer._run_manifest.finish',
                        lambda *a: marker)
    collect = Mock()
    monkeypatch.setattr('kumo.engine.trainer.trainer._collect_garbage', collect)
    replay = Mock(return_value=PredictionExplanationSupport(True, True))
    monkeypatch.setattr('kumo_core.prediction_explanations.explain_prediction',
                        replay)
    table = SimpleNamespace(query='prediction query', tt_id='current-table')
    return SimpleNamespace(trainer=trainer, table=table, snapshot=snapshot,
                           replay=replay, score=score, selection=selection,
                           materialized=materialized, encoders=encoders,
                           collect=collect, tracking=marker)


def test_prediction_explanations_opt_in(prediction):
    p = prediction
    result = p.trainer.predict(p.table, snapshot=p.snapshot)
    p.replay.assert_not_called()
    assert result.selection is p.selection
    assert result.tracking is p.tracking
    assert all(state.reason == 'disabled'
               for state in result.explanations.availability.values())
    p.collect.assert_called_once_with('predict')


@pytest.mark.parametrize('failure', [False, True])
def test_prediction_replay_preserves_outputs(prediction, monkeypatch, failure,
                                             caplog):
    p = prediction

    def replay(**kwargs):
        p.score.assert_called_once()
        assert kwargs['materialized'] is p.materialized
        assert kwargs['encoders'] is p.encoders
        assert kwargs['accelerator'] == 'cpu'
        assert kwargs['output_dir'].endswith('/xai')
        _cohorts(kwargs['output_dir'])
        if failure:
            raise RuntimeError('sample generation failed')
        _samples(kwargs['output_dir'])
        return PredictionExplanationSupport(True, True)

    p.replay.side_effect = replay
    with monkeypatch.context() as lazy:
        lazy.setattr('kumo.engine.explain.results.read_cohorts',
                     Mock(side_effect=AssertionError('must stay lazy')))
        lazy.setattr('kumo.engine.explain.results.read_sample',
                     Mock(side_effect=AssertionError('must stay lazy')))
        result = p.trainer.predict(p.table, snapshot=p.snapshot,
                                   explanations=True)
    assert result.selection is p.selection
    assert result.tracking is p.tracking
    p.collect.assert_called_once_with('predict')
    xai = result.explanations
    assert xai.availability['cohorts'].available
    assert xai.column_analysis().cohorts[0].targets is None
    assert xai.cohort_keys() == (('users', 0, 'age'),)
    assert xai.classes(('users', 0, 'age')) == ()
    assert xai.cohort(('users', 0, 'age')).predictions == [.2, .8]
    reopened = PredictionExplanations.load(xai.run_root)
    assert reopened.column_analysis() == xai.column_analysis()
    if failure:
        assert xai.availability['samples'].reason == 'failed'
        assert 'sample generation failed' in xai.availability[
            'cohorts'].generation_failure
        assert reopened.availability['samples'].reason == 'unknown'
        assert 'Prediction outputs remain available' in caplog.text
    else:
        references = xai.samples()
        assert len(references) == 1
        assert references[0].original_ids == {'first': 'current-user'}
        assert xai.sample('7').subgraph_store.nodes[7].pkey == 'current-user'
        assert reopened.sample('7') == xai.sample('7')


def test_prediction_unsupported_is_explicit(prediction):
    p = prediction
    p.replay.return_value = PredictionExplanationSupport(False, False)
    result = p.trainer.predict(p.table, snapshot=p.snapshot, explanations=True)
    assert all(state.reason == 'unsupported'
               for state in result.explanations.availability.values())
    with pytest.raises(ExplanationUnavailableError):
        result.explanations.column_analysis()


@pytest.mark.parametrize('value', [None, 0, 1, 'yes'])
def test_prediction_flag_validated_first(value):
    with pytest.raises(ValueError, match='explanations must be a boolean'):
        Trainer(ModelPlan()).predict(explanations=value)


def test_retained_prediction_paths_and_plot(tmp_path, monkeypatch):
    root = tmp_path / 'retained'
    _cohorts(f'{root}/xai')
    _samples(f'{root}/xai')
    xai = PredictionExplanations.load(root)
    monkeypatch.chdir(tmp_path.parent)
    assert xai.availability['cohorts'].available
    assert xai.samples()[0].available
    figure, axes = xai.plot_cohort(('users', 0, 'age'))
    assert len(axes[0].lines) == 1  # prediction only; no observed targets
    import matplotlib.pyplot as plt
    plt.close(figure)
    with pytest.raises(ValueError, match='nonnegative integer string'):
        xai.sample('../escape')
    save_json({'cohorts': None},
              f'{root}/xai/column_analysis/pred/column_analysis.json')
    with pytest.raises(InvalidExplanationError):
        xai.column_analysis()


def test_prediction_run_and_export_absence(tmp_path):
    with pytest.raises(InvalidRunError):
        PredictionExplanations.load(tmp_path / 'missing')
    file = tmp_path / 'file'
    with filesystem.open_file(str(file), 'wt') as out:
        out.write('not a run')
    with pytest.raises(InvalidRunError):
        PredictionExplanations.load(file)
    xai = PredictionExplanations.unavailable()
    assert all(state.reason == 'unknown' for state in xai.availability.values())
    with pytest.raises(ExplanationUnavailableError):
        xai.samples()
