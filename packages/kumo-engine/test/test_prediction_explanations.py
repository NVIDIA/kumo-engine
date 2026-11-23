# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct lifecycle and retained prediction explanation contracts."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest
from kumo_core.io import filesystem
from kumo_core.io.json import save_json
from kumo_core.prediction_explanation_result import (
    PredictionExplanationCoverage,
    PredictionExplanationOptions,
    PredictionExplanationResult,
    PredictionExplanationSupport,
    ProducerOutcome,
)
from kumo_core.task_dataset import TaskDatasetMaterializeResult
from kumo_core.types.enums import TaskType
from kumoapi.explain.column_analysis import ColumnAnalysisOutput, ColumnCohorts
from kumoapi.explain.common import EntityMappings
from kumoapi.explain.subgraph import Subgraph, SubgraphNode, SubgraphStore
from kumoapi.typing import Stype

from kumo.engine import (
    ExplanationUnavailableError,
    InvalidRunError,
    PredictionExplanations,
    Trainer,
)
from kumo.engine._explainability._prediction import (
    PredictionExplainabilityView,
)
from kumo.engine._manifest import (
    ModelManifest,
    PredictionManifest,
    PredictionParents,
    PredictionTableRecord,
    RunScoring,
    SnapshotManifest,
    SnapshotParents,
)
from kumo.engine.explain import _prediction as lifecycle
from kumo.engine.explain._prediction_manifest import (
    PredictionExplanationManifest,
    write_record,
)


def _record(root, *, enabled=True, state='complete', result=None):
    record = PredictionExplanationManifest(
        schema_version=1, kind='prediction_explanations',
        run_id='recorded-run', enabled=enabled, state=state, result=result)
    write_record(str(root), record)
    return record


def _complete():
    return PredictionExplanationResult(
        ProducerOutcome('complete'), ProducerOutcome('complete'),
        PredictionExplanationCoverage(2, 1, 'checkpoint/step=0.ckpt'))


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


def _samples(root, task_type=TaskType.REGRESSION):
    collector = f'{root}/subgraph_collector/pred'
    save_json(EntityMappings(mappings={'first': ['7']},
                             orig_mappings={'first': ['current-user']}),
              f'{collector}/strategy_mappings.json')
    graph = Subgraph(subgraph_store=SubgraphStore(
        root_id=7, task_type=task_type,
        nodes={7: SubgraphNode(7, 'users', 'current-user')},
        edges={}, tables={'users': [7]}))
    save_json(graph, f'{collector}/subgraph_json/7.json')


@pytest.mark.parametrize('failed', ['cohorts', 'samples'])
def test_failed_producer_blocks_partial_artifacts(tmp_path, failed):
    _cohorts(f'{tmp_path}/xai')
    _samples(f'{tmp_path}/xai')
    broken = ProducerOutcome('failed', 'producer callback failed')
    complete = (_complete().samples if failed == 'cohorts' else
                _complete().cohorts)
    xai = PredictionExplanations.from_run(
        tmp_path, enabled=True, result=PredictionExplanationResult(
            broken if failed == 'cohorts' else complete,
            broken if failed == 'samples' else complete))
    assert xai.availability[failed].reason == 'failed'
    assert not xai.availability[failed].available
    assert 'callback failed' in xai.availability[failed].generation_failure
    if failed == 'cohorts':
        for read in (xai.column_analysis, xai.cohort_keys,
                     lambda: xai.classes(('users', 0, 'age')),
                     lambda: xai.cohort(('users', 0, 'age')),
                     lambda: xai.plot_cohort(('users', 0, 'age'))):
            with pytest.raises(ExplanationUnavailableError):
                read()
        assert xai.availability['samples'].available
        assert xai.sample('7').subgraph_store.root_id == 7
    else:
        with pytest.raises(ExplanationUnavailableError):
            xai.samples()
        with pytest.raises(ExplanationUnavailableError):
            xai.sample('7')
        assert xai.availability['cohorts'].available
        assert xai.column_analysis().cohorts


def test_retained_prediction_paths_and_plot(tmp_path, monkeypatch):
    root = tmp_path / 'retained'
    _cohorts(f'{root}/xai')
    _samples(f'{root}/xai')
    _record(root, result=_complete())
    xai = PredictionExplanations.load(root)
    monkeypatch.chdir(tmp_path.parent)
    assert xai.availability['cohorts'].available
    assert xai.samples()[0].available
    figure, axes = xai.plot_cohort(('users', 0, 'age'))
    assert len(axes[0].lines) == 1  # prediction only; no observed targets
    import matplotlib.pyplot as plt
    plt.close(figure)


@pytest.mark.parametrize('reopen', [False, True])
@pytest.mark.parametrize('recorded_ids,available', [
    (('7',), False), (('8',), True), (('7', '8'), True),
])
def test_sample_availability_requires_recorded_graph(tmp_path, recorded_ids,
                                                     available, reopen):
    _samples(f'{tmp_path}/xai')
    collector = f'{tmp_path}/xai/subgraph_collector/pred'
    graphs = f'{collector}/subgraph_json'
    filesystem.copy(f'{graphs}/7.json', f'{graphs}/8.json')
    filesystem.rm(f'{graphs}/7.json')
    save_json(EntityMappings(
        mappings={'first': list(recorded_ids)},
        orig_mappings={'first': ['current-user'] * len(recorded_ids)}),
        f'{collector}/strategy_mappings.json')
    result = _complete()
    _record(tmp_path, result=result)

    xai = (PredictionExplanations.load(tmp_path) if reopen else
           PredictionExplanations.from_run(tmp_path, enabled=True, result=result))
    assert xai.availability['samples'].available is available
    references = xai.samples()
    assert {item.artifact_id: item.available for item in references} == {
        id: id == '8' for id in recorded_ids
    }
    with pytest.raises(ExplanationUnavailableError):
        xai.sample('7')
    if available:
        assert xai.sample('8').subgraph_store.root_id == 7
    else:
        with pytest.raises(ExplanationUnavailableError):
            xai.sample('8')


def test_prediction_run_and_export_absence(tmp_path):
    with pytest.raises(InvalidRunError):
        PredictionExplanations.load(tmp_path)
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


@pytest.mark.parametrize('outcomes', [
    PredictionExplanationResult(ProducerOutcome('unsupported'),
                                ProducerOutcome('unsupported')),
    PredictionExplanationResult(ProducerOutcome('failed', 'cohort failed'),
                                ProducerOutcome('failed', 'collector failed')),
    replace(_complete(), samples=ProducerOutcome('failed', 'collector failed')),
    replace(_complete(), cohorts=ProducerOutcome('failed', 'cohort failed')),
])
def test_retained_producer_status_matches_live(tmp_path, outcomes):
    _cohorts(f'{tmp_path}/xai')
    _samples(f'{tmp_path}/xai')
    _record(tmp_path, result=outcomes)
    live = PredictionExplanations.from_run(tmp_path, enabled=True,
                                           result=outcomes)
    reopened = PredictionExplanations.load(tmp_path)
    assert reopened.availability == live.availability
    for view in (live.visualize(), reopened.visualize()):
        ids = _ids(view)
        assert ids[0] == 'status'
        assert ('groups' in ids) == (outcomes.cohorts.status == 'complete')
        assert ('subgraphs' in ids) == (outcomes.samples.status == 'complete')
    if outcomes.cohorts.status == 'complete':
        assert reopened.column_analysis().cohorts
    else:
        with pytest.raises(ExplanationUnavailableError):
            reopened.column_analysis()


def test_running_manifest_blocks_existing_artifacts(tmp_path):
    _cohorts(f'{tmp_path}/xai')
    _samples(f'{tmp_path}/xai')
    _record(tmp_path, state='running', result=PredictionExplanationResult(
        ProducerOutcome('unsupported'),
        ProducerOutcome('failed', 'Explanation replay did not complete')))
    reopened = PredictionExplanations.load(tmp_path)
    assert all(not state.available for state in reopened.availability.values())
    assert reopened.availability['cohorts'].reason == 'unsupported'
    assert 'did not complete' in reopened.availability[
        'samples'].generation_failure
    with pytest.raises(ExplanationUnavailableError):
        reopened.column_analysis()
    with pytest.raises(ExplanationUnavailableError):
        reopened.sample('7')


def test_copied_prediction_run_resolves_new_root(tmp_path):
    original, copied = tmp_path / 'original', tmp_path / 'renamed'
    _cohorts(f'{original}/xai')
    _samples(f'{original}/xai')
    _record(original, result=_complete())
    filesystem.copy(str(original), str(copied))
    filesystem.rm(str(original))
    reopened = PredictionExplanations.load(copied)
    assert reopened.column_analysis().cohorts
    assert reopened.sample('7').subgraph_store.root_id == 7
    assert reopened.run_id == 'recorded-run'
    assert reopened.coverage == _complete().coverage


@pytest.fixture
def replay(tmp_path, monkeypatch):
    call = Mock(return_value=_complete())
    monkeypatch.setattr(lifecycle, 'explain_prediction', call)
    # Support lookup is independent of loading model weights or prediction data.
    monkeypatch.setattr(lifecycle, 'prediction_model_support',
                        lambda task, model: PredictionExplanationSupport(
                            task != TaskType.TEMPORAL_LINK_PREDICTION, True))
    return call


def _replay(root, **kwargs):
    return lifecycle.explain_run(
        run_root=str(root), run_id='actual-prediction-id',
        task_type=kwargs.pop('task_type', TaskType.REGRESSION),
        materialized=TaskDatasetMaterializeResult('manifest', 'runtime', 'bundle'),
        model_dir='published-model', **kwargs)


def test_disabled_replay_is_retained(tmp_path, replay):
    xai = _replay(tmp_path)
    replay.assert_not_called()
    assert xai.run_id == 'actual-prediction-id'
    assert all(state.reason == 'disabled' for state in xai.availability.values())
    assert PredictionExplanations.load(tmp_path).availability == xai.availability


def test_options_and_public_coverage_live_and_reopened(tmp_path, replay):
    options = PredictionExplanationOptions(cohort_rows=12, sample_rows=5)
    xai = _replay(tmp_path, explanations=options)
    assert replay.call_args.kwargs['options'] is options
    assert xai.coverage == _complete().coverage
    reopened = PredictionExplanations.load(tmp_path)
    assert reopened.coverage == xai.coverage
    assert reopened.run_id == 'actual-prediction-id'


def test_link_replay_failure_really_executes_producer(tmp_path, replay):
    replay.side_effect = RuntimeError('collector callback failed')
    xai = _replay(tmp_path, explanations=True,
                  task_type=TaskType.TEMPORAL_LINK_PREDICTION)
    replay.assert_called_once()
    reopened = PredictionExplanations.load(tmp_path)
    for result in (xai, reopened):
        assert result.availability['cohorts'].reason == 'unsupported'
        assert result.availability['samples'].reason == 'failed'
        assert 'collector callback failed' in result.availability[
            'samples'].generation_failure


@pytest.mark.parametrize('enabled', [False, True])
def test_initial_status_write_failure_is_explicit(tmp_path, replay, monkeypatch,
                                                  enabled):
    monkeypatch.setattr(lifecycle, 'write_record',
                        Mock(side_effect=OSError('disk full')))
    xai = _replay(tmp_path, explanations=enabled)
    replay.assert_not_called()
    assert xai.availability['cohorts'].reason == (
        'failed' if enabled else 'disabled')
    if enabled:
        assert 'disk full' in xai.availability['cohorts'].generation_failure


def test_final_write_failure_preserves_completed_live_outcomes(
        tmp_path, replay, monkeypatch):
    _cohorts(f'{tmp_path}/xai')
    _samples(f'{tmp_path}/xai')
    calls = []

    def write(root, record):
        calls.append(record.state)
        if record.state == 'complete':
            raise OSError('final write failed')
        write_record(root, record)

    monkeypatch.setattr(lifecycle, 'write_record', write)
    xai = _replay(tmp_path, explanations=True)
    assert calls == ['running', 'complete']
    assert xai.coverage == _complete().coverage
    assert xai.column_analysis().cohorts
    assert xai.sample('7').subgraph_store.root_id == 7
    reopened = PredictionExplanations.load(tmp_path)
    assert not any(state.available for state in reopened.availability.values())
    assert 'did not complete' in reopened.availability['samples'].generation_failure


def test_producer_read_permissions_are_independent(tmp_path, monkeypatch):
    _cohorts(f'{tmp_path}/xai')
    _samples(f'{tmp_path}/xai')
    xai = PredictionExplanations.from_run(tmp_path, enabled=True,
                                          result=_complete())
    with monkeypatch.context() as patch:
        patch.setattr('kumo.engine.explain.results.mapped_sample_ids',
                      Mock(side_effect=PermissionError('sample metadata denied')))
        assert xai.column_analysis().cohorts
    with monkeypatch.context() as patch:
        patch.setattr('kumo.engine.explain.results.read_cohorts',
                      Mock(side_effect=PermissionError('cohorts denied')))
        assert xai.sample('7').subgraph_store.root_id == 7


@pytest.mark.parametrize('change', [
    {'schema_version': 99}, {'schema_version': True}, {'schema_version': 1.0}, {'kind': 'training_explanations'}, {'run_id': ''},
    {'enabled': 'true'},
    {'result': {'cohorts': {'status': 'failed', 'error': None},
                'samples': {'status': 'unsupported'}}},
    {'result': {'cohorts': {'status': 'unsupported', 'error': 'failure'},
                'samples': {'status': 'unsupported'}}},
])
def test_invalid_status_rejected(tmp_path, change):
    record = _record(tmp_path, result=_complete())
    value = record.model_dump(mode='json')
    value.update(change)
    with filesystem.open_file(f'{tmp_path}/xai/manifest.json', 'wt') as target:
        json.dump(value, target)
    with pytest.raises(InvalidRunError):
        PredictionExplanations.load(tmp_path)


@pytest.mark.parametrize('coverage', [
    {'cohort_rows_processed': -1}, {'sampled_graphs': True},
    {'sampled_graphs': 1.5}, {'checkpoint_path': '../other'},
    {'checkpoint_path': '/absolute'}, {'checkpoint_path': 'file://checkpoint'},
])
def test_invalid_essential_coverage_rejected(tmp_path, coverage):
    record = _record(tmp_path, result=_complete()).model_dump(mode='json')
    record['result']['coverage'].update(coverage)
    with filesystem.open_file(f'{tmp_path}/xai/manifest.json', 'wt') as target:
        json.dump(record, target)
    with pytest.raises(InvalidRunError):
        PredictionExplanations.load(tmp_path)


def test_future_status_fields_are_ignored(tmp_path):
    record = _record(tmp_path, result=_complete()).model_dump(mode='json')
    record['future'] = {'new': True}
    record['result']['cohorts']['future'] = 1
    record['result']['coverage']['future'] = 1
    with filesystem.open_file(f'{tmp_path}/xai/manifest.json', 'wt') as target:
        json.dump(record, target)
    assert PredictionExplanations.load(tmp_path).coverage == _complete().coverage


@pytest.mark.parametrize('value', [None, 0, 1, 'yes'])
def test_prediction_flag_validated_first(value):
    from kumoapi.model_plan import ModelPlan

    with pytest.raises(ValueError, match='explanations must be a boolean'):
        Trainer(ModelPlan()).predict(explanations=value)


@pytest.mark.parametrize('field', ['cohort_rows', 'sample_rows'])
@pytest.mark.parametrize('value', [0, -1, True, False, 1.5, '10'])
def test_public_options_validate_limits(field, value):
    error = ValueError if type(value) is int else TypeError
    with pytest.raises(error, match='positive'):
        PredictionExplanationOptions(**{field: value})


@pytest.mark.parametrize('field', ['schema_version', 'kind'])
def test_missing_status_discriminator_rejected(tmp_path, field):
    record = _record(tmp_path, result=_complete()).model_dump(mode='json')
    del record[field]
    with filesystem.open_file(f'{tmp_path}/xai/manifest.json', 'wt') as target:
        json.dump(record, target)
    with pytest.raises(InvalidRunError):
        PredictionExplanations.load(tmp_path)


def test_unsupported_producers_survive_reopen(tmp_path, replay):
    replay.return_value = PredictionExplanationResult(
        ProducerOutcome('unsupported'), ProducerOutcome('unsupported'))
    xai = _replay(tmp_path, explanations=True)
    assert PredictionExplanations.load(tmp_path).availability == xai.availability
    assert all(state.reason == 'unsupported' for state in xai.availability.values())


def test_model_support_error_retains_known_link_eligibility(
        tmp_path, replay, monkeypatch):
    monkeypatch.setattr(lifecycle, 'prediction_model_support',
                        Mock(side_effect=OSError('config unavailable')))
    xai = _replay(tmp_path, explanations=True,
                  task_type=TaskType.TEMPORAL_LINK_PREDICTION)
    replay.assert_not_called()
    assert xai.availability['cohorts'].reason == 'unsupported'
    assert xai.availability['samples'].reason == 'failed'
    assert 'config unavailable' in PredictionExplanations.load(
        tmp_path).availability['samples'].generation_failure


def _ids(view):
    return [card['id'] for card in view.data['tabs'][0]['cards']]


def _live(run_id='recorded-run'):
    return PredictionManifest(
        id=run_id, location=f's3://root/predictions/{run_id}',
        parents=PredictionParents(
            model=ModelManifest(id='m', location='s3://root/models/m'),
            snapshot=SnapshotManifest(id='g@d', parents=SnapshotParents(
                source_tables={}))),
        prediction_table=PredictionTableRecord(
            query_sha256='q', source='pql', anchors=[], rows=2),
        scoring=RunScoring(
            binary_classification_threshold=None, threshold_source='none',
            num_classes_to_return=None, count_source='none', num_workers=None),
        outputs=[], status='succeeded', timings=None, stats=None)


@pytest.mark.parametrize('task', [
    TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION,
    TaskType.MULTICLASS_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION
])
def test_prediction_widget_shows_cohorts_and_subgraphs(tmp_path, task):
    cohort = _cohorts(f'{tmp_path}/xai')
    cohort.task_type = task
    labeled = task in (TaskType.MULTICLASS_CLASSIFICATION,
                       TaskType.MULTILABEL_CLASSIFICATION)
    if labeled:
        cohort.cohorts[0].predictions = {'blue': [.2, .8], 'red': [.8, .2]}
    save_json(cohort,
              f'{tmp_path}/xai/column_analysis/pred/column_analysis.json')
    _samples(f'{tmp_path}/xai', task)
    _record(tmp_path, result=_complete())
    xai = PredictionExplanations.load(tmp_path)
    view = PredictionExplainabilityView(xai, _live(), 'recorded-run')
    data = view.data
    assert data['kind'] == 'prediction' and len(data['tabs']) == 1
    assert data['header']['summary'] == '2 prediction rows'
    identity = dict(data['header']['identity'])
    assert identity['Prediction type'] == task.value.replace(
        '_', ' ').capitalize()
    assert identity['Batch'] == 'recorded-run' and 'Architecture' not in identity
    assert _ids(view) == ['groups', 'subgraphs']
    groups, graphs = data['tabs'][0]['cards']
    col = groups['columns'][groups['default']]
    col = col['by_class'][0] if labeled else col
    assert [g['predicted'] for g in col['groups']] == [.2, .8]
    assert 'actual' not in col['groups'][0] and 'average_actual' not in col
    assert graphs['samples'][0]['nodes'][0]['value'] == 'current-user'
    json.dumps(data, allow_nan=False)
    svg = view._repr_svg_()
    assert '<title>Batch explanations</title>' in svg
    assert 'Batch: recorded-run' in svg and 'Architecture' not in svg
    assert 'young' in svg and 'average actual' not in svg and 'current-user' not in svg
    assert 'To activate the widget, rerun this cell.' in view._snapshot.plain()
    assert 'kumo-explainability' in view.html()
    assert xai.visualize().data['tabs'] == data['tabs']
    # A live manifest of another batch is not shown under this run.
    other = PredictionExplainabilityView(xai, _live('other'), 'recorded-run')
    assert 'Model ID' not in dict(other.data['header']['identity'])


@pytest.mark.parametrize(
    'state', ['disabled', 'samples_only', 'unreadable_graph', 'missing',
              'malformed'])
@pytest.mark.parametrize('attached', ['loaded', 'unknown', 'complete'])
def test_prediction_widget_respects_producer_status(tmp_path, state, attached):
    """Status precedence and unreadable artifacts; producer outcomes are in
    test_retained_producer_status_matches_live."""
    _cohorts(f'{tmp_path}/xai')
    _samples(f'{tmp_path}/xai')
    result = _complete()
    if state in ('samples_only', 'unreadable_graph'):
        result = replace(result, cohorts=ProducerOutcome('unsupported'))
        _samples(f'{tmp_path}/xai', TaskType.LINK_PREDICTION)
    if state == 'unreadable_graph':
        # An unreadable graph ordered before the readable one.
        collector = tmp_path / 'xai/subgraph_collector/pred'
        save_json(EntityMappings(mappings={'first': ['3', '7']},
                                 orig_mappings={'first': ['x', 'y']}),
                  str(collector / 'strategy_mappings.json'))
        (collector / 'subgraph_json/3.json').write_text('not json')
    _record(tmp_path, enabled=state != 'disabled',
            result=None if state == 'disabled' else result)
    path = tmp_path / 'xai/column_analysis/pred/column_analysis.json'
    if state == 'missing':
        path.unlink()
    elif state == 'malformed':
        path.write_text('{}')
    source = (PredictionExplanations.load(tmp_path)
              if attached == 'loaded' else PredictionExplanations.from_run(
                  tmp_path, enabled=True,
                  result=_complete() if attached == 'complete' else None))
    view = source.visualize()
    ids = _ids(view)
    assert ids[0] == 'status' and 'groups' not in ids
    assert ('subgraphs' in ids) == (state != 'disabled')
    assert ('1 retained example subgraphs.'
            in view._snapshot.plain()) == ('subgraphs' in ids)


def test_prediction_widget_keeps_all_zero_variation_cohorts(tmp_path):
    artifact = _cohorts(f'{tmp_path}/xai')
    cohort = artifact.cohorts[0]
    cohort.cohorts = [f'cohort-{i}' for i in range(12)]
    cohort.populations = [1 / 12] * 12
    cohort.predictions = [0.] * 12
    cohort.prediction_score = 0.
    save_json(artifact,
              f'{tmp_path}/xai/column_analysis/pred/column_analysis.json')
    _record(tmp_path, result=_complete())
    view = PredictionExplanations.load(tmp_path).visualize()
    column = next(iter(view.data['tabs'][0]['cards'][-1]['columns'].values()))
    assert len(column['groups']) == 12 and column['average_predicted'] == 0.
    assert 'cohort-11' in view._repr_svg_()
    json.dumps(view.data, allow_nan=False)
