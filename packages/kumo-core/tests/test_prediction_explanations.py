# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import kumo_core.prediction_explanations as explanations
import pytest
from kumo_core.explain.column_analysis import ColumnAnalysis
from kumo_core.explain.subgraph_collector import SubgraphCollector
from kumo_core.io.json import save_json
from kumo_core.prediction_explanation_result import (
    PredictionExplanationCoverage,
    PredictionExplanationOptions,
)
from kumo_core.resolved_task import ResolvedNumericalTask
from kumo_core.task_dataset import TaskDatasetMaterializeResult
from kumo_core.training.config import (
    LinkPredModuleKwargs,
    LinkPredModuleType,
    TrainerConfig,
)
from kumo_core.types.enums import TaskType
from kumoapi.explain.column_analysis import ColumnAnalysisOutput


@pytest.mark.parametrize('field', ['cohort_rows', 'sample_rows'])
@pytest.mark.parametrize('value', [True, False, 1.0, '5', 0, -1])
def test_invalid_explanation_budget(field, value):
    error = ValueError if type(value) is int else TypeError
    with pytest.raises(error, match=field):
        PredictionExplanationOptions(**{field: value})


@pytest.mark.parametrize('task_type,module,expected', [
    (TaskType.REGRESSION, None, (True, True)),
    (TaskType.BINARY_CLASSIFICATION, None, (True, True)),
    (TaskType.MULTICLASS_CLASSIFICATION, None, (True, True)),
    (TaskType.MULTILABEL_CLASSIFICATION, None, (True, True)),
    (TaskType.STATIC_LINK_PREDICTION, LinkPredModuleType.HYBRID, (False, True)),
    (TaskType.TEMPORAL_LINK_PREDICTION, LinkPredModuleType.ID, (False, True)),
    (TaskType.STATIC_LINK_PREDICTION, LinkPredModuleType.SHALLOW_RHS, (False, True)),
    (TaskType.STATIC_LINK_PREDICTION, LinkPredModuleType.TWO_TOWER, (False, False)),
    (TaskType.FORECASTING, None, (False, False)),
])
def test_prediction_support(task_type, module, expected):
    support = explanations.prediction_explanation_support(task_type, module)
    assert (support.cohorts, support.samples) == expected


@pytest.fixture
def replay(tmp_path, monkeypatch):
    monkeypatch.setattr(explanations.DiskgraphTaskManifest, 'load',
                        Mock(return_value=SimpleNamespace(task_type='regression')))
    monkeypatch.setattr(ResolvedNumericalTask, 'from_manifest', Mock())
    trainer = Mock(explanation_producers=(),
                   best_ckpt=str(tmp_path / 'model/checkpoints/best.ckpt'))
    load = Mock(return_value=trainer)
    monkeypatch.setattr(explanations.DiskgraphNestedTrainer, 'load', load)
    output = str(tmp_path / 'xai')

    def run(**kwargs):
        return explanations.explain_prediction(
            materialized=TaskDatasetMaterializeResult('manifest', 'runtime', 'bundle'),
            model_dir=str(tmp_path / 'model'), run_id='run', output_dir=output,
            encoders=Mock(), **kwargs)

    return trainer, load, output, run


def _write_artifacts(output, *, partial=False):
    save_json(ColumnAnalysisOutput(task_type=TaskType.REGRESSION),
              f'{output}/column_analysis/pred/column_analysis.json')
    ids = ['0', '1'] if partial else ['0']
    save_json({'mappings': {'first': ids}},
              f'{output}/subgraph_collector/pred/strategy_mappings.json')
    save_json({}, f'{output}/subgraph_collector/pred/subgraph_json/0.json')


def _producers(output):
    cohort = ColumnAnalysis(TaskType.REGRESSION, output)
    sample = SubgraphCollector(TaskType.REGRESSION, output, ['nested_json'])
    return cohort, sample


def test_prediction_requires_producer_completion(replay):
    trainer, _, output, run = replay
    producers = _producers(output)
    trainer.explain.return_value = producers
    _write_artifacts(output)
    result = run()
    assert result.cohorts.status == result.samples.status == 'failed'
    assert 'did not complete' in result.failure
    # Each sampled graph keeps its recorded prediction beside it.
    assert trainer.explain.call_args.kwargs['collector_formats'] == [
        'nested_json', 'output']


@pytest.mark.parametrize('failed', ['cohorts', 'samples'])
def test_prediction_outcomes_are_independent(replay, failed):
    trainer, _, output, run = replay
    cohort, sample = _producers(output)
    cohort.completed = sample.completed = True
    if failed == 'cohorts':
        cohort.failure = 'feature batch failed'
    else:
        sample.failure = 'graph serialization failed'
    trainer.explain.return_value = (cohort, sample)
    _write_artifacts(output)
    result = run()
    assert result.cohorts.status == ('failed' if failed == 'cohorts' else 'complete')
    assert result.samples.status == ('failed' if failed == 'samples' else 'complete')


def test_completed_cohorts_survive_graph_forward_error(replay):
    trainer, _, output, run = replay
    cohort, sample = _producers(output)
    cohort.completed = True
    trainer.explanation_producers = (cohort, sample)
    trainer.explain.side_effect = RuntimeError('graph forward failed')
    _write_artifacts(output)
    result = run()
    assert result.cohorts.status == 'complete'
    assert result.samples.status == 'failed'
    assert 'graph forward failed' in result.samples.error


def test_observed_coverage_survives_partial_failure(replay):
    trainer, _, output, run = replay
    cohort, sample = _producers(output)
    cohort.completed = True
    cohort.processed_rows = 3
    trainer.explanation_producers = (cohort, sample)
    trainer.explain.side_effect = RuntimeError('graph forward failed')
    _write_artifacts(output, partial=True)
    result = run()
    assert result.coverage == PredictionExplanationCoverage(
        cohort_rows_processed=3, sampled_graphs=1,
        checkpoint_path='checkpoints/best.ckpt')
    assert result.cohorts.status == 'complete'
    assert result.samples.status == 'failed'


def test_unresolved_checkpoint_preserves_failure(replay):
    trainer, _, _, run = replay
    trainer.predict_setup.side_effect = RuntimeError('initialization failed')
    trainer.best_ckpt = None
    result = run()
    assert result.coverage.checkpoint_path is None
    assert 'initialization failed' in result.failure


def test_sample_inventory_reads_mapping_and_graphs_once(replay, monkeypatch):
    trainer, _, output, run = replay
    cohort, sample = _producers(output)
    cohort.completed = sample.completed = True
    trainer.explain.return_value = (cohort, sample)
    _write_artifacts(output)
    save_json({'mappings': {'first': ['0'], 'random': ['0']}},
              f'{output}/subgraph_collector/pred/strategy_mappings.json')
    original_load = explanations.load_json
    loads = []
    stats = []
    original_exists = explanations.filesystem.exists

    def load(path, schema):
        loads.append(path)
        return original_load(path, schema)

    def exists(path):
        stats.append(path)
        return original_exists(path)

    monkeypatch.setattr(explanations, 'load_json', load)
    monkeypatch.setattr(explanations.filesystem, 'exists', exists)
    result = run()
    assert result.coverage.sampled_graphs == 1
    assert sum(path.endswith('strategy_mappings.json') for path in loads) == 1
    assert sum(path.endswith('subgraph_json/0.json') for path in stats) == 1


def test_unobserved_counts_are_none(replay):
    trainer, _, _, run = replay
    trainer.predict_setup.side_effect = RuntimeError('initialization failed')
    result = run()
    assert result.coverage.cohort_rows_processed is None
    assert result.coverage.sampled_graphs is None


def test_empty_cohort_population_is_observed_zero(replay):
    trainer, _, output, run = replay
    cohort, sample = _producers(output)
    cohort.completed = sample.completed = True
    trainer.explain.return_value = (cohort, sample)
    _write_artifacts(output)
    assert run().coverage.cohort_rows_processed == 0


def test_checkpoint_uri_is_retained_as_relative_path(replay, tmp_path):
    trainer, _, _, run = replay
    trainer.best_ckpt = f'file://{tmp_path}/model/checkpoints/best.ckpt'
    trainer.explain.return_value = ()
    assert run().coverage.checkpoint_path == 'checkpoints/best.ckpt'


@pytest.mark.parametrize('partial', [False, True])
def test_missing_outputs_report_paths(replay, partial):
    trainer, _, output, run = replay
    cohort, sample = _producers(output)
    cohort.completed = sample.completed = True
    trainer.explain.return_value = (cohort, sample)
    if partial:
        _write_artifacts(output, partial=True)
    result = run()
    assert result.samples.status == 'failed'
    assert output in result.samples.error
    if partial:
        assert '1 of 2 sampled graphs missing' in result.samples.error
    else:
        assert 'Missing cohort artifact' in result.cohorts.error
        assert output in result.cohorts.error


@pytest.mark.parametrize('threshold,expected', [(None, 0.83), (0.73, 0.73)])
def test_prediction_uses_saved_threshold(replay, threshold, expected):
    trainer, load, _, run = replay
    trainer.explain.return_value = ()
    observed = []

    def load_model(*args, binary_threshold=0.83, **kwargs):
        observed.append(binary_threshold)
        return trainer

    load.side_effect = load_model
    run(binary_threshold=threshold)
    assert observed == [expected]


@pytest.mark.parametrize('module,expected', [
    (LinkPredModuleType.HYBRID, 'failed'),
    (LinkPredModuleType.TWO_TOWER, 'unsupported'),
])
def test_load_error_retains_support(replay, tmp_path, monkeypatch, module, expected):
    _, load, _, run = replay
    monkeypatch.setattr(explanations.DiskgraphTaskManifest, 'load', Mock(
        return_value=SimpleNamespace(task_type='static_link_prediction')))
    save_json(TrainerConfig(encoder=None, model=None, head=None,
                            module_kwargs=LinkPredModuleKwargs(module=module)),
              str(tmp_path / 'model/config.json'))
    load.side_effect = RuntimeError('checkpoint load failed')
    result = run()
    assert result.cohorts.status == 'unsupported'
    assert result.samples.status == expected


@pytest.mark.parametrize('failed', ['cohorts', 'samples'])
def test_artifact_probe_errors_preserve_healthy_producer(replay, monkeypatch,
                                                         failed):
    trainer, _, output, run = replay
    cohort, sample = _producers(output)
    cohort.completed = sample.completed = True
    trainer.explain.return_value = (cohort, sample)
    _write_artifacts(output)
    exists = explanations.filesystem.exists
    suffix = ('column_analysis.json' if failed == 'cohorts'
              else 'strategy_mappings.json')

    def probe(path):
        if path.endswith(suffix):
            raise PermissionError('artifact access denied')
        return exists(path)

    monkeypatch.setattr(explanations.filesystem, 'exists', probe)
    result = run()
    assert result.cohorts.status == ('failed' if failed == 'cohorts' else 'complete')
    assert result.samples.status == ('failed' if failed == 'samples' else 'complete')
    assert 'artifact access denied' in result.failure
