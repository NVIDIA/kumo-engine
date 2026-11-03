# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumo_core.model_plan import ModelPlan as CoreModelPlan
from kumo_core.model_plan import TrainingTableGenerationPlan
from kumo_core.resolved_task import ResolvedNumericalTask
from kumo_core.types import ExplainMethod
from kumoapi.model_plan import ModelPlan

from kumo.engine import (
    ExplanationUnavailableError,
    Trainer,
    TrainingExplanations,
    TrainingTable,
)
from kumo.engine.explain import results as explanation_results


@pytest.mark.parametrize('disabled,explain_fails,aggregate_fails', [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (False, False, True),
])
@pytest.mark.parametrize('accelerator', ['cpu', 'gpu', 'mps', 'auto'])
@pytest.mark.parametrize('refitted_trainval', [False, True])
def test_fit_runs_shared_explanations_after_training(tmp_path, snapshot,
                                                     monkeypatch, caplog,
                                                     staged_seed_dir, disabled,
                                                     explain_fails,
                                                     aggregate_fails,
                                                     accelerator,
                                                     refitted_trainval):
    plan = ModelPlan()
    plan.training_job.disable_explain = disabled
    trainer = Trainer(plan, accelerator=accelerator)
    monkeypatch.setattr(type(trainer), 'scratch_dir',
                        property(lambda _: tmp_path))
    monkeypatch.setattr('kumo.engine.trainer.trainer.new_run_id',
                        lambda: 'explanation-test')
    table = TrainingTable(
        query='PREDICT users.label FOR EACH users.id',
        train_table_path=tmp_path / 'train.parquet',
        manifest_path=tmp_path / 'manifest.json',
        staging_dir=staged_seed_dir,
        required_stages=('train', 'val', 'test'),
        generation_plan=TrainingTableGenerationPlan(),
    )
    events = []
    run_root = tmp_path / 'runs' / f'{trainer._id}-fit-explanation-test'
    automl = run_root / 'models' / 'users' / 'automl'
    selected_trial = automl / 'version_4'
    selected_trial.mkdir(parents=True)
    (selected_trial / 'config.json').write_text(json.dumps({
        'refit_cfg': {'refitted_trainval': refitted_trainval},
    }))
    explanation_trial = (selected_trial.with_name('version_4_refit_trainval')
                         if refitted_trainval else selected_trial)
    explanation_trial.mkdir(exist_ok=True)

    def explain(**kwargs):
        events.append('explain')
        if explain_fails:
            raise RuntimeError('explanation failure')
        gradient_dir = explanation_trial / 'explain_test'
        gradient_dir.mkdir()
        (gradient_dir / 'scores.csv').write_text(
            'table,column,score\nusers,age,1.000000\n')

    best_trainer = SimpleNamespace(
        explain=Mock(side_effect=explain),
        subgraph_collector_dir=lambda: str(tmp_path / 'subgraph_collector'),
        log_dir=str(tmp_path),
        original_log_dir=str(selected_trial),
        refit_cfg=SimpleNamespace(refitted_trainval=refitted_trainval),
    )

    def aggregate_features(*args):
        events.append('aggregate')
        if aggregate_fails:
            raise RuntimeError('aggregation failure')

    aggregate = Mock(side_effect=aggregate_features)
    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.aggregate_subgraph_features', aggregate)

    def train(training_input, **kwargs):
        assert kwargs['bindings'].runtime.trainer_runtime_kwargs()[
            'accelerator'] == accelerator
        events.append('train')
        assert training_input.home_dir == str(run_root)
        (automl / 'best_trainer.json').write_text(json.dumps({
            'best_trainer': str(selected_trial),
        }))
        return SimpleNamespace(
            metrics={'test_accuracy': 0.8},
            artifact_paths={'selected_checkpoint': '/model/selected.ckpt'},
            explain_disabled=training_input.model_plan.training_job.
            disable_explain,
            best_trainer=best_trainer,
        )

    core_plan = CoreModelPlan()
    core_plan.training_job.disable_explain = disabled
    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.build_generated_training_task_dataset',
        lambda **kwargs: SimpleNamespace(feature_schema={'node_types': {}}))
    monkeypatch.setattr(
        'kumo_core.training.workflow.materialize_generated_training_task',
        lambda *args, **kwargs: SimpleNamespace(
            task=Mock(spec=ResolvedNumericalTask), model_plan=core_plan,
            bundle_dir=str(tmp_path)))
    monkeypatch.setattr(
        'kumo_core.training.orchestration.run_callable_training', train)
    monkeypatch.setattr('kumo.engine.trainer.trainer.write_training_spec',
                        lambda *args, **kwargs: events.append('finish'))

    with monkeypatch.context() as lazy_checks:
        for name in ('read_gradient', 'read_cohorts', 'read_sample',
                     'list_samples', 'resolve_run'):
            lazy_checks.setattr(
                explanation_results, name,
                Mock(side_effect=AssertionError('fit must not read artifacts')))
        result = trainer.fit(table, snapshot=snapshot)

    assert result.metrics == {'test_accuracy': 0.8}
    assert result.artifact_paths == {
        'selected_checkpoint': '/model/selected.ckpt'
    }
    assert trainer._training_result is result
    assert result.explanations.run_root == str(run_root)
    assert result.explanations.trial_root == str(explanation_trial)
    availability = result.explanations.availability
    if disabled or explain_fails:
        assert availability['gradient'].reason == ('disabled' if disabled else
                                                   'failed')
        with pytest.raises(ExplanationUnavailableError):
            result.explanations.gradient_scores()
    else:
        assert result.explanations.gradient_scores().tables[
            'users'].columns == {'age': 1.0}
        assert availability['gradient'].available
        assert not availability['cohorts'].available
        reopened = TrainingExplanations.load(run_root)
        assert reopened.trial_root == result.explanations.trial_root
        assert (reopened.gradient_scores() ==
                result.explanations.gradient_scores())
    if explain_fails or aggregate_fails:
        assert availability['gradient'].generation_failure is not None
    if disabled:
        assert events == ['train', 'finish']
        best_trainer.explain.assert_not_called()
    else:
        assert events == (['train', 'explain', 'finish'] if explain_fails else
                          ['train', 'explain', 'aggregate', 'finish'])
        # Leave fit-mode selection and supported-task checks to the shared
        # numerical Trainer.explain, just as the hosted training path does.
        best_trainer.explain.assert_called_once_with(
            explain_methods=[
                ExplainMethod.COLUMN_ANALYSIS, ExplainMethod.GRADIENT,
                ExplainMethod.SUBGRAPH_COLLECTOR
            ], stage='test')
    if disabled or explain_fails:
        aggregate.assert_not_called()
    else:
        from kumoapi.train import TrainingStage
        aggregate.assert_called_once_with(
            str(tmp_path / 'subgraph_collector' / 'test'), TrainingStage.TEST)
    if explain_fails or aggregate_fails:
        assert 'Training succeeded' in caplog.text
        assert ('explanation failure'
                if explain_fails else 'aggregation failure') in caplog.text
    else:
        assert 'post-training explanations failed' not in caplog.text


def test_load_model_does_not_search_for_training_explanations(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.read_bundle_artifact_paths', lambda _: {})
    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.read_training_spec',
        lambda _: SimpleNamespace(
            model_plan={}, snapshot_id='snapshot',
            predictive_query='PREDICT users.label FOR EACH users.id',
            training_table_generation={}))
    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.trainer_scratch_dir',
        Mock(side_effect=AssertionError('model load must not search scratch')))
    monkeypatch.setattr(
        explanation_results, 'resolve_run',
        Mock(side_effect=AssertionError('model load must not resolve a run')))

    restored = Trainer.load(tmp_path / 'published-model')

    explanations = restored._training_result.explanations
    assert explanations.run_root is None
    assert explanations.trial_root is None
    assert all(not artifact.available and artifact.reason == 'model_only'
               for artifact in explanations.availability.values())
    with pytest.raises(ExplanationUnavailableError, match='model_only'):
        explanations.gradient_scores()
