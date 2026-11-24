# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import kumo_core.modeling.context as modeling_context
import pytest
import torch
from kumo_core.explain.base import Explainer
from kumo_core.explain.column_analysis import ColumnAnalysis
from kumo_core.prediction_explanation_result import (
    PredictionExplanationOptions,
)
from kumo_core.train.base_module import BaseLightningModule
from kumo_core.train.trainer import Trainer
from kumo_core.training.config import FitMode, LinkPredModuleType
from kumo_core.types import ExplainMethod, TaskType
from kumoapi.explain.common import EntityMappings
from kumoapi.train import TrainingStage
from torch_geometric.data import HeteroData


class _Producer(Explainer):
    supported_task_types = {TaskType.STATIC_LINK_PREDICTION, TaskType.REGRESSION}
    requires_grad = True

    def __init__(self, task_type, root_dir):
        super().__init__(task_type, root_dir)
        self.pre_step_count = 0

    def explain_pre_step(self, batch, module, datamodule, stage, batch_idx):
        self.pre_step_count += 1


class _Keys:
    entity_table_names = ['users']

    def __getitem__(self, key):
        return key[2] + 100


class _Model(torch.nn.Module):
    is_explaining = BaseLightningModule.is_explaining
    on_predict_batch_start = BaseLightningModule.on_predict_batch_start
    on_predict_batch_end = BaseLightningModule.on_predict_batch_end

    def __init__(self):
        super().__init__()
        self.explainers = []
        self.outputs = None
        self.disable_eval_metrics = False
        self.eval_top_k = 7
        self.pred_top_k = 9
        self.valid_rhs_node_idx = torch.tensor([1, 3])
        self.fail = False

    def forward(self, batch):
        assert modeling_context.is_explaining() == bool(self.explainers)
        assert torch.is_grad_enabled() == any(
            producer.requires_grad for producer in self.explainers)
        if self.fail:
            raise RuntimeError('forward failed')
        return {'id': torch.tensor([5, 6])}

    def reset_embedding_hook(self):
        pass


class _Inference:
    limit_val_batches = 0.4
    limit_test_batches = 0.5
    limit_predict_batches = 0.6

    def __init__(self):
        self.fail_pass = None
        self.calls = 0

    def predict(self, *, model, ckpt_path, datamodule=None, dataloaders=None):
        self.calls += 1
        model.fail = self.fail_pass == self.calls
        model.trainer = SimpleNamespace(datamodule=datamodule)
        if datamodule is not None and datamodule.selected_rows is not None:
            dataloaders = datamodule.selected_rows
        batch = HeteroData()
        batch['users'].input_id = torch.tensor([0, 1])
        batch['users'].batch_size = (len(dataloaders)
                                     if isinstance(dataloaders, torch.Tensor)
                                     else 2)
        model.on_predict_batch_start(batch, 0)
        outputs = model(batch)
        model.on_predict_batch_end(outputs, batch, 0)
        return [outputs]

    validate = predict
    test = predict


class _Harness:
    explain = Trainer.explain
    _run_explain = Trainer._run_explain
    _explain_prediction = Trainer._explain_prediction
    _samples_requested = Trainer._samples_requested
    _collect_subgraphs = Trainer._collect_subgraphs
    _explain_runtime = Trainer._explain_runtime
    subgraph_collector_dir = Trainer.subgraph_collector_dir

    def __init__(self, root):
        self.fit_mode = FitMode.fit
        self.refit_cfg = SimpleNamespace(refitted_trainval=False,
                                         refitted_full=False)
        self._fully_initialized = True
        self.task = SimpleNamespace(task_type=TaskType.STATIC_LINK_PREDICTION)
        self.module_kwargs = SimpleNamespace(module=LinkPredModuleType.HYBRID)
        self.pl_model = _Model()
        self.pl_trainer = _Inference()
        self.datamodule = SimpleNamespace(
            eval_loader_kwargs={'batch_size': 16}, active_stage=None,
            selected_rows=None,
            begin_explain_selection_capture=self.begin_capture,
            clear_explain_selection_capture=self.clear_capture)
        self.best_ckpt = '/tmp/test-explain.ckpt'
        self.log_dir = str(root)
        self.feature_store = _Keys()
        self.explain_runtime = SimpleNamespace(
            max_batches=50, prediction_selection_nodes=1000, max_facts=16,
            mapping_file_name='strategy_mappings.json',
            get_prediction_inputs=self.prediction_inputs,
            prediction_rows=self.prediction_rows,
            get_explain_node_inputs=self.inputs,
            select_input_nodes=self.select,
            get_eval_dataloader=lambda *args: 'selected-loader')
        self.fail_selection = False

    def begin_capture(self, stage):
        self.datamodule.active_stage = stage

    def clear_capture(self):
        self.datamodule.active_stage = None

    def _check_readiness_for_prediction(self, fit_mode):
        pass

    def inputs(self, datamodule, stage):
        return torch.tensor([5, 6]), torch.tensor([10, 11])

    def prediction_inputs(self, datamodule, limit):
        if self.fail_selection:
            raise RuntimeError('selection failed')
        return self.inputs(datamodule, TrainingStage.PRED)

    @contextmanager
    def prediction_rows(self, datamodule, rows):
        previous = datamodule.selected_rows
        datamodule.selected_rows = rows
        try:
            yield
        finally:
            datamodule.selected_rows = previous

    def select(self, *args):
        if self.fail_selection:
            raise RuntimeError('selection failed')
        return torch.tensor([10, 11]), EntityMappings(
            mappings={'first': ['0', '1']}, orig_mappings={'first': [5, 6]})

    def set_explainers(self, stage, methods, root_dir=None,
                       collector_formats=None):
        self.pl_model.explainers.extend([
            _Producer(self.task.task_type, self.log_dir) for _ in methods
        ])


def test_failed_selected_replay_restores_caller_state(tmp_path):
    trainer = _Harness(tmp_path)
    sentinel = _Producer(trainer.task.task_type, str(tmp_path))
    original_explainers = [sentinel]
    trainer.pl_model.explainers = original_explainers
    trainer.pl_trainer.fail_pass = 2
    with pytest.raises(RuntimeError, match='forward failed'):
        trainer.explain([ExplainMethod.GRADIENT, ExplainMethod.SUBGRAPH_COLLECTOR],
                        stage=TrainingStage.PRED, top_k=4,
                        fit_mode_to_use=FitMode.refit_full,
                        valid_rhs_node_idx=torch.tensor([2]))
    assert sentinel.pre_step_count == 1
    assert trainer.pl_model.explainers is original_explainers
    assert original_explainers == [sentinel]
    assert trainer.fit_mode is FitMode.fit
    assert trainer.datamodule.selected_rows is None
    assert trainer.datamodule.eval_loader_kwargs['batch_size'] == 16
    assert trainer.pl_trainer.limit_predict_batches == 0.6
    assert trainer.pl_model.pred_top_k == 9
    assert not trainer.pl_model.disable_eval_metrics
    assert not modeling_context.is_explaining()


@pytest.mark.parametrize('stage', [TrainingStage.TEST, TrainingStage.VAL])
def test_eval_selection_failure_restores_loader(tmp_path, stage):
    trainer = _Harness(tmp_path)
    trainer.pl_model.outputs = {'id': torch.tensor([5, 6])}
    trainer.fail_selection = True
    with pytest.raises(RuntimeError, match='selection failed'):
        trainer.explain([ExplainMethod.SUBGRAPH_COLLECTOR], stage=stage, top_k=4)
    assert trainer.datamodule.active_stage is None
    assert trainer.datamodule.eval_loader_kwargs['batch_size'] == 16
    assert trainer.pl_trainer.limit_test_batches == 0.5
    assert trainer.pl_trainer.limit_val_batches == 0.4
    assert not modeling_context.is_explaining()


def test_explain_preserves_existing_context(tmp_path):
    trainer = _Harness(tmp_path)
    trainer.pl_trainer.fail_pass = 1
    with (modeling_context.explaining(), torch.no_grad()):
        with pytest.raises(RuntimeError, match='forward failed'):
            trainer.explain([ExplainMethod.GRADIENT], stage=TrainingStage.PRED)
        assert modeling_context.is_explaining()
        assert not torch.is_grad_enabled()


@pytest.mark.parametrize('oom_failures', [1, 4])
def test_explain_oom_restores_batch_size(tmp_path, monkeypatch, oom_failures):
    """CPU-injected OOM exercises retry cleanup without a CUDA requirement."""
    trainer = _Harness(tmp_path)
    original_forward = trainer.pl_model.forward
    forwards = 0

    def forward(batch):
        nonlocal forwards
        forwards += 1
        if forwards <= oom_failures:
            raise torch.cuda.OutOfMemoryError('injected OOM')
        return original_forward(batch)

    monkeypatch.setattr(trainer.pl_model, 'forward', forward)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    with torch.no_grad():
        if oom_failures == 4:
            with pytest.raises(torch.cuda.OutOfMemoryError, match='injected'):
                trainer.explain([ExplainMethod.GRADIENT],
                                stage=TrainingStage.PRED)
        else:
            trainer.explain([ExplainMethod.GRADIENT], stage=TrainingStage.PRED)
        assert trainer.datamodule.eval_loader_kwargs['batch_size'] == 16
        assert trainer.pl_trainer.limit_predict_batches == 0.6
        assert trainer.pl_model.explainers == []
        assert not trainer.pl_model.disable_eval_metrics
        assert trainer.datamodule.active_stage is None
        assert not modeling_context.is_explaining()
        assert not torch.is_grad_enabled()


class _Cohort(ColumnAnalysis):
    def explain_pre_step(self, batch, **kwargs):
        assert kwargs['datamodule'] is not None
        self.processed_rows += batch['users'].batch_size

    def explain_post_step(self, **kwargs):
        pass


@pytest.mark.parametrize('cohort_rows', [None, 3])
def test_prediction_graph_budget_is_independent(tmp_path, cohort_rows):
    trainer = _Harness(tmp_path)
    trainer.task.task_type = TaskType.REGRESSION
    received = []

    def producers(stage, methods, **kwargs):
        for method in methods:
            producer = (_Cohort(TaskType.REGRESSION, str(tmp_path))
                        if method == ExplainMethod.COLUMN_ANALYSIS else
                        _Producer(TaskType.REGRESSION, str(tmp_path)))
            trainer.pl_model.explainers.append(producer)

    @contextmanager
    def selected_rows(datamodule, rows):
        received.append(rows.tolist())
        with trainer.prediction_rows(datamodule, rows):
            yield

    nodes = torch.tensor([5, 5, 7, 8, 9, 10, 11])
    row_ids = torch.tensor([20, 5, 11, 19, 25, 29, 40])
    trainer.set_explainers = producers
    trainer.explain_runtime.get_prediction_inputs = lambda dm, limit: (
        nodes[:limit], row_ids[:limit])
    trainer.explain_runtime.prediction_rows = selected_rows
    outcomes = trainer.explain(
        [ExplainMethod.COLUMN_ANALYSIS, ExplainMethod.SUBGRAPH_COLLECTOR],
        stage=TrainingStage.PRED, options=PredictionExplanationOptions(
            cohort_rows=cohort_rows, sample_rows=5))
    expected_cohorts = 3 if cohort_rows is not None else 7
    assert len(received[0]) == expected_cohorts
    assert received[0] == row_ids[:expected_cohorts].tolist()
    assert received[1] == [5, 11, 19, 20, 25]
    assert outcomes[0].processed_rows == expected_cohorts


@pytest.mark.parametrize('batch_size,expected_limit', [
    (16, 1000), (44, 1001), (128, 1600),
])
def test_default_cohort_budget_rounds_to_batches(tmp_path, batch_size,
                                                 expected_limit):
    trainer = _Harness(tmp_path)
    trainer.datamodule.eval_loader_kwargs['batch_size'] = batch_size
    limits = []

    def inputs(datamodule, limit):
        limits.append(limit)
        return torch.tensor([5]), torch.tensor([20])

    trainer.explain_runtime.get_prediction_inputs = inputs
    trainer.explain([ExplainMethod.GRADIENT], stage=TrainingStage.PRED)
    assert limits == [expected_limit]


def test_unsupported_prediction_skips_all_inference(tmp_path):
    trainer = _Harness(tmp_path)
    trainer.module_kwargs.module = LinkPredModuleType.TWO_TOWER
    assert trainer.explain([ExplainMethod.SUBGRAPH_COLLECTOR],
                           stage=TrainingStage.PRED) == ()
    assert trainer.pl_trainer.calls == 0
