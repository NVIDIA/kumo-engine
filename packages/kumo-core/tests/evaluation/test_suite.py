# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
import torchmetrics
from kumoapi.encoder import Index
from kumoapi.task import TaskType
from kumoapi.train import TrainingStage
from kumoapi.typing import ColStatType, Stype

from kumo_core.evaluation.suite import MetricStore
from kumo_core.encoders.target.base import TargetEncoder


def test_tune_mode() -> None:
    # Loss
    assert MetricStore.tune_mode('loss') == 'min'

    # Classification:
    assert MetricStore.tune_mode('acc') == 'max'
    assert MetricStore.tune_mode('f1') == 'max'
    assert MetricStore.tune_mode('precision') == 'max'
    assert MetricStore.tune_mode('recall') == 'max'

    # Binary classification:
    assert MetricStore.tune_mode('auprc') == 'max'
    assert MetricStore.tune_mode('ap') == 'max'
    assert MetricStore.tune_mode('auroc') == 'max'
    assert MetricStore.tune_mode('ndcg') == 'max'
    assert MetricStore.tune_mode('recall@3') == 'max'
    assert MetricStore.tune_mode('precision@3') == 'max'
    assert MetricStore.tune_mode('ndcg@3') == 'max'

    # Multi-label classification:
    assert MetricStore.tune_mode('auprc_macro') == 'max'
    assert MetricStore.tune_mode('auprc_micro') == 'max'
    assert MetricStore.tune_mode('auprc_per_label') == 'max'
    assert MetricStore.tune_mode('auroc_macro') == 'max'
    assert MetricStore.tune_mode('auroc_micro') == 'max'
    assert MetricStore.tune_mode('auroc_per_label') == 'max'
    assert MetricStore.tune_mode('ap_macro') == 'max'
    assert MetricStore.tune_mode('ap_micro') == 'max'
    assert MetricStore.tune_mode('ap_per_label') == 'max'

    # Regression:
    assert MetricStore.tune_mode('mae') == 'min'
    assert MetricStore.tune_mode('mape') == 'min'
    assert MetricStore.tune_mode('mse') == 'min'
    assert MetricStore.tune_mode('rmse') == 'min'
    assert MetricStore.tune_mode('smape') == 'min'
    assert MetricStore.tune_mode('wape') == 'min'
    assert MetricStore.tune_mode('r2') == 'max'

    # Link prediction:
    assert MetricStore.tune_mode('map@3') == 'max'
    assert MetricStore.tune_mode('precision@3') == 'max'
    assert MetricStore.tune_mode('recall@3') == 'max'
    assert MetricStore.tune_mode('ndcg@3') == 'max'
    assert MetricStore.tune_mode('f1@3') == 'max'
    assert MetricStore.tune_mode('mrr@3') == 'max'


@pytest.mark.parametrize('task_type,metrics', [
    (TaskType.BINARY_CLASSIFICATION,
     ['auroc', 'ap', 'ndcg', 'recall@3', 'precision@3', 'ndcg@3']),
    (TaskType.MULTICLASS_CLASSIFICATION, ['acc', 'precision', 'recall', 'f1']),
    (TaskType.MULTILABEL_CLASSIFICATION, [
        'auprc_macro', 'auprc_micro', 'auprc_per_label', 'auroc_macro',
        'auroc_micro', 'auroc_per_label', 'ap_macro', 'ap_micro',
        'ap_per_label'
    ]),
    (TaskType.MULTILABEL_RANKING,
     ['map@3', 'ndcg@3', 'mrr@3', 'precision@3', 'recall@3', 'f1@3']),
    (TaskType.REGRESSION,
     ['mae', 'mape', 'mse', 'rmse', 'smape', 'wape', 'r2']),
    (TaskType.TEMPORAL_LINK_PREDICTION,
     ['map@3', 'precision@3', 'recall@3', 'ndcg@3', 'f1@3', 'mrr@3']),
    (TaskType.STATIC_LINK_PREDICTION,
     ['map@3', 'precision@3', 'recall@3', 'ndcg@3', 'f1@3', 'mrr@3']),
])
def test_valid_metrics(task_type: TaskType, metrics: list[str]) -> None:
    MetricStore.validate_metrics(task_type, metrics)


@pytest.mark.parametrize('task_type,metrics', [
    (TaskType.BINARY_CLASSIFICATION, ['mse']),
    (TaskType.REGRESSION, ['auroc']),
    (TaskType.STATIC_LINK_PREDICTION, ['f1']),
    (TaskType.MULTILABEL_CLASSIFICATION, ['mse']),
])
def test_invalid_metrics(task_type: TaskType, metrics: list[str]) -> None:
    with pytest.raises(ValueError):
        MetricStore.validate_metrics(task_type, metrics)


def test_metric_store() -> None:
    task_type = TaskType.BINARY_CLASSIFICATION
    encoder = Index()
    encoder._stats = {
        ColStatType.CATEGORY_COUNTS: [['A', 1], ['B', 1]],
    }
    metrics = ['auroc', 'ap', 'ndcg', 'recall@3', 'precision@3', 'ndcg@3']

    metric_store = MetricStore(
        task_type=task_type,
        target_encoder=TargetEncoder.from_encoder(task_type, encoder),
    ).init_metrics(metrics, module=torch.nn.Module())
    assert metric_store.is_initialized
    assert len(metric_store._metric_collections) == 3

    for stage in TrainingStage.list_stages():
        m = metric_store.get_metric_collection(stage)
        assert isinstance(m, torchmetrics.MetricCollection)
        assert set(m.keys()) == {f'{stage}_{metric}' for metric in metrics}


@pytest.mark.parametrize('device', ['cpu', 'cuda', 'mps'])
@pytest.mark.parametrize('stage', TrainingStage.list_stages())
def test_large_classification_metrics(
    device: str,
    stage: TrainingStage,
) -> None:
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')

    generator = torch.Generator().manual_seed(17)
    # More than 2**16 distinct thresholds triggers the MPS padding bug.
    pred = torch.rand(70_257, generator=generator)
    target = torch.randint(2, (len(pred), ), generator=generator)
    encoder = Index()
    encoder._stats = {ColStatType.CATEGORY_COUNTS: [['A', 1], ['B', 1]]}
    target_encoder = TargetEncoder.from_encoder(
        TaskType.BINARY_CLASSIFICATION, encoder)

    def make_store(device: str) -> MetricStore:
        return MetricStore(
            TaskType.BINARY_CLASSIFICATION, target_encoder).init_metrics(
                ['ap', 'auroc', 'auprc'], torch.nn.Module()).to(device)

    reference = make_store('cpu').get_metric_collection(stage)
    reference.update(pred, target)
    expected = reference.compute()
    store = make_store(device)
    for _ in range(2):
        for batch_pred, batch_target in zip(pred.split(512),
                                           target.split(512)):
            metric = store.get_metric_collection(stage)
            metric.update(batch_pred.to(device), batch_target.to(device))

        metric = store.get_metric_collection(stage)
        for member in metric.values(copy_state=False):
            assert member.compute_on_cpu == (device == 'mps')
            assert member.preds[0].device.type == (
                'cpu' if device == 'mps' else device)
        for name, value in metric.compute().items():
            torch.testing.assert_close(value.cpu(), expected[name])
        metric.reset()


@pytest.mark.parametrize('device', ['cpu', 'cuda', 'mps'])
def test_large_multilabel_metrics(device: str) -> None:
    from kumo_core.train.base_module import BaseLightningModule

    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')

    encoder = Index()
    encoder._stats = {ColStatType.CATEGORY_COUNTS: [['A', 1], ['B', 1]]}
    target_encoder = TargetEncoder.from_encoder(
        TaskType.MULTILABEL_CLASSIFICATION, encoder)
    generator = torch.Generator().manual_seed(17)
    pred = torch.randn(70_257, 2, generator=generator)
    target = torch.randint(2, pred.shape, generator=generator)
    expected = None
    for metric_device in ['cpu', device]:
        store = MetricStore(
            TaskType.MULTILABEL_CLASSIFICATION, target_encoder).init_metrics(
                ['ap_macro', 'auroc_macro', 'auprc_macro'],
                torch.nn.Module()).to(metric_device)
        for batch_pred, batch_target in zip(pred.split(512),
                                           target.split(512)):
            store.get_metric_collection(TrainingStage.TRAIN).update(
                batch_pred.to(metric_device), batch_target.to(metric_device))
        module = SimpleNamespace(
            metric_store=store,
            task_type=TaskType.MULTILABEL_CLASSIFICATION,
            max_multilabel_samples=100_000_000,
            device=torch.device(metric_device),
            is_fitting=True,
            log_epoch=lambda *args: None,
        )
        actual = BaseLightningModule.compute_metrics(module,
                                                    TrainingStage.TRAIN)
        if expected is None:
            expected = actual
        else:
            for name, value in actual.items():
                torch.testing.assert_close(value.cpu(), expected[name])


def test_validate_diversity() -> None:
    stype_dict = {
        'users': {},
        'articles': {
            'NUM': Stype.numerical,
            'CAT': Stype.categorical,
        },
    }

    with pytest.raises(ValueError, match=r"does not point to a column"):
        MetricStore.validate_metrics(
            TaskType.TEMPORAL_LINK_PREDICTION,
            ['diversity@10'],
            stype_dict,
            entity_table_names=['users', 'articles'],
        )

    with pytest.raises(ValueError, match=r"not point to a categorical column"):
        MetricStore.validate_metrics(
            TaskType.TEMPORAL_LINK_PREDICTION,
            ['diversity[FOO]@10'],
            stype_dict,
            entity_table_names=['users', 'articles'],
        )

    with pytest.raises(ValueError, match=r"not point to a categorical column"):
        MetricStore.validate_metrics(
            TaskType.TEMPORAL_LINK_PREDICTION,
            ['diversity[NUM]@10'],
            stype_dict,
            entity_table_names=['users', 'articles'],
        )

    MetricStore.validate_metrics(
        TaskType.TEMPORAL_LINK_PREDICTION,
        ['diversity[CAT]@10'],
        stype_dict,
        entity_table_names=['users', 'articles'],
    )
