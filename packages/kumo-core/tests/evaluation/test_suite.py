# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
