# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from kumoapi.task import TaskType

from kumo_core.training.task_config import TaskConfig


@pytest.mark.parametrize('task_type', list(TaskType))
def test_task_config(task_type: TaskType) -> None:
    cfg = TaskConfig(task_type, name='TEST')

    if task_type == TaskType.BINARY_CLASSIFICATION:
        assert cfg.metrics == ['auroc', 'auprc', 'ap']
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        assert cfg.metrics == ['acc']
    if task_type == TaskType.MULTILABEL_CLASSIFICATION:
        assert cfg.metrics == [
            'ap_macro',
            'ap_micro',
            'ap_per_label',
            'auroc_macro',
            'auroc_micro',
            'auroc_per_label',
            'auprc_macro',
            'auprc_micro',
            'auprc_per_label',
        ]
    if task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
        assert cfg.metrics == ['mae', 'mse', 'rmse']
    if task_type in {
            TaskType.TEMPORAL_LINK_PREDICTION,
            TaskType.STATIC_LINK_PREDICTION,
            TaskType.MULTILABEL_RANKING,
    }:
        assert cfg.metrics == [
            'map@1',
            'map@10',
            'precision@1',
            'precision@10',
            'recall@1',
            'recall@10',
            'f1@1',
            'f1@10',
        ]
