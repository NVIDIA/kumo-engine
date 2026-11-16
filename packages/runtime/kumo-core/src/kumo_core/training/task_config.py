# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import field
from itertools import product

from kumoapi.task import TaskType
from pydantic.dataclasses import dataclass
from torch_geometric.typing import EdgeType


@dataclass
class TaskConfig:
    r"""Task-specific properties for training."""
    task_type: TaskType
    name: str
    metrics: list[str] = field(default_factory=list)
    target_to_entity_paths: list[list[EdgeType]] | None = field(
        default_factory=list)
    # Static prediction parameters:
    exclude_cols_dict: dict[str, list[str]] = field(default_factory=dict)
    # Predictive query configuration:
    unix_start_offset: int | None = None
    unix_end_offset: int | None = None
    unix_step_size: int | None = None
    # Forecasting parameters:
    forecast_length: int = 1
    lag_timesteps: int = 0

    def __post_init__(self) -> None:
        if len(self.metrics) == 0:
            if self.task_type == TaskType.BINARY_CLASSIFICATION:
                self.metrics = ['auroc', 'auprc', 'ap']
            elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                self.metrics = ['acc']
            elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
                names = ['ap', 'auroc', 'auprc']
                averages = ['macro', 'micro', 'per_label']
                self.metrics = [
                    f'{name}_{average}'
                    for name, average in product(names, averages)
                ]
            elif self.task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
                self.metrics = ['mae', 'mse', 'rmse']
            elif self.task_type in {
                    TaskType.TEMPORAL_LINK_PREDICTION,
                    TaskType.STATIC_LINK_PREDICTION,
                    TaskType.MULTILABEL_RANKING,
            }:
                names = ['map', 'precision', 'recall', 'f1']
                ks = [1, 10]
                self.metrics = [
                    f'{name}@{k}' for name, k in product(names, ks)
                ]
