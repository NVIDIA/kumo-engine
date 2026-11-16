# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from kumoapi.common import StrEnum


class TaskType(StrEnum):
    BINARY_CLASSIFICATION = 'binary_classification'
    MULTICLASS_CLASSIFICATION = 'multiclass_classification'
    MULTILABEL_CLASSIFICATION = 'multilabel_classification'
    MULTILABEL_RANKING = 'multilabel_ranking'
    REGRESSION = 'regression'
    TEMPORAL_LINK_PREDICTION = 'temporal_link_prediction'
    STATIC_LINK_PREDICTION = 'static_link_prediction'
    FORECASTING = 'forecasting'

    LINK_PREDICTION = 'link_prediction'  # Deprecated.

    @staticmethod
    def get_node_pred_tasks() -> List['TaskType']:
        return [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.MULTILABEL_RANKING,
            TaskType.REGRESSION,
            TaskType.FORECASTING,
        ]

    @property
    def is_node_pred(self) -> bool:
        return self in self.get_node_pred_tasks()

    @staticmethod
    def get_link_pred_tasks() -> List['TaskType']:
        return [
            TaskType.TEMPORAL_LINK_PREDICTION,
            TaskType.STATIC_LINK_PREDICTION,
            TaskType.LINK_PREDICTION,
        ]

    @property
    def is_link_pred(self) -> bool:
        return self in self.get_link_pred_tasks()

    @staticmethod
    def get_classification_tasks() -> List['TaskType']:
        return [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]

    @property
    def is_classification(self) -> bool:
        return self in self.get_classification_tasks()

    @staticmethod
    def get_multilabel_tasks() -> List['TaskType']:
        return [
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.MULTILABEL_RANKING,
        ]

    @property
    def is_multilabel(self) -> bool:
        return self in self.get_multilabel_tasks()

    @staticmethod
    def get_ranking_tasks() -> List['TaskType']:
        return TaskType.get_link_pred_tasks() + [TaskType.MULTILABEL_RANKING]

    @property
    def is_ranking(self) -> bool:
        return self in self.get_ranking_tasks()
