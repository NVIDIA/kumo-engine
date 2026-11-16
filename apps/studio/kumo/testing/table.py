# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import torch
from kumo_core.types import TaskType

from kumo.graph import ForeignKey


# TODO(dong) make it able to fake data based on table config.
class MockTrainTable:
    """A helper class that makes it possible to write tests
    that depend on training table.
    """
    def __init__(self, task_type: TaskType):
        self.name = task_type.name

        if task_type.is_link_pred:
            self.fkeys = [
                ForeignKey('pkey_0', 'dim_table_0'),
                ForeignKey('pkey_1', 'dim_table_1'),
            ]
        else:
            self.fkeys = [ForeignKey('pkey_0', 'dim_table_0')]

        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self._setup_classification_mock_data()

        self.encoders = None

    def _setup_classification_mock_data(self):
        self.data = MagicMock()
        self.data.fkeys = torch.arange(8)
        self.data.train_mask = torch.BoolTensor(
            [True, True, True, True, True, True, False, False])
        self.data.val_mask = torch.BoolTensor(
            [False, False, False, False, False, True, False])
        self.data.test_mask = torch.BoolTensor(
            [False, False, False, False, False, False, True])
