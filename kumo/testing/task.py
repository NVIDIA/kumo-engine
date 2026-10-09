# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from kumo_core.types import TaskType

from kumo.testing.table import MockTrainTable


# TODO(dong) make it able to fake data based on task config.
class MockTask:
    """A helper class that makes it possible to write tests
    that depend on time, by mocking out training tabdatetime.now().
    """
    def __init__(self, task_type: TaskType):
        # TODO(dong) setup MockTask inherits from Task to reuse attributes.
        self.name = 'mock_task'
        self.task_type = task_type

        self.exclude_cols = {}
        self.use_lp_shallow_rhs = False
        self.use_lp_id_awareness = False
        self.use_lp_hybrid = False

        self.train_table = MockTrainTable(task_type)
        self.graph = MagicMock()
        self.tables = []
