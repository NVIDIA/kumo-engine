# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from kumo_core.types import TaskType
from kumoapi.typing import Dtype

from rest.backends.ml_backend.data_util import (
    get_max_partition_size_with_top_k,
)


def test_get_max_partition_size_with_top_k():
    task = SimpleNamespace(
        name='temporal-link-prediction',
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        target_to_entity_paths=[[], [('lhs', 'key', 'rhs')]],
        graph=SimpleNamespace(table=lambda _name: SimpleNamespace(
            pkey=SimpleNamespace(dtype=Dtype.string))),
    )
    max_rows = get_max_partition_size_with_top_k(
        max_partition_size=100_000_000,
        max_rows_per_partition=300_000_000,
        top_k=500,
        task=task,
    )
    assert max_rows == 37500
