# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumo_core.types import ColStatType, PredTime, TaskType, TaskTypeWithReason
from kumo_core.types.conversions import can_cast_safely
from kumo_core.types.edges import EdgeKey
from kumoapi.typing import Dtype
from pydantic import TypeAdapter


@pytest.mark.parametrize(
    ("raw_dtype", "destination", "expected"),
    [
        (np.dtype("int64"), Dtype.float32, True),
        (pd.Float64Dtype(), Dtype.int32, True),
        (pa.string(), Dtype.binary, True),
        (torch.bool, Dtype.bool, True),
    ],
)
def test_dtype_safe_cast_normalizes_supported_library_dtypes(
    raw_dtype: object,
    destination: Dtype,
    expected: bool,
) -> None:
    assert can_cast_safely(destination, raw_dtype) is expected


def test_dtype_safe_cast_preserves_unsupported_dtype_error() -> None:
    with pytest.raises(KeyError, match="Unsupported data type"):
        can_cast_safely(Dtype.string, object())


def test_category_counts_do_not_support_timedelta() -> None:
    assert Dtype.timedelta.is_numerical()
    assert not ColStatType.CATEGORY_COUNTS.supports(Dtype.timedelta)


def test_task_type_legacy_values_and_invalid_input_behavior() -> None:
    assert TaskType("classification") is TaskType.MULTICLASS_CLASSIFICATION
    assert TaskType("multilabel") is TaskType.MULTILABEL_CLASSIFICATION
    assert TaskType("logistic_regression") is TaskType.INVALID

    with pytest.raises(AttributeError):
        TaskType(1)


def test_task_type_with_reason_validates_and_serializes() -> None:
    result = TaskTypeWithReason(
        task_type="regression",
        reason="numeric target",
    )

    assert result.task_type is TaskType.REGRESSION
    assert TypeAdapter(TaskTypeWithReason).dump_python(
        result,
        mode="json",
    ) == {
        "task_type": "regression",
        "reason": "numeric target",
    }


def test_pred_time_remains_a_plain_enum_with_json_value() -> None:
    assert not isinstance(PredTime.MAX_TRAIN, str)
    assert TypeAdapter(PredTime).dump_python(
        PredTime.MAX_TRAIN,
        mode="json",
    ) == "max_train"


def test_edge_key_valid_behavior_and_serialization() -> None:
    edge = EdgeKey("events", "session", "sessions")
    reverse = edge.reverse()

    assert reverse.to_str() == "sessions__rev_session__events"
    assert reverse.to_tuple() == ("sessions", "rev_session", "events")
    assert reverse.reverse() == edge
    assert dataclasses.asdict(reverse) == {
        "src_table": "sessions",
        "link_key": "rev_session",
        "dst_table": "events",
        "is_reverse": True,
    }


def test_edge_key_reverse_removes_only_the_exact_prefix() -> None:
    reverse = EdgeKey("events", "ever", "sessions", is_reverse=True)

    assert reverse.link_key == "rev_ever"
    assert reverse.reverse().link_key == "ever"
