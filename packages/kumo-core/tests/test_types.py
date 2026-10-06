# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.types import (
    PredictionArtifactType,
    TaskType,
    TaskTypeWithReason,
    TrainTableType,
)
from kumoapi.typing import Dtype, Stype


def test_portable_task_type_preserves_selected_training_semantics():
    assert TaskType.BINARY_CLASSIFICATION.is_node_pred
    assert TaskType.REGRESSION.has_numerical_target
    assert TaskType.STATIC_LINK_PREDICTION.is_link_pred
    assert TaskType.TEMPORAL_LINK_PREDICTION.is_ranking
    assert not TaskType.FORECASTING.supports_xai


def test_portable_task_type_with_reason_preserves_fields():
    result = TaskTypeWithReason(TaskType.REGRESSION, "numeric target")

    assert result.task_type is TaskType.REGRESSION
    assert result.reason == "numeric target"


def test_portable_train_table_and_artifact_types_preserve_values():
    assert TrainTableType.TEMPORAL_NODE_PRED.is_temporal
    assert TrainTableType.STATIC_LINK_PRED.is_link_pred
    assert (PredictionArtifactType.PREDICTIONS.output_file_name ==
            "predictions.parquet")


def test_portable_dtype_and_stype_preserve_training_facing_behavior():
    assert Dtype.int64.is_int()
    assert Dtype.floatlist.list_element_type() is Dtype.float
    assert Dtype.string.supports_cast_to_temporal()
    assert Stype.numerical.supports_dtype(Dtype.float32)
    assert Stype.sequence.supports_dtype(Dtype.floatlist)
    assert not Stype.sequence.supports_dtype(Dtype.stringlist)
    # `str` is the bare value now that `Stype` is kumoapi's. The old core copy
    # rendered `"kumo.timestamp"`; only `kumo/data/cache/exact_stats.py` read
    # that prefix, and it strips an optional one either way.
    assert str(Stype.timestamp) == "timestamp"
    assert Dtype.floatlist.title_case() == "FloatList"
