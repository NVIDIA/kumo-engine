# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from kumo_core.types import TaskType
from kumoapi.typing import TimeUnit

from kumo.pquery import PredictiveQueryV2
from kumo.pquery.train_table_generator import TrainingTableGeneratorV2
from kumo.pquery.utils import table_name
from kumo.task.split import DateOffsetSplit
from kumo.testing.pquery import get_validated_query
from statedb.utils.errors import UnsupportedInDiskgraphError


def test_registered_scale_validation_describes_native_physical_table(
    create_node_pred_graph,
    monkeypatch,
    tempdir,
) -> None:
    from kumo.pquery.pquery_validators import pre_train_table_validator as mod

    graph = create_node_pred_graph("scale_contract", tempdir.name)
    query = get_validated_query(
        graph,
        "PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 20) "
        "FOR EACH USERS.USER_ID",
    )
    generator = PredictiveQueryV2(
        name="scale_contract",
        graph=graph,
        validated_predictive_query=query,
        split=DateOffsetSplit([-2, 0], unit=TimeUnit.DAYS, timeframe=2),
    ).table_generator
    monkeypatch.setattr(mod, "TRAIN_ROWS_ERROR_LIMIT", 0)
    monkeypatch.setattr(mod, "get_max_train_rows", lambda: 200_000_000)
    entity_table = graph.table(table_name(query.entity_column))
    monkeypatch.setattr(
        entity_table,
        "num_rows",
        lambda **_kwargs: 1_000_000,
    )
    monkeypatch.setattr(
        generator,
        "generate_timestamps",
        lambda: [pd.Timestamp("2024-01-01")],
    )

    response = generator.validate_scale()

    assert response is not None
    assert response.ok
    assert len(response.warnings) == 1
    warning = response.warnings[0]
    assert warning.title == "Training table may be very large"
    assert "DiskGraph will generate the full physical training table" in (
        warning.message)
    assert "bounded subset of at most 200,000,000 rows" in warning.message
    assert "automatically downsampling" not in warning.message


def test_native_downsampling_defaults_disabled(monkeypatch) -> None:
    from kumo.pquery import train_table_generator as mod

    monkeypatch.delenv("DISKGRAPH_PRODUCER_DOWNSAMPLING_ENABLED",
                       raising=False)
    monkeypatch.setattr(mod, "get_max_train_rows", lambda: 200_000_000)
    generator = object.__new__(TrainingTableGeneratorV2)
    generator.task_type = TaskType.REGRESSION

    config = generator._build_diskgraph_downsampling_config(False)

    assert config == {
        "enabled": False,
        "max_rows": 200_000_000,
        "preserve_pre_downsample_tt": True,
        "is_link_prediction": False,
        "list_distinct_target_is_array": False,
        "uses_arl_labels": False,
    }


def test_native_downsampling_enabled_remains_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("DISKGRAPH_PRODUCER_DOWNSAMPLING_ENABLED", "1")
    generator = object.__new__(TrainingTableGeneratorV2)
    generator.task_type = TaskType.REGRESSION

    with pytest.raises(
        UnsupportedInDiskgraphError,
        match="DiskGraph downsampling is not yet wired",
    ):
        generator._build_diskgraph_downsampling_config(False)
