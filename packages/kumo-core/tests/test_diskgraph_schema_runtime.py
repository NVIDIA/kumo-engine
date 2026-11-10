# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify native feature metadata is prepared without loading graph data."""

from unittest.mock import Mock

import pytest
from kumo_core.diskgraph.metadata import DiskgraphSchema
from kumo_core.encoders.config import EncodersConfig
from kumo_core.training import runtime
from kumoapi.encoder import Numerical


def test_runtime_prepares_context_without_opening_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepare real mappers and schema without native handles or DataFrames."""
    schema = DiskgraphSchema(
        table_names=["events", "users"],
        pkey_dict={"events": "id", "users": "id"},
        fk_specs=[("events.user_id", "users.id")],
        raw_edge_types=[],
        edge_types=[("events", "user_id", "users"), ("users", "rev_user_id", "events")],
        edge_type_remap={},
        num_nodes_dict={"events": 3, "users": 2},
    )
    load_schema = Mock(return_value=schema)
    monkeypatch.setattr(runtime, "load_diskgraph_schema", load_schema)
    encoders = EncodersConfig(encoders={"events": {"value": Numerical()}})

    context = runtime.LocalDiskgraphTrainingRuntime().create_feature_context(
        index_path="/index",
        encoders=encoders,
        entity_table_names=["events"],
        time_col_dict={"events": "created_at"},
    )

    assert context.schema is schema
    assert context._engine is None
    assert context.entity_table_names == ["events"]
    assert context.get_pkey_name("users") == "id"
    assert context.time_col_dict == {"events": "created_at"}
    assert set(context.table_schema_dict["events"].mapper_dict) == {"value"}
    assert context.table_schema_dict["users"].mapper_dict == {}
    load_schema.assert_called_once_with("/index")
