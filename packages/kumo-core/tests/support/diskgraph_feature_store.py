# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native read contexts backed by mocked Arrow tables for component tests."""

from collections.abc import Mapping
from typing import Any, TypedDict
from unittest.mock import Mock

import pandas as pd
import pyarrow as pa
from kumo_core.diskgraph.metadata import DiskgraphSchema
from kumo_core.features.reads import DiskgraphFeatureContext
from kumo_core.features.stores.feature import FeatureStoreTableSchema


class FeatureRequest(TypedDict):
    """Columns and ordered node IDs requested from the mocked engine."""

    node_type: str
    node_ids: list[int]
    feature_names: list[str]


def make_feature_context(
    df_dict: Mapping[str, pd.DataFrame | pa.Table],
    table_schema_dict: Mapping[str, FeatureStoreTableSchema | dict[str, Any]],
    entity_table_names: list[str] | None = None,
) -> DiskgraphFeatureContext:
    """Build a native read context with a mocked engine for supplied tables.

    Args:
        df_dict: Tables in DiskGraph node-ID order.
        table_schema_dict: Column schemas and tensor mappers for graph tables.
        entity_table_names: Entity tables in task order, if needed by the test.

    Returns:
        A context exercising production metadata and feature-mapping code.

    Raises:
        KeyError: If a schema has no corresponding input table.
    """
    schemas = {
        name: FeatureStoreTableSchema.cast(schema)
        for name, schema in table_schema_dict.items()
    }
    tables = {
        name: frame if isinstance(frame, pa.Table) else pa.Table.from_pandas(frame)
        for name, frame in df_dict.items()
    }
    store = DiskgraphFeatureContext(
        index_path="/mock-snapshot",
        schema=DiskgraphSchema(
            table_names=list(schemas),
            pkey_dict={
                name: schema.pkey_name
                for name, schema in schemas.items()
                if schema.pkey_name is not None
            },
            fk_specs=[],
            raw_edge_types=[],
            edge_types=[],
            edge_type_remap={},
            num_nodes_dict={name: len(tables[name]) for name in schemas},
        ),
        table_schema_dict=schemas,
        entity_table_names=entity_table_names or [],
    )

    def fetch_features(
        requests: list[FeatureRequest],
    ) -> list[dict[str, str | pa.RecordBatch]]:
        events: list[dict[str, str | pa.RecordBatch]] = []
        for request in requests:
            table = tables[request["node_type"]].select(request["feature_names"])
            rows = table.take(pa.array(request["node_ids"], type=pa.int64()))
            events.extend(
                {"type": "feature", "batch": batch} for batch in rows.to_batches()
            )
        return events

    store._engine = Mock(fetch_features=Mock(side_effect=fetch_features))
    return store
