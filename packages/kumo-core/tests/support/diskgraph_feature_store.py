# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DiskGraph stores backed by mocked Arrow reads for component tests."""

from collections.abc import Mapping
from typing import Any, TypedDict
from unittest.mock import Mock

import pandas as pd
import pyarrow as pa
from kumo_core.diskgraph.stores import DiskgraphFeatureStore
from kumo_core.features.stores.feature import FeatureStoreTableSchema


class FeatureRequest(TypedDict):
    """Columns and ordered node IDs requested from the mocked engine."""

    node_type: str
    node_ids: list[int]
    feature_names: list[str]


def make_diskgraph_feature_store(
    df_dict: Mapping[str, pd.DataFrame | pa.Table],
    table_schema_dict: Mapping[str, FeatureStoreTableSchema | dict[str, Any]],
    entity_table_names: list[str] | None = None,
) -> DiskgraphFeatureStore:
    """Build a real store with a mocked engine reading the supplied tables.

    Args:
        df_dict: Tables in DiskGraph node-ID order.
        table_schema_dict: Column schemas and tensor mappers for graph tables.
        entity_table_names: Entity tables in task order, if needed by the test.

    Returns:
        A store exercising production metadata, query, and feature-mapping code.

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
    store = DiskgraphFeatureStore(
        index_path="/mock-snapshot",
        table_schema_dict=schemas,
        diskgraph_num_nodes={name: len(tables[name]) for name in schemas},
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
