# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit node reads for DiskGraph and legacy in-memory consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, TypeAlias

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from kumoapi.encoder import Datetime
from torch import Tensor
from torch_frame import TensorFrame

from kumo_core.diskgraph.metadata import DiskgraphSchema
from kumo_core.diskgraph.stores import (
    _decimal_primary_key_arrow_type,
    _node_column_engine,
)
from kumo_core.encoders.tensor_frame import get_tf
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.features.stores.feature import FeatureStore, FeatureStoreTableSchema

if TYPE_CHECKING:
    from diskgraph import SamplerEngine

READ_BATCH_SIZE = 50_000


@dataclass
class DiskgraphFeatureContext:
    """Metadata for node reads, with a process-local, lazily opened engine.

    Args:
        index_path: Local DiskGraph index directory.
        schema: Table and relationship metadata for the index.
        table_schema_dict: Column mappers and temporal column names per table.
        entity_table_names: Entity tables used by the task.
    """

    index_path: str
    schema: DiskgraphSchema
    table_schema_dict: dict[str, FeatureStoreTableSchema]
    entity_table_names: list[str]
    _engine: SamplerEngine | None = field(
        default=None, init=False, repr=False, compare=False
    )
    train_table_name: ClassVar[None] = None

    @property
    def engine(self) -> SamplerEngine:
        """Return the native engine, opening it on the first read."""
        if self._engine is None:
            self._engine = _node_column_engine(self.index_path)
        return self._engine

    def __getstate__(self) -> dict[str, object]:
        """Return serializable state so workers open their own native engine."""
        return {**self.__dict__, "_engine": None}

    @property
    def table_names(self) -> list[str]:
        """Return graph table names from the index schema."""
        return self.schema.table_names

    @property
    def time_col_dict(self) -> dict[str, str]:
        """Return the configured creation-time column for each temporal table."""
        return {
            table: schema.time_col_name
            for table, schema in self.table_schema_dict.items()
            if schema.time_col_name is not None
        }

    def get_pkey_name(self, table_name: str) -> str | None:
        """Return the primary-key column for a table.

        Args:
            table_name: Graph table to inspect.

        Returns:
            The primary-key column name, or None if absent.
        """
        return self.schema.pkey_dict.get(table_name)

    def get_time_col_name(self, table_name: str) -> str | None:
        """Return the creation-time column for a table.

        Args:
            table_name: Graph table to inspect.

        Returns:
            The time column name, or None for a non-temporal table.

        Raises:
            KeyError: If the table does not exist.
        """
        return self.table_schema_dict[table_name].time_col_name

    def get_tensor_mapper(self, table_name: str, col_name: str) -> TensorMapper:
        """Return a column mapper, including unencoded temporal columns.

        Args:
            table_name: Table containing the column.
            col_name: Column to decode.

        Returns:
            The fitted mapper, or a timestamp mapper for a temporal column.

        Raises:
            KeyError: If the table or column does not exist.
        """
        schema = self.table_schema_dict[table_name]
        if col_name in schema.mapper_dict:
            return schema.mapper_dict[col_name]
        if col_name in (schema.time_col_name, schema.end_time_col_name):
            return TensorMapper.from_encoder(Datetime())
        raise KeyError(f"Unknown column '{table_name}.{col_name}'")


FeatureSource: TypeAlias = FeatureStore | DiskgraphFeatureContext


def _read_columns(
    context: DiskgraphFeatureContext,
    table_name: str,
    node_ids: Tensor,
    columns: list[str],
) -> pa.Table:
    batches: list[pa.RecordBatch] = []
    for ids in node_ids.cpu().split(READ_BATCH_SIZE):
        if ids.numel() == 0:
            continue
        request = [
            {
                "node_type": table_name,
                "node_ids": ids.tolist(),
                "feature_names": columns,
            }
        ]
        chunk_batches = [
            event["batch"]
            for event in context.engine.fetch_features(request)
            if event["type"] == "feature"
        ]
        num_rows = sum(batch.num_rows for batch in chunk_batches)
        if num_rows != ids.numel():
            raise RuntimeError(
                f"DiskGraph returned {num_rows} rows for {ids.numel()} "
                f"requested nodes in '{table_name}'"
            )
        batches.extend(chunk_batches)
    if not batches:
        return pa.table({column: pa.array([]) for column in columns})
    table = pa.Table.from_batches(batches)
    missing = set(columns).difference(table.column_names)
    if missing:
        raise KeyError(f"DiskGraph omitted columns {sorted(missing)} in '{table_name}'")
    return table.select(columns)


def read_tensor_frame(
    source: FeatureSource,
    table_name: str,
    node_ids: Tensor,
) -> TensorFrame:
    """Read and encode requested nodes in order, preserving duplicate IDs.

    Args:
        source: DiskGraph context or legacy feature store.
        table_name: Graph table to read.
        node_ids: Node IDs in the required output order.

    Returns:
        A TensorFrame with one row per requested node.

    Raises:
        KeyError: If table metadata or requested columns are missing.
        RuntimeError: If the engine returns an incomplete response.
    """
    if not isinstance(source, DiskgraphFeatureContext):
        return source[table_name, "tf", node_ids]
    mappers = source.table_schema_dict[table_name].mapper_dict
    if not mappers:
        return TensorFrame({}, {}, num_rows=node_ids.numel())
    return get_tf(_read_columns(source, table_name, node_ids, list(mappers)), mappers)


def read_primary_keys(
    source: FeatureSource,
    table_name: str,
    node_ids: Tensor,
) -> np.ndarray:
    """Read primary keys for the requested nodes without substituting node IDs.

    Args:
        source: DiskGraph context or legacy feature store.
        table_name: Graph table to read.
        node_ids: Node IDs in the required output order.

    Returns:
        Primary keys in request order, retaining their native values.

    Raises:
        KeyError: If the table has no primary-key column or it is omitted.
        RuntimeError: If the engine returns an incomplete response.
    """
    if not isinstance(source, DiskgraphFeatureContext):
        return source[table_name, "pkey", node_ids]
    column = source.get_pkey_name(table_name)
    if column is None:
        raise KeyError(f"DiskGraph table '{table_name}' has no primary key")
    values = _read_columns(source, table_name, node_ids, [column]).column(column)
    decimal_type = _decimal_primary_key_arrow_type(
        source.index_path, table_name, column
    )
    if decimal_type is not None:
        values = values.cast(decimal_type)
    return values.to_numpy(zero_copy_only=False).copy()


def read_node_times(
    source: FeatureSource,
    table_name: str,
    node_ids: Tensor,
    end_time: bool = False,
) -> Tensor:
    """Read creation times as nanoseconds, retaining the missing-time sentinel.

    Args:
        source: DiskGraph context or legacy feature store.
        table_name: Temporal graph table to read.
        node_ids: Node IDs in the required output order.
        end_time: Read the expiry column instead of the creation-time column.

    Returns:
        Int64 nanosecond timestamps; nulls use the minimum int64 value.

    Raises:
        KeyError: If the table has no time column or it is omitted.
        RuntimeError: If the engine returns an incomplete response.
    """
    if not isinstance(source, DiskgraphFeatureContext):
        with source.safe_batch_mode(batch_size=READ_BATCH_SIZE):
            return source[table_name, "end_time" if end_time else "time", node_ids]
    column = (
        source.table_schema_dict[table_name].end_time_col_name
        if end_time
        else source.get_time_col_name(table_name)
    )
    if column is None:
        raise KeyError(f"DiskGraph table '{table_name}' has no time column")
    if node_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    values = _read_columns(source, table_name, node_ids, [column]).column(column)
    timestamps = pc.cast(pc.cast(values, pa.timestamp("ns")), pa.int64())
    timestamps = pc.fill_null(timestamps, np.iinfo(np.int64).min)
    return torch.from_numpy(timestamps.to_numpy(zero_copy_only=False).copy())
