# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DiskGraph indexed feature reads and training configuration."""

import json
import logging
import multiprocessing
import os
import sys
from typing import TYPE_CHECKING, Any, Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from kumoapi.encoder import Datetime
from torch_frame import TensorFrame

from kumo_core.diskgraph.metadata import DiskgraphSchema, load_diskgraph_schema
from kumo_core.encoders.config import EncodersConfig
from kumo_core.encoders.tensor_frame import get_tf
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.features.stores.feature import (
    FeatureStore,
    FeatureStoreAttr,
    FeatureStoreTableSchema,
)
from kumo_core.sources.cpu import cpu_budget
from kumo_core.training.config import shm_safe_num_workers
from kumo_core.types.features import FeatureStoreData

if TYPE_CHECKING:
    from diskgraph import SamplerEngine

logger = logging.getLogger(__name__)

def _datetime_series_to_ns(
    ser: pd.Series,
    nat_sentinel: Optional[int] = None,
) -> np.ndarray:
    r"""Convert a (possibly NaT-bearing) datetime ``Series`` to int64 NANOSECONDS,
    independent of the source datetime *resolution*.

    Kumo Python treats every in-memory timestamp as nanoseconds (the engine owns
    its own unit and is converted at its boundaries). A bare
    ``ser.astype('int64')`` is NOT resolution-stable under pandas >= 2: it
    preserves the source arrow resolution, so a ``datetime64[s]`` column (what
    the DiskGraph engine emits for ``timestamp[s]``) yields *seconds* — ~1e9x too
    small for ns-expecting consumers. Forcing ``datetime64[ns]`` first normalizes
    every resolution to nanoseconds.

    ``nat_sentinel`` controls the value substituted for ``NaT`` rows: ``None``
    (default) preserves the legacy ``astype('int64')`` behavior (``NaT`` becomes
    ``numpy.iinfo(int64).min``); pass ``-1`` for the ``RHSTime`` never-expire /
    missing-time sentinel.
    """
    vals = ser.astype("datetime64[ns]").astype("int64").to_numpy()
    if nat_sentinel is not None:
        vals = vals.copy()
        vals[pd.isna(ser).to_numpy()] = nat_sentinel
    return vals


def diskgraph_auto_config(
    cpu_count: int,
    ram_bytes: int,
    index_bytes: int,
    estimated_edges: int,
) -> dict:
    """Auto-configure diskgraph sampling parameters.

    Returns dict with keys:
    - ``mode``: ``"fused"`` or ``"standalone"``
    - ``num_workers``: recommended DataLoader workers
    - ``rayon_threads``: rayon threads per worker

    ``estimated_edges`` is the per-batch I/O volume estimate:
    ``batch_size * per_seed_fanout * 0.25`` (dedup/sparsity factor).

    Heuristic based on empirical sweep data across subgraph sizes,
    worker counts, and rayon thread counts on a production ranking job's
    graph (500GB index, 64 vCPUs, 500GB RAM).
    """
    # On non-Linux platforms (macOS, Windows) multiprocessing uses 'spawn',
    # where re-importing the whole module graph per worker makes worker startup
    # expensive enough that single-process loading is the better default. The
    # Native handles are recreated in workers, so this is a cost heuristic,
    # not a correctness guard.
    if multiprocessing.get_start_method(allow_none=True) in (None, 'spawn') \
            and sys.platform != 'linux':
        return {"mode": "fused", "num_workers": 0, "rayon_threads": 1}

    if estimated_edges < 5_000:
        return {
            "mode": "fused",
            "num_workers": shm_safe_num_workers(max(2, min(cpu_count // 4,
                                                           16))),
            "rayon_threads": 1,
        }

    usable_memory = int(ram_bytes * 0.7)
    cache_ratio = usable_memory / max(index_bytes, 1)
    effective_cpus = (int(cpu_count *
                          0.85) if cache_ratio > 0.9 else int(cpu_count * 0.5))

    if estimated_edges < 100_000:
        num_workers = min(effective_cpus // 2, 16)
    else:
        num_workers = min(effective_cpus // 4, 8)
    num_workers = max(2, num_workers)

    rayon_threads = max(2, min(effective_cpus // num_workers, 8))
    num_workers = shm_safe_num_workers(num_workers)
    if num_workers > 0:
        rayon_threads = max(2, min(effective_cpus // num_workers, 8))

    return {
        "mode": "standalone",
        "num_workers": num_workers,
        "rayon_threads": rayon_threads,
    }


def diskgraph_index_size_bytes(index_path: str) -> int:
    """Read the finalized DiskGraph snapshot size without walking its tree.

    The snapshot manifest is written atomically only after ingest succeeds. A
    directory walk is unsafe because a pool stage/evict can mutate the live
    directory while a trainer is selecting its runtime configuration.
    """
    manifest_path = os.path.join(index_path, "_MANIFEST.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return int(sum(entry["size_bytes"] for entry in manifest["files"]))


def _decimal_primary_key_arrow_type(
    index_path: str,
    table_name: str,
    primary_key_name: str,
) -> pa.DataType | None:
    """Recover decimal PK metadata lost by the feature-fetch string view."""
    config_path = os.path.join(index_path, 'diskgraph_config.json')
    try:
        with open(config_path) as file:
            config = json.load(file)
    except FileNotFoundError:
        return None

    for table in config.get('tables', []):
        if table.get('name') != table_name:
            continue
        for column in table.get('columns', []):
            if column.get('name') != primary_key_name:
                continue
            dtype = column.get('ingested_dtype')
            if dtype not in {'decimal128', 'decimal256'}:
                return None
            try:
                precision = int(column['precision'])
                scale = int(column['scale'])
                constructor = (pa.decimal128
                               if dtype == 'decimal128' else pa.decimal256)
                return constructor(precision, scale)
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"DiskGraph decimal primary key '{table_name}."
                    f"{primary_key_name}' is missing valid precision/scale"
                ) from error
    return None


_NODE_COLUMN_READ_BATCH = 1 << 20


def _iter_node_column(
    engine: Any,
    table_name: str,
    column: str,
    start: int,
    stop: int,
) -> Iterator[tuple[int, pa.Array]]:
    for chunk_start in range(start, stop, _NODE_COLUMN_READ_BATCH):
        node_ids = list(
            range(chunk_start, min(chunk_start + _NODE_COLUMN_READ_BATCH,
                                   stop)))
        request = [{
            'node_type': table_name,
            'node_ids': node_ids,
            'feature_names': [column],
        }]
        batches = [
            event['batch'] for event in engine.fetch_features(request)
            if event['type'] == 'feature'
        ]
        if not batches:
            raise RuntimeError(
                f"DiskGraph returned no '{table_name}.{column}' values for "
                f'node IDs [{node_ids[0]}, {node_ids[-1]}]')
        values = pa.Table.from_batches(batches)
        if column not in values.column_names:
            raise RuntimeError(f"DiskGraph response for '{table_name}' "
                               f"omitted column '{column}'")
        if values.num_rows != len(node_ids):
            raise RuntimeError(
                f"DiskGraph returned {values.num_rows} '{table_name}.{column}' "
                f'values for {len(node_ids)} requested node IDs')
        yield chunk_start, values.column(column).combine_chunks()


def _node_column_engine(index_path: str) -> Any:
    from diskgraph import SamplerEngine
    return SamplerEngine(index_path, num_threads=cpu_budget())


def read_node_column(
    index_path: str,
    table_name: str,
    column: str,
    num_nodes: int,
    engine: Any | None = None,
) -> pa.ChunkedArray:
    """``column`` for node IDs ``[0, num_nodes)``, scanned on a multi-threaded
    engine."""
    chunks = [
        values for _, values in _iter_node_column(
            engine or _node_column_engine(index_path), table_name, column, 0,
            num_nodes)
    ]
    return pa.chunked_array(chunks, type=None if chunks else pa.null())


def iter_primary_key_batches(
    index_path: str,
    table_name: str,
    *,
    start: int = 0,
    stop: int,
    batch_size: int = 50_000,
    engine: Any | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield canonical node IDs and primary keys without graph sampling."""
    if start < 0 or stop < start:
        raise ValueError('primary-key range must satisfy 0 <= start <= stop')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    schema = load_diskgraph_schema(index_path)
    pkey_name = schema.pkey_dict.get(table_name)
    if not pkey_name:
        raise KeyError(
            f"DiskGraph table '{table_name}' has no primary-key column")
    decimal_primary_key_type = _decimal_primary_key_arrow_type(
        index_path,
        table_name,
        pkey_name,
    )

    feature_engine = engine or _node_column_engine(index_path)
    for chunk_start, primary_key_column in _iter_node_column(
            feature_engine, table_name, pkey_name, start, stop):
        if (decimal_primary_key_type is not None
                and primary_key_column.type != decimal_primary_key_type):
            try:
                primary_key_column = primary_key_column.cast(
                    decimal_primary_key_type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
                raise RuntimeError(
                    f"DiskGraph returned invalid values for decimal primary "
                    f"key '{table_name}.{pkey_name}'") from error
        for offset in range(0, len(primary_key_column), batch_size):
            batch = primary_key_column.slice(offset, batch_size)
            node_ids = list(
                range(chunk_start + offset,
                      chunk_start + offset + len(batch)))
            canonical = np.asarray(
                feature_engine.canonical_node_mask(table_name, node_ids,
                                                   batch),
                dtype=np.bool_,
            )
            if canonical.shape != (len(node_ids), ):
                raise RuntimeError(
                    f"DiskGraph returned an invalid canonical-node mask for "
                    f"'{table_name}': {canonical.shape}")
            yield (np.asarray(node_ids, dtype=np.uint64)[canonical],
                   batch.to_pandas().to_numpy(copy=False)[canonical])


class DiskgraphFeatureStore(FeatureStore):
    """Serve metadata and indexed features for RHS embeddings and explanation.

    Sampled features live on runtime batches. This store supplies RHS feature
    prefetching and additional primary keys and features for explanation.
    """

    def __init__(
        self,
        index_path: str,
        table_schema_dict: dict[str, FeatureStoreTableSchema],
        diskgraph_num_nodes: dict[str, int],
        entity_table_names: list[str],
    ) -> None:
        """Initialize metadata without materializing graph-table data.

        Args:
            index_path: Local snapshot directory for lazy feature reads.
            table_schema_dict: Graph table column schemas and tensor mappers.
            diskgraph_num_nodes: Graph table sizes from snapshot metadata.
            entity_table_names: Entity tables in task order.
        """
        super().__init__()
        self._table_schema_dict = table_schema_dict
        self._diskgraph_num_nodes = diskgraph_num_nodes
        self._entity_table_names = list(entity_table_names)
        self._index_path = index_path
        self._engine: SamplerEngine | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state; workers recreate the native engine lazily."""
        state = self.__dict__.copy()
        state["_engine"] = None
        return state

    @property
    def table_names(self) -> list[str]:
        """Return graph table names."""
        return list(self._table_schema_dict)

    @property
    def entity_table_names(self) -> list[str]:
        """Return entity tables in task order."""
        return list(self._entity_table_names)

    @property
    def train_table_name(self) -> None:
        """Return None; training tables belong to the native runtime."""
        return None

    @property
    def train_fkey_names(self) -> list[str]:
        """Return no training foreign keys; these belong to the native runtime."""
        return []

    @property
    def has_train_table_test_mask(self) -> bool:
        """Return False; this store contains only graph tables."""
        return False

    @property
    def has_train_table_target_col(self) -> bool:
        """Return False; targets belong to the native runtime."""
        return False

    @property
    def has_train_table_weight_col(self) -> bool:
        """Return False; training weights belong to the native runtime."""
        return False

    def get_pkey_name(self, table_name: str) -> str | None:
        """Return a table's primary key column, or None if it has none."""
        return self._table_schema_dict[table_name].pkey_name

    def get_time_col_name(self, table_name: str) -> str | None:
        """Return a table's creation-time column, or None if absent."""
        schema = self._table_schema_dict.get(table_name)
        return schema.time_col_name if schema is not None else None

    def get_end_time_col_name(self, table_name: str) -> str | None:
        """Return a table's expiration-time column, or None if absent."""
        schema = self._table_schema_dict.get(table_name)
        return schema.end_time_col_name if schema is not None else None

    def get_target_tensor_mapper(self) -> TensorMapper | None:
        """Return None; target encoding belongs to the native runtime."""
        return None

    def get_node_id(self, table_name: str, pkey: np.ndarray) -> torch.Tensor:
        """Reject key resolution, which belongs to the native runtime.

        Args:
            table_name: Graph table name.
            pkey: Primary keys to resolve.

        Raises:
            NotImplementedError: This store accepts resolved DiskGraph node IDs.
        """
        raise NotImplementedError("Resolve primary keys through the DiskGraph runtime")

    @classmethod
    def for_explain(
        cls,
        *,
        index_path: str,
        encoders: EncodersConfig,
        entity_table_names: list[str],
        time_col_dict: dict[str, str] | None = None,
        end_time_col_dict: dict[str, str] | None = None,
        schema: DiskgraphSchema | None = None,
    ) -> "DiskgraphFeatureStore":
        """Build a store for RHS reads and explanation without loading graph tables.

        Args:
            index_path: Local DiskGraph snapshot directory.
            encoders: Encoders defining the model's graph-table features.
            entity_table_names: Entity tables in task order.
            time_col_dict: Table creation-time columns.
            end_time_col_dict: Table expiration-time columns.
            schema: Already loaded snapshot metadata, if available.

        Returns:
            A store with metadata and lazy indexed feature reads.

        Raises:
            OSError: If snapshot metadata cannot be read.
        """
        time_col_dict = time_col_dict or {}
        end_time_col_dict = end_time_col_dict or {}

        if schema is None:
            schema = load_diskgraph_schema(index_path)

        table_schema_dict: dict[str, FeatureStoreTableSchema] = {}

        for table_name in schema.table_names:
            encoder_dict = encoders.encoders.get(table_name, {})
            mapper_dict = {
                col_name: TensorMapper.from_encoder(encoder)
                for col_name, encoder in encoder_dict.items()
            }
            table_schema_dict[table_name] = FeatureStoreTableSchema(
                mapper_dict=mapper_dict,
                pkey_name=schema.pkey_dict.get(table_name),
                time_col_name=time_col_dict.get(table_name),
                end_time_col_name=end_time_col_dict.get(table_name),
            )

        return cls(
            index_path=index_path,
            table_schema_dict=table_schema_dict,
            diskgraph_num_nodes=schema.num_nodes_dict,
            entity_table_names=entity_table_names,
        )

    def get_tensor_mapper(self, table_name: str, col_name: str) -> TensorMapper:
        """Return the feature mapper, including unencoded temporal metadata.

        Args:
            table_name: Graph table name.
            col_name: Feature or temporal column name.

        Returns:
            The configured mapper or a datetime mapper for temporal metadata.

        Raises:
            KeyError: If the table or column is unknown.
        """
        schema = self._table_schema_dict.get(table_name)
        if (
            schema is not None
            and col_name not in schema.mapper_dict
            and col_name in (schema.time_col_name, schema.end_time_col_name)
        ):
            # Temporal columns are exposed to explanation without being
            # ordinary encoded features. Keep the model's TensorFrame schema
            # unchanged while supplying the collector's timestamp metadata.
            return TensorMapper.from_encoder(Datetime())
        return self._table_schema_dict[table_name].mapper_dict[col_name]

    def _get_tensor(self, attr: FeatureStoreAttr) -> FeatureStoreData | None:
        table_name = attr.table_name
        assert isinstance(table_name, str)
        schema = self._table_schema_dict[table_name]
        if attr.attr_name == "tf":
            feature_names = list(schema.mapper_dict)
        elif attr.attr_name in {"pkey", "time", "end_time"}:
            column_name = {
                "pkey": schema.pkey_name,
                "time": schema.time_col_name,
                "end_time": schema.end_time_col_name,
            }[attr.attr_name]
            if column_name is None:
                return None
            feature_names = [column_name]
        else:
            raise KeyError(f"Unsupported DiskGraph attribute '{attr.attr_name}'")

        node_ids = self._get_index(attr)
        if attr.attr_name == "tf" and not feature_names:
            return TensorFrame({}, {}, num_rows=len(node_ids))
        if len(node_ids) == 0:
            if attr.attr_name == "tf":
                return get_tf(
                    pa.table({name: [] for name in feature_names}), schema.mapper_dict
                )
            if attr.attr_name == "pkey":
                return np.empty(0, dtype=object)
            return torch.empty(0, dtype=torch.long)

        table = self._fetch_features(table_name, node_ids, feature_names)
        if attr.attr_name == "tf":
            mappers = {
                name: mapper
                for name, mapper in schema.mapper_dict.items()
                if name in table.column_names
            }
            if not mappers:
                return get_tf(
                    pa.table({name: [] for name in feature_names}), schema.mapper_dict
                )
            return get_tf(table, mappers)
        if feature_names[0] not in table.column_names:
            if attr.attr_name == "pkey":
                return np.arange(len(node_ids))
            return torch.zeros(len(node_ids), dtype=torch.long)
        column = table.column(feature_names[0])
        if attr.attr_name == "pkey":
            return column.to_numpy(zero_copy_only=False).copy()
        timestamps = pd.to_datetime(column.to_pandas(), errors="coerce")
        return torch.from_numpy(_datetime_series_to_ns(timestamps))

    def _get_tensor_size(self, attr: FeatureStoreAttr) -> tuple[int, ...] | None:
        table_name = attr.table_name
        if isinstance(table_name, str) and table_name in self._diskgraph_num_nodes:
            return (self._diskgraph_num_nodes[table_name],)
        raise KeyError(f"Unknown DiskGraph table '{table_name}'")

    def _fetch_features(
        self,
        table_name: str,
        node_ids: torch.Tensor,
        feature_names: list[str],
    ) -> pa.Table:
        """Fetch requested rows in bounded chunks through one lazy engine."""
        if self._engine is None:
            from diskgraph import SamplerEngine

            self._engine = SamplerEngine(self._index_path)

        batches = []
        batch_size = 50_000
        for chunk in node_ids.split(batch_size):
            request = [
                {
                    "node_type": table_name,
                    "node_ids": chunk.cpu().tolist(),
                    "feature_names": feature_names,
                }
            ]
            for event in self._engine.fetch_features(request):
                if event["type"] == "feature":
                    batches.append(event["batch"])

        return pa.Table.from_batches(batches) if batches else pa.table({})
