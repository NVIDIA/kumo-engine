# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free DiskGraph feature and graph stores for numerical training.

Supports two backends:
- **Arrow Flight** (remote): communicates with ``diskgraph-gateway`` over RPC.
- **Pybind** (local/co-located): calls ``diskgraph-pybind`` native module
  directly, eliminating RPC/serde overhead.

"""

import json
import logging
import multiprocessing
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.flight as flight
import torch
from kumo_core.diskgraph.metadata import load_diskgraph_schema
from kumo_core.encoders.config import EncodersConfig
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.features.stores.feature import FeatureStoreTableSchema
from kumo_core.features.stores.graph import GraphStore, GraphStoreAttr
from kumo_core.features.stores.in_memory_feature import (
    InMemoryFeatureStore,
    get_tf,
)
from kumo_core.sources.cpu import cpu_budget
from kumo_core.training.config import shm_safe_num_workers
from kumoapi.encoder import Datetime
from torch_geometric.sampler import NumNeighbors
from torch_geometric.typing import EdgeType

logger = logging.getLogger(__name__)

NumNeighborsType = Union[NumNeighbors, List[int], Dict[EdgeType, List[int]]]


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
    # SamplerEngine / Flight handles themselves are now safe to ship across any
    # start method (see DiskgraphFeatureStore.__getstate__), so this is a
    # cost heuristic, not a correctness guard.
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


_DISKGRAPH_STRICT_HOP_VALIDATION = os.getenv(
    "DISKGRAPH_STRICT_HOP_VALIDATION",
    "").lower() in {"1", "true", "yes", "on"}


def _types_from_schema_mapping(
        mapping: dict) -> Tuple[List[str], List[EdgeType]]:
    node_types = list(mapping.get("node_types", []))
    edge_types = [(rel["src_type"], rel["name"], rel["dst_type"])
                  for rel in mapping.get("edge_relations", [])
                  if rel.get("name")]
    return node_types, edge_types


def _fetch_flight_schema_mapping(server_address: str) -> dict:
    client = flight.connect(server_address)
    action = flight.Action("get_type_name_mapping", b"")
    results = list(client.do_action(action))
    if not results:
        raise RuntimeError("Gateway returned no schema mapping payload")
    return json.loads(results[0].body.to_pybytes())


@dataclass
class ResolvedSchema:
    """Result of auto-deriving edge_types/node_types/edge_type_remap."""

    edge_types: List[Tuple[str, str, str]]
    node_types: List[str]
    edge_type_remap: Optional[Dict[Tuple[str, str, str], Tuple[str, str, str]]]
    schema_mapping: Optional[dict]  # non-None when fetched from flight gateway


def resolve_schema(
    edge_types: Optional[List[Tuple[str, str, str]]],
    node_types: Optional[List[str]],
    edge_type_remap: Optional[Dict[Tuple[str, str, str], Tuple[str, str, str]]],
    index_path: Optional[str],
    server_address: Optional[str],
) -> ResolvedSchema:
    """Derive missing edge_types/node_types from index metadata or gateway."""
    schema_mapping = None

    if index_path is not None and (edge_types is None or node_types is None):
        schema = load_diskgraph_schema(index_path)
        if edge_types is None:
            edge_types = schema.edge_types
        if node_types is None:
            node_types = schema.node_types
        if edge_type_remap is None:
            edge_type_remap = schema.edge_type_remap

    if edge_types is None or node_types is None:
        if server_address is None:
            raise ValueError(
                "edge_types/node_types missing and no schema source")
        schema_mapping = _fetch_flight_schema_mapping(server_address)
        mapped_node_types, mapped_edge_types = _types_from_schema_mapping(
            schema_mapping)
        if node_types is None:
            node_types = mapped_node_types
        if edge_types is None:
            edge_types = mapped_edge_types

    return ResolvedSchema(
        edge_types=edge_types,
        node_types=node_types,
        edge_type_remap=edge_type_remap,
        schema_mapping=schema_mapping,
    )


class DiskgraphGraphStore(GraphStore):
    """GraphStore adapter for a diskgraph backend.

    This store exposes edge metadata (edge types) needed by
    ``filter_custom_hetero_store`` to assemble ``edge_index`` from sampler
    output. Edge indices themselves are not fetched via GraphStore in this
    path; they come directly from the sampler.

    Supports two modes:
    - ``server_address``: remote Arrow Flight gateway.
    - ``index_path``: local pybind engine (co-located mode).

    In either mode, ``edge_types`` / ``node_types`` may be omitted and will be
    derived from local index metadata (preferred when ``index_path`` is
    available) or from the gateway schema mapping action.
    """
    def __init__(
        self,
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
        node_types: Optional[List[str]] = None,
        server_address: Optional[str] = None,
        index_path: Optional[str] = None,
        edge_type_remap: Optional[Dict[Tuple[str, str, str],
                                       Tuple[str, str, str]]] = None,
    ):
        super().__init__()
        if server_address is None and index_path is None:
            raise ValueError(
                "DiskgraphGraphStore requires either server_address or index_path"
            )

        schema = resolve_schema(edge_types, node_types, edge_type_remap,
                                index_path, server_address)

        self.server_address = server_address
        self.index_path = index_path
        self.node_types = schema.node_types
        self.edge_type_remap = schema.edge_type_remap
        self._edge_attrs = [
            GraphStoreAttr(edge_type=et) for et in schema.edge_types
        ]

    def get_all_edge_attrs(self) -> List[GraphStoreAttr]:
        return [
            GraphStoreAttr(edge_type=attr.edge_type, size=attr.size)
            for attr in self._edge_attrs
        ]

    def _get_edge_index(
        self,
        _edge_attr: GraphStoreAttr,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "DiskgraphGraphStore does not hold edge indices locally. "
            "Edge indices are returned by the sampler.")


class DiskgraphFeatureStore(InMemoryFeatureStore):
    """Feature store that overlays fused Arrow Flight features on top of
    in-memory data.

    During each sampling batch, the Rust engine can stream node features
    via side-channel Arrow Flight streams (fused feature fetching).  This
    store caches those Arrow tables and returns them as ``TensorFrame``
    for ``attr_name='tf'`` queries on graph tables, while still serving the
    training table from in-memory tensors.

    Use :meth:`for_explain` to initialize without loading full graph-table
    parquets into pandas.
    """
    def __init__(
        self,
        *args,
        diskgraph_num_nodes: Optional[Dict[str, int]] = None,
        strict_fused_graph_tf: bool = False,
        index_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Fused feature cache: maps table_name -> (ArrowTable, global_node_ids).
        # Populated by the sampler after each sample call, consumed by
        # _get_tensor.  The node_ids allow validation that the cached data
        # matches what the caller requested.
        self._fused_cache: Dict[str, tuple[pa.Table, torch.Tensor]] = {}
        self._diskgraph_num_nodes = diskgraph_num_nodes or {}
        self._strict_fused_graph_tf = strict_fused_graph_tf
        self._index_path = index_path
        self._engine = None
        # Set by the training runtime after construction (see
        # ``set_sampler_engine``) to enable random-access feature reads
        # (e.g. for RHS fusion fetch_feats).
        self._sampler_engine = None

    def __getstate__(self) -> Dict[str, Any]:
        # The pybind SamplerEngine handles (``_engine`` for out-of-core reads
        # and ``_sampler_engine``, a back-reference set by the sampler) are
        # native objects that cannot be pickled. Drop them so the feature store
        # can be shipped to DataLoader workers under non-``fork`` start methods
        # (``forkserver`` is the Linux default on Python 3.14). ``_engine`` is
        # lazily re-created on cache miss; ``_sampler_engine`` is re-attached by
        # the worker's sampler via ``set_sampler_engine`` / ``_get_engine``.
        state = self.__dict__.copy()
        state["_engine"] = None
        state["_sampler_engine"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._engine = None
        self._sampler_engine = None

    @property
    def entity_table_names(self) -> List[str]:
        # Parent ``InMemoryFeatureStore.entity_table_names`` reads
        # ``self._train_table_schema.fkeys`` and crashes when the
        # schema is ``None`` (the TT-NG ``for_explain`` path). Return
        # the explicit list stashed by ``for_explain`` when present.
        if getattr(self, '_explain_entity_table_names', None) is not None:
            return list(self._explain_entity_table_names)
        return super().entity_table_names

    @classmethod
    def for_explain(
        cls,
        *,
        index_path: str,
        encoders: EncodersConfig,
        entity_table_names: List[str],
        time_col_dict: Optional[Dict[str, str]] = None,
        end_time_col_dict: Optional[Dict[str, str]] = None,
    ) -> "DiskgraphFeatureStore":
        """Streaming-only ``DiskgraphFeatureStore`` for
        ``Trainer.explain()`` on the TT-NG path.

        No ``train_table_df`` / ``train_fkeys`` / ``train_mask_names`` /
        ``diskgraph_train_table_data`` — those were legacy train-loop
        concepts TT-NG doesn't use. ``train_table_schema=None`` is passed
        through so ``InMemoryFeatureStore.__init__`` skips
        ``set_train_table(...)`` and entity-table-mapping setup.

        ``SubgraphCollector`` consumes
        ``feature_store[ns, 'pkey'|'tf', node_ids]``; both go through
        ``SamplerEngine.fetch_features``.
        """
        time_col_dict = dict(time_col_dict or {})
        end_time_col_dict = dict(end_time_col_dict or {})

        schema = load_diskgraph_schema(index_path)

        table_schema_dict: Dict[str, FeatureStoreTableSchema] = {}
        df_dict: Dict[str, pd.DataFrame] = {}
        diskgraph_num_nodes: Dict[str, int] = {}

        with open(os.path.join(index_path, "index_metadata.json")) as f:
            index_meta = json.load(f)
        node_type_to_meta = {
            nt["name"]: nt
            for nt in index_meta.get("node_types", [])
        }

        for table_name in schema.table_names:
            encoder_dict = encoders[
                table_name] if table_name in encoders else {}
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
            # FS expects a per-table entry in df_dict, but row reads
            # for graph tables go through fetch_features.
            cols = set(mapper_dict.keys())
            if table_name in schema.pkey_dict:
                cols.add(schema.pkey_dict[table_name])
            if table_name in time_col_dict:
                cols.add(time_col_dict[table_name])
            if table_name in end_time_col_dict:
                cols.add(end_time_col_dict[table_name])
            df_dict[table_name] = pd.DataFrame(
                {col: pd.Series([], dtype="float64")
                 for col in cols})

            nt_meta = node_type_to_meta.get(table_name, {})
            max_id = nt_meta.get("max_id")
            if isinstance(max_id, int) and max_id >= 0:
                diskgraph_num_nodes[table_name] = max_id + 1

        instance = cls(
            df_dict=df_dict,
            table_schema_dict=table_schema_dict,
            train_table_schema=None,
            diskgraph_num_nodes=diskgraph_num_nodes,
            strict_fused_graph_tf=False,
            index_path=index_path,
        )
        # Stash entity_table_names so the overridden property can
        # return it without going through ``_train_table_schema.fkeys``
        # (which is ``None`` on this construction path).
        instance._explain_entity_table_names = list(entity_table_names)
        return instance

    def set_sampler_engine(self, engine) -> None:
        """Set the Rust sampler engine for random-access feature reads."""
        self._sampler_engine = engine

    def get_tensor_mapper(self, table_name: str, col_name: str) -> TensorMapper:
        schema = self._table_schema_dict.get(table_name)
        if (schema is not None and col_name not in schema.mapper_dict and
                col_name in (schema.time_col_name, schema.end_time_col_name)):
            # Temporal columns are exposed to explanation without being
            # ordinary encoded features. Keep the model's TensorFrame schema
            # unchanged while supplying the collector's timestamp metadata.
            return TensorMapper.from_encoder(Datetime())
        return super().get_tensor_mapper(table_name, col_name)

    def _cache_fused_features(self, type_name: str, arrow_table: pa.Table,
                              node_ids: torch.Tensor) -> None:
        self._fused_cache[type_name] = (arrow_table, node_ids)

    def _clear_fused_cache(self) -> None:
        self._fused_cache.clear()

    def _read_features_by_ids(self, table_name: str, node_ids: torch.Tensor):
        """Read features for specific node IDs from the diskgraph index
        via the ``fetch_features`` pybind API.

        Used as the cache-miss fallback when callers like
        ``RHSEmbedding.fetch_feats`` request features outside the
        sampling context.
        """
        if self._sampler_engine is None:
            raise RuntimeError(
                "DiskgraphFeatureStore has no sampler engine reference. "
                "Call set_sampler_engine() after sampler creation.")

        engine = self._sampler_engine
        mapper_dict = self._table_schema_dict[table_name].mapper_dict
        feature_names = list(mapper_dict.keys())
        ids_list = node_ids.cpu().tolist()

        # Batch large requests to avoid overwhelming the engine
        BATCH = 50_000
        all_feature_batches = []
        for start in range(0, len(ids_list), BATCH):
            chunk = ids_list[start:start + BATCH]
            request = [{
                "node_type": table_name,
                "node_ids": chunk,
                "feature_names": feature_names,
            }]
            for event in engine.fetch_features(request):
                if event["type"] == "feature":
                    all_feature_batches.append(event["batch"])

        if not all_feature_batches:
            return self._get_fused_tf_empty(table_name)

        table = pa.Table.from_batches(all_feature_batches)
        mapper_dict = self._table_schema_dict[table_name].mapper_dict
        relevant = {
            k: v
            for k, v in mapper_dict.items() if k in table.column_names
        }

        if not relevant:
            return super()._get_tf(table_name, None)
        return get_tf(table.select(list(relevant.keys())), relevant)

    def _get_fused_tf_empty(self, table_name):
        """Return an empty TensorFrame for a table with zero sampled nodes.

        Used when a schema type is included in the batch with an empty
        node tensor (zero nodes sampled) — no features to look up, but
        the caller expects a TensorFrame with the correct schema.
        """
        schema = self._table_schema_dict.get(table_name)
        mapper_dict = schema.mapper_dict if schema else {}
        df = pd.DataFrame(
            {col: pd.Series(dtype="object")
             for col in mapper_dict})
        return get_tf(df, mapper_dict)

    def _get_tensor(self, attr):
        # Router: check fused cache first for tf/time/pkey, then handle
        # empty indices, then fall back to out-of-core fetch for non-train
        # tables.

        # --- tf (TensorFrame features) ---
        if attr.attr_name == "tf":
            # Fused cache hit
            if attr.table_name in self._fused_cache:
                _, cached_ids = self._fused_cache[attr.table_name]
                requested = (attr.index if hasattr(attr, "index")
                             and attr.index is not None else None)
                # Validate that the cached data matches the requested nodes.
                # A mismatch means stale data from a previous sample — evict
                # and fall through to the cache-miss path.
                if requested is not None and not (
                        requested.shape == cached_ids.shape
                        and torch.equal(requested.long(), cached_ids.long())):
                    logger.warning(
                        "Fused cache eviction for '%s': "
                        "requested.shape=%s, cached.shape=%s, "
                        "requested[:5]=%s, cached[:5]=%s",
                        attr.table_name,
                        requested.shape,
                        cached_ids.shape,
                        requested[:5].tolist(),
                        cached_ids[:5].tolist(),
                    )
                    self._fused_cache.pop(attr.table_name)
                else:
                    return self._get_fused_tf(attr.table_name)

            # Empty index → empty TensorFrame (unsampled schema type)
            if (hasattr(attr, "index") and attr.index is not None
                    and len(attr.index) == 0):
                return self._get_fused_tf_empty(attr.table_name)

            # Cache miss for graph table — fall through to out-of-core fetch.

            # Non-strict cache miss → out-of-core fetch.
            if attr.table_name != self.train_table_name:
                if (hasattr(attr, "index") and attr.index is not None
                        and len(attr.index) > 0):
                    return self._fetch_out_of_core_feature(
                        attr.table_name, attr.index)

        # --- time / end_time ---
        if attr.attr_name in ("time", "end_time"):
            if attr.table_name in self._fused_cache:
                if attr.attr_name == "time":
                    return self._get_fused_time(attr)
                # end_time from fused cache
                return self._get_fused_end_time(attr)
            # Cache miss — fetch from DG index directly.
            if attr.table_name != self.train_table_name:
                idx = getattr(attr, "index", None)
                if idx is None or (isinstance(idx, slice)
                                   and idx == slice(None)):
                    # Full table request — get all node IDs
                    num_nodes = self._diskgraph_num_nodes.get(
                        attr.table_name, 0)
                    if num_nodes > 0:
                        node_ids = torch.arange(num_nodes, dtype=torch.long)
                        schema = self._table_schema_dict[attr.table_name]
                        col_name = (schema.time_col_name if attr.attr_name
                                    == "time" else schema.end_time_col_name)
                        if col_name:
                            return self._fetch_out_of_core_time_col(
                                attr.table_name, node_ids, col_name)
                elif hasattr(idx, '__len__') and len(idx) > 0:
                    col_name = (
                        self._table_schema_dict[attr.table_name].time_col_name
                        if attr.attr_name == "time" else self.
                        _table_schema_dict[attr.table_name].end_time_col_name)
                    if col_name:
                        return self._fetch_out_of_core_time_col(
                            attr.table_name, idx, col_name)

        # --- pkey ---
        if attr.attr_name == "pkey":
            idx = getattr(attr, "index", None)
            if idx is not None and attr.table_name in self._fused_cache:
                return DiskgraphFeatureStore._get_fused_pkey(self, attr)
            # Cache miss — fetch pkey from DG index directly.
            if attr.table_name != self.train_table_name:
                schema = self._table_schema_dict.get(attr.table_name)
                pkey_name = schema.pkey_name if schema else None
                if pkey_name:
                    if idx is None or (isinstance(idx, slice)
                                       and idx == slice(None)):
                        num_nodes = self._diskgraph_num_nodes.get(
                            attr.table_name, 0)
                        if num_nodes > 0:
                            node_ids = torch.arange(num_nodes, dtype=torch.long)
                            return self._fetch_out_of_core_pkey(
                                attr.table_name, node_ids, pkey_name)
                    elif hasattr(idx, '__len__') and len(idx) > 0:
                        return self._fetch_out_of_core_pkey(
                            attr.table_name, idx, pkey_name)

        return super()._get_tensor(attr)

    def _get_tensor_size(self, attr):
        if (attr.table_name != self.train_table_name
                and attr.table_name in self._diskgraph_num_nodes):
            return (self._diskgraph_num_nodes[attr.table_name], )
        return super()._get_tensor_size(attr)

    def _get_fused_pkey(self, attr):
        """Return primary key values for sampled nodes.

        If the fused cache contains an Arrow table with the pkey column,
        read the string PK from it (with ``.copy()`` to detach from the
        Arrow buffer).  Otherwise fall back to exposing the integer
        global node IDs directly.
        """
        table_name = attr.table_name
        idx = attr.index

        if table_name in self._fused_cache:
            arrow_table, _ = self._fused_cache[table_name]
            schema = self._table_schema_dict.get(table_name)
            pkey_name = schema.pkey_name if schema else None
            if pkey_name and pkey_name in arrow_table.column_names:
                col = arrow_table.column(pkey_name)
                return col.to_pandas().values.copy()

        # Fallback: sampled node ids are already global node ids.
        if isinstance(idx, torch.Tensor):
            return idx.numpy()
        return idx

    def _fetch_out_of_core_feature(self, table_name: str,
                                   node_ids: torch.Tensor):
        """Lazily create an engine and fetch features via fetch_features API.

        Used for non-train table cache misses (e.g. side lookups in
        RHSEmbedding.fetch_feats).  Falls back to the sampler engine if
        available, otherwise creates a dedicated engine from index_path.
        """
        if self._sampler_engine is not None:
            return self._read_features_by_ids(table_name, node_ids)

        if self._index_path is None:
            raise KeyError("DiskgraphFeatureStore expected fused features for "
                           f"table '{table_name}', but none were cached for "
                           "this batch and no index_path is available for "
                           "out-of-core reads.")

        if self._engine is None:
            from diskgraph import SamplerEngine

            self._engine = SamplerEngine(self._index_path)

        engine = self._engine
        mapper_dict = self._table_schema_dict[table_name].mapper_dict
        feature_names = list(mapper_dict.keys())
        ids_list = node_ids.cpu().tolist()

        BATCH = 50_000
        all_feature_batches = []
        for start in range(0, len(ids_list), BATCH):
            chunk = ids_list[start:start + BATCH]
            request = [{
                "node_type": table_name,
                "node_ids": chunk,
                "feature_names": feature_names,
            }]
            for event in engine.fetch_features(request):
                if event["type"] == "feature":
                    all_feature_batches.append(event["batch"])

        if not all_feature_batches:
            return self._get_fused_tf_empty(table_name)

        table = pa.Table.from_batches(all_feature_batches)
        relevant = {
            k: v
            for k, v in mapper_dict.items() if k in table.column_names
        }

        if not relevant:
            return super()._get_tf(table_name, None)
        return get_tf(table.select(list(relevant.keys())), relevant)

    def _get_fused_tf(self, table_name):
        """Convert cached Arrow table to TensorFrame.

        Fused features arrive in local_id order from the engine — row i
        corresponds to local_id i, which matches the ``node_dict``
        ordering from ``HeteroSamplerOutput``.  The caller
        (``_get_tensor``) validates that the cached node IDs match
        the requested index before calling this method.
        """
        arrow_table, _ = self._fused_cache[table_name]
        mapper_dict = self._table_schema_dict[table_name].mapper_dict
        relevant = {
            k: v
            for k, v in mapper_dict.items() if k in arrow_table.column_names
        }
        if not relevant:
            logger.warning(
                "Fused features for '%s' have no columns matching mapper_dict "
                "(fused_cols=%s, mapper_cols=%s). Falling back to in-memory.",
                table_name,
                list(arrow_table.column_names),
                list(mapper_dict.keys()),
            )
            return super()._get_tf(table_name, None)
        return get_tf(arrow_table.select(list(relevant.keys())), relevant)

    def _get_fused_time(self, attr):
        """Return fused time tensor aligned with sampled subgraph rows."""
        table_name = attr.table_name
        arrow_table, _ = self._fused_cache[table_name]
        time_col = self._table_schema_dict[table_name].time_col_name
        if not time_col or time_col not in arrow_table.column_names:
            return super()._get_tensor(attr)

        ser = pd.to_datetime(
            arrow_table.column(time_col).to_pandas(), errors="coerce")
        vals = _datetime_series_to_ns(ser)
        return torch.from_numpy(vals)

    def _get_fused_end_time(self, attr):
        """Return fused end_time tensor from cached Arrow table."""
        table_name = attr.table_name
        arrow_table, _ = self._fused_cache[table_name]
        end_time_col = self._table_schema_dict[table_name].end_time_col_name
        if not end_time_col or end_time_col not in arrow_table.column_names:
            return super()._get_tensor(attr)
        ser = pd.to_datetime(
            arrow_table.column(end_time_col).to_pandas(), errors="coerce")
        vals = _datetime_series_to_ns(ser)
        return torch.from_numpy(vals)

    def _fetch_out_of_core_time_col(self, table_name: str,
                                    node_ids: torch.Tensor,
                                    col_name: str) -> torch.Tensor:
        """Fetch a time/end_time column for specific node IDs from the DG index.

        Used when time attributes are requested outside of a sampling context
        (e.g. explain step, create_datamodule for link prediction).
        """
        if self._index_path is None:
            raise KeyError(f"DiskgraphFeatureStore: '{col_name}' requested for "
                           f"'{table_name}' but fused cache is empty and no "
                           "index_path available.")

        if self._engine is None:
            from diskgraph import SamplerEngine
            self._engine = SamplerEngine(self._index_path)

        ids_list = node_ids.cpu().tolist()

        BATCH = 50_000
        all_batches = []
        for start in range(0, len(ids_list), BATCH):
            chunk = ids_list[start:start + BATCH]
            request = [{
                "node_type": table_name,
                "node_ids": chunk,
                "feature_names": [col_name],
            }]
            for event in self._engine.fetch_features(request):
                if event["type"] == "feature":
                    all_batches.append(event["batch"])

        if not all_batches:
            return torch.zeros(len(ids_list), dtype=torch.long)

        table = pa.Table.from_batches(all_batches)
        if col_name not in table.column_names:
            return torch.zeros(len(ids_list), dtype=torch.long)

        ser = pd.to_datetime(
            table.column(col_name).to_pandas(), errors="coerce")
        vals = _datetime_series_to_ns(ser)
        return torch.from_numpy(vals)

    def _fetch_out_of_core_pkey(self, table_name: str, node_ids: torch.Tensor,
                                pkey_name: str) -> np.ndarray:
        """Fetch primary key column for specific node IDs from the DG index."""
        if self._index_path is None:
            raise KeyError(
                f"DiskgraphFeatureStore: pkey requested for '{table_name}' "
                "but fused cache is empty and no index_path available.")

        if self._engine is None:
            from diskgraph import SamplerEngine
            self._engine = SamplerEngine(self._index_path)

        ids_list = node_ids.cpu().tolist()

        BATCH = 50_000
        all_batches = []
        for start in range(0, len(ids_list), BATCH):
            chunk = ids_list[start:start + BATCH]
            request = [{
                "node_type": table_name,
                "node_ids": chunk,
                "feature_names": [pkey_name],
            }]
            for event in self._engine.fetch_features(request):
                if event["type"] == "feature":
                    all_batches.append(event["batch"])

        if not all_batches:
            return np.arange(len(ids_list))

        table = pa.Table.from_batches(all_batches)
        if pkey_name not in table.column_names:
            return np.arange(len(ids_list))

        return table.column(pkey_name).to_pandas().values.copy()

    def _multi_get_tensor(self, attrs):
        """Override to eagerly clean up fused cache after all attrs are read.

        The fused cache holds large Arrow tables from the sampling engine.
        Cleaning up in a finally block ensures memory is released even if
        one of the ``_get_tensor`` calls raises.
        """
        try:
            return [self._get_tensor(attr) for attr in attrs]
        finally:
            self._clear_fused_cache()
