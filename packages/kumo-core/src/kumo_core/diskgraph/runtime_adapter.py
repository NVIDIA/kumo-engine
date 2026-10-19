# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter from a kumo-diskgraph native runtime batch to PyG ``HeteroData``.

Pure transform of the dict yielded by
``diskgraph.RuntimeIndex(...).iter(...)`` into a ``HeteroData`` (graph
topology + seed-table block) plus per-namespace raw feature tables. Without an
engine bundle, ``graphs`` / ``raw_features`` are empty and only the seed block
is populated.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, TypedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from kumo_core.diskgraph.metadata import normalize_diskgraph_edge_type
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.features.stores.in_memory_feature import get_tf
from kumo_core.encoders.target.base import TargetEncoder
from torch_frame import TensorFrame
from torch_frame.data import MultiNestedTensor
from torch_geometric.data import HeteroData
from torch_geometric.sampler.base import HeteroSamplerOutput

# A column value as handed across the diskgraph FFI boundary: the wheel
# always builds these through pyarrow (see diskgraph-pybind's py_*_array
# helpers), so this is Arrow-or-absent, never a plain Python list.
ArrowArray = pa.Array | pa.ChunkedArray | None

# A target value before ``_shape_target`` unwraps a still-multi-column
# RecordBatch (the wheel's multi-TARGET node-prediction shape).
TargetLike = pa.RecordBatch | ArrowArray

_CLASSIFICATION_TASK_TYPES = frozenset({
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "multilabel_ranking",
})

# LP targets are RHS node ids — must stay int64 (float32 loses ids > 2**24).
_LINK_PRED_TASK_TYPES = frozenset({
    "temporal_link_prediction",
    "static_link_prediction",
})

DEFAULT_GRAPH_CHANNEL = "main"

# mapper_key -> column-name -> ``TensorMapper``.
MapperBank = Mapping[str, Mapping[str, TensorMapper]]

# numpy datetime64 unit -> nanoseconds-per-tick. The engine stamps seed times +
# node ``time`` in its configured unit (``index_metadata.json::timestamp_unit``,
# default seconds); Kumo Python is ns-canonical, so the adapter multiplies by
# this factor to normalize engine-unit ticks -> nanoseconds at the boundary.
_NP_UNIT_TO_NS_FACTOR = {
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}

# --- FFI-boundary shapes -----------------------------------------------
# These TypedDicts document (without restructuring) the dicts the native
# ``diskgraph.RuntimeIndex(...).iter(...)`` yields. They stay dicts at
# runtime -- the wheel builds them, not this module -- but replace bare
# ``Any``/``Mapping[str, Any]`` typing with the actual optional-key shape.


class EdgeTypeKey(TypedDict):
    src: str
    rel: str
    dst: str


class EdgeIndexEntry(TypedDict):
    edge_type: EdgeTypeKey
    src: ArrowArray
    dst: ArrowArray


class EdgeCountsEntry(TypedDict):
    edge_type: EdgeTypeKey
    counts: ArrowArray


class GraphChannel(TypedDict, total=False):
    node_ids: dict[str, ArrowArray]
    batch_dict: dict[str, ArrowArray]
    edge_index_dict: list[EdgeIndexEntry]
    num_sampled_nodes_dict: dict[str, ArrowArray]
    num_sampled_edges_dict: list[EdgeCountsEntry]


class RawTask(TypedDict, total=False):
    input_ids: ArrowArray
    seed_node_ids: ArrowArray
    seed_timestamps: ArrowArray
    target: TargetLike
    target_mask: ArrowArray
    weight: ArrowArray
    tt_stage_node_ids: ArrowArray
    amount_m: int
    num_positives: int
    rhs_seed_type: str
    lp_pos_local: ArrowArray
    lp_pos_counts: ArrowArray
    lp_pos_rhs_global: ArrowArray
    lp_pos_mask: ArrowArray
    lp_pos_weight: ArrowArray
    lp_exclude_counts: ArrowArray
    lp_exclude_rhs_global: ArrowArray
    neg_mask: ArrowArray


class RawFeatureBlock(TypedDict, total=False):
    record_batches: list[pa.RecordBatch]
    node_ids: ArrowArray


class RuntimeBatch(TypedDict, total=False):
    task: RawTask
    graphs: dict[str, GraphChannel]
    raw_features: dict[str, RawFeatureBlock]


class NodeTypeMeta(TypedDict, total=False):
    mapper_key: str
    feature_columns: list[str]


class FeatureSchema(TypedDict, total=False):
    node_types: dict[str, NodeTypeMeta]


# --- Adapter-owned shapes ------------------------------------------------
# Unlike the TypedDicts above, this module builds + consumes these itself,
# so they're real dataclasses rather than dict shims.


@dataclass(slots=True)
class NormalizedTask:
    """The FFI task block, resolved to one shape and one flat namespace.

    One dataclass rather than ``SeedRef`` / ``Supervision`` / ``TaskIdentity``
    sub-objects: every field here is a leaf value the wheel may or may not
    send, addressed the same way regardless of which of "seed", "supervision"
    or "identity" a reader would file it under -- so the grouping bought
    nothing a reader used, and cost an extra ``.seed_ref.`` /
    ``.supervision.`` /  ``.identity.`` hop at every call site. ``slots=True``
    is what makes the ragged-field loop in ``_normalize_task_payload`` safe:
    a renamed or misspelled key there raises ``AttributeError`` instead of
    silently creating a new attribute nothing reads.
    """
    input_ids: ArrowArray
    seed_node_id: ArrowArray
    seed_time: ArrowArray = None
    tt_stage_node_ids: ArrowArray = None
    target: ArrowArray = None
    target_mask: ArrowArray = None
    weight: ArrowArray = None
    lp_pos_counts: ArrowArray = None
    lp_pos_rhs_global: ArrowArray = None
    lp_pos_mask: ArrowArray = None
    lp_pos_weight: ArrowArray = None
    lp_exclude_counts: ArrowArray = None
    lp_exclude_rhs_global: ArrowArray = None


@dataclass
class RawFeatures:
    tables: dict[str, pa.RecordBatch] = field(default_factory=dict)
    row_ids: dict[str, torch.Tensor] = field(default_factory=dict)


def engine_int64_to_ns(arr: "pa.Array | pa.ChunkedArray",
                       index_np_time_unit: str) -> "pa.Array":
    """Scale an int64 engine-unit timestamp Arrow array to int64 NANOSECONDS,
    vectorized. ``ns`` is a no-op. NULLs are preserved (propagate through
    ``pa.compute.multiply``)."""
    import pyarrow.compute as pc
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    factor = _NP_UNIT_TO_NS_FACTOR.get(index_np_time_unit, 1)
    if factor == 1:
        return arr
    return pc.multiply(arr.cast(pa.int64()), pa.scalar(factor, type=pa.int64()))


def runtime_batch_to_hetero(
    batch: RuntimeBatch,
    *,
    train_table_name: str,
    channel: str = DEFAULT_GRAPH_CHANNEL,
    mapper_bank: MapperBank | None = None,
    feature_schema: FeatureSchema | None = None,
    task_type: str | None = None,
    seed_node_type: str | None = None,
    pk_col_dict: Mapping[str, str] | None = None,
    time_col_dict: Mapping[str, str] | None = None,
    target_encoder: TargetEncoder | None = None,
    index_np_time_unit: str = "s",
    bidirectional_output: bool = False,
    attach_pkeys: bool = True,
) -> tuple[HeteroData, dict[str, pa.RecordBatch], dict[str, torch.Tensor]]:
    """Convert one native runtime batch into ``HeteroData`` + raw features.

    When ``mapper_bank`` + ``feature_schema`` are supplied, ``hetero[ns].tf`` is
    populated via ``get_tf``; raw tables are returned for the explainability
    trace. ``task_type`` selects the seed-block ``y`` shape (``[N, 1]`` int64 /
    float32 for classification / regression; 1-D float32 when ``None``).

    ``index_np_time_unit`` is the engine's configured timestamp unit (numpy
    datetime64 unit, default ``'s'``). Engine seed times and node ``time`` columns
    arrive in that unit; the adapter normalizes BOTH to int64 NANOSECONDS (Kumo's
    canonical unit) so the temporal encoder's ``rel_time = node.time - seed_time``
    and the temporal-LP RHS validity are all-ns.
    """
    if (mapper_bank is None) != (feature_schema is None):
        raise ValueError(
            "mapper_bank and feature_schema must both be provided or "
            "both be None")

    hetero = HeteroData()

    task = _normalize_task_payload(batch["task"])
    main = _normalize_graph_channel(_channel(batch, channel), task,
                                    seed_node_type=seed_node_type)
    # Limit the back-fill to graph schema node types (see _attach_graph_topology).
    # Only available on the runtime path (feature_schema present); ``None`` =
    # legacy behavior.
    allowed_node_types: frozenset[str] | None = None
    if feature_schema is not None:
        allowed_node_types = frozenset((feature_schema.get("node_types")
                                        or {}).keys())
    _attach_graph_topology(
        hetero,
        main,
        bidirectional_output=bidirectional_output,
        allowed_node_types=allowed_node_types,
    )
    _attach_seed_table(
        hetero,
        task,
        train_table_name,
        task_type=task_type,
        target_encoder=target_encoder,
        index_np_time_unit=index_np_time_unit,
    )
    _attach_lp_indices(hetero, batch["task"], train_table_name,
                       seed_node_type=seed_node_type)
    # evaluation_step reads batch[entity].batch_size to slice embeddings to the
    # seed rows; PyG doesn't auto-derive it, so set it from len(input_ids).
    batch_size = _value_len(task.input_ids)
    # Set on every node type in the batch, including parity back-filled empty
    # blocks (_attach_graph_topology), so the set matches PyG.
    for node_type in hetero.node_types:
        hetero[node_type].batch_size = batch_size

    # LP modules read supervision off the LHS entity block (== seed_node_type),
    # not the train_table placeholder. Mirror the tensors onto the entity block
    # so get_ground_truth / common_step find .y / .seed_time / .weight / .target_mask.
    if (task_type in _LINK_PRED_TASK_TYPES and seed_node_type
            and seed_node_type in hetero.node_types):
        seed_tbl = hetero[train_table_name]
        entity_block = hetero[seed_node_type]
        for _attr in ("y", "seed_time", "weight", "target_mask"):
            _val = getattr(seed_tbl, _attr, None)
            if _val is not None:
                setattr(entity_block, _attr, _val)
        # Ragged multi-positive COO on the entity + seed blocks.
        _attach_lp_edge_label_index(hetero, task, train_table_name,
                                    seed_node_type)
        # Static-eval past-visit exclude set (TT-NG only; no-op otherwise).
        _attach_lp_exclude(hetero, task, train_table_name, seed_node_type)

    raw_features = _collect_raw_features(
        batch.get("raw_features", {}) or {},
        main.get("node_ids", {}) or {},
    )
    _alias_forecasting_seed_features(
        raw_features,
        train_table_name=train_table_name,
        mapper_bank=mapper_bank,
        feature_schema=feature_schema,
        task_type=task_type,
    )

    if mapper_bank is not None and feature_schema is not None:
        attach_tensor_frames(
            hetero,
            raw_features.tables,
            mapper_bank,
            feature_schema,
        )
        _attach_placeholder_tensor_frames(hetero, main)

    _attach_pkeys_and_times(
        hetero,
        raw_features.tables,
        pk_col_dict or {},
        time_col_dict or {},
        index_np_time_unit=index_np_time_unit,
        attach_pkeys=attach_pkeys,
    )

    # SubgraphCollector resolves batch.input_type to the first node_type with an
    # input_id attr, which must be the entity namespace (not the seed table) —
    # so stamp input_id on the entity block here.
    if seed_node_type and seed_node_type in hetero.node_types:
        tt_stage_node_ids = task.tt_stage_node_ids
        if tt_stage_node_ids is not None:
            hetero[seed_node_type].input_id = _values_to_tensor(
                tt_stage_node_ids, dtype=torch.long)
        elif task.input_ids is not None:
            hetero[seed_node_type].input_id = _values_to_tensor(
                task.input_ids, dtype=torch.long)

    return hetero, raw_features.tables, raw_features.row_ids


def attach_tensor_frames(
    hetero: HeteroData,
    raw_feature_tables: Mapping[str, pa.RecordBatch],
    mapper_bank: MapperBank,
    feature_schema: FeatureSchema,
) -> None:
    """Lower per-namespace raw ``RecordBatch``es into ``TensorFrame`` as
    ``hetero[ns].tf`` (in place). Namespaces with no ``mapper_key`` are skipped;
    the placeholder pass fills them.

    A schema node type that is present in ``hetero`` (e.g. a parity back-filled
    empty block from ``_attach_graph_topology``) but has NO raw RecordBatch
    still gets a MAPPER-SHAPED empty TF here: ``_record_batch_to_tf`` builds the
    encoder-correct ``feat_dict`` stypes at zero rows from the mapper. Without
    this the placeholder pass would attach a bare ``TensorFrame({}, {})`` whose
    empty ``feat_dict`` makes the feature encoder ``tf.feat_dict[stype]``
    KeyError (tensor_frame_encoder.py:240) for a type the encoder must encode.
    """
    nodes_meta = feature_schema.get("node_types") or {}
    for namespace, record_batch in raw_feature_tables.items():
        node_meta = nodes_meta.get(namespace) or {}
        mapper_key = node_meta.get("mapper_key")
        if mapper_key is None:
            continue
        if mapper_key not in mapper_bank:
            raise KeyError(
                f"feature_schema references mapper_key={mapper_key!r} "
                f"for namespace {namespace!r}, but mapper_bank has no "
                f"such key.  Available: {sorted(mapper_bank)}")
        mapper_dict = dict(mapper_bank[mapper_key])
        hetero[namespace].tf = _record_batch_to_tf(
            record_batch, mapper_dict, namespace=namespace
        )

    # Schema types present (back-filled) but without raw features: build an
    # empty mapper-shaped TF so the encoder finds its expected stypes.
    for namespace in hetero.node_types:
        if "tf" in hetero[namespace] or namespace in raw_feature_tables:
            continue
        node_meta = nodes_meta.get(namespace) or {}
        mapper_key = node_meta.get("mapper_key")
        if mapper_key is None:
            continue  # no encoder for this type -> placeholder pass handles it
        if mapper_key not in mapper_bank:
            raise KeyError(
                f"feature_schema references mapper_key={mapper_key!r} "
                f"for namespace {namespace!r}, but mapper_bank has no "
                f"such key.  Available: {sorted(mapper_bank)}")
        mapper_dict = dict(mapper_bank[mapper_key])
        empty_rb = pa.RecordBatch.from_pydict(
            {col: pa.array([], type=pa.string())
             for col in mapper_dict})
        hetero[namespace].tf = _record_batch_to_tf(
            empty_rb, mapper_dict, namespace=namespace
        )


def _attach_pkeys_and_times(
    hetero: HeteroData,
    raw_feature_tables: Mapping[str, pa.RecordBatch],
    pk_col_dict: Mapping[str, str],
    time_col_dict: Mapping[str, str],
    *,
    index_np_time_unit: str = "s",
    attach_pkeys: bool = True,
) -> None:
    """Decorate per-namespace blocks with ``.pkey`` / ``.time`` from
    ``raw_feature_tables[ns]``; PK / time column names come from the caller's
    ``pk_col_dict`` / ``time_col_dict``.

    ``attach_pkeys=False`` skips the ``.pkey`` decoration only. ``.time`` is
    always attached -- the temporal encoder needs it on every stage.

    TODO(wheel-gap, kumo-diskgraph#279): at the seed namespace the wheel's PK
    column is misaligned with ``task.seed_node_ids``; not compensated here —
    the fix lands wheel-side.
    """
    for namespace, record_batch in raw_feature_tables.items():
        column_names = set(record_batch.column_names)

        pk_col = pk_col_dict.get(namespace)
        if attach_pkeys and pk_col is not None and pk_col in column_names:
            pk_values = record_batch.column(pk_col)
            # numpy object array (not a list): supports both SubgraphCollector
            # patterns (fancy indexing + .tolist()) and string/numeric PKs.
            import numpy as np
            hetero[namespace].pkey = np.array(pk_values.to_pylist(),
                                              dtype=object)

        time_col = time_col_dict.get(namespace)
        if (time_col is not None and time_col in column_names
                and 'time' not in hetero[namespace]):
            # Normalize the node ``time`` column to int64 NANOSECONDS (Kumo
            # canonical), so the temporal encoder's ``rel_time = node.time -
            # seed_time`` is computed in a single (ns) unit. torch.as_tensor
            # rejects datetime64, so we go through int64.
            time_arr = record_batch.column(time_col)
            if isinstance(time_arr, pa.ChunkedArray):
                time_arr = time_arr.combine_chunks()
            if pa.types.is_timestamp(time_arr.type):
                # A real timestamp carries its resolution, so cast straight to
                # ns; this is resolution-stable regardless of the engine unit.
                time_arr = time_arr.cast(pa.timestamp("ns")).cast(pa.int64())
            else:
                # Bare int64 ticks in the engine's configured unit -> scale to ns.
                time_arr = engine_int64_to_ns(time_arr, index_np_time_unit)
            hetero[namespace].time = _values_to_tensor(time_arr,
                                                       dtype=torch.long)

    # The SubgraphCollector (XAI) reads ``batch[table].pkey`` for EVERY graph
    # table, but a given TT-NG seed only materializes features (hence pkeys)
    # for the tables it actually reaches; unreached tables appear as empty
    # back-fill blocks (see ``_attach_graph_topology``). Give each such empty,
    # pkey-bearing block an empty pkey array so the collector builds an empty
    # per-table frame (correct: zero nodes in this subgraph) rather than
    # crashing on a missing attribute. Non-empty blocks are never fabricated:
    # if one lacks a pkey it is a real gap and left to surface loudly.
    import numpy as np
    for namespace in hetero.node_types:
        block = hetero[namespace]
        if pk_col_dict.get(namespace) is None or "pkey" in block:
            continue
        n_id = getattr(block, "n_id", None)
        num_rows = int(n_id.numel()) if n_id is not None else 0
        if num_rows == 0:
            block.pkey = np.array([], dtype=object)


def _attach_placeholder_tensor_frames(
    hetero: HeteroData,
    main: GraphChannel,
) -> None:
    """Attach an empty TF for every node type with ``n_id`` but no ``tf``.

    Without it the HeteroGNN forward's ``x_src_dict[src]`` lookup raises
    ``KeyError``; the empty TF feeds a zero-channel input the encoder projects up.

    Iterates every node type in ``hetero`` (incl. parity back-filled empty
    blocks from ``_attach_graph_topology``) so a zero-length RHS block also gets
    its placeholder TF.
    """
    for node_type in hetero.node_types:
        block = hetero[node_type]
        if "tf" in block:
            continue
        n_id = getattr(block, "n_id", None)
        num_rows = int(n_id.numel()) if n_id is not None else 0
        block.tf = TensorFrame({}, {}, num_rows=num_rows)


def _record_batch_to_tf(
    record_batch: pa.RecordBatch,
    mapper_dict: dict[str, TensorMapper],
    *,
    namespace: str | None = None,
) -> TensorFrame:
    """Encode mapper columns, ignoring auxiliary time and primary key columns.

    Args:
        record_batch: Features and optional auxiliary columns for one namespace.
        mapper_dict: Fitted mappers defining the required encoder inputs.
        namespace: Node type included in missing-column diagnostics.

    Returns:
        Encoded features, or an empty frame for a type without fitted encoders.

    Raises:
        RuntimeError: If a required encoder column is absent from the batch.
    """
    missing = sorted(set(mapper_dict) - set(record_batch.schema.names))
    if missing:
        raise RuntimeError(
            f"feature batch for namespace {namespace or '<unknown>'!r} is "
            f"missing {len(missing)} column(s) the fitted encoders require: "
            f"{missing!r}. Present: {record_batch.schema.names!r}. Either the "
            "projection handed to RuntimeIndex.iter() omits an encoder "
            "column, or the graph snapshot lacks a fitted encoder column."
        )
    if not mapper_dict:
        return TensorFrame({}, {}, num_rows=0)
    return get_tf(record_batch, dict(mapper_dict))


def _channel(batch: RuntimeBatch, channel: str) -> GraphChannel:
    graphs = batch.get("graphs") or {}
    return graphs.get(channel) or {}


def _normalize_task_payload(task: RawTask) -> NormalizedTask:
    normalized = NormalizedTask(
        input_ids=task.get("input_ids"),
        seed_node_id=task.get("seed_node_ids"),
        seed_time=task.get("seed_timestamps"),
        tt_stage_node_ids=task.get("tt_stage_node_ids"),
    )

    # Set target only when one is present (predict has none). The wheel
    # emits ``task.target`` as a RecordBatch for multi-TARGET; unwrap the
    # single column to an Array for ``_shape_target``.
    target = task.get("target")
    if target is not None:
        if isinstance(target, pa.RecordBatch):
            # Ragged multi-positive carries no single target column (positives
            # live in ``lp_pos_*``); enforce the one-column shape only otherwise.
            if (target.num_columns != 1 and task.get("lp_pos_counts") is None):
                raise ValueError(
                    f"adapter expects single-TARGET RecordBatch; got "
                    f"{target.num_columns} columns: {target.schema.names}")
            if target.num_columns == 1:
                target = target.column(0)
                # With ``lp_pos_counts`` the positives go ragged via
                # ``lp_pos_*`` and ``.y`` stays unset; set target only otherwise.
                if task.get("lp_pos_counts") is None:
                    normalized.target = target
        else:
            normalized.target = target
        # target_mask: True == TARGET1 PK didn't resolve to an RHS node id;
        # carried so the LP module can drop those rows.
        if task.get("target_mask") is not None:
            normalized.target_mask = task.get("target_mask")
        if task.get("weight") is not None:
            normalized.weight = task.get("weight")

    # Ragged multi-positive LP fields (anchor-major), set independently of
    # the single-target path above. ``lp_exclude_*`` carry the static-eval
    # past-visit exclusion set (present only on TT-NG VAL/TEST static
    # batches). ``lp_pos_local`` is deliberately absent: ``_attach_lp_indices``
    # reads it off the raw ``task`` dict directly, so it has no normalized
    # field to land in.
    for _ragged_key in ("lp_pos_counts", "lp_pos_rhs_global", "lp_pos_mask",
                        "lp_pos_weight", "lp_exclude_counts",
                        "lp_exclude_rhs_global"):
        if task.get(_ragged_key) is not None:
            setattr(normalized, _ragged_key, task.get(_ragged_key))

    return normalized


def _normalize_graph_channel(
    main: GraphChannel,
    task: NormalizedTask,
    *,
    seed_node_type: str | None = None,
) -> GraphChannel:
    if main.get("node_ids") or not seed_node_type:
        return main
    seed_ids = task.seed_node_id
    if _value_len(seed_ids) == 0:
        return main
    return {
        **dict(main),
        "node_ids": {
            seed_node_type: seed_ids
        },
        "batch_dict": {
            seed_node_type: pa.array(range(_value_len(seed_ids)),
                                     type=pa.uint32())
        },
    }


def _value_len(values: ArrowArray) -> int:
    if values is None:
        return 0
    return len(values)


def _values_to_tensor(values: ArrowArray, *,
                      dtype: torch.dtype | None = None) -> torch.Tensor:
    if values is None:
        return torch.empty((0, ), dtype=dtype or torch.long)
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    # Arrow-backed numpy views are read-only; these tensors are only ever
    # read (indexed into / passed to ops), never written in place, so the
    # copy torch would otherwise recommend is unnecessary. Suppress the
    # warning rather than pay for a copy.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable",
            category=UserWarning,
        )
        tensor = torch.as_tensor(values.to_numpy(zero_copy_only=False))
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _values_to_list(values: ArrowArray) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    return values.to_pylist()


def _attach_graph_topology(
    hetero: HeteroData,
    main: GraphChannel,
    *,
    bidirectional_output: bool = False,
    allowed_node_types: frozenset[str] | None = None,
) -> None:
    sampler_out = _graph_channel_to_sampler_output(main)
    if bidirectional_output:
        # Legacy DiskGraph sampling calls HeteroSamplerOutput.to_bidirectional()
        # before materializing HeteroData. The TT-NG runtime yields a plain
        # pydict, so recreate that sampler-output surface and reuse PyG's
        # transform instead of duplicating its relation pairing logic.
        sampler_out = sampler_out.to_bidirectional(keep_orig_edges=True)

    for node_type, ids_tensor in sampler_out.node.items():
        hetero[node_type].n_id = ids_tensor

    # PARITY BACK-FILL (mirrors PyG ``filter_hetero_data``,
    # torch_geometric/loader/utils.py:180-186 "disconnected graph sampling"):
    # a batch's sampled neighborhood may not reach every SCHEMA node type, so
    # the engine returns ``node_ids`` for only the visited types. PyG always
    # materializes an EMPTY block for any unreached node type; the kumo adapter
    # historically did not, leaving e.g. ``node_id_dict[rhs]`` absent. That is a
    # pre-existing adapter gap (NOT introduced by ts-grouping) — it was dormant
    # because flat mixed-time batches almost always reach every type, but a
    # single-timestamp grouped batch at an early anchor can sample zero RHS
    # nodes, surfacing it as ``KeyError`` in the LP modules. Back-fill every
    # schema node type (derived from the engine's all-relations
    # ``num_sampled_edges_dict`` + any returned ``edge_index_dict``) with a
    # zero-length block so downstream code sees the same node-type set as PyG.
    schema_node_types: set[str] = set()
    for entry in main.get("num_sampled_edges_dict") or []:
        src, _, dst = _decode_sampled_edge_counts(entry)[0]
        schema_node_types.add(src)
        schema_node_types.add(dst)
    for entry in main.get("edge_index_dict") or []:
        edge_type, _ = _decode_edge_index(entry)
        schema_node_types.add(edge_type[0])
        schema_node_types.add(edge_type[2])
    # The engine reports a per-relation count for EVERY sampled relation,
    # including the TT-NG extension's supervision seed_table->entity relations
    # (endpoints like ``<tt_id>/train_seeds``). Those seed namespaces have no
    # encoder, so back-filling a block for them makes the feature encoder
    # ``enc_dict[node_type]`` KeyError (node-prediction; LP is unaffected, its
    # supervision rides the entity/RHS blocks). Restrict to the schema allow-set
    # when the caller knows it; ``None`` (legacy) keeps every type.
    if allowed_node_types is not None:
        schema_node_types &= allowed_node_types
    backfilled_node_types: list[str] = []
    for node_type in schema_node_types:
        if node_type in hetero.node_types and "n_id" in hetero[node_type]:
            continue
        hetero[node_type].n_id = _values_to_tensor(None, dtype=torch.long)
        backfilled_node_types.append(node_type)

    for node_type, batch_indices in (sampler_out.batch or {}).items():
        hetero[node_type].batch = batch_indices
    # Back-filled empty types also need an (empty) ``batch`` vector: the feature
    # encoder indexes ``batch_dict[node_type]`` for every type in ``tf_dict``
    # (tensor_frame_encoder.py:434), and the placeholder-TF pass below adds a TF
    # for these types — so a missing ``batch`` would KeyError there.
    for node_type in backfilled_node_types:
        hetero[node_type].batch = _values_to_tensor(None, dtype=torch.long)

    for edge_type, row in sampler_out.row.items():
        # Skip relations filtered out above: materializing the edge store would
        # resurrect the encoder-less seed namespace via PyG's implicit node store.
        if allowed_node_types is not None and (
                edge_type[0] not in allowed_node_types
                or edge_type[2] not in allowed_node_types):
            continue
        col = sampler_out.col[edge_type]
        if row.numel() == 0:
            hetero[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            hetero[edge_type].edge_index = torch.stack((row, col),
                                                       dim=0).contiguous()

    # Types the engine reported no per-hop counts for get one total:
    for node_type in hetero.node_types:
        if "num_sampled_nodes" in hetero[node_type]:
            continue
        if sampler_out.num_sampled_nodes is not None:
            n_id = getattr(hetero[node_type], "n_id", None)
            default = [int(n_id.numel()) if n_id is not None else 0]
            hetero[node_type].num_sampled_nodes = list(
                sampler_out.num_sampled_nodes.get(node_type, default))
        else:
            n_id = getattr(hetero[node_type], "n_id", None)
            hetero[node_type].num_sampled_nodes = [
                int(n_id.numel()) if n_id is not None else 0
            ]

    # Create per-hop empty edge stores for any schema relation whose BOTH
    # endpoint types are now present (incl. parity back-filled empty blocks) but
    # which was not materialized above — e.g. an all-zero-count relation between
    # two unsampled types. The engine reports counts for every schema relation;
    # dropping a zero-column edge whose endpoints exist would leave the HeteroGNN
    # / explainer indexing a missing store. Relations touching a still-absent
    # type are skipped (loss-free) to avoid a dangling ``x_dict`` KeyError.
    present_node_types = set(hetero.node_types)
    if sampler_out.num_sampled_edges is not None:
        for edge_type in sampler_out.num_sampled_edges:
            src_type, _, dst_type = edge_type
            if (src_type not in present_node_types
                    or dst_type not in present_node_types):
                continue
            store = hetero[edge_type]
            if "edge_index" not in store or store.edge_index is None:
                store.edge_index = torch.empty((2, 0), dtype=torch.long)

    for edge_type in hetero.edge_types:
        if (sampler_out.num_sampled_edges is not None
                and edge_type in sampler_out.num_sampled_edges):
            hetero[edge_type].num_sampled_edges = list(
                sampler_out.num_sampled_edges[edge_type])
            continue
        ei = hetero[edge_type].edge_index
        n_edges = int(ei.shape[1]) if ei is not None and ei.numel() else 0
        hetero[edge_type].num_sampled_edges = [n_edges]
    if sampler_out.orig_row is not None and sampler_out.orig_col is not None:
        for edge_type, row in sampler_out.orig_row.items():
            col = sampler_out.orig_col[edge_type]
            if edge_type not in hetero.edge_types:
                continue
            if row.numel() == 0:
                hetero[edge_type]._orig_edge_index = torch.empty(
                    (2, 0), dtype=torch.long)
            else:
                hetero[edge_type]._orig_edge_index = torch.stack(
                    (row, col), dim=0).contiguous()


def _graph_channel_to_sampler_output(
    main: GraphChannel, ) -> HeteroSamplerOutput:
    node = {
        node_type: _values_to_tensor(ids, dtype=torch.long)
        for node_type, ids in (main.get("node_ids") or {}).items()
    }
    batch = {
        node_type: _values_to_tensor(batch_indices, dtype=torch.long)
        for node_type, batch_indices in (main.get("batch_dict") or {}).items()
    }

    row: dict[tuple[str, str, str], torch.Tensor] = {}
    col: dict[tuple[str, str, str], torch.Tensor] = {}
    edge: dict[tuple[str, str, str], torch.Tensor | None] = {}
    for entry in main.get("edge_index_dict") or []:
        edge_type, edge_index = _decode_edge_index(entry)
        row[edge_type] = edge_index[0]
        col[edge_type] = edge_index[1]
        edge[edge_type] = None

    num_sampled_nodes = {
        node_type: _values_to_list(counts)
        for node_type, counts in (
            main.get("num_sampled_nodes_dict") or {}).items()
    }
    num_sampled_edges = {
        edge_type: counts
        for edge_type, counts in (_decode_sampled_edge_counts(entry)
                                  for entry in (
                                      main.get("num_sampled_edges_dict") or []))
    }

    # Create empty row/col stores for reported schema relations with both
    # endpoint node types present. This mirrors PyG disconnected-relation
    # handling and lets to_bidirectional pair empty/populated counterparts.
    present_node_types = set(node)
    for edge_type in num_sampled_edges:
        src_type, _, dst_type = edge_type
        if src_type not in present_node_types or dst_type not in present_node_types:
            continue
        row.setdefault(edge_type, torch.empty((0, ), dtype=torch.long))
        col.setdefault(edge_type, torch.empty((0, ), dtype=torch.long))
        edge.setdefault(edge_type, None)

    return HeteroSamplerOutput(
        node=node,
        row=row,
        col=col,
        edge=edge,
        batch=batch or None,
        num_sampled_nodes=num_sampled_nodes or None,
        num_sampled_edges=num_sampled_edges or None,
    )


def _decode_edge_index(
    entry: EdgeIndexEntry, ) -> tuple[tuple[str, str, str], torch.Tensor]:
    key = entry["edge_type"]
    edge_type = normalize_diskgraph_edge_type(key["src"], key["rel"],
                                              key["dst"])
    src = _values_to_tensor(entry.get("src"), dtype=torch.long)
    dst = _values_to_tensor(entry.get("dst"), dtype=torch.long)
    if src.numel() == 0:
        return edge_type, torch.empty((2, 0), dtype=torch.long)
    return edge_type, torch.stack((src, dst), dim=0).contiguous()


def _decode_sampled_edge_counts(
    entry: EdgeCountsEntry, ) -> tuple[tuple[str, str, str], list[Any]]:
    key = entry["edge_type"]
    edge_type = normalize_diskgraph_edge_type(key["src"], key["rel"],
                                              key["dst"])
    return edge_type, _values_to_list(entry.get("counts"))


def _attach_seed_table(
    hetero: HeteroData,
    task: NormalizedTask,
    train_table_name: str,
    *,
    task_type: str | None = None,
    target_encoder: TargetEncoder | None = None,
    index_np_time_unit: str = "s",
) -> None:
    input_ids = task.input_ids
    batch_size = _value_len(input_ids)
    seed_block = hetero[train_table_name]
    seed_block.input_id = _values_to_tensor(input_ids, dtype=torch.long)
    seed_block.batch = torch.arange(batch_size, dtype=torch.long)
    # Placeholder TF so the encoder emits ``x_dict[train_table_name]`` (used
    # only for batch_size, then dropped); all signal flows through the entity block.
    seed_block.tf = TensorFrame({}, {}, num_rows=batch_size)

    seed_time = task.seed_time
    if _all_present(seed_time):
        # Engine emits seed times in its configured unit; normalize to int64 ns
        # (Kumo-canonical) before stamping. Vectorized; no-op when already ns.
        seed_time = engine_int64_to_ns(seed_time, index_np_time_unit)
        seed_time_tensor = _values_to_tensor(seed_time, dtype=torch.long)
        seed_block.time = seed_time_tensor
        # Temporal LP reads the query anchor as ``seed_time`` on the seed block.
        seed_block.seed_time = seed_time_tensor

    target = task.target
    if target is not None and task_type != "forecasting":
        seed_block.y = _shape_target(target, task_type, target_encoder)
    target_mask = task.target_mask
    if target_mask is not None:
        # bool[B], True == TARGET1 PK did not resolve to an RHS node id.
        seed_block.target_mask = _values_to_tensor(target_mask,
                                                   dtype=torch.bool)
    weight = task.weight
    if _all_present(weight):
        seed_block.weight = _values_to_tensor(weight, dtype=torch.float32)


def _attach_lp_indices(
    hetero: HeteroData,
    task: RawTask,
    train_table_name: str,
    seed_node_type: str | None = None,
) -> None:
    """Attach two-tower LP index tensors from the FFI ``task`` dict.

    The RHS node array is seeds-first (``[B positives, B*M negatives,
    ...neighbors]``), so ``src_index`` / ``dst_pos_index`` / ``dst_neg_index`` /
    ``dst_neg_mask`` are built directly from ``num_positives`` (B) and
    ``amount_m`` (M). No-op when ``amount_m`` is absent / 0 (shallow / eval) or
    ``rhs_seed_type`` is not a real node type.
    """
    amount_m = task.get("amount_m")
    if amount_m is None or int(amount_m) <= 0:
        return
    rhs_seed_type = task.get("rhs_seed_type")
    if rhs_seed_type is None or rhs_seed_type not in hetero.node_types:
        return

    M = int(amount_m)
    B = int(task["num_positives"])

    # common_step reads ``src_index`` off the entity block (== seed_node_type),
    # not the train-table placeholder; keep it on the train-table block too.
    src_index = torch.arange(B, dtype=torch.long)
    hetero[train_table_name].src_index = src_index
    if seed_node_type is not None and seed_node_type in hetero.node_types:
        hetero[seed_node_type].src_index = src_index
    # lp_pos_local: engine's chosen rhs slot per anchor; else seeds-first arange(B).
    lp_pos_local = task.get("lp_pos_local")
    if lp_pos_local is not None:
        hetero[rhs_seed_type].dst_pos_index = _values_to_tensor(
            lp_pos_local, dtype=torch.long)
    else:
        hetero[rhs_seed_type].dst_pos_index = torch.arange(B, dtype=torch.long)
    hetero[rhs_seed_type].dst_neg_index = torch.arange(B, B + B * M,
                                                       dtype=torch.long).view(
                                                           B, M)
    neg_mask_tensor = _values_to_tensor(task.get("neg_mask"), dtype=torch.bool)
    if task.get("neg_mask") is not None:
        assert neg_mask_tensor.numel() == B * M, (
            f"dst_neg_mask length {neg_mask_tensor.numel()} != B*M "
            f"({B}*{M})")
    hetero[rhs_seed_type].dst_neg_mask = neg_mask_tensor.view(B, M)

    # Temporal two-tower: the RHS seed block is ``[B positives, B*M anchor-major
    # negatives]`` but its seed_time is only length B, so the temporal encoder
    # indexes out of bounds. Expand to the RHS layout (positives keep anchor
    # times; each negative repeats its anchor's). First B entries unchanged, so
    # the LHS tower is unaffected; static LP ignores seed_time so left ungated.
    seed_block = hetero[train_table_name]
    anchor_time = getattr(seed_block, "time", None)
    if anchor_time is not None and anchor_time.numel() == B:
        expanded = torch.cat([anchor_time, anchor_time.repeat_interleave(M)])
        seed_block.time = expanded
        if getattr(seed_block, "seed_time", None) is not None:
            seed_block.seed_time = expanded


def _attach_lp_edge_label_index(
    hetero: HeteroData,
    task: NormalizedTask,
    train_table_name: str,
    seed_node_type: str,
) -> None:
    """Build the ragged multi-positive ``edge_label_index`` COO from the
    anchor-major ``lp_pos_*`` fields.

    ``lhs_batch = repeat_interleave(arange(B), lp_pos_counts)``, ``rhs_index =
    lp_pos_rhs_global``, dropping rows where ``lp_pos_mask`` is True. Sets
    ``edge_label_index`` + filtered ``edge_weight`` on the entity + seed blocks.
    No-op when ``lp_pos_counts`` is absent (non-ragged path).
    """
    counts = task.lp_pos_counts
    rhs_global = task.lp_pos_rhs_global
    if counts is None or rhs_global is None:
        return

    counts_t = _values_to_tensor(counts, dtype=torch.long)
    rhs_index = _values_to_tensor(rhs_global, dtype=torch.long)
    B = counts_t.numel()
    lhs_batch = torch.repeat_interleave(torch.arange(B, dtype=torch.long),
                                        counts_t)

    pos_mask = task.lp_pos_mask
    if pos_mask is not None:
        keep = ~_values_to_tensor(pos_mask, dtype=torch.bool)
        lhs_batch = lhs_batch[keep]
        rhs_index = rhs_index[keep]
    else:
        keep = None

    edge_label_index = torch.stack([lhs_batch, rhs_index])

    edge_weight = None
    pos_weight = task.lp_pos_weight
    if pos_weight is not None:
        edge_weight = _values_to_tensor(pos_weight, dtype=torch.float32)
        if keep is not None:
            edge_weight = edge_weight[keep]

    entity_block = hetero[seed_node_type]
    seed_block = hetero[train_table_name]
    entity_block.edge_label_index = edge_label_index
    seed_block.edge_label_index = edge_label_index
    if edge_weight is not None:
        entity_block.edge_weight = edge_weight
        seed_block.edge_weight = edge_weight


def _expand_lp_exclude(
    counts: ArrowArray,
    rhs_global: ArrowArray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the anchor-major ragged exclude set into a flat (lhs_batch,
    rhs_index) pair, mirroring the ``lp_pos`` expansion."""
    counts_t = _values_to_tensor(counts, dtype=torch.long)
    rhs_index = _values_to_tensor(rhs_global, dtype=torch.long)
    B = counts_t.numel()
    lhs_batch = torch.repeat_interleave(torch.arange(B, dtype=torch.long),
                                        counts_t)
    return lhs_batch, rhs_index


def _attach_lp_exclude(
    hetero: HeteroData,
    task: NormalizedTask,
    train_table_name: str,
    seed_node_type: str,
) -> None:
    """Attach the static-eval past-visit exclude set as ``exclude_lhs_batch`` /
    ``exclude_rhs_index`` on the entity + seed blocks, from the anchor-major
    ``lp_exclude_*`` fields. The eval modules read these on the TT-NG path
    instead of ``get_past_visit_ground_truth``.

    No-op when ``lp_exclude_counts`` is absent (train / pred / temporal / legacy).
    """
    counts = task.lp_exclude_counts
    rhs_global = task.lp_exclude_rhs_global
    if counts is None or rhs_global is None:
        return

    exclude_lhs_batch, exclude_rhs_index = _expand_lp_exclude(
        counts, rhs_global)

    entity_block = hetero[seed_node_type]
    seed_block = hetero[train_table_name]
    entity_block.exclude_lhs_batch = exclude_lhs_batch
    entity_block.exclude_rhs_index = exclude_rhs_index
    seed_block.exclude_lhs_batch = exclude_lhs_batch
    seed_block.exclude_rhs_index = exclude_rhs_index


def _shape_target(
    target: ArrowArray,
    task_type: str | None,
    target_encoder: TargetEncoder | None = None,
) -> torch.Tensor | MultiNestedTensor:
    """Promote target to the shape expected by the Kumo task modules.

    ``target`` has already passed through ``_normalize_task_payload``, which
    unwraps any single-column TT-NG multi-TARGET ``RecordBatch`` into a plain
    Arrow array -- so this never sees a ``RecordBatch``.
    """
    # LP targets are int64 RHS node ids — keep 1-D; get_ground_truth reads
    # ``batch[lhs].y`` as node ids.
    if task_type in _LINK_PRED_TASK_TYPES:
        return _values_to_tensor(target, dtype=torch.long).view(-1)
    if task_type in {"multilabel_classification", "multilabel_ranking"}:
        return _shape_multilabel_target(target, target_encoder)

    if (task_type is not None and task_type in _CLASSIFICATION_TASK_TYPES
            and target_encoder is not None):
        return _categorical_indices(target, target_encoder).long().view(-1, 1)

    y = _arrow_array_to_tensor(target)

    if task_type is None:
        return y
    if task_type in _CLASSIFICATION_TASK_TYPES:
        return y.long().view(-1, 1)
    if y.dim() == 1:
        return y.to(dtype=torch.float32).view(-1, 1)
    return y.to(dtype=torch.float32)


def _shape_multilabel_target(
    target: ArrowArray,
    target_encoder: TargetEncoder | None,
) -> MultiNestedTensor:
    sep = None
    if target_encoder is not None:
        sep = getattr(target_encoder.tensor_mapper, "sep", None)
    values = _array_to_multilabel_lists(target, sep=sep or "|")
    if target_encoder is None:
        mapped = [[int(v) for v in row] for row in values]
    else:
        mapping = target_encoder.tensor_mapper.mapping
        mapped = []
        for row in values:
            labels, indices = _target_label_indices(list(row), mapping)
            if (indices < 0).any():
                unknown = [
                    labels[i] for i, idx in enumerate(indices) if idx < 0
                ]
                raise ValueError(
                    "TT-NG multilabel target contains labels not present in "
                    f"target encoder mapping: {unknown[:5]}")
            mapped.append([int(i) for i in indices])

    flat: list[int] = []
    offsets = [0]
    for row in mapped:
        flat.extend(row)
        offsets.append(len(flat))
    return MultiNestedTensor(
        num_rows=len(mapped),
        num_cols=1,
        values=torch.tensor(flat, dtype=torch.long),
        offset=torch.tensor(offsets, dtype=torch.long),
    )


def _coerce_numeric_label(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "" or stripped != value:
        return value
    try:
        int_value = int(stripped)
    except ValueError:
        try:
            float_value = float(stripped)
        except ValueError:
            return value
        return float_value if np.isfinite(float_value) else value
    return int_value if stripped == str(int_value) else value


def _target_label_indices(
    values: list[Any],
    mapping: pd.Index,
) -> tuple[list[Any], np.ndarray]:
    candidates = [values]
    string_values = [str(v) for v in values]
    if string_values != values:
        candidates.append(string_values)
    numeric_values = [_coerce_numeric_label(v) for v in values]
    if numeric_values != values:
        candidates.append(numeric_values)

    indices = mapping.get_indexer(values)
    for candidate in candidates:
        candidate_indices = mapping.get_indexer(candidate)
        if (candidate_indices >= 0).all():
            return candidate, candidate_indices
    return values, indices


def _categorical_indices(
    target: ArrowArray,
    target_encoder: TargetEncoder,
) -> torch.Tensor:
    values = _arrow_array_to_pylist(target)

    mapping = target_encoder.tensor_mapper.mapping
    values, indices = _target_label_indices(values, mapping)
    if (indices < 0).any():
        unknown = [values[i] for i, idx in enumerate(indices) if idx < 0]
        raise ValueError(
            "TT-NG categorical target contains labels not present in target "
            f"encoder mapping: {unknown[:5]}")
    return torch.tensor(indices, dtype=torch.long)


def _array_to_multilabel_lists(values: ArrowArray, *,
                               sep: str) -> list[list[Any]]:
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    py_values = values.to_pylist()

    out: list[list[Any]] = []
    for value in py_values:
        if value is None:
            out.append([])
        elif isinstance(value, str):
            out.append([v for v in value.split(sep) if v != ""])
        elif isinstance(value, (list, tuple, np.ndarray)):
            out.append([v for v in list(value) if v is not None])
        else:
            out.append([value])
    return out


def _arrow_array_to_tensor(values: ArrowArray) -> torch.Tensor:
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if isinstance(values, pa.Array) and pa.types.is_list(values.type):
        return torch.tensor(values.to_pylist(), dtype=torch.float32)
    return _values_to_tensor(values, dtype=torch.float32)


def _arrow_array_to_pylist(values: ArrowArray) -> list[Any]:
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    return values.to_pylist()


def _all_present(values: ArrowArray) -> bool:
    """``True`` iff ``values`` is a non-empty Arrow array with no ``None``
    entries. ``False`` for a bare ``ChunkedArray`` (callers only ever pass
    the un-chunked ``pa.Array`` the wheel emits)."""
    if values is None:
        return False
    if isinstance(values, pa.Array):
        return len(values) > 0 and values.null_count == 0
    return False


def _collect_raw_features(
    raw_features: Mapping[str, RawFeatureBlock],
    node_ids: Mapping[str, ArrowArray],
) -> RawFeatures:
    out = RawFeatures()
    for namespace, block in raw_features.items():
        record_batches = block.get("record_batches") or []
        if not record_batches:
            continue
        out.tables[namespace] = _concat_record_batches(record_batches)
        ids = block.get("node_ids")
        if ids is None:
            ids = node_ids.get(namespace)
        if ids is not None:
            out.row_ids[namespace] = _values_to_tensor(ids, dtype=torch.long)
    return out


def _alias_forecasting_seed_features(
    raw_features: RawFeatures,
    *,
    train_table_name: str,
    mapper_bank: MapperBank | None,
    feature_schema: FeatureSchema | None,
    task_type: str | None,
) -> None:
    raw_feature_tables = raw_features.tables
    raw_feature_row_ids = raw_features.row_ids
    if (task_type != "forecasting" or mapper_bank is None
            or feature_schema is None
            or train_table_name in raw_feature_tables):
        return

    nodes_meta = feature_schema.get("node_types") or {}
    seed_meta = nodes_meta.get(train_table_name)
    if not isinstance(seed_meta, Mapping):
        return
    mapper_key = seed_meta.get("mapper_key")
    expected = set(seed_meta.get("feature_columns") or [])
    if not expected and mapper_key in mapper_bank:
        expected = set(mapper_bank[mapper_key])
    if not expected:
        return

    candidates = [
        namespace for namespace, record_batch in raw_feature_tables.items()
        if namespace not in nodes_meta
        and expected.intersection(record_batch.column_names)
    ]
    if len(candidates) != 1:
        return
    namespace = candidates[0]
    raw_feature_tables[train_table_name] = raw_feature_tables[namespace]
    if namespace in raw_feature_row_ids:
        raw_feature_row_ids[train_table_name] = raw_feature_row_ids[namespace]


def _concat_record_batches(batches: list[pa.RecordBatch], ) -> pa.RecordBatch:
    if len(batches) == 1:
        return batches[0]
    table = pa.Table.from_batches(batches)
    combined = table.combine_chunks()
    out = combined.to_batches()
    if not out:
        # All-empty batches concatenate to nothing; return the first input.
        return batches[0]
    return out[0]
