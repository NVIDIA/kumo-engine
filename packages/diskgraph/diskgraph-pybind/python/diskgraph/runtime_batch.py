# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mutable runtime sampling records owning zero-copy Arrow payloads."""

from dataclasses import dataclass, field

import pyarrow as pa


@dataclass(slots=True)
class RuntimeTask:
    """Seed identities, supervision and optional link-prediction labels.

    Optional arrays are absent for tasks that do not produce those labels.
    Arrow buffers retain their native owners for the lifetime of the arrays.

    Args:
        input_ids: Positions of the seed rows within this batch.
        tt_stage_node_ids: Canonical row IDs within the task stage.
        seed_node_ids: Corresponding parent-graph entity IDs.
        seed_timestamps: Seed cutoffs in the snapshot's clock unit.
        target: Single or multi-column supervision from the seed table.
        target_mask: Rows whose target IDs could not be resolved.
        weight: Optional per-seed loss weights.
        amount_m: Negatives per positive, or zero without negative sampling.
        num_positives: Number of positive seed rows.
        rhs_seed_type: Node type of the sampled RHS candidates.
        neg_mask: Validity of each anchor-major negative slot.
        lp_pos_counts: Number of positive labels per anchor.
        lp_pos_rhs_global: Concatenated global RHS IDs for positive labels.
        lp_pos_mask: Positive labels that must be excluded from the loss.
        lp_pos_weight: Optional weight for each positive label.
        lp_pos_local: Positive-label positions in the sampled RHS node array.
        lp_exclude_counts: Number of historical exclusions per anchor.
        lp_exclude_rhs_global: Concatenated global RHS IDs to exclude.
    """

    input_ids: pa.Array = field(default_factory=lambda: pa.array([], type=pa.uint32()))
    tt_stage_node_ids: pa.Array = field(
        default_factory=lambda: pa.array([], type=pa.uint64())
    )
    seed_node_ids: pa.Array = field(
        default_factory=lambda: pa.array([], type=pa.uint64())
    )
    seed_timestamps: pa.Array | None = None
    target: pa.RecordBatch | pa.Array | pa.ChunkedArray | None = None
    target_mask: pa.Array | None = None
    weight: pa.Array | None = None
    amount_m: int = 0
    num_positives: int = 0
    rhs_seed_type: str | None = None
    neg_mask: pa.Array | None = None
    lp_pos_counts: pa.Array | None = None
    lp_pos_rhs_global: pa.Array | None = None
    lp_pos_mask: pa.Array | None = None
    lp_pos_weight: pa.Array | None = None
    lp_pos_local: pa.Array | None = None
    lp_exclude_counts: pa.Array | None = None
    lp_exclude_rhs_global: pa.Array | None = None


@dataclass(slots=True)
class RuntimeEdgeType:
    """Physical relation labels from the graph snapshot.

    Args:
        src: Source node table name.
        rel: Relation identifier, including the sampled direction.
        dst: Destination node table name.
    """

    src: str
    rel: str
    dst: str


@dataclass(slots=True)
class RuntimeEdge:
    """One relation's aligned source and destination local indices.

    Args:
        edge_type: Physical relation whose endpoints these arrays index.
        src: Positions in the graph's source-type node array.
        dst: Corresponding positions in the destination-type node array.
    """

    edge_type: RuntimeEdgeType
    src: pa.Array
    dst: pa.Array


@dataclass(slots=True)
class RuntimeEdgeCounts:
    """Per-hop sampled edge counts for a relation, including empty relations.

    Args:
        edge_type: Physical relation whose counts are reported.
        counts: Number of edge occurrences sampled at each hop.
    """

    edge_type: RuntimeEdgeType
    counts: pa.Array


@dataclass(slots=True)
class RuntimeGraph:
    """Topology, seed provenance and per-hop counts for a graph channel.

    Args:
        node_ids: Sampled parent-graph IDs by node type.
        batch_dict: Owning seed position for each sampled node.
        edge_index_dict: Aligned local source/destination arrays by relation.
        num_sampled_nodes_dict: Per-hop sampled node counts by node type.
        num_sampled_edges_dict: Per-hop counts for every reachable relation.
    """

    node_ids: dict[str, pa.Array] = field(default_factory=dict)
    batch_dict: dict[str, pa.Array] = field(default_factory=dict)
    edge_index_dict: list[RuntimeEdge] = field(default_factory=list)
    num_sampled_nodes_dict: dict[str, pa.Array] = field(default_factory=dict)
    num_sampled_edges_dict: list[RuntimeEdgeCounts] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeFeatureBlock:
    """Arrow feature chunks for one namespace and their aligned sampled rows.

    Args:
        record_batches: Chunks in sampled-node order, potentially from several hops.
        key: Graph node type or task-stage feature namespace.
        num_rows: Total row count across all chunks.
        node_ids: Optional explicit row identities; otherwise use graph node IDs.
    """

    record_batches: list[pa.RecordBatch] = field(default_factory=list)
    key: str = ""
    num_rows: int = 0
    node_ids: pa.Array | None = None


@dataclass(slots=True)
class RuntimeBatch:
    """Complete sampled batch returned by native iteration and sample_request.

    Named graph channels and feature namespaces remain ordinary dictionaries.
    Containers can be changed without copying the Arrow payload buffers.
    ``metadata`` holds consumer annotations, independent of native sampling.

    Args:
        task: Seed identities and supervision.
        graphs: Topology channels keyed by their sampling-plan names.
        raw_features: Feature blocks keyed by graph/task namespace.
        metadata: Consumer-owned annotations, empty in native sampling output.
        history: Optional temporal-query output in its existing query format.
        history_merges: Event-stream names mapped to Arrow offsets/order pairs.
    """

    task: RuntimeTask = field(default_factory=RuntimeTask)
    graphs: dict[str, RuntimeGraph] = field(default_factory=dict)
    raw_features: dict[str, RuntimeFeatureBlock] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    history: object | None = None
    history_merges: dict[str, tuple[pa.Array, pa.Array]] = field(default_factory=dict)
