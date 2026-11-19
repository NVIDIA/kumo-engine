# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arrow graph outputs for consumers that assemble their own model payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import pyarrow as pa


@dataclass(frozen=True)
class ColumnarNodes:
    """Aligned columns for node occurrences, isolated per example.

    ``ids`` contains primary-key values (index-local row IDs when no primary
    key is declared). ``owners`` is a non-null uint32 or uint64 Arrow array of positions
    into ``Batch.metadata["sample_ids"]``. The same entity may occur under
    several owners. ``features`` is an Arrow table containing exactly the
    declared Raw fields, in declaration order and declared dtypes. Arrow
    validity bitmaps retain NULLs; no separate masks or filled values are used.
    With no declared fields, the table has zero columns; use ``len(ids)`` for
    the node count. All three outputs otherwise have the same row order.
    """

    ids: pa.Array | pa.ChunkedArray
    owners: pa.Array
    features: pa.Table


@dataclass(frozen=True)
class ColumnarEdges:
    """Selected edges in the graph topology's declared orientation.

    ``index`` is an Arrow table with non-null uint32 or uint64 ``source`` and
    ``destination`` columns, indexing rows in the corresponding node tables.
    Endpoints always belong to the same example. Only edges selected by the
    sampling layers are included, never all links among returned nodes.
    In relations topology, reverse traversals are reoriented, including
    self-relations. Chunks may
    repeat a pair selected in both directions. Consumers must normalize these
    pairs or request ``Task.batches(..., deduplicate_edges=True)`` to retain
    the first occurrence per relation, at the cost of edge-array allocation.
    Deduplication uses node positions, preserving separate examples even when
    entity IDs repeat. Chunk order is not traversal provenance. Sampled
    topology instead preserves native direction/order and exposes hop counts
    on ColumnarGraph; deduplication is refused for that topology.
    Sampled edges also carry the declared logical ``relation`` and traversal
    ``direction``. Those optional fields remain unset for relations output,
    whose logical identity is already the mapping key and orientation is normalized.
    """

    source_type: str
    destination_type: str
    index: pa.Table
    relation: str | None = None
    direction: Literal["forward", "reverse"] | None = None


@dataclass(frozen=True)
class ColumnarGraph:
    """CPU Arrow graph buffers emitted by ``Task.batches(output_format="columnar")``.

    ``nodes`` uses table names. With relations topology, ``edges`` uses declared
    relation names. Sampled topology uses native physical relation-name strings,
    retaining edge order and per-hop counts without reorienting or deduplicating
    edges. Its edge values expose the logical ``relation`` and traversal
    ``direction``, alongside endpoint types. A model adapter can construct
    ``(edge.source_type, key, edge.destination_type)`` without changing arrays.
    Counts use the same string keys as edges and align with
    the native node and edge ordering, not a newly computed shortest-path order.
    The first ``Batch.batch_size`` rows of ``nodes[root_type]`` are the roots,
    in batch order, including repeated entities at distinct prediction times.
    Isolated roots and empty declared node/edge types are retained.

    Buffers remain valid after advancing or closing the iterator/task. Treat
    them as read-only. Compatible native Arrow buffers are shared; declared
    dtype restoration may allocate. ``Batch.to`` leaves these CPU buffers in
    Arrow form. This output avoids tensor encoding and is intended for callers
    that construct their own model representation, such as foundation models.
    """

    root_type: str
    nodes: Mapping[str, ColumnarNodes]
    edges: Mapping[str, ColumnarEdges]
    num_sampled_nodes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    num_sampled_edges: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
