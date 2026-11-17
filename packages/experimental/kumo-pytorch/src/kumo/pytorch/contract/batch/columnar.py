# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arrow graph outputs for consumers that assemble their own model payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
    """Selected edges in declared left-to-right relation orientation.

    ``index`` is an Arrow table with non-null uint32 or uint64 ``source`` and
    ``destination`` columns, indexing rows in the corresponding node tables.
    Endpoints always belong to the same example. Only edges selected by the
    sampling layers are included, never all links among returned nodes.
    Reverse traversals are reoriented, including self-relations. Chunks may
    repeat a pair selected in both directions. Consumers must normalize these
    pairs or request ``Task.batches(..., deduplicate_edges=True)`` to retain
    the first occurrence per relation, at the cost of edge-array allocation.
    Deduplication uses node positions, preserving separate examples even when
    entity IDs repeat. Chunk order is not traversal provenance.
    """

    source_type: str
    destination_type: str
    index: pa.Table


@dataclass(frozen=True)
class ColumnarGraph:
    """CPU Arrow graph buffers emitted by ``Task.batches(output_format="columnar")``.

    ``nodes`` uses table names and ``edges`` uses declared relation names.
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
