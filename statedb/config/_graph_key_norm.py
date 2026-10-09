# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared ``graph_key`` -> ``GraphIDv2`` normalization.

Lives in ``statedb/config/`` so both data_backend and ml_backend
workflows can import it without creating a cross-backend dependency
edge.
"""
import hashlib

from kumo_core.types.schema import FPID

from statedb.config.common import GraphIDv2


def _graph_key_to_v2(graph_key: str) -> GraphIDv2:
    """Normalize ``graph_key`` to a ``GraphIDv2`` for snapshot-id composition.

    ``DataSnapshot.GRAPH.canonicalize`` requires a parseable ``GraphIDv2``.
    Already-canonical keys (``"NA-<fpid>"``) are returned as-is; bare
    display names (e.g. ``"FINANCIAL"``) get a deterministic md5-derived
    fpid so concurrent ingests for the same ``(graph_key, version)`` pair
    collapse onto a single snapshot lock and SSD-cache key.
    """
    try:
        return GraphIDv2(graph_key)
    except ValueError:
        synthetic_fpid = FPID(hashlib.md5(graph_key.encode()).hexdigest())
        return GraphIDv2.compose(None, synthetic_fpid)
