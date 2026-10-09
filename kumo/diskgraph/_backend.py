# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DiskGraph identity for the Kumo diskgraph layer.

Defined here (not in `kumo/diskgraph/__init__.py`) so that `kumo/diskgraph/`
submodules can import these without creating a circular dependency on
`kumo/diskgraph/__init__.py`.

Everything here is pure data — no pybind, no workflow orchestration.
"""

from __future__ import annotations

import os
from typing import Optional

from io_utils import kumo_dataclass

# ---------------------------------------------------------------------------
# DiskgraphHandle
# ---------------------------------------------------------------------------
#
# Defined here (not in kumo/diskgraph/service.py) to avoid a circular
# import: other kumo.diskgraph submodules need :class:`DiskgraphHandle`
# without pulling in the service module's heavier deps. The service module
# re-exports it so the public API is unchanged: callers still do
# ``from kumo.diskgraph import DiskgraphHandle``.


@kumo_dataclass(frozen=True)
class DiskgraphHandle:
    """Opaque identity for a DiskGraph snapshot, optionally bound to a
    materialized task dataset on top of that snapshot.

    Carries ``(graph_key, graph_data_version)`` — and, when a particular
    task dataset has been materialized, the opaque ``job_id`` of that
    materialization. The service impl is responsible for mapping the
    handle to wherever things physically live: the snapshot path AND, if
    ``job_id`` is set, the manifest dir (locally or downloaded from s3).

    In the remote/pool world the same handle may route to different pods
    on different calls; callers must not assume any stable physical
    location across calls.

    Frozen: handles are immutable and hashable. Must be JSON-serializable
    — Temporal activities receive handles as arguments.
    """
    graph_key: str
    graph_data_version: str
    job_id: Optional[str] = None

    @classmethod
    def for_test(
        cls,
        graph_key: str = 'NA-test',
        graph_data_version: str = 'v0',
    ) -> 'DiskgraphHandle':
        """Standard test-fixture constructor."""
        return cls(graph_key=graph_key, graph_data_version=graph_data_version)

    @classmethod
    def from_index_path(cls, index_path: str) -> 'DiskgraphHandle':
        """Parse a ``{data_root}/{graph_key}/{graph_data_version}`` path.

        The path must have at least two components; callers are expected to
        pass a path built from that layout.
        """
        parts = os.path.normpath(index_path).split(os.sep)
        assert len(parts) >= 2, (
            f"diskgraph index path must have at least two components "
            f"({{graph_key}}/{{graph_data_version}}); got {index_path!r}")
        return cls(graph_key=parts[-2], graph_data_version=parts[-1])
