# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import pytest
from kumo_core.diskgraph.sidecars import (
    canonical_graph_snapshot_id,
    job_sidecar_root,
)
from kumo_core.diskgraph.snapshot import DiskgraphIngestResult
from kumo_core.diskgraph.stats import compute_table_stats
from kumo_core.diskgraph.types import InputDataStats
from kumo_core.types import TableStatType


def test_ingest_result_carries_runtime_stats_without_changing_marker() -> None:
    result = DiskgraphIngestResult.for_index("/index")
    enriched = dataclasses.replace(
        result,
        pkey_duplicate_counts={"users": 2},
    )

    assert enriched.pkey_duplicate_counts == {"users": 2}
    assert "pkey_duplicate_counts" not in json.loads(enriched.to_json())
    assert (DiskgraphIngestResult.from_json(
        enriched.to_json()).pkey_duplicate_counts is None)


def test_portable_stats_and_input_shape() -> None:
    result = compute_table_stats(
        "/source",
        is_directory=lambda path: True,
        count_rows=lambda path: 7,
        file_size=lambda path: 0,
        directory_size=lambda path: 11 if path == "/source" else 13,
        index_path="/index",
    )
    assert result[TableStatType.NUM_ROWS].value == 7
    assert result[TableStatType.SOURCE_SIZE].value == 11
    assert result[TableStatType.MATERIALIZED_SIZE].value == 13
    assert InputDataStats(graph_num_nodes=7).json()["graph_num_nodes"] == 7


def test_portable_sidecar_identity_matches_legacy_layout() -> None:
    snapshot = "graph-1bf8d6a669f6a6caba68a2393f1044e3@v1"
    assert canonical_graph_snapshot_id(snapshot) == snapshot
    with pytest.raises(ValueError, match="must already use canonical"):
        canonical_graph_snapshot_id("NA-1bf8d6a669f6a6caba68a2393f1044e3@v1")
    assert job_sidecar_root(
        snapshot, "/persist") == ("/persist/diskgraph_job_sidecars/"
                                  "NA-1bf8d6a669f6a6caba68a2393f1044e3/v1")
