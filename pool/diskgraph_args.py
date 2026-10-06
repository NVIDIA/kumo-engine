# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed transport dataclass for diskgraph ingest args, in its own
module so the workflow side can import it without pulling in the rest
of :mod:`pool.activities` (which would hit an import cycle through
``rest.backends.ml_backend.activities``).
"""

from dataclasses import dataclass
from typing import Dict

from rest.config.data_backend import ConnectorExecutionRef
from rest.config.graph_v2 import GraphDefinition
from rest.config.table_v2 import TableDefinition


@dataclass(frozen=True)
class DiskgraphIngestArgs:
    """Daemon ``ensure_snapshot`` inputs."""
    graph_definition: GraphDefinition
    graph_display_name: str
    data_persist_dir: str
    table_data_versions: Dict[str, str]
    connector_specs: Dict[str, ConnectorExecutionRef]
    all_graph_tables: Dict[str, TableDefinition]
    source_size_bytes_by_table: Dict[str, int]
