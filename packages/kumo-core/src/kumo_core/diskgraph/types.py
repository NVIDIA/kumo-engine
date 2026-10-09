# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable values shared by DiskGraph runtimes and platform adapters."""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field


@dataclass
class TableStat:
    value: int | float
    approx: bool = False
    data_version: str = ""


@dataclass
class InputDataStats:
    table_num_nodes: dict[str, int] = field(default_factory=dict)
    table_size_bytes: dict[str, int] = field(default_factory=dict)
    table_time_ranges: dict[str, tuple[datetime.datetime | None,
                                       datetime.datetime
                                       | None]] = field(default_factory=dict)
    graph_num_nodes: int = 0
    graph_num_edges: int = 0
    graph_num_downsampled_edges: int = 0
    graph_size_bytes: int = 0
    num_target_entities: int = 0
    num_in_memory_features: int = 0

    def json(self) -> dict[str, object]:
        """Legacy-compatible dictionary serialization."""
        return asdict(self)
