# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engines that execute a contract against a dataset.

Each engine implements the `Engine` interface and checks a contract before it
reads any data, refusing what it can't run. The package ships two: the DiskGraph
engine, which is the one to use for real data, and the in-memory engine, which
runs the contract in Python over a few rows for trying things out. The DiskGraph
engine lists what it doesn't support yet in `DiskGraphEngine.unsupported`.

Constructing one is the contract; the short names exist so configuration can
resolve to the same classes:

    engine = DiskGraphEngine(index="/data/index")
    engine = engine_from_name("diskgraph", index="/data/index")   # identical
"""

from kumo.pytorch.engines.diskgraph import DiskGraphEngine
from kumo.pytorch.engines.in_memory import InMemoryEngine

__all__ = [
    "DiskGraphEngine",
    "InMemoryEngine",
]
