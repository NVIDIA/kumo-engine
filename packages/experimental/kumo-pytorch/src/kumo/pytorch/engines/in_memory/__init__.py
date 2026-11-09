# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-memory engine, for trying the library out on a few rows of data.

It holds tables as Python mappings and evaluates the contract with plain scans,
so it is not meant for real data or production use; use `DiskGraphEngine` for
that. The tests also run it next to DiskGraph and compare the results, which is
why it declares every operation the contract defines.
"""

from kumo.pytorch.engines.in_memory.engine import InMemoryEngine

__all__ = ["InMemoryEngine"]
