# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Samples assembled from rows held in memory, independent of any engine.

Serving assembles caller-supplied rows with `RowSource`, in the DiskGraph
engine's clock; the in-memory engine reads its datasets through it too, in
whole seconds.
"""

from kumo.pytorch.rows.clock import Clock
from kumo.pytorch.rows.source import RowSource

__all__ = ["Clock", "RowSource"]
