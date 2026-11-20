# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic ownership of conformance checks across CI workers."""

import hashlib
import re


def owner(nodeid: str, count: int) -> int:
    if count < 1:
        raise ValueError("Conformance shard count must be positive")
    # Keep one seed's properties together so their prepared fixtures stay cached.
    seed = re.search(r"::test_property\[.*-seed(\d+)\]$", nodeid)
    key = (
        int(seed[1])
        if seed
        else int.from_bytes(hashlib.sha256(nodeid.encode()).digest()[:8], "big")
    )
    return key % count
