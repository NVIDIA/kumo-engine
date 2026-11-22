# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split complete pytest runs without repeating a generated seed batch."""

import hashlib
import json
import re
from zlib import crc32


def shard_for(nodeid):
    seed = re.search(r"::test_property\[.*-seed(\d+)\]$", nodeid)
    return (int(seed[1]) if seed else crc32(nodeid.encode())) % 3


def collection_hash(nodeids):
    return hashlib.sha256("\n".join(sorted(nodeids)).encode()).hexdigest()


def verify_shards(paths, *, revision, run_id, attempt, seed_start=None):
    """Verify the disjoint union against the complete collection seen by every worker."""
    reports = [json.loads(path.read_text()) for path in paths]
    if len(reports) != 3 or {r["shard"] for r in reports} != {0, 1, 2}:
        raise ValueError("Expected conformance shards 0, 1 and 2")
    first = reports[0]
    expected = (revision, str(run_id), str(attempt), "all", 12)
    population = set()
    for r in reports:
        identity = (r["revision"], r["run_id"], r["attempt"], r["engine"], r["seeds"])
        if identity != expected or r["seed_start"] != first["seed_start"]:
            raise ValueError("Mismatched conformance run or seed range")
        if seed_start is not None and r["seed_start"] != seed_start:
            raise ValueError("Incorrect conformance seed start")
        ids = r["passed"]
        if r["complete"] is not True or not ids or len(ids) != len(set(ids)):
            raise ValueError("Incomplete conformance shard")
        if any(shard_for(nodeid) != r["shard"] for nodeid in ids):
            raise ValueError("Conformance test assigned to the wrong shard")
        population.update(ids)
    digest = collection_hash(population)
    if any(r["collection_sha256"] != digest for r in reports):
        raise ValueError("Conformance shard union differs from the complete collection")
    return len(population)
