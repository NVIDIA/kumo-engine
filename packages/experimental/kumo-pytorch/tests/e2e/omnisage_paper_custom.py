# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct DiskGraph fetches for the paper benchmark's shared root and cutoff schedule."""

import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import diskgraph
import numpy as np
import pyarrow as pa
import torch

from tests.e2e.omnisage_custom import _sequence, _static_features, _unique_pairs


@contextmanager
def custom_graphs(index, fixture, num_threads, seed):
    index = Path(index)
    manifest = json.loads((index / "index_manifest.json").read_text())
    metadata = json.loads((index / "index_metadata.json").read_text())
    if manifest["num_shards"] != 1 or metadata["timestamp_unit"] != "seconds":
        raise ValueError("the custom paper loader requires one shard and a second-resolution clock")
    sampler = diskgraph.SamplerEngine(str(index), num_threads=num_threads, random_seed=seed)
    types = json.loads(sampler.get_type_mapping())["node_types"]
    rel = manifest["relation_mapping"]

    def walk(name, event, left, right, quotas):
        paths = [
            [rel[left]["forward"], rel[right]["reverse"]],
            [rel[right]["forward"], rel[left]["reverse"]],
        ]
        return diskgraph.WalkSpec(name, paths, quotas, 0.25, 1024, {event: "at"})

    pb = walk("pb", "membership", "pin_links", "board_links", {"pins": 25, "boards": 75})
    pp = walk("pp", "related", "related_from", "related_to", {"pins": 50})
    plans = {
        kind: diskgraph.SamplingPlan([diskgraph.LayerSpec({}, None, walks)])
        for kind, walks in (("pins", [pb, pp]), ("boards", [pb]))
    }
    degree_paths = {
        "pins": [
            ("pin_links", "membership"),
            ("related_from", "related"),
            ("related_to", "related"),
        ],
        "boards": [("board_links", "membership")],
    }
    degrees = {
        kind: [_sequence(kind, rel[path]["forward"], table, columns=[]) for path, table in paths]
        for kind, paths in degree_paths.items()
    }
    fields = {"pins": "image", "boards": "annotation"}
    public, vectors = _static_features(index, sampler, fields, 1152)

    def fetch(root_type, root_ids, times):
        # This loader accepts the same ordered requests as the package, including cutoffs.
        times = np.asarray(times, dtype=np.int64)
        physical = sampler.lookup_node_ids(
            root_type, pa.array(np.asarray(root_ids), type=pa.int64())
        )
        if any(i is None for i in physical):
            raise ValueError("unknown paper-workload root")
        records = defaultdict(list)
        edges = defaultdict(list)
        for event in sampler.sample(
            {root_type: physical},
            plans[root_type],
            fused=False,
            seed_timestamps={root_type: times.tolist()},
            swap_edge_direction=False,
            seed_timestamp_unit=diskgraph.TimeUnit.Seconds,
        ):
            if event["type"] == "node":
                records[types[event["type_id"]]].append(event["batch"])
            elif event["type"] == "walk_edges":
                edges[event["walk"], event["destination_type"]].append(event["batch"])
        graph = {"nodes": {}, "edges": {}}
        for table, field in fields.items():
            rows = records[table]
            ids = (
                np.concatenate([b.column("id_b").to_numpy() for b in rows]).astype(np.int64)
                if rows
                else np.empty(0, dtype=np.int64)
            )
            owners = (
                np.concatenate([b.column("id_a").to_numpy() for b in rows]).astype(np.int64)
                if rows
                else np.empty(0, dtype=np.int64)
            )
            keys, at, restore = _unique_pairs(ids, times[owners])
            degree = np.zeros(len(keys), dtype=np.int64)
            if len(keys):
                for query in degrees[table]:
                    result = sampler.query_temporal_lengths(
                        {table: keys.tolist()}, {table: at.tolist()}, query
                    )
                    degree += result["sequences"][0]["lengths"].to_numpy().astype(np.int64)
            valid = torch.ones(len(ids), dtype=torch.bool)
            graph["nodes"][table] = {
                "ids": torch.from_numpy(public[table][ids]),
                "batch": torch.from_numpy(owners),
                "features": {
                    field: {"values": torch.from_numpy(vectors[table][ids]), "valid": valid},
                    "degree": {"values": torch.from_numpy(degree[restore]), "valid": valid.clone()},
                },
            }
        for name, table in [("pb", "pins"), ("pb", "boards")] + (
            [("pp", "pins")] if root_type == "pins" else []
        ):
            rows = edges[name, table]

            def column(key, dtype):
                return (
                    torch.as_tensor(
                        np.concatenate([b.column(key).to_numpy() for b in rows]), dtype=dtype
                    )
                    if rows
                    else torch.empty(0, dtype=dtype)
                )

            graph["edges"][f"{name}.{table}"] = {
                "index": torch.stack(
                    (column("source", torch.long), column("destination", torch.long))
                ),
                "score": column("score", torch.float32),
                "rank": column("rank", torch.long),
                "source_type": root_type,
                "destination_type": table,
            }
        return graph

    try:
        yield fetch
    finally:
        vectors.clear()
        sampler = None
