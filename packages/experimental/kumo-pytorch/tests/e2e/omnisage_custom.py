# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Specialized single-worker OmniSage loader using DiskGraph's native index and sampler."""

import json
from pathlib import Path
from types import SimpleNamespace

import diskgraph
import numpy as np
import pyarrow as pa
import torch

from kumo.pytorch.parquet_sources import open_parquet_file, parquet_objects
from tests.e2e.omnisage_fixture import (
    CANDIDATE_NAMES,
    DRAW_COUNT,
    FUTURE_WINDOW,
    KEEP,
    RESTART,
    WALK_BUDGET,
)


def _offsets(counts):
    return np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]


def _unique_pairs(ids, ticks):
    """Index repeated occurrences, keeping the full native precision of each cutoff."""
    times, codes = np.unique(ticks, return_inverse=True)
    _, index, inverse = np.unique(ids * len(times) + codes, return_index=True, return_inverse=True)
    return ids[index], ticks[index], inverse


def _sequence(root, relation, destination, *, columns, start=None, stop=0):
    frame = dict(
        name="window",
        root_type=root,
        path=[dict(relation=relation, destination_type=destination)],
        time_column="at",
        cutoff_unit="seconds",
        start_seconds=start,
        stop_seconds=stop,
        distinct=False,
        columns=columns,
    )
    sequence = dict(name="rows", frame="window", columns=columns, order="ascending", limit=None)
    return json.dumps(dict(frames=[frame], sequences=[sequence], aggregates=[]))


def _draw_rows(owners, ids, n, rng):
    """Uniform draws over distinct destination IDs per owner. Repeated actions count once."""
    if not len(ids):
        return ids, np.zeros(n + 1, dtype=np.int64)
    order = np.lexsort((ids, owners))
    owners, ids = owners[order], ids[order]
    first = np.r_[True, (owners[1:] != owners[:-1]) | (ids[1:] != ids[:-1])]
    owners, ids = owners[first], ids[first]
    order = np.lexsort((rng.random(len(ids)), owners))
    owners, ids = owners[order], ids[order]
    counts = np.bincount(owners, minlength=n)
    positions = np.arange(len(ids)) - np.repeat(_offsets(counts)[:-1], counts)
    return ids[positions < DRAW_COUNT], _offsets(np.minimum(counts, DRAW_COUNT))


def _static_features(index, sampler, field_names, width):
    public = {}
    vectors = {}
    stats = json.loads((index / "ingest_stats.json").read_text())["stats"]["node_counts"]
    for table, field in field_names.items():
        size = stats[table]
        vector_width = width if table == "pins" else min(128, width)
        ids = np.empty(size, dtype=np.int64)
        values = np.empty((size, vector_width), dtype=np.float32)
        for start in range(0, size, 2048):
            stop = min(start + 2048, size)
            offset = start
            for e in sampler.fetch_features(
                [
                    {
                        "node_type": table,
                        "node_ids": list(range(start, stop)),
                        "feature_names": ["id", field],
                    }
                ]
            ):
                if e["type"] != "feature":
                    continue
                batch = e["batch"]
                arr = batch.column(field)
                if arr.null_count or arr.values.null_count:
                    raise ValueError("this benchmark uses non-null feature vectors")
                end = offset + len(arr)
                ids[offset:end] = batch.column("id").to_numpy()
                values[offset:end] = arr.flatten().to_numpy().reshape(len(arr), vector_width)
                offset = end
            if offset != stop:
                raise ValueError("incomplete DiskGraph static feature projection")
        public[table], vectors[table] = ids, values
    return public, vectors


class CustomLoader:
    """Cache projected static columns and anchors; query the native temporal index each batch."""

    def __init__(self, index, anchor_source, batch_size, *, width, seed=42, num_threads=1):
        index = Path(index)
        self.batch_size = batch_size
        self.seed = seed
        self.sampler = diskgraph.SamplerEngine(
            str(index), num_threads=num_threads, random_seed=seed
        )
        mapping = json.loads(self.sampler.get_type_mapping())
        self.node_types = mapping["node_types"]
        manifest = json.loads((index / "index_manifest.json").read_text())
        metadata = json.loads((index / "index_metadata.json").read_text())
        if metadata["timestamp_unit"] != "seconds" or manifest["num_shards"] != 1:
            raise ValueError(
                "this specialized loader requires a one-shard, second-resolution index"
            )
        relations = manifest["relation_mapping"]
        pins = relations["pin_links"]
        boards = relations["board_links"]
        paths = [[pins["forward"], boards["reverse"]], [boards["forward"], pins["reverse"]]]
        walk = diskgraph.WalkSpec("walk", paths, KEEP, RESTART, WALK_BUDGET, {"membership": "at"})
        self.plan = diskgraph.SamplingPlan([diskgraph.LayerSpec({}, None, [walk])])
        self.degrees = {
            root: _sequence(root, relation["forward"], "membership", columns=[])
            for root, relation in (("pins", pins), ("boards", boards))
        }
        self.future = _sequence(
            "pins",
            relations["acted"]["forward"],
            "actions",
            columns=["pin"],
            start=0,
            stop=FUTURE_WINDOW,
        )
        self.field_names = {"pins": "image", "boards": "annotation"}
        self.public, self.vectors = _static_features(index, self.sampler, self.field_names, width)
        self.lookup_order = self.public["pins"].argsort()
        self.lookup_keys = self.public["pins"][self.lookup_order]
        path = str(getattr(anchor_source, "path", anchor_source))
        parts = []
        for obj in parquet_objects(path):
            with open_parquet_file(obj) as reader:
                parts.append(reader.read(columns=["sid", "entity", "at", "positives"]))
        anchors = pa.concat_tables(parts).combine_chunks()
        self.sids = anchors["sid"].to_numpy()
        self.roots = self._physical(anchors["entity"].to_numpy())
        self.times = anchors["at"].cast(pa.timestamp("s", tz="UTC")).cast(pa.int64()).to_numpy()
        positives = anchors["positives"].chunk(0)
        self.supplied_offsets = positives.offsets.to_numpy().astype(np.int64)
        self.supplied = self._physical(positives.flatten().field("pin").to_numpy())
        arrays = [
            *self.public.values(),
            *self.vectors.values(),
            self.lookup_order,
            self.lookup_keys,
            self.sids,
            self.roots,
            self.times,
            self.supplied_offsets,
            self.supplied,
        ]
        self.auxiliary_bytes = sum(a.nbytes for a in arrays)

    def _physical(self, public):
        positions = np.searchsorted(self.lookup_keys, public)
        if np.any(positions >= len(self.lookup_keys)) or np.any(
            self.lookup_keys[positions] != public
        ):
            raise KeyError("candidate or anchor references an unknown pin")
        return self.lookup_order[positions]

    def _walk(self, ids, ticks):
        """One native call for all distinct root/time pairs. Restore duplicate ownership later."""
        keys, times, inverse = _unique_pairs(ids, ticks)
        result = {
            name: (
                np.full((len(keys), keep), -1, dtype=np.int64),
                np.zeros((len(keys), keep), dtype=np.float32),
            )
            for name, keep in KEEP.items()
        }
        if len(keys):
            nodes, owners, edges = {}, {}, []
            for e in self.sampler.sample(
                {"pins": keys.tolist()},
                self.plan,
                fused=False,
                seed_timestamps={"pins": times.tolist()},
                swap_edge_direction=False,
                seed_timestamp_unit=diskgraph.TimeUnit.Seconds,
            ):
                if e["type"] == "node":
                    table = self.node_types[e["type_id"]]
                    batch = e["batch"]
                    nodes.setdefault(table, []).append(batch.column("id_b").to_numpy())
                    owners.setdefault(table, []).append(batch.column("id_a").to_numpy())
                elif e["type"] == "walk_edges":
                    edges.append(e)
            physical = {table: np.concatenate(parts) for table, parts in nodes.items()}
            pin_owners = np.concatenate(owners["pins"])
            for e in edges:
                table, edge = e["destination_type"], e["batch"]
                rows = pin_owners[edge.column("source").to_numpy()]
                rank = edge.column("rank").to_numpy()
                result[table][0][rows, rank] = physical[table][
                    edge.column("destination").to_numpy()
                ]
                result[table][1][rows, rank] = edge.column("score").to_numpy()
        return {
            table: (neighbors[inverse], weights[inverse])
            for table, (neighbors, weights) in result.items()
        }

    def _future_draw(self, roots, ticks, rng):
        seq = self.sampler.query_temporal(
            {"pins": roots.tolist()}, {"pins": ticks.tolist()}, self.future, output_format="model"
        )["sequences"][0]
        offsets = seq["offsets"].to_numpy()
        owners = np.repeat(np.arange(len(roots)), np.diff(offsets).astype(np.intp))
        candidates, boundaries = _draw_rows(
            owners, seq["batch"].column("pin").to_numpy(), len(roots), rng
        )
        return self._physical(candidates), boundaries

    @staticmethod
    def _scored_draw(query, rng):
        neighbors, scores = query["pins"]
        # Independent exponential clocks select distinct IDs proportional to score.
        clocks = np.full(scores.shape, np.inf)
        np.divide(rng.exponential(size=scores.shape), scores, out=clocks, where=scores > 0)
        slots = np.argsort(clocks, axis=1)[:, :DRAW_COUNT]
        result = np.take_along_axis(neighbors, slots, axis=1)
        valid = result >= 0
        return result[valid], _offsets(valid.sum(axis=1))

    def _tensor_graph(self, roots, neighbors, bounds):
        graph = dict(nodes={}, edges={})
        if bounds is not None:
            graph["candidate_offsets"] = torch.from_numpy(bounds)
        occurrences = {}
        for table, (ids, weights) in neighbors.items():
            owner, rank = np.nonzero(ids >= 0)
            native_ids = ids[owner, rank]
            dest = np.arange(len(owner), dtype=np.int64)
            if table == "pins":
                native_ids = np.r_[roots, native_ids]
                dest += len(roots)
                node_owner = np.r_[np.arange(len(roots)), owner]
            else:
                node_owner = owner
            occurrences[table] = native_ids, node_owner
            valid = torch.ones(len(native_ids), dtype=torch.bool)
            graph["nodes"][table] = dict(
                ids=torch.from_numpy(self.public[table][native_ids]),
                batch=torch.from_numpy(node_owner),
                features={
                    self.field_names[table]: dict(
                        values=torch.from_numpy(self.vectors[table][native_ids]),
                        valid=valid,
                    ),
                    "degree": dict(valid=valid.clone()),
                },
            )
            graph["edges"][f"walk.{table}"] = dict(
                index=torch.from_numpy(np.stack((owner, dest))),
                source_type="pins",
                destination_type=table,
                score=torch.from_numpy(weights[owner, rank]),
                rank=torch.from_numpy(rank),
            )
        return graph, occurrences

    def _batch(self, start, stop, rng):
        roots, ticks = self.roots[start:stop], self.times[start:stop]
        query = self._walk(roots, ticks)
        bounds = self.supplied_offsets[start : stop + 1]
        groups = {
            "supplied": (self.supplied[bounds[0] : bounds[-1]], bounds - bounds[0]),
            "scored": self._scored_draw(query, rng),
            "future": self._future_draw(roots, ticks, rng),
        }
        candidate_ids = np.concatenate([ids for ids, _ in groups.values()])
        group_times = {
            name: np.repeat(ticks, np.diff(offsets)) for name, (_, offsets) in groups.items()
        }
        sampled = self._walk(candidate_ids, np.concatenate(list(group_times.values())))
        graphs = {"query": self._tensor_graph(roots, query, None)}
        offset = 0
        metadata = {}
        for name in CANDIDATE_NAMES:
            ids, boundaries = groups[name]
            end = offset + len(ids)
            part = {
                table: (nodes[offset:end], scores[offset:end])
                for table, (nodes, scores) in sampled.items()
            }
            graphs[name] = self._tensor_graph(ids, part, boundaries)
            public = self.public["pins"][ids].tolist()
            metadata[name] = tuple(
                tuple(public[a:b]) for a, b in zip(boundaries[:-1], boundaries[1:])
            )
            offset = end

        all_ticks = {"query": ticks, **group_times}
        for table in KEEP:
            ids = np.concatenate([occs[table][0] for _, occs in graphs.values()])
            times = np.concatenate(
                [all_ticks[name][occs[table][1]] for name, (_, occs) in graphs.items()]
            )
            keys, cutoffs, inverse = _unique_pairs(ids, times)
            if len(keys):
                result = self.sampler.query_temporal_lengths(
                    {table: keys.tolist()}, {table: cutoffs.tolist()}, self.degrees[table]
                )
                all_degrees = result["sequences"][0]["lengths"].to_numpy().astype(np.int64)[inverse]
            else:
                all_degrees = np.empty(0, dtype=np.int64)
            at = 0
            for graph, occs in graphs.values():
                end = at + len(occs[table][0])
                graph["nodes"][table]["features"]["degree"]["values"] = torch.from_numpy(
                    all_degrees[at:end]
                )
                at = end

        return SimpleNamespace(
            inputs={name: graph for name, (graph, _) in graphs.items()},
            metadata={
                "sample_ids": self.sids[start:stop].tolist(),
                "candidate_entity_ids": metadata,
            },
        )

    def close(self):
        """Release index mappings and the projected feature and anchor arrays."""
        self.sampler = None
        self.vectors.clear()
        self.public.clear()
        self.lookup_order = self.lookup_keys = None
        self.sids = self.roots = self.times = self.supplied_offsets = self.supplied = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def batches(self):
        if self.sampler is None:
            raise RuntimeError("custom loader is closed")
        rng = np.random.default_rng(self.seed)
        for start in range(0, len(self.sids), self.batch_size):
            yield self._batch(start, min(start + self.batch_size, len(self.sids)), rng)
