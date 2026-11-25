# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check model tensors against the original Parquet rows, outside the timed loops."""

from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from tests.e2e.omnisage_fixture import (
    CANDIDATE_NAMES,
    DRAW_COUNT,
    FUTURE_WINDOW,
    KEEP,
    RESTART,
    WALK_BUDGET,
)


def _seconds(table):
    return table["at"].cast(pa.timestamp("s", tz="UTC")).cast(pa.int64()).to_pylist()


class Oracle:
    def __init__(self, workload):
        def read(table):
            return pq.read_table(workload.paths[table])

        anchors = read("anchors")
        self.anchors = {
            sid: (pin, cutoff, tuple(entry["pin"] for entry in positives))
            for sid, pin, cutoff, positives in zip(
                anchors["sid"].to_pylist(),
                anchors["entity"].to_pylist(),
                _seconds(anchors),
                anchors["positives"].to_pylist(),
                strict=True,
            )
        }
        assert len(self.anchors) == len(anchors)
        self.arrays = {}
        for kind, name in (("pins", "image"), ("boards", "annotation")):
            table = read(kind)
            row = {pk: i for i, pk in enumerate(table["id"].to_pylist())}
            array = table[name].combine_chunks()
            self.arrays[kind] = name, row, array.values.to_numpy().reshape(len(table), -1)
        actions = read("actions")
        self.actions = defaultdict(list)
        for actor, at, pin in zip(
            actions["actor"].to_pylist(), _seconds(actions), actions["pin"].to_pylist(), strict=True
        ):
            self.actions[actor].append((at, pin))
        membership = read("membership")
        self.at = {at: defaultdict(list) for _, at, _ in self.anchors.values()}
        for pin, board, event in zip(
            membership["pin"].to_pylist(),
            membership["board"].to_pylist(),
            _seconds(membership),
            strict=True,
        ):
            for cutoff, adj in self.at.items():
                if event <= cutoff:
                    adj["pins", pin].append(("boards", board))
                    adj["boards", board].append(("pins", pin))
        self.stationary = {}

    def _walk(self, root, at):
        """Solve the exact restart walk on this small connected temporal component."""
        start = ("pins", root)
        if (at, start) not in self.stationary:
            adj = self.at[at]
            nodes = [start]
            positions = {start: 0}
            for node in nodes:
                for other in adj.get(node, ()):
                    if other not in positions:
                        positions[other] = len(nodes)
                        nodes.append(other)
            transition = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
            for i, node in enumerate(nodes):
                neighbors = adj.get(node, ())
                for other, count in Counter(neighbors).items():
                    transition[i, positions[other]] = count / len(neighbors)
            identity = np.eye(len(nodes))
            # Row i contains the exact discounted occupancy when starting at i.
            scores = np.linalg.solve(identity - (1 - RESTART) * transition, RESTART * identity)
            for i, node in enumerate(nodes):
                self.stationary[at, node] = dict(zip(nodes, scores[i], strict=True))
        return self.stationary[at, start]

    def _check_scores(self, kind, root, at, actual):
        exact = self._walk(root, at)
        assert len(actual) <= KEEP[kind]
        actual.sort()  # rank, destination ID, score
        assert [rank for rank, _, _ in actual] == list(range(len(actual)))
        ids = [pk for _, pk, _ in actual]
        assert len(set(ids)) == len(ids)
        scores = [s for _, _, s in actual]
        assert scores == sorted(scores, reverse=True)
        observed = {pk: s for _, pk, s in actual}
        eligible = {
            pk: v
            for (table, pk), v in exact.items()
            if table == kind and (table != "pins" or pk != root)
        }
        assert set(ids) <= eligible.keys()
        # Forward push stops when residual(v) < degree(v)/budget. On the undirected
        # membership multigraph, degree is stationary for the walk. Residual mass
        # can therefore add at most degree(v)/budget to the reported score of v.
        tolerance = 1e-6
        threshold = scores[-1] if len(actual) == KEEP[kind] else 0.0
        for pk, score in eligible.items():
            bound = len(self.at[at][kind, pk]) / WALK_BUDGET + tolerance
            if pk in observed:
                assert 0 < observed[pk] <= score + tolerance
                assert observed[pk] >= score - bound, (kind, pk, root, at, observed[pk], score)
            else:
                assert score - bound <= threshold, (kind, pk, root, at, score, threshold)

    def _check_graph(self, graph, roots, cutoffs, counts):
        ids = {}
        owners = {}
        by_owner = {kind: [[] for _ in roots] for kind in KEEP}
        for kind in KEEP:
            node = graph["nodes"][kind]
            assert node["ids"].dtype == torch.long and node["batch"].dtype == torch.long
            ids[kind] = node["ids"].tolist()
            owners[kind] = node["batch"].tolist()
            assert len(ids[kind]) == len(owners[kind])
            assert len(set(zip(owners[kind], ids[kind], strict=True))) == len(ids[kind])
            assert all(0 <= owner < len(roots) for owner in owners[kind])
            degree = node["features"]["degree"]
            assert degree["values"].dtype == torch.long
            assert degree["valid"].dtype == torch.bool
            assert degree["valid"].shape == (len(ids[kind]),) and degree["valid"].all()
            assert degree["values"].tolist() == [
                len(self.at[cutoffs[owner]].get((kind, pk), ()))
                for pk, owner in zip(ids[kind], owners[kind], strict=True)
            ]
            name, positions, expected = self.arrays[kind]
            feature = node["features"][name]
            assert feature["values"].dtype == torch.float32
            assert feature["valid"].dtype == torch.bool
            assert feature["valid"].shape == (len(ids[kind]),) and feature["valid"].all()
            rows = [positions[pk] for pk in ids[kind]]
            np.testing.assert_array_equal(feature["values"].numpy(), expected[rows])
            counts["feature_elements"] += feature["values"].numel()
            counts["node_occurrences"] += len(ids[kind])
        assert ids["pins"][: len(roots)] == roots
        assert owners["pins"][: len(roots)] == list(range(len(roots)))
        for kind in KEEP:
            edge = graph["edges"][f"walk.{kind}"]
            assert edge["index"].dtype == edge["rank"].dtype == torch.long
            assert edge["score"].dtype == torch.float32
            src, dst = edge["index"].tolist()
            reached = set()
            for i, j, rank, score in zip(
                src, dst, edge["rank"].tolist(), edge["score"].tolist(), strict=True
            ):
                assert 0 <= i < len(roots) and 0 <= j < len(ids[kind])
                assert owners[kind][j] == i
                by_owner[kind][i].append((rank, ids[kind][j], score))
                reached.add(j)
            expected = set(range(len(ids[kind])))
            if kind == "pins":
                expected.difference_update(range(len(roots)))
            assert reached == expected
            counts["scored_edges"] += len(src)
            for i, root in enumerate(roots):
                self._check_scores(kind, root, cutoffs[i], by_owner[kind][i])
        counts["graph_roots"] += len(roots)
        return [{pk for _, pk, _ in hits} for hits in by_owner["pins"]]

    def verify_epoch(self, batches):
        """Validate each occurrence and require every source anchor exactly once."""
        counts = Counter()
        seen = set()
        for batch in batches:
            ids = list(batch.metadata["sample_ids"])
            assert len(set(ids)) == len(ids) and seen.isdisjoint(ids)
            seen.update(ids)
            counts["batches"] += 1
            counts["anchors"] += len(ids)
            rows = [self.anchors[sid] for sid in ids]
            roots = [row[0] for row in rows]
            cutoffs = [row[1] for row in rows]
            scored = self._check_graph(batch.inputs["query"], roots, cutoffs, counts)
            for name in CANDIDATE_NAMES:
                groups = batch.metadata["candidate_entity_ids"][name]
                for (pin, cutoff, supplied), group, hits in zip(rows, groups, scored, strict=True):
                    if name == "supplied":
                        assert tuple(group) == supplied
                    else:
                        eligible = (
                            hits
                            if name == "scored"
                            else {
                                pk
                                for t, pk in self.actions[pin]
                                if cutoff < t <= cutoff + FUTURE_WINDOW
                            }
                        )
                        assert len(set(group)) == len(group) == min(DRAW_COUNT, len(eligible))
                        assert set(group) <= eligible
                graph = batch.inputs[name]
                assert (
                    graph["candidate_offsets"].tolist()
                    == np.cumsum([0, *map(len, groups)]).tolist()
                )
                flat = [pk for group in groups for pk in group]
                times = [at for at, group in zip(cutoffs, groups, strict=True) for _ in group]
                self._check_graph(graph, flat, times, counts)
                counts[f"{name}_candidates"] += len(flat)
        assert seen == self.anchors.keys()
        return dict(counts)
