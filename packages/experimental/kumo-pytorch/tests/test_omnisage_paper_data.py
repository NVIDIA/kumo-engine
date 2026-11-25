# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the paper-recipe model inputs against the physical engagement and feature rows."""

from collections import Counter
from contextlib import nullcontext

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kumo.pytorch import api
from tests.e2e.omnisage_paper_data import (
    FUTURE_WINDOW,
    make_paper_fixture,
    paper_batches,
    paper_selection,
)
from tests.e2e.s3_endpoint import local_s3


def _seconds(column):
    return column.cast(pa.timestamp("s", tz="UTC")).cast(pa.int64()).to_pylist()


def _check_graphs(batch, tables):
    membership = tables["membership"]
    related = tables["related"]
    all_times = torch.unique(torch.cat(tuple(batch.times.values()))).tolist()
    degree = {}
    for at in all_times:
        by_type = {"pins": Counter(), "boards": Counter()}
        for pin, board, t in zip(
            membership["pin"].to_pylist(),
            membership["board"].to_pylist(),
            _seconds(membership["at"]),
            strict=True,
        ):
            if t <= at:
                by_type["pins"][pin] += 1
                by_type["boards"][board] += 1
        for src, dst, t in zip(
            related["src"].to_pylist(),
            related["dst"].to_pylist(),
            _seconds(related["at"]),
            strict=True,
        ):
            if t <= at:
                by_type["pins"][src] += 1
                by_type["pins"][dst] += 1
        degree[at] = by_type
    for root, graph in batch.graphs.items():
        times = batch.times[root]
        assert torch.equal(graph["nodes"][root]["ids"][: len(times)], batch.root_ids[root])
        assert set(graph["edges"]) == (
            {"pb.pins", "pb.boards", "pp.pins"} if root == "pins" else {"pb.pins", "pb.boards"}
        )
        for kind, features in (("pins", "image"), ("boards", "annotation")):
            node = graph["nodes"][kind]
            keys = node["ids"].tolist()
            cutoffs = times[node["batch"].long()].tolist()
            assert node["features"]["degree"]["values"].tolist() == [
                degree[at][kind][pk] for pk, at in zip(keys, cutoffs, strict=True)
            ]
            raw = tables[kind][features].combine_chunks()
            width = raw.type.list_size
            values = raw.values.to_numpy().reshape(-1, width)
            np.testing.assert_array_equal(
                node["features"][features]["values"].numpy(), values[keys]
            )


def _check_sequence(batch, users, actions, fixture, step):
    for i, row in enumerate(users.slice(step * fixture.batch_size, fixture.batch_size).to_pylist()):
        cutoff = int(row["at"].timestamp())
        valid = batch.history[i][batch.history[i] >= 0]
        history = [
            (r["pin"], r["second"])
            for r in actions
            if r["user"] == row["entity"] and r["second"] <= cutoff
        ]
        assert (
            list(
                zip(
                    batch.root_ids["pins"][valid].tolist(),
                    batch.times["pins"][valid].tolist(),
                    strict=True,
                )
            )
            == history[-fixture.history_length :]
        )
        assert all(t >= 0 for t in batch.history[i][: len(history[-fixture.history_length :])])
        future = batch.future[i].item()
        assert future >= 0 and batch.times["pins"][future] == cutoff
        assert batch.root_ids["pins"][future].item() in {
            r["pin"]
            for r in actions
            if r["user"] == row["entity"] and cutoff < r["second"] < cutoff + FUTURE_WINDOW
        }


def _check_pairs(batch, tables, step, size):
    for table in ("pins", "boards"):
        src = tables[table + "_examples"].slice(step * size * 2, size)
        q, qi, p, pi = batch.pairs[table + "_logged"]
        assert batch.root_ids[q][qi].tolist() == src["entity"].to_pylist()
        assert batch.root_ids[p][pi].tolist() == [v[0]["pin"] for v in src["logged"].to_pylist()]
    for src, key in (("pins_pp", "pp.pins"), ("pins_pb", "pb.boards"), ("boards_bp", "pb.pins")):
        q, qi, p, pi = batch.pairs[src]
        graph = batch.graphs[q]
        edge = graph["edges"][key]
        candidates = graph["nodes"][p]["ids"][edge["index"][1]]
        for idx, pk in zip(qi, batch.root_ids[p][pi], strict=True):
            assert pk in candidates[edge["index"][0] == idx]


@pytest.mark.parametrize("on_s3", [False, True], ids=["local", "s3"])
def test_paper_recipe_feeds_source_backed_occurrences(tmp_path, on_s3):
    pytest.importorskip("diskgraph")
    if on_s3:
        pytest.importorskip("moto.server")
    with local_s3() if on_s3 else nullcontext() as client:
        work = make_paper_fixture(
            tmp_path / "data",
            pins=40,
            steps=2,
            batch_size=4,
            history_length=5,
            random_negatives=9,
            client=client,
        )
        truth = {p.stem: pq.read_table(p) for p in work.root.glob("*.parquet")}
        # Algorithm 2 retains all edges in this low-degree fixture (d_min = 10).
        pin_degree = Counter(truth["membership"]["pin"].to_pylist())
        pin_degree.update(truth["related"]["src"].to_pylist())
        pin_degree.update(truth["related"]["dst"].to_pylist())
        board_degree = Counter(truth["membership"]["board"].to_pylist())
        assert max(pin_degree.values()) < 10
        assert max(board_degree.values()) < 10
        if on_s3:
            # Ensure the selector and graph engine do real S3 reads.
            work.root.rename(tmp_path / "source-truth")
        engine = api.DiskGraphEngine(num_threads=1, random_seed=42)
        requests = paper_selection(work, engine)
        found_padded = found_repeated = False
        with paper_batches(work, engine=engine, requests=requests) as stream:
            for step, batch in enumerate(stream):
                assert all(len(v) == work.batch_size for v in batch.features.values())
                assert all(len(v) == work.random_negatives for v in batch.negatives.values())
                _check_pairs(batch, truth, step, work.batch_size)
                _check_sequence(
                    batch, truth["users_examples"], truth["activity"].to_pylist(), work, step
                )
                _check_graphs(batch, truth)
                for seq in batch.history:
                    ids = batch.root_ids["pins"][seq[seq >= 0]].tolist()
                    found_repeated |= len(set(ids)) < len(ids)
                    found_padded |= len(ids) < work.history_length
        assert found_padded and found_repeated
