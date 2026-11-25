# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-level checks for chronology and catalog eligibility in the real-log adapter."""

from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from tests.e2e.omnisage_event_fixture import _snapshot, make_event_fixture
from tests.e2e.omnisage_paper_data import paper_batches, paper_selection
from tests.test_omnisage_paper_data import _check_graphs


def test_event_fixture_uses_observed_edges_and_future_labels(tmp_path):
    pytest.importorskip("diskgraph")
    source = [
        (100, 10, 10),
        (200, 20, 10),
        (100, 30, 20),
        (100, 40, 20),
        (200, 10, 20),
        (100, 20, 30),
        (200, 30, 30),
        (200, 20, 40),
        (100, 10, 41),
        (100, 30, 50),
        (200, 10, 51),
        (300, 50, 60),
        (100, 20, 68),
        (200, 30, 69),
        (200, 10, 80),
        (100, 20, 80),
    ]
    # Input is disordered, while the ordering of records within each timestamp stays intact.
    rows = source[5:10] + source[:5] + source[10:]
    actor, item, second = map(np.asarray, zip(*rows, strict=True))
    path = tmp_path / "events.npz"
    np.savez(path, actor=actor, item=item, time=second)
    fixture, stats = make_event_fixture(
        tmp_path / "parquet",
        path,
        steps=3,
        batch_size=4,
        history_length=3,
        random_negatives=25,
        max_events=15,
        future_window_seconds=20,
    )
    assert stats["raw_events"] == 16 and stats["used_events"] == 14
    assert stats["catalog"] == {"pins": 5, "boards": 3, "users": 3}
    assert (
        stats["unique_actor_seconds"] == 13
        and stats["same_actor_second_rows_omitted_from_sequence"] == 1
    )
    # The chronological cap excludes both tied events at second 80. Reconstruct native IDs
    # directly from the original event log, without reading the adapter's mapping helpers.
    snapshot = stats["snapshot_second"]
    assert snapshot == 30
    assert stats["snapshot_before"] == stats["snapshot_after"]
    assert stats["sequence_events"] == 7
    actors, items, pairs, related = {}, {}, {}, {}
    history = defaultdict(dict)
    born = {"pins": {}, "boards": {}}
    for u, i, at in source[:-2]:
        ui, pi = actors.setdefault(u, len(actors)), items.setdefault(i, len(items))
        born["pins"].setdefault(pi, at)
        born["boards"].setdefault(ui, at)
        if at <= snapshot:
            pairs.setdefault((ui, pi), at)
        history[ui].setdefault(at, pi)
    for seq in history.values():
        records = list(seq.items())
        for (_, i), (time, j) in zip(records[:-1], records[1:], strict=True):
            if i != j and time <= snapshot:
                edge = tuple(sorted((i, j)))
                related[edge] = min(time, related.get(edge, time))
    tables = {p.stem: pq.read_table(p) for p in fixture.root.glob("*.parquet")}
    for name, lhs, rhs, expected in (
        ("membership", "board", "pin", pairs),
        ("related", "src", "dst", related),
    ):
        actual = {(r[lhs], r[rhs]): int(r["at"].timestamp()) for r in tables[name].to_pylist()}
        assert len(expected) == len(tables[name]) and actual == expected
    engine = api.DiskGraphEngine(num_threads=1, random_seed=42)
    requests = paper_selection(fixture, engine)
    with paper_batches(fixture, engine=engine, requests=requests) as batches:
        for step, batch in enumerate(batches):
            cutoff = fixture.cutoffs[step]
            assert all((times > snapshot).all() for times in batch.times.values())
            for table in ("pins", "boards"):
                for ids in (batch.features[table], batch.negatives[table]):
                    assert all(
                        born[table][i] <= cutoff for i in batch.root_ids[table][ids].tolist()
                    )
                _, qi, _, pi = batch.pairs[table + "_logged"]
                for q, p in zip(
                    batch.root_ids[table][qi].tolist(),
                    batch.root_ids["pins"][pi].tolist(),
                    strict=True,
                ):
                    labels = []
                    for uid, events in history.items():
                        past = [i for t, i in events.items() if snapshot < t <= cutoff]
                        ahead = [i for t, i in events.items() if cutoff < t < cutoff + 20]
                        if past and ahead and (uid == q if table == "boards" else past[-1] == q):
                            labels.append(ahead[0])
                    assert p in labels and born["pins"][p] <= cutoff
            users = tables["users_examples"].slice(step * 4, 4)["entity"].to_pylist()
            for idx, uid in enumerate(users):
                past = [(i, t) for t, i in history[uid].items() if snapshot < t <= cutoff][-3:]
                roots = batch.history[idx][batch.history[idx] >= 0]
                assert (
                    list(
                        zip(
                            batch.root_ids["pins"][roots].tolist(),
                            batch.times["pins"][roots].tolist(),
                            strict=True,
                        )
                    )
                    == past
                )
                future = batch.future[idx].item()
                target = batch.root_ids["pins"][future].item()
                assert batch.times["pins"][future] == cutoff
                assert target in [i for t, i in history[uid].items() if cutoff < t < cutoff + 20]
            _check_graphs(batch, tables)


def test_pruning_uses_only_snapshot_degrees_and_both_endpoint_decisions():
    # Actor 0 and item 0 both have 31+ neighbors. Every edge not incident to a hub has
    # two low-degree endpoints and survives. Include late activity with a different hub.
    ids = np.arange(1, 33)
    board, pin = np.r_[0 * ids, ids, ids], np.r_[ids, 0 * ids, ids]
    times = np.full(len(pin), 10)
    pp = (0 * ids, ids, np.full(len(ids), 9))
    pb = (board, pin, times)
    out, rel, before, after = _snapshot(pb, pp, 10, 33, 33, 42)
    assert after["pb_edges"] < before["pb_edges"] and after["pp_edges"] < before["pp_edges"]
    retained = set(zip(out[0], out[1], strict=True))
    assert all((pk, pk) in retained for pk in ids)
    assert any((0, pk) not in retained for pk in ids)
    assert any((pk, 0) not in retained for pk in ids)
    assert before["pin_degree"]["max"] == 64
    assert before["board_degree"]["max"] == 32
    late_pb = (np.r_[board, ids], np.r_[pin, ids[::-1]], np.r_[times, times[: len(ids)] + 1])
    late_pp = tuple(
        np.r_[col, extra] for col, extra in zip(pp, (ids, ids[::-1], np.full(32, 12)), strict=True)
    )
    same, same_rel, same_before, same_after = _snapshot(late_pb, late_pp, 10, 33, 33, 42)
    assert same_before == before and same_after == after
    for got, expected in zip((*same, *same_rel), (*out, *rel), strict=True):
        np.testing.assert_array_equal(got, expected)
