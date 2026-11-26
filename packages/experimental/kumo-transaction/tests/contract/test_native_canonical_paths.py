# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical neighbor selection and root/event unions on generic source tables."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tfm.config import RelationalTaskConfig
from tfm.contracts.graph import GraphAnchor
from tfm.contracts.relational import RelationalDataset, RelationalForeignKey, RelationalTable
from tfm.contracts.table_source import TableSource
from tfm.data.graph import GraphSampler
from tfm.data.native_inputs import NativeInputs
from tfm.data.native_validation import validate_graph_policy
from tfm.data.relational import RelationalSource, key_token

START = datetime(2024, 1, 1, tzinfo=UTC)


def signature(batch):
    nodes = {
        name: sorted(
            (int(component), tuple(values.tolist()), int(clock))
            for component, values, clock in zip(
                batch.node_sample[name],
                batch.node_features[name],
                batch.node_available_us[name],
                strict=True,
            )
        )
        for name in batch.node_features
    }
    edges = {
        role: sorted(
            (
                int(batch.node_sample[role[0]][s]),
                tuple(batch.node_features[role[0]][s].tolist()),
                tuple(batch.node_features[role[2]][t].tolist()),
                int(clock),
            )
            for (s, t), clock in zip(indices.T.tolist(), batch.edge_available_us[role], strict=True)
        )
        for role, indices in batch.edge_index.items()
    }
    return nodes, edges


@pytest.mark.parametrize("sharing", ["shared", "disjoint"])
@pytest.mark.parametrize("composite", [False, True])
@pytest.mark.parametrize("threads", [1, 2])
def test_native_recent_ties_select_canonical_keys_before_ten_neighbor_fanout(
    tmp_path, sharing, composite, threads
):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    # Six newer rows plus six equal-time rows compete for ten slots. Physical
    # input order disagrees with canonical key order, including numeric strings.
    keys = ["z", "a", "10", "2", "b", "y", "n5", "n1", "n3", "n0", "n4", "n2", "future"]
    rows = {
        "owners": [{"id": "one", "signal": 100.0}],
        "activity": [
            dict(
                id=key,
                kind="k",
                owner="one",
                at=START + timedelta(days=1 if i < 6 else 2 if i < 12 else 5),
                signal=float(i + 1),
            )
            for i, key in enumerate(keys)
        ],
    }
    files = {}
    for name, values in rows.items():
        files[name] = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(values), files[name])
    tables = (
        RelationalTable("owners", ("id",), numeric={"signal": "signal"}, immutable=True),
        RelationalTable(
            "activity",
            ("id", "kind") if composite else ("id",),
            numeric={"signal": "signal"},
            event_time="at",
            available_at="at",
        ),
    )
    roles = (RelationalForeignKey("activity_owner", "owners", "activity", ("owner",)),)
    paths = {"history": ("activity_owner",)}
    source = TableSource(tables, roles, files)
    anchors = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.table(
            dict(
                sample_id=["first", "repeat"],
                id=["one", "one"],
                cutoff=[START + timedelta(days=3)] * 2,
                split=["train", "train"],
            )
        ),
        anchors,
    )
    task = RelationalTaskConfig(root="owners", history_paths=paths, cutoffs={"train": "2024-01-04"})
    reference = GraphSampler(
        RelationalSource(
            RelationalDataset(tables, roles, {n: tuple(v) for n, v in rows.items()}), task
        ).graph(),
        depth=6,
        fanout=10,
    )
    with NativeInputs(
        source,
        root="owners",
        history_paths=paths,
        anchors=anchors,
        work_dir=tmp_path / "work",
        max_events=32,
        batch_size=2,
        depth=6,
        fanout=10,
        num_threads=threads,
    ) as inputs:
        validate_graph_policy(
            inputs.normalized, root="owners", paths=paths, fanout=10, observation_seconds=None
        )
        for _ in range(2):
            with inputs.batches("train", mode="graph", sharing=sharing, shuffle=False) as batches:
                (actual,) = list(batches)
            expected = reference.sample(
                tuple(replace(a, entity_id=key_token(("one",))) for a in actual.anchors),
                "owners",
                sharing=sharing,
            )
            assert signature(actual) == signature(expected)
            assert actual.num_sampled_nodes == expected.num_sampled_nodes
            assert actual.num_sampled_edges == expected.num_sampled_edges
            selected = set(actual.node_features["activity"][:, 0].tolist())
            assert selected == {2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}
            assert all(values.shape[1] == 2 for values in actual.node_features.values())
            actual.assert_validated()


@pytest.mark.parametrize("mode", ["sequence", "hybrid"])
@pytest.mark.parametrize("max_events", [3, 64])
def test_native_history_unions_incoming_outgoing_and_self_events_per_root(
    tmp_path, mode, max_events
):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    rows = {
        "owners": [{"id": "a", "signal": 10.0}, {"id": "b", "signal": 20.0}],
        "accounts": [
            {"id": 11, "owner": "a", "opened": START},
            {"id": 12, "owner": "a", "opened": START},
            {"id": 21, "owner": "b", "opened": START},
        ],
        "activity": [
            {
                "id": "t10",
                "sender": 11,
                "receiver": 12,
                "at": START + timedelta(days=1),
                "amount": 10.0,
            },
            {
                "id": "t2",
                "sender": 11,
                "receiver": 21,
                "at": START + timedelta(days=1),
                "amount": 20.0,
            },
            {
                "id": "t1",
                "sender": 21,
                "receiver": 11,
                "at": START + timedelta(days=2),
                "amount": 30.0,
            },
            {
                "id": "t3",
                "sender": 21,
                "receiver": 12,
                "at": START + timedelta(days=2),
                "amount": 40.0,
            },
            {
                "id": "future",
                "sender": 11,
                "receiver": 21,
                "at": START + timedelta(days=5),
                "amount": 50.0,
            },
        ],
    }
    tables = (
        RelationalTable("owners", ("id",), numeric={"signal": "signal"}, immutable=True),
        RelationalTable("accounts", ("id",), event_time="opened", available_at="opened"),
        RelationalTable(
            "activity", ("id",), numeric={"amount": "amount"}, event_time="at", available_at="at"
        ),
    )
    roles = (
        RelationalForeignKey("owns", "owners", "accounts", ("owner",)),
        RelationalForeignKey("sent", "accounts", "activity", ("sender",)),
        RelationalForeignKey("received", "accounts", "activity", ("receiver",)),
    )
    files = {}
    for name, values in rows.items():
        files[name] = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(values), files[name])
    paths = {"outgoing": ("owns", "sent"), "incoming": ("owns", "received")}
    task = RelationalTaskConfig(root="owners", history_paths=paths, cutoffs={"train": "2024-01-04"})
    reference = RelationalSource(
        RelationalDataset(tables, roles, {n: tuple(v) for n, v in rows.items()}), task
    )
    anchor_file = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.table(
            dict(
                sample_id=["a0", "b0", "repeat", "a1", "b1"],
                id=["a", "b", "a", "a", "b"],
                cutoff=[START + timedelta(days=d) for d in (3, 3, 3, 6, 6)],
                split=["train"] * 3 + ["later"] * 2,
            )
        ),
        anchor_file,
    )
    with NativeInputs(
        TableSource(tables, roles, files),
        root="owners",
        history_paths=paths,
        anchors=anchor_file,
        work_dir=tmp_path / "work",
        max_events=max_events,
        batch_size=8,
        depth=6,
        fanout=10,
        num_threads=2,
    ) as inputs:
        fitted = inputs.task(mode)
        for split in ("train", "later"):
            (delivered,) = list(
                fitted.batches(split=split, batch_size=8, shuffle=False, output_format="model")
            )
            mask = delivered.inputs["history"]["sequence"]["valid"].bool()
            amounts = delivered.inputs["history_aux"]["__amount"]["values"]
            requests = delivered.metadata["request_metadata"]
            for i, request in enumerate(requests):
                cutoff = START + timedelta(days=3 if split == "train" else 6)
                history = reference.history(
                    GraphAnchor(
                        str(delivered.metadata["sample_ids"][i]),
                        key_token((request["entity_id"],)),
                        cutoff,
                    ),
                    max_events=max_events,
                )
                assert amounts[i][mask[i]].tolist() == [
                    event.numeric["amount"] for event in history.events
                ]
            with inputs.batches(split, mode=mode, sharing="shared", shuffle=False) as batches:
                (batch,) = list(batches)
            batch.assert_validated()
            assert batch.history.event_mask.sum(1).tolist() == mask.sum(1).tolist()
            assert batch.history.sequence_ids == tuple(delivered.metadata["sample_ids"])
            if split == "train" and max_events == 64:
                assert batch.history.event_mask.sum(1).tolist() == [4, 3, 4]
            batch.history.value_ids.add_(1)
            with pytest.raises(ValueError, match="mutat|changed"):
                batch.assert_validated()


def test_native_canonical_order_preserves_wide_clock_ranges_and_unused_nested_columns(tmp_path):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    owners = tmp_path / "owners.parquet"
    events = tmp_path / "events.parquet"
    pq.write_table(pa.table({"id": ["one"]}), owners)
    times = [datetime(y, 1, 1, tzinfo=UTC) for y in (1900, 1960, 2030)]
    pq.write_table(
        pa.table(
            {
                "id": ["old", "middle", "new"],
                "owner": ["one"] * 3,
                "unused": [{"x": 1, "y": 2}] * 3,
                "at": times,
                "signal": [1.0, 2.0, 3.0],
            }
        ),
        events,
    )
    source = TableSource(
        (
            RelationalTable("owners", ("id",), immutable=True),
            RelationalTable(
                "activity",
                ("id",),
                numeric={"signal": "signal"},
                event_time="at",
                available_at="at",
            ),
        ),
        (RelationalForeignKey("owns", "owners", "activity", ("owner",)),),
        {"owners": owners, "activity": events},
    )
    anchors = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": ["one"],
                "id": ["one"],
                "cutoff": [datetime(2031, 1, 1, tzinfo=UTC)],
                "split": ["train"],
            }
        ),
        anchors,
    )
    with NativeInputs(
        source,
        root="owners",
        history_paths={"history": ("owns",)},
        anchors=anchors,
        work_dir=tmp_path / "work",
        max_events=8,
        batch_size=1,
        depth=1,
        fanout=2,
    ) as inputs:
        with inputs.batches("train", mode="graph", shuffle=False) as batches:
            (batch,) = list(batches)
        batch.assert_validated()
        assert set(batch.node_features["activity"][:, 0].tolist()) == {2.0, 3.0}
        assert set(batch.node_available_us["activity"].tolist()) == {
            int(t.timestamp()) * 1_000_000 for t in times[1:]
        }


def test_native_canonical_clock_encoding_rejects_overflow_before_ingest(tmp_path):
    from tfm.contracts.errors import UnsupportedCombinationError
    from tfm.data.native_index import _clock_anchor

    path = tmp_path / "events.parquet"
    pq.write_table(
        pa.table({"at": [datetime(1900, 1, 1, tzinfo=UTC), datetime(2200, 1, 1, tzinfo=UTC)]}), path
    )
    with pytest.raises(UnsupportedCombinationError, match="events.at.*clock span"):
        _clock_anchor({"name": "events", "source": {"paths": [str(path)]}}, "at")
