# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed padding keeps caller identities and native graph ownership distinct."""

from __future__ import annotations

from pathlib import Path

import pytest

from kumo.pytorch import api


@pytest.fixture(scope="module")
def graph_plan(tmp_path_factory):
    """Open one real native index for all tiny request and AnchorTable controls."""
    pytest.importorskip("diskgraph")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path_factory.mktemp("padding-native")
    accounts = pa.table({"account_id": [1]})
    transactions = pa.table(
        {
            "transaction_id": [100],
            "sender_id": [1],
            "event_time": pa.array([1], type=pa.timestamp("s", tz="UTC")),
        }
    )
    sources = {}
    for name, table in (("accounts", accounts), ("transactions", transactions)):
        path = root / f"{name}.parquet"
        pq.write_table(table, path)
        sources[name] = api.Parquet(str(path))
    schema = api.DataSchema(
        tables=(
            api.TableSpec(
                "accounts",
                "account_id",
                {"account_id": api.DType("int64")},
                time_column=api.NON_TEMPORAL,
            ),
            api.TableSpec(
                "transactions",
                "transaction_id",
                {
                    "transaction_id": api.DType("int64"),
                    "sender_id": api.DType("int64"),
                    "event_time": api.DType("timestamp", unit="s", timezone="UTC"),
                },
                time_column="event_time",
            ),
        ),
        relations=(
            api.RelationSpec("sender", "accounts", "account_id", "transactions", "sender_id"),
        ),
    )
    contract = api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            entity=api.EntitySpec("accounts"),
            neighborhoods={
                "activity": api.NeighborhoodSpec(
                    layers=(
                        api.GraphLayer((api.GraphSample("sender", fanout=1, strategy="recent"),)),
                    )
                )
            },
        ),
        api.BatchSpec(
            inputs={"graph": api.Input(api.NeighborhoodSource("activity"), api.Graph(), None)}
        ),
    )
    # The same public saved contract drives both inspectable and model delivery.
    contract = api.DataLoaderSpec.from_json(contract.to_json())
    engine = api.DiskGraphEngine(index=root / "index", num_threads=1)
    with api.prepare(contract, api.Dataset(sources), engine) as plan:
        yield plan


def _anchor_table(path: Path, names: tuple[str, ...]) -> api.AnchorTable:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table(
            {
                "sample_id": names,
                "account_id": [1] * len(names),
                "cutoff": pa.array(
                    list(range(10, 10 + len(names))), type=pa.timestamp("s", tz="UTC")
                ),
            }
        ),
        path,
    )
    return api.AnchorTable(
        source=api.Parquet(str(path)),
        entity_id="account_id",
        as_of_time="cutoff",
        sample_id="sample_id",
    )


def _assert_delivery(batches, names, positions, padding):
    sample_ids, cutoffs, flags, sizes = [], [], [], []
    for batch in batches:
        ids = list(batch.metadata["sample_ids"])
        sizes.append(len(ids))
        assert len(set(ids)) == len(ids)
        sample_ids.extend(ids)
        cutoffs.extend(batch.metadata["as_of_time"].tolist())
        flags.extend(batch.metadata["is_padding"].tolist())
        assert batch.metadata["entity_ids"].tolist() == [1] * len(ids)
        graph = batch.inputs["graph"]
        # Every request owns its own copy of the account and its one past transaction.
        assert set(graph["nodes"]) == {"accounts", "transactions"}
        for table, expected_id in (("accounts", 1), ("transactions", 100)):
            node = graph["nodes"][table]
            assert sorted(zip(node["ids"].tolist(), node["batch"].tolist(), strict=True)) == [
                (expected_id, owner) for owner in range(len(ids))
            ]
        edge = graph["edges"]["sender"]
        source = graph["nodes"][edge["source_type"]]
        destination = graph["nodes"][edge["destination_type"]]
        assert sorted(
            (int(source["batch"][left]), int(destination["batch"][right]))
            for left, right in edge["index"].t().tolist()
        ) == [(owner, owner) for owner in range(len(ids))]
    assert len(sample_ids) == len(set(sample_ids)) == len(positions)
    assert cutoffs == [10 + position for position in positions]
    assert flags == list(padding)
    real_ids = [
        sample_id for sample_id, padded in zip(sample_ids, flags, strict=True) if not padded
    ]
    assert real_ids == [
        names[position] for position, padded in zip(positions, padding, strict=True) if not padded
    ]
    assert {
        sample_id for sample_id, padded in zip(sample_ids, flags, strict=True) if padded
    }.isdisjoint(real_ids)
    return sizes


@pytest.mark.parametrize(
    ("names", "rank", "world_size", "positions", "padding"),
    [
        # Rank 1 owns positions 1 and 3; its third row repeats position 1.
        pytest.param(
            ("a", "b", "c", "b#padding2", "e"),
            1,
            2,
            (1, 3, 1),
            (False, False, True),
            id="reported-collision",
        ),
        pytest.param(
            ("a", "b#padding2", "c", "b#padding2#padding2", "e"),
            1,
            2,
            (1, 3, 1),
            (False, False, True),
            id="repeated-suffix-collision",
        ),
        pytest.param(
            ("a", "b", "c", "d", "e"),
            1,
            2,
            (1, 3, 1),
            (False, False, True),
            id="ordinary-padding",
        ),
        pytest.param(
            ("a", "b", "c", "b#padding2", "e"),
            0,
            1,
            (0, 1, 2, 3, 4),
            (False,) * 5,
            id="single-rank-custom-ids",
        ),
    ],
)
def test_native_graph_padding_preserves_identity(
    graph_plan, tmp_path, names, rank, world_size, positions, padding
):
    requests = tuple(api.SampleRequest(name, 1, 10 + index) for index, name in enumerate(names))
    task = graph_plan.materialize(requests)
    from_table = graph_plan.materialize(_anchor_table(tmp_path / "anchors.parquet", names))
    for source, selected_task, explicit in (
        ("explicit requests", task, requests),
        ("materialized requests", task, None),
        ("materialized AnchorTable", from_table, None),
    ):
        for output_format in ("inspectable", "model"):
            batches = selected_task.batches(
                explicit,
                batch_size=3,
                rank=rank,
                world_size=world_size,
                shuffle=False,
                seed=0,
                output_format=output_format,
            )
            # Inspect each batch before advancing a native reader that may reuse buffers.
            assert _assert_delivery(batches, names, positions, padding) == (
                [3, 2] if world_size == 1 else [3]
            ), (source, output_format)
    assert tuple(request.sample_id for request in requests) == names
