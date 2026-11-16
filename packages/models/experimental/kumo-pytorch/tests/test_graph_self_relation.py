# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G-GRAPH-001 rule 4: traversal never reverses a declared logical edge."""

import pytest

from conformance.assertions import bound_tables
from kumo.pytorch import api


@pytest.mark.parametrize("engine_name", ["in_memory", "diskgraph"])
@pytest.mark.parametrize("pk_left", [True, False])
@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("connected", [True, False])
def test_self_relation_orientation(engine_name, pk_left, direction, connected):
    # Account 2 names account 1 as its parent. The join, not an engine output,
    # determines the answer: PK -> FK is 1 -> 2; FK -> PK is 2 -> 1.
    rows = [
        {"id": 1, "parent_id": None, "opened_at": 1, "balance": 10.0},
        {"id": 2, "parent_id": 1 if connected else None, "opened_at": 10, "balance": 20.0},
    ]
    left_key, right_key = ("id", "parent_id") if pk_left else ("parent_id", "id")
    expected_edge = (1, 2) if pk_left else (2, 1)
    root = expected_edge[0 if direction == "forward" else 1]
    contract = api.DataLoaderSpec(
        api.DataSchema(
            (
                api.TableSpec(
                    "accounts",
                    "id",
                    {
                        "id": api.DType("int64"),
                        "parent_id": api.DType("int64"),
                        "opened_at": api.DType("timestamp", unit="s", timezone="UTC"),
                        "balance": api.DType("float64"),
                    },
                    time_column="opened_at",
                ),
            ),
            (api.RelationSpec("parent", "accounts", left_key, "accounts", right_key),),
        ),
        api.SampleSpec(
            api.EntitySpec("accounts"),
            neighborhoods={
                "family": api.NeighborhoodSpec(
                    (
                        api.GraphLayer(
                            (api.GraphSample("parent", 4, direction=direction, strategy="recent"),)
                        ),
                    )
                )
            },
        ),
        api.BatchSpec(
            {
                "graph": api.Input(
                    api.NeighborhoodSource("family"),
                    api.Graph({"accounts": api.Fields((api.Encode("balance", api.Raw()),))}),
                    None,
                )
            }
        ),
    )
    contract = api.DataLoaderSpec.from_json(contract.to_json())
    if engine_name == "diskgraph":
        pytest.importorskip("diskgraph")
    requests = [api.SampleRequest("root", root, 100)]
    with (
        bound_tables(contract, {"accounts": rows}, engine_name) as (dataset, engine),
        api.prepare(contract, dataset, engine) as plan,
        plan.materialize(requests) as task,
    ):
        batches = list(
            task.batches(
                batch_size=1,
                output_format="model" if engine_name == "diskgraph" else "inspectable",
            )
        )
    assert len(batches) == 1
    batch = batches[0]
    assert list(batch.metadata["sample_ids"]) == ["root"]
    graph = batch.inputs["graph"]
    nodes = graph["nodes"]["accounts"]
    ids = nodes["ids"].tolist()
    assert set(ids) == ({1, 2} if connected else {root})
    assert nodes["batch"].tolist() == [0] * len(ids)
    assert nodes["features"]["balance"]["values"].tolist() == [10.0 * i for i in ids]
    edge = graph["edges"]["parent"]
    assert edge["source_type"] == edge["destination_type"] == "accounts"
    actual = [(ids[a], ids[b]) for a, b in zip(*edge["index"].tolist(), strict=True)]
    assert actual == ([expected_edge] if connected else [])
    by_id = {row["id"]: row for row in rows}
    assert all(by_id[a][left_key] == by_id[b][right_key] for a, b in actual)
