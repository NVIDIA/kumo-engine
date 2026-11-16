# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph assembly preserves native order and fetches each node's columns together."""

from types import SimpleNamespace

import pytest
import torch

from kumo.pytorch import api
from kumo.pytorch.encoding import OutputBinding, fit_encoder
from kumo.pytorch.engines.diskgraph import DiskGraphSource

pa = pytest.importorskip("pyarrow")


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("global_base", [0, 2**63])
def test_graph_output_preserves_chunks_orientation_and_features(reverse, empty, global_base):
    source = object.__new__(DiskGraphSource)
    source._pa = pa
    source._node_types = ["users", "events"]
    relation = api.RelationSpec("activity", "users", "id", "events", "user_id")
    source._relations = {relation.name: relation}
    source._relation_mapping = {
        relation.name: {
            "forward": "opposite" if reverse else "physical",
            "reverse": "physical" if reverse else "opposite",
        }
    }
    source._relations_by_id = [
        {
            "name": "physical",
            "src_type": "events" if reverse else "users",
            "dst_type": "users" if reverse else "events",
        }
    ]
    source._tables = {
        "users": SimpleNamespace(primary_key="id", columns={"id": api.DType("int64")}),
        "events": SimpleNamespace(
            primary_key="id",
            columns={"id": api.DType("string"), "amount": api.DType("float32")},
        ),
    }
    tables = {
        "users": pa.table({"id": [101, 202]}),
        "events": pa.table(
            {
                "id": [f"e{i}" for i in range(8)],
                "amount": pa.array([None] * 7 + [9.5], type=pa.float32()),
            }
        ),
    }
    calls = []

    def fetch_features(requests):
        for request in requests:
            calls.append(request)
            selected = tables[request["node_type"]].select(request["feature_names"])
            # Return separate feature chunks, including an empty chunk.
            selected = selected.take(
                pa.array([value - global_base for value in request["node_ids"]], type=pa.int64())
            )
            yield {
                "type": "feature",
                "batch": pa.record_batch(
                    [pa.array([], type=field.type) for field in selected.schema],
                    schema=selected.schema,
                ),
            }
            for batch in selected.to_batches(max_chunksize=1):
                yield {"type": "feature", "batch": batch}

    source._sampler = SimpleNamespace(fetch_features=fetch_features)

    def event(kind, left, right):
        return {
            "type": kind,
            "type_id": 1,
            "rel_id": 0,
            "batch": pa.record_batch(
                [
                    pa.array(left, type=pa.uint32()),
                    pa.array(
                        [value + global_base if kind == "node" else value for value in right],
                        type=pa.uint64(),
                    ),
                ],
                names=["id_a", "id_b"],
            ),
        }

    events = [dict(event("node", [0, 1], [0, 1]), type_id=0), event("node", [], [])]
    if not empty:
        events.extend([event("node", [0], [2]), event("node", [1, 1], [7, 2])])
        # The last chunk repeats an edge of the first; G-GRAPH-001 rule 3 keeps one.
        for left, right in (([0, 1], [0, 2]), ([1], [1]), ([1], [2])):
            events.append(event("edge", right if reverse else left, left if reverse else right))
    events.append(event("edge", [], []))
    vocabulary = api.Vocabulary(values=(("e2", 2), ("e7", 3)))
    node_fields = {
        "events": api.Fields((api.Encode("id", vocabulary), api.Encode("amount", api.Raw())))
    }
    encoders = {
        "events": {
            "id": OutputBinding(fit_encoder("id", vocabulary, api.DType("string"), ())),
            "amount": OutputBinding(fit_encoder("amount", api.Raw(), api.DType("float32"), ())),
        }
    }
    graph = source._graph_output(
        "users",
        api.NeighborhoodSpec((api.GraphLayer((api.GraphSample("activity", 3),)),)),
        events,
        {"physical": "activity"},
        node_fields,
        encoders,
    )
    nodes, edge = graph["nodes"], graph["edges"]["activity"]
    assert nodes["users"]["ids"].tolist() == [101, 202]
    assert nodes["users"]["batch"].tolist() == [0, 1]
    if empty:
        assert torch.equal(nodes["events"]["ids"], torch.empty(0, dtype=torch.long))
    else:
        assert nodes["events"]["ids"] == ("e2", "e7", "e2")
    assert nodes["events"]["batch"].tolist() == ([] if empty else [0, 1, 1])
    assert nodes["events"]["features"]["id"]["values"].tolist() == ([] if empty else [2, 3, 2])
    amount = nodes["events"]["features"]["amount"]
    assert amount["values"].tolist() == ([] if empty else [0.0, 9.5, 0.0])
    assert amount["valid"].tolist() == ([] if empty else [False, True, False])
    assert edge["source_type"] == "users" and edge["destination_type"] == "events"
    assert edge["index"].shape == (2, 0 if empty else 3)
    assert edge["index"].tolist() == ([[], []] if empty else [[0, 1, 1], [0, 2, 1]])
    for tensor in (nodes["users"]["batch"], nodes["events"]["batch"], edge["index"]):
        assert tensor.dtype == torch.long and tensor.is_contiguous()
    nodes["users"]["batch"][0] = 99
    assert events[0]["batch"].column("id_a").to_pylist() == [0, 1]
    assert [(call["node_type"], call["feature_names"]) for call in calls] == (
        [("users", ["id"])] if empty else [("events", ["id", "amount"]), ("users", ["id"])]
    )


@pytest.mark.parametrize(
    "dtype,values,expected",
    [
        (pa.int32(), [-2, 5], [-2, 5]),
        (pa.int64(), [-(2**63), 2**63 - 1], [-(2**63), 2**63 - 1]),
        (pa.uint64(), [0, 2**63 - 1], [0, 2**63 - 1]),
        (pa.bool_(), [True, False], [1, 0]),
        (pa.string(), ["café", ""], ("café", "")),
        (pa.int64(), [1, None], (1, None)),
        (pa.float64(), [1.0, 2.0], (1.0, 2.0)),
        (pa.string(), [], []),
    ],
)
def test_graph_identity_preserves_types_and_owns_tensor_storage(dtype, values, expected):
    column = pa.array(values, type=dtype)
    actual = _identity_graph(column)["nodes"]["users"]["ids"]
    if isinstance(expected, tuple):
        assert actual == expected
    else:
        assert actual.dtype == torch.long and actual.is_contiguous()
        assert actual.tolist() == expected
        actual.fill_(0)
        assert column.to_pylist() == values


def test_graph_identity_rejects_uint64_overflow():
    with pytest.raises(ValueError, match="Overflow"):
        _identity_graph(pa.array([2**63], type=pa.uint64()))


def _identity_graph(column):
    source = object.__new__(DiskGraphSource)
    source._pa = pa
    source._node_types = ["users"]
    source._relations = source._relations_by_id = source._relation_mapping = {}
    source._tables = {"users": SimpleNamespace(primary_key="id")}
    source._feature_arrays = lambda *_: {"id": column}
    event = {
        "type": "node",
        "type_id": 0,
        "batch": pa.record_batch(
            [pa.array([0] * len(column), type=pa.uint32()), pa.array(range(len(column)))],
            names=["id_a", "id_b"],
        ),
    }
    return source._graph_output("users", SimpleNamespace(layers=()), [event], {}, {}, {})
