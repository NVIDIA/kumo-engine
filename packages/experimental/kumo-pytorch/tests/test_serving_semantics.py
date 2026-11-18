# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allowed serving differences must not conceal wrong values or connectivity."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from kumo.pytorch import api as r
from kumo.pytorch.encoding import FittedEncoder
from tests.graph_semantics import SourceRows, assert_graph
from tests.serving_assertions import _compare, assert_serving_batches


def graph_fixture(selected=(11, 12)):
    schema = r.DataSchema(
        (
            r.TableSpec("users", "id", {"id": r.DType("int64")}, time_column=r.NON_TEMPORAL),
            r.TableSpec(
                "events",
                "id",
                {
                    "id": r.DType("int64"),
                    "owner": r.DType("int64"),
                    "at": r.DType("timestamp", unit="s", timezone="UTC"),
                    "price": r.DType("float64"),
                },
                time_column="at",
            ),
        ),
        (r.RelationSpec("activity", "users", "id", "events", "owner"),),
    )
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(
            r.EntitySpec("users"),
            neighborhoods={
                "recent": r.NeighborhoodSpec((r.GraphLayer((r.GraphSample("activity", 2),)),))
            },
        ),
        r.BatchSpec(
            {
                "graph": r.Input(
                    r.NeighborhoodSource("recent"),
                    r.Graph({"events": r.Fields((r.Encode("price", r.ZScore()),))}),
                    None,
                )
            }
        ),
    )
    tables = {
        "users": [{"id": 1}, {"id": 2}],
        "events": [
            {"id": identity, "owner": owner, "at": at, "price": float(identity)}
            for identity, owner, at in (
                (10, 1, 7),
                (11, 1, 8),
                (12, 1, 8),
                (13, 1, 8),
                (14, 1, 10),
                (15, 2, 8),
            )
        ],
    }
    graph = {
        "nodes": {
            "users": {"ids": torch.tensor([1]), "batch": torch.tensor([0]), "features": {}},
            "events": {
                "ids": torch.tensor(selected),
                "batch": torch.zeros(len(selected), dtype=torch.int64),
                "features": {
                    "price": {
                        "values": torch.tensor(
                            [identity / 10 for identity in selected], dtype=torch.float32
                        ),
                        "valid": torch.ones(len(selected), dtype=torch.bool),
                    }
                },
            },
        },
        "edges": {
            "activity": {
                "source_type": "users",
                "destination_type": "events",
                "index": torch.tensor([[0] * len(selected), list(range(len(selected)))]),
            }
        },
    }
    state = {
        "encoders": {
            '["inputs","graph","graph","events","price"]': FittedEncoder(
                "price", r.ZScore(), r.DType("float64"), mean=0.0, scale=10.0
            ).state_dict()
        }
    }
    return graph, contract, [r.SampleRequest("a", 1, 9)], SourceRows(schema, tables), state


def validate(fixture):
    graph, contract, requests, source, state = fixture
    assert_graph(graph, contract, "graph", requests, source, state)


@pytest.mark.parametrize("selected", [(11, 12), (12, 11), (11, 13), (13, 12)])
def test_graph_allows_reordering_and_eligible_boundary_ties(selected):
    validate(graph_fixture(selected))


@pytest.mark.parametrize("count", [32, 33])
def test_graph_revisited_tied_relation_uses_lazy_complete_witness(count):
    graph, contract, requests, source, state = graph_fixture(tuple(range(100, 100 + count)))
    layers = (
        r.GraphLayer((r.GraphSample("activity", 16),)),
        r.GraphLayer((r.GraphSample("activity", 1, direction="reverse"),)),
        r.GraphLayer((r.GraphSample("activity", 16),)),
    )
    contract = replace(
        contract,
        sample=replace(contract.sample, neighborhoods={"recent": r.NeighborhoodSpec(layers)}),
    )
    source.tables["events"] = [
        {"id": i, "owner": 1, "at": 8, "price": float(i)} for i in range(100, 100 + count)
    ]
    # All 32 events require two different valid fanout-16 selections. Eager
    # enumeration would construct C(32,16) = 601,080,390 choices before checking.
    if count == 32:
        assert_graph(graph, contract, "graph", requests, source, state)
    else:
        with pytest.raises(AssertionError):
            assert_graph(graph, contract, "graph", requests, source, state)


@pytest.mark.parametrize("selected", [(10, 11), (11, 14), (11, 15), (11,), (11, 12, 13), (11, 11)])
def test_graph_rejects_wrong_membership_multiplicity_and_fanout(selected):
    with pytest.raises(AssertionError):
        validate(graph_fixture(selected))


@pytest.mark.parametrize(
    "fault",
    [
        "value",
        "valid",
        "dtype",
        "edge-negative",
        "edge-too-large",
        "edge-missing",
        "edge-duplicate",
        "owner",
        "root",
        "orientation",
    ],
)
def test_graph_rejects_corrupt_values_structure_and_edges(fault):
    fixture = graph_fixture()
    graph = fixture[0]
    node = graph["nodes"]["events"]
    edge = graph["edges"]["activity"]
    if fault == "value":
        node["features"]["price"]["values"][0] += 1
    elif fault == "valid":
        node["features"]["price"]["valid"][0] = False
    elif fault == "dtype":
        node["features"]["price"]["values"] = node["features"]["price"]["values"].double()
    elif fault == "edge-negative":
        edge["index"][1] -= len(node["ids"])
    elif fault == "edge-too-large":
        edge["index"][1, 0] = 99
    elif fault == "edge-missing":
        edge["index"] = edge["index"][:, :1]
    elif fault == "edge-duplicate":
        edge["index"] = torch.cat((edge["index"], edge["index"][:, :1]), dim=1)
    elif fault == "owner":
        node["batch"][0] = 1
    elif fault == "root":
        graph["nodes"]["users"]["ids"][0] = 2
    else:
        edge["source_type"], edge["destination_type"] = "events", "users"
    with pytest.raises(AssertionError):
        validate(fixture)


def test_only_declared_float64_statistics_allow_rounding():
    expected = {
        "mean": {"values": torch.tensor([2.0], dtype=torch.float64), "valid": torch.tensor([True])},
        "count": {"values": torch.tensor([2]), "valid": torch.tensor([True])},
    }
    actual = deepcopy(expected)
    actual["mean"]["values"] += 1e-13
    _compare(actual, expected, "entity", {"entity.mean.values"})
    for path in ("mean", "count"):
        bad = deepcopy(actual)
        bad[path]["values"] += 1
        with pytest.raises(AssertionError):
            _compare(bad, expected, "entity", {"entity.mean.values"})
    bad = deepcopy(actual)
    bad["mean"]["valid"][0] = False
    with pytest.raises(AssertionError):
        _compare(bad, expected, "entity", {"entity.mean.values"})
    with pytest.raises(AssertionError):
        _compare(actual, expected, "entity", set())


def test_serving_sequence_ties_are_source_validated_on_both_sides():
    from conformance.assertions import PropertyFailure
    from conformance.sequence_cases import COMPONENT_TABLES
    from tests.conformance.test_sampling_harness import _tied_sequence_fixture

    case, answer, event = _tied_sequence_fixture()

    def batch(events):
        inputs = {}
        for name, rows in events.items():
            fields = {
                field: {
                    "values": torch.tensor(
                        [[dict((f, v) for f, _, v in row[2])[field] for row in rows]], dtype=dtype
                    ),
                    "valid": torch.ones((1, len(rows)), dtype=torch.bool),
                }
                for field, dtype in (
                    ("id", torch.int64),
                    ("amount", torch.float64),
                    ("delta", torch.float64),
                )
            }
            fields["component"] = {
                "values": torch.tensor([[tuple(COMPONENT_TABLES).index(row[1]) for row in rows]]),
                "valid": torch.ones((1, len(rows)), dtype=torch.bool),
            }
            fields["valid"] = torch.ones((1, len(rows)), dtype=torch.bool)
            inputs[name] = fields
        return r.Batch(inputs, {}, {"sample_ids": ["a"]})

    original = batch(answer["a"])
    answer["a"]["merged"] = (event(7, "orders", 1), event(8, "clicks", 10), event(8, "orders", 4))
    answer["a"]["pushed"] = (event(7, "recent", 1), event(8, "recent", 3))
    alternative = batch(answer["a"])
    artifact = SimpleNamespace(
        contract=case.contract, fitted_state=SimpleNamespace(to_dict=lambda: {})
    )

    def check(left, right):
        assert_serving_batches(
            [left], [right], artifact, case.tables, case.requests, sequence_case=case
        )

    check(original, alternative)
    for fault in ("value", "dtype", "mask", "padding", "component", "batch-dimension"):
        bad = deepcopy(alternative)
        column = bad.inputs["merged"]["amount"]
        if fault == "value":
            column["values"][0, 0] += 1
        elif fault == "dtype":
            column["values"] = column["values"].float()
        elif fault == "mask":
            column["valid"][0, 0] = False
        elif fault == "padding":
            bad.inputs["merged"]["valid"][0, 0] = False
        elif fault == "component":
            bad.inputs["merged"]["component"]["valid"][0, 0] = False
        else:
            for name, field in bad.inputs["merged"].items():
                if name == "valid":
                    bad.inputs["merged"][name] = field.repeat(2, 1)
                else:
                    for key in field:
                        field[key] = field[key].repeat(2, 1)
        for left, right in ((original, bad), (bad, alternative)):
            with pytest.raises(PropertyFailure):
                check(left, right)


@pytest.mark.parametrize("dtype", ["float32", "int32", "list"])
def test_graph_numeric_id_dtype_is_not_relaxed_by_identity_matching(dtype):
    fixture = graph_fixture()
    for node in fixture[0]["nodes"].values():
        node["ids"] = (
            node["ids"].float().tolist()
            if dtype == "list"
            else node["ids"].to(getattr(torch, dtype))
        )
    with pytest.raises(AssertionError):
        validate(fixture)


def test_hm_graph_serving_matches_complete_source_semantics(tmp_path):
    from e2e.datasets.hm import generate
    from e2e.flows.common import CustomerRows, anchors, dataset, serving_requests
    from e2e.flows.hm_gnn import contract
    from tests.native_fixtures import request_anchor_table
    from tests.serving_assertions import assert_serving_batches

    hm = generate("XS", 7, tmp_path / "hm")
    spec = contract()
    engine = r.DiskGraphEngine(index=tmp_path / "index", num_threads=1, num_shards=1)
    store = CustomerRows.open(hm.root)
    requests = serving_requests(hm.root, store, count=4)
    with r.prepare(spec, dataset(hm.root), engine) as plan:
        fitted = plan.materialize(anchors(hm.root), fit_split="train").fitted_state
        task = plan.materialize(
            request_anchor_table(requests, schema=spec.schema), fitted=fitted, inference=True
        )
        artifact = task.to_artifact()
        expected = list(task.batches(batch_size=1, shuffle=False, output_format="model", seed=0))
    server = r.ServingAssembler(artifact)
    source = SourceRows(spec.schema, store.tables)
    by_id = {b.metadata["sample_ids"][0]: b for b in expected}
    for request in requests:
        served = server.assemble(store.lookup(server.rows_spec, request), [request], seed=0)
        assert_serving_batches(served, [by_id[request.sample_id]], artifact, source, [request])
