# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keyless source occurrences work throughout the public loader and serving flow."""

from dataclasses import replace
from unittest.mock import patch

import pyarrow as pa
import pytest

import kumo.pytorch.api as r
from conformance.assertions import bound_case
from conformance.cases import GUARANTEES, load_case
from kumo.pytorch import build


def _case():
    return load_case(GUARANTEES / "G-KEYLESS-001/golden/C-KEYLESS-001.json")


@pytest.fixture(params=("in_memory", "diskgraph"))
def engine_name(request):
    if request.param == "diskgraph":
        pytest.importorskip("diskgraph")
    return request.param


def _graph_case():
    case = _case()
    tables = []
    for table in case.contract.schema.tables:
        if table.name == "users":
            table = replace(
                table,
                columns={
                    **table.columns,
                    "created": r.DType("timestamp", unit="s", timezone="UTC"),
                },
                time_column="created",
            )
        tables.append(table)
    history = r.EventSpec(r.Rows(("activity",)), {"amount": r.Field(column="amount")})
    layers = tuple(
        r.GraphLayer((r.GraphSample("activity", 10, direction=direction, strategy="recent"),))
        for direction in ("forward", "reverse", "forward")
    )
    sample = replace(
        case.contract.sample,
        events={"history": history},
        neighborhoods={"activity": r.NeighborhoodSpec(layers)},
        targets={
            "next": r.RowTarget(
                r.Rows(("activity",), time=r.TimeWindow("at", start=0, stop=2)),
                {"amount": r.Field(column="amount")},
            )
        },
    )
    fields = r.Fields((r.Encode("amount", r.Raw()),))
    batch = replace(
        case.contract.batch,
        inputs={
            **case.contract.batch.inputs,
            "history": r.Input(r.EventSource(("history",)), fields, r.Packed()),
            "graph": r.Input(r.NeighborhoodSource("activity"), r.Graph({"events": fields}), None),
        },
    )
    return replace(
        case,
        contract=replace(
            case.contract,
            schema=r.DataSchema(tuple(tables), case.contract.schema.relations),
            sample=sample,
            batch=batch,
        ),
        tables={**case.tables, "users": [{**row, "created": 0} for row in case.tables["users"]]},
    )


@pytest.mark.parametrize("candidate", (False, True))
@pytest.mark.parametrize("inferred", (False, True))
def test_keyless_entities_are_refused_before_opening(candidate, inferred, engine_name):
    case = _case()
    entity = r.EntitySpec("events")
    sample = (
        replace(case.contract.sample, candidate_sets={"items": r.CandidateSetSpec(entity)})
        if candidate
        else replace(case.contract.sample, entity=entity)
    )
    contract = replace(case.contract, sample=sample)
    if inferred:
        contract = replace(
            contract,
            schema=replace(
                contract.schema,
                tables=tuple(
                    replace(table, columns=None) if table.name == "events" else table
                    for table in contract.schema.tables
                ),
            ),
        )
    expected_path = (
        "sample.candidate_sets.items.entity.table" if candidate else "sample.entity.table"
    )
    with bound_case(case, engine_name, refusal=True) as (dataset, engine):
        with (
            patch.object(
                r.Dataset, "resolve_schema", side_effect=AssertionError("read source metadata")
            ),
            patch.object(engine, "open", side_effect=AssertionError("opened invalid entity")),
        ):
            issues = r.check(contract, dataset, engine)
            assert any(
                issue.path == expected_path and "requires a primary key" in issue.message
                for issue in issues
            )
            with pytest.raises(r.ValidationError, match="requires a primary key"):
                r.prepare(contract, dataset, engine)


@pytest.mark.parametrize(
    ("engine_name", "output_format"),
    (("in_memory", "inspectable"), ("diskgraph", "inspectable"), ("diskgraph", "model")),
    indirect=["engine_name"],
)
def test_keyless_graph_history_targets_and_features(engine_name, output_format):
    case = _graph_case()
    with bound_case(case, engine_name) as (dataset, engine):
        with r.prepare(case.contract, dataset, engine) as plan:
            with plan.materialize(case.requests) as task:
                batch = next(
                    iter(task.batches(case.requests, batch_size=3, output_format=output_format))
                )
                assert batch.inputs["values"]["count"]["values"].tolist() == [2, 3, 0]
                assert batch.inputs["history"]["amount"]["values"].tolist() == [2, 2, 2, 2, 3]
                graph = batch.inputs["graph"]
                nodes = graph["nodes"]["events"]
                ids, owners = nodes["ids"].tolist(), nodes["batch"].tolist()
                assert len(set(zip(owners, ids))) == 5
                assert [owners.count(i) for i in range(3)] == [2, 3, 0]
                assert sorted(nodes["features"]["amount"]["values"].tolist()) == [2, 2, 2, 2, 3]
                edges = graph["edges"]["activity"]["index"]
                assert edges.shape == (2, 5)
                assert len(set(map(tuple, edges.t().tolist()))) == 5
                preview = task.preview(case.requests).to_dict()
                assert preview["samples"][1]["targets"]["next"]["fields"][0]["value"] == 100


@pytest.mark.parametrize("arrow", (False, True))
def test_keyless_artifact_and_serving_keep_identical_rows(engine_name, arrow, tmp_path):
    case = _case()
    with bound_case(case, engine_name) as (dataset, engine):
        with r.prepare(case.contract, dataset, engine) as plan:
            with plan.materialize(case.requests) as task:
                task.to_artifact().save(tmp_path / "artifact")
    artifact = r.Artifact.load(tmp_path / "artifact")
    server = r.ServingAssembler(artifact)
    assert all(
        column.role != "primary key"
        for column in server.rows_spec.tables["events"].columns.values()
    )
    rows = {
        name: pa.Table.from_pylist(values) if arrow else values
        for name, values in case.tables.items()
    }
    batch = server.assemble(rows, case.requests)[0]
    assert batch.inputs["values"]["count"]["values"].tolist() == [2, 3, 0]
    assert batch.inputs["values"]["sum"]["values"].tolist() == [4, 7, 0]
    assert artifact.contract.to_dict() == case.contract.to_dict()


def test_keyless_builder_and_inferred_columns(tmp_path):
    import pyarrow.parquet as pq

    case = _case()
    event = next(table for table in case.contract.schema.tables if table.name == "events")
    assert build.Table("events", None, event.columns, time="at").lower() == event
    inferred = replace(event, columns=None)
    schema = replace(
        case.contract.schema,
        tables=tuple(inferred if t.name == "events" else t for t in case.contract.schema.tables),
    )
    sources = {}
    for table in case.contract.schema.tables:
        path = tmp_path / f"{table.name}.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                case.tables[table.name],
                schema=pa.schema(
                    [(name, dtype.to_arrow()) for name, dtype in table.columns.items()]
                ),
            ),
            path,
        )
        sources[table.name] = r.Parquet(str(path))
    resolved = replace(case.contract, schema=schema).resolve(dataset=r.Dataset(sources))
    assert resolved.schema == case.contract.schema


def test_keyless_non_primary_key_relation_remains_valid_on_reference():
    case = _case()
    # Reference joins are defined by relation columns, not by PK/FK declarations.
    users = replace(
        case.contract.schema.tables[1], columns={"id": r.DType("int64"), "group": r.DType("int64")}
    )
    relation = replace(case.contract.schema.relations[0], left_key="group")
    contract = replace(
        case.contract, schema=r.DataSchema((case.contract.schema.tables[0], users), (relation,))
    )
    tables = {**case.tables, "users": [{"id": 1, "group": 1}, {"id": 2, "group": 2}]}
    dataset = r.Dataset({name: r.InMemoryRows(rows) for name, rows in tables.items()})
    preview = r.preview(contract, dataset, r.InMemoryEngine(), case.requests)
    assert preview.batch.inputs["values"]["count"]["values"].tolist() == [2, 3, 0]


@pytest.mark.parametrize(
    ("engine_name", "output_format"),
    (("in_memory", "inspectable"), ("diskgraph", "inspectable"), ("diskgraph", "model")),
    indirect=["engine_name"],
)
def test_keyless_constant_only_events(engine_name, output_format):
    case = _case()
    event = r.EventSpec(
        r.Rows(("activity",)), {"marker": r.Constant(value=1, dtype=r.DType("int64"))}
    )
    contract = replace(
        case.contract,
        sample=replace(case.contract.sample, events={"history": event}),
        batch=r.BatchSpec(
            {
                "history": r.Input(
                    r.EventSource(("history",)),
                    r.Fields((r.Encode("marker", r.Raw()),)),
                    r.Packed(),
                )
            }
        ),
    )
    with bound_case(replace(case, contract=contract), engine_name) as (dataset, engine):
        with r.prepare(contract, dataset, engine) as plan:
            with plan.materialize(case.requests) as task:
                batch = next(
                    iter(task.batches(case.requests, batch_size=3, output_format=output_format))
                )
                assert batch.inputs["history"]["marker"]["values"].tolist() == [1] * 5


def test_keyless_non_temporal_count_only_frame(engine_name):
    case = _case()
    tables = tuple(
        replace(table, time_column=r.NON_TEMPORAL) if table.name == "events" else table
        for table in case.contract.schema.tables
    )
    sample = replace(
        case.contract.sample, entity_values={"count": r.Aggregate(r.Rows(("activity",)), "count")}
    )
    contract = replace(
        case.contract,
        schema=replace(case.contract.schema, tables=tables),
        sample=sample,
        batch=r.BatchSpec(
            {
                "values": r.Input(
                    r.EntitySource(), r.Fields((r.Encode("count", r.Raw()),)), r.Separate()
                )
            }
        ),
    )
    with bound_case(replace(case, contract=contract), engine_name) as (dataset, engine):
        batch = r.preview(contract, dataset, engine, case.requests).batch
        assert batch.inputs["values"]["count"]["values"].tolist() == [4, 4, 0]


def test_keyless_tie_order_survives_an_ordered_serving_subset():
    case = _case()
    event = r.EventSpec(
        r.Rows(("activity",)), {"amount": r.Field(column="amount")}, order="descending", limit=1
    )
    contract = replace(
        case.contract,
        sample=replace(case.contract.sample, events={"history": event}),
        batch=r.BatchSpec(
            {
                "history": r.Input(
                    r.EventSource(("history",)),
                    r.Fields((r.Encode("amount", r.Raw()),)),
                    r.Packed(),
                )
            }
        ),
    )
    # Removing an unrelated row must not change numeric source-order ties.
    tables = {
        **case.tables,
        "events": [
            {"owner": 2, "at": 9, "amount": 99},
            *({"owner": 1, "at": 9, "amount": i} for i in range(12)),
        ],
    }
    requests = (r.SampleRequest("tied", 1, 9),)
    dataset = r.Dataset({name: r.InMemoryRows(rows) for name, rows in tables.items()})
    with r.prepare(contract, dataset, r.InMemoryEngine()) as plan:
        with plan.materialize(requests) as task:
            offline = next(iter(task.batches(requests, batch_size=1)))
            server = r.ServingAssembler(task.to_artifact())
    served = server.assemble({**tables, "events": tables["events"][1:]}, requests)[0]
    assert offline.inputs["history"]["amount"]["values"].tolist() == [11]
    assert served.inputs["history"]["amount"]["values"].tolist() == [11]


def test_keyless_serving_graphs_keep_occurrences_after_fetch(engine_name):
    from tests.serving_fetch import fetch

    case = _graph_case()
    case = replace(
        case,
        tables={
            **case.tables,
            "events": [{"owner": 2, "at": 5, "amount": 42}, *case.tables["events"]],
        },
        requests=(case.requests[1],),
    )
    with bound_case(case, engine_name) as (dataset, engine):
        with r.prepare(case.contract, dataset, engine) as plan:
            with plan.materialize(case.requests) as task:
                offline = next(iter(task.batches(case.requests, batch_size=1))).inputs["graph"]
                server = r.ServingAssembler(task.to_artifact())
    rows = fetch(server.rows_spec, server.contract.schema, case.tables, case.requests[0])
    served = server.assemble(rows, case.requests)[0].inputs["graph"]
    # Opaque IDs are local; compare features and connectivity instead of their values.
    for graph in (offline, served):
        nodes = graph["nodes"]["events"]
        assert sorted(nodes["features"]["amount"]["values"].tolist()) == [2, 2, 3]
        assert len(set(nodes["ids"].tolist())) == 3
        assert graph["edges"]["activity"]["index"].shape == (2, 3)


def test_keyless_serving_reduction_retains_local_floating_rounding():
    case = _case()
    tables = tuple(
        replace(table, columns={**table.columns, "amount": r.DType("float64")})
        if table.name == "events"
        else table
        for table in case.contract.schema.tables
    )
    sample = replace(
        case.contract.sample,
        entity_values={"sum": r.Aggregate(r.Rows(("activity",)), "sum", "amount")},
    )
    contract = replace(
        case.contract,
        schema=replace(case.contract.schema, tables=tables),
        sample=sample,
        batch=r.BatchSpec(
            {
                "values": r.Input(
                    r.EntitySource(), r.Fields((r.Encode("sum", r.Raw()),)), r.Separate()
                )
            }
        ),
    )
    rows = {
        **case.tables,
        "events": [
            {"owner": 1, "at": i + 1, "amount": amount}
            for i, amount in enumerate((1e16, -1e16, 1.0))
        ],
    }
    requests = (r.SampleRequest("rounding", 1, 9),)
    dataset = r.Dataset({name: r.InMemoryRows(values) for name, values in rows.items()})
    with r.prepare(contract, dataset, r.InMemoryEngine()) as plan:
        with plan.materialize(requests) as task:
            offline = next(iter(task.batches(requests, batch_size=1)))
            server = r.ServingAssembler(task.to_artifact())
    served = server.assemble({**rows, "events": rows["events"][::-1]}, requests)[0]
    # Binary64: (1e16 - 1e16) + 1 = 1; (1 - 1e16) + 1e16 = 0.
    assert offline.inputs["values"]["sum"]["values"].tolist() == [1.0]
    assert served.inputs["values"]["sum"]["values"].tolist() == [0.0]
