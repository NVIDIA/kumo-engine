# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-owned identity is shared by samples, candidates, and graph roots."""

import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from conformance.assertions import bound_case
from conformance.cases import GUARANTEES, load_case
from kumo.pytorch import api, build


def _case():
    return load_case(GUARANTEES / "G-ENTITY-001/golden/C-ENTITY-001.json")


@pytest.fixture(params=("in_memory", "diskgraph"))
def engine_name(request):
    if request.param == "diskgraph":
        pytest.importorskip("diskgraph")
    return request.param


def test_table_only_entity_defaults_its_type_and_preserves_optional_settings():
    entity = api.EntitySpec("users")
    assert entity.entity_type == "users"
    customized = api.EntitySpec(
        "users",
        entity_type="customer",
        filters=(api.RowFilter("external", "equal", 2),),
        available_from="created",
    )
    assert customized.entity_type == "customer"
    case = _case()
    sample = replace(case.contract.sample, entity=customized)
    table = replace(
        case.contract.schema.tables[0],
        columns={
            "pk": api.DType("int64"),
            "external": api.DType("int64"),
            "created": api.DType("timestamp", unit="s", timezone="UTC"),
        },
    )
    contract = replace(case.contract, schema=api.DataSchema((table,), ()), sample=sample)
    dataset = api.Dataset({"users": api.InMemoryRows([{"pk": 1, "external": 2, "created": 5}])})
    result = api.preview(contract, dataset, api.InMemoryEngine(), case.requests[:1])
    assert result.samples[0]["entity"]["type"] == "customer"
    assert result.to_dict()["samples"][0]["entity"]["values"] == {"external": 2}
    with pytest.raises(ValueError, match="did not exist"):
        api.preview(contract, dataset, api.InMemoryEngine(), (api.SampleRequest("early", 1, 5),))
    rejected = replace(
        contract,
        sample=replace(
            sample, entity=replace(customized, filters=(api.RowFilter("external", "equal", 1),))
        ),
    )
    with pytest.raises(ValueError, match="filters"):
        api.preview(rejected, dataset, api.InMemoryEngine(), case.requests[:1])


def test_entity_and_builder_key_overrides_are_not_accepted():
    with pytest.raises(TypeError, match="key"):
        api.EntitySpec("users", key="external")
    with pytest.raises(TypeError):
        api.EntitySpec("users", "external")
    table = build.Table(
        "users",
        key="pk",
        columns={"pk": api.DType("int64"), "external": api.DType("int64")},
        time=api.NON_TEMPORAL,
    )
    with pytest.raises(TypeError, match="key"):
        build.Sample(entity=table, key="external")
    with pytest.raises(TypeError):
        build.Sample(table, "external")
    sample = build.Sample(entity=table).lower(build.Schema(tables=(table,)))
    assert sample.entity == api.EntitySpec("users")
    customized = build.Sample(entity=table, entity_type="customer").lower(
        build.Schema(tables=(table,))
    )
    assert customized.entity.entity_type == "customer"
    payload = _case().contract.to_dict()
    payload["sample"]["entity"]["key"] = "external"
    with pytest.raises(TypeError, match="key"):
        api.DataLoaderSpec.from_json(json.dumps(payload))


def test_primary_key_answers_metadata_and_artifact_contract(engine_name, tmp_path):
    case = _case()
    assert "key" not in case.contract.to_dict()["sample"]["entity"]
    assert "entity: users.pk (users)" in case.contract.explain()
    with bound_case(case, engine_name) as (dataset, engine):
        assert not api.check(case.contract, dataset, engine)
        with api.prepare(case.contract, dataset, engine) as plan:
            with plan.materialize(case.requests) as task:
                formats = (
                    ("inspectable", "model") if engine_name == "diskgraph" else ("inspectable",)
                )
                for output_format in formats:
                    batch = next(
                        iter(task.batches(case.requests, batch_size=2, output_format=output_format))
                    )
                    assert batch.inputs["values"]["external"]["values"].tolist() == [2, 1]
                    assert list(batch.metadata["entity_ids"]) == [1, 2]
                task.to_artifact().save(tmp_path)
    restored = api.DataLoaderSpec.from_json((tmp_path / "contract.json").read_text())
    assert restored == case.contract
    assert restored.fingerprint == case.contract.fingerprint


@pytest.mark.parametrize("candidate", (False, True))
def test_unknown_entity_table_is_rejected_before_engine_opening(candidate, engine_name):
    case = _case()
    entity = api.EntitySpec("absent")
    sample = (
        replace(case.contract.sample, candidate_sets={"items": api.CandidateSetSpec(entity)})
        if candidate
        else replace(case.contract.sample, entity=entity)
    )
    contract = replace(case.contract, sample=sample)
    with bound_case(case, engine_name, refusal=True) as (dataset, engine):
        issues = api.check(contract, dataset, engine)
        expected_path = (
            "sample.candidate_sets.items.entity.table" if candidate else "sample.entity.table"
        )
        assert any(
            issue.path == expected_path and "unknown table" in issue.message for issue in issues
        )
        with patch.object(
            engine, "open", side_effect=AssertionError("opened invalid entity table")
        ):
            with pytest.raises(api.ValidationError, match="unknown table"):
                api.prepare(contract, dataset, engine)


def test_candidates_use_their_own_tables_primary_key():
    case = _case()
    items = api.TableSpec(
        "items",
        "item_pk",
        {"item_pk": api.DType("int64"), "external": api.DType("int64")},
        time_column=api.NON_TEMPORAL,
    )
    schema = replace(case.contract.schema, tables=(*case.contract.schema.tables, items))
    candidate = api.CandidateSetSpec(
        api.EntitySpec("items"), values={"external": api.Field(column="external")}
    )
    sample = replace(case.contract.sample, candidate_sets={"items": candidate})
    batch = api.BatchSpec(
        {
            "items": api.Input(
                api.CandidateSource("items"),
                api.Fields(
                    fields=(api.Encode(field="external", encoding=api.Raw()),), row=api.Columns()
                ),
                api.Padded(),
            )
        }
    )
    contract = api.DataLoaderSpec(schema, sample, batch)
    requests = (
        api.SampleRequest(
            "one", 1, 10, candidates={"items": (api.CandidateRequest(2), api.CandidateRequest(1))}
        ),
    )
    dataset = api.Dataset(
        {
            "users": api.InMemoryRows(case.tables["users"]),
            "items": api.InMemoryRows(
                [{"item_pk": 1, "external": 2}, {"item_pk": 2, "external": 1}]
            ),
        }
    )
    result = api.preview(contract, dataset, api.InMemoryEngine(), requests).batch
    assert result.inputs["items"]["external"]["values"].tolist() == [[1, 2]]
    assert result.metadata["candidate_entity_ids"]["items"] == ((2, 1),)


def test_graph_roots_and_fitted_attributes_use_the_same_primary_key(engine_name):
    case = _case()
    relation = api.RelationSpec("peer", "users", "pk", "users", "external")
    table = replace(
        case.contract.schema.tables[0],
        columns={
            "pk": api.DType("int64"),
            "external": api.DType("int64"),
            "created": api.DType("timestamp", unit="s", timezone="UTC"),
        },
        time_column="created",
    )
    schema = api.DataSchema((table,), (relation,))
    neighborhood = api.NeighborhoodSpec(
        (api.GraphLayer((api.GraphSample("peer", 1, strategy="recent"),)),)
    )
    sample = replace(case.contract.sample, neighborhoods={"peer": neighborhood})
    graph = api.Input(
        api.NeighborhoodSource("peer"),
        api.Graph(
            {"users": api.Fields(fields=(api.Encode(field="external", encoding=api.Vocabulary()),))}
        ),
        None,
    )
    contract = replace(
        case.contract,
        schema=schema,
        sample=sample,
        batch=replace(case.contract.batch, inputs={**case.contract.batch.inputs, "graph": graph}),
    )
    case = replace(
        case,
        contract=contract,
        tables={"users": [{**row, "created": 0} for row in case.tables["users"]]},
    )
    with bound_case(case, engine_name) as (dataset, engine):
        with api.prepare(contract, dataset, engine) as plan:
            with plan.materialize(case.requests) as task:
                vocabulary = (
                    task.bound_contract()
                    .batch.inputs["graph"]
                    .representation.node_fields["users"]
                    .fields[0]
                    .encoding.values
                )
                formats = (
                    ("inspectable", "model") if engine_name == "diskgraph" else ("inspectable",)
                )
                for output_format in formats:
                    batch = next(
                        iter(task.batches(case.requests, batch_size=2, output_format=output_format))
                    )
                    nodes = batch.inputs["graph"]["nodes"]["users"]
                    edges = batch.inputs["graph"]["edges"]["peer"]["index"]
                    assert edges.shape == (2, 2)
                    assert batch.inputs["values"]["external"]["values"].tolist() == [2, 1]
                    for source, destination in edges.t().tolist():
                        owner = int(nodes["batch"][source])
                        assert int(nodes["ids"][source]) == case.requests[owner].entity_id
                        assert int(nodes["ids"][destination]) == 3 - case.requests[owner].entity_id
                    for raw_id, encoded in zip(
                        nodes["ids"].tolist(),
                        nodes["features"]["external"]["values"].tolist(),
                        strict=True,
                    ):
                        assert encoded == dict(vocabulary)[3 - raw_id]


@pytest.mark.parametrize("key,columns", [("", ("pk",)), (None, ("pk",)), ("pk", ("external",))])
def test_schema_already_rejects_invalid_primary_key_declarations(key, columns):
    with pytest.raises(ValueError, match="primary key"):
        api.TableSpec(
            "users",
            key,
            {column: api.DType("int64") for column in columns},
            time_column=api.NON_TEMPORAL,
        )
