# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serving from caller-supplied rows (G-SERVING-001): the startup/per-call split,
`ServingAssembler.rows_spec`, and the refusals of rows that don't fit the contract.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pytest
import torch

import kumo.pytorch.preparation as preparation
import kumo.pytorch.serving as serving
from conformance.cases import discover_cases
from conformance.graph_cases import generate_graph_case
from conformance.sequence_cases import generate_sequence_case
from kumo.pytorch import api
from kumo.pytorch.serving import RowLimit, RowsJoin, ServingRowsSpec
from tests import lifecycle_app
from tests.native_fixtures import prepare_rows, request_anchor_table
from tests.serving_fetch import assert_offline_batch, fetch, offline_batches
from tests.test_diskgraph_backend import _fixture_rows, _requests, _schema, _temporal_contract
from tests.test_inference_lifecycle import assert_same_batches

ROOT = Path(__file__).parents[1]
Row = dict[str, object]
Tables = dict[str, list[Row]]
SEED = 7


def _example() -> ModuleType:
    return lifecycle_app


def _artifact(
    contract: api.DataLoaderSpec, tables: Tables, requests: Sequence[api.SampleRequest]
) -> api.Artifact:
    with prepare_rows(contract, tables) as plan:
        anchors = request_anchor_table(requests, schema=plan.contract.schema)
        with plan.materialize(anchors) as task:
            return task.to_artifact()


def _bounded_rows_contract() -> tuple[api.DataLoaderSpec, Tables, list[api.SampleRequest]]:
    """C-ROWS-001 with only its bounded count: the spec's window is bounded."""

    case = discover_cases()["C-ROWS-001"]
    contract = case.contract
    sample = replace(
        contract.sample,
        entity_values={"recent_count": contract.sample.entity_values["recent_count"]},
    )
    entity = contract.batch.inputs["entity"]
    fields = tuple(use for use in entity.representation.fields if use.field == "recent_count")
    batch = replace(
        contract.batch,
        inputs={
            "entity": replace(entity, representation=replace(entity.representation, fields=fields))
        },
    )
    return replace(contract, sample=sample, batch=batch), case.tables, list(case.requests)


def _graph_case(path: str, ties: str) -> tuple[api.DataLoaderSpec, Tables, list[api.SampleRequest]]:
    dims = {
        "schema_shape.graph_path": path,
        "schema_shape.graph_orientation": "aligned",
        "contract.graph_direction": "forward",
        "contract.graph_fanout": 2,
        "time.graph_ties": ties,
        "time.cutoffs": "mixed",
    }
    case = generate_graph_case(11, 0, dims, "XS")
    return case.contract, case.tables, list(case.requests)


def _limited_events_case(ties: str) -> tuple[api.DataLoaderSpec, Tables, list[api.SampleRequest]]:
    """A generated sequence case without its unlimited reader: the spec keeps the newest 2."""

    dims = {
        "contract.event_order": "descending",
        "contract.event_limit": "limited",
        "contract.event_keep": "last",
        "contract.event_max": "none",
        "contract.event_delta_unit": "seconds",
        "time.ties": ties,
        "time.cutoffs": "mixed",
    }
    case = generate_sequence_case(13, 0, dims, "XS")
    contract = case.contract
    sample = replace(
        contract.sample,
        events={name: spec for name, spec in contract.sample.events.items() if name != "recent"},
    )
    batch = replace(
        contract.batch,
        inputs={name: spec for name, spec in contract.batch.inputs.items() if name != "pushed"},
    )
    return replace(contract, sample=sample, batch=batch), case.tables, list(case.requests)


def _event_lookup_case() -> tuple[api.DataLoaderSpec, Tables, list[api.SampleRequest]]:
    """A latest lookup as of each event: the newest row overall can be after the event."""

    day, clock = 86_400, api.DType("timestamp", unit="s", timezone="UTC")
    key = api.DType("int64")
    schema = api.DataSchema(
        tables=(
            api.TableSpec("users", "user_id", {"user_id": key}, time_column=api.NON_TEMPORAL),
            api.TableSpec(
                "purchases",
                "purchase_id",
                {"purchase_id": key, "user_id": key, "product_id": key, "at": clock},
                time_column="at",
            ),
            api.TableSpec(
                "prices",
                "price_id",
                {"price_id": key, "product_id": key, "at": clock, "price": api.DType("float64")},
                time_column="at",
            ),
        ),
        relations=(
            api.RelationSpec("user_purchases", "users", "user_id", "purchases", "user_id"),
            api.RelationSpec("purchase_prices", "purchases", "product_id", "prices", "product_id"),
        ),
    )
    price = api.Field(
        column="price", path=("purchase_prices",), selection="latest", reference_time="event"
    )
    sample = api.SampleSpec(
        entity=api.EntitySpec("users"),
        events={
            "purchases": api.EventSpec(
                api.Rows(("user_purchases",), api.TimeWindow("at")), fields={"price": price}
            )
        },
    )
    batch = api.BatchSpec(
        inputs={
            "recent": api.Input(
                api.EventSource(("purchases",)),
                api.Fields((api.Encode("price", api.Raw()),), row=api.Columns()),
                api.Padded(),
            )
        }
    )
    tables: Tables = {
        "users": [{"user_id": 1}],
        "purchases": [{"purchase_id": 1, "user_id": 1, "product_id": 10, "at": 5 * day}],
        "prices": [
            {"price_id": 1, "product_id": 10, "at": 1 * day, "price": 5.0},
            {"price_id": 2, "product_id": 10, "at": 8 * day, "price": 9.0},
        ],
    }
    return api.DataLoaderSpec(schema, sample, batch), tables, [api.SampleRequest("r", 1, 10 * day)]


def _example_case() -> tuple[api.DataLoaderSpec, Tables, list[api.SampleRequest]]:
    example = _example()
    return example.contract(), example.tables(), example.anchors(example.SCORE_AT, split=None)[:8]


CASES: dict[str, Callable[[], tuple[api.DataLoaderSpec, Tables, list[api.SampleRequest]]]] = {
    "example": _example_case,
    "event_reference_lookup": _event_lookup_case,
    "bounded_window": _bounded_rows_contract,
    "event_limit": lambda: _limited_events_case("distinct"),
    "event_limit_tied": lambda: _limited_events_case("tied"),
    "graph_three_hop": lambda: _graph_case("three_hop", "distinct"),
    "graph_wide": lambda: _graph_case("wide", "distinct"),
    "graph_back_and_forth_tied": lambda: _graph_case("back_and_forth", "tied"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_fetching_what_rows_spec_lists_gives_batch_inference(name: str) -> None:
    contract, tables, requests = CASES[name]()
    artifact, expected = offline_batches(contract, tables, requests, case_id=name)
    server = api.ServingAssembler(artifact)
    for request in requests:
        rows = fetch(server.rows_spec, artifact.contract.schema, tables, request)
        (served,) = server.assemble(rows, [request], seed=SEED)
        assert_offline_batch(served, expected[request.sample_id], artifact, tables, request)


def test_an_event_limit_alone_limits_the_rows_to_fetch() -> None:
    contract, _, _ = _limited_events_case("distinct")
    orders = ServingRowsSpec.from_contract(contract).tables["orders"]
    assert orders.limit == RowLimit(2, "newest", "at", "request", "events.orders")


def test_the_bounded_window_is_the_widest_reader_window() -> None:
    contract, _, _ = _bounded_rows_contract()
    events = ServingRowsSpec.from_contract(contract).tables["events"]
    assert events.window == api.TimeWindow("at", -1, 0)
    assert "request.as_of_time - 1s < events.at <= request.as_of_time" in str(
        ServingRowsSpec.from_contract(contract)
    )


def test_spec_fetch_of_a_graph_contract_equals_batch_inference(tmp_path: Path) -> None:
    # columns, events, two-hop graph and future-row targets
    contract = _temporal_contract(_schema())
    rows = _fixture_rows()
    artifact, expected = offline_batches(
        contract, rows, _requests(), case_id="train_split_graph", fit_split="train"
    )
    server = api.ServingAssembler(artifact)
    spec = server.rows_spec
    assert spec.tables["card_transactions"].required
    assert len(spec.tables["card_transactions"].routes) == 2
    assert not spec.tables["merchants"].required
    assert not spec.tables["card_transactions"].columns["description"].required
    for request in _requests():
        fetched = fetch(spec, artifact.contract.schema, rows, request)
        assert "merchants" not in fetched
        assert all("description" not in row for row in fetched["card_transactions"])
        (served,) = server.assemble(fetched, [request], seed=SEED)
        assert_offline_batch(served, expected[request.sample_id], artifact, rows, request)


EXAMPLE_SPEC = """\
Rows to supply per request (entity: users, keyed by request.entity_id)

users        required · 1 row per request
  match      users.user_id = request.entity_id
  columns    user_id  int64   primary key
             segment  string  read by: user.segment

purchases    required · 0..n rows per request
  match      purchases.user_id = request.entity_id   (relation user_purchases)
             purchases.at <= request.as_of_time
  history    ALL matching rows: entity_values.purchases_so_far counts the full
             history (events.purchases keeps only the newest 8, but the count
             needs every row)
  columns    purchase_id  int64              primary key
             user_id      int64              join key
             at           timestamp[s, UTC]  time column
             amount       float64            read by: recent.amount
             category     string             read by: recent.category

Not needed for serving: rows after request.as_of_time (read only by target
"active", which inference never computes)"""


@pytest.fixture(scope="module")
def example_server() -> api.ServingAssembler:
    contract, tables, requests = _example_case()
    return api.ServingAssembler(_artifact(contract, tables, requests))


def test_rows_spec_renders_the_example_contract(example_server: api.ServingAssembler) -> None:
    assert str(example_server.rows_spec) == EXAMPLE_SPEC


def test_rows_spec_reads_programmatically(example_server: api.ServingAssembler) -> None:
    spec = example_server.rows_spec
    assert spec.entity_table == "users"
    users, purchases = spec.tables["users"], spec.tables["purchases"]
    assert users.required and users.hop == 0
    assert users.join == RowsJoin("user_id", request_field="entity_id")
    assert purchases.join == RowsJoin(
        "user_id", relation="user_purchases", from_table="users", from_column="user_id", from_hop=0
    )
    assert (purchases.hop, purchases.time_column, purchases.window, purchases.limit) == (
        1,
        "at",
        None,
        None,
    )
    assert "entity_values.purchases_so_far" in purchases.route.history
    assert purchases.columns["at"].dtype == api.DType("timestamp", unit="s", timezone="UTC")
    assert {name: column.role for name, column in purchases.columns.items()} == {
        "purchase_id": "primary key",
        "user_id": "join key",
        "at": "time column",
        "amount": "read",
        "category": "read",
    }
    assert purchases.columns["amount"].read_by == ("recent.amount",)
    assert users.columns["segment"].read_by == ("user.segment",)


def test_rows_spec_limits_graph_hops_and_events() -> None:
    contract, _, _ = _graph_case("three_hop", "distinct")
    spec = ServingRowsSpec.from_contract(contract)
    shops = spec.tables["shops"]
    assert shops.hop == 2
    assert shops.join.from_table == "events" and shops.join.from_hop == 1
    assert shops.limit == RowLimit(2, "newest", "opened", "events row", "neighborhoods.g hop 2")
    assert "hop 2 · shops.shop_id IN (events.shop_id from hop 1)" in str(spec)


def _user_rows(user_id: int) -> Tables:
    example = _example()
    tables = example.tables()
    return {
        "users": [row for row in tables["users"] if row["user_id"] == user_id],
        "purchases": [
            row
            for row in tables["purchases"]
            if row["user_id"] == user_id and row["at"] <= example.SCORE_AT
        ],
    }


def _request(user_id: int = 4) -> api.SampleRequest:
    example = _example()
    return api.SampleRequest(f"user-{user_id}", user_id, example.SCORE_AT)


def _refused(server: api.ServingAssembler, rows: Mapping[str, object], pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern) as error:
        server.assemble(rows, [_request()])  # type: ignore[arg-type]
    assert "ServingAssembler.rows_spec" in str(error.value)


def test_serving_refuses_rows_that_do_not_fit_the_contract(
    example_server: api.ServingAssembler,
) -> None:
    rows = _user_rows(4)
    purchases = rows["purchases"]
    _refused(example_server, {**rows, "purchase": []}, r"\['purchase'\].*did you mean 'purchases'")
    _refused(example_server, {"users": rows["users"]}, r"required tables \['purchases'\]")
    _refused(
        example_server,
        {**rows, "purchases": [{**purchases[0], "amout": 1.0}, *purchases[1:]]},
        r"rows\['purchases'\]\[0\] has column 'amout'.*did you mean 'amount'",
    )
    _refused(
        example_server,
        {**rows, "purchases": [{k: v for k, v in purchases[0].items() if k != "at"}]},
        r"rows\['purchases'\]\[0\] lacks required columns \['at'\]",
    )
    _refused(
        example_server,
        {**rows, "purchases": [*purchases[:1], {**purchases[1], "at": "yesterday"}]},
        r"rows\['purchases'\]\[1\]\['at'\]: 'yesterday' is not a valid timestamp\[s, UTC\]",
    )
    _refused(
        example_server,
        {**rows, "purchases": [{**purchases[0], "purchase_id": None}]},
        r"rows\['purchases'\]\[0\] has a NULL primary key 'purchase_id'",
    )
    _refused(
        example_server,
        {**rows, "purchases": [purchases[0], purchases[0]]},
        r"rows\['purchases'\]\[1\] repeats primary key purchase_id=.* of rows\['purchases'\]\[0\]",
    )
    _refused(
        example_server,
        {**rows, "users": []},
        r"'user-4': entity_id 4 has no row in rows\['users'\]",
    )
    arrow = pa.Table.from_pylist([{**row, "amount": str(row["amount"])} for row in purchases])
    _refused(
        example_server,
        {**rows, "purchases": arrow},
        r"rows\['purchases'\] column 'amount' is Arrow string, not the declared float64",
    )
    with pytest.raises(ValueError, match="at least one request"):
        example_server.assemble(rows, [])


def test_resolve_never_reads_supplied_rows(example_server: api.ServingAssembler) -> None:
    # artifact contracts pin every column type, so a request whose column is all
    # None neither resolves differently nor is refused
    contract, tables, requests = _example_case()
    artifact = _artifact(contract, tables, requests)
    assert artifact.contract.resolve() is artifact.contract
    blank = {
        **tables,
        "users": [{**row, "segment": None} for row in tables["users"]],
    }
    request = _request(4)
    _, expected = offline_batches(contract, blank, [request], artifact=artifact)
    rows = _user_rows(4)
    rows["users"] = [{**row, "segment": None} for row in rows["users"]]
    arrow = {name: pa.Table.from_pylist(values) for name, values in rows.items()}
    assert pa.types.is_null(arrow["users"].schema.field("segment").type)
    encoded = {
        **arrow,
        "users": pa.table(
            {
                "user_id": arrow["users"]["user_id"],
                "segment": pa.array(["loyal"]).dictionary_encode(),
            }
        ),
    }
    assert_same_batches(
        example_server.assemble(encoded, [request]),
        example_server.assemble(
            {**rows, "users": [{**rows["users"][0], "segment": "loyal"}]}, [request]
        ),
    )
    for supplied in (rows, arrow):
        (served,) = example_server.assemble(supplied, [request])
        assert_offline_batch(served, expected[request.sample_id], artifact, blank, request)


def test_the_plan_is_built_once_and_rows_are_never_fingerprinted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, tables, requests = _example_case()
    artifact = _artifact(contract, tables, requests)
    built = []
    original = serving.fit_plan

    def counting(*args, **kwargs):
        built.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(serving, "fit_plan", counting)
    server = api.ServingAssembler(artifact)

    def refuse(value: object) -> str:
        raise AssertionError("serving fingerprinted per call")

    monkeypatch.setattr(preparation, "_fingerprint", refuse)
    for user_id in (1, 2, 3, 4, 5):
        server.assemble(_user_rows(user_id), [_request(user_id)])
    assert built == [1]


def _layout(value: object, path: str = "inputs") -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Every tensor's path, shape and dtype."""

    if isinstance(value, Mapping):
        layout = {}
        for key, inner in value.items():
            layout.update(_layout(inner, f"{path}.{key}"))
        return layout
    assert isinstance(value, torch.Tensor), path
    return {path: (tuple(value.shape), value.dtype)}


def test_a_model_trained_on_model_format_batches_takes_served_batches(tmp_path: Path) -> None:
    pytest.importorskip("diskgraph")
    pq = pytest.importorskip("pyarrow.parquet")
    example = _example()
    tables = example.tables()
    (tmp_path / "parquet").mkdir()
    sources = {}
    for name, rows in tables.items():
        table = pa.Table.from_pylist(rows)
        if "at" in table.column_names:
            at = table["at"].cast(pa.timestamp("s", tz="UTC"))
            table = table.set_column(table.schema.get_field_index("at"), "at", at)
        pq.write_table(table, tmp_path / "parquet" / f"{name}.parquet")
        sources[name] = api.Parquet(str(tmp_path / "parquet" / f"{name}.parquet"))
    dataset = api.Dataset(sources)
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1, num_shards=1)

    torch.manual_seed(0)
    with api.prepare(example.contract(), dataset, engine) as plan:
        anchors = example.anchors(example.TRAIN_AT, split="train")
        with plan.materialize(anchors, fit_split="train") as task:
            model = example.ActivityModel(task.output_schema)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
            for batch in task.batches(split="train", output_format="model"):
                labels = (batch.targets["active"]["values"] > 0).float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            task.to_artifact().save(tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    served_model = example.ActivityModel(artifact.output_schema)
    served_model.load_state_dict(model.state_dict())
    served_model.eval()

    server = api.ServingAssembler(artifact)
    requests = [api.SampleRequest(f"user-{u}", u, example.SCORE_AT) for u in (1, 2, 3)]
    with api.prepare(artifact.contract, dataset, engine) as plan:
        with plan.materialize(requests, fitted=artifact.fitted_state, inference=True) as task:
            model_format = {
                request.sample_id: next(iter(task.batches([request], output_format="model")))
                for request in requests
            }
    for request in requests:
        rows = fetch(server.rows_spec, artifact.contract.schema, tables, request)
        (served,) = server.assemble(rows, [request])
        assert _layout(served.inputs) == _layout(model_format[request.sample_id].inputs)
        with torch.no_grad():
            assert served_model(served).shape == (1,)


@pytest.fixture(params=["bounded_window", "event_limit"])
def native_serving_case(tmp_path, request, native_anchor_delivery):
    from conformance.assertions import bound_tables
    from tests.helpers import parquet_anchors

    name = request.param
    contract, tables, requests = CASES[name]()
    by_id = {r.sample_id: r for r in requests}
    anchors = parquet_anchors(
        tmp_path / "anchors.parquet",
        {
            "sample": list(by_id),
            "entity": [r.entity_id for r in requests],
            "at": [r.as_of_time for r in requests],
        },
        "entity",
        "at",
        sample_id="sample",
    )
    with (
        bound_tables(contract, tables, "diskgraph") as (dataset, engine),
        api.prepare(contract, dataset, engine) as plan,
    ):
        with plan.materialize(anchors) as task:
            artifact = task.to_artifact()
        with plan.materialize(anchors, fitted=artifact.fitted_state, inference=True) as task:
            batches = list(native_anchor_delivery(task, batch_size=1, seed=SEED))
    assert sorted(batch.metadata["sample_ids"][0] for batch in batches) == sorted(by_id)
    return name, artifact, tables, by_id, batches


def test_serving_fetch_covers_native_parquet_anchors(native_serving_case):
    name, artifact, tables, by_id, batches = native_serving_case
    spec = ServingRowsSpec.from_contract(artifact.contract)
    for batch in batches:
        (sample,) = batch.metadata["sample_ids"]
        rows = fetch(spec, artifact.contract.schema, tables, by_id[sample])
        if name == "bounded_window":
            assert batch.inputs["entity"]["recent_count"]["values"].item() == len(rows["events"])
        else:
            merged = batch.inputs["merged"]
            valid = merged["id"]["valid"]
            delivered = zip(
                merged["component"]["values"][valid].tolist(),
                merged["id"]["values"][valid].tolist(),
                strict=True,
            )
            assert sorted(delivered) == sorted(
                [(0, r["order_id"]) for r in rows["orders"]]
                + [(1, r["click_id"]) for r in rows["clicks"]]
            )


def test_serving_assembly_matches_native_parquet_anchors(native_serving_case, monkeypatch):
    _, artifact, tables, by_id, expected = native_serving_case

    def fallback(*args, **kwargs):
        pytest.fail("ServingAssembler.assemble tried to open InMemorySource")

    # assemble() binds its source directly, bypassing the engine's open() guard.
    monkeypatch.setattr(serving, "InMemorySource", fallback, raising=False)

    spec = ServingRowsSpec.from_contract(artifact.contract)
    calls = []
    for batch in expected:
        (sample,) = batch.metadata["sample_ids"]
        request = by_id[sample]
        calls.append((fetch(spec, artifact.contract.schema, tables, request), request))

    def outputs():
        server = api.ServingAssembler(artifact)
        for rows, request in calls:
            batches = iter(server.assemble(rows, [request], seed=SEED))
            yield next(batches)  # Empty delivery for a requested anchor must fail.
            yield from batches

    results = outputs()
    try:
        first = next(results)
    except (ValueError, NotImplementedError) as exc:
        refusal = re.compile(
            r"(?:native|diskgraph) serving(?: assembly)? (?:is )?"
            r"(?:unsupported|not supported|not implemented)[.!]?",
            re.IGNORECASE,
        )
        messages = (
            [i.message for i in exc.issues] if isinstance(exc, api.ValidationError) else [str(exc)]
        )
        if not messages or not all(refusal.fullmatch(message) for message in messages):
            raise
        return
    assert_same_batches([first, *results], expected)
