# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native serving, its row requirements and its agreement with offline inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Lock
from types import ModuleType

import pyarrow as pa
import pytest
import torch

from conformance.assertions import bound_tables
from conformance.cases import discover_cases
from conformance.graph_cases import generate_graph_case
from conformance.sequence_cases import generate_sequence_case
from kumo.pytorch import api
from kumo.pytorch.assembly import _fingerprint, fit_plan
from kumo.pytorch.preparation import reused_state
from kumo.pytorch.rows import Clock, RowSource
from kumo.pytorch.serving import RowLimit, RowsJoin, ServingRowsSpec
from kumo.pytorch.task import request_batches
from tests import lifecycle_app
from tests.helpers import serving_anchors
from tests.serving_fetch import fetch
from tests.test_diskgraph_backend import _fixture_rows, _requests, _schema, _temporal_contract
from tests.test_inference_lifecycle import _dataset, assert_same_batches

ROOT = Path(__file__).parents[1]
Row = dict[str, object]
Tables = dict[str, list[Row]]
SEED = 7


def _example() -> ModuleType:
    return lifecycle_app


def _artifact(
    contract: api.DataLoaderSpec, tables: Tables, requests: Sequence[api.SampleRequest]
) -> api.Artifact:
    dataset = api.Dataset({name: api.InMemoryRows(rows) for name, rows in tables.items()})
    with api.prepare(contract, dataset, api.InMemoryEngine()) as plan:
        anchors = [replace(request, split=None) for request in requests]
        with plan.materialize(anchors) as task:
            return task.to_artifact()


def _batch_inference(
    artifact: api.Artifact,
    dataset: api.Dataset,
    engine: api.Engine,
    requests: Sequence[api.SampleRequest],
) -> dict[str, api.Batch]:
    with api.prepare(artifact.contract, dataset, engine) as plan:
        with plan.materialize(requests, fitted=artifact.fitted_state, inference=True) as task:
            return {
                batch.metadata["sample_ids"][0]: batch
                for batch in task.batches(batch_size=1, seed=SEED)
            }


def _reference_batch(artifact: api.Artifact, rows: Tables, request: api.SampleRequest) -> api.Batch:
    """Evaluate the fetched rows directly with the artifact's frozen encoders."""
    contract = artifact.contract
    schema = contract.schema
    tables = {
        t.name: [{column: row.get(column) for column in t.columns} for row in rows.get(t.name, [])]
        for t in schema.tables
    }
    source = RowSource(schema, tables, Clock.of(schema))
    inputs, targets = reused_state(contract, artifact.fitted_state, None)
    plan = fit_plan(
        contract.sample,
        contract.batch,
        (),
        source,
        schema_fingerprint=_fingerprint(schema),
        contract_fingerprint=contract.fingerprint,
        contract=contract,
        input_state=inputs,
        target_encoders=targets,
        inference=True,
    )
    return next(
        request_batches(
            plan,
            [request],
            batch_size=1,
            epoch=0,
            seed=SEED,
            shuffle=False,
            rank=0,
            world_size=1,
            drop_last=False,
        )
    )


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
        "contract.graph_temporal": "inherit",
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
    artifact = _artifact(contract, tables, requests)
    dataset, engine = _dataset("in_memory", tables, Path())
    expected = _batch_inference(artifact, dataset, engine, requests)
    spec = ServingRowsSpec.from_contract(artifact.contract)
    for request in requests:
        rows = fetch(spec, artifact.contract.schema, tables, request)
        assert_same_batches(
            [_reference_batch(artifact, rows, request)], [expected[request.sample_id]]
        )


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
    dataset, engine = _dataset("in_memory", rows, tmp_path)
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            artifact = task.to_artifact()
    expected = _batch_inference(artifact, dataset, engine, _requests())
    spec = ServingRowsSpec.from_contract(artifact.contract)
    assert spec.tables["card_transactions"].required
    assert len(spec.tables["card_transactions"].routes) == 2
    assert not spec.tables["merchants"].required
    assert not spec.tables["card_transactions"].columns["description"].required
    for request in _requests():
        fetched = fetch(spec, artifact.contract.schema, rows, request)
        assert "merchants" not in fetched
        assert all("description" not in row for row in fetched["card_transactions"])
        assert_same_batches(
            [_reference_batch(artifact, fetched, request)], [expected[request.sample_id]]
        )


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
def example_artifact() -> api.Artifact:
    contract, tables, requests = _example_case()
    with bound_tables(contract, tables, "diskgraph") as (dataset, engine):
        anchors = serving_anchors(engine.index.parent / "example.parquet", requests)
        with api.prepare(contract, dataset, engine) as plan, plan.materialize(anchors) as task:
            return task.to_artifact()


@pytest.fixture(scope="module")
def example_server(example_artifact) -> api.ServingAssembler:
    return api.ServingAssembler(example_artifact)


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


@pytest.mark.parametrize("arrow", [False, True], ids=["row-mappings", "arrow-tables"])
def test_supplied_rows_require_native_serving(arrow, example_server):
    contract, tables, requests = _example_case()
    rows = fetch(example_server.rows_spec, contract.schema, tables, requests[0])
    if arrow:
        rows = {name: pa.Table.from_pylist(values) for name, values in rows.items()}
    with pytest.raises(NotImplementedError, match="native serving assembly is unsupported"):
        example_server.assemble(rows=rows, requests=requests[:1])


def test_anchors_require_a_prepared_diskgraph_plan(example_server, tmp_path):
    _, _, requests = _example_case()
    anchors = serving_anchors(tmp_path / "anchors.parquet", requests)
    with pytest.raises(ValueError, match="prepared DiskGraph plan"):
        example_server.assemble(anchors)
    with pytest.raises(ValueError, match="Parquet AnchorTable"):
        example_server.assemble(api.AnchorTable(api.InMemoryRows([]), "entity", "at"))


def test_serving_plan_must_match_the_artifact(example_artifact):
    contract, tables, _ = _bounded_rows_contract()
    with bound_tables(contract, tables, "diskgraph") as (dataset, engine):
        with api.prepare(contract, dataset, engine) as plan:
            with pytest.raises(ValueError, match="different contracts"):
                api.ServingAssembler(example_artifact, plan=plan)


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

    requests = [api.SampleRequest(f"user-{u}", u, example.SCORE_AT) for u in (1, 2, 3)]
    anchors = serving_anchors(tmp_path / "score.parquet", requests)
    with api.prepare(artifact.contract, dataset, engine, inference=True) as plan:
        with plan.materialize(anchors, fitted=artifact.fitted_state) as task:
            expected = list(task.batches(batch_size=1, output_format="model"))
        server = api.ServingAssembler(artifact, plan=plan)
        for batch, reference in zip(server.assemble(anchors, batch_size=1), expected, strict=True):
            assert _layout(batch.inputs) == _layout(reference.inputs)
            with torch.no_grad():
                assert served_model(batch).shape == (1,)


@contextmanager
def _diskgraph_inference(contract, tables, requests):
    with bound_tables(contract, tables, "diskgraph") as (dataset, engine):
        anchors = serving_anchors(engine.index.parent / "anchors.parquet", requests)
        with api.prepare(contract, dataset, engine) as train:
            with train.materialize(anchors) as task:
                artifact = task.to_artifact()
        with api.prepare(artifact.contract, dataset, engine, inference=True) as plan:
            with plan.materialize(anchors, fitted=artifact.fitted_state) as task:
                expected = list(task.batches(batch_size=1, seed=SEED, output_format="model"))
            yield artifact, plan, anchors, expected


@pytest.mark.parametrize("name", ["example", "bounded_window", "event_limit"])
def test_serving_equals_diskgraph_batch_inference(name: str, monkeypatch) -> None:
    from kumo.pytorch.assembly import PreparedBatchPlan
    from kumo.pytorch.engines.native_anchors import NativeAnchors

    def fallback(*args, **kwargs):
        pytest.fail("expected native DiskGraph delivery from Parquet anchors")

    monkeypatch.setattr(api.InMemoryEngine, "open", fallback)
    contract, tables, requests = CASES[name]()
    with _diskgraph_inference(contract, tables, requests) as (artifact, plan, anchors, expected):
        server = api.ServingAssembler(artifact, plan=plan)
        # Serving must reuse the caller's open index and emit native iterator batches.
        monkeypatch.setattr(api.DiskGraphEngine, "open", fallback)
        monkeypatch.setattr(PreparedBatchPlan, "batches", fallback)
        monkeypatch.setattr(NativeAnchors, "_requests", fallback)
        native_batches = NativeAnchors._typed_batches
        served_ids = []

        def tracked(*args, **kwargs):
            for batch in native_batches(*args, **kwargs):
                served_ids.extend(batch.metadata["sample_ids"])
                yield batch

        monkeypatch.setattr(NativeAnchors, "_typed_batches", tracked)
        assert_same_batches(server.assemble(anchors, batch_size=1, seed=SEED), expected)
        assert served_ids == [r.sample_id for r in requests]


def test_concurrent_servers_share_a_plan(tmp_path, monkeypatch):
    contract, tables, requests = _bounded_rows_contract()
    with _diskgraph_inference(contract, tables, requests) as (artifact, plan, _, expected):
        start = Barrier(2)
        active = Lock()
        materialize = plan.materialize

        @contextmanager
        def one_task_at_a_time(*args, **kwargs):
            assert active.acquire(blocking=False), "native tasks overlapped on a shared plan"
            try:
                with materialize(*args, **kwargs) as task:
                    yield task
            finally:
                active.release()

        monkeypatch.setattr(plan, "materialize", one_task_at_a_time)

        def serve(i):
            server = api.ServingAssembler(artifact, plan=plan)
            anchors = serving_anchors(tmp_path / f"{i}.parquet", requests[i::2])
            start.wait(timeout=10)
            return server.assemble(anchors, batch_size=1, seed=SEED)

        with ThreadPoolExecutor(max_workers=2) as pool:
            for i, batches in enumerate(pool.map(serve, range(2))):
                assert_same_batches(batches, expected[i::2])


def test_serving_refuses_mixed_graph_and_token_budget_before_python_batching(monkeypatch):
    from kumo.pytorch.assembly import PreparedBatchPlan

    contract, tables, requests = _limited_events_case("distinct")
    layer = api.GraphLayer((api.GraphSample("user_clicks", 1),))
    sample = replace(contract.sample, neighborhoods={"g": api.NeighborhoodSpec((layer,))})
    graph = api.Input(api.NeighborhoodSource("g"), api.Graph(), None)
    batch = replace(
        contract.batch,
        inputs={**contract.batch.inputs, "graph": graph},
        batching=api.Batching(max_samples=2, max_tokens=64, length_input="merged"),
    )
    with bound_tables(contract, tables, "diskgraph") as (dataset, engine):
        anchors = serving_anchors(engine.index.parent / "anchors.parquet", requests)
        with api.prepare(replace(contract, sample=sample, batch=batch), dataset, engine) as plan:
            with plan.materialize(anchors) as task:
                server = api.ServingAssembler(task.to_artifact(), plan=plan)
            monkeypatch.setattr(
                PreparedBatchPlan, "batches", lambda *a, **kw: pytest.fail("fallback")
            )
            with pytest.raises(NotImplementedError, match="native serving assembly is unsupported"):
                server.assemble(anchors)


def _clock_case(unit: str) -> tuple[api.DataLoaderSpec, Tables, list[api.SampleRequest]]:
    """Events a few ticks apart: a one-second window, a recency and event ages."""

    clock, key = api.DType("timestamp", unit=unit, timezone="UTC"), api.DType("int64")
    schema = api.DataSchema(
        tables=(
            api.TableSpec("users", "user_id", {"user_id": key}, time_column=api.NON_TEMPORAL),
            api.TableSpec(
                "events",
                "event_id",
                {"event_id": key, "user_id": key, "at": clock, "amount": api.DType("float64")},
                time_column="at",
            ),
        ),
        relations=(api.RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
    )
    history = api.Rows(("user_events",), api.TimeWindow("at"))
    sample = api.SampleSpec(
        entity=api.EntitySpec("users"),
        entity_values={
            "last_second": api.Aggregate(
                api.Rows(("user_events",), api.TimeWindow("at", -1, 0)), "count", dtype=key
            ),
            "age": api.Aggregate(history, "recency", "at", dtype=api.DType("float64")),
        },
        events={
            "events": api.EventSpec(
                history, {"amount": api.Field("amount"), "age": api.EventTimeDelta()}
            )
        },
    )
    batch = api.BatchSpec(
        inputs={
            "user": api.Input(
                api.EntitySource(),
                api.Fields((api.Encode("last_second", api.Raw()), api.Encode("age", api.Raw()))),
                api.Separate(),
            ),
            "events": api.Input(
                api.EventSource(("events",)),
                api.Fields(
                    (api.Encode("amount", api.Raw()), api.Encode("age", api.Raw())),
                    row=api.Columns(),
                ),
                api.Padded(),
            ),
        }
    )
    per_second = {"us": 10**6, "ns": 10**9}[unit]
    base = 1_700_000_000 * per_second
    events = [(1, 1, 7), (2, 1, 5), (3, 1, 2), (4, 1, 9), (5, 1, 2 - per_second), (6, 2, 3)]
    tables: Tables = {
        "users": [{"user_id": 1}, {"user_id": 2}],
        "events": [
            {"event_id": e, "user_id": u, "at": base + t, "amount": float(e)} for e, u, t in events
        ],
    }
    at = pa.scalar(base + 8, type=pa.timestamp(unit, tz="UTC"))
    requests = [api.SampleRequest("u1", 1, at), api.SampleRequest("u2", 2, at)]
    return api.DataLoaderSpec(schema, sample, batch), tables, requests


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_sub_second_clocks_are_served_as_diskgraph_batch_inference(unit: str) -> None:
    contract, tables, requests = _clock_case(unit)
    with _diskgraph_inference(contract, tables, requests) as (artifact, plan, anchors, expected):
        server = api.ServingAssembler(artifact, plan=plan)
        batches = server.assemble(anchors, batch_size=1, seed=SEED)
        assert_same_batches(batches, expected)
        served = batches[0]
    # Ticks 2 to 7 are within the second before the cutoff at tick 8; tick 9 is after it.
    assert served.inputs["user"]["last_second"]["values"].tolist() == [3]
    tick = {"us": 1e-6, "ns": 1e-9}[unit]
    assert served.inputs["user"]["age"]["values"].tolist() == pytest.approx([tick])
