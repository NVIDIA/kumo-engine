# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each entry of `DiskGraphEngine.unsupported` is refused, with its code, by a minimal contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from conformance.assertions import write_parquet
from kumo.pytorch.api import (
    NON_TEMPORAL,
    ActiveInterval,
    Aggregate,
    AnchorTable,
    Batching,
    BatchSpec,
    CandidateColumn,
    CandidateSetSpec,
    Columns,
    DataLoaderSpec,
    DataSchema,
    Dataset,
    DatedEntityValue,
    DiskGraphEngine,
    DType,
    EntitySource,
    EntitySpec,
    EventSpec,
    Field,
    Graph,
    GraphLayer,
    GraphSample,
    InMemoryRows,
    Input,
    NeighborhoodSource,
    NeighborhoodSpec,
    Parquet,
    Raw,
    RelationSpec,
    RowFilter,
    Rows,
    RowTarget,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    TimeWindow,
    check,
    prepare,
)
from kumo.pytorch.contract.batch import Encode, Fields
from kumo.pytorch.engines.unsupported import DISKGRAPH_UNSUPPORTED
from kumo.pytorch.materialization import _compile_diskgraph_config

STAMP = DType("timestamp", unit="s", timezone="UTC")
USERS = TableSpec(
    "users",
    "user_id",
    {"user_id": DType("int64"), "group_id": DType("int64"), "region": DType("string")},
    time_column=NON_TEMPORAL,
)
ACCOUNTS = TableSpec(
    "accounts",
    "account_id",
    {"account_id": DType("int64"), "user_id": DType("int64"), "opened_at": STAMP},
    time_column="opened_at",
)
EVENTS = TableSpec(
    "events",
    "event_id",
    {
        "event_id": DType("int64"),
        "user_id": DType("int64"),
        "account_id": DType("int64"),
        "group_id": DType("int64"),
        "occurred_at": STAMP,
        "synced_at": STAMP,
        "ends_at": STAMP,
        "amount": DType("float64"),
        "price": DType("decimal128", precision=10, scale=2),
    },
    time_column="occurred_at",
)
RELATIONS = (
    RelationSpec("user_events", "users", "user_id", "events", "user_id"),
    RelationSpec("user_accounts", "users", "user_id", "accounts", "user_id"),
    RelationSpec("account_events", "accounts", "account_id", "events", "account_id"),
)
SCHEMA = DataSchema((USERS, ACCOUNTS, EVENTS), RELATIONS)
WINDOW = TimeWindow("occurred_at", -100, 0)
HISTORY = Rows(("user_events",), WINDOW)
BASE = DataLoaderSpec(
    schema=SCHEMA,
    sample=SampleSpec(
        entity=EntitySpec("users", entity_type="user"),
        entity_values={"n": Aggregate(HISTORY, "count", dtype=DType("int64"))},
    ),
    batch=BatchSpec(
        inputs={
            "values": Input(
                EntitySource(), Fields(fields=(Encode("n", Raw()),), row=Columns()), Separate()
            )
        },
        batching=Batching(max_samples=2),
    ),
)
PARQUET = Dataset({name: Parquet(f"/warehouse/{name}") for name in ("users", "accounts", "events")})


def _sample(**changes: object) -> DataLoaderSpec:
    return replace(BASE, sample=replace(BASE.sample, **changes))


def _value(name: str, value: object) -> DataLoaderSpec:
    return _sample(entity_values={**BASE.sample.entity_values, name: value})


def _graph(traversal: GraphSample) -> DataLoaderSpec:
    return _sample(neighborhoods={"g": NeighborhoodSpec((GraphLayer((traversal,)),))})


#: One minimal contract (and dataset) per entry that check() refuses.
CHECK_CASES: dict[str, Callable[[], tuple[DataLoaderSpec, Dataset]]] = {
    "DG-SOURCE-KIND": lambda: (
        BASE,
        Dataset({name: InMemoryRows([]) for name in ("users", "accounts", "events")}),
    ),
    "DG-RELATION-KEYS": lambda: (
        replace(
            BASE,
            schema=DataSchema(
                SCHEMA.tables,
                (*RELATIONS, RelationSpec("by_group", "users", "group_id", "events", "group_id")),
            ),
        ),
        PARQUET,
    ),
    "DG-TIMED-INTERMEDIATE": lambda: (
        _value(
            "through_accounts",
            Aggregate(
                Rows(("user_accounts", "account_events"), WINDOW), "count", dtype=DType("int64")
            ),
        ),
        PARQUET,
    ),
    "DG-ROW-FILTER": lambda: (
        _value(
            "large",
            Aggregate(
                Rows(("user_events",), WINDOW, filters=(RowFilter("amount", "greater_than", 5),)),
                "count",
                dtype=DType("int64"),
            ),
        ),
        PARQUET,
    ),
    "DG-ACTIVE-INTERVAL": lambda: (
        _value(
            "open",
            Aggregate(
                Rows(("user_events",), active=ActiveInterval("occurred_at", "ends_at")),
                "count",
                dtype=DType("int64"),
            ),
        ),
        PARQUET,
    ),
    "DG-ENTITY-VALUE-KIND": lambda: (
        _value("account_opened", Field("opened_at", path=("user_accounts",))),
        PARQUET,
    ),
    "DG-JOINED-FIELD": lambda: (
        _sample(
            events={"seen": EventSpec(HISTORY, {"owner": Field("region", path=("user_events",))})}
        ),
        PARQUET,
    ),
    "DG-DATED-ENTITY-VALUES": lambda: (
        _sample(dated_entity_values={"seen": DatedEntityValue(HISTORY, Field("amount"))}),
        PARQUET,
    ),
    "DG-CURRENT-EVENTS": lambda: (
        _sample(
            current_events={
                "now": EventSpec(
                    Rows(("user_events",), TimeWindow("occurred_at", -1, 0)),
                    {"amount": Field("amount")},
                    limit=1,
                )
            }
        ),
        PARQUET,
    ),
    "DG-CANDIDATE-SETS": lambda: (
        _sample(candidate_sets={"accounts": CandidateSetSpec(EntitySpec("accounts"))}),
        PARQUET,
    ),
    "DG-REMOTE-PARQUET": lambda: (
        BASE,
        Dataset({**PARQUET.tables, "accounts": Parquet("s3://bucket/accounts")}),
    ),
    "DG-UNIFORM-GRAPH": lambda: (
        _graph(GraphSample("user_events", 2, strategy="uniform")),
        PARQUET,
    ),
    "DG-GRAPH-CLOCK-OVERRIDE": lambda: (
        _graph(GraphSample("user_events", 2, strategy="recent", time_column="synced_at")),
        PARQUET,
    ),
    "DG-STATIC-TIMED-TRAVERSAL": lambda: (
        _graph(GraphSample("user_events", 2, strategy="recent", temporal="static")),
        PARQUET,
    ),
    "DG-REDUCTION-OPERAND": lambda: (
        _value(
            "top_price",
            Aggregate(HISTORY, "max", "price", dtype=DType("decimal128", precision=10, scale=2)),
        ),
        PARQUET,
    ),
}


ROWS: dict[str, list[dict[str, object]]] = {
    "users": [{"user_id": 1, "group_id": 1, "region": "EU"}],
    "accounts": [{"account_id": 1, "user_id": 1, "opened_at": 0}],
    "events": [
        {
            "event_id": index,
            "user_id": 1,
            "account_id": 1,
            "group_id": 1,
            "occurred_at": index,
            "synced_at": index,
            "ends_at": index,
            "amount": 1.0,
            "price": None,
        }
        for index in (1, 2)
    ],
}


def _written(tmp_path: Path, types: dict[str, dict[str, str]] | None = None) -> Dataset:
    """Tiny Parquet files for every table, with optional physical-type overrides."""
    for table in SCHEMA.tables:
        write_parquet(
            table,
            ROWS[table.name],
            tmp_path / f"{table.name}.parquet",
            types=(types or {}).get(table.name),
        )
    return Dataset({t.name: Parquet(str(tmp_path / f"{t.name}.parquet")) for t in SCHEMA.tables})


def _int64_clock(tmp_path: Path) -> None:
    dataset = _written(tmp_path, {"accounts": {"opened_at": "int64"}})
    _compile_diskgraph_config(SCHEMA, dataset, num_shards=1, dataset_name="t")


def _nested(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset = _written(tmp_path)
    path = dataset.tables["accounts"].path
    data = pq.read_table(path)
    index = data.column_names.index("user_id")
    nested = pa.array([[value] for value in data.column("user_id").to_pylist()])
    pq.write_table(data.set_column(index, "user_id", nested), path)
    _compile_diskgraph_config(SCHEMA, dataset, num_shards=1, dataset_name="t")


def _renamed(tmp_path: Path) -> None:
    dataset = Dataset(
        {**PARQUET.tables, "accounts": Parquet("/warehouse/accounts", {"opened": "opened_at"})}
    )
    _compile_diskgraph_config(SCHEMA, dataset, num_shards=1, dataset_name="t")


#: Further variants of entries whose refusal covers several declarations.
VARIANT_CASES: tuple[tuple[str, str, Callable[[], DataLoaderSpec]], ...] = (
    (
        "DG-ROW-FILTER",
        "entity filter",
        lambda: _sample(
            entity=EntitySpec(
                "users", entity_type="user", filters=(RowFilter("region", "equal", "EU"),)
            )
        ),
    ),
    (
        "DG-ROW-FILTER",
        "event rows",
        lambda: _sample(
            events={
                "big": EventSpec(
                    Rows(("user_events",), WINDOW, filters=(RowFilter("amount", "not_null"),)),
                    {"amount": Field("amount")},
                )
            }
        ),
    ),
    (
        "DG-ENTITY-VALUE-KIND",
        "own time column",
        lambda: _value("dated", Field("region", time_column="region")),
    ),
    (
        "DG-JOINED-FIELD",
        "row-target field",
        lambda: _sample(
            targets={
                "next": RowTarget(
                    Rows(("user_events",), TimeWindow("occurred_at", 0, 100)),
                    {"owner": Field("region", path=("user_events",))},
                )
            }
        ),
    ),
)


@pytest.mark.parametrize(
    ("code", "build"),
    [(code, build) for code, _, build in VARIANT_CASES],
    ids=[f"{code}-{label}" for code, label, _ in VARIANT_CASES],
)
def test_check_refuses_each_variant_with_the_entry_code(code: str, build) -> None:
    issues = DiskGraphEngine(index="/i").check_contract(build().resolve())
    assert any(issue.code == code for issue in issues), issues
    assert all(issue.code in DiskGraphEngine.unsupported for issue in issues), issues


#: Entries refused while prepare() binds the data, each with the call that refuses it.
PREPARE_CASES: dict[str, Callable[[Path], None]] = {
    "DG-COLUMN-RENAME": _renamed,
    "DG-INT64-CLOCK": _int64_clock,
    "DG-NESTED-COLUMN": _nested,
}


def _duplicate_layer_relation(tmp_path: Path) -> None:
    pytest.importorskip("diskgraph")
    contract = _sample(
        neighborhoods={
            "g": NeighborhoodSpec(
                (
                    GraphLayer(
                        (
                            GraphSample("user_events", 1, strategy="recent"),
                            GraphSample("user_events", 2, strategy="recent"),
                        )
                    ),
                )
            )
        }
    )
    graph = Input(NeighborhoodSource("g"), Graph(), None)
    contract = replace(
        contract, batch=replace(contract.batch, inputs={**contract.batch.inputs, "g": graph})
    )
    dataset = _written(tmp_path)
    engine = DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    with prepare(contract, dataset, engine) as plan:
        task = plan.materialize([SampleRequest("a", 1, 100)])
        next(iter(task.batches(batch_size=1)))


#: Entries refused when batches are requested, each with the call that refuses it.
BATCHES_CASES: dict[str, Callable[[Path], None]] = {
    "DG-DUPLICATE-LAYER-RELATION": _duplicate_layer_relation,
}

#: Variants refused at check() through an anchor table.
ANCHOR_CASES: dict[str, AnchorTable] = {
    "DG-CANDIDATE-SETS": AnchorTable(
        Parquet("/warehouse/anchors"),
        "user_id",
        "at",
        candidates={"accounts": CandidateColumn(column="accounts", entity_id="account_id")},
    ),
    "DG-REMOTE-PARQUET": AnchorTable(Parquet("s3://bucket/anchors"), "user_id", "at"),
}


@pytest.mark.parametrize("code", sorted(ANCHOR_CASES))
def test_an_anchor_table_is_refused_at_check_with_the_entry_code(code: str) -> None:
    contract = _sample(entity=EntitySpec("users", entity_type="user"))
    issues = check(contract, PARQUET, DiskGraphEngine(index="/i"), anchors=ANCHOR_CASES[code])
    refusals = [issue for issue in issues if issue.code is not None]
    assert [issue.code for issue in refusals] == [code], issues
    assert refusals[0].suggestion == DISKGRAPH_UNSUPPORTED[code].next_step


@pytest.mark.parametrize("code", sorted(BATCHES_CASES))
def test_batches_refuse_with_the_entry_code(code: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=rf"\[{code}\]"):
        BATCHES_CASES[code](tmp_path)


def test_every_entry_has_a_minimal_refused_case_at_its_stage() -> None:
    stages = {"check()": CHECK_CASES, "prepare()": PREPARE_CASES, "batches()": BATCHES_CASES}
    assert set(CHECK_CASES) | set(PREPARE_CASES) | set(BATCHES_CASES) == set(
        DiskGraphEngine.unsupported
    )
    for code, entry in DiskGraphEngine.unsupported.items():
        assert code in stages[entry.refused_at], f"{code} is refused at {entry.refused_at}"
        assert entry.description and entry.next_step
        assert "engine" not in entry.next_step.lower(), f"{code}: a next step, not an engine"


def test_the_base_contract_is_accepted() -> None:
    assert check(BASE, PARQUET, DiskGraphEngine(index="/i")) == ()


@pytest.mark.parametrize("code", sorted(CHECK_CASES))
def test_check_refuses_with_the_entry_code(code: str) -> None:
    contract, dataset = CHECK_CASES[code]()
    contract.validate()  # a valid contract: the refusal is the engine's alone
    issues = check(contract, dataset, DiskGraphEngine(index="/i"))
    entry = DISKGRAPH_UNSUPPORTED[code]
    refusals = [issue for issue in issues if issue.code == code]
    assert refusals, issues
    assert all(issue.suggestion == entry.next_step for issue in refusals)
    assert {issue.code for issue in issues} <= {code}, issues


@pytest.mark.parametrize("code", sorted(PREPARE_CASES))
def test_prepare_refuses_with_the_entry_code(code: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=rf"\[{code}\]"):
        PREPARE_CASES[code](tmp_path)


def test_check_contract_emits_only_listed_codes() -> None:
    engine = DiskGraphEngine(index="/i")
    for build in CHECK_CASES.values():
        contract, _ = build()
        for issue in engine.check_contract(contract.resolve()):
            assert issue.code in DiskGraphEngine.unsupported, issue


def test_the_engine_source_raises_only_listed_codes() -> None:
    import inspect
    import re

    from kumo.pytorch import materialization
    from kumo.pytorch.engines import diskgraph, parquet_anchors

    source = "".join(inspect.getsource(m) for m in (diskgraph, parquet_anchors, materialization))
    used = set(re.findall(r'"(DG-[A-Z0-9-]+)"', source))
    assert used and used <= set(DiskGraphEngine.unsupported)
    assert "ValidationIssue(" not in inspect.getsource(DiskGraphEngine.check_contract)
