# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from kumo.pytorch import api
from kumo.pytorch.column_stats import table_column_encoders
from tests.test_diskgraph_backend import _fixture_rows, _requests, _schema, _write_parquet

pytest.importorskip("diskgraph")

LATEST = datetime(2024, 12, 1, tzinfo=UTC)


def _contract() -> api.DataLoaderSpec:
    history = api.Rows(("user_card_transactions",), api.TimeWindow("occurred_at", None, 0))
    sample = api.SampleSpec(
        entity=api.EntitySpec("users", entity_type="user"),
        entity_values={
            "country": api.Field(column="country"),
            "transaction_count": api.Aggregate(history, "count", dtype=api.DType("int64")),
        },
        events={
            "transactions": api.EventSpec(
                history,
                fields={
                    "amount": api.Field(column="amount"),
                    "channel": api.Field(column="channel"),
                },
                limit=5,
            )
        },
        neighborhoods={
            "activity": api.NeighborhoodSpec(
                layers=(
                    api.GraphLayer((api.GraphSample("user_accounts", 6),)),
                    api.GraphLayer((api.GraphSample("account_card_transactions", 4),)),
                )
            )
        },
        targets={"transactions_next_7d": api.ExternalTarget(dtype=api.DType("float64"))},
    )
    batch = api.BatchSpec(
        inputs={
            "entity": api.Input(
                api.EntitySource(),
                api.Fields(
                    fields=(
                        api.Encode(field="country", encoding=api.Vocabulary()),
                        api.Encode(field="transaction_count", encoding=api.ZScore()),
                    )
                ),
                api.Separate(),
            ),
            "transactions": api.Input(
                api.EventSource(("transactions",)),
                api.Fields(
                    fields=(
                        api.Encode(field="amount", encoding=api.Buckets(bins=4)),
                        api.Encode(field="channel", encoding=api.Vocabulary()),
                    )
                ),
                api.Padded(),
            ),
            "graph": api.Input(
                api.NeighborhoodSource("activity"),
                api.Graph(
                    node_fields={
                        "accounts": api.Fields(
                            fields=(api.Encode(field="account_type", encoding=api.Vocabulary()),)
                        ),
                        "card_transactions": api.Fields(
                            fields=(api.Encode(field="amount", encoding=api.ZScore()),)
                        ),
                    }
                ),
                None,
            ),
        },
        targets={"spend": api.LabelTarget("transactions_next_7d", api.Raw())},
        batching=api.Batching(max_samples=2),
    )
    return api.DataLoaderSpec(schema=_schema(), sample=sample, batch=batch)


def _setup(engine_name: str, directory: Path):
    rows = _fixture_rows()
    if engine_name == "in_memory":
        dataset = api.Dataset({name: api.InMemoryRows(values) for name, values in rows.items()})
        return dataset, api.InMemoryEngine()
    if not (directory / "parquet").exists():
        directory.mkdir(parents=True, exist_ok=True)
        _write_parquet(rows, directory / "parquet")
    dataset = api.Dataset(
        {name: api.Parquet(str(directory / "parquet" / f"{name}.parquet")) for name in rows}
    )
    return dataset, api.DiskGraphEngine(index=directory / "index", num_threads=2)


def _encoders(contract, dataset, engine, **materialize):
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train", **materialize) as task:
            return task.fitted_state


def _column(table: str, column: str, cutoff: datetime | None) -> list:
    clock = {"users": "signup_at", "accounts": "opened_at", "card_transactions": "occurred_at"}
    return [
        row[column]
        for row in _fixture_rows()[table]
        if cutoff is None or row[clock[table]].replace(tzinfo=UTC) <= cutoff
    ]


def _state(fitted: api.FittedState, name: str) -> dict:
    (key,) = (key for key in fitted["encoders"] if name in key)
    return fitted["encoders"][key]


def _statistics(fitted: api.FittedState) -> dict:
    return fitted.fit["inputs"]["statistics"]


def test_a_shared_encoder_combines_its_columns_unless_one_is_computed(tmp_path):
    contract = _contract()
    inputs = dict(contract.batch.inputs)
    inputs["transactions"] = replace(
        inputs["transactions"],
        representation=api.Fields(
            fields=(
                api.Encode(field="amount", encoding=api.UseEncoding("amounts")),
                api.Encode(field="channel", encoding=api.UseEncoding("kinds")),
            )
        ),
    )
    graph = inputs["graph"].representation
    inputs["graph"] = replace(
        inputs["graph"],
        representation=replace(
            graph,
            node_fields={
                **graph.node_fields,
                "accounts": api.Fields(
                    fields=(api.Encode(field="account_type", encoding=api.UseEncoding("kinds")),)
                ),
            },
        ),
    )
    sample = replace(
        contract.sample,
        entity_values={
            **contract.sample.entity_values,
            "spend": api.Aggregate(
                api.Rows(("user_card_transactions",), api.TimeWindow("occurred_at", None, 0)),
                "sum",
                "amount",
                dtype=api.DType("float64"),
            ),
        },
    )
    inputs["entity"] = replace(
        inputs["entity"],
        representation=api.Fields(
            fields=(api.Encode(field="spend", encoding=api.UseEncoding("amounts")),)
        ),
    )
    shared = replace(
        contract,
        sample=sample,
        batch=replace(
            contract.batch,
            inputs=inputs,
            encodings={"kinds": api.Vocabulary(), "amounts": api.ZScore()},
        ),
    )
    assert '["shared","amounts"]' not in table_column_encoders(shared)
    fitted = _encoders(shared, *_setup("in_memory", tmp_path))
    values = _column("accounts", "account_type", LATEST) + _column(
        "card_transactions", "channel", LATEST
    )
    counts = {value: values.count(value) for value in values}
    expected = sorted(counts, key=lambda value: (-counts[value], value))
    assert [value for value, _ in _state(fitted, '"kinds"')["entries"]] == expected


# 6226f4ac fits this state, reading the graph once for every fitting anchor.
SHARED_GRAPH_STATE = "sha256:a737a0089a9acc4f75c8c6055315623ccb2b20b29a379129c4af4719602bd710"


def test_a_graph_field_sharing_an_encoder_with_an_aggregate_fits_as_before(tmp_path):
    rows = _fixture_rows()
    draw = random.Random(1)
    for row in rows["card_transactions"]:
        row["amount"] = draw.uniform(0, 1) * 10 ** draw.randint(-3, 6)
    history = api.Rows(("user_card_transactions",), api.TimeWindow("occurred_at", None, 0))
    layers = (
        api.GraphLayer(
            (api.GraphSample("user_accounts", 6), api.GraphSample("user_card_transactions", 3))
        ),
        api.GraphLayer((api.GraphSample("account_card_transactions", 4),)),
    )
    sample = api.SampleSpec(
        entity=api.EntitySpec("users", entity_type="user"),
        entity_values={
            "spend": api.Aggregate(history, "sum", "amount", dtype=api.DType("float64"))
        },
        neighborhoods={"activity": api.NeighborhoodSpec(layers=layers)},
    )
    amounts = api.UseEncoding("amounts")
    graph = api.Graph(
        node_fields={"card_transactions": api.Fields(fields=(api.Encode("amount", amounts),))}
    )
    batch = api.BatchSpec(
        inputs={
            "graph": api.Input(api.NeighborhoodSource("activity"), graph, None),
            "user": api.Input(
                api.EntitySource(),
                api.Fields(fields=(api.Encode("spend", amounts),)),
                api.Separate(),
            ),
        },
        encodings={"amounts": api.ZScore()},
        batching=api.Batching(max_samples=2),
    )
    contract = api.DataLoaderSpec(schema=_schema(), sample=sample, batch=batch)
    anchors = [
        api.SampleRequest(f"{user}-{hour}", user, datetime(2024, 1, 1) + timedelta(hours=hour))
        for hour in range(0, 35_000, 7)
        for user in (2, 3)
    ]
    engine = api.DiskGraphEngine(index=tmp_path / "index")
    with api.prepare(contract, _write_parquet(rows, tmp_path / "parquet"), engine) as plan:
        with plan.materialize(anchors) as task:
            assert task.manifest["fitted_state_fingerprint"] == SHARED_GRAPH_STATE


@pytest.mark.parametrize("engine_name", ("in_memory", "diskgraph"))
@pytest.mark.parametrize("cutoff", ("latest", "2024-11-21T16:00:00", None))
def test_pinned_encoders_match_numpy_over_rows_up_to_the_cutoff(tmp_path, engine_name, cutoff):
    dataset, engine = _setup(engine_name, tmp_path)
    options = {} if cutoff == "latest" else {"encoder_cutoff": cutoff}
    fitted = _encoders(_contract(), dataset, engine, **options)
    bound = (
        LATEST
        if cutoff == "latest"
        else None
        if cutoff is None
        else datetime.fromisoformat(cutoff).replace(tzinfo=UTC)
    )

    def vocabulary(table, column):
        values = _column(table, column, bound)
        counts = {value: values.count(value) for value in values}
        return [value for value in sorted(counts, key=lambda value: (-counts[value], value))]

    for name, table, column in (
        ('"country"', "users", "country"),
        ('"channel"', "card_transactions", "channel"),
        ('"account_type"', "accounts", "account_type"),
    ):
        assert [value for value, _ in _state(fitted, name)["entries"]] == vocabulary(table, column)
    amounts = np.asarray(_column("card_transactions", "amount", bound))
    zscore = _state(fitted, '"card_transactions","amount"')
    assert zscore["mean"] == pytest.approx(amounts.mean(), rel=1e-12)
    assert zscore["scale"] == pytest.approx(amounts.std() or 1.0, rel=1e-12)
    buckets = _state(fitted, '"transactions","fields","amount"')
    # t-digest cuts: each within one rank of the exact quantile.
    ranks = [np.mean(amounts <= cut) for cut in buckets["boundaries"]]
    assert np.allclose(ranks, [0.25, 0.5, 0.75][: len(ranks)], atol=1.5 / len(amounts))
    assert _statistics(fitted)["cutoff"] == (None if bound is None else int(bound.timestamp()))


def test_a_cutoff_after_the_latest_anchor_warns(tmp_path, caplog):
    dataset, engine = _setup("in_memory", tmp_path)
    for cutoff in ("2025-01-01", None):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="kumo.pytorch.preparation"):
            _encoders(_contract(), dataset, engine, encoder_cutoff=cutoff)
        assert "after the latest fitting anchor" in caplog.text
    with pytest.raises(ValueError, match="encoder_cutoff 'soon' is not an ISO 8601 timestamp"):
        _encoders(_contract(), dataset, engine, encoder_cutoff="soon")


COLUMN_TYPES = {
    "flag": (api.DType("bool"), [True, False, True]),
    "n": (api.DType("int32"), [3, -1, 3]),
    "u": (api.DType("uint64"), [2**64 - 1, 1, 2**64 - 1]),
    "f": (api.DType("float32"), [0.5, -0.0, 0.5]),
    "s": (api.DType("string"), ["a", "", "a"]),
    "b": (api.DType("bytes"), [b"\x00", b"\xff", b"\x00"]),
    "d": (api.DType("date32"), [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1)]),
    "t": (
        api.DType("timestamp", unit="s", timezone="UTC"),
        [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)] * 2,
    ),
    "p": (
        api.DType("decimal128", precision=10, scale=2),
        [Decimal("1.50"), Decimal("2.25"), Decimal("1.50")],
    ),
}


@pytest.mark.parametrize("engine_name", ("in_memory", "diskgraph"))
def test_vocabulary_learns_every_stored_column_type(tmp_path, engine_name):
    rows = [
        {"id": i, **{n: values[i] for n, (_, values) in COLUMN_TYPES.items()}} for i in range(3)
    ]
    columns = {"id": api.DType("int64"), **{n: dtype for n, (dtype, _) in COLUMN_TYPES.items()}}
    schema = api.DataSchema(
        tables=(api.TableSpec("items", "id", columns, time_column=api.NON_TEMPORAL),)
    )
    sample = api.SampleSpec(
        entity=api.EntitySpec("items", entity_type="item"),
        entity_values={n: api.Field(column=n) for n in COLUMN_TYPES},
    )
    fields = tuple(api.Encode(field=n, encoding=api.Vocabulary()) for n in COLUMN_TYPES)
    batch = api.BatchSpec(
        inputs={"entity": api.Input(api.EntitySource(), api.Fields(fields=fields), api.Separate())},
        batching=api.Batching(max_samples=2),
    )
    contract = api.DataLoaderSpec(schema=schema, sample=sample, batch=batch)
    if engine_name == "in_memory":
        dataset, engine = api.Dataset({"items": api.InMemoryRows(rows)}), api.InMemoryEngine()
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        types = pa.schema([(name, dtype.to_arrow()) for name, dtype in columns.items()])
        pq.write_table(pa.Table.from_pylist(rows, types), tmp_path / "items.parquet")
        dataset = api.Dataset({"items": api.Parquet(str(tmp_path / "items.parquet"))})
        engine = api.DiskGraphEngine(index=tmp_path / "index")
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize([api.SampleRequest("item-0", 0, 0)]) as task:
            for name, (_, values) in COLUMN_TYPES.items():
                counts = {value: values.count(value) for value in values}
                expected = sorted(counts, key=lambda value: (-counts[value], value))
                entries = task.output("entity").field(name=name).encoder.entries
                assert [value for value, _ in entries] == expected, name
