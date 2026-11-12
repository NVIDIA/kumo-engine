# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from kumo.pytorch import api
from kumo.pytorch.column_stats import TableColumnEncoder, fit_from_stats
from kumo.pytorch.encoding import fit_encoder
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


def _fitted_from_stats(config, values, *, truncated=False):
    import diskgraph
    import pyarrow as pa

    dtype = api.DType("float64")
    encoder = TableColumnEncoder(config, dtype, (("t", "c"),))
    computed = diskgraph.col_stats_from_arrow(
        pa.array(values, pa.float64()), None, None, list(encoder.stats)
    )
    if truncated:
        computed["counts"]["truncated"] = True
    files = {("t", "c", stat): {"data": data} for stat, data in computed.items()}
    return fit_from_stats("e", encoder, files), fit_encoder("e", config, dtype, values)


@pytest.mark.parametrize(
    "values",
    (
        [1.0, 2.0, 3.0, 4.0, 5.0] * 200,
        [1.0, 1.0, 1.0, 100.0],
        [1.0, 1.0001, 1.0002, 1.0003],
        [-3.5, -1.0, 0.0, 0.0, 2.0, 7.25],
        [4.0],
    ),
)
@pytest.mark.parametrize("separate_zero", (False, True))
def test_buckets_from_counted_values_are_numpy_quantiles(values, separate_zero):
    for bins in (3, 4, 5):
        config = api.Buckets(bins=bins, separate_zero=separate_zero)
        pinned, fitted = _fitted_from_stats(config, values)
        assert pinned.boundaries == fitted.boundaries


def test_buckets_beyond_the_category_cap_come_from_log_bins():
    rng = np.random.default_rng(0)
    values = rng.lognormal(3, 1, 10_000).tolist()
    pinned, fitted = _fitted_from_stats(api.Buckets(bins=8), values, truncated=True)
    np.testing.assert_allclose(pinned.boundaries, fitted.boundaries, rtol=2 ** (1 / 128) - 1)
    narrow, fitted = _fitted_from_stats(
        api.Buckets(bins=4), [1.0, 1.0001, 1.0002, 1.0003], truncated=True
    )
    assert len(narrow.boundaries) == len(fitted.boundaries)


def test_truncated_counts_pin_no_vocabulary():
    assert _fitted_from_stats(api.Vocabulary(), [1.0, 2.0], truncated=True)[0] is None


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
    statistics = _statistics(_encoders(shared, *_setup("in_memory", tmp_path)))
    assert sorted(statistics['["shared","kinds"]']["files"]) == [
        "accounts/account_type.counts",
        "card_transactions/channel.counts",
    ]
    assert '["shared","amounts"]' not in statistics


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
    expected = np.unique(np.quantile(amounts, np.linspace(0, 1, 5)[1:-1]))
    assert buckets["boundaries"] == expected.tolist()
    assert {
        identity: sorted(record["files"]) for identity, record in _statistics(fitted).items()
    } == {
        '["inputs","entity","fields","country"]': ["users/country.counts"],
        '["inputs","graph","graph","accounts","account_type"]': ["accounts/account_type.counts"],
        '["inputs","graph","graph","card_transactions","amount"]': [
            "card_transactions/amount.moments"
        ],
        '["inputs","transactions","fields","amount"]': [
            "card_transactions/amount.bins",
            "card_transactions/amount.counts",
        ],
        '["inputs","transactions","fields","channel"]': ["card_transactions/channel.counts"],
    }


def test_a_cutoff_after_the_latest_anchor_warns(tmp_path, caplog):
    dataset, engine = _setup("in_memory", tmp_path)
    for cutoff in ("2025-01-01", None):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="kumo.pytorch.preparation"):
            _encoders(_contract(), dataset, engine, encoder_cutoff=cutoff)
        assert "after the latest fitting anchor" in caplog.text
    with pytest.raises(ValueError, match="encoder_cutoff 'soon' is not an ISO 8601 timestamp"):
        _encoders(_contract(), dataset, engine, encoder_cutoff="soon")
