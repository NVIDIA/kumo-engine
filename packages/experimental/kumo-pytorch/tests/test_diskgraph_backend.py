# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end parity for the intentionally narrow first DiskGraph slice."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kumo.pytorch import api

FIXTURE = Path(__file__).parent / "data" / "synthetic_banking_subset.json"


def _index_at(path: Path) -> SimpleNamespace:
    """Where prepare() ensures the index; tests only need its location."""
    return SimpleNamespace(path=path)


def _fixture_rows() -> dict[str, list[dict[str, object]]]:
    payload = json.loads(FIXTURE.read_text())
    rows = payload["tables"]
    for table, columns in {
        "users": ("signup_at",),
        "accounts": ("opened_at", "closed_at"),
        "card_transactions": ("occurred_at",),
    }.items():
        for row in rows[table]:
            for column in columns:
                if row[column] is not None:
                    row[column] = datetime.fromisoformat(row[column])
    return rows


def _schema() -> api.DataSchema:
    return api.DataSchema(
        tables=(
            api.TableSpec(
                "users",
                "user_id",
                {
                    "user_id": api.DType("int64"),
                    "signup_at": api.DType("timestamp", unit="s", timezone="UTC"),
                    "country": api.DType("string"),
                    "risk_segment": api.DType("string"),
                },
                time_column="signup_at",
            ),
            api.TableSpec(
                "accounts",
                "account_id",
                {
                    "account_id": api.DType("int64"),
                    "user_id": api.DType("int64"),
                    "account_type": api.DType("string"),
                    "opened_at": api.DType("timestamp", unit="s", timezone="UTC"),
                    "closed_at": api.DType("timestamp", unit="s", timezone="UTC"),
                },
                time_column="opened_at",
            ),
            api.TableSpec(
                "card_transactions",
                "card_transaction_id",
                {
                    "card_transaction_id": api.DType("int64"),
                    "user_id": api.DType("int64"),
                    "account_id": api.DType("int64"),
                    "occurred_at": api.DType("timestamp", unit="s", timezone="UTC"),
                    "merchant_id": api.DType("int64"),
                    "amount": api.DType("float64"),
                    "channel": api.DType("string"),
                    "description": api.DType("string"),
                },
                time_column="occurred_at",
            ),
            api.TableSpec(
                "merchants",
                "merchant_id",
                {
                    "merchant_id": api.DType("int64"),
                    "merchant_category": api.DType("string"),
                    "risk_score": api.DType("float64"),
                },
                time_column=api.NON_TEMPORAL,
            ),
        ),
        relations=(
            api.RelationSpec("user_accounts", "users", "user_id", "accounts", "user_id"),
            api.RelationSpec(
                "account_card_transactions",
                "accounts",
                "account_id",
                "card_transactions",
                "account_id",
            ),
            api.RelationSpec(
                "user_card_transactions", "users", "user_id", "card_transactions", "user_id"
            ),
            api.RelationSpec(
                "transaction_merchant",
                "card_transactions",
                "merchant_id",
                "merchants",
                "merchant_id",
            ),
        ),
    )


def _contract(schema: api.DataSchema) -> api.DataLoaderSpec:
    neighborhood = api.NeighborhoodSpec(
        layers=(
            api.GraphLayer((api.GraphSample("user_accounts", 6, strategy="recent"),)),
            api.GraphLayer((api.GraphSample("account_card_transactions", 4, strategy="recent"),)),
        )
    )
    return api.DataLoaderSpec(
        schema=schema,
        sample=api.SampleSpec(
            entity=api.EntitySpec("users", entity_type="user", available_from="signup_at"),
            neighborhoods={"activity": neighborhood},
            targets={"transactions_next_7d": api.ExternalTarget(dtype=api.DType("float64"))},
        ),
        batch=api.BatchSpec(
            inputs={
                "graph": api.Input(
                    api.NeighborhoodSource("activity"), api.Graph(node_fields={}), None
                )
            },
            targets={"future_activity": api.LabelTarget("transactions_next_7d", api.Raw())},
            batching=api.Batching(max_samples=2),
        ),
    )


def _requests() -> tuple[api.SampleRequest, ...]:
    cutoff = datetime.fromisoformat("2024-12-01T00:00:00")
    return (
        api.SampleRequest(
            "user-2-at-2024-12-01",
            2,
            cutoff,
            targets=(api.TargetValue("transactions_next_7d", 3.0),),
            split="train",
        ),
        api.SampleRequest(
            "user-3-at-2024-12-01",
            3,
            cutoff,
            targets=(api.TargetValue("transactions_next_7d", 2.0),),
            split="train",
        ),
    )


def _temporal_sample() -> api.SampleSpec:
    accounts = api.Rows(("user_accounts",), api.TimeWindow("opened_at", None, 0))
    history = api.Rows(("user_card_transactions",), api.TimeWindow("occurred_at", None, 0))
    future = api.Rows(("user_card_transactions",), api.TimeWindow("occurred_at", 0, 7 * 86400))
    per_account = api.GroupBy(
        history,
        keys=("account_id",),
        values={
            "transaction_count": api.Reduce("count", dtype=api.DType("int64")),
            "account_spend": api.Reduce("sum", "amount", dtype=api.DType("float64")),
        },
    )
    per_transaction_count = api.GroupBy(
        per_account,
        keys=("transaction_count",),
        values={
            "account_groups": api.Reduce("count", dtype=api.DType("int64")),
            "bucket_spend": api.Reduce("sum", "account_spend", dtype=api.DType("float64")),
        },
    )
    return api.SampleSpec(
        entity=api.EntitySpec("users", entity_type="user"),
        entity_values={
            "country": api.Field(column="country"),
            "risk_segment": api.Field(column="risk_segment"),
            "transaction_count": api.Aggregate(history, "count", dtype=api.DType("int64")),
            "merchant_count": api.Aggregate(
                history, "count_distinct", "merchant_id", dtype=api.DType("int64")
            ),
            "total_spend": api.Aggregate(history, "sum", "amount", dtype=api.DType("float64")),
            "mean_spend": api.Aggregate(history, "mean", "amount", dtype=api.DType("float64")),
            "minimum_spend": api.Aggregate(history, "min", "amount", dtype=api.DType("float64")),
            "maximum_spend": api.Aggregate(history, "max", "amount", dtype=api.DType("float64")),
            "variance_population": api.Aggregate(
                history, "var_population", "amount", dtype=api.DType("float64")
            ),
            "variance_sample": api.Aggregate(
                history, "var_sample", "amount", dtype=api.DType("float64")
            ),
            "std_population": api.Aggregate(
                history, "std_population", "amount", dtype=api.DType("float64")
            ),
            "std_sample": api.Aggregate(
                history, "std_sample", "amount", dtype=api.DType("float64")
            ),
            "last_activity_seconds": api.Aggregate(
                history, "recency", "occurred_at", dtype=api.DType("float64")
            ),
            "usual_channel": api.Aggregate(history, "mode", "channel", dtype=api.DType("string")),
            "spend_across_account_groups": api.Aggregate(
                per_transaction_count, "sum", "bucket_spend", dtype=api.DType("float64")
            ),
        },
        events={
            "accounts": api.EventSpec(
                accounts,
                fields={
                    "account_type": api.Field(column="account_type"),
                    "age_days": api.EventTimeDelta("days", dtype=api.DType("float64")),
                },
            ),
            "transactions": api.EventSpec(
                history,
                fields={
                    "amount": api.Field(column="amount"),
                    "channel": api.Field(column="channel"),
                    "age_hours": api.EventTimeDelta("hours", dtype=api.DType("float64")),
                },
                limit=5,
                order="descending",
            ),
        },
        targets={
            "transactions_next_7d": api.ExternalTarget(dtype=api.DType("float64")),
            "future_spend": api.Aggregate(future, "sum", "amount", dtype=api.DType("float64")),
            "future_count": api.Aggregate(future, "count", dtype=api.DType("int64")),
            "next_transaction": api.RowTarget(
                future,
                fields={
                    "amount": api.Field(column="amount"),
                    "channel": api.Field(column="channel"),
                },
                selection="first",
            ),
        },
        neighborhoods={
            "activity": api.NeighborhoodSpec(
                layers=(
                    api.GraphLayer((api.GraphSample("user_accounts", 6, strategy="recent"),)),
                    api.GraphLayer(
                        (api.GraphSample("account_card_transactions", 4, strategy="recent"),)
                    ),
                )
            )
        },
    )


def _temporal_contract(schema: api.DataSchema) -> api.DataLoaderSpec:
    return api.DataLoaderSpec(
        schema=schema,
        sample=_temporal_sample(),
        batch=api.BatchSpec(
            inputs={
                "entity": api.Input(
                    api.EntitySource(),
                    api.Fields(
                        fields=(
                            api.Encode(
                                field="country",
                                encoding=api.Vocabulary(values=(("C021", 2), ("C002", 3))),
                            ),
                            api.Encode(
                                field="risk_segment", encoding=api.Vocabulary(values=(("low", 2),))
                            ),
                            api.Encode(field="transaction_count", encoding=api.Raw()),
                            api.Encode(field="total_spend", encoding=api.Raw()),
                        ),
                        row=api.Columns(),
                    ),
                    api.Separate(),
                ),
                "transactions": api.Input(
                    api.EventSource(("transactions",)),
                    api.Fields(
                        fields=(
                            api.Encode(field="amount", encoding=api.Raw()),
                            api.Encode(
                                field="channel",
                                encoding=api.Vocabulary(values=(("online", 2), ("store", 3))),
                            ),
                            api.Encode(field="age_hours", encoding=api.Raw()),
                        ),
                        row=api.Columns(),
                    ),
                    api.Padded(),
                ),
                "timeline": api.Input(
                    api.EventSource(("accounts", "transactions"), max_events=4, keep="last"),
                    api.Fields(
                        fields=(
                            api.Encode(
                                field="account_type",
                                encoding=api.Vocabulary(values=(("checking", 2), ("savings", 3))),
                            ),
                            api.Encode(field="amount", encoding=api.Raw()),
                            api.Encode(
                                field="channel",
                                encoding=api.Vocabulary(values=(("online", 2), ("store", 3))),
                            ),
                            api.Encode(field="age_days", encoding=api.Raw()),
                            api.Encode(field="age_hours", encoding=api.Raw()),
                        ),
                        row=api.Columns(),
                    ),
                    api.Packed(),
                ),
                "graph": api.Input(
                    api.NeighborhoodSource("activity"),
                    api.Graph(
                        node_fields={
                            "accounts": api.Fields(
                                fields=(
                                    api.Encode(field="account_type", encoding=api.Vocabulary()),
                                )
                            ),
                            "card_transactions": api.Fields(
                                fields=(api.Encode(field="amount", encoding=api.ZScore()),)
                            ),
                        }
                    ),
                    None,
                ),
            },
            targets={
                "future_spend": api.LabelTarget("future_spend", api.Raw()),
                "next_transaction": api.StructuredTarget(
                    "next_transaction",
                    (
                        api.Encode(field="amount", encoding=api.Raw()),
                        api.Encode(
                            field="channel",
                            encoding=api.Vocabulary(values=(("online", 2), ("store", 3))),
                        ),
                    ),
                ),
            },
            batching=api.Batching(max_samples=2),
        ),
    )


def _pragma_contract(schema: api.DataSchema) -> api.DataLoaderSpec:
    history = api.Rows(path=("user_card_transactions",), time=api.TimeWindow(column="occurred_at"))
    roles = {
        name: f"[{name.upper()}]"
        for name in ("pad", "unknown", "missing", "mask", "entity", "event")
    }
    return api.DataLoaderSpec(
        schema=schema,
        sample=api.SampleSpec(
            entity=api.EntitySpec(table="users"),
            entity_values={
                "country": api.Field(column="country"),
                "risk_segment": api.Field(column="risk_segment"),
                "transaction_count": api.Aggregate(
                    source=history, operation="count", dtype=api.DType(name="int64")
                ),
            },
            events={
                "transactions": api.EventSpec(
                    rows=history,
                    fields={
                        "amount": api.Field(column="amount"),
                        "channel": api.Field(column="channel"),
                        "description": api.Field(column="description"),
                        "time": api.Field(column="occurred_at"),
                        "hour": api.Derived(
                            operation=api.CalendarPart(part="hour"),
                            inputs=("time",),
                            dtype=api.DType(name="int64"),
                        ),
                        "month": api.Derived(
                            operation=api.CalendarPart(part="month"),
                            inputs=("time",),
                            dtype=api.DType(name="int64"),
                        ),
                        "weekday": api.Derived(
                            operation=api.CalendarPart(part="weekday"),
                            inputs=("time",),
                            dtype=api.DType(name="int64"),
                        ),
                        "day": api.Derived(
                            operation=api.CalendarPart(part="day"),
                            inputs=("time",),
                            dtype=api.DType(name="int64"),
                        ),
                        "age_seconds": api.EventTimeDelta(
                            unit="seconds", dtype=api.DType(name="float64")
                        ),
                    },
                )
            },
            targets={"transactions_next_7d": api.ExternalTarget(dtype=api.DType(name="float64"))},
        ),
        batch=api.BatchSpec(
            encodings={"categories": api.Vocabulary()},
            id_spaces={"pragma": api.IdSpace(specials=roles)},
            field_names={"fields": api.FieldNames()},
            inputs={
                "entity": api.Input(
                    source=api.EntitySource(),
                    representation=api.Fields(
                        fields=(
                            api.Encode(
                                field="country",
                                encoding=api.UseEncoding(name="categories"),
                                id_space="pragma",
                            ),
                            api.Encode(
                                field="risk_segment",
                                encoding=api.UseEncoding(name="categories"),
                                id_space="pragma",
                            ),
                            api.Encode(
                                field="transaction_count",
                                encoding=api.Buckets(bins=4, separate_zero=True),
                                id_space="pragma",
                            ),
                        ),
                        row=api.Tokens(max_tokens=8),
                        field_names="fields",
                        field_space="pragma",
                    ),
                    layout=api.Padded(),
                ),
                "history": api.Input(
                    source=api.EventSource(events=("transactions",), max_events=16, keep="last"),
                    representation=api.Fields(
                        fields=(
                            api.Encode(
                                field="amount",
                                encoding=api.Buckets(bins=4, separate_zero=True),
                                id_space="pragma",
                            ),
                            api.Encode(
                                field="channel",
                                encoding=api.UseEncoding(name="categories"),
                                id_space="pragma",
                            ),
                            api.Encode(
                                field="description",
                                encoding=api.Text(vocabulary_size=64, empty="unknown"),
                                id_space="pragma",
                            ),
                        ),
                        row=api.Tokens(max_tokens=8),
                        field_names="fields",
                        field_space="pragma",
                    ),
                    layout=api.Nested(fields=api.Packed(), events=api.Padded()),
                ),
                "history_calendar": api.Input(
                    source=api.EventsOf(input="history"),
                    representation=api.Fields(
                        fields=tuple(
                            api.Encode(field=name, encoding=api.Raw())
                            for name in ("hour", "weekday", "day")
                        )
                    ),
                    layout=api.Padded(),
                ),
                "history_timing": api.Input(
                    source=api.EventsOf(input="history"),
                    representation=api.Fields(
                        fields=(api.Encode(field="age_seconds", encoding=api.Raw()),)
                    ),
                    layout=api.Padded(),
                ),
            },
            targets={
                "masked_values": api.MaskedValuesTarget(
                    input="history",
                    token_probability=0.15,
                    event_probability=0.1,
                    key_probability=0.1,
                    unknown_probability=0.1,
                    fields=("amount", "channel", "description"),
                    ensure_at_least_one=True,
                )
            },
            batching=api.Batching(
                max_samples=2, max_tokens=16, length_input="history_timing", bucket_by_length=True
            ),
        ),
    )


def _write_parquet(rows: dict[str, list[dict[str, object]]], root: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root.mkdir()
    sources = {}
    for table, values in rows.items():
        path = root / f"{table}.parquet"
        pq.write_table(pa.Table.from_pylist(values), path)
        sources[table] = api.Parquet(str(path))
    return api.Dataset(sources)


def _preview(contract, dataset, engine, requests):
    with api.prepare(contract, dataset, engine) as plan:
        return plan.materialize(requests, fit_split="train").preview(requests, seed=7).batch


@pytest.mark.parametrize("engine_name", ("in_memory", "diskgraph"))
def test_named_dictionary_placement_preserves_columns_targets_and_graphs(tmp_path, engine_name):
    from tests.helpers import reorder_dicts
    from tests.test_preview import _equal as assert_tree_equal

    rows = _fixture_rows()
    if engine_name == "diskgraph":
        pytest.importorskip("diskgraph")
        dataset = _write_parquet(rows, tmp_path / "tables")
        engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1, num_shards=1)
    else:
        dataset = api.Dataset({name: api.InMemoryRows(values) for name, values in rows.items()})
        engine = api.InMemoryEngine()
    contract = _temporal_contract(_schema())
    payload = contract.to_dict()
    expected = None
    fitted_fingerprint = None
    for variant in (payload, reorder_dicts(payload), reorder_dicts(payload, shuffled=True)):
        restored = api.DataLoaderSpec.from_dict(variant)
        assert restored.fingerprint == contract.fingerprint
        with api.prepare(restored, dataset, engine) as plan:
            with plan.materialize(_requests(), fit_split="train") as task:
                bound = task.bound_contract()
                batch = task.preview(_requests(), seed=7).batch
                if expected is not None:
                    assert bound.fingerprint == fitted_fingerprint
                    assert_tree_equal(batch.inputs, expected.inputs)
                    assert_tree_equal(batch.targets, expected.targets)
                expected, fitted_fingerprint = (batch, bound.fingerprint)


@pytest.mark.parametrize("max_events", [None, 2])
def test_diskgraph_flat_model_stream_matches_inspectable(tmp_path, monkeypatch, max_events):
    pytest.importorskip("diskgraph")
    from dataclasses import replace

    from tests.test_preview import _equal as assert_tree_equal

    rows, schema = (_fixture_rows(), _schema())
    parquet = _write_parquet(rows, tmp_path / "parquet")
    artifact = _index_at(tmp_path / "index")
    base = _pragma_contract(schema)
    requests = _requests()
    for layout in (api.Packed(), api.Padded()):
        for supervision in (False, True):
            stream = api.Fields(
                fields=(
                    api.Encode(field="channel", encoding=api.Vocabulary(), id_space="stream"),
                    api.Encode(field="amount", encoding=api.Buckets(bins=4), id_space="stream"),
                    api.Encode(
                        field="month",
                        encoding=api.Vocabulary(
                            values=tuple((month, month + 1) for month in range(1, 13))
                        ),
                        id_space="stream",
                    ),
                    api.Encode(
                        field="description",
                        encoding=api.Text(vocabulary_size=32),
                        id_space="stream",
                    ),
                ),
                row=api.Tokens(missing="skip_event"),
                history=api.Flatten(max_tokens=100, before="bos", after="eos"),
            )
            batch = api.BatchSpec(
                inputs={
                    "history": api.Input(
                        api.EventSource(("transactions",), max_events=max_events), stream, layout
                    )
                },
                targets={"next": api.NextTokenTarget("history")} if supervision else {},
                id_spaces={
                    "stream": api.IdSpace(
                        specials={
                            "pad": "[PAD]",
                            "unknown": "[UNK]",
                            "missing": "[MISSING]",
                            "bos": "<bos>",
                            "eos": "<eos>",
                        }
                    )
                },
                batching=api.Batching(
                    max_tokens=10000, length_input="history", bucket_by_length=True
                )
                if max_events is not None
                else api.Batching(),
            )
            contract = replace(base, batch=batch)
            with api.prepare(
                contract, parquet, api.DiskGraphEngine(index=artifact.path, num_threads=1)
            ) as plan:
                task = plan.materialize(requests, fit_split="train")
                expected = next(iter(task.batches(requests, batch_size=2)))
                plan._source._adapt_temporal_result = lambda *_: pytest.fail(
                    "flat model execution must not construct inspectable events"
                )
                actual = next(iter(task.batches(requests, batch_size=2, output_format="model")))
                assert_tree_equal(expected.inputs, actual.inputs)
                assert_tree_equal(expected.targets, actual.targets)
                assert_tree_equal(expected.metadata, actual.metadata)
                bound = task.bound_contract()
            from kumo.pytorch.engines.diskgraph import DiskGraphSource

            with monkeypatch.context() as patch:
                patch.setattr(
                    DiskGraphSource,
                    "build",
                    lambda *_: pytest.fail(
                        "supplied stream state must not fit through inspectable records"
                    ),
                )
                bound = replace(
                    bound,
                    batch=replace(
                        bound.batch,
                        batching=api.Batching(
                            max_samples=2,
                            max_tokens=200,
                            length_input="history",
                            bucket_by_length=True,
                        ),
                    ),
                )
                with api.prepare(
                    bound, parquet, api.DiskGraphEngine(index=artifact.path, num_threads=1)
                ) as restored:
                    restored_task = restored.materialize(requests, fit_split="train")
                    batches = list(restored_task.batches(requests, output_format="model"))
                    assert sum(len(batch.metadata["sample_ids"]) for batch in batches) == 2


def _canonical_graph(graph: dict[str, object]) -> dict[str, object]:
    nodes = graph["nodes"]
    edges = graph["edges"]
    canonical_edges = {}
    for relation, edge in edges.items():
        source_ids = nodes[edge["source_type"]]["ids"]
        destination_ids = nodes[edge["destination_type"]]["ids"]
        canonical_edges[relation] = sorted(
            (
                (int(source_ids[source]), int(destination_ids[destination]))
                for source, destination in edge["index"].t().tolist()
            )
        )
    return {
        "nodes": {
            table: sorted(
                (
                    (int(raw_id), int(owner))
                    for raw_id, owner in zip(value["ids"], value["batch"], strict=True)
                )
            )
            for table, value in nodes.items()
        },
        "edges": canonical_edges,
    }


def _train_one_pyg_step(batch: object) -> float:
    torch = pytest.importorskip("torch")
    nn = torch.nn
    HeteroData = pytest.importorskip("torch_geometric.data").HeteroData
    hetero = pytest.importorskip("torch_geometric.nn")
    data = HeteroData()
    graph = batch.inputs["graph"]
    for node_type, node in graph["nodes"].items():
        data[node_type].node_id = node["ids"]
        data[node_type].batch = node["batch"]
    for relation, edge in graph["edges"].items():
        source = edge["source_type"]
        destination = edge["destination_type"]
        data[source, relation, destination].edge_index = edge["index"]
        data[destination, f"reverse_{relation}", source].edge_index = edge["index"].flip(0)

    class ActivityGNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            node_types, edge_types = data.metadata()
            hidden_size = 16
            self.node_vectors = nn.ParameterDict(
                {node_type: nn.Parameter(torch.randn(1, hidden_size)) for node_type in node_types}
            )
            self.layers = nn.ModuleList(
                [
                    hetero.HeteroConv(
                        {
                            edge_type: hetero.SAGEConv((-1, -1), hidden_size)
                            for edge_type in edge_types
                        },
                        aggr="sum",
                    )
                    for _ in range(3)
                ]
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self) -> object:
            node_vectors = {
                node_type: self.node_vectors[node_type].expand(data[node_type].num_nodes, -1)
                for node_type in data.node_types
            }
            for layer in self.layers:
                node_vectors = {
                    node_type: values.relu()
                    for node_type, values in layer(node_vectors, data.edge_index_dict).items()
                }
            users = hetero.global_mean_pool(node_vectors["users"], data["users"].batch)
            return self.head(users).squeeze(-1)

    torch.manual_seed(7)
    model = ActivityGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    predictions = model()
    targets = batch.targets["future_activity"]["values"].to(predictions.dtype)
    before = model.head.weight.detach().clone()
    loss = nn.functional.mse_loss(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert predictions.shape == targets.shape == (2,)
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.head.weight.detach())
    return float(loss.detach())


def test_synthetic_banking_materialization_sampling_and_reopen_are_reproducible(
    tmp_path: Path,
) -> None:
    pytest.importorskip("diskgraph")
    rows = _fixture_rows()
    rows["card_transactions"].append(
        {
            "card_transaction_id": 999999,
            "user_id": 2,
            "account_id": 4,
            "occurred_at": datetime.fromisoformat("2024-12-01T00:00:00"),
            "merchant_id": 139849,
            "amount": 1000.0,
            "channel": "online",
            "description": "included at the anchor",
        }
    )
    schema = _schema()
    contract = _contract(schema)
    requests = _requests()
    reference = _preview(
        contract,
        api.Dataset({name: api.InMemoryRows(values) for name, values in rows.items()}),
        api.InMemoryEngine(),
        requests,
    )
    parquet = _write_parquet(rows, tmp_path / "parquet")
    indexes = [tmp_path / f"index-{run}" for run in range(2)]
    batches = [
        _preview(contract, parquet, api.DiskGraphEngine(index=index, num_threads=1), requests)
        for index in indexes
    ]
    expected_graph = _canonical_graph(reference.inputs["graph"])
    assert expected_graph["nodes"] == {
        "accounts": [(4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 1), (11, 1), (12, 1)],
        "card_transactions": [
            (61, 0),
            (102, 0),
            (141, 0),
            (387, 0),
            (713, 1),
            (732, 1),
            (774, 1),
            (826, 1),
            (999999, 0),
        ],
        "users": [(2, 0), (3, 1)],
    }
    assert [_canonical_graph(batch.inputs["graph"]) for batch in batches] == [
        expected_graph,
        expected_graph,
    ]
    assert [batch.targets["future_activity"]["values"].tolist() for batch in batches] == [
        [3.0, 2.0],
        [3.0, 2.0],
    ]
    assert (
        json.loads((indexes[0] / "index_manifest.json").read_text())["config_fingerprint"]
        == json.loads((indexes[1] / "index_manifest.json").read_text())["config_fingerprint"]
    )
    assert all((index / "index_manifest.json").is_file() for index in indexes)
    assert _train_one_pyg_step(batches[0]) >= 0.0
    with api.prepare(
        contract, parquet, api.DiskGraphEngine(index=indexes[0], num_threads=1)
    ) as plan:
        model_batch = next(
            iter(
                plan.materialize(requests, fit_split="train").batches(
                    requests, output_format="model", seed=7
                )
            )
        )
    assert _canonical_graph(model_batch.inputs["graph"]) == expected_graph
    assert model_batch.targets["future_activity"]["values"].tolist() == [3.0, 2.0]
    signup = next(row["signup_at"] for row in rows["users"] if row["user_id"] == 2)
    engine = api.DiskGraphEngine(index=indexes[0], num_threads=1)
    for cutoff in (signup - timedelta(seconds=1), signup):
        request = api.SampleRequest("before-signup", 2, cutoff)
        with pytest.raises(ValueError, match="entity did not exist before"):
            with api.prepare(contract, parquet, engine) as plan:
                plan.materialize((request,))
    request = api.SampleRequest(
        "after-signup", 2, signup + timedelta(seconds=1), targets=requests[0].targets
    )
    with api.prepare(contract, parquet, engine) as plan:
        task = plan.materialize((request,))
        for output_format in ("inspectable", "model"):
            batch = next(iter(task.batches((request,), output_format=output_format)))
            assert batch.inputs["graph"]["nodes"]["users"]["ids"].tolist() == [2]
            for cutoff in (signup - timedelta(seconds=1), signup):
                unavailable = api.SampleRequest(
                    "before-signup", 2, cutoff, targets=requests[0].targets
                )
                with pytest.raises(ValueError, match="entity did not exist before"):
                    next(iter(task.batches((unavailable,), output_format=output_format)))


def test_diskgraph_check_refuses_unsupported_work_before_ingestion(tmp_path: Path) -> None:
    schema = _schema()
    dataset = api.Dataset({table.name: api.Parquet("/does/not/exist") for table in schema.tables})
    engine = api.DiskGraphEngine(index=tmp_path / "not-created")
    contract = _contract(schema)
    assert not api.check(contract, dataset, engine)
    uniform = api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            entity=contract.sample.entity,
            neighborhoods={
                "activity": api.NeighborhoodSpec(
                    layers=(
                        api.GraphLayer((api.GraphSample("user_accounts", 2, strategy="uniform"),)),
                    )
                )
            },
        ),
        contract.batch,
    )
    assert any(
        issue.path.startswith("sample.neighborhoods")
        for issue in api.check(uniform, dataset, engine)
    )
    with_values = api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            entity=contract.sample.entity,
            entity_values={"country": api.Field(column="country")},
            neighborhoods=contract.sample.neighborhoods,
            targets=contract.sample.targets,
        ),
        contract.batch,
    )
    assert not api.check(with_values, dataset, engine)
    anchor_table = api.AnchorTable(
        source=api.Parquet("/does/not/exist"), entity_id="user_id", as_of_time="as_of_time"
    )
    assert any(
        issue.path.startswith("anchors.source")
        for issue in api.check(contract, dataset, engine, anchors=anchor_table)
    )
    assert not (tmp_path / "not-created").exists()


def test_diskgraph_check_refuses_timed_intermediate_rows(tmp_path: Path) -> None:
    schema = _schema()
    rows = api.Rows(
        ("user_accounts", "account_card_transactions"), api.TimeWindow("occurred_at", None, 0)
    )
    contract = api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            entity=api.EntitySpec("users", entity_type="user"),
            entity_values={
                "spend": api.Aggregate(rows, "sum", "amount", dtype=api.DType("float64"))
            },
            events={
                "transactions": api.EventSpec(rows, fields={"amount": api.Field(column="amount")})
            },
        ),
        api.BatchSpec(
            inputs={
                "spend": api.Input(
                    api.EntitySource(),
                    api.Fields(
                        fields=(api.Encode(field="spend", encoding=api.Raw()),), row=api.Columns()
                    ),
                    api.Separate(),
                )
            }
        ),
    )
    dataset = api.Dataset({table.name: api.Parquet("/does/not/exist") for table in schema.tables})
    issues = api.check(contract, dataset, api.DiskGraphEngine(index=tmp_path / "index"))
    assert {issue.path for issue in issues if "intermediate" in issue.message} == {
        "sample.entity_values.spend.source.path[0]",
        "sample.events.transactions.rows.path[0]",
    }
    assert not (tmp_path / "index").exists()


def test_diskgraph_refuses_future_intermediate_edge_that_leaks_old_rows(tmp_path: Path) -> None:
    cutoff = datetime(2024, 1, 10)
    schema = api.DataSchema(
        tables=(
            api.TableSpec(
                "accounts",
                "account_id",
                {"account_id": api.DType("int64")},
                time_column=api.NON_TEMPORAL,
            ),
            api.TableSpec(
                "transfers",
                "transfer_id",
                {
                    "transfer_id": api.DType("int64"),
                    "from_id": api.DType("int64"),
                    "to_id": api.DType("int64"),
                    "occurred_at": api.DType("timestamp", unit="s", timezone="UTC"),
                    "amount": api.DType("float64"),
                },
                time_column="occurred_at",
            ),
        ),
        relations=(
            api.RelationSpec("out", "accounts", "account_id", "transfers", "from_id"),
            api.RelationSpec("to", "transfers", "to_id", "accounts", "account_id"),
        ),
    )
    contract = api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            entity=api.EntitySpec("accounts", entity_type="account"),
            entity_values={
                "spend": api.Aggregate(
                    api.Rows(("out", "to", "out"), api.TimeWindow("occurred_at", None, 0)),
                    "sum",
                    "amount",
                    dtype=api.DType("float64"),
                )
            },
        ),
        api.BatchSpec(
            inputs={
                "account": api.Input(
                    api.EntitySource(),
                    api.Fields(
                        fields=(api.Encode(field="spend", encoding=api.Raw()),), row=api.Columns()
                    ),
                    api.Separate(),
                )
            },
            batching=api.Batching(max_samples=1),
        ),
    )
    dataset = api.Dataset(
        {
            "accounts": api.InMemoryRows([{"account_id": 1}, {"account_id": 2}]),
            "transfers": api.InMemoryRows(
                [
                    {
                        "transfer_id": 10,
                        "from_id": 1,
                        "to_id": 2,
                        "occurred_at": datetime(2024, 1, 11),
                        "amount": 0.0,
                    },
                    {
                        "transfer_id": 11,
                        "from_id": 2,
                        "to_id": 1,
                        "occurred_at": datetime(2024, 1, 9),
                        "amount": 500.0,
                    },
                ]
            ),
        }
    )
    request = api.SampleRequest("account-1", 1, cutoff, split="train")
    with api.prepare(contract, dataset, api.InMemoryEngine()) as plan:
        batch = next(plan.materialize((request,), fit_split="train").batches())
    assert batch.inputs["account"]["spend"]["values"].tolist() == [500.0]
    issues = api.check(contract, dataset, api.DiskGraphEngine(index=tmp_path / "index"))
    assert any("intermediate" in issue.message for issue in issues)
    assert not (tmp_path / "index").exists()


@pytest.mark.parametrize(
    ("left_key", "right_key"), (("country", "account_type"), ("user_id", "account_id"))
)
@pytest.mark.parametrize("used", (False, True))
def test_diskgraph_check_refuses_non_fk_pk_relations(
    tmp_path: Path, left_key: str, right_key: str, used: bool
) -> None:
    base = _schema()
    schema = api.DataSchema(
        base.tables,
        (
            *base.relations,
            api.RelationSpec("unsupported", "users", left_key, "accounts", right_key),
        ),
    )
    contract = _contract(schema)
    if used:
        contract = api.DataLoaderSpec(
            schema,
            api.SampleSpec(
                entity=contract.sample.entity,
                neighborhoods={
                    "activity": api.NeighborhoodSpec(
                        (api.GraphLayer((api.GraphSample("unsupported", 2, strategy="recent"),)),)
                    )
                },
            ),
            contract.batch,
        )
    dataset = api.Dataset({table.name: api.Parquet("/does/not/exist") for table in schema.tables})
    issues = api.check(contract, dataset, api.DiskGraphEngine(index=tmp_path / "index"))
    index = next(
        index for index, relation in enumerate(schema.relations) if relation.name == "unsupported"
    )
    assert any(issue.path == f"schema.relations[{index}]" for issue in issues)
    assert not (tmp_path / "index").exists()


def test_prepare_builds_declared_columns_drops_null_clocks_and_reuses_index(tmp_path: Path) -> None:
    diskgraph = pytest.importorskip("diskgraph")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    users = tmp_path / "users.parquet"
    events = tmp_path / "events.parquet"
    pq.write_table(pa.table({"user_id": [1]}), users)
    pq.write_table(
        pa.table(
            {
                "event_id": [10, 11],
                "user_id": [1, 1],
                "occurred_at": [datetime(2024, 1, 1), None],
                "unused_nested": [[1], [2]],
            }
        ),
        events,
    )
    schema = api.DataSchema(
        tables=(
            api.TableSpec(
                "users", "user_id", {"user_id": api.DType("int64")}, time_column=api.NON_TEMPORAL
            ),
            api.TableSpec(
                "events",
                "event_id",
                {
                    "event_id": api.DType("int64"),
                    "user_id": api.DType("int64"),
                    "occurred_at": api.DType("timestamp", unit="s", timezone="UTC"),
                },
                time_column="occurred_at",
            ),
        ),
        relations=(api.RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
    )
    contract = api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            entity=api.EntitySpec("users", entity_type="user"),
            neighborhoods={
                "history": api.NeighborhoodSpec(
                    (api.GraphLayer((api.GraphSample("user_events", 4, strategy="recent"),)),)
                )
            },
        ),
        api.BatchSpec(
            inputs={"graph": api.Input(api.NeighborhoodSource("history"), api.Graph(), None)},
            batching=api.Batching(max_samples=1),
        ),
    )
    dataset = api.Dataset({"users": api.Parquet(str(users)), "events": api.Parquet(str(events))})
    index = tmp_path / "index"
    engine = api.DiskGraphEngine(index=index, num_threads=1)
    request = api.SampleRequest("one", 1, datetime(2024, 2, 1))
    with pytest.raises(FileNotFoundError):
        api.prepare(contract, dataset, api.DiskGraphEngine(index=index, ingest="never"))
    with pytest.warns(UserWarning, match="dropped source rows"):
        with api.prepare(contract, dataset, engine) as plan:
            graph = next(plan.materialize((request,)).batches((request,))).inputs["graph"]
    assert graph["nodes"]["events"]["ids"].tolist() == [10]
    config = json.loads((index / "diskgraph_config.json").read_text())
    event_table = next(table for table in config["tables"] if table["name"] == "events")
    assert {column["name"] for column in event_table["columns"]} == {
        "event_id",
        "user_id",
        "occurred_at",
    }
    stats = json.loads((index / "ingest_stats.json").read_text())["stats"]
    assert stats["source_rows_by_table"]["events"] == 2
    assert stats["node_counts"]["events"] == 1
    sampler = diskgraph.SamplerEngine(str(index), num_threads=1)
    assert sampler.lookup_node_ids("events", pa.array([10, 11])) == [0, None]
    relation = json.loads((index / "index_manifest.json").read_text())["relation_mapping"]
    plan_json = json.dumps(
        {
            "frames": [
                {
                    "name": "events",
                    "root_type": "users",
                    "path": [
                        {
                            "relation": relation["user_events"]["forward"],
                            "destination_type": "events",
                        }
                    ],
                    "time_column": "occurred_at",
                    "start_seconds": None,
                    "stop_seconds": 0,
                    "distinct": True,
                    "columns": ["event_id"],
                }
            ],
            "sequences": [
                {
                    "name": "history",
                    "frame": "events",
                    "columns": ["event_id"],
                    "order": "ascending",
                    "limit": None,
                }
            ],
        }
    )
    result = sampler.query_temporal(
        {"users": [0, 0]}, {"users": [1704067200, 1706745600]}, plan_json
    )
    assert result["sequences"][0]["offsets"] == [0, 1, 2]
    assert result["stats"]["rows_decoded"] == 2
    before = (index / "index_manifest.json").stat().st_mtime_ns
    with pytest.warns(UserWarning, match="dropped source rows"):
        with api.prepare(contract, dataset, api.DiskGraphEngine(index=index, ingest="never")):
            pass
    assert (index / "index_manifest.json").stat().st_mtime_ns == before
    changed = api.DataSchema(
        tables=(
            next(table for table in schema.tables if table.name == "users"),
            api.TableSpec(
                "events",
                "event_id",
                {
                    "user_id": api.DType("int64"),
                    "event_id": api.DType("int32"),
                    "occurred_at": api.DType("timestamp", unit="s", timezone="UTC"),
                },
                time_column="occurred_at",
            ),
        ),
        relations=schema.relations,
    )
    changed_contract = api.DataLoaderSpec(changed, contract.sample, contract.batch)
    with pytest.raises(ValueError, match="different schema"):
        api.prepare(changed_contract, dataset, engine)


def test_diskgraph_temporal_sequences_aggregates_and_labels_match_reference(tmp_path: Path) -> None:
    pytest.importorskip("diskgraph")
    rows = _fixture_rows()
    schema = _schema()
    sample = _temporal_sample()
    requests = _requests()
    memory_dataset = api.Dataset({name: api.InMemoryRows(values) for name, values in rows.items()})
    parquet = _write_parquet(rows, tmp_path / "parquet")
    artifact = _index_at(tmp_path / "index")
    reference_source = api.InMemoryEngine().open(memory_dataset, schema)
    diskgraph_source = api.DiskGraphEngine(index=artifact.path, num_threads=1).open(parquet, schema)
    try:
        expected = reference_source.build(sample, requests)
        actual = diskgraph_source.build(sample, requests)
    finally:
        reference_source.close()
        diskgraph_source.close()
    for expected_sample, actual_sample in zip(expected, actual, strict=True):
        assert actual_sample.history.events == expected_sample.history.events
        expected_values = {
            value.key: value.value for value in expected_sample.history.entity.values
        }
        actual_values = {value.key: value.value for value in actual_sample.history.entity.values}
        assert actual_values.keys() == expected_values.keys()
        for name, expected_value in expected_values.items():
            if isinstance(expected_value, float):
                assert actual_values[name] == pytest.approx(expected_value)
            else:
                assert actual_values[name] == expected_value
        expected_targets = {target.name: target.value for target in expected_sample.targets}
        actual_targets = {target.name: target.value for target in actual_sample.targets}
        assert actual_targets.keys() == expected_targets.keys()
        assert actual_targets["future_spend"] == pytest.approx(expected_targets["future_spend"])
        assert actual_targets["future_count"] == expected_targets["future_count"]
        assert actual_targets["next_transaction"] == expected_targets["next_transaction"]
    profile = diskgraph_source.last_temporal_profile
    assert profile is not None
    assert profile["samples"] == 2
    assert profile["native_stats"]["frames_materialized"] == 3
    assert set(profile["native_timings_ns"]) == {
        "traversal",
        "decode_and_select",
        "sequence_assembly",
        "aggregation",
        "total",
    }
    contract = _temporal_contract(schema)
    expected_batch = api.preview(
        contract, memory_dataset, api.InMemoryEngine(), requests, seed=7
    ).batch
    actual_batch = api.preview(
        contract, parquet, api.DiskGraphEngine(index=artifact.path, num_threads=1), requests, seed=7
    ).batch

    def assert_equal(actual_value: object, expected_value: object) -> None:
        if isinstance(expected_value, torch.Tensor):
            assert isinstance(actual_value, torch.Tensor)
            torch.testing.assert_close(actual_value, expected_value)
        elif isinstance(expected_value, Mapping):
            assert isinstance(actual_value, Mapping)
            assert actual_value.keys() == expected_value.keys()
            for key in expected_value:
                assert_equal(actual_value[key], expected_value[key])
        else:
            assert actual_value == expected_value

    assert_equal(actual_batch.inputs["entity"], expected_batch.inputs["entity"])
    assert_equal(actual_batch.inputs["transactions"], expected_batch.inputs["transactions"])
    assert_equal(actual_batch.inputs["timeline"], expected_batch.inputs["timeline"])
    assert _canonical_graph(actual_batch.inputs["graph"]) == _canonical_graph(
        expected_batch.inputs["graph"]
    )
    for table, expected_node in expected_batch.inputs["graph"]["nodes"].items():
        actual_node = actual_batch.inputs["graph"]["nodes"][table]
        for field, expected_feature in expected_node["features"].items():
            actual_feature = actual_node["features"][field]
            expected_by_node = {
                (int(raw_id), int(owner)): (value.item(), bool(valid))
                for raw_id, owner, value, valid in zip(
                    expected_node["ids"],
                    expected_node["batch"],
                    expected_feature["values"],
                    expected_feature["valid"],
                    strict=True,
                )
            }
            actual_by_node = {
                (int(raw_id), int(owner)): (value.item(), bool(valid))
                for raw_id, owner, value, valid in zip(
                    actual_node["ids"],
                    actual_node["batch"],
                    actual_feature["values"],
                    actual_feature["valid"],
                    strict=True,
                )
            }
            assert actual_by_node.keys() == expected_by_node.keys()
            for identity, (expected_value, expected_valid) in expected_by_node.items():
                actual_value, actual_valid = actual_by_node[identity]
                assert actual_valid == expected_valid
                if isinstance(expected_value, float):
                    assert actual_value == pytest.approx(expected_value)
                else:
                    assert actual_value == expected_value
    assert_equal(actual_batch.targets, expected_batch.targets)
    with api.prepare(
        contract, parquet, api.DiskGraphEngine(index=artifact.path, num_threads=1)
    ) as plan:
        task = plan.materialize(requests, fit_split="train")
        inspectable_batch = next(
            iter(task.batches(requests, batch_size=2, output_format="inspectable", seed=7))
        )
        plan._source._adapt_temporal_result = lambda *_args: pytest.fail(
            "model output entered the inspectable Event/FieldValue adapter"
        )
        model_batch = next(
            iter(task.batches(requests, batch_size=2, output_format="model", seed=7))
        )
        assert_equal(model_batch.inputs, inspectable_batch.inputs)
        assert_equal(model_batch.targets, inspectable_batch.targets)
        assert_equal(model_batch.metadata, inspectable_batch.metadata)
        assert task._assembler.last_batch_profile["output_format"] == "model"
        assert plan._source.last_temporal_profile["output_format"] == "model"
        assert set(plan._source.last_graph_profile) == {
            "samples",
            "sampling_ns",
            "event_decode_ns",
            "identity_fetch_ns",
            "feature_fetch_ns",
            "feature_encode_ns",
            "topology_assembly_ns",
            "total_ns",
        }


def test_diskgraph_pragma_batches_match_reference_without_record_adaptation(tmp_path: Path) -> None:
    pytest.importorskip("diskgraph")
    rows = _fixture_rows()
    schema = _schema()
    contract = _pragma_contract(schema)
    requests = _requests()
    memory_dataset = api.Dataset({name: api.InMemoryRows(values) for name, values in rows.items()})
    parquet = _write_parquet(rows, tmp_path / "parquet")
    artifact = _index_at(tmp_path / "index")

    def assert_equal(actual: object, expected: object) -> None:
        if isinstance(expected, torch.Tensor):
            assert isinstance(actual, torch.Tensor)
            torch.testing.assert_close(actual, expected)
        elif isinstance(expected, Mapping):
            assert isinstance(actual, Mapping)
            assert actual.keys() == expected.keys()
            for key in expected:
                assert_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            assert len(actual) == len(expected)
            for actual_item, expected_item in zip(actual, expected, strict=True):
                assert_equal(actual_item, expected_item)
        else:
            assert actual == expected

    with (
        api.prepare(contract, memory_dataset, api.InMemoryEngine()) as reference_plan,
        api.prepare(
            contract, parquet, api.DiskGraphEngine(index=artifact.path, num_threads=1)
        ) as diskgraph_plan,
    ):
        reference_task = reference_plan.materialize(requests, fit_split="train")
        diskgraph_task = diskgraph_plan.materialize(requests, fit_split="train")
        assert diskgraph_task.fitted_state == reference_task.fitted_state
        assert diskgraph_plan._source.last_temporal_profile["output_format"] == "model"
        expected = list(reference_task.batches(requests, seed=17, epoch=3))
        inspectable = list(diskgraph_task.batches(requests, seed=17, epoch=3))
        diskgraph_plan._source._adapt_temporal_result = lambda *_args: pytest.fail(
            "model output entered the inspectable Event/FieldValue adapter"
        )
        model = list(diskgraph_task.batches(requests, seed=17, epoch=3, output_format="model"))
        length_profile = diskgraph_plan._source.last_length_profile
        assert length_profile["native_stats"]["frames_counted"] == 1
        assert length_profile["native_stats"]["rowpack_timestamps"] == 0
        assert length_profile["native_stats"]["sort_key_timestamps"] > 0
        assert set(length_profile["native_timings_ns"]) == {"traversal", "time_filter", "total"}
    assert len(expected) == len(inspectable) == len(model)
    for expected_batch, inspectable_batch, model_batch in zip(
        expected, inspectable, model, strict=True
    ):
        assert_equal(inspectable_batch.inputs, expected_batch.inputs)
        assert_equal(inspectable_batch.targets, expected_batch.targets)
        assert_equal(model_batch.inputs, expected_batch.inputs)
        assert_equal(model_batch.targets, expected_batch.targets)


def test_diskgraph_model_uses_shared_anchor_shuffle(tmp_path: Path) -> None:
    pytest.importorskip("diskgraph")
    schema = _schema()
    contract = _temporal_contract(schema)
    requests = tuple(
        api.SampleRequest(f"shuffled-{index}", 2, datetime(2024, 12, 1), split="train")
        for index in range(6)
    )
    dataset = _write_parquet(_fixture_rows(), tmp_path / "parquet")
    with api.prepare(
        contract, dataset, api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    ) as plan:
        task = plan.materialize(requests, fit_split="train")
        expected_ids = tuple(
            requests[index].sample_id
            for index in api.shard_positions(len(requests), seed=0, epoch=2, shuffle=True)
        )
        for output_format in ("inspectable", "model"):
            actual_ids = tuple(
                sample_id
                for batch in task.batches(
                    requests, output_format=output_format, seed=0, epoch=2, shuffle=True
                )
                for sample_id in batch.metadata["sample_ids"]
            )
            assert actual_ids == expected_ids
            direct_ids = tuple(
                sample_id
                for batch in task._assembler.batches(
                    requests, output_format=output_format, seed=0, epoch=2, shuffle=True
                )
                for sample_id in batch.metadata["sample_ids"]
            )
            assert direct_ids == expected_ids


def _anchor_parquet(path: Path, rows: list[dict[str, object]]) -> api.AnchorTable:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path / "anchors.parquet")
    return api.AnchorTable(
        source=api.Parquet(str(path)),
        entity_id="user_id",
        as_of_time="as_of",
        sample_id="sample_id",
        split="split",
        targets={"is_fraud": "is_fraud"},
    )


def test_diskgraph_anchor_check_validates_every_parquet_file(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.table({"user_id": [1], "as_of": [0]}), tmp_path / "part-0.parquet")
    pq.write_table(pa.table({"user_id": [2]}), tmp_path / "part-1.parquet")
    anchors = api.AnchorTable(
        source=api.Parquet(str(tmp_path)), entity_id="user_id", as_of_time="as_of"
    )
    from kumo.pytorch.engines.parquet_anchors import check_parquet_anchors

    assert check_parquet_anchors(anchors)[0].path == "anchors.columns"


def test_diskgraph_reads_parquet_anchor_tables_with_external_labels(tmp_path: Path) -> None:
    """G1 + G3: a Parquet AnchorTable with splits and an upstream label column feeds the
    model path, matching the same anchors passed as requests, on both output formats."""
    pytest.importorskip("diskgraph")
    from dataclasses import replace

    from tests.test_preview import _equal as assert_tree_equal

    rows, schema = (_fixture_rows(), _schema())
    parquet = _write_parquet(rows, tmp_path / "parquet")
    base = _pragma_contract(schema)
    contract = replace(
        base,
        sample=replace(
            base.sample, targets={"is_fraud": api.ExternalTarget(dtype=api.DType("float64"))}
        ),
        batch=replace(
            base.batch,
            targets={"is_fraud": api.LabelTarget(source="is_fraud", encoding=api.Raw())},
            batching=api.Batching(max_samples=2),
        ),
    )
    cutoff = datetime.fromisoformat("2024-12-01T00:00:00")
    anchor_rows = [
        {"sample_id": "a", "user_id": 2, "as_of": cutoff, "split": "train", "is_fraud": 1.0},
        {"sample_id": "b", "user_id": 3, "as_of": cutoff, "split": "train", "is_fraud": 0.0},
        {"sample_id": "c", "user_id": 3, "as_of": cutoff, "split": "test", "is_fraud": 1.0},
    ]
    anchors = _anchor_parquet(tmp_path / "anchors", anchor_rows)
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    assert not api.check(contract, parquet, engine, anchors=anchors)
    requests = tuple(
        api.SampleRequest(
            row["sample_id"],
            row["user_id"],
            row["as_of"],
            targets=(api.TargetValue("is_fraud", row["is_fraud"]),),
            split=row["split"],
        )
        for row in anchor_rows
    )
    with api.prepare(contract, parquet, engine) as plan:
        from_table = plan.materialize(anchors, fit_split="train")
        from_requests = plan.materialize(requests, fit_split="train")
        assert from_table.fitted_state == from_requests.fitted_state
        assert from_table.count("train") == 2 and from_table.count("test") == 1
        for output_format in ("inspectable", "model"):
            got = list(from_table.batches(split="train", output_format=output_format))
            want = list(
                from_requests.batches(
                    [r for r in requests if r.split == "train"], output_format=output_format
                )
            )
            assert len(got) == len(want) == 1
            assert_tree_equal(got[0].inputs, want[0].inputs)
            assert got[0].targets["is_fraud"]["values"].tolist() == [1.0, 0.0]
            test_batch = next(iter(from_table.batches(split="test", output_format=output_format)))
            assert test_batch.targets["is_fraud"]["values"].tolist() == [1.0]
