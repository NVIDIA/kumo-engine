# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object references lower to exactly the same typed contract as explicit declarations."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kumo.pytorch import build
from kumo.pytorch.api import (
    NON_TEMPORAL,
    Aggregate,
    BatchSpec,
    Buckets,
    DataLoaderSpec,
    DataSchema,
    DType,
    Encode,
    EntitySource,
    EntitySpec,
    EventSource,
    EventSpec,
    Field,
    Fields,
    Graph,
    GraphLayer,
    GraphSample,
    IdSpace,
    Input,
    NeighborhoodSource,
    NeighborhoodSpec,
    Nested,
    Packed,
    Padded,
    Raw,
    RelationSpec,
    Rows,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    TimeWindow,
    Tokens,
    Vocabulary,
    ZScore,
)
from tests.helpers import Plans

I64, F64, STRING = DType("int64"), DType("float64"), DType("string")
TIME = DType("timestamp", unit="s", timezone="UTC")
users = build.Table(
    name="users", key="user_id", columns={"user_id": I64, "country": STRING}, time=NON_TEMPORAL
)
accounts = build.Table(
    name="accounts",
    key="account_id",
    columns={"account_id": I64, "user_id": I64, "balance": F64},
    time=NON_TEMPORAL,
)
transactions = build.Table(
    name="transactions",
    key="transaction_id",
    columns={
        "transaction_id": I64,
        "account_id": I64,
        "occurred_at": TIME,
        "amount": F64,
    },
    time="occurred_at",
)
user_accounts = build.Link(name="user_accounts", left=users.user_id, right=accounts.user_id)
account_transactions = build.Link(
    name="account_transactions", left=accounts.account_id, right=transactions.account_id
)
SCHEMA = build.Schema(
    tables=(users, accounts, transactions), links=(user_accounts, account_transactions)
)
history = build.Events(
    via=(user_accounts, account_transactions),
    fields={"amount": build.Read(column=transactions.amount)},
    limit=16,
)
activity = build.Neighborhood(
    layers=(
        build.Hop(link=user_accounts, fanout=2, strategy="uniform", temporal="static"),
        build.Hop(link=account_transactions, fanout=1, strategy="recent", time="occurred_at"),
    )
)
SAMPLE = build.Sample(
    entity=users,
    values={
        "country": build.Read(column=users.country),
        "spend": build.Sum(
            column=transactions.amount, via=(user_accounts, account_transactions), dtype=F64
        ),
        "txn_count": build.Count(via=(user_accounts, account_transactions), dtype=I64),
    },
    events={"history": history},
    neighbors={"activity": activity},
)
GRAPH = Graph(
    node_fields={
        "users": Fields(
            fields=(Encode(field="country", encoding=Vocabulary(values=(("US", 2),))),)
        ),
        "accounts": Fields(fields=(Encode(field="balance", encoding=ZScore()),)),
    }
)
BATCH = build.Batch(
    inputs={"graph": build.Input(source=activity, representation=GRAPH, layout=None)}
)
PATH = ("user_accounts", "account_transactions")
WINDOW = TimeWindow(column="occurred_at")
EXPECTED = DataLoaderSpec(
    schema=DataSchema(
        tables=(
            TableSpec(
                name="users",
                primary_key="user_id",
                columns={"user_id": I64, "country": STRING},
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                name="accounts",
                primary_key="account_id",
                columns={"account_id": I64, "user_id": I64, "balance": F64},
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                name="transactions",
                primary_key="transaction_id",
                columns={
                    "transaction_id": I64,
                    "account_id": I64,
                    "occurred_at": TIME,
                    "amount": F64,
                },
                time_column="occurred_at",
            ),
        ),
        relations=(
            RelationSpec(
                name="user_accounts",
                left_table="users",
                left_key="user_id",
                right_table="accounts",
                right_key="user_id",
            ),
            RelationSpec(
                name="account_transactions",
                left_table="accounts",
                left_key="account_id",
                right_table="transactions",
                right_key="account_id",
            ),
        ),
    ),
    sample=SampleSpec(
        entity=EntitySpec(table="users"),
        entity_values={
            "country": Field(column="country"),
            "spend": Aggregate(
                source=Rows(path=PATH, time=WINDOW), operation="sum", column="amount", dtype=F64
            ),
            "txn_count": Aggregate(
                source=Rows(path=PATH, time=WINDOW), operation="count", dtype=I64
            ),
        },
        events={
            "history": EventSpec(
                rows=Rows(path=PATH, time=WINDOW),
                fields={"amount": Field(column="amount")},
                limit=16,
            )
        },
        neighborhoods={
            "activity": NeighborhoodSpec(
                layers=(
                    GraphLayer(
                        samples=(
                            GraphSample(
                                relation="user_accounts",
                                fanout=2,
                                strategy="uniform",
                                temporal="static",
                            ),
                        )
                    ),
                    GraphLayer(
                        samples=(
                            GraphSample(
                                relation="account_transactions",
                                fanout=1,
                                strategy="recent",
                                time_column="occurred_at",
                            ),
                        )
                    ),
                )
            )
        },
    ),
    batch=BatchSpec(
        inputs={
            "graph": Input(
                source=NeighborhoodSource(name="activity"), representation=GRAPH, layout=None
            )
        }
    ),
)


def test_built_contract_is_the_identical_artifact():
    actual = build.contract(schema=SCHEMA, sample=SAMPLE, batch=BATCH)
    assert actual == EXPECTED
    assert actual.fingerprint == EXPECTED.fingerprint
    actual.validate()
    assert DataLoaderSpec.from_json(actual.to_json()) == EXPECTED


def test_windows_use_the_destination_clock_or_explicit_bounds():
    assert SAMPLE.lower(SCHEMA).entity_values["spend"].source.time == WINDOW
    count = build.Sample(entity=users, values={"count": build.Count(via=user_accounts, dtype=I64)})
    assert count.lower(SCHEMA).entity_values["count"].source.time is None
    recent = build.Sample(
        entity=users,
        values={
            "sum": build.Sum(
                column=transactions.amount,
                via=(user_accounts, account_transactions),
                since=-86_400,
                dtype=F64,
            )
        },
    )
    assert recent.lower(SCHEMA).entity_values["sum"].source.time == TimeWindow(
        "occurred_at", -86_400, 0
    )


@pytest.mark.parametrize(
    ("name", "operation"),
    (
        ("Count", "count"),
        ("CountDistinct", "count_distinct"),
        ("Sum", "sum"),
        ("Mean", "mean"),
        ("Min", "min"),
        ("Max", "max"),
        ("Mode", "mode"),
        ("StdPopulation", "std_population"),
        ("StdSample", "std_sample"),
        ("VariancePopulation", "var_population"),
        ("VarianceSample", "var_sample"),
        ("Recency", "recency"),
    ),
)
def test_each_reduction_uses_explicit_result_type(name, operation):
    dtype = I64 if name in {"Count", "CountDistinct"} else F64
    kwargs = {"via": (user_accounts, account_transactions), "dtype": dtype}
    if name != "Count":
        kwargs["column"] = transactions.occurred_at if name == "Recency" else transactions.amount
    value = getattr(build, name)(**kwargs)
    result = (
        build.Sample(entity=users, values={"value": value}).lower(SCHEMA).entity_values["value"]
    )
    assert result.operation == operation
    assert result.dtype == dtype


def test_read_field_type_comes_only_from_table():
    assert users.country.dtype == STRING
    assert SAMPLE.lower(SCHEMA).entity_values["country"] == Field(column="country")
    with pytest.raises(AttributeError, match="country_typo"):
        users.country_typo
    odd = build.Table(name="odd", key="id", columns={"id": I64, "name": STRING}, time=NON_TEMPORAL)
    assert odd["name"] == build.Column(table="odd", name="name", dtype=STRING)


def test_undeclared_or_disconnected_links_are_refused():
    other = build.Table(name="other", key="id", columns={"id": I64}, time=NON_TEMPORAL)
    with pytest.raises(ValueError):
        build.Schema(
            tables=(users,), links=(build.Link(name="x", left=users.user_id, right=other.id),)
        )
    with pytest.raises(ValueError, match="connect"):
        build.Sample(
            entity=users, values={"bad": build.Count(via=account_transactions, dtype=I64)}
        ).lower(SCHEMA)


def test_events_need_a_declared_destination_clock():
    notes = build.Table(
        name="notes", key="id", columns={"id": I64, "user_id": I64}, time=NON_TEMPORAL
    )
    link = build.Link(name="notes", left=users.user_id, right=notes.user_id)
    schema = build.Schema(tables=(users, notes), links=(link,))
    with pytest.raises(ValueError, match="clock"):
        build.Sample(
            entity=users,
            events={"notes": build.Events(via=link, fields={"id": build.Read(column=notes.id)})},
        ).lower(schema)


def test_structural_equality_does_not_substitute_an_undeclared_component():
    twin = replace(history)
    assert twin == history
    batch = build.Batch(
        inputs={
            "history": build.Input(
                source=twin,
                representation=Fields(fields=(Encode(field="amount", encoding=Raw()),)),
                layout=Padded(),
            )
        }
    )
    with pytest.raises(ValueError, match="declared"):
        build.contract(schema=SCHEMA, sample=SAMPLE, batch=batch)


def test_object_source_resolves_to_the_declared_event_name():
    batch = build.Batch(
        inputs={
            "history": build.Input(
                source=history,
                representation=Fields(fields=(Encode(field="amount", encoding=Raw()),)),
                layout=Padded(),
            )
        }
    )
    actual = build.contract(schema=SCHEMA, sample=SAMPLE, batch=batch)
    assert actual.batch.inputs["history"].source == EventSource(events=("history",))


def test_undeclared_neighborhood_is_refused():
    orphan = build.Neighborhood(layers=(build.Hop(link=user_accounts, fanout=2),))
    with pytest.raises(ValueError, match="declared"):
        build.contract(
            schema=SCHEMA,
            sample=SAMPLE,
            batch=build.Batch(
                inputs={
                    "graph": build.Input(source=orphan, representation=Graph(), layout=None),
                }
            ),
        )


def test_common_encodings_and_layouts_are_not_redeclared_by_the_builder():
    fields = Fields(fields=(Encode(field="country", encoding=Vocabulary(values=(("US", 2),))),))
    actual = build.contract(
        schema=SCHEMA,
        sample=SAMPLE,
        batch=build.Batch(
            inputs={
                "profile": build.Input(source=users, representation=fields, layout=Separate()),
            }
        ),
    )
    assert actual.batch == BatchSpec(
        inputs={"profile": Input(source=EntitySource(), representation=fields, layout=Separate())}
    )


def test_explicit_output_order_is_independent_of_sample_declaration_order():
    contracts = []
    for values in (
        {"z": build.Read(column=transactions.amount), "a": build.Read(column=transactions.amount)},
        {"a": build.Read(column=transactions.amount), "z": build.Read(column=transactions.amount)},
    ):
        event = replace(history, fields=values)
        sample = build.Sample(entity=users, events={"history": event})
        batch = build.Batch(
            id_spaces={"model": IdSpace(specials={"pad": "[PAD]"})},
            inputs={
                "history": build.Input(
                    source=event,
                    representation=Fields(
                        fields=(
                            Encode(field="z", encoding=Buckets(boundaries=(0,)), id_space="model"),
                            Encode(field="a", encoding=Buckets(boundaries=(0,)), id_space="model"),
                        ),
                        row=Tokens(max_tokens=8),
                    ),
                    layout=Nested(fields=Packed(), events=Padded()),
                )
            },
        )
        contracts.append(build.contract(schema=SCHEMA, sample=sample, batch=batch))
    assert contracts[0] == contracts[1]
    assert contracts[0].fingerprint == contracts[1].fingerprint


def test_built_graph_contract_runs():
    tables = {
        "users": [{"user_id": 1, "country": "US"}],
        "accounts": [{"account_id": 10, "user_id": 1, "balance": 10.0}],
        "transactions": [
            {"transaction_id": 100, "account_id": 10, "occurred_at": 10, "amount": 5.0}
        ],
    }
    contract = build.contract(schema=SCHEMA, sample=SAMPLE, batch=BATCH)
    requests = (SampleRequest(sample_id="u1", entity_id=1, as_of_time=30),)
    with Plans(contract.schema, tables).prepare(contract, requests) as task:
        batch = next(task.batches(requests, batch_size=1, seed=17))
        assert batch.inputs["graph"]["nodes"]["users"]["features"]["country"][
            "values"
        ].tolist() == [2]
