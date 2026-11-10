# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from kumo.pytorch.api import (
    NON_TEMPORAL,
    Aggregate,
    Columns,
    DataLoaderSpec,
    DataSchema,
    Derived,
    DType,
    EntitySource,
    EntitySpec,
    EventSpec,
    Field,
    Input,
    Raw,
    RelationSpec,
    Rows,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    TimeWindow,
)
from kumo.pytorch.contract.batch import BatchSpec, Encode, Fields
from kumo.pytorch.engines.in_memory.samples import InMemorySampleBuilder


def test_entity_values_unify_fields_aggregates_and_derived_values() -> None:
    schema = DataSchema(
        tables=(
            TableSpec(
                "users",
                "user_id",
                {"user_id": DType("int64"), "country": DType("string")},
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                "transactions",
                "transaction_id",
                {
                    "transaction_id": DType("int64"),
                    "user_id": DType("int64"),
                    "at": DType("timestamp", unit="s", timezone="UTC"),
                    "amount": DType("float64"),
                },
                time_column="at",
            ),
        ),
        relations=(
            RelationSpec("user_transactions", "users", "user_id", "transactions", "user_id"),
        ),
    )
    historical = Rows(path=("user_transactions",), time=TimeWindow("at", stop=0))
    sample = SampleSpec(
        entity=EntitySpec("users", entity_type="user"),
        entity_values={
            "country": Field(column="country"),
            "transaction_count": Aggregate(historical, "count", dtype=DType("int64")),
            "total_spend": Aggregate(historical, "sum", column="amount", dtype=DType("float64")),
            "average_spend": Derived(
                "divide", inputs=("total_spend", "transaction_count"), dtype=DType("float64")
            ),
        },
        events={
            "transactions": EventSpec(rows=historical, fields={"amount": Field(column="amount")})
        },
    )
    contract = DataLoaderSpec(
        schema=schema,
        sample=sample,
        batch=BatchSpec(
            inputs={
                "entity": Input(
                    EntitySource(),
                    Fields(
                        fields=(
                            Encode(field="country", encoding=Raw()),
                            Encode(field="transaction_count", encoding=Raw()),
                            Encode(field="average_spend", encoding=Raw()),
                        ),
                        row=Columns(),
                    ),
                    Separate(),
                )
            }
        ),
        version=3,
    )
    contract.validate()
    built = InMemorySampleBuilder(
        schema,
        {
            "users": [{"user_id": 7, "country": "US"}],
            "transactions": [
                {"transaction_id": 1, "user_id": 7, "at": 10, "amount": 12.0},
                {"transaction_id": 2, "user_id": 7, "at": 20, "amount": 8.0},
            ],
        },
    ).build(sample, [SampleRequest("sample-7", entity_id=7, as_of_time=30)])[0]
    assert built.history.entity_id == 7
    assert built.history.entity_type == "user"
    assert {value.key: value.value for value in built.history.entity.values} == {
        "country": "US",
        "transaction_count": 2.0,
        "total_spend": 20.0,
        "average_spend": 10.0,
    }


def test_public_api_exposes_only_the_new_entity_contract_names() -> None:
    from kumo.pytorch import api

    assert hasattr(api, "EntitySpec")
    assert hasattr(api, "Input")
    assert hasattr(api, "EntitySource")
    assert hasattr(api, "Columns")
    assert hasattr(api, "Field")
    assert not hasattr(api, "EntityInput")
    assert not hasattr(api, "EventValues")
    assert not hasattr(api, "TransactionTokenSequence")
    assert not hasattr(api, "SubjectSpec")
    assert not hasattr(api, "EntityValues")
    assert not hasattr(api, "FieldSpec")
