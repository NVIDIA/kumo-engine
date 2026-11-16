# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt

import pytest

from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import (
    ActiveInterval,
    Aggregate,
    Compare,
    Constant,
    DatedEntityValue,
    EntitySpec,
    EventSpec,
    ExternalTarget,
    Field,
    RowFilter,
    Rows,
    SampleRequest,
    SampleSpec,
    TargetValue,
    TimeWindow,
)
from kumo.pytorch.contract.schema import NON_TEMPORAL, DataSchema, RelationSpec, TableSpec
from kumo.pytorch.engines.in_memory.samples import InMemorySampleBuilder


def _field(history, key: str):
    return next(field for field in history.entity.values if field.key == key)


def _event_field(event, key: str):
    return next((field for field in event.fields if field.key == key), None)


def _sample_builder() -> InMemorySampleBuilder:
    schema = DataSchema(
        tables=(
            TableSpec(
                "users",
                "user_id",
                {
                    "user_id": DType("int64"),
                    "signup_at": DType("timestamp", unit="s", timezone="UTC"),
                    "country": DType("string"),
                },
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                "user_states",
                "state_id",
                {
                    "state_id": DType("int64"),
                    "user_id": DType("int64"),
                    "recorded_at": DType("timestamp", unit="s", timezone="UTC"),
                    "risk": DType("string"),
                },
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                "accounts",
                "account_id",
                {
                    "account_id": DType("int64"),
                    "user_id": DType("int64"),
                    "account_type": DType("string"),
                    "opened_at": DType("timestamp", unit="s", timezone="UTC"),
                    "closed_at": DType("timestamp", unit="s", timezone="UTC"),
                },
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                "card_transactions",
                "transaction_id",
                {
                    "transaction_id": DType("int64"),
                    "user_id": DType("int64"),
                    "account_id": DType("int64"),
                    "merchant_id": DType("int64"),
                    "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
                    "amount": DType("float64"),
                },
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                "merchants",
                "merchant_id",
                {"merchant_id": DType("int64"), "name": DType("string")},
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                "merchant_risks",
                "risk_id",
                {
                    "risk_id": DType("int64"),
                    "merchant_id": DType("int64"),
                    "recorded_at": DType("timestamp", unit="s", timezone="UTC"),
                    "risk": DType("string"),
                },
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                "transfers",
                "transfer_id",
                {
                    "transfer_id": DType("int64"),
                    "user_id": DType("int64"),
                    "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
                    "amount": DType("float64"),
                },
                time_column=NON_TEMPORAL,
            ),
        ),
        relations=(
            RelationSpec("user_states", "users", "user_id", "user_states", "user_id"),
            RelationSpec("user_accounts", "users", "user_id", "accounts", "user_id"),
            RelationSpec(
                "user_card_transactions", "users", "user_id", "card_transactions", "user_id"
            ),
            RelationSpec(
                "transaction_merchant",
                "card_transactions",
                "merchant_id",
                "merchants",
                "merchant_id",
            ),
            RelationSpec(
                "merchant_risks", "merchants", "merchant_id", "merchant_risks", "merchant_id"
            ),
            RelationSpec("user_transfers", "users", "user_id", "transfers", "user_id"),
        ),
    )
    tables = {
        "users": [
            {"user_id": 1, "signup_at": 10, "country": "GB"},
            {"user_id": 2, "signup_at": 10, "country": "US"},
        ],
        "user_states": [
            {"state_id": 1, "user_id": 1, "recorded_at": 50, "risk": "bronze"},
            {"state_id": 2, "user_id": 1, "recorded_at": 99, "risk": "silver"},
            {"state_id": 3, "user_id": 1, "recorded_at": 100, "risk": "gold"},
            {"state_id": 4, "user_id": 1, "recorded_at": 105, "risk": "platinum"},
        ],
        "accounts": [
            {
                "account_id": 10,
                "user_id": 1,
                "account_type": "checking",
                "opened_at": 20,
                "closed_at": None,
            },
            {
                "account_id": 11,
                "user_id": 1,
                "account_type": "credit_card",
                "opened_at": 30,
                "closed_at": 95,
            },
        ],
        "card_transactions": [
            {
                "transaction_id": 102,
                "user_id": 1,
                "account_id": 11,
                "merchant_id": 50,
                "occurred_at": 90,
                "amount": 20.0,
            },
            {
                "transaction_id": 101,
                "user_id": 1,
                "account_id": 11,
                "merchant_id": 50,
                "occurred_at": 90,
                "amount": 10.0,
            },
            {
                "transaction_id": 103,
                "user_id": 1,
                "account_id": 11,
                "merchant_id": 50,
                "occurred_at": 100,
                "amount": 30.0,
            },
            {
                "transaction_id": 104,
                "user_id": 1,
                "account_id": 11,
                "merchant_id": 50,
                "occurred_at": 110,
                "amount": 40.0,
            },
        ],
        "merchants": [{"merchant_id": 50, "name": "Example Store"}],
        "merchant_risks": [
            {"risk_id": 1, "merchant_id": 50, "recorded_at": 80, "risk": "low"},
            {"risk_id": 2, "merchant_id": 50, "recorded_at": 95, "risk": "medium"},
            {"risk_id": 3, "merchant_id": 50, "recorded_at": 105, "risk": "high"},
        ],
        "transfers": [],
    }
    return InMemorySampleBuilder(schema, tables)


def _recipe() -> SampleSpec:
    card_rows = Rows(
        path=("user_card_transactions",), time=TimeWindow(column="occurred_at", start=None, stop=0)
    )
    return SampleSpec(
        entity=EntitySpec("users", entity_type="customer"),
        entity_values={
            "country": Field(column="country"),
            "risk_segment": Field(
                column="risk", path=("user_states",), selection="latest", time_column="recorded_at"
            ),
            "transaction_count": Aggregate(card_rows, "count", dtype=DType("int64")),
            "active_account_count": Aggregate(
                Rows(path=("user_accounts",), active=ActiveInterval("opened_at", "closed_at")),
                "count",
                dtype=DType("int64"),
            ),
            "average_transfer_amount": Aggregate(
                Rows(
                    path=("user_transfers",),
                    time=TimeWindow(column="occurred_at", start=None, stop=0),
                ),
                "mean",
                column="amount",
                dtype=DType("float64"),
            ),
        },
        dated_entity_values={
            "milestone": DatedEntityValue(
                Rows(time=TimeWindow(column="signup_at", start=None, stop=0)),
                Constant(value="signup", dtype=DType("string")),
            ),
            "first_account_type": DatedEntityValue(
                Rows(
                    path=("user_accounts",), time=TimeWindow(column="opened_at", start=None, stop=0)
                ),
                Field(column="account_type"),
                first_per_value=True,
            ),
        },
        events={
            "card_transaction": EventSpec(
                card_rows,
                fields={
                    "event_source": Constant(value="card_transaction", dtype=DType("string")),
                    "transaction_id": Field(column="transaction_id"),
                    "amount": Field(column="amount"),
                    "merchant_risk": Field(
                        column="risk",
                        path=("transaction_merchant", "merchant_risks"),
                        selection="latest",
                        time_column="recorded_at",
                        reference_time="event",
                    ),
                },
            )
        },
        targets={"fraud": ExternalTarget(dtype=DType("int64"))},
    )


def test_sample_builder_builds_raw_point_in_time_history() -> None:
    request = SampleRequest(
        "user-1-at-100", entity_id=1, as_of_time=100, targets=(TargetValue("fraud", 1),)
    )
    sample = _sample_builder().build(_recipe(), [request])[0]
    history = sample.history
    assert sample.sample_id == "user-1-at-100"
    assert history.entity_id == 1
    assert history.entity_type == "customer"
    assert history.evaluation_time == 100
    assert _field(history, "country").value == "GB"
    assert _field(history, "risk_segment").value == "gold"
    assert _field(history, "transaction_count").value == 3.0
    assert _field(history, "active_account_count").value == 1.0
    assert not any(field.key == "average_transfer_amount" for field in history.entity.values)
    assert [_event_field(event, "transaction_id").value for event in history.events] == [
        101,
        102,
        103,
    ]
    assert [_event_field(event, "merchant_risk").value for event in history.events] == [
        "low",
        "low",
        "medium",
    ]
    assert all(event.timestamp <= 100 for event in history.events)
    milestone_values = {value.value for value in history.entity.dated_values}
    assert milestone_values == {"signup", "checking", "credit_card"}
    assert sample.targets == (TargetValue("fraud", 1),)
    assert not any(field.key == "fraud" for field in history.entity.values)
    assert not any(_event_field(event, "fraud") for event in history.events)


def test_same_entity_at_different_times_produces_independent_samples() -> None:
    samples = _sample_builder().build(
        _recipe(),
        [SampleRequest("before-boundary", 1, 100), SampleRequest("after-later-event", 1, 111)],
    )
    assert [len(sample.history.events) for sample in samples] == [3, 4]
    assert _field(samples[0].history, "risk_segment").value == "gold"
    assert _field(samples[1].history, "risk_segment").value == "platinum"
    assert samples[0].history is not samples[1].history


def test_context_recipe_cannot_request_future_rows() -> None:
    recipe = SampleSpec(
        entity=EntitySpec("users", entity_type="customer"),
        events={
            "card_transaction": EventSpec(
                Rows(
                    path=("user_card_transactions",),
                    time=TimeWindow(column="occurred_at", start=None, stop=1),
                ),
                fields={"amount": Field(column="amount")},
            )
        },
    )
    with pytest.raises(ValueError, match="context.*after"):
        _sample_builder().build(recipe, [SampleRequest("sample", 1, 100)])


def test_filters_and_lookback_are_configuration_not_engine_code() -> None:
    recipe = SampleSpec(
        entity=EntitySpec("users", entity_type="customer"),
        entity_values={
            "large_recent_transactions": Aggregate(
                Rows(
                    path=("user_card_transactions",),
                    filters=(RowFilter("amount", "greater_than", 15.0),),
                    time=TimeWindow(column="occurred_at", start=-15, stop=0),
                ),
                "count",
                dtype=DType("int64"),
            )
        },
    )
    sample = _sample_builder().build(recipe, [SampleRequest("sample", 1, 100)])[0]
    assert _field(sample.history, "large_recent_transactions").value == 2.0


def test_comparisons_emit_boolean_features_and_targets() -> None:
    historical = Rows(path=("user_card_transactions",), time=TimeWindow("occurred_at"))
    next_fifteen_seconds = Rows(
        path=("user_card_transactions",), time=TimeWindow("occurred_at", start=0, stop=15)
    )
    recipe = SampleSpec(
        entity=EntitySpec("users", entity_type="customer"),
        entity_values={
            "has_transacted": Compare(
                left=Aggregate(historical, "count", dtype=DType("int64")),
                operation="greater_than",
                right=0,
                dtype=DType("bool"),
            )
        },
        targets={
            "will_transact": Compare(
                left=Aggregate(next_fifteen_seconds, "count", dtype=DType("int64")),
                operation="greater_than",
                right=0,
                dtype=DType("bool"),
            )
        },
    )
    samples = _sample_builder().build(
        recipe, [SampleRequest("active", 1, 100), SampleRequest("inactive", 2, 100)]
    )
    assert _field(samples[0].history, "has_transacted").value is True
    assert _field(samples[0].history, "has_transacted").dtype == DType("bool")
    assert {target.name: target.value for target in samples[0].targets} == {"will_transact": True}
    assert _field(samples[1].history, "has_transacted").value is False
    assert {target.name: target.value for target in samples[1].targets} == {"will_transact": False}


def test_compare_rejects_unknown_operations() -> None:
    with pytest.raises(ValueError, match="unsupported comparison operation"):
        Compare(
            Aggregate(Rows(), "count", dtype=DType("int64")),
            "approximately",
            0,
            dtype=DType("bool"),
        )


@pytest.mark.parametrize(
    ("left", "operation", "right", "message"),
    (
        (
            Aggregate(Rows(), "count", dtype=DType("int64")),
            "greater_than",
            True,
            "not a valid",
        ),
        (
            Aggregate(Rows(), "mode", "channel", dtype=DType("bool")),
            "greater_than",
            True,
            "only supports",
        ),
        (
            Aggregate(Rows(), "mode", "channel", dtype=DType("string")),
            "equal",
            1,
            "not a valid",
        ),
        (
            Aggregate(Rows(), "min", "description", dtype=DType("string")),
            "equal",
            False,
            "not a valid",
        ),
    ),
)
def test_compare_rejects_incompatible_types(
    left: Aggregate, operation: str, right, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Compare(left, operation, right, dtype=DType("bool"))


@pytest.mark.parametrize(
    ("operation", "value"),
    (
        ("equal", 3),
        ("not_equal", 4),
        ("greater_than", 2),
        ("greater_or_equal", 3),
        ("less_than", 4),
        ("less_or_equal", 3),
    ),
)
def test_numerical_comparison_operations(operation: str, value: int) -> None:
    historical = Rows(path=("user_card_transactions",), time=TimeWindow("occurred_at"))
    recipe = SampleSpec(
        entity=EntitySpec("users", entity_type="customer"),
        entity_values={
            "matches": Compare(
                Aggregate(historical, "count", dtype=DType("int64")),
                operation,
                value,
                dtype=DType("bool"),
            )
        },
    )
    sample = _sample_builder().build(recipe, [SampleRequest("sample", 1, 100)])[0]
    assert _field(sample.history, "matches").value is True


def test_integer_mode_supports_typed_equality_comparison() -> None:
    historical = Rows(path=("user_card_transactions",), time=TimeWindow("occurred_at"))
    comparison = Compare(
        Aggregate(historical, "mode", "merchant_id", dtype=DType("int64")),
        "equal",
        50,
        dtype=DType("bool"),
    )
    recipe = SampleSpec(
        entity=EntitySpec("users", entity_type="customer"),
        entity_values={"uses_merchant_50": comparison},
    )
    sample = _sample_builder().build(recipe, [SampleRequest("sample", 1, 100)])[0]
    assert comparison.dtype == DType("bool")
    assert _field(sample.history, "uses_merchant_50").value is True


def test_missing_aggregate_remains_missing_after_comparison() -> None:
    recipe = SampleSpec(
        entity=EntitySpec("users", entity_type="customer"),
        entity_values={
            "has_positive_transfer_average": Compare(
                left=Aggregate(
                    Rows(
                        path=("user_transfers",),
                        time=TimeWindow(column="occurred_at", start=None, stop=0),
                    ),
                    "mean",
                    column="amount",
                    dtype=DType("float64"),
                ),
                operation="greater_than",
                right=0,
                dtype=DType("bool"),
            )
        },
    )
    sample = _sample_builder().build(recipe, [SampleRequest("sample", 1, 100)])[0]
    assert not any(
        field.key == "has_positive_transfer_average" for field in sample.history.entity.values
    )


def test_datetime_values_are_compared_as_seconds() -> None:
    cutoff = dt.datetime(1970, 1, 1, 0, 1, 40, tzinfo=dt.UTC)
    sample = _sample_builder().build(_recipe(), [SampleRequest("datetime-cutoff", 1, cutoff)])[0]
    assert sample.history.evaluation_time == 100
    assert len(sample.history.events) == 3
