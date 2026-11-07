# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Four model-specific sample fixtures over the synthetic relational dataset.

The physical tables and relationships live in ``data_schema.py``. These recipes
decide which entity a sample describes, which rows become ordered events, and
which raw fields each sample contains.
"""

from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import (
    ActiveInterval,
    Aggregate,
    CalendarPart,
    Constant,
    DatedEntityValue,
    Derived,
    EntitySpec,
    EventSpec,
    EventTimeDelta,
    Field,
    MapValues,
    RowFilter,
    Rows,
    SampleSpec,
    TimeWindow,
)


def _column(name: str) -> tuple[str, Field]:
    return name, Field(column=name)


def _event(
    source: str,
    relation: str,
    time_column: str,
    fields: tuple[tuple[str, Field], ...],
) -> EventSpec:
    return EventSpec(
        Rows(path=(relation,), time=TimeWindow(time_column)),
        fields={
            "event_source": Constant(value=source, dtype=DType("string")),
            **dict(fields),
            "time": Field(column=time_column),
            "hour": Derived(
                inputs=("time",), operation=CalendarPart(part="hour"), dtype=DType("int64")
            ),
            "weekday": Derived(
                inputs=("time",), operation=CalendarPart(part="weekday"), dtype=DType("int64")
            ),
            "day": Derived(
                inputs=("time",), operation=CalendarPart(part="day"), dtype=DType("int64")
            ),
            "month": Derived(
                inputs=("time",), operation=CalendarPart(part="month"), dtype=DType("int64")
            ),
            "age_seconds": EventTimeDelta(unit="seconds", dtype=DType("float64")),
        },
    )


_USER_ENTITY_VALUES = dict(
    (
        _column("country"),
        _column("region"),
        _column("home_currency"),
        _column("income_band"),
        _column("risk_segment"),
        _column("acquisition_channel"),
        _column("birth_year"),
    )
)

_ACTIVE_ACCOUNT_COUNT = Aggregate(
    Rows(
        path=("user_accounts",),
        active=ActiveInterval("opened_at", "closed_at"),
    ),
    "count",
    dtype=DType("int64"),
)

_USER_DATED_VALUES = {
    "lifelong_event": DatedEntityValue(
        Rows(time=TimeWindow("signup_at")),
        Constant(value="signup", dtype=DType("string")),
        fields={"age_seconds": EventTimeDelta(unit="seconds", dtype=DType("float64"))},
    ),
    "lifelong_product": DatedEntityValue(
        Rows(path=("user_accounts",), time=TimeWindow("opened_at")),
        Field(column="account_type"),
        first_per_value=True,
        fields={"age_seconds": EventTimeDelta(unit="seconds", dtype=DType("float64"))},
    ),
}

_BANKING_EVENT = _event(
    "banking_transaction",
    "user_banking_transactions",
    "occurred_at",
    (
        _column("transaction_type"),
        _column("amount"),
        _column("currency"),
        _column("channel"),
        _column("counterparty_type"),
        _column("description"),
        _column("status"),
    ),
)

_CARD_FIELDS = (
    _column("amount"),
    _column("currency"),
    _column("mcc"),
    _column("channel"),
    _column("entry_mode"),
    _column("country"),
    _column("description"),
    _column("status"),
    _column("recurring_candidate_flag"),
)

_MERCHANT_RISK = (
    "merchant_risk",
    Field(
        column="risk_score",
        path=("transaction_merchant",),
    ),
)


PRAGMA_SAMPLE = SampleSpec(
    entity=EntitySpec("users", entity_type="user", available_from="signup_at"),
    entity_values={**_USER_ENTITY_VALUES, "active_account_count": _ACTIVE_ACCOUNT_COUNT},
    dated_entity_values=_USER_DATED_VALUES,
    events={
        "banking_transaction": _BANKING_EVENT,
        "transfer": _event(
            "transfer",
            "user_transfers",
            "occurred_at",
            (
                _column("amount"),
                _column("currency"),
                _column("transfer_type"),
                _column("counterparty_type"),
                _column("channel"),
                _column("status"),
            ),
        ),
        "card_transaction": _event(
            "card_transaction",
            "user_card_transactions",
            "occurred_at",
            (*_CARD_FIELDS, _MERCHANT_RISK),
        ),
        "card_payment": _event(
            "card_payment",
            "user_card_payments",
            "occurred_at",
            (
                _column("amount"),
                _column("currency"),
                _column("payment_type"),
                _column("status"),
            ),
        ),
        "card_dispute": _event(
            "card_dispute",
            "user_card_disputes",
            "opened_at",
            (
                _column("reason"),
                _column("outcome"),
            ),
        ),
        "investment_transaction": _event(
            "investment_transaction",
            "user_investment_transactions",
            "occurred_at",
            (
                _column("asset_class"),
                _column("side"),
                _column("gross_amount"),
                _column("currency"),
                _column("channel"),
                _column("status"),
            ),
        ),
        "communication": _event(
            "communication",
            "user_communications",
            "sent_at",
            (
                _column("product_id"),
                _column("channel"),
                _column("status"),
            ),
        ),
        "product_adoption": _event(
            "product_adoption",
            "user_product_adoptions",
            "adopted_at",
            (_column("product_id"),),
        ),
    },
)


NUFORMER_SAMPLE = SampleSpec(
    entity=EntitySpec("users", entity_type="user", available_from="signup_at"),
    events={
        "banking_transaction": EventSpec(
            rows=_BANKING_EVENT.rows,
            fields={
                **_BANKING_EVENT.fields,
                "absolute_amount": Derived(
                    inputs=("amount",), operation="abs", dtype=DType("float64")
                ),
                "direction": Derived(
                    inputs=("transaction_type",),
                    operation=MapValues(
                        groups={
                            "inflow": ("deposit", "interest"),
                            "outflow": ("withdrawal", "fee"),
                        },
                        unknown="error",
                    ),
                    dtype=DType("string"),
                ),
            },
        )
    },
)


PANTHER_SAMPLE = SampleSpec(
    entity=EntitySpec("users", entity_type="user", available_from="signup_at"),
    entity_values=dict((_column("country"),)),
    events={
        "card_transaction": _event(
            "card_transaction",
            "user_card_transactions",
            "occurred_at",
            (
                _column("channel"),
                _column("amount"),
                _column("mcc"),
                _MERCHANT_RISK,
            ),
        ),
    },
)


TREASURE_SAMPLE = SampleSpec(
    entity=EntitySpec(
        "accounts",
        entity_type="card_account",
        filters=(RowFilter("account_type", "equal", "credit_card"),),
        available_from="opened_at",
    ),
    entity_values={
        "country": Field(
            column="country",
            path=("user_accounts",),
        ),
        "active_account_count": Aggregate(
            Rows(
                path=("user_accounts", "user_accounts"),
                active=ActiveInterval("opened_at", "closed_at"),
            ),
            "count",
            dtype=DType("int64"),
        ),
    },
    events={
        "card_transaction": EventSpec(
            Rows(
                path=("account_card_transactions",),
                time=TimeWindow("occurred_at"),
            ),
            fields=dict(
                (
                    _column("amount"),
                    _column("currency"),
                    _column("mcc"),
                    _column("channel"),
                    _column("entry_mode"),
                    _column("country"),
                    _column("recurring_candidate_flag"),
                    _column("status"),
                )
            ),
        ),
    },
)
