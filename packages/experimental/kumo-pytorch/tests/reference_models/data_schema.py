# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical table and relationship fixture for the synthetic dataset.

This schema says what data exists. It does not decide which rows are model
events, which entity a sample is about, or which fields a model consumes.
Those choices belong to the sample definitions.
"""

from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.schema import NON_TEMPORAL, DataSchema, RelationSpec, TableSpec

SYNTHETIC_SCHEMA = DataSchema(
    tables=(
        TableSpec(
            "users",
            "user_id",
            {
                "user_id": DType(name="int64"),
                "signup_at": DType(name="timestamp", unit="s", timezone="UTC"),
                "country": DType(name="string"),
                "region": DType(name="string"),
                "home_currency": DType(name="string"),
                "income_band": DType(name="string"),
                "risk_segment": DType(name="string"),
                "acquisition_channel": DType(name="string"),
                "birth_year": DType(name="int64"),
            },
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "products",
            "product_id",
            {"product_id": DType(name="int64"), "product_type": DType(name="string")},
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "accounts",
            "account_id",
            {
                "account_id": DType(name="int64"),
                "user_id": DType(name="int64"),
                "product_id": DType(name="int64"),
                "account_type": DType(name="string"),
                "opened_at": DType(name="timestamp", unit="s", timezone="UTC"),
                "closed_at": DType(name="timestamp", unit="s", timezone="UTC"),
            },
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "merchants",
            "merchant_id",
            {"merchant_id": DType(name="int64"), "risk_score": DType(name="float64")},
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "devices",
            "device_id",
            {"device_id": DType(name="int64"), "user_id": DType(name="int64")},
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "banking_transactions",
            "transaction_id",
            {
                "transaction_id": DType(name="int64"),
                "user_id": DType(name="int64"),
                "account_id": DType(name="int64"),
                "occurred_at": DType(name="timestamp", unit="s", timezone="UTC"),
                "transaction_type": DType(name="string"),
                "amount": DType(name="float64"),
                "currency": DType(name="string"),
                "channel": DType(name="string"),
                "counterparty_type": DType(name="string"),
                "description": DType(name="string"),
                "status": DType(name="string"),
            },
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "transfers",
            "transfer_id",
            {
                "transfer_id": DType(name="int64"),
                "user_id": DType(name="int64"),
                "occurred_at": DType(name="timestamp", unit="s", timezone="UTC"),
                "amount": DType(name="float64"),
                "currency": DType(name="string"),
                "transfer_type": DType(name="string"),
                "counterparty_type": DType(name="string"),
                "channel": DType(name="string"),
                "status": DType(name="string"),
            },
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "card_transactions",
            "card_transaction_id",
            {
                "card_transaction_id": DType(name="int64"),
                "user_id": DType(name="int64"),
                "account_id": DType(name="int64"),
                "occurred_at": DType(name="timestamp", unit="s", timezone="UTC"),
                "merchant_id": DType(name="int64"),
                "amount": DType(name="float64"),
                "currency": DType(name="string"),
                "mcc": DType(name="string"),
                "channel": DType(name="string"),
                "entry_mode": DType(name="string"),
                "country": DType(name="string"),
                "description": DType(name="string"),
                "status": DType(name="string"),
                "recurring_candidate_flag": DType(name="bool"),
            },
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "card_payments",
            "payment_id",
            {
                "payment_id": DType(name="int64"),
                "user_id": DType(name="int64"),
                "occurred_at": DType(name="timestamp", unit="s", timezone="UTC"),
                "amount": DType(name="float64"),
                "currency": DType(name="string"),
                "payment_type": DType(name="string"),
                "status": DType(name="string"),
            },
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "card_disputes",
            "dispute_id",
            {
                "dispute_id": DType(name="int64"),
                "user_id": DType(name="int64"),
                "card_transaction_id": DType(name="int64"),
                "opened_at": DType(name="timestamp", unit="s", timezone="UTC"),
                "reason": DType(name="string"),
                "outcome": DType(name="string"),
            },
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "investment_transactions",
            "investment_transaction_id",
            {
                "investment_transaction_id": DType(name="int64"),
                "user_id": DType(name="int64"),
                "occurred_at": DType(name="timestamp", unit="s", timezone="UTC"),
                "asset_class": DType(name="string"),
                "side": DType(name="string"),
                "gross_amount": DType(name="float64"),
                "currency": DType(name="string"),
                "channel": DType(name="string"),
                "status": DType(name="string"),
            },
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "communications",
            "communication_id",
            {
                "communication_id": DType(name="int64"),
                "user_id": DType(name="int64"),
                "sent_at": DType(name="timestamp", unit="s", timezone="UTC"),
                "product_id": DType(name="int64"),
                "channel": DType(name="string"),
                "status": DType(name="string"),
            },
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "product_adoptions",
            "adoption_id",
            {
                "adoption_id": DType(name="int64"),
                "user_id": DType(name="int64"),
                "adopted_at": DType(name="timestamp", unit="s", timezone="UTC"),
                "product_id": DType(name="int64"),
            },
            time_column=NON_TEMPORAL,
        ),
    ),
    relations=(
        RelationSpec("user_accounts", "users", "user_id", "accounts", "user_id"),
        RelationSpec("account_product", "accounts", "product_id", "products", "product_id"),
        RelationSpec("user_devices", "users", "user_id", "devices", "user_id"),
        RelationSpec(
            "user_banking_transactions", "users", "user_id", "banking_transactions", "user_id"
        ),
        RelationSpec(
            "account_banking_transactions",
            "accounts",
            "account_id",
            "banking_transactions",
            "account_id",
        ),
        RelationSpec("user_transfers", "users", "user_id", "transfers", "user_id"),
        RelationSpec("user_card_transactions", "users", "user_id", "card_transactions", "user_id"),
        RelationSpec(
            "account_card_transactions", "accounts", "account_id", "card_transactions", "account_id"
        ),
        RelationSpec(
            "transaction_merchant", "card_transactions", "merchant_id", "merchants", "merchant_id"
        ),
        RelationSpec("user_card_payments", "users", "user_id", "card_payments", "user_id"),
        RelationSpec("user_card_disputes", "users", "user_id", "card_disputes", "user_id"),
        RelationSpec(
            "transaction_disputes",
            "card_transactions",
            "card_transaction_id",
            "card_disputes",
            "card_transaction_id",
        ),
        RelationSpec(
            "user_investment_transactions", "users", "user_id", "investment_transactions", "user_id"
        ),
        RelationSpec("user_communications", "users", "user_id", "communications", "user_id"),
        RelationSpec("user_product_adoptions", "users", "user_id", "product_adoptions", "user_id"),
    ),
)
