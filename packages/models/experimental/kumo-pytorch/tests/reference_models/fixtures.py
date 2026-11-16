# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small relational fixtures exercising the four public model recipes."""

from contextlib import contextmanager
from dataclasses import replace

from kumo.pytorch.api import (
    DataLoaderSpec,
    Dataset,
    InMemoryEngine,
    InMemoryRows,
    SampleRequest,
    prepare,
)
from kumo.pytorch.architectures import build_architecture
from tests.reference_models.architecture_definitions import ARCHITECTURES, MODEL_BINDINGS
from tests.reference_models.batch_adapters import MODEL_BATCH_ADAPTERS
from tests.reference_models.batch_definitions import BATCH_SPECS
from tests.reference_models.data_recipes import (
    NUFORMER_SAMPLE,
    PANTHER_SAMPLE,
    PRAGMA_SAMPLE,
    TREASURE_SAMPLE,
)
from tests.reference_models.data_schema import SYNTHETIC_SCHEMA

SAMPLES = {
    "pragma": PRAGMA_SAMPLE,
    "nuformer": NUFORMER_SAMPLE,
    "panther": PANTHER_SAMPLE,
    "treasure": TREASURE_SAMPLE,
}


def _tables() -> dict[str, list[dict[str, object]]]:
    tables: dict[str, list[dict[str, object]]] = {
        "users": [],
        "products": [],
        "accounts": [],
        "merchants": [],
        "devices": [],
        "banking_transactions": [],
        "transfers": [],
        "card_transactions": [],
        "card_payments": [],
        "card_disputes": [],
        "investment_transactions": [],
        "communications": [],
        "product_adoptions": [],
    }
    tables["products"] = [
        {"product_id": 1, "product_type": "checking_account"},
        {"product_id": 2, "product_type": "credit_card"},
    ]
    tables["merchants"] = [
        {"merchant_id": 50, "risk_score": 0.1},
        {"merchant_id": 51, "risk_score": 0.8},
    ]
    for user_id in (1, 2):
        checking_id = 10 + user_id
        card_id = 20 + user_id
        tables["users"].append(
            {
                "user_id": user_id,
                "signup_at": 10,
                "birth_year": 1990,
                "country": "C001",
                "region": "region_01",
                "home_currency": "USD",
                "income_band": "middle",
                "risk_segment": "low",
                "acquisition_channel": "organic",
            }
        )
        tables["accounts"].extend(
            [
                {
                    "account_id": checking_id,
                    "user_id": user_id,
                    "product_id": 1,
                    "account_type": "checking",
                    "opened_at": 20,
                    "closed_at": None,
                },
                {
                    "account_id": card_id,
                    "user_id": user_id,
                    "product_id": 2,
                    "account_type": "credit_card",
                    "opened_at": 30,
                    "closed_at": None,
                },
            ]
        )
        for index, occurred_at in enumerate((60, 80)):
            tables["banking_transactions"].append(
                {
                    "transaction_id": user_id * 100 + index,
                    "user_id": user_id,
                    "account_id": checking_id,
                    "occurred_at": occurred_at,
                    "transaction_type": "deposit" if index == 0 else "withdrawal",
                    "amount": 100.0 if index == 0 else 25.0,
                    "currency": "USD",
                    "channel": "mobile",
                    "counterparty_type": "employer" if index == 0 else "atm",
                    "description": "salary payment" if index == 0 else "cash withdrawal",
                    "status": "settled",
                }
            )
            tables["card_transactions"].append(
                {
                    "card_transaction_id": user_id * 1_000 + index,
                    "user_id": user_id,
                    "account_id": card_id,
                    "occurred_at": occurred_at + 5,
                    "merchant_id": 50 + index,
                    "amount": 12.5 + index,
                    "currency": "USD",
                    "mcc": "5815",
                    "channel": "online",
                    "entry_mode": "card_on_file",
                    "country": "C001",
                    "description": "monthly music plan",
                    "status": "approved",
                    "recurring_candidate_flag": True,
                }
            )
    return tables


def small_config(name):
    """Model sizes small enough for deterministic CPU forward/backward tests."""
    config = ARCHITECTURES[name]
    encoder = config.encoder
    if name == "pragma":
        encoder = replace(
            encoder,
            width=24,
            ffn_width=48,
            attention_heads=3,
            profile_layers=1,
            event_layers=1,
            history_layers=1,
            dropout=0.0,
            event_layout="padded",
        )
    elif name == "panther":
        encoder = replace(
            encoder,
            transformer=replace(
                encoder.transformer,
                width=24,
                ffn_width=48,
                attention_heads=3,
                layers=1,
                dropout=0.0,
            ),
            prototypes=8,
        )
    else:
        encoder = replace(encoder, width=24, ffn_width=48, attention_heads=3, layers=1, dropout=0.0)
    return replace(config, encoder=encoder)


def requests_for(name):
    entities = (21, 22) if name == "treasure" else (1, 2)
    return tuple(
        SampleRequest(sample_id=f"{name}-{index}", entity_id=entity, as_of_time=100)
        for index, entity in enumerate(entities, 1)
    )


@contextmanager
def model_task(
    name,
    *,
    tables=None,
    batch=None,
    sample=None,
    schema=None,
    requests=None,
    fitted=None,
    targets=True,
):
    """Fit the ordinary loader contract; callers build model modules separately."""
    batch = batch or BATCH_SPECS[name]
    if not targets:
        batch = replace(batch, targets={})
    contract = DataLoaderSpec(
        schema=schema or SYNTHETIC_SCHEMA, sample=sample or SAMPLES[name], batch=batch
    )
    dataset = Dataset(
        tables={table: InMemoryRows(rows=rows) for table, rows in (tables or _tables()).items()}
    )
    with prepare(contract=contract, dataset=dataset, engine=InMemoryEngine()) as plan:
        with plan.materialize(requests or requests_for(name), fitted=fitted) as task:
            yield task


def architecture_for(task, name, *, config=None):
    return build_architecture(
        task=task,
        config=config or small_config(name),
        bindings=MODEL_BINDINGS[name],
        adapt=MODEL_BATCH_ADAPTERS[name],
    )


def adapted_batch(task, name, *, seed=1729, epoch=0):
    return MODEL_BATCH_ADAPTERS[name](
        next(task.batches(batch_size=task.count(), epoch=epoch, seed=seed))
    )
