# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transaction/profile semantics mapped to a small explicit relational graph."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import datetime

from tfm.contracts.schema import TransactionEvent, TransferEdge
from tfm.data.graph import GraphLink, GraphRow, GraphTable, RelationalGraph
from tfm.data.validation.schema import PROHIBITED_INPUT_KEYS

TRANSACTION_FORBIDDEN = PROHIBITED_INPUT_KEYS | frozenset(
    {"entity_id", "event_id", "timestamp", "fraud", "label", "split", "available_at", "chargeback"}
)


def transaction_graph(
    events: Sequence[TransactionEvent],
    transfers: Sequence[TransferEdge] = (),
    *,
    observed_through: datetime | None = None,
    amount_field: str = "amount",
) -> RelationalGraph:
    """Induced graph over selected owners; static identity roots have only a bias.

    Event/transfer fields are observed at event time under these adapters' data
    contract. Post-outcome labels never enter rows. Only amount is a numeric
    target; categorical identifiers are joins, never floating features. Missing
    amount is masked for targets and zero plus presence flag for inputs.
    """
    if not amount_field.strip() or amount_field.strip().casefold() in TRANSACTION_FORBIDDEN:
        raise ValueError("amount_field must name an observed numeric amount, not a forbidden field")
    owners = sorted({e.entity_id for e in events})
    rows = [GraphRow("users", key, {"bias": 1.0}) for key in owners]
    links: list[GraphLink] = []
    for event in sorted(events, key=lambda e: (e.event_timestamp, e.entity_id, e.event_id)):
        key = json.dumps(
            [event.entity_id, event.event_id], ensure_ascii=False, separators=(",", ":")
        )
        raw = event.numeric.get(amount_field, math.nan)
        finite = math.isfinite(raw)
        values = {
            "amount": raw,
            "amount_log": math.copysign(math.log1p(abs(raw)), raw) if finite else 0.0,
            "present": float(finite),
        }
        rows.append(GraphRow("events", key, values, event.event_timestamp, event.event_timestamp))
        links.append(
            GraphLink(("users", "events", "events"), event.entity_id, key, event.event_timestamp)
        )
    population = set(owners)
    for i, edge in enumerate(
        sorted(transfers, key=lambda e: (e.timestamp, e.from_entity, e.to_entity, e.amount))
    ):
        if edge.from_entity not in population or edge.to_entity not in population:
            continue
        key = f"transfer-{i}"
        rows.append(
            GraphRow(
                "transfers",
                key,
                {
                    "amount": edge.amount,
                    "amount_log": math.copysign(math.log1p(abs(edge.amount)), edge.amount),
                },
                edge.timestamp,
                edge.timestamp,
            )
        )
        links.extend(
            [
                GraphLink(("users", "sent", "transfers"), edge.from_entity, key, edge.timestamp),
                GraphLink(("users", "received", "transfers"), edge.to_entity, key, edge.timestamp),
            ]
        )
    return RelationalGraph(
        (
            GraphTable("users", ("bias",), immutable=True),
            GraphTable("events", ("amount_log", "present"), ("amount",)),
            GraphTable("transfers", ("amount_log",), ("amount",)),
        ),
        (
            ("users", "events", "events"),
            ("users", "sent", "transfers"),
            ("users", "received", "transfers"),
        ),
        tuple(rows),
        tuple(links),
        TRANSACTION_FORBIDDEN,
        observed_through,
    )
