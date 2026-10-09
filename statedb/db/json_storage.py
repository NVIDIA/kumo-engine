# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone

from io_utils.json import to_json_dict
from statedb.db.runtime import StateTx


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def encode_timestamp(tx: StateTx, value: datetime) -> datetime | str:
    return value if tx.backend == "postgres" else value.isoformat()


def payload_dict_for_storage(
    record: object,
    *,
    extra_fields: Mapping[str, object] | None = None,
    omit_fields: Iterable[str] = (),
) -> dict[str, object]:
    payload = to_json_dict(record)
    for field_name in omit_fields:
        payload.pop(field_name, None)
    payload.update(extra_fields or {})
    return payload


def payload_json_for_storage(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True)
