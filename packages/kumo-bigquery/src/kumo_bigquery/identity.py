# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from kumo_core.sources.identity import (
    SourceVersion,
    epoch_second,
    hour_version,
)

LOGICAL_VIEW = "VIEW"
UNDATED_VERSION = "-1"


def is_logical_view(table: Any) -> bool:
    return str(table.table_type) == LOGICAL_VIEW


def stamp(table: Any) -> str:
    """A view's modification time describes its definition rather than its
    rows, so it takes the hour window; materialized views and external tables
    are dated anyway, matching what production already does."""
    if is_logical_view(table):
        return hour_version()
    if table.modified is None:
        return SourceVersion(UNDATED_VERSION)
    return SourceVersion(epoch_second(table.modified), "timestamp")


def data_version(client: Any, fqn: str) -> str:
    return stamp(client.get_table(fqn))
