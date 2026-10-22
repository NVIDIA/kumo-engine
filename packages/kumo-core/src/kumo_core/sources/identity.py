# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import time
from typing import Callable, Mapping
from uuid import uuid4

from ..types.enums import ConnectorType

logger = logging.getLogger(__name__)

#: A query's proven dependency set is an incomplete proxy for what it reads,
#: so an aligned wall-clock bucket bounds how long a version may stand. The
#: platform reads the same default from ``CUSTOM_SQL_MAX_STALENESS``.
CUSTOM_SQL_MAX_STALENESS_HOURS = 24


class SourceVersion(str):
    """A version string that also says what kind of evidence it is:
    ``exact``, ``timestamp``, ``listing``, ``hour_bucket`` or ``unknown``."""

    kind: str
    dependencies: Mapping[str, str] | None

    def __new__(cls, value: str, kind: str = "unknown",
                dependencies: Mapping[str, str] | None = None
                ) -> "SourceVersion":
        version = super().__new__(cls, value)
        version.kind = kind
        version.dependencies = dependencies
        return version


def version_kind(version: object) -> str:
    return getattr(version, "kind", "unknown")


def epoch_second(moment: datetime) -> str:
    """Truncated by string rather than through ``int``: the two disagree
    just below the epoch, and this value is a raw path segment."""
    return str(moment.timestamp()).split(".")[0]


def hour_version() -> str:
    """Fallback for a source with no readable version; it expires itself
    rather than needing invalidation."""
    return SourceVersion(
        epoch_second(
            datetime.now(timezone.utc).replace(minute=0, second=0,
                                               microsecond=0)), "hour_bucket")


def sql_data_version(
    sql: str,
    connector_type: ConnectorType,
    probe: Callable[[str], str | None],
    cols: Mapping[str, object] | None = None,
    *,
    table: str | None = None,
) -> str:
    """Change-detection version for a query, from what the query reads.

    A query has no catalog entry to stamp, so its version is a fingerprint of
    the normalized SQL, the version of every table it is proven to read, and
    the output contract. ``probe`` stamps one referenced table by its bare
    dialect-cased leaf, which is the only name a connector resolves; ``table``
    names the Custom SQL table in the warning each sentinel is announced with.

    Uncertainty is folded in as a fresh sentinel rather than dropped, and an
    aligned wall-clock bucket puts a ceiling on the rest.
    """
    from ..custom_sql import (
        custom_sql_data_version,
        normalized_sql_identifier_leaf,
        referenced_tables_or_none,
    )

    versions: dict[str, str] = {}
    referenced = referenced_tables_or_none(sql, connector_type)
    if referenced is None:
        logger.warning(
            "Custom SQL table %s: SQL could not be parsed for dependency "
            "extraction; re-materializing conservatively.", table)
        # The map is empty in this branch, so the synthetic key cannot
        # collide with a real table name.
        versions["__unparsed__"] = _indeterminate()
        referenced = []
    for reference in referenced:
        leaf = normalized_sql_identifier_leaf(reference, connector_type)
        refusal: Exception | None = None
        try:
            stamped = probe(leaf)
        except Exception as error:
            stamped, refusal = None, error
        if stamped is None:
            logger.warning(
                "Custom SQL table %s: could not resolve version for "
                "referenced table %s (%r); re-materializing conservatively.",
                table, leaf, refusal)
            versions[leaf] = _indeterminate()
        else:
            versions[leaf] = str(stamped)
    return SourceVersion(
        custom_sql_data_version(
            sql, connector_type, versions, cols or {},
            max_staleness_bucket=int(time()) //
            (CUSTOM_SQL_MAX_STALENESS_HOURS * 3600)), "hour_bucket", versions)


#: The prefix every sentinel carries, so one can be recognized without being
#: reproducible.
_INDETERMINATE = "indeterminate:"


def _indeterminate() -> str:
    """A version that cannot repeat, for an input nobody could check.

    A fixed marker would make every unstampable reference hash alike, and the
    second table to hit one would restore a snapshot built over the first's
    data. The cost is re-staging on every run, which is warned about wherever
    a sentinel is minted.
    """
    return f"{_INDETERMINATE}{uuid4().hex}"
