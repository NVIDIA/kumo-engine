# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static analysis and identity helpers for Custom SQL Tables.

The identity and gate layer lives in :mod:`kumo_core.custom_sql` so the hosted
platform and the SDK's ``LocalBackend`` derive the same synthetic name -- and
so the same graph id -- from the same SQL. What remains here is the analysis
only the platform performs.
"""
from __future__ import annotations

import hashlib
from typing import List, Mapping, Optional, Tuple

import sqlglot
from kumo_core.custom_sql import (
    SQL_SAMPLE_MAX_SCAN_BYTES,
    CustomSqlOutputSchemaError,
    UnsupportedColumnTypeError,
    advisory_read_only_check,
    custom_sql_data_version,
    custom_sql_synthetic_name,
    normalize_sql,
    normalized_sql_identifier_leaf,
    referenced_tables_or_none,
    sample_over_budget,
    sqlglot_dialect,
    subquery_safe_sql,
    validate_connector_sql_scope,
    validate_referenced_table_scope,
    validate_sql_output_contract,
)
from kumo_core.types import ConnectorType
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import build_scope

__all__ = [
    'SQL_SAMPLE_MAX_SCAN_BYTES',
    'CustomSqlOutputSchemaError',
    'UnsupportedColumnTypeError',
    'advisory_read_only_check',
    'custom_sql_data_version',
    'custom_sql_stats_version',
    'custom_sql_synthetic_name',
    'normalize_sql',
    'normalized_sql_identifier',
    'normalized_sql_identifier_leaf',
    'referenced_tables',
    'referenced_tables_or_none',
    'sample_over_budget',
    'source_table_for_col',
    'sqlglot_dialect',
    'subquery_safe_sql',
    'validate_connector_sql_scope',
    'validate_referenced_table_scope',
    'validate_sql_output_contract',
]


def referenced_tables(sql: str, connector_type: ConnectorType) -> List[str]:
    """Best-effort external tables for lineage display."""
    return referenced_tables_or_none(sql, connector_type) or []


def normalized_sql_identifier(name: str, connector_type: ConnectorType) -> str:
    """Dialect-normalize a possibly dotted identifier."""
    dialect = sqlglot_dialect(connector_type)
    try:
        table = sqlglot.maybe_parse(name, dialect=dialect, into=exp.Table)
    except SqlglotError:
        return name
    if table is None:
        return name
    return ".".join(
        part.name
        for part in normalize_identifiers(table, dialect=dialect).parts)


def source_table_for_col(
    sql: str,
    connector_type: ConnectorType,
    col_name: str,
) -> Optional[Tuple[str, str]]:
    """Trace a direct output to its normalized external table and column.

    Only unambiguous pass-through columns are traced. Computed or star outputs,
    CTE/subquery sources, and parse failures return ``None``. View definitions
    are not recursively resolved.
    """
    dialect = sqlglot_dialect(connector_type)
    try:
        expression = sqlglot.parse_one(sql, dialect=dialect)
        qualified = qualify(expression, dialect=dialect, schema=None,
                            validate_qualify_columns=False)
    except SqlglotError:
        return None
    root = build_scope(qualified)
    if root is None:
        return None
    target = normalized_sql_identifier(col_name, connector_type)
    traced: Optional[Tuple[str, str]] = None
    for projection in qualified.expressions:
        output_name = normalized_sql_identifier(projection.alias_or_name,
                                                connector_type)
        if output_name != target:
            continue
        if traced is not None:
            return None  # Output name produced by more than one projection.
        column = (projection.unalias()
                  if isinstance(projection, exp.Alias) else projection)
        if not isinstance(column, exp.Column):
            return None  # Computed/aliased expression, or ``*``.
        source = root.sources.get(column.table)
        if not isinstance(source, exp.Table):
            return None  # CTE/subquery/derived source, or unresolved.
        table_name = ".".join(part.name for part in source.parts)
        traced = (table_name, column.name)
    return traced


def custom_sql_stats_version(
    sql: str,
    connector_type: ConnectorType,
    cols: Mapping[str, object],
) -> str:
    """Definition identity for validate-time approximate column stats.

    Upstream table versions are deliberately excluded so ordinary data churn
    does not orphan preview stats; semantic SQL or output-dtype changes do.
    """
    hasher = hashlib.md5()
    hasher.update(normalize_sql(sql, connector_type).encode())
    hasher.update(b":cols:")
    for name in sorted(cols):
        # Use the enum ``value`` rather than ``str``. There is now a single
        # ``Dtype`` (kumoapi's), but tokens hashed here outlive the process
        # that wrote them, and ``value`` is the half of the enum that is wire
        # contract; ``str`` is presentation and has changed before.
        raw = cols[name]
        dtype = str(getattr(raw, "value", raw))
        hasher.update(f"{len(name)}:{name}:{len(dtype)}:{dtype}".encode())
    # Namespaced so the token can never collide with a real content data_version.
    return f"sqldef_{hasher.hexdigest()}"
