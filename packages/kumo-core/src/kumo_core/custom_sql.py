# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static analysis and identity helpers for Custom SQL Tables.

Shared by the hosted platform and the SDK: ``custom_sql_synthetic_name`` lands
inside a graph's identity, so a second copy would diverge cloud and local
graph ids with no error.
"""
from __future__ import annotations

import hashlib
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.tokens import TokenType

from .types import ConnectorType

_CONNECTOR_DIALECTS: Mapping[ConnectorType, str] = {
    ConnectorType.SNOWFLAKE: "snowflake",
    ConnectorType.DATABRICKS: "databricks",
    ConnectorType.BIGQUERY: "bigquery",
}

# sqlglot models these as generic qualified calls, indistinguishable from
# dataset UDFs by node type.
_BIGQUERY_BUILTIN_ROUTINE_NAMESPACES = frozenset({
    'AEAD',
    'AI',
    'HLL_COUNT',
    'KEYS',
    'KLL_QUANTILES',
    'ML',
    'NET',
    'OBJ',
    'SAFE',
    'VECTOR_INDEX',
})

_READ_ONLY_QUERY = (exp.Select, exp.SetOperation, exp.Subquery)

# Rejected anywhere in the query, including inside a CTE or subquery body.
_WRITE_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Copy,
    exp.Command,
)

#: ``LIMIT`` does not reduce a BigQuery scan, so a stype-inference sample of a
#: query is gated on this and also runs under it as ``maximum_bytes_billed``.
SQL_SAMPLE_MAX_SCAN_BYTES = 10 * 1024**3

#: Matches the hosted `SNOWFLAKE_SAMPLE_QUERY_TIMEOUT_SECONDS` and
#: `DATABRICKS_SAMPLE_QUERY_TIMEOUT_SECONDS`, so cloud and local refuse alike.
SQL_SAMPLE_TIMEOUT_SECONDS = 60


def sample_timeout(timeout: Optional[int],
                   query: Optional[str]) -> Optional[int]:
    """The bound a stype-inference sample runs under, or ``None``.

    Only a query is bounded; a physical sample already is, by its ``LIMIT``.
    Never below one second: ``0`` is *unlimited* on all three warehouses, so
    clamping up is what keeps a caller asking for "no time at all" from being
    read as "as long as you like".

    Beside the constant it falls back to rather than in any one reader, so the
    three warehouses that run Custom SQL cannot drift on it.
    """
    if query is None:
        return None
    if timeout is not None:
        return max(1, int(timeout))
    return SQL_SAMPLE_TIMEOUT_SECONDS


def sample_over_budget(estimate: Optional[int], *, gated: bool) -> bool:
    """Whether running a query to sample it would cost more than Kumo spends.

    ``gated`` is BigQuery and only BigQuery. Under a gate a ``None`` estimate
    is over budget: a dry run reports no cost for external and federated
    tables and some wildcard/``SEARCH`` shapes, which is what this stops.
    """
    if not gated:
        return False
    return estimate is None or estimate > SQL_SAMPLE_MAX_SCAN_BYTES


class CustomSqlOutputSchemaError(ValueError):
    pass


class UnsupportedColumnTypeError(CustomSqlOutputSchemaError):
    pass


def sqlglot_dialect(connector_type: ConnectorType) -> str:
    """Return the sqlglot dialect for a warehouse that supports Custom SQL Tables."""
    dialect = _CONNECTOR_DIALECTS.get(connector_type)
    if dialect is None:
        supported = ", ".join(sorted(d.value for d in _CONNECTOR_DIALECTS))
        raise ValueError(
            f"Custom SQL Tables are not supported for '{connector_type.value}' "
            f"connectors. Supported warehouses: {supported}.")
    return dialect


def _parse_executable(sql: str, connector_type: ConnectorType, *,
                      many: bool = False) -> Any:
    """Parse the text that will execute, never the text as written.

    A trailing ``;;``, ``;-- comment`` or ``; /*c*/`` parses as an
    ``exp.Block``, in which ``traverse_scope`` finds no scopes at all.
    """
    import sqlglot
    text = subquery_safe_sql(sql, connector_type)
    dialect = sqlglot_dialect(connector_type)
    if many:
        return sqlglot.parse(text, dialect=dialect)
    return sqlglot.parse_one(text, dialect=dialect)


def advisory_read_only_check(sql: str, connector_type: ConnectorType) -> None:
    """Reject definite writes and multiple statements as fast feedback."""
    try:
        statements = [
            stmt for stmt in _parse_executable(sql, connector_type, many=True)
            if stmt is not None and not isinstance(stmt, exp.Semicolon)
        ]
    except SqlglotError:
        # Unparseable by sqlglot: defer to the warehouse rather than block a
        # valid query.
        return
    if len(statements) > 1:
        raise ValueError("Only a single SQL statement is allowed; found "
                         f"{len(statements)}.")
    if not statements:
        return
    statement = statements[0]
    if (not isinstance(statement, _READ_ONLY_QUERY)
            or statement.find(*_WRITE_EXPRESSIONS) is not None):
        raise ValueError(
            "Only read-only SELECT statements (optionally with leading CTEs) "
            "are allowed; data-modifying or DDL statements are not permitted.")
    if statement.find(exp.Placeholder, exp.Parameter) is not None:
        raise ValueError(
            "Custom SQL does not accept unbound parameters; supply a fully "
            "bound query before creating the table.")


def validate_sql_output_contract(sql: str,
                                 connector_type: ConnectorType) -> None:
    """Require a flat row result and deliberate names for computed outputs."""
    try:
        query = _parse_executable(sql, connector_type)
    except SqlglotError:
        return  # Fail open for new warehouse syntax.
    while isinstance(query, (exp.Subquery, exp.SetOperation)):
        query = query.this
    if not isinstance(query, exp.Select):
        return
    if query.args.get("kind") is not None:
        raise CustomSqlOutputSchemaError(
            "Custom SQL must return flat named columns; SELECT AS STRUCT and "
            "SELECT AS VALUE are unsupported. Flatten the result into columns.")
    if any(not isinstance(item, (exp.Alias, exp.Column, exp.Star))
           for item in query.expressions):
        raise CustomSqlOutputSchemaError(
            "Computed Custom SQL output columns require an explicit alias "
            "(for example, `<expression> AS output_name`).")


def referenced_tables_or_none(
        sql: str, connector_type: ConnectorType) -> Optional[List[str]]:
    """Return external tables, excluding CTE and derived-source aliases.

    ``None`` means dependencies cannot be proven and must not be treated as the
    dependency-free result ``[]`` by scope or change-detection callers.
    """
    dialect = sqlglot_dialect(connector_type)
    try:
        expression = _parse_executable(sql, connector_type)
    except SqlglotError:
        return None
    # A table-valued or data-source function (EXTERNAL_QUERY, read_files, a
    # Snowflake TABLE function, a stage path) hides its dependency in its
    # arguments, where the namespace check cannot reach it.
    if expression.find(exp.TableFromRows) is not None:
        return None
    if any(not isinstance(table.this, exp.Identifier)
           for table in expression.find_all(exp.Table)):
        return None
    tables = {
        ".".join(part.sql(dialect=dialect) for part in source.parts)
        for scope in traverse_scope(expression)
        for source in scope.sources.values() if isinstance(source, exp.Table)
    }
    return sorted(tables)


def _qualified_routines_or_none(
    sql: str,
    connector_type: ConnectorType,
) -> Optional[List[Tuple[str, List[exp.Identifier]]]]:
    """Qualified scalar routines and the namespace parts before their name."""
    dialect = sqlglot_dialect(connector_type)
    try:
        expression = _parse_executable(sql, connector_type)
    except SqlglotError:
        return None

    def identifier_chain(
            node: exp.Expression) -> Optional[List[exp.Identifier]]:
        if isinstance(node, exp.Identifier):
            return [node]
        if (isinstance(node, exp.Dot)
                and isinstance(node.expression, exp.Identifier)):
            prefix = identifier_chain(node.this)
            return None if prefix is None else prefix + [node.expression]
        return None

    routines: List[Tuple[str, List[exp.Identifier]]] = []
    for dot in expression.find_all(exp.Dot):
        if isinstance(dot.expression, exp.Func):
            qualifiers = identifier_chain(dot.this)
            if qualifiers:
                if (connector_type == ConnectorType.BIGQUERY
                        and len(qualifiers) == 1 and not qualifiers[0].quoted
                        and qualifiers[0].name.upper()
                        in _BIGQUERY_BUILTIN_ROUTINE_NAMESPACES):
                    continue
                routines.append((dot.sql(dialect=dialect), qualifiers))

    # BigQuery represents a backtick-quoted project.dataset.routine as one
    # Identifier on the function rather than a Dot chain.
    if connector_type == ConnectorType.BIGQUERY:
        for function in expression.find_all(exp.Anonymous):
            name = function.this
            if (isinstance(name, exp.Identifier) and name.quoted
                    and '.' in name.name):
                parts = name.name.split('.')
                routines.append((function.sql(dialect=dialect), [
                    exp.to_identifier(part, quoted=True) for part in parts[:-1]
                ]))
    return routines


def _parse_error(sql: str, connector_type: ConnectorType) -> Optional[str]:
    """The dialect parser's own complaint, or ``None`` if it had none."""
    try:
        _parse_executable(sql, connector_type)
    except SqlglotError as error:
        return " ".join(str(error).split())
    return None


def validate_referenced_table_scope(
    sql: str,
    connector_type: ConnectorType,
    namespace: Sequence[str],
) -> List[str]:
    """Return dependencies after proving they stay in the connector target.

    Unqualified tables resolve in the connector's configured namespace. A
    qualified table may repeat the configured schema/dataset, or the complete
    database/catalog/project plus schema/dataset, but may not escape it. SQL
    whose dependencies cannot be established is rejected.
    """
    refs = referenced_tables_or_none(sql, connector_type)
    routines = _qualified_routines_or_none(sql, connector_type)
    if refs is None or routines is None:
        detail = _parse_error(sql, connector_type)
        if detail is not None:
            raise ValueError(
                f"Custom SQL is not valid "
                f"{sqlglot_dialect(connector_type)} SQL, so its table "
                f"references cannot be safely verified: {detail}")
        raise ValueError(
            "Custom SQL uses warehouse syntax whose table references cannot "
            "be safely verified. Rewrite the query so its source tables can "
            "be resolved within the connector's configured namespace.")

    def canonical(name: str, *, quoted: bool = False) -> str:
        if connector_type == ConnectorType.SNOWFLAKE and not quoted:
            return name.upper()
        if connector_type == ConnectorType.DATABRICKS:
            return name.casefold()
        return name

    allowed = [canonical(part) for part in namespace]
    dialect = sqlglot_dialect(connector_type)

    def validate_qualifiers(ref: str, qualifiers: Sequence[exp.Identifier],
                            kind: str) -> None:
        normalized = [
            canonical(part.name, quoted=part.quoted) for part in qualifiers
        ]
        if normalized and normalized != allowed[-len(normalized):]:
            target = ".".join(namespace)
            raise ValueError(
                f"Custom SQL {kind} reference '{ref}' is outside this "
                f"connector's configured namespace '{target}'.")

    import sqlglot
    for ref in refs:
        table = sqlglot.parse_one(ref, dialect=dialect, into=exp.Table)
        validate_qualifiers(ref, table.parts[:-1], 'table')
    for ref, qualifiers in routines:
        validate_qualifiers(ref, qualifiers, 'routine')
    return refs


#: Where each warehouse keeps the two parts of the namespace a query resolves
#: in. Also read by the SDK, so the scope a query is proved to stay inside and
#: the namespace its graph is identified by stay one thing.
CUSTOM_SQL_NAMESPACE_ATTRS: Mapping[ConnectorType, Tuple[str, str]] = {
    ConnectorType.SNOWFLAKE: ('database', 'schema_name'),
    ConnectorType.DATABRICKS: ('catalog', 'schema_name'),
    ConnectorType.BIGQUERY: ('project_id', 'dataset_id'),
}


def validate_connector_sql_scope(
    sql: str,
    connector_type: ConnectorType,
    connector: object,
) -> List[str]:
    """Prove that a query stays inside its connector's configured namespace."""
    try:
        attrs = CUSTOM_SQL_NAMESPACE_ATTRS[connector_type]
        namespace = tuple(getattr(connector, attr) for attr in attrs)
    except (KeyError, AttributeError) as error:
        raise ValueError(
            f"Custom SQL is not supported for {connector_type.value}."
        ) from error
    if not all(isinstance(part, str) and part for part in namespace):
        raise ValueError(
            "Custom SQL requires a configured warehouse namespace.")
    return validate_referenced_table_scope(sql, connector_type, namespace)


def normalized_sql_identifier_leaf(name: str,
                                   connector_type: ConnectorType) -> str:
    """Dialect-normalize the table-name leaf of a qualified identifier.

    A dummy database qualifier forces SQLGlot to apply table rather than column
    normalization to a one-part name. Unparseable input is returned unchanged.
    """
    dialect = sqlglot_dialect(connector_type)
    import sqlglot
    try:
        table = sqlglot.maybe_parse(name, dialect=dialect, into=exp.Table)
    except SqlglotError:
        return name
    if table is None or not table.parts:
        return name
    leaf = table.parts[-1]
    qualified = exp.Table(this=exp.to_identifier(leaf.name, quoted=leaf.quoted),
                          db=exp.to_identifier("_"))
    return normalize_identifiers(qualified, dialect=dialect).this.name


def subquery_safe_sql(sql: str, connector_type: ConnectorType) -> str:
    """Strip only trailing terminators/comments for subquery embedding."""
    dialect = sqlglot_dialect(connector_type)
    import sqlglot
    try:
        tokens = sqlglot.tokenize(sql, dialect=dialect)
    except SqlglotError:
        return sql.strip().rstrip(';').rstrip()
    body = [t for t in tokens if t.token_type != TokenType.SEMICOLON]
    if not body:
        return sql.strip().rstrip(';').rstrip()
    return sql[:body[-1].end + 1]


def normalize_sql(sql: str, connector_type: ConnectorType) -> str:
    """Canonicalize SQL so formatting-only edits (whitespace, comments, keyword
    case) leave the change-detection fingerprint unchanged."""
    dialect = sqlglot_dialect(connector_type)
    import sqlglot
    try:
        expression = sqlglot.parse_one(sql, dialect=dialect)
    except SqlglotError:
        return " ".join(sql.split())
    # BigQuery identifiers are case-sensitive even when unquoted, so a
    # case-only table change must not reuse another table's identity.
    fold_identifiers = connector_type != ConnectorType.BIGQUERY
    return expression.sql(dialect=dialect, normalize=fold_identifiers,
                          comments=False, pretty=False)


_CUSTOM_SQL_NAME_PREFIX = "custom_sql_"


def custom_sql_synthetic_name(sql: str, connector_type: ConnectorType) -> str:
    """Identity whose normalization ignores formatting but not semantic edits."""
    digest = hashlib.md5(normalize_sql(sql,
                                       connector_type).encode()).hexdigest()
    return f"{_CUSTOM_SQL_NAME_PREFIX}{digest}"


def custom_sql_data_version(
    sql: str,
    connector_type: ConnectorType,
    referenced_table_versions: Mapping[str, str],
    cols: Mapping[str, object],
    *,
    max_staleness_bucket: Optional[int] = None,
) -> str:
    """Fingerprint SQL inputs and the physical output contract.

    Output columns are included because projections and casts change Parquet
    even when SQL and source versions do not.
    """
    hasher = hashlib.md5()
    # Length-prefixed so no combination of names, versions, cols and SQL text
    # can produce the same byte stream.
    for name in sorted(referenced_table_versions):
        version = referenced_table_versions[name]
        hasher.update(f"{len(name)}:{name}:{len(version)}:{version}".encode())
    hasher.update(b":sql:")
    hasher.update(normalize_sql(sql, connector_type).encode())
    hasher.update(b":cols:")
    for name in sorted(cols):
        dtype = str(cols[name])
        hasher.update(f"{len(name)}:{name}:{len(dtype)}:{dtype}".encode())
    # A section of its own, so the staleness window cannot collide with a real
    # table name in the map above.
    if max_staleness_bucket is not None:
        hasher.update(b":staleness:")
        hasher.update(str(max_staleness_bucket).encode())
    return hasher.hexdigest()


def sample_select(source: str, num_rows: int, projection: str = "*") -> str:
    num_rows = int(num_rows)
    if num_rows < 0:
        raise ValueError(
            f"sample_query requires a non-negative row limit, got {num_rows}.")
    return f'SELECT {projection} FROM {source} LIMIT {num_rows}'
