# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from kumo_core.sources import sql
from kumo_core.sources.errors import PermanentSourceError
from kumoapi.typing import Dtype

QUOTE = "`"


def quote_identifier(name: str) -> str:
    """BigQuery has no in-band escape inside a quoted identifier, so a name
    that could end the quoting is refused rather than interpolated."""
    return sql.quote_identifier(name, quote=QUOTE, escape=False)


def qualified_name(*parts: str) -> str:
    return sql.qualified_name(*parts, quote=quote_identifier)


def table_path(project: str, dataset: str, table: str) -> str:
    for part in (project, dataset, table):
        if not part or "/" in part:
            raise ValueError(f"{part!r} is not a table path component")
    return f"projects/{project}/datasets/{dataset}/tables/{table}"


def order_by_schema(
    columns: Mapping[str, Dtype],
    schema: Sequence[str],
) -> dict[str, Dtype]:
    """A read session returns the source's column order whatever the
    projection asked for, so without this the two modes would disagree about
    the order of the same columns."""
    missing = [name for name in columns if name not in set(schema)]
    if missing:
        raise ValueError(f"source has no column named {sorted(missing)}")
    return {name: columns[name] for name in schema if name in columns}


def build_projection(
    columns: Mapping[str, Dtype],
    aliases: Mapping[str, str] | None = None,
) -> str:
    """BigQuery needs no cast for any dtype this reader carries, so the
    dtypes select no expression today; they are carried so that one could."""
    return sql.render_projection(columns, aliases, quote=quote_identifier)


def output_columns(
    columns: Mapping[str, Dtype],
    aliases: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    aliases = sql.validated_aliases(columns, aliases)
    return tuple(aliases.get(name, name) for name in columns)


def home_dataset(project: str, dataset: str,
                 max_bytes_billed: int | None = None,
                 job_timeout_ms: int | None = None) -> Any:
    """A job config whose default dataset is the connector's own, so
    unqualified names in a user's query resolve there.

    ``max_bytes_billed`` is a ceiling BigQuery enforces itself, so it holds
    for queries a dry run cannot cost at all. An unset option is left out
    rather than passed as ``None``: ``QueryJobConfig`` stringifies this one
    field, so a ``None`` reaches the API as ``"None"`` and every job that
    carries the config is rejected.
    """
    from google.cloud import bigquery

    options = {
        name: value
        for name, value in (("maximum_bytes_billed", max_bytes_billed),
                            ("job_timeout_ms", job_timeout_ms))
        if value is not None
    }
    return bigquery.QueryJobConfig(default_dataset=f"{project}.{dataset}",
                                   **options)


@dataclass(frozen=True)
class QueryPlan:
    """What a dry run establishes: the ordered result schema, and what
    running the query would scan."""

    schema: tuple[Any, ...]
    scanned_bytes: int | None


def dry_run(client: Any, sql: str, *, project: str, dataset: str) -> Any:
    config = home_dataset(project, dataset)
    config.dry_run = True
    config.use_query_cache = False
    return client.query(sql, job_config=config)


def read_only_job(client: Any, sql: str, *, project: str, dataset: str) -> Any:
    """A dry run compiles the query without scanning a row, and classifies
    it: a script or a procedure is not a read however it parses."""
    job = dry_run(client, sql, project=project, dataset=dataset)
    if job.statement_type != "SELECT":
        raise PermanentSourceError(
            f"custom sql must be a single read-only SELECT; bigquery "
            f"classified this as {job.statement_type}")
    return job


def query_plan(client: Any, sql: str, *, project: str,
               dataset: str) -> QueryPlan:
    job = read_only_job(client, sql, project=project, dataset=dataset)
    return QueryPlan(tuple(job.schema), job.total_bytes_processed)


def query_result(execute: Callable[..., Any], query: str,
                 job_config: Any = None) -> Any:
    settings = {} if job_config is None else {'job_config': job_config}
    return execute(query, **settings).result()


def sample_request(sql: str, rows: int, *, project: str, dataset: str,
                   max_bytes_billed: int,
                   job_timeout_ms: int | None = None) -> tuple[str, Any]:
    from kumo_core.custom_sql import sample_select

    return (sample_select(f'({sql})', rows),
            home_dataset(project, dataset, max_bytes_billed, job_timeout_ms))


def table_names(client: Any, dataset: str) -> list[str]:
    return sorted(str(table.table_id) for table in client.list_tables(dataset))
