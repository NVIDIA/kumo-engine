# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Writing local Parquet output into BigQuery, for hosted Kumo and Kumo
Engine alike.

Shards are conformed to one BigQuery schema and committed by a single
operation, so a failed write leaves the destination unchanged:

* ``STAGED`` uploads them under a unique GCS prefix and runs one load job.
* ``STREAMED`` appends them through Storage Write API pending streams and
  commits every stream at once. ``OVERWRITE`` commits into an expiring table
  and replaces the destination with one ``WRITE_TRUNCATE`` query.
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from google.cloud import bigquery
from kumo_bigquery import staging, streaming
from kumo_bigquery.staging import StagingLocation
from kumo_core.sources.errors import (
    PermanentSourceError,
    SourceAccessDeniedError,
    SourceNotFoundError,
)

OUTPUT_TABLE_ID = re.compile(r'[A-Za-z0-9_-]{1,1024}')
EXPORT_PREFIX = '_kumo.export'
TEMP_TABLE_PREFIX = '_kumo_export_'
TEMP_TABLE_TTL = timedelta(days=1)
_TYPE_NAMES = {
    pa.bool_(): 'BOOLEAN',
    pa.int64(): 'INTEGER',
    pa.float64(): 'FLOAT',
    pa.string(): 'STRING',
    pa.timestamp('us', tz='UTC'): 'TIMESTAMP',
    pa.date32(): 'DATE',
}
# Types a table can hold beyond those output is created with.
_TABLE_TYPES = {
    **{
        name: arrow_type
        for arrow_type, name in _TYPE_NAMES.items()
    },
    'NUMERIC': pa.decimal128(38, 9),
    'BIGNUMERIC': pa.decimal256(76, 38),
    'DATETIME': pa.timestamp('us'),
    'TIME': pa.time64('us'),
    'BYTES': pa.binary(),
    'JSON': pa.string(),
    'GEOGRAPHY': pa.string(),
}
_LEGACY_NAMES = {
    'INT64': 'INTEGER',
    'FLOAT64': 'FLOAT',
    'BOOL': 'BOOLEAN',
    'DECIMAL': 'NUMERIC',
    'BIGDECIMAL': 'BIGNUMERIC',
}


class OutputSchemaMismatchError(PermanentSourceError):
    """Output columns do not fit the existing destination table.

    ``problems`` names each column and its types, without the table's name.
    """
    def __init__(self, message: str, *, table: str,
                 problems: Sequence[str]) -> None:
        super().__init__(message, table=table)
        self.problems = list(problems)


class MissingWritePermissions(SourceAccessDeniedError):
    """The identity lacks permissions on an existing destination table."""


class ProbeCleanupError(PermanentSourceError):
    """A write probe table could not be removed; it expires on its own."""


def is_output_table_id(table_name: str) -> bool:
    return OUTPUT_TABLE_ID.fullmatch(table_name) is not None


def write_permissions(*, replace: bool) -> list[str]:
    required = {'bigquery.tables.get', 'bigquery.tables.updateData'}
    if replace:
        required.add('bigquery.tables.update')
    return sorted(required)


def load_job_config(
    *,
    replace: bool,
    schema: Optional[Sequence[Any]] = None,
) -> bigquery.LoadJobConfig:
    """Parquet load; list columns (embeddings) become REPEATED fields."""
    config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=(bigquery.WriteDisposition.WRITE_TRUNCATE if replace
                           else bigquery.WriteDisposition.WRITE_APPEND),
    )
    if schema:
        config.autodetect = False
        config.schema = schema
    else:
        config.autodetect = True
    parquet_options = bigquery.ParquetOptions()
    parquet_options.enable_list_inference = True
    config.parquet_options = parquet_options
    return config


# Schema ######################################################################


def _canonical(arrow_type: pa.DataType) -> Optional[pa.DataType]:
    if pa.types.is_boolean(arrow_type):
        return pa.bool_()
    if pa.types.is_integer(arrow_type):
        return pa.int64()
    if pa.types.is_floating(arrow_type):
        return pa.float64()
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return pa.string()
    if pa.types.is_timestamp(arrow_type):
        # Nanosecond Parquet timestamps load as INTEGER.
        return pa.timestamp('us', tz='UTC')
    if pa.types.is_date(arrow_type):
        return pa.date32()
    if (pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)
            or pa.types.is_fixed_size_list(arrow_type)):
        inner = _canonical(arrow_type.value_type)
        if inner is None or pa.types.is_list(inner):
            return None
        return pa.list_(inner)
    return None


def _type_name(field: Any) -> str:
    return _LEGACY_NAMES.get(field.field_type, field.field_type)


def _arrow_type(field: Any) -> Optional[pa.DataType]:
    """The Arrow type output must have to land in a table column."""
    arrow_type = _TABLE_TYPES.get(_type_name(field))
    if arrow_type is not None and field.mode == 'REPEATED':
        return pa.list_(arrow_type)
    return arrow_type


def _widens(source: pa.DataType, column: pa.DataType) -> bool:
    """Integers fit FLOAT and NUMERIC columns exactly, as BigQuery's own
    INT64 coercions do."""
    if pa.types.is_list(source) and pa.types.is_list(column):
        return _widens(source.value_type, column.value_type)
    return pa.types.is_integer(source) and (pa.types.is_floating(column)
                                            or pa.types.is_decimal(column))


def bq_field(field: pa.Field) -> Optional[bigquery.SchemaField]:
    arrow_type, mode = field.type, 'NULLABLE'
    if pa.types.is_list(arrow_type):
        arrow_type, mode = arrow_type.value_type, 'REPEATED'
    name = _TYPE_NAMES.get(arrow_type)
    return bigquery.SchemaField(field.name, name, mode=mode) if name else None


def bq_schema(target: pa.Schema) -> Optional[list[bigquery.SchemaField]]:
    """The BigQuery schema of conformed output, or ``None`` if a column has
    no fixed mapping."""
    fields = [bq_field(field) for field in target]
    return None if None in fields else fields  # type: ignore[return-value]


def nonempty(files: Sequence[Path]) -> list[Path]:
    """An empty shard's column types are not trustworthy."""
    return [path for path in files if pq.ParquetFile(path).metadata.num_rows
            ] or list(files[:1])


@dataclass(frozen=True)
class Plan:
    """What every shard is cast to, and the BigQuery schema it lands in."""
    target: pa.Schema
    schema: Optional[list[bigquery.SchemaField]]


def _source_fields(
    files: Sequence[Path],
    extra_columns: Optional[Mapping[str, Any]],
) -> list[tuple[str, pa.DataType]]:
    try:
        schema = pa.unify_schemas(
            [pq.read_schema(path).remove_metadata() for path in files],
            promote_options='permissive')
    except (pa.ArrowInvalid, pa.ArrowTypeError) as error:
        raise PermanentSourceError(
            f'Prediction shards have incompatible schemas: {error}') from None
    extra = extra_columns or {}
    return ([(field.name, field.type)
             for field in schema if field.name not in extra] +
            [(name, pa.scalar(value).type) for name, value in extra.items()])


def plan(
    files: Sequence[Path],
    extra_columns: Optional[Mapping[str, Any]] = None,
    existing: Any = None,
    *,
    replace: bool,
    ref: str,
) -> Plan:
    """One nullable schema for every shard: BigQuery autodetects a multi-file
    load from a single file, and a stream has one writer schema.

    ``APPEND`` to an existing table is strict: every output column must be a
    table column of the same type, or an integer into a FLOAT or NUMERIC
    column. An all-null column fits any column and is left to BigQuery's
    NULL. A new table types an all-null column STRING, the one choice nothing
    in the output contradicts.
    """
    fields = _source_fields(files, extra_columns)
    if not replace and existing is not None:
        return _append_plan(fields, existing, ref)
    target = pa.schema([
        pa.field(
            name,
            pa.string() if pa.types.is_null(arrow_type) else
            (_canonical(arrow_type) or arrow_type))
        for name, arrow_type in fields
    ])
    schema = bq_schema(target)
    if schema is not None and existing is not None:
        schema = _keep_column_metadata(schema, existing.schema)
    return Plan(target, schema)


def _append_plan(fields: list[tuple[str, pa.DataType]], existing: Any,
                 ref: str) -> Plan:
    columns = {field.name.casefold(): field for field in existing.schema}
    target, problems = [], []
    for name, arrow_type in fields:
        column = columns.get(name.casefold())
        if column is None:
            problems.append(f'{name} ({_describe(arrow_type)}) is not a '
                            f'table column')
            continue
        if pa.types.is_null(arrow_type):
            continue
        wanted = _arrow_type(column)
        source = _canonical(arrow_type) or arrow_type
        if wanted is None or not (source.equals(wanted)
                                  or _widens(source, wanted)):
            problems.append(f'{name} is {_describe(arrow_type)} but the table '
                            f'column is {_type_name(column)} {column.mode}')
            continue
        target.append(pa.field(column.name, wanted))
    if problems:
        raise OutputSchemaMismatchError(
            f'Cannot APPEND to {ref}: {"; ".join(problems)}. APPEND needs '
            f'every output column in the table with the same type; use '
            f'WriteMode.OVERWRITE or another table.', table=ref,
            problems=problems)
    return Plan(pa.schema(target), list(existing.schema))


def _describe(arrow_type: pa.DataType) -> str:
    field = bq_field(pa.field('_', _canonical(arrow_type) or arrow_type))
    if field is None:
        return str(arrow_type)
    return f'{field.field_type} {field.mode}'


def _keep_column_metadata(
        schema: list[bigquery.SchemaField],
        current: Sequence[bigquery.SchemaField]) -> list[bigquery.SchemaField]:
    """``OVERWRITE`` keeps the descriptions and policy tags of columns that
    keep their name and type."""
    kept = {
        (field.name.casefold(), _type_name(field), field.mode): field
        for field in current
    }
    merged = []
    for field in schema:
        old = kept.get((field.name.casefold(), field.field_type, field.mode))
        if old is None:
            merged.append(field)
            continue
        merged.append(
            bigquery.SchemaField(field.name, field.field_type, mode=field.mode,
                                 description=old.description,
                                 policy_tags=old.policy_tags))
    return merged


def _cast(table: pa.Table, target: pa.Schema,
          extra_columns: Optional[Mapping[str, Any]]) -> pa.Table:
    extra = {
        name.casefold(): value
        for name, value in (extra_columns or {}).items()
    }
    present = {name.casefold(): name for name in table.column_names}
    columns = []
    for field in target:
        key = field.name.casefold()
        if key in extra:
            column = pa.array([extra[key]] * table.num_rows)
        elif key in present:
            column = table.column(present[key])
        else:
            column = pa.nulls(table.num_rows, field.type)
        column = pc.cast(
            column, options=pc.CastOptions(field.type,
                                           allow_time_truncate=True))
        if pa.types.is_list(field.type):
            # A null REPEATED value fails a stream; a load writes [].
            column = pc.if_else(pc.is_null(column), pa.scalar([], field.type),
                                column)
        columns.append(column)
    return pa.Table.from_arrays(columns, schema=target)


def conformed_batches(
    path: Path,
    target: pa.Schema,
    extra_columns: Optional[Mapping[str, Any]] = None,
) -> Iterator[pa.RecordBatch]:
    for batch in pq.ParquetFile(path).iter_batches():
        yield from _cast(pa.Table.from_batches([batch]), target,
                         extra_columns).to_batches()


@contextmanager
def conformed(
    path: Path,
    target: pa.Schema,
    extra_columns: Optional[Mapping[str, Any]],
    scratch: str,
) -> Iterator[Path]:
    if not extra_columns and pq.read_schema(path).remove_metadata().equals(
            target):
        yield path
        return
    descriptor, name = tempfile.mkstemp(suffix='.parquet', dir=scratch)
    os.close(descriptor)
    out = Path(name)
    try:
        with pq.ParquetWriter(out, target) as writer:
            for batch in conformed_batches(path, target, extra_columns):
                writer.write_batch(batch)
        yield out
    finally:
        out.unlink(missing_ok=True)


# Preflight ###################################################################


def existing_table(client: Any, ref: str) -> Any:
    from google.api_core.exceptions import NotFound
    try:
        table = client.get_table(ref)
    except NotFound:
        return None
    if table.table_type != 'TABLE':
        raise PermanentSourceError(
            f'{ref} is a {table.table_type}, not a table; choose another '
            f'table name.', table=ref)
    return table


def require_dataset(client: Any, dataset: str, location: str) -> None:
    from google.api_core.exceptions import Forbidden, NotFound
    try:
        found = client.get_dataset(dataset)
    except NotFound:
        raise SourceNotFoundError(
            f'BigQuery dataset {dataset} does not exist or is not visible to '
            f'this identity. Create it in location {location} or point the '
            f'connector at an existing dataset.', table=dataset) from None
    except Forbidden:
        raise SourceAccessDeniedError(
            f'This identity cannot read BigQuery dataset {dataset}; grant '
            f'roles/bigquery.dataEditor on it.', table=dataset) from None
    if (found.location or '').casefold() != location.casefold():
        raise PermanentSourceError(
            f'BigQuery dataset {dataset} is in location {found.location}, but '
            f'the connector uses {location}; set '
            f'location={found.location!r}.', table=dataset)


def probe_write_access(
    client: Any,
    dataset: str,
    table_name: str,
    *,
    replace: bool,
    create: bool,
    probe_stream: Optional[Callable[[str], None]] = None,
) -> Any:
    """Checks the identity can write ``table_name`` in ``dataset``, and
    create tables there when ``create`` is set or the table is missing.
    ``probe_stream`` is called with the table a streamed write would write
    first: the destination, or else the probe table.

    Returns the existing destination table, if any.
    """
    from google.api_core.exceptions import Forbidden

    ref = f'{dataset}.{table_name}'
    existing = existing_table(client, ref)
    if existing is not None:
        required = write_permissions(replace=replace)
        granted = set(
            client.test_iam_permissions(ref, required).get('permissions', ()))
        missing = sorted(set(required) - granted)
        if missing:
            raise MissingWritePermissions(
                f'This identity cannot {"overwrite" if replace else "append"} '
                f'BigQuery table {ref}: missing {", ".join(missing)}.',
                table=ref)
        if probe_stream is not None and not create:
            probe_stream(ref)
    if create or existing is None:
        probe = bigquery.Table(
            f'{dataset}.__kumo_write_probe_{uuid.uuid4().hex}')
        probe.expires = datetime.now(timezone.utc) + timedelta(hours=1)
        try:
            client.create_table(probe)
        except Forbidden:
            raise SourceAccessDeniedError(
                f'This identity cannot create tables in BigQuery dataset '
                f'{dataset}; grant roles/bigquery.dataEditor on it.',
                table=dataset) from None
        try:
            if probe_stream is not None:
                probe_stream(str(probe.reference))
        finally:
            _drop_probe(client, probe, dataset)
    return existing


def _drop_probe(client: Any, probe: Any, dataset: str) -> None:
    try:
        client.delete_table(probe, not_found_ok=True)
    except Exception as error:
        raise ProbeCleanupError(
            f'Write probe cleanup failed ({type(error).__name__}); the '
            f'probe expires automatically.', table=dataset) from None


def require_job_access(client: Any, project: str) -> None:
    from google.api_core.exceptions import Forbidden
    try:
        client.query('SELECT 1',
                     job_config=bigquery.QueryJobConfig(dry_run=True))
    except Forbidden:
        raise SourceAccessDeniedError(
            f'This identity cannot run BigQuery jobs in project {project}; '
            f'grant roles/bigquery.jobUser there or set billing_project_id.',
            table=project) from None


# Jobs ########################################################################


def _run(job: Any, ref: str) -> None:
    from google.api_core.exceptions import BadRequest

    try:
        job.result()
    except BadRequest as error:
        detail = '; '.join(
            str(item.get('message'))
            for item in (job.errors or error.errors or ())
            if item.get('message')) or error.message
        raise PermanentSourceError(
            f'BigQuery {job.job_type} job {job.job_id} into {ref} failed: '
            f'{detail}', table=ref) from None


def _create_table(client: Any, table: Any) -> None:
    from google.api_core.exceptions import BadRequest
    try:
        client.create_table(table)
    except BadRequest as error:
        raise PermanentSourceError(
            f'Cannot create BigQuery table {table.reference}: '
            f'{error.message}', table=str(table.reference)) from None


def _like(table: Any, existing: Any) -> None:
    """Writes into a partitioned or clustered table must restate both."""
    if existing is not None:
        table.time_partitioning = existing.time_partitioning
        table.range_partitioning = existing.range_partitioning
        table.clustering_fields = existing.clustering_fields


def _restore_column_metadata(client: Any, ref: str, existing: Any) -> None:
    """A query's result carries no column descriptions or policy tags."""
    table = client.get_table(ref)
    schema = _keep_column_metadata(list(table.schema), existing.schema)
    if schema != list(table.schema):
        table.schema = schema
        client.update_table(table, ['schema'])


def load_uris(client: Any, uris: str | list[str], ref: str, *, replace: bool,
              schema: Optional[Sequence[Any]] = None) -> None:
    job = client.load_table_from_uri(
        uris, ref, job_config=load_job_config(replace=replace, schema=schema))
    _run(job, ref)


# Transports ##################################################################


def staged_write(
    client: Any,
    storage: Any,
    location: StagingLocation,
    files: Sequence[Path],
    ref: str,
    *,
    replace: bool,
    existing: Any = None,
    extra_columns: Optional[Mapping[str, Any]] = None,
    worker_ceiling: Optional[int] = None,
) -> None:
    """Uploads conformed shards under ``location`` and loads them with one
    job. The prefix is unique and cleared first, so its wildcard names exactly
    these objects; listing 2,000 URIs instead made the same load ~50x
    slower."""
    files = nonempty(files)
    layout = plan(files, extra_columns, existing, replace=replace, ref=ref)
    with tempfile.TemporaryDirectory(prefix='kumo-bq-export-') as scratch:
        staging.staged_write(
            storage, location=location, files=[
                partial(conformed, path, layout.target, extra_columns, scratch)
                for path in files
            ], load=lambda _: load_uris(client, location.object_uri(), ref,
                                        replace=replace, schema=layout.schema),
            worker_ceiling=worker_ceiling)


def streamed_write(
    client: Any,
    write_client: Any,
    files: Sequence[Path],
    ref: str,
    *,
    replace: bool,
    existing: Any = None,
    extra_columns: Optional[Mapping[str, Any]] = None,
    credentials: Any = None,
    worker_ceiling: Optional[int] = None,
) -> None:
    """Appends conformed shards through Storage Write API pending streams and
    commits them at once. ``APPEND`` commits into the destination;
    ``OVERWRITE`` commits into an expiring table and replaces the destination
    with one ``WRITE_TRUNCATE`` query over it, which keeps the destination's
    partitioning and clustering; column descriptions and policy tags are
    restored right after. A copy job would be free but skips rows still in
    the streaming buffer."""
    files = nonempty(files)
    layout = plan(files, extra_columns, existing, replace=replace, ref=ref)
    target = layout.target
    if layout.schema is None:
        unmapped = [field.name for field in target if bq_field(field) is None]
        raise PermanentSourceError(
            f'Cannot stream columns {", ".join(unmapped)} to BigQuery; '
            f'configure a GCS staging_uri to load them instead.', table=ref)
    project, dataset, _ = ref.split('.')
    created = None
    if not replace and existing is not None:
        destination = ref
    else:
        destination = (f'{project}.{dataset}.{TEMP_TABLE_PREFIX}'
                       f'{uuid.uuid4().hex}' if replace else ref)
        created = bigquery.Table(destination, schema=layout.schema)
        if replace:
            created.expires = datetime.now(timezone.utc) + TEMP_TABLE_TTL
        _create_table(client, created)
    keep = False
    try:
        table = bigquery.TableReference.from_string(destination)
        parent = write_client.table_path(table.project, table.dataset_id,
                                         table.table_id)
        names = streaming.write_streams(write_client, parent, target, [
            partial(conformed_batches, path, target, extra_columns)
            for path in files
        ], credentials=credentials, worker_ceiling=worker_ceiling)
        streaming.commit_streams(write_client, parent, names)
        keep = not replace
        if replace:
            # Copy jobs skip rows still in the streaming buffer; a query
            # reads them.
            config = bigquery.QueryJobConfig(
                destination=ref,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
            _like(config, existing)
            _run(client.query(f'SELECT * FROM `{destination}`', config), ref)
            if existing is not None:
                _restore_column_metadata(client, ref, existing)
    finally:
        if created is not None and not keep:
            with suppress(Exception):
                client.delete_table(destination, not_found_ok=True)
