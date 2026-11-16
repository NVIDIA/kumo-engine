# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from kumo_core.training import custom_table
from kumo_core.training import export as training_table_export
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType
from pyarrow.fs import FileSystem
from temporalio import activity

from io_utils import fsutil, logging
from io_utils.errors import restricted_diagnostic_boundary
from io_utils.pyarrow import arrow_cloud_filesystem
from kumo.connector.file_connector import FileConnector
from kumo.validation import ValidationError, ValidationResponse
from rest.config.training_job import (
    CUSTOM_TT_ALLOW_ROW_COUNT_MISMATCH,
    CustomTrainingTable,
)
from rest.utils.driver_common import get_connector_from_spec
from service_framework.deploy_env_config import DeployEnvConfig
from statedb.utils.errors import UnsupportedInDiskgraphError
from temporalio_commons.heartbeat_utils import auto_heartbeater

from ..connector_activity_registry import connector_authorized_activity
from .ingest_table_data_activity_v2 import SAFE_TABLE_INGESTION_ERROR_MESSAGE

log = logging.getLogger(__name__)


def anchor_is_sentinel(
    task_type: TaskType,
    query_type: QueryType,
    list_distinct_target_is_array: bool,
    entity_time_col: Optional[str],
) -> bool:
    """Adapt hosted query facts to the shared anchor decision."""
    return custom_table.anchor_is_sentinel(
        task_type,
        temporal=query_type == QueryType.TEMPORAL,
        entity_has_time_column=entity_time_col is not None,
        list_distinct_target_is_array=list_distinct_target_is_array,
    )


def _synthesize_sentinel_timestamp(
    parquet_dir: str,
    *,
    task_type: TaskType,
    query_type: QueryType,
    list_distinct_target_is_array: bool,
    entity_time_col: Optional[str],
) -> None:
    """Append a zero-filled anchor column to customer-ingested
    parquet parts when DG would have emitted one (static node-pred
    without entity_time_col, or static LP with array target), so the
    weighted-TT validator's schema-parity check matches. Per-part
    atomic via tmp + os.replace.
    """
    if not anchor_is_sentinel(task_type, query_type,
                              list_distinct_target_is_array, entity_time_col):
        return

    parquet_files = _list_parquet_files(parquet_dir)
    if not parquet_files:
        log.info(
            "Sentinel-TIMESTAMP synthesis: no parquet files under %s; "
            "skipping.", parquet_dir)
        return

    # Footer-read to skip parts that already have TIMESTAMP, then
    # rewrite the rest in parallel (each part is I/O-bound).
    targets: List[str] = []
    for parquet_path in parquet_files:
        if not _append_zero_timestamp_supported(parquet_path):
            raise UnsupportedInDiskgraphError(
                f"Sentinel-TIMESTAMP synthesis is only supported for "
                f"local, S3, GCS, and ADLS parquet paths; got {parquet_path}. "
                f"UCV-backed custom training tables must be staged to "
                f"one of the supported backends before ingest.")
        schema_names = set(_read_parquet_schema([parquet_path]).names)
        if custom_table.timestamp_column() not in schema_names:
            targets.append(parquet_path)

    if not targets:
        return

    log.info(
        "Sentinel-TIMESTAMP synthesis: appending zero-filled TIMESTAMP to "
        "%d part(s) under %s (task_type=%s, query_type=%s, "
        "list_distinct_target_is_array=%s)", len(targets), parquet_dir,
        task_type, query_type, list_distinct_target_is_array)
    _SYNTHESIZE_MAX_WORKERS = min(16, len(targets))
    if _SYNTHESIZE_MAX_WORKERS <= 1:
        for parquet_path in targets:
            _append_zero_timestamp_column(parquet_path)
        return

    with ThreadPoolExecutor(max_workers=_SYNTHESIZE_MAX_WORKERS) as pool:
        futures = [
            pool.submit(_append_zero_timestamp_column, p) for p in targets
        ]
        for fut in futures:
            fut.result()


def _append_zero_timestamp_supported(parquet_path: str) -> bool:
    """Whether :func:`_append_zero_timestamp_column` can rewrite ``parquet_path``
    in place. Local / S3 / GCS / ADLS are supported.
    """
    return (fsutil.is_local(parquet_path) or fsutil.is_s3(parquet_path)
            or fsutil.is_gcs(parquet_path) or fsutil.is_adls(parquet_path))


def _append_zero_timestamp_column(parquet_path: str) -> None:
    """Append an int64 zero-filled ``TIMESTAMP`` column to ``parquet_path``
    via a row-group-streaming rewrite. Local uses atomic rename;
    remote backends (S3, GCS, ADLS) use tmp+move — a Temporal retry
    mid-move is handled idempotently by the schema-presence check.
    """
    fs, bare = _filesystem_and_path(parquet_path)
    if fs is not None:
        tmp_bare = bare + '.tmp'
        # Pre-clean any leftover .tmp from a prior crashed run.
        try:
            fs.delete_file(tmp_bare)
        except Exception:
            pass
        try:
            _stream_append_zero_timestamp(bare, tmp_bare, filesystem=fs)
        except Exception:
            try:
                fs.delete_file(tmp_bare)
            except Exception:
                pass
            raise
        fs.move(tmp_bare, bare)
        return

    tmp_path = parquet_path + '.tmp'
    # Pre-clean any leftover .tmp from a prior crashed run.
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    try:
        _stream_append_zero_timestamp(parquet_path, tmp_path, filesystem=None)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    os.replace(tmp_path, parquet_path)


def _stream_append_zero_timestamp(
    src_path: str,
    dst_path: str,
    *,
    filesystem: Optional[FileSystem],
) -> None:
    """Copy ``src_path`` to ``dst_path`` one row group at a time with a
    zero-filled ``TIMESTAMP`` int64 column appended. Peak memory is
    bounded by the row-group size.
    """
    if filesystem is not None:
        pf = pq.ParquetFile(src_path, filesystem=filesystem)
    else:
        pf = pq.ParquetFile(src_path)
    src_schema = pf.schema_arrow
    new_field = pa.field(custom_table.timestamp_column(), pa.int64())
    dst_schema = pa.schema(list(src_schema) + [new_field])

    writer_kwargs = {}
    if filesystem is not None:
        writer_kwargs['filesystem'] = filesystem
    with pq.ParquetWriter(dst_path, dst_schema, **writer_kwargs) as writer:
        for rg_idx in range(pf.num_row_groups):
            rg_table = pf.read_row_group(rg_idx, use_threads=True)
            zero_col = pa.array([0] * rg_table.num_rows, type=pa.int64())
            rg_table = rg_table.append_column(custom_table.timestamp_column(),
                                              zero_col)
            writer.write_table(rg_table)


def read_only_custom_sql(source_table, connector) -> str:
    """Validate and normalize a Custom SQL training source before it is
    embedded in the connector's subquery position, the same boundary the
    graph-ingest activity applies (``ingest_table_data_activity_v2``).
    """
    from kumo_core.types import ConnectorType

    from kumo.data.custom_sql import (
        advisory_read_only_check,
        subquery_safe_sql,
        validate_connector_sql_scope,
    )
    connector_type = ConnectorType(
        str(
            getattr(connector._get_connector_type(), 'value',
                    connector._get_connector_type())).casefold())
    advisory_read_only_check(source_table.sql, connector_type)
    validate_connector_sql_scope(source_table.sql, connector_type, connector)
    return subquery_safe_sql(source_table.sql, connector_type)


def require_custom_sql_support(source_table, connector) -> None:
    """Refuse a Custom SQL source whose exporter cannot read a query.

    Gated on what the callee accepts, not on what the source carries. Only
    Databricks' `export_table_read_only` grew a `sql` parameter; Snowflake's
    is `(source_name, destination, cols)` and BigQuery's
    `export_table_to_blob_storage` is `(table_name, dest_path, columns)`.
    Keying the kwarg off the source alone hands an unexpected keyword to the
    two connectors that cannot take it, and the broad `except Exception`
    below reports that `TypeError` as the generic sanitized ingest failure,
    which cannot tell the user what is wrong.

    The capability is declared on the connector rather than read off
    `export`: `@blocking_method()` wraps every export in an `RpcFuncSync`
    whose signature is `(*args, **kwargs)`, so `inspect.signature` shows no
    real parameter for any connector and cannot tell the three apart.

    Raised before that `try` so the reason survives to the user. Reachable
    only if the Custom SQL gates in `pquery_api_driver._from_custom_tt` and
    `TrainingTable.update` are widened past Databricks; this makes that
    widening fail loudly at the connector that cannot serve it.
    """
    if not getattr(source_table, 'sql', None):
        return
    if getattr(connector, 'exports_custom_sql', False):
        return
    raise UnsupportedInDiskgraphError(
        'Custom SQL sources are currently supported only for Databricks '
        f'connectors; {type(connector).__name__} cannot export a query.')


@connector_authorized_activity
@auto_heartbeater
def ingest_custom_train_table(
        custom_train_table_spec: CustomTrainingTable,
        custom_train_table_path: str,
        deploy_config: DeployEnvConfig,  # noqa: ARG001 — workflow signature
        id: str,  # noqa: ARG001 — workflow signature
        task_type: TaskType,
        query_type: QueryType,
        list_distinct_target_is_array: bool,
        entity_time_col: Optional[str] = None,  # noqa: ARG001
) -> Tuple[str, ValidationResponse]:
    """Stage a customer-supplied training table for diskgraph rehydration.

    Connector-aware dispatch:

    - ``FileConnector`` (Parquet/CSV): copy source into
      ``custom_train_table_path`` as a directory of parquet parts.
    - Warehouse connectors with ``export_table_read_only`` acquire source
      data through their provider reader. Other warehouses use
      ``export_table_to_blob_storage``.
    - Anything else: refuse with :class:`UnsupportedInDiskgraphError`.

    Appends a zero-filled ``TIMESTAMP`` column when the task shape
    requires it, so the weighted-TT validator's schema-parity check
    passes.
    """
    response = ValidationResponse()
    from rest.utils.driver_common import extract_source_table_type
    source_table, table_name, _ = extract_source_table_type(
        custom_train_table_spec.source_table)
    log.info("Ingesting custom training table %s into %s", table_name,
             custom_train_table_path)

    try:
        connector = get_connector_from_spec(
            custom_train_table_spec.connector_spec)
    except Exception as error:
        log.error(
            'Custom training table connector resolution failed '
            '(error_type=%s)',
            type(error).__name__)
        response.errors.append(
            ValidationError(
                title="Failed to ingest custom training table",
                message=SAFE_TABLE_INGESTION_ERROR_MESSAGE,
            ))
        return custom_train_table_path, response

    try:
        if isinstance(connector, FileConnector):
            # Customer data is downloaded only under the exact connector
            # identity.  Kumo-owned persistence then happens after that scope
            # exits, which is essential when both locations use S3 or GCS.
            try:
                with connector._local_source(table_name) as (staged, _):
                    from kumo.connector.source_bridge import publish_source
                    publish_source(staged, custom_train_table_path)
                _synthesize_sentinel_timestamp(
                    custom_train_table_path,
                    task_type=task_type,
                    query_type=query_type,
                    list_distinct_target_is_array=list_distinct_target_is_array,
                    entity_time_col=entity_time_col,
                )
            except Exception as error:
                log.error(
                    'Custom training table file ingest failed '
                    '(error_type=%s)',
                    type(error).__name__)
                response.errors.append(
                    ValidationError(
                        title="Failed to ingest custom training table",
                        message=SAFE_TABLE_INGESTION_ERROR_MESSAGE,
                    ))
            return custom_train_table_path, response

        read_only_export = getattr(connector, 'export_table_read_only', None)
        warehouse_export = getattr(connector, 'export_table_to_blob_storage',
                                   None)
        if read_only_export is not None or warehouse_export is not None:
            export = (read_only_export
                      if read_only_export is not None else warehouse_export)
            # Outside the try: an exporter that cannot read a query is a
            # configuration error the user can act on, not a warehouse
            # failure to sanitize.
            require_custom_sql_support(source_table, connector)
            # Pre-clean: not every warehouse export overwrites, so a
            # stale part-*.parquet could silently ingest stale rows.
            try:
                # Passed only when the source is a query and the exporter
                # accepts one; `require_custom_sql_support` has already
                # refused the combination that cannot be served.
                optional = ({
                    'sql': read_only_custom_sql(source_table, connector)
                } if getattr(source_table, 'sql', None) else {})
                fsutil.rm(custom_train_table_path, ignore_errors=True)
                connector.open()
                export(table_name, custom_train_table_path, **optional)
                if not _list_parquet_files(custom_train_table_path):
                    response.errors.append(
                        ValidationError(
                            title="Custom training table is empty",
                            message=(f"The custom training table {table_name!r} "
                                     "produced no parquet files.")))
                    return custom_train_table_path, response
                _synthesize_sentinel_timestamp(
                    custom_train_table_path,
                    task_type=task_type,
                    query_type=query_type,
                    list_distinct_target_is_array=list_distinct_target_is_array,
                    entity_time_col=entity_time_col,
                )
            except Exception as error:
                log.error(
                    'Custom training table warehouse ingest failed '
                    '(error_type=%s)',
                    type(error).__name__)
                response.errors.append(
                    ValidationError(
                        title="Failed to ingest custom training table",
                        message=SAFE_TABLE_INGESTION_ERROR_MESSAGE,
                    ))
            return custom_train_table_path, response

        raise UnsupportedInDiskgraphError(
            f"Custom training table ingest is not supported for connector "
            f"type {type(connector).__name__}. Use a FileConnector "
            f"(Parquet/CSV) or a warehouse connector "
            f"(Snowflake/BigQuery/Databricks).")
    finally:
        try:
            connector.close()
        except Exception as error:
            log.warning(
                'Custom training table connector close failed '
                '(error_type=%s)',
                type(error).__name__,
            )


def _list_parquet_files(path: str) -> list:
    """Recursively list ``*.parquet`` files under ``path``. Returns ``[]``
    if ``path`` doesn't exist. Defensive against fsutil impls that
    return basenames instead of full paths.
    """
    if not fsutil.exists(path):
        return []
    if not fsutil.isdir(path):
        return [path] if path.endswith('.parquet') else []
    out = []
    for entry in fsutil.listdir(path):
        # Defensive: tolerate impls that return basenames.
        if '/' not in entry and not entry.startswith(path):
            entry = osp.join(path, entry)
        if fsutil.isdir(entry):
            out.extend(_list_parquet_files(entry))
        elif entry.endswith('.parquet'):
            out.append(entry)
    return out


@contextlib.contextmanager
def _materialized_parquet_files(path: str) -> Iterator[List[str]]:
    """Yield parquet paths for ``path``. S3 / GCS / ADLS / local
    paths are yielded as-is; UCV is staged to a temp dir because
    PyArrow has no Databricks Unity Catalog Volume backend.
    """
    if fsutil.is_ucv(path):
        with fsutil.TemporaryDirectory() as temp_dir:
            if fsutil.isdir(path):
                for sub in fsutil.listdir(path, suffix='parquet'):
                    fsutil.copy_file(sub, temp_dir)
            else:
                fsutil.copy_file(path, temp_dir)
            yield _list_parquet_files(temp_dir)
        return

    if (fsutil.is_local(path) or fsutil.is_s3(path) or fsutil.is_gcs(path)
            or fsutil.is_adls(path)):
        yield _list_parquet_files(path)
        return

    raise ValueError(
        f"Unsupported train table path '{path}'. Operation only supports "
        "local, S3, GCS, ADLS, and UCV paths.")


def exported_reference(
    path: str,
    split_col: Optional[str],
    column_names: List[str],
) -> Optional[Tuple[List[str], pa.Schema]]:
    """Exported stage rows, when the supplied table carries its split."""
    split = (split_col or '').strip()
    if not split or split not in column_names:
        return None
    return _generated_export(path)


def exported_reference_rows(path: str, split_col: Optional[str],
                            column_names: List[str]) -> Optional[int]:
    exported = exported_reference(path, split_col, column_names)
    return None if exported is None else _count_parquet_rows(exported[0])


def _generated_export(path: str) -> Optional[Tuple[List[str], pa.Schema]]:
    """Seed parts and exported columns of a generated bundle's table."""
    bundle = osp.dirname(path.rstrip('/'))
    spec = osp.join(bundle, training_table_export.INGEST_SPEC)
    if fsutil.is_ucv(path) or not fsutil.exists(spec):
        return None
    stages = training_table_export.stage_parts(
        json.loads(fsutil.read_text(spec)))
    files = [osp.join(bundle, part) for parts in stages.values()
             for part in parts]
    schema = training_table_export.export_schema(_read_parquet_schema(files))
    return files, schema.remove(
        schema.get_field_index(training_table_export.SPLIT_COLUMN))


def _filesystem_and_path(path: str) -> Tuple[Optional[FileSystem], str]:
    """Return a ``(pyarrow.fs.FileSystem, bare_path)`` pair for ``path``,
    or ``(None, path)`` for plain local paths.
    """
    if fsutil.is_s3(path) or fsutil.is_gcs(path):
        # Route S3 / GCS through the credentialed, fork-safe choke point.
        # Normalize s3a first so the strip and filesystem both see s3://.
        path = fsutil.s3a_as_s3(path)
        bare = path.removeprefix("s3://").removeprefix("gs://")
        return arrow_cloud_filesystem(path), bare
    if fsutil.is_adls(path):
        fs, bare = FileSystem.from_uri(path)
        return fs, bare
    return None, path


def _open_parquet_file(path: str) -> pq.ParquetFile:
    """Open ``path`` as a footer-only ``ParquetFile``."""
    fs, bare = _filesystem_and_path(path)
    if fs is not None:
        return pq.ParquetFile(bare, filesystem=fs)
    return pq.ParquetFile(path)


def _read_parquet_schema(parquet_files: List[str]) -> pa.Schema:
    """The schema of the first part, off its footer.

    ``kumo_core``'s reading of it, not pyarrow's: a connector export may land
    a dictionary-encoded column that pyarrow reports as its value type, which
    would then pass every gate and be dropped row by row at ingest. The
    in-process backend reads the same footer the same way.
    """
    parquet = _open_parquet_file(parquet_files[0])
    return custom_table.stored_schema(parquet.schema_arrow,
                                      parquet.metadata.metadata)


_DIVERGENCE_SAMPLE = 5


def _schema_divergence_errors(first: pa.Schema,
                              parquet_files: List[str]) -> list:
    """Refuses parts that do not store what :obj:`first` says they do.

    Every gate here reads part zero's footer while the row scan reads them
    all, and a projected read of a part that lacks a requested column omits
    it silently rather than failing -- so a well-formed part zero would carry
    a part with no ``TARGET1``, or with ``ENTITY1`` stored as a string, past
    the whole check. Footers only; no row group is read.

    Reported errors are capped at :data:`_DIVERGENCE_SAMPLE` parts; a table
    with thousands of divergent parts gets a count of them, not one error
    per column per part.
    """
    errors = []
    affected = 0
    for path in parquet_files[1:]:
        schema = _read_parquet_schema([path])
        part_errors = []
        for field_ in first:
            if field_.name not in schema.names:
                part_errors.append(
                    ValidationError(
                        title="Column missing in a part of the supplied table",
                        message=(
                            f"Part {path} does not carry {field_.name!r}, "
                            f"which the other parts do. Every part of a "
                            f"supplied training table has to store the same "
                            f"columns.")))
            elif schema.field(field_.name).type != field_.type:
                part_errors.append(
                    ValidationError(
                        title="Column type differs between parts",
                        message=(f"Part {path} stores {field_.name!r} as "
                                 f"{schema.field(field_.name).type}, and "
                                 f"another part stores it as {field_.type}."),
                    ))
        if not part_errors:
            continue
        affected += 1
        if affected <= _DIVERGENCE_SAMPLE:
            errors.extend(part_errors)
    if affected > _DIVERGENCE_SAMPLE:
        errors.append(
            ValidationError(
                title="More parts diverge from the first",
                message=(f"{affected} parts of the supplied table diverge from "
                         f"the first; only the first {_DIVERGENCE_SAMPLE} are "
                         f"listed above.")))
    return errors


def missing_source_column_errors(
    source_column_names: List[str],
    generated_table_path: str,
    weight_col_name: Optional[str],
    split_col_name: Optional[str] = None,
) -> list:
    """Refuse a custom source that does not name what the generated table does.

    Names only, and the same rule on every connector. Read before an export
    is paid for, so the most common mistake -- a column called ``ENTITY``
    where DiskGraph reads ``ENTITY1`` -- is refused at author time rather
    than after the rows have moved.

    The anchor is excluded: a static query's is a sentinel that ingest writes,
    so a source is not expected to carry one yet.
    """
    with restricted_diagnostic_boundary(
            safe_message='Custom training table schema read failed',
            exception_types=OSError):
        exported = exported_reference(generated_table_path, split_col_name,
                                      source_column_names)
        if exported is not None:
            generated_schema = exported[1]
        else:
            with _materialized_parquet_files(generated_table_path) as files:
                if not files:
                    raise ValueError('The generated training table is empty.')
                generated_schema = _read_parquet_schema(files)

    return custom_table.missing_column_errors(
        source_column_names,
        required=custom_table.generated_required_columns(
            generated_schema.names, anchor_optional=True),
        weight_col=weight_col_name,
        split_col=split_col_name,
    )


def _count_parquet_rows(parquet_files: List[str]) -> int:
    """Sum ``num_rows`` across the footer metadata of every parquet
    part. Footer-only — no data is read."""
    total = 0
    for f in parquet_files:
        total += _open_parquet_file(f).metadata.num_rows
    return total


def _validate_supplied_train_table(
    generated_table_path: Optional[str],
    supplied_table_path: str,
    weight_col_name: Optional[str],
    task_type: TaskType,
    split_col_name: Optional[str] = None,
    anchored: bool = True,
) -> ValidationResponse:
    """Check schema and generated-table correspondence from parquet footers.

    DiskGraph validates row contents during ingestion and applies its
    ``ingest_on_error`` policy there.
    """
    response = ValidationResponse()

    with contextlib.ExitStack() as stack:
        supplied_files = stack.enter_context(
            _materialized_parquet_files(supplied_table_path))
        gen_files = (None
                     if generated_table_path is None else stack.enter_context(
                         _materialized_parquet_files(generated_table_path)))
        if not supplied_files:
            response.errors.append(
                ValidationError(
                    title="Supplied training table is empty", message=(
                        f"Supplied training table at {supplied_table_path} "
                        "contains no parquet files.")))
            return response
        if gen_files is not None and not gen_files:
            response.errors.append(
                ValidationError(
                    title="Generated training table is empty", message=(
                        f"Generated training table at {generated_table_path} "
                        "contains no parquet files.")))
            return response

        schema = _read_parquet_schema(supplied_files)
        divergent = _schema_divergence_errors(schema, supplied_files)
        if divergent:
            response.errors.extend(divergent)
            return response

        exported = (None if generated_table_path is None else
                    exported_reference(generated_table_path, split_col_name,
                                       schema.names))
        if exported is not None:
            gen_files, generated_schema = exported
        else:
            generated_schema = (None if gen_files is None else
                                _read_parquet_schema(gen_files))
        rules = custom_table.SuppliedTableRules(
            task_type=task_type,
            weight_col=weight_col_name,
            split_col=(split_col_name or '').strip() or None,
            anchored=(anchored if generated_schema is None else
                      custom_table.timestamp_column()
                      in generated_schema.names),
            allow_row_count_mismatch=(
                CUSTOM_TT_ALLOW_ROW_COUNT_MISMATCH.parse_value()),
        )

        # The contract's names here; a generated column that went missing is
        # reported by the correspondence check below, which can say where it
        # came from. The same call the in-process backend makes.
        errors = custom_table.missing_column_errors(
            schema.names,
            required=custom_table.contract_required_columns(rules),
            weight_col=weight_col_name,
            split_col=rules.split_col,
        )
        errors += custom_table.column_type_errors(schema, rules)
        errors += custom_table.weight_arity_errors(schema, rules)
        if errors:
            response.errors.extend(errors)
            return response

        if generated_schema is None:
            return response

        supplied_rows = _count_parquet_rows(supplied_files)
        generated = custom_table.GeneratedTable(
            schema=generated_schema,
            rows=_count_parquet_rows(gen_files),
        )
        errors, warnings = custom_table.correspondence_errors(
            schema, supplied_rows, generated, rules)
        if rules.allow_row_count_mismatch and supplied_rows != generated.rows:
            log.warning(
                "Custom training table row count mismatch allowed by %s: "
                "supplied table has %d rows, generated table has %d rows.",
                CUSTOM_TT_ALLOW_ROW_COUNT_MISMATCH.name, supplied_rows,
                generated.rows)
        response.errors.extend(errors)
        response.warnings.extend(warnings)

    return response


@activity.defn
@auto_heartbeater
def validate_weighted_train_table(
    original_train_table_path: Optional[str],
    custom_train_table_path: str,
    weight_col_name: Optional[str],
    deploy_config: DeployEnvConfig,
    id: str,
    task_type: TaskType,
    split_col_name: Optional[str] = None,
    anchored: bool = True,
) -> ValidationResponse:
    """Validate the supplied training table a connector has just produced."""
    with restricted_diagnostic_boundary(
            safe_message='Supplied training table read failed',
            exception_types=OSError):
        return _validate_supplied_train_table(
            generated_table_path=original_train_table_path,
            supplied_table_path=custom_train_table_path,
            weight_col_name=weight_col_name,
            task_type=task_type,
            split_col_name=split_col_name,
            anchored=anchored,
        )
