# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Contains functions to write processed prediction artifacts to customer
data warehouses and S3. All the functions in this module assume that the
input artifact is a parquet-formatted table, and no additional processing
is required.
"""
import os.path as osp
import uuid
from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd
from kumo_core.artifact_export import (
    MultiDimensionalCsvError,
    flatten_arrays_for_csv,
)

from io_utils import fsutil, logging, parquet_util
from io_utils.retry import retry
from kumo.connector import (
    BigQueryConnector,
    DatabricksConnector,
    SnowflakeConnector,
)
from kumo.connector.bigquery_connector import BigQueryOutputSchemaError
from kumo.connector.connector import ConnectorError
from rest.backends.ml_backend.utils import (
    canonical_file_path,
    is_partitioned,
    validated_file_type,
)
from rest.config.file_type import FileType
from rest.config.jobs import WriteMode
from statedb.utils.errors import ArtifactExportError
from temporalio_commons.error_utils import is_error_non_retryable

log = logging.getLogger(__name__)

_PARQUET_COPY_MAX_WORKERS = 8


@dataclass
class OutputStats:
    size_bytes: int
    display_path: str


class _StagingTableConnector(Protocol):
    def drop_table_if_exists(self, table_name: str) -> None:
        ...


def _drop_staging_table(
    connector: _StagingTableConnector,
    table_name: str,
) -> None:
    try:
        connector.drop_table_if_exists(table_name)
    except Exception as error:
        log.warning(
            'Output staging-table cleanup failed (error_type=%s)',
            type(error).__name__,
        )


CSVShardConverter = Callable[[str, str], None]


def _artifact_size(path: str) -> int:
    """Return size for either a single Parquet file or a dataset directory."""
    return fsutil.dirsize(path) if fsutil.isdir(path) else fsutil.filesize(path)


def write_artifact_to_s3(
    artifact_path: str,
    out_path: str,
    file_type: FileType | None = None,
    *,
    artifact_files: list[str] | None = None,
    csv_shard_converter: CSVShardConverter | None = None,
    export_id: str | None = None,
) -> OutputStats:
    r"""
    Write processed prediction artifact to S3.
    Args:
        artifact_path (str): Internal path to the artifact.
        out_path (str): Output path.
        file_type (FileType|None): File type of the out_path.
        csv_shard_converter: Optional credential-free converter used to stage
            one parquet shard as a local CSV file. The caller may use this
            seam to isolate Pandas/PyArrow allocation in a disposable process;
            destination upload always remains in this process.
    Returns:
        OutputStats: Output stats.
    """
    from kumo_core import artifact_export as shared_export

    if artifact_files is not None:
        try:
            size, display = shared_export.export_selected_files(
                artifact_files,
                out_path,
                fs=fsutil,
                temporary_directory=fsutil.TemporaryDirectory,
                identity=export_id or artifact_path,
                file_type=validated_file_type(out_path,
                                              file_type).value.lower(),
                convert_csv=csv_shard_converter
                or _convert_parquet_to_local_csv,
                copy_parquet=lambda files,
                destination: _write_artifact_to_s3_parquet(
                    artifact_path, destination, artifact_files=files),
            )
        except shared_export.ArtifactDestinationConflict as error:
            raise ArtifactExportError(message=str(error),
                                      non_retryable=True) from None
        return OutputStats(size, display)

    # Legacy scanning/copy rules stay with this adapter. Destination ownership
    # and publication are the same lower-level decisions as selected export.
    multi_sharded_artifact = (fsutil.isdir(artifact_path) and len(
        fsutil.listdir(artifact_path, max_items=10)) > 1)
    file_type = validated_file_type(out_path, file_type)
    try:
        out_path, _ = shared_export.prepare_export_destination(
            out_path, output_is_dir=is_partitioned(out_path)
            or multi_sharded_artifact, file_type=file_type.value.lower(),
            fs=fsutil, identity=export_id or artifact_path)
    except shared_export.ArtifactDestinationConflict as error:
        raise ArtifactExportError(message=str(error),
                                  non_retryable=True) from None

    # When writing to customer s3 buckets, this flag is necessary for
    # customers to read the resulting data
    storage_options = None
    if fsutil.is_s3(out_path):
        storage_options = {
            "s3_additional_kwargs": {
                "ACL": "bucket-owner-full-control"
            }
        }
    if file_type == FileType.CSV:
        _write_artifact_to_s3_csv(
            artifact_path, out_path, storage_options, csv_shard_converter
            or _convert_parquet_to_local_csv, artifact_files=artifact_files)
    elif file_type == FileType.PARQUET:
        _write_artifact_to_s3_parquet(artifact_path, out_path,
                                      artifact_files=artifact_files)
    else:
        raise ArtifactExportError(
            message=(f"Unsupported file type '{file_type}' for S3 export. "
                     "Please use 'parquet' or 'csv' format."),
            non_retryable=True,
        )

    return OutputStats(
        *shared_export.publish_export_destination(out_path, fs=fsutil))


@retry(tries=5, delay=60, backoff=2, max_delay=240, logger=log)
def _write_artifact_to_s3_parquet(
    artifact_path: str,
    output_path: str,
    *,
    artifact_files: list[str] | None = None,
):
    # Since artifact is already in parquet format, we just need to copy it to
    # the output path. Since it's just a copy operation, it's safe to retry in
    # case of intermittent AWS S3 errors like rate limitation etc.
    #
    # Note that for s3->s3 copy, s3_filesystem already sets 'ACL' to
    # 'bucket-owner-full-control' in extra args.
    if artifact_files is not None:
        from kumo_core.artifact_export import copy_selected_parquet_files
        copy_selected_parquet_files(artifact_files, output_path, fs=fsutil)
        return
    fsutil.copy(artifact_path, output_path, exclude=["_SUCCESS"],
                overwrite=True)


# Convert one parquet shard to a local CSV incrementally. This function is
# deliberately credential-free: activity code may execute it in a disposable
# child process without serializing a ConnectorOpenSpec or provider credential.
def _convert_parquet_to_local_csv(in_parquet_file: str,
                                  local_csv_file: str) -> None:
    log.info('Convert parquet to local CSV from %s to %s', in_parquet_file,
             local_csv_file)
    from kumo_core.artifact_export import parquet_to_csv
    parquet_to_csv(
        in_parquet_file, local_csv_file, batches=lambda path:
        (batch.df for batch in parquet_util.to_batches(path)),
        csv_ready=_csv_ready)


# Stage the credential-free transform locally, then upload from the parent
# process where the connector's request-local credential scope is active.
def _convert_parquet_to_s3_csv(
    in_parquet_file: str,
    out_csv_file: str,
    storage_options: dict,
    csv_shard_converter: CSVShardConverter,
) -> None:
    del storage_options  # ACL behavior is applied by the destination fsutil.
    log.info('Convert parquet to CSV from %s to %s', in_parquet_file,
             out_csv_file)

    # tempfile.TemporaryDirectory creates mode 0700 directories. Cleanup is
    # guaranteed on converter, upload, and cancellation failures.
    with fsutil.TemporaryDirectory() as temp_dir:
        temp_file = osp.join(temp_dir, osp.basename(out_csv_file))
        csv_shard_converter(in_parquet_file, temp_file)
        fsutil.copy_file(temp_file, out_csv_file, overwrite=True)


def _write_artifact_to_s3_csv(
    artifact_path: str,
    out_path: str,
    storage_options: dict[str, dict[str, str]] | None,
    csv_shard_converter: CSVShardConverter,
    *,
    artifact_files: list[str] | None = None,
):
    artifact_files = (artifact_files if artifact_files is not None else
                      [artifact_path] if not fsutil.isdir(artifact_path) else
                      fsutil.listdir(artifact_path))
    is_single_file_output = (len(artifact_files) == 1
                             and not is_partitioned(out_path))

    shard_num = 0
    for shard_num, path in enumerate(artifact_files):
        if is_single_file_output:
            filename = canonical_file_path(out_path)
        else:
            filename = osp.join(out_path, f"part-{shard_num:05}.csv")
        _convert_parquet_to_s3_csv(path, filename, storage_options or {},
                                   csv_shard_converter)


def write_artifact_to_databricks(
    artifact_path: str,
    connector: DatabricksConnector,
    output_table_name: str,
    write_mode: WriteMode,
    *,
    known_size_bytes: int | None = None,
    staged_files: list[str] | None = None,
) -> OutputStats:
    # Reject a schema outside the persisted connector target before opening a
    # provider connection or inspecting artifact data.
    schema_name, table_name = (
        connector.require_output_table_within_connector_target(
            output_table_name))
    connector.open()
    data_size = (known_size_bytes if known_size_bytes is not None else
                 _artifact_size(artifact_path))
    stage_kwargs = ({
        'staged_files': staged_files
    } if staged_files is not None else {})
    connector.write_artifacts_to_table(
        output_table_name,
        artifact_path,
        replace=(write_mode == WriteMode.OVERWRITE),
        **stage_kwargs,
    )
    display_table_name = f"{connector.catalog}.{schema_name}.{table_name}"
    return OutputStats(size_bytes=data_size, display_path=display_table_name)


def write_artifact_to_snowflake(
    artifact_path: str,
    connector: SnowflakeConnector,
    write_mode: WriteMode,
    table_name: str,
    staging_table_name: str | None = None,
    *,
    known_size_bytes: int | None = None,
    artifact_files: list[str] | None = None,
) -> OutputStats:
    """Writes artifact to Snowflake.

    Args:
        artifact_path: Internal kumo path to the artifact.
        connector: authorized Snowflake connector with credentials set
        write_mode: Write mode (APPEND or OVERWRITE)
        table_name: Name of the table to write to
        staging_table_name: Optional name of staging table for
            OVERWRITE operations
        artifact_files: The exact caller-resolved object paths to load. A
            batch-prediction selection names one attempt directory per output
            shard, so the artifact root alone addresses only the first shard.
            Omit to retain the legacy whole-prefix load.

    Returns:
        OutputStats: Statistics about the written data
    """
    if staging_table_name is not None and write_mode != WriteMode.OVERWRITE:
        raise ArtifactExportError(
            message='A Snowflake staging table may be used only with '
            'OVERWRITE mode.',
            non_retryable=True,
        )
    try:
        connector.open()
    except Exception as e:
        log.warning(
            'Snowflake output connector authentication failed (%s)',
            type(e).__name__,
        )
        raise ArtifactExportError(
            message="Unable to authenticate the Snowflake output connector.",
            non_retryable=True,
        ) from None

    try:
        from kumo_snowflake.write import export_table_files
        data_size = (known_size_bytes if known_size_bytes is not None else
                     _artifact_size(artifact_path))
        export_table_files(connector, table_name, artifact_path,
                           replace=write_mode == WriteMode.OVERWRITE,
                           artifact_files=artifact_files,
                           staging_table_name=staging_table_name)
    except Exception as e:
        log.warning('Snowflake artifact export failed (%s)', type(e).__name__)
        raise ArtifactExportError(
            message="Failed to export artifacts to Snowflake",
            non_retryable=is_error_non_retryable(e),
        ) from None

    # Irrespective of whether using a staging table or not, the display path
    # should always be the final output table name.
    display_table_name = (
        f"{connector.database}.{connector.schema_name}.{table_name}")
    return OutputStats(size_bytes=data_size, display_path=display_table_name)


def write_artifact_to_bigquery(
    artifact_path: str,
    connector: BigQueryConnector,
    write_mode: WriteMode,
    table_name: str,
    staging_table_name: str | None = None,
    *,
    known_size_bytes: int | None = None,
    artifact_files: list[str] | None = None,
) -> OutputStats:
    if staging_table_name is not None and write_mode != WriteMode.OVERWRITE:
        raise ArtifactExportError(
            message='A BigQuery staging table may be used only with '
            'OVERWRITE mode.',
            non_retryable=True,
        )
    try:
        connector.require_supported_artifact_path(artifact_path)
    except ConnectorError:
        raise ArtifactExportError(
            message='BigQuery artifact export requires a gs:// artifact path.',
            non_retryable=True,
        ) from None
    connector.require_output_table_within_connector_target(table_name)
    if staging_table_name is not None:
        connector.require_output_table_within_connector_target(
            staging_table_name)
    try:
        connector.open()
    except Exception as error:
        log.warning(
            'BigQuery output connector authentication failed (%s)',
            type(error).__name__,
        )
        raise ArtifactExportError(
            message='Unable to authenticate the BigQuery output connector.',
            non_retryable=True,
        ) from None
    data_size = (known_size_bytes if known_size_bytes is not None else
                 _artifact_size(artifact_path))

    is_overwrite = write_mode == WriteMode.OVERWRITE

    # Create a local variable for staging_table_name that we can modify
    local_staging_table_name = staging_table_name

    if (is_overwrite and local_staging_table_name is None):
        # Create a default staging table
        local_staging_table_name = f'{table_name}_STAGING'

    if local_staging_table_name is not None:
        local_staging_table_name = (
            f'{local_staging_table_name}_{uuid.uuid4().hex[:12]}')
        connector.require_output_table_within_connector_target(
            local_staging_table_name)
        output_table_name = local_staging_table_name
    else:
        output_table_name = table_name

    try:
        if artifact_files is None:
            connector.write_artifacts_to_table(output_table_name, artifact_path,
                                               replace=is_overwrite)
        else:
            connector.write_artifacts_to_table(output_table_name, artifact_path,
                                               replace=is_overwrite,
                                               artifact_files=artifact_files)
        if local_staging_table_name is not None:
            connector.overwrite_table(local_staging_table_name, table_name)
    except BigQueryOutputSchemaError as e:
        raise ArtifactExportError(message=e.message,
                                  non_retryable=True) from None
    except Exception as e:
        log.warning('BigQuery artifact export failed (%s)', type(e).__name__)
        raise ArtifactExportError(
            message="Failed to export artifacts to BigQuery",
            non_retryable=is_error_non_retryable(e),
        ) from None

    finally:
        if local_staging_table_name is not None:
            _drop_staging_table(connector, local_staging_table_name)

    # Irrespective of whether using a staging table or not, the display path
    # should always be the final output table name.
    display_table_name = (
        f"{connector.project_id}.{connector.dataset_id}.{table_name}")
    return OutputStats(size_bytes=data_size, display_path=display_table_name)


def _csv_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Return :obj:`df` with its embedding columns rendered for CSV.

    The rule is :func:`kumo_core.artifact_export.flatten_arrays_for_csv`, which
    the local sink calls as well. Only the classification is this activity's:
    a frame carrying a nested array will carry one again on every attempt, so
    the refusal is marked non-retryable here rather than in the rule, where a
    local caller has no use for it.

    Caught by type, and only that type. Rendering a frame runs pandas, numpy
    and PyArrow, any of which raises :class:`ValueError` for its own reasons;
    catching the base class would restate an out-of-memory or a dtype fault as
    "use Parquet instead", non-retryably, which is both wrong and permanent.
    Anything other than the one refusal this rule makes on purpose propagates
    as itself and is classified by whoever understands it.
    """
    try:
        return flatten_arrays_for_csv(df)
    except MultiDimensionalCsvError as error:
        raise ArtifactExportError(
            message=str(error),
            non_retryable=True,
        ) from None
