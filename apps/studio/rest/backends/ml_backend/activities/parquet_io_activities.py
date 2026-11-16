# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Global semaphore for limiting concurrent (parquet) IO operations which lead
# to high memory consumption.
import json
import os
import os.path as osp
import tempfile
import uuid
from asyncio import CancelledError
from concurrent.futures import TimeoutError
from contextlib import ExitStack
from threading import Semaphore
from typing import Dict, List, Mapping, Union

import pyarrow.parquet as pq
from kumo_core.job_config import PredictionConfig
from kumo_core.training import export as training_table_export
from kumo_core.types import PredictionArtifactType
from kumoapi.jobs import TrainingTableArtifactType
from temporalio import activity
from temporalio.exceptions import ApplicationError

from common_utils.error import PublicException
from io_utils import fsutil, logging, parquet_util
from io_utils.errors import restricted_diagnostic_boundary
from kumo.connector import (
    BigQueryConnector,
    DatabricksConnector,
    SnowflakeConnector,
)
from kumo.connector.connector import ConnectorError
from kumo.connector.file_connector import FileConnector
from kumo.data.generic.diskgraph_train_table import _IMMUTABLE_BUNDLES_DIRNAME
from kumo.train.artifact_manifest import (
    PredictionManifestRef,
    PredictionProducerMismatchError,
    PredictionSelectionRef,
    SelectedPredictionArtifact,
    load_prediction_selection,
    publish_prediction_selection,
)
from rest.backends.data_backend.activities.custom_train_table import (
    _filesystem_and_path,
)
from rest.backends.data_backend.connector_activity_registry import (
    connector_authorized_activity,
    connector_resolution_helper,
)
from rest.backends.ml_backend.artifact_export import (
    _convert_parquet_to_local_csv,
    write_artifact_to_bigquery,
    write_artifact_to_databricks,
    write_artifact_to_s3,
    write_artifact_to_snowflake,
)
from rest.backends.ml_backend.prediction_artifacts import (
    selected_prediction_artifact_files,
)
from rest.config.artifact_export import (
    ArtifactExportInput,
    FileOutputConfig,
    NamedOutputConfig,
    resolve_file_output_path,
)
from rest.config.batch_prediction_job import (
    BatchPredictionJobInput,
    DatabricksVolumeShard,
    OutputArtifactStats,
    requested_prediction_artifact_types,
)
from rest.config.data_backend import ConnectorExecutionRef, ConnectorOpenSpec
from rest.utils.driver_common import get_connector_from_spec
from service_framework.env_var import EnvVar
from statedb.config.common import ConnectorID
from statedb.utils.errors import (
    ArtifactExportError,
    PredictionOutputProcessingError,
)
from temporalio_commons.error_utils import is_error_non_retryable
from temporalio_commons.heartbeat_utils import auto_heartbeater
from temporalio_commons.process_pool_utils import HealingProcessPoolExecutor

log = logging.getLogger(__name__)

# As of 03/11/2024, offline benchmark indicates that processing(resharding)
# large parquet files with ~1GB (uncompressed) shard size consumes ~3GB memory.
# Allowing max 3 concurrent parquent IO activities indicates ~9GB peak memory
# footprint.
_MAX_CONCURRENT_PARQUET_IO = EnvVar(
    'MAX_CONCURRENT_PARQUET_IO',
    type=int,
    required=False,
    default_value=3,
    description="""
    Max allowed concurrent parquet IO operations, e.g., postprocessing BP
    output files.""",
)
parquet_io_semaphore = Semaphore(_MAX_CONCURRENT_PARQUET_IO.parse_value())

_CSV_CONVERSION_POLL_INTERVAL_SECONDS = 1.0


class _DisposableCSVConverter:
    """Run credential-free Pandas/PyArrow conversion in one killable child.

    The child protocol is intentionally limited to two safe path strings: the
    internal artifact shard and a mode-0700 local staging path. Connector
    specs, destination metadata, and credentials stay in the activity process.
    One child is reused for every shard in an output and is always reaped when
    that output completes, fails, or is cancelled.
    """
    def __init__(self) -> None:
        self._executor: HealingProcessPoolExecutor | None = None

    def __enter__(self) -> '_DisposableCSVConverter':
        return self

    def __exit__(self, *_args: object) -> None:
        if self._executor is not None:
            log.info('Terminating disposable CSV conversion process.')
            self._executor.terminate()
            self._executor = None

    def __call__(self, in_parquet_file: str, local_csv_file: str) -> None:
        if self._executor is None:
            self._executor = HealingProcessPoolExecutor(1)
        future = self._executor.submit(
            _convert_parquet_to_local_csv,
            in_parquet_file,
            local_csv_file,
        )
        while True:
            try:
                future.result(timeout=_CSV_CONVERSION_POLL_INTERVAL_SECONDS)
                return
            except TimeoutError:
                if activity.in_activity():
                    # Heartbeats are both liveness and the delivery mechanism
                    # for Temporal cancellation. Polling once per second makes
                    # cancellation independent of native Arrow/Pandas work in
                    # the child; __exit__ then kills and joins that process.
                    activity.heartbeat()
                    if activity.is_cancelled():
                        raise CancelledError('Artifact CSV export cancelled')


class PredictionOutputFileMissingError(PredictionOutputProcessingError):
    """Raised when expected prediction output file is missing."""
    template = (
        "Batch prediction job completed but the expected output was not found. "
        "This may indicate an internal processing error. Please retry the job "
        "or contact support if the issue persists.")


@activity.defn
def collect_prediction_job_output(
    job_input: BatchPredictionJobInput,
    job_config_path: str,
) -> dict[PredictionArtifactType, str]:
    # auto_heartbeater is injected by the worker interceptor because the
    # dispatch sets heartbeat_timeout; no explicit decorator needed.
    return _collect_prediction_artifacts(job_input, job_config_path)


@activity.defn
def finalize_prediction_job_output(
    job_input: BatchPredictionJobInput,
    job_config_path: str,
    graph_snapshot_id: str,
    selection_id: str,
    shard_manifests: list[PredictionManifestRef],
    expected_shards: int,
) -> PredictionSelectionRef:
    """Materialize and select exactly the accepted shard attempts."""
    from kumo.train.artifact_manifest import hash_prediction_config
    from kumo.train.prediction_partitions import (
        prediction_output_partition_count,
    )

    prediction_config = PredictionConfig.load(job_config_path)
    selection_attempt_id = f'temporal-{activity.info().attempt:05d}'
    source_config_sha256 = hash_prediction_config(job_config_path)
    try:
        return publish_prediction_selection(
            shard_manifests,
            job_id=str(job_input.id),
            selection_id=selection_id,
            selection_attempt_id=selection_attempt_id,
            canonical_output_root=prediction_config.output_dir,
            graph_snapshot_id=graph_snapshot_id,
            source_config_sha256=source_config_sha256,
            expected_shards=expected_shards,
            expected_output_partitions=(
                prediction_output_partition_count(prediction_config)),
        )
    except PredictionProducerMismatchError as error:
        raise PredictionOutputProcessingError(
            message=str(error), non_retryable=True) from error


def _collect_prediction_artifacts(
    job_input: BatchPredictionJobInput,
    job_config_path: str,
) -> dict[PredictionArtifactType, str]:
    r"""
    Returns: Dictionary of PredictionArtifactType to the path of the
        producer-finalized artifact.
    """
    prediction_config = PredictionConfig.load(job_config_path)
    output_dir = prediction_config.output_dir
    requested_prediction_artifact_types(job_input.config)
    # Examine existence of expected output artifacts
    result: dict[PredictionArtifactType, str] = {}
    for output in job_input.config.outputs:
        artifact_type = PredictionArtifactType(output.artifact_type)
        if artifact_type in result:
            continue
        output_path = osp.join(output_dir, artifact_type.output_file_name)
        if not fsutil.exists(output_path):
            raise PredictionOutputFileMissingError(non_retryable=True)
        if parquet_util.is_empty(output_path):
            raise PredictionOutputProcessingError(
                message=(f"Batch prediction produced an empty "
                         f"{artifact_type.value.lower()} artifact."),
                non_retryable=True,
            )
        result[artifact_type] = output_path

    log.info(f"Collected prediction artifacts: {result}")
    return result


@connector_authorized_activity
@auto_heartbeater
def write_artifacts_to_customer(
    input: ArtifactExportInput,
    artifact_stats: Dict[Union[PredictionArtifactType,
                               TrainingTableArtifactType], OutputArtifactStats],
) -> list[OutputArtifactStats]:
    resolved_connectors: dict[
        Union[PredictionArtifactType, TrainingTableArtifactType],
        list[ConnectorOpenSpec],
    ] = {}
    try:
        resolved_connectors = _resolve_output_connectors(input)
        return _write_artifacts_to_customer(
            input,
            artifact_stats=artifact_stats,
            resolved_connectors=resolved_connectors,
        )
    except ConnectorError as err:
        log.warning(
            'Customer output connector rejected the write (%s)',
            type(err).__name__,
        )
        raise ArtifactExportError(
            message='The selected output connector rejected the write. '
            'Verify its native target permissions.',
            non_retryable=is_error_non_retryable(err),
        ) from None
    except (ApplicationError, PublicException):
        raise
    except Exception as err:
        log.warning(
            'Customer output write failed (%s)',
            type(err).__name__,
        )
        raise ArtifactExportError(
            message="Failed to export artifacts to customer warehouse",
            non_retryable=is_error_non_retryable(err),
        ) from None
    finally:
        # Resolved specs may contain decrypted credentials. Drop every strong
        # reference before returning from the activity, including error paths.
        resolved_connectors.clear()


@connector_authorized_activity
@auto_heartbeater
def cleanup_databricks_volume_outputs(
    job_input: BatchPredictionJobInput,
    uploads: list[DatabricksVolumeShard],
) -> None:
    """Remove producer uploads when prediction cannot reach COPY INTO."""
    from rest.backends.data_backend import connector_broker

    by_output: dict[int, list[str]] = {}
    for upload in uploads:
        by_output.setdefault(upload.output_index,
                             []).extend(upload.staged_files)
    errors: list[Exception] = []
    for output_index, files in by_output.items():
        output = job_input.config.outputs[output_index]
        reference = job_input.output_connectors[ConnectorID(
            output.connector_id)]
        try:
            connector = get_connector_from_spec(
                connector_broker.connector_credential_broker.resolve(reference))
            if not isinstance(connector, DatabricksConnector):
                raise TypeError(
                    'Databricks staged output has a non-Databricks connector')
            try:
                connector.cleanup_managed_volume_files(files)
            finally:
                connector.close()
        except Exception as error:
            errors.append(error)
    if errors:
        raise errors[0]


@connector_resolution_helper
def _resolve_output_connectors(
    input: ArtifactExportInput,
) -> dict[Union[PredictionArtifactType, TrainingTableArtifactType],
          list[ConnectorOpenSpec]]:
    """Resolve workflow-safe references at the Temporal activity boundary."""
    from rest.backends.data_backend import connector_broker

    resolved = {}
    for artifact_type, output_configs in input.output_config.items():
        resolved[artifact_type] = []
        for output_config in output_configs:
            reference = output_config.connector
            if not isinstance(reference, ConnectorExecutionRef):
                raise TypeError(
                    'Artifact workflow outputs must contain connector '
                    'references')
            resolved[artifact_type].append(
                connector_broker.connector_credential_broker.resolve(reference))
    return resolved


@activity.defn
@auto_heartbeater
def prepare_training_table_export(train_table_path: str) -> str:
    return exported_training_table(train_table_path)


def exported_training_table(train_table_path: str) -> str:
    """The generated table's rows with their split, beside its bundle."""
    bundle = osp.dirname(train_table_path.rstrip('/'))
    bundles = osp.dirname(bundle)
    if (osp.basename(bundles) != _IMMUTABLE_BUNDLES_DIRNAME
            or not fsutil.exists(
                osp.join(bundle, training_table_export.INGEST_SPEC))):
        return train_table_path
    base = osp.join(osp.dirname(bundles), '_exports', osp.basename(bundle))
    pointer = osp.join(base, f'{training_table_export.FORMAT}.json')
    if fsutil.exists(pointer):
        return json.loads(fsutil.read_text(pointer))['path']
    output = osp.join(base,
                      f'{training_table_export.FORMAT}-{uuid.uuid4().hex}')
    fsutil.makedirs(output, exist_ok=True)
    filesystem, source = _filesystem_and_path(bundle)
    with fsutil.TemporaryDirectory() as temp:
        for part in training_table_export.iter_export_parts(
                source, temp, filesystem):
            fsutil.copy_file(part, osp.join(output, osp.basename(part)),
                             overwrite=True)
            os.remove(part)
    if fsutil.write_text_if_absent(pointer, json.dumps({'path': output})):
        return output
    fsutil.rm(output, ignore_errors=True)
    return json.loads(fsutil.read_text(pointer))['path']


@activity.defn
@auto_heartbeater
def get_artifact_stats(
    artifact_paths: Dict[Union[PredictionArtifactType,
                               TrainingTableArtifactType], str],
    prediction_selection: PredictionSelectionRef | None = None,
    prediction_artifact_types: list[PredictionArtifactType] | None = None,
) -> Dict[Union[PredictionArtifactType, TrainingTableArtifactType],
          OutputArtifactStats]:
    resolved_paths, artifact_files, selected_artifacts = (
        _resolve_selected_prediction_artifacts(
            artifact_paths,
            prediction_selection,
            prediction_artifact_types or [],
        ))
    result = {}
    for artifact_type, path in resolved_paths.items():
        selected_artifact = selected_artifacts.get(artifact_type)
        if (selected_artifact is not None
                and selected_artifact.num_rows is not None):
            assert selected_artifact.num_cols is not None
            assert selected_artifact.size_bytes is not None
            result[artifact_type] = OutputArtifactStats(
                artifact_type=artifact_type,
                num_rows=selected_artifact.num_rows,
                num_cols=selected_artifact.num_cols,
                size_bytes=selected_artifact.size_bytes,
                output_path=path,
            )
            continue
        selected_files = artifact_files.get(artifact_type)
        with restricted_diagnostic_boundary(
                safe_message='Prediction artifact Parquet read failed',
                exception_types=OSError):
            parquet_metadata = (_get_selected_parquet_metadata(selected_files)
                                if selected_files is not None else
                                parquet_util.get_parquet_metadata(path))
        size_bytes = int(
            round(parquet_metadata.avg_row_size * parquet_metadata.num_rows))
        result[artifact_type] = OutputArtifactStats(
            artifact_type=artifact_type,
            num_rows=parquet_metadata.num_rows,
            num_cols=parquet_metadata.num_cols,
            size_bytes=size_bytes,
            output_path=path,
        )
    return result


def _resolve_selected_prediction_artifacts(
    artifact_paths: Mapping[Union[PredictionArtifactType,
                                  TrainingTableArtifactType], str],
    selection_ref: PredictionSelectionRef | None,
    expected_artifact_types: List[PredictionArtifactType],
) -> tuple[
        dict[Union[PredictionArtifactType, TrainingTableArtifactType], str],
        dict[PredictionArtifactType, list[str]],
        dict[PredictionArtifactType, SelectedPredictionArtifact],
]:
    """Resolve a bounded selection reference and reject root extras."""
    resolved_paths = dict(artifact_paths)
    selected_files: dict[PredictionArtifactType, list[str]] = {}
    selected_artifacts: dict[PredictionArtifactType,
                             SelectedPredictionArtifact] = {}
    expected = set(expected_artifact_types)
    if selection_ref is None:
        if expected:
            raise ValueError('Prediction artifact selection is missing')
        return resolved_paths, selected_files, selected_artifacts

    selection = load_prediction_selection(selection_ref, validate_objects=False)
    selected = {
        artifact.artifact_type: artifact
        for artifact in selection.artifacts
    }
    if set(selected) != expected:
        raise ValueError('Prediction selection artifact set changed')
    for artifact_type, artifact in selected.items():
        files = list(selected_prediction_artifact_files(artifact))
        resolved_paths[artifact_type] = artifact.root
        selected_files[artifact_type] = files
        selected_artifacts[artifact_type] = artifact
    return resolved_paths, selected_files, selected_artifacts


def _get_selected_parquet_metadata(
        files: list[str]) -> parquet_util.ParquetMetadata:
    """Read footer metadata from only the selected immutable files."""
    if not files:
        raise ValueError('Selected prediction artifact has no files')
    schema = None
    num_rows = 0
    sample = None
    for path in files:
        with restricted_diagnostic_boundary(
                safe_message='Prediction artifact Parquet read failed',
                exception_types=OSError):
            with fsutil.get_binary_handle(path, 'rb') as handle:
                parquet_file = pq.ParquetFile(handle)
                current_schema = parquet_file.schema_arrow.remove_metadata()
                current_num_rows = parquet_file.metadata.num_rows
                current_sample = (next(
                    parquet_file.iter_batches(batch_size=1000)) if
                                  sample is None and current_num_rows else None)
        if schema is None:
            schema = current_schema
        elif not current_schema.equals(schema, check_metadata=False):
            raise ValueError('Selected prediction schemas are inconsistent')
        num_rows += current_num_rows
        if current_sample is not None:
            sample = current_sample
    if schema is None or num_rows <= 0 or sample is None:
        raise ValueError('Selected prediction artifact is empty')
    return parquet_util.ParquetMetadata(
        cols=schema.names,
        num_rows=num_rows,
        num_cols=len(schema.names),
        avg_row_size=int(round(sample.nbytes / sample.num_rows)),
    )


def _write_artifacts_to_customer(
    input: ArtifactExportInput,
    *,
    artifact_stats: Mapping[Union[PredictionArtifactType,
                                  TrainingTableArtifactType],
                            OutputArtifactStats],
    resolved_connectors: Mapping[Union[PredictionArtifactType,
                                       TrainingTableArtifactType],
                                 list[ConnectorOpenSpec]],
) -> List[OutputArtifactStats]:
    r"""
    Writes producer-finalized artifacts to the requested customer outputs.
    """
    artifact_paths, selected_artifact_files, _ = (
        _resolve_selected_prediction_artifacts(
            input.artifact_paths,
            input.prediction_selection,
            list(input.prediction_artifact_types),
        ))
    output_artifact_stats: list[OutputArtifactStats] = []
    for artifact_type, output_configs in input.output_config.items():
        artifact_path = artifact_paths[artifact_type]
        selected_files = selected_artifact_files.get(artifact_type)
        source_stats = artifact_stats[artifact_type]
        connector_specs = resolved_connectors[artifact_type]
        if len(connector_specs) != len(output_configs):
            raise ValueError(
                f'Artifact {artifact_type} has {len(output_configs)} outputs '
                f'but {len(connector_specs)} resolved connectors')
        for output_config, connector_spec in zip(output_configs,
                                                 connector_specs, strict=True):
            connector = get_connector_from_spec(connector_spec)
            if isinstance(output_config, NamedOutputConfig):
                write_mode = output_config.out_table.write_mode
                table_name = output_config.out_table.table_name
                staging_table_name = output_config.out_table.staging_table_name
            else:
                write_mode = None
                table_name = None
                staging_table_name = None

            log.info(f"Writing artifact {artifact_path} to output: "
                     f"{output_config}")
            try:
                if isinstance(output_config, FileOutputConfig):
                    if not isinstance(connector, FileConnector):
                        raise ArtifactExportError(
                            message=('Selected output connector is not an '
                                     'object-store connector.'),
                            non_retryable=True,
                        )
                    output_path = resolve_file_output_path(
                        output_config, connector)
                    with parquet_io_semaphore:
                        with ExitStack() as stack:
                            staged_artifact = artifact_path
                            staged_files = selected_files
                            # BP file output uses the persisted Parquet files
                            # directly. Keep the legacy training-table export
                            # behavior outside this change.
                            if (not isinstance(artifact_type,
                                               PredictionArtifactType)
                                    and not fsutil.is_local(artifact_path)):
                                temp_dir = stack.enter_context(
                                    tempfile.TemporaryDirectory(
                                        prefix='kumo-artifact-export-'))
                                staged_artifact = osp.join(temp_dir, 'artifact')
                                if selected_files is None:
                                    fsutil.copy(
                                        artifact_path,
                                        staged_artifact,
                                        overwrite=True,
                                        raise_on_exists=False,
                                    )
                                else:
                                    fsutil.makedirs(staged_artifact,
                                                    exist_ok=True)
                                    staged_files = []
                                    for selected_file in selected_files:
                                        staged_file = osp.join(
                                            staged_artifact,
                                            osp.basename(selected_file),
                                        )
                                        fsutil.copy_file(
                                            selected_file,
                                            staged_file,
                                            overwrite=True,
                                        )
                                        staged_files.append(staged_file)
                            with connector.filesystem_credential_scope(
                                    output_path):
                                with _DisposableCSVConverter() as csv_converter:
                                    stats = write_artifact_to_s3(
                                        staged_artifact,
                                        output_path,
                                        output_config.file_type,
                                        artifact_files=staged_files,
                                        csv_shard_converter=csv_converter,
                                        export_id=(
                                            f'{artifact_path}:'
                                            f'{source_stats.num_rows}:'
                                            f'{source_stats.size_bytes}'),
                                    )
                elif isinstance(connector, DatabricksConnector):
                    assert table_name is not None
                    volume_stage = (output_config.managed_volume_stage
                                    if isinstance(output_config,
                                                  NamedOutputConfig) else None)
                    if volume_stage is None:
                        raise ArtifactExportError(
                            message=(
                                'Databricks artifact export requires a '
                                'producer-uploaded Unity Catalog volume '
                                'manifest. Configure the Databricks output '
                                'when creating the batch prediction job.'),
                            non_retryable=True,
                        )
                    stats = write_artifact_to_databricks(
                        volume_stage.path,
                        connector,
                        table_name,
                        write_mode,
                        known_size_bytes=source_stats.size_bytes,
                        staged_files=volume_stage.files,
                    )
                elif (isinstance(output_config, NamedOutputConfig)
                      and isinstance(connector, BigQueryConnector)):
                    assert table_name is not None
                    # Only the GCS fast path consumes explicit remote URIs.
                    # Preserve existing staging for worker-local artifacts.
                    stats = write_artifact_to_bigquery(
                        artifact_path, connector, write_mode, table_name,
                        staging_table_name,
                        known_size_bytes=source_stats.size_bytes,
                        artifact_files=(selected_files if
                                        artifact_path.startswith('gs://') else
                                        None))
                elif (isinstance(output_config, NamedOutputConfig)
                      and isinstance(connector, SnowflakeConnector)):
                    assert table_name is not None
                    # The selection spans one attempt root per output
                    # shard; the artifact root alone names only the first.
                    # Stage-backed paths keep the prefix load: their FROM
                    # clause is unquoted, so a selection is not a usable
                    # coordinate there.
                    stage_backed = artifact_path.startswith(
                        ('@', 'spcs://'))
                    stats = write_artifact_to_snowflake(
                        artifact_path, connector, write_mode, table_name,
                        staging_table_name,
                        known_size_bytes=source_stats.size_bytes,
                        artifact_files=(None if stage_backed else
                                        selected_files))
                else:
                    raise ArtifactExportError(
                        message=(f"Unsupported connector type: "
                                 f"{type(connector).__name__}. "
                                 "Please use S3, Snowflake, BigQuery, or "
                                 "Databricks connectors for exporting "
                                 "artifacts."), non_retryable=True)
            finally:
                connector.close()

            output_artifact_stats.append(
                OutputArtifactStats(
                    artifact_type=artifact_type,
                    num_rows=source_stats.num_rows,
                    num_cols=source_stats.num_cols,
                    size_bytes=source_stats.size_bytes,
                    output_path=stats.display_path,
                ))

    return output_artifact_stats
