# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import posixpath
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, Optional, Union, cast
from urllib.parse import urlsplit
from uuid import UUID

from kumo_core.sources.layout import within_root
from kumo_core.types import PredictionArtifactType
from kumoapi.jobs import ADLSPredictionOutput as APIADLSPredictionOutput
from kumoapi.jobs import FileTrainingTableOutputConfig
from kumoapi.jobs import GCSPredictionOutput as APIGCSPredictionOutput
from kumoapi.jobs import NamedTrainingTableOutputConfig, PredictionOutputConfig
from kumoapi.jobs import S3PredictionOutput as APIS3PredictionOutput
from kumoapi.jobs import TrainingTableArtifactType, TrainingTableOutputConfig
from kumoapi.jobs import WriteMode as KumoAPIWriteMode
from kumoapi.typing import _FORBID_EXTRA_CONFIG, compatible_field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import Self

from io_utils import logging
from kumo.connector.any_uri_file_connector import AnyURIFileConnector
from kumo.train.artifact_manifest import PredictionSelectionRef
from rest.config.batch_prediction_job import (
    BatchPredictionJobInput,
    OutputArtifactStats,
    requested_prediction_artifact_types,
)
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.file_type import FileType
from rest.config.prediction_output import (
    ADLSPredictionOutput,
    BigQueryPredictionOutput,
    DatabricksPredictionOutput,
    GCSPredictionOutput,
    LocalPredictionOutput,
    S3PredictionOutput,
    SnowflakePredictionOutput,
)
from statedb.config.common import ConnectorID

log = logging.getLogger(__name__)

#: The rule that an object output name a destination under its connector's
#: root. It lives in ``kumo_core`` so that the SDK's local S3 sink enforces
#: the same one; the name here is the one this module has always exported.
require_output_path_within_connector = within_root

# Hard execution deadline for independently launched artifact workflows.
ARTIFACT_EXPORT_WORKFLOW_EXECUTION_TIMEOUT = timedelta(days=30)


@pydantic_dataclass(config=_FORBID_EXTRA_CONFIG)
class HostedArtifactExportRequest:
    """Request for exporting a prediction or training-table artifact."""
    request_id: str
    job_id: str
    prediction_output: Optional[PredictionOutputConfig] = None
    training_table_output: Optional[TrainingTableOutputConfig] = None

    @compatible_field_validator('request_id')
    def validate_request_id(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError('request_id must be a canonical UUID') from error
        if str(parsed) != value:
            raise ValueError('request_id must be a canonical UUID')
        return value

    def __post_init__(self) -> None:
        configs = [self.prediction_output, self.training_table_output]
        if sum(config is not None for config in configs) != 1:
            raise ValueError(
                'Exactly one of "prediction_output" or '
                '"training_table_output" must be set.')


@dataclass
class ArtifactOutTable:
    table_name: str
    # The staging table name is used only by connectors that stage before an
    # OVERWRITE operation.
    staging_table_name: Optional[str] = None
    write_mode: KumoAPIWriteMode = KumoAPIWriteMode.OVERWRITE


@dataclass
class FileOutputConfig:
    """Credential-free durable object-store destination.

    Exactly one coordinate representation is populated. ``output_path`` is a
    canonical full URI and ``path`` is connector-relative.
    """
    connector: ConnectorExecutionRef
    output_path: str | None = None
    path: str | None = None
    file_type: Optional[FileType] = None

    def __post_init__(self) -> None:
        if sum(value is not None
               for value in (self.output_path, self.path)) != 1:
            raise ValueError('File output requires exactly one destination')


def resolve_file_output_path(
    output: FileOutputConfig,
    connector,
) -> str:
    """Resolve a safe workflow coordinate after activity-side hydration."""
    if output.output_path is not None:
        if isinstance(connector, AnyURIFileConnector):
            return connector._table_path(output.output_path)
        require_output_path_within_connector(output.output_path,
                                             connector.root_dir)
        return output.output_path
    if isinstance(connector, AnyURIFileConnector):
        raise ValueError('Rootless connectors require a full destination')
    root = connector.root_dir
    expected = {'s3', 'gs', 'abfs', 'abfss'}
    if not root or urlsplit(root).scheme.casefold() not in expected:
        raise ValueError('Relative destination does not match connector')
    assert output.path is not None
    output_path = posixpath.join(root.rstrip('/'), output.path)
    require_output_path_within_connector(output_path, root)
    return output_path


def file_output_config(
    connector: ConnectorExecutionRef,
    coordinate: str,
    file_type: Optional[FileType] = None,
) -> FileOutputConfig:
    """Build the durable form without changing the external string shape."""
    if urlsplit(coordinate).scheme:
        return FileOutputConfig(connector, output_path=coordinate,
                                file_type=file_type)
    return FileOutputConfig(connector, path=coordinate, file_type=file_type)


@dataclass
class NamedOutputConfig:
    connector: ConnectorExecutionRef
    out_table: ArtifactOutTable
    output_index: int | None = None
    managed_volume_stage: Optional['DatabricksManagedVolumeStage'] = None


@dataclass
class DatabricksManagedVolumeStage:
    path: str
    files: list[str]


@dataclass
class ArtifactExportInput:
    # For each artifact type, the Kumo-internal path where it is stored.
    artifact_paths: Dict[Union[PredictionArtifactType,
                               TrainingTableArtifactType], str]
    # For each artifact type, every authorized customer destination describing
    # where and how the shared producer artifact will be written.
    output_config: Dict[Union[PredictionArtifactType,
                              TrainingTableArtifactType],
                        list[Union[FileOutputConfig, NamedOutputConfig]]]
    # Batch-prediction ownership remains a bounded pointer/manifest reference.
    # Activities resolve and validate the exact selected files; workflow
    # histories never duplicate an O(number-of-parts) inventory.
    prediction_selection: PredictionSelectionRef | None = None
    prediction_artifact_types: tuple[PredictionArtifactType, ...] = ()
    known_artifact_stats: Dict[
        Union[PredictionArtifactType, TrainingTableArtifactType],
        OutputArtifactStats,
    ] = field(default_factory=dict)

    @classmethod
    def from_out_config_n_spec(
        cls,
        artifact_paths: Dict[Union[PredictionArtifactType,
                                   TrainingTableArtifactType], str],
        output_config: Union[PredictionOutputConfig, TrainingTableOutputConfig],
        connector_spec: ConnectorExecutionRef,
    ) -> Self:
        r"""Build durable export input from an API output and connector ref.

        Args:
            artifact_paths: Paths to generated training or prediction
                artifacts in Kumo-internal storage.
            output_config: Customer-facing training or prediction destination.
            connector_spec: Authorized credential-free destination connector
                reference.

        Returns:
            An ``ArtifactExportInput`` suitable for durable workflow storage.
        """
        output_configs = {}

        artifact_type = output_config.artifact_type
        if isinstance(output_config, NamedTrainingTableOutputConfig):
            output_configs[artifact_type] = [
                NamedOutputConfig(
                    connector=connector_spec,
                    out_table=ArtifactOutTable(
                        table_name=output_config.table_name,
                        write_mode=output_config.write_mode,
                    ),
                )
            ]
        elif isinstance(output_config, FileTrainingTableOutputConfig):
            output_configs[artifact_type] = [
                file_output_config(connector_spec, output_config.path)
            ]

        else:
            artifact_type = cls._map_artifact_type(artifact_type)
            if isinstance(output_config,
                          (APIS3PredictionOutput, APIGCSPredictionOutput,
                           APIADLSPredictionOutput)):
                output_configs[artifact_type] = [
                    file_output_config(
                        connector_spec,
                        output_config.file_path,
                        getattr(output_config, 'file_type', None),
                    )
                ]
            else:
                table_name = output_config.table_name
                write_mode = output_config.write_mode
                staging_table_name = getattr(output_config,
                                             'staging_table_name', None)

                output_configs[artifact_type] = [
                    NamedOutputConfig(
                        connector=connector_spec, out_table=ArtifactOutTable(
                            table_name=table_name,
                            staging_table_name=staging_table_name,
                            write_mode=write_mode))
                ]

        return cls(
            artifact_paths=artifact_paths,
            output_config=output_configs,
        )

    @staticmethod
    def _map_artifact_type(kumo_api_artifact_type) -> PredictionArtifactType:
        # Bridge equivalent artifact enums from API and internal job modules.
        return cast(PredictionArtifactType, kumo_api_artifact_type)

    @classmethod
    def from_bp_input(
        cls,
        job_input: BatchPredictionJobInput,
        artifact_paths: Dict[PredictionArtifactType, str],
        *,
        prediction_selection: PredictionSelectionRef | None = None,
        databricks_stages: Dict[int, DatabricksManagedVolumeStage]
        | None = None,
        known_artifact_stats: Dict[PredictionArtifactType, OutputArtifactStats]
        | None = None,
    ) -> Self:
        # Create a durable mapping from each requested output to its authorized
        # connector reference and output configuration. Credentials are never
        # placed in workflow input.
        output_config = {}
        requested_artifacts = requested_prediction_artifact_types(
            job_input.config)
        requested_artifact_order = tuple(
            dict.fromkeys(
                cls._map_artifact_type(output.artifact_type)
                for output in job_input.config.outputs))
        unexpected_artifacts = set(artifact_paths) - requested_artifacts
        if unexpected_artifacts:
            raise ValueError(
                'Batch prediction produced unrequested artifacts: '
                f'{sorted(str(value) for value in unexpected_artifacts)}')
        artifact_paths = dict(artifact_paths)
        databricks_stages = databricks_stages or {}
        selected_artifacts = (requested_artifact_order
                              if prediction_selection is not None else ())

        for output_index, output_conf in enumerate(job_input.config.outputs):
            artifact_type = cls._map_artifact_type(output_conf.artifact_type)

            if isinstance(output_conf, LocalPredictionOutput):
                # LOCAL outputs remain in the Kumo dataplane for download and
                # do not need a customer connector export workflow.
                continue

            if isinstance(output_conf, (S3PredictionOutput, GCSPredictionOutput,
                                        ADLSPredictionOutput)):
                connector_id = ConnectorID(output_conf.connector_id)
                connector = job_input.output_connectors.get(connector_id)
                if connector is None:
                    raise ValueError(
                        f"Prediction output with ID '{connector_id}' is "
                        'missing its authorized connector reference')
                out_config = file_output_config(
                    connector,
                    output_conf.file_path,
                    output_conf.file_type,
                )

            else:
                connector_id = ConnectorID(output_conf.connector_id)

                staging_table_name = None
                if isinstance(output_conf, BigQueryPredictionOutput):
                    staging_table_name = output_conf.staging_table_name
                    write_mode = output_conf.write_mode
                elif isinstance(output_conf, SnowflakePredictionOutput):
                    staging_table_name = output_conf.staging_table_name
                    write_mode = output_conf.write_mode
                elif isinstance(output_conf, DatabricksPredictionOutput):
                    write_mode = output_conf.write_mode
                else:
                    raise ValueError(
                        f"Unsupported connector type: {type(output_conf)}")

                connector = job_input.output_connectors.get(connector_id)
                out_table = ArtifactOutTable(
                    table_name=output_conf.table_name,
                    staging_table_name=staging_table_name,
                    write_mode=write_mode)
                assert connector is not None
                out_config = NamedOutputConfig(connector=connector,
                                               out_table=out_table,
                                               output_index=output_index)
                if isinstance(output_conf, DatabricksPredictionOutput):
                    stage = databricks_stages.get(output_index)
                    if stage is None:
                        raise ValueError(
                            'Databricks prediction output is missing its '
                            'producer-uploaded managed-volume stage')
                    out_config.managed_volume_stage = stage
                    artifact_paths.setdefault(artifact_type, stage.path)

            output_config.setdefault(artifact_type, []).append(out_config)

        missing_artifacts = requested_artifacts - (set(artifact_paths)
                                                   | set(selected_artifacts))
        if missing_artifacts:
            raise ValueError(
                'Batch prediction is missing requested artifact paths: '
                f'{sorted(str(value) for value in missing_artifacts)}')

        known_artifact_stats = known_artifact_stats or {}
        unexpected_stats = set(known_artifact_stats) - requested_artifacts
        if unexpected_stats:
            raise ValueError(
                'Batch prediction has stats for unrequested artifacts: '
                f'{sorted(str(value) for value in unexpected_stats)}')

        return cls(
            artifact_paths=artifact_paths,
            output_config=output_config,
            prediction_selection=prediction_selection,
            prediction_artifact_types=selected_artifacts,
            known_artifact_stats=known_artifact_stats,
        )

    def __post_init__(self):
        available_artifacts = (set(self.artifact_paths)
                               | set(self.prediction_artifact_types))
        for artifact_type in self.output_config:
            if artifact_type not in available_artifacts:
                raise ValueError(
                    f"Artifact type {artifact_type} path not found in "
                    f"the producer contract")
        expected_selected = tuple(dict.fromkeys(self.prediction_artifact_types))
        if expected_selected != self.prediction_artifact_types:
            raise ValueError('Selected prediction artifact set is invalid')
        if bool(self.prediction_selection) != bool(expected_selected):
            raise ValueError(
                'Prediction selection and artifact set must be provided together'
            )

    @property
    def artifact_types(
        self,
    ) -> tuple[Union[PredictionArtifactType, TrainingTableArtifactType], ...]:
        return tuple(
            dict.fromkeys((*self.artifact_paths,
                           *self.prediction_artifact_types)))
