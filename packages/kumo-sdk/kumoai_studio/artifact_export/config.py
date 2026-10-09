# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from kumoapi.jobs import (
    ADLSModelArtifactOutput,
    DatabricksModelArtifactOutput,
    GCSModelArtifactOutput,
    MetadataField,
    ModelOutputConfig,
    S3ModelArtifactOutput,
    SnowflakeModelArtifactOutput,
    WriteMode,
)
from kumoapi.plugin import ModelRegistryLocation
from kumoapi.prediction_outputs import resolve_prediction_output_paths
from pydantic import field_validator, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from kumoai_studio.connector.adls_connector import ADLSConnector
from kumoai_studio.connector.base import Connector
from kumoai_studio.connector.databricks_connector import DatabricksConnector
from kumoai_studio.connector.gcs_connector import GCSConnector
from kumoai_studio.connector.s3_connector import S3Connector
from kumoai_studio.connector.snowflake_connector import SnowflakeConnector


@dataclass(frozen=True)
class ModelImportConfig:
    r"""Locate a published model to import into this process.

    The mirror of :class:`ModelExportConfig`, and addressed the same way: a
    Snowflake or Databricks connector names a registry and takes ``location``;
    an S3, GCS, or ADLS connector names a place under a root and takes
    ``path``. Exactly one is set, and which one follows from the connector.

    A plain dataclass, unlike its export counterpart, because it never crosses
    a wire: importing is a local-backend operation, so nothing here is
    serialized.

    .. code-block:: python

        import kumoai_studio

        registry = kumoai_studio.SnowflakeConnector(
            name="ml-registry",
            warehouse="COMPUTE_WH",
            database="PROD",
            schema_name="ML",
        )
        trainer = kumoai_studio.import_model(kumoai_studio.ModelImportConfig(
            connector=registry,
            location=kumoai_studio.ModelRegistryLocation("CHURN_GNN", version="V3"),
        ))

    Two connectors, because a model and its data are two places. ``connector``
    is where the *model* is published; ``graph_connector`` is where the tables
    it was trained over live *now*, which is often a different schema and
    sometimes a different account. The definition in the bundle says what the
    graph is; only a caller can say where it is.

    Args:
        connector: Project-owned connector holding the published model.
        path: Where the bundle sits under a rooted connector. Required for an
            object store, and refused for a registry.
        location: The model, and optionally the version, to read. A version of
            ``None`` takes the registry's default. Required for a Snowflake or
            Databricks connector, and refused for an object store.
        graph_connector: Where the graph's tables are read from. One connector
            serves every table; pass a mapping of table name to connector for a
            graph whose tables live in different places.
    """
    connector: Connector
    path: Optional[str] = None
    location: Optional[ModelRegistryLocation] = None
    # Defaulted so it can sit after the two addressing fields, and required in
    # `__post_init__`: a model cannot be scored without naming its data, and
    # saying so here beats failing later with a half-built query.
    graph_connector: Optional[Union[Connector, Mapping[str, Connector]]] = None

    def __post_init__(self) -> None:
        _check_addressing(self.connector, path=self.path,
                          location=self.location, path_field='path',
                          location_field='location')
        if self.graph_connector is None:
            raise ValueError(
                'graph_connector is required: an imported model names the '
                'tables it was trained over, but not where they are now. Pass '
                'the connector holding that data, or a mapping of table name '
                'to connector.')

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable project owning the source connector."""
        return self.connector.kumo_project_id

    @property
    def is_registry(self) -> bool:
        r"""Whether this import addresses a model by name rather than path."""
        return isinstance(self.connector, _REGISTRY_CONNECTORS)


def _check_addressing(connector: Any, *, path: Optional[str],
                      location: Optional[Any], path_field: str,
                      location_field: str) -> None:
    r"""Refuses a destination the connector does not address models by.

    Shared by both directions so export and import cannot drift on what counts
    as a registry, and checked against the *connector* rather than merely for
    one-of-two: a path and a registry name are not interchangeable, so naming
    the field this connector actually takes is what makes the error actionable.
    """
    kind = type(connector).__name__
    if not isinstance(connector, _PATH_CONNECTORS + _REGISTRY_CONNECTORS):
        raise ValueError(
            'connector must be S3Connector, GCSConnector, ADLSConnector, '
            'SnowflakeConnector, or DatabricksConnector')
    if isinstance(connector, _REGISTRY_CONNECTORS):
        if path is not None:
            raise ValueError(
                f'{kind} names a registry, so it takes '
                f'{location_field}=ModelRegistryLocation(...). {path_field} '
                f'addresses a location under a root.')
        if location is None:
            raise ValueError(
                f'{kind} names a registry, so '
                f'{location_field}=ModelRegistryLocation(...) is required.')
        if not isinstance(location, ModelRegistryLocation):
            raise ValueError(
                f'{location_field} must be a ModelRegistryLocation')
        return
    if location is not None:
        raise ValueError(
            f'{kind} addresses a location under a root, so it takes '
            f'{path_field}. A model version is named only by a registry.')
    if not isinstance(path, str) or not path or path != path.strip():
        raise ValueError(f'{path_field} must be a non-empty path without '
                         f'surrounding whitespace')


#: Connectors that address a model by name and version rather than by path.
#: Registered here rather than tested inline so the two directions -- export
#: and import -- cannot drift on what counts as a registry.
_REGISTRY_CONNECTORS = (SnowflakeConnector, DatabricksConnector)

#: Connectors that address a model by a path under a root.
_PATH_CONNECTORS = (S3Connector, GCSConnector, ADLSConnector)


@dataclass(frozen=True)
class ModelExportConfig:
    r"""Configure connector-backed export of an offline model bundle.

    Users provide an SDK connector object, never its internal ID. The SDK
    snapshots the connector's project and emits the strict credential-free API
    representation. Rooted S3, GCS, and ADLS connectors accept an arbitrary-
    depth relative path or same-root full URI. Rootless object-store
    connectors require a full provider URI.

    .. code-block:: python

        import kumoai_studio

        output = kumoai_studio.S3Connector(
            name="model-output",
            root_dir="s3://company-ml/",
        )
        config = kumoai_studio.ModelExportConfig(
            training_job_id="training-job-...",
            output_connector=output,
            output_path="models/churn/2026/07/production",
        )
        job = kumoai_studio.export_model(config)

        # The equivalent same-root full path is also accepted:
        config = kumoai_studio.ModelExportConfig(
            training_job_id="training-job-...",
            output_connector=output,
            output_path=(
                "s3://company-ml/models/churn/2026/07/production"
            ),
        )

    A Snowflake or Databricks connector names a *registry* instead, where a
    model is addressed by name and version. Such a destination is given as
    ``output_location`` rather than ``output_path``; which of the two a config
    carries follows from its connector, and exactly one must be set.

    .. code-block:: python

        registry = kumoai_studio.SnowflakeConnector(
            name="ml-registry",
            warehouse="COMPUTE_WH",
            database="PROD",
            schema_name="ML",
        )
        config = kumoai_studio.ModelExportConfig(
            training_job_id="training-job-...",
            output_connector=registry,
            output_location=kumoai_studio.ModelRegistryLocation("CHURN_GNN"),
        )

    Args:
        training_job_id: Training job containing the model bundle.
        output_connector: Project-owned connector used to write artifacts. An
            S3, GCS, or ADLS connector takes ``output_path``; a Snowflake or
            Databricks connector takes ``output_location``.
        output_path: Relative or same-root URI for a rooted connector; full
            provider URI for a rootless connector. Paths may be nested.
            Required for an object-store connector, and refused for a registry.
        output_location: The model name and version to register. Required for a
            Snowflake or Databricks connector, and refused for an object store,
            which has no registry to version within.
    """
    training_job_id: str
    output_connector: Union[S3Connector, GCSConnector, ADLSConnector,
                            SnowflakeConnector, DatabricksConnector]
    # Optional so that a registry destination can supply `output_location`
    # instead. Kept in position, with every later field already defaulted, so
    # that positional construction of an object-store export is unchanged.
    output_path: Optional[str] = None
    output_location: Optional[ModelRegistryLocation] = None

    def __post_init__(self) -> None:
        _check_addressing(self.output_connector, path=self.output_path,
                          location=self.output_location,
                          path_field='output_path',
                          location_field='output_location')

    @property
    def is_registry(self) -> bool:
        r"""Whether this export addresses a model by name rather than path."""
        return isinstance(self.output_connector, _REGISTRY_CONNECTORS)

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable project owning the output connector."""
        return self.output_connector.kumo_project_id

    def _to_api_config(self) -> ModelOutputConfig:
        connector = self.output_connector
        if self.is_registry:
            assert self.output_location is not None  # `__post_init__`
            location = self.output_location
            registry = (SnowflakeModelArtifactOutput if isinstance(
                connector, SnowflakeConnector) else
                        DatabricksModelArtifactOutput)
            return ModelOutputConfig(
                training_job_id=self.training_job_id,
                output=registry(
                    connector_id=connector.id,
                    model_name=location.model_name,
                    version_name=location.version,
                    comment=location.comment,
                ),
            )
        # Both guaranteed by `__post_init__` on this frozen dataclass: a
        # non-registry connector is one of the three object stores, and it
        # carries a non-empty `output_path`.
        assert self.output_path is not None
        if isinstance(connector, S3Connector):
            output = S3ModelArtifactOutput(
                connector_id=connector.id,
                path=connector._rooted_destination(self.output_path),
            )
        elif isinstance(connector, GCSConnector):
            output = GCSModelArtifactOutput(
                connector_id=connector.id,
                path=connector._rooted_destination(self.output_path),
            )
        elif isinstance(connector, ADLSConnector):
            output = ADLSModelArtifactOutput(
                connector_id=connector.id,
                path=connector._rooted_destination(self.output_path),
            )
        else:
            # Unreachable while `_REGISTRY_CONNECTORS` and the branches above
            # partition the annotated union. Named rather than folded into the
            # ADLS branch so that a connector added to one and not the other
            # fails here instead of being silently published as ADLS.
            raise AssertionError(
                f'{type(connector).__name__} addresses neither a registry nor '
                f'a path under a root')
        return ModelOutputConfig(
            training_job_id=self.training_job_id,
            output=output,
        )


def compatible_field_validator(field_name: str):
    """Decorator factory that creates a field validator.

    Usage:
        @compatible_field_validator('field_name')
        def validate_field(cls, v, values_or_info):
            # Your validation logic here
            return v
    """
    def decorator(func):
        @field_validator(field_name)
        @classmethod
        @functools.wraps(func)
        def wrapper(cls, v, info):
            # Convert info to values dict for compatibility
            values = info.data if hasattr(info, 'data') else {}
            return func(cls, v, values)

        return wrapper

    return decorator


# TODO: probably will need to be removed b/c using __post_init__ instead
def compatible_model_validator(mode='before'):
    """Decorator factory that creates a model validator.

    Usage:
        @compatible_model_validator()
        def validate_model(cls, values):
            # Your validation logic here
            return values
    """
    def decorator(func):
        @model_validator(mode=mode)
        @classmethod
        @functools.wraps(func)
        def wrapper(cls, values):
            return func(cls, values)

        return wrapper

    return decorator


@dataclass(frozen=True)
class QueryConnectorConfig:
    # If using OVERWRITE, big query connector will first write to a staging
    # table followed by overwriting to the destination table.
    # When using APPEND, it is strongly recommended to use
    # MetadataField.JOB_TIMESTAMP to indicate the timestamp of the job.
    write_mode: WriteMode = WriteMode.APPEND


@dataclass(frozen=True)
class BigQueryOutputConfig(QueryConnectorConfig):
    pass


@dataclass(frozen=True)
class SnowflakeConnectorConfig(QueryConnectorConfig):
    pass


@dataclass(frozen=True)
class DatabricksOutputConfig(QueryConnectorConfig):
    # Databricks OVERWRITE is replay-safe. APPEND remains available when
    # explicitly requested, but disables server-side export retries.
    write_mode: WriteMode = WriteMode.OVERWRITE


CONNECTOR_CONFIG_MAPPING = {
    'BigQueryConnector': BigQueryOutputConfig,
    'SnowflakeConnector': SnowflakeConnectorConfig,
    'DatabricksConnector': DatabricksOutputConfig,
    # 'S3Connector': S3OutputConfig,
}


@pydantic_dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class OutputConfig:
    """Output configuration associated with a Batch Prediction Job.
    Specifies the output types and optionally output data source
    configuration.

    Args:
        output_types(`Set[str]`): The types of outputs that should be produced
            by the prediction job. Can include either ``'predictions'``,
            ``'embeddings'``, or both.
        output_connector(`Connector` or None): The output data source that Kumo
                should write batch predictions to, if it is None,
                produce local download output only.
        output_table_name(`str` or `Tuple[str, str]` or None): Destination for
            the batch-prediction output. Rooted S3, GCS, and ADLS connectors
            accept an arbitrary-depth relative path or same-root full URI.
            Rootless S3, GCS, and ADLS connectors require a full provider URI.
            Kumo appends the output type (for example ``_predictions``) to this
            value. In the case of a Databricks connector, this should be a
            tuple of two strings: the schema name and the output prediction
            table name.
        output_paths: Exact destinations keyed by predictions or embeddings.
            Keys select the artifacts; do not also supply output_types or
            output_table_name. A trailing slash selects a directory.
        output_metadata_fields: ``JOB_TIMESTAMP`` and/or ``ANCHOR_TIMESTAMP``
            columns for every selected artifact, including embeddings.
        connector_specific_config(`QueryConnectorConfig` or None): The custom
            connector specific output config for predictions, for
            example whether to append or overwrite existing table.
    """
    output_types: Optional[Set[str]] = None
    output_connector: Optional[Connector] = None
    output_table_name: Optional[Union[str, Tuple]] = None
    output_metadata_fields: Optional[List[MetadataField]] = None
    connector_specific_config: Optional[Union[
        BigQueryOutputConfig,
        SnowflakeConnectorConfig,
        DatabricksOutputConfig,
    ]] = None

    output_paths: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if self.output_paths is not None:
            paths = resolve_prediction_output_paths(
                self.output_paths, output_types=self.output_types,
                output_table_name=self.output_table_name)
            if not isinstance(self.output_connector,
                              (S3Connector, GCSConnector, ADLSConnector)):
                raise ValueError(
                    'output_paths requires an S3, GCS, or ADLS connector')
            # Reject aliases; retain caller paths.
            resolve_prediction_output_paths({
                kind:
                self.output_connector._rooted_destination(path)
                for kind, path in paths.items()
            })
            object.__setattr__(self, 'output_paths', paths)
            object.__setattr__(self, 'output_types', set(paths))
        elif self.output_types is None:
            raise ValueError('Specify output_types or output_paths')

    @compatible_field_validator('connector_specific_config')
    def validate_connector_config(cls, v: Any, values: Dict) -> Any:
        """Validate the connector specific output config. Raises ValueError if
        there is a mismatch between the connector type and the config type.
        """
        # Skip validation if no connector or no specific config
        if values.get('output_connector') is None or v is None:
            return v

        connector_type = type(values['output_connector']).__name__
        expected_config_type = CONNECTOR_CONFIG_MAPPING.get(connector_type)

        # If we don't have a mapping for this connector type, it doesn't
        # support specific configs yet
        if expected_config_type is None:
            raise ValueError(
                f"Connector type '{connector_type}' does not support "
                f"specific output configurations")

        # Check if the provided config is of the correct type
        if not isinstance(v, expected_config_type):
            raise ValueError(
                f"Connector type '{connector_type}' requires output "
                f"config of type '{expected_config_type.__name__}', but "
                f"got '{type(v).__name__}'")

        return v


@pydantic_dataclass(frozen=True, config={'arbitrary_types_allowed': True})
class TrainingTableExportConfig(OutputConfig):
    """Export configuration associated with a Training Table.

    Args:
        output_types(`Set[str]`): The artifact to export from the training
            table job. Currently only `'training_table'` is supported.
            Which exports the full training table to the output connector.
        output_connector(`Connector`): The output data source that Kumo should
            write training table artifacts to.
        output_table_name(str): The destination in the output data source.
            Rooted object-store connectors accept a relative or same-root
            path; rootless connectors require a full provider URI. For a
            Databricks connector, this should be a tuple containing the schema
            name and output prediction table name.
        connector_specific_config(QueryConnectorConfig or None):
            Defines custom connector specific output
            for example whether to append or overwrite
            existing table. This is supported for BigQuery, Snowflake, and
            Databricks.
    """
    output_connector: Optional[Connector] = None
    output_table_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.output_paths is not None:
            raise ValueError(
                'output_paths is only supported for prediction artifacts')
        if self.output_types != {'training_table'}:
            raise ValueError("output_type must be set(['training_table'])"
                             f" (got {self.output_types})")
        if self.output_connector is None:
            raise ValueError("output_connector is required")
        if self.output_table_name is None:
            raise ValueError("output_table_name is required")
        if self.output_metadata_fields is not None:
            raise ValueError(
                "output_metadata_fields is not supported for training"
                "table export")
