# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The graph, connectors and model registry the job scripts share.

Configured by environment variables the job definition sets; credentials come
from the platform identity.

- ``KUMO_SOURCE``: ``databricks`` (``KUMO_DATABRICKS_CATALOG``/``_SCHEMA``/
  ``_WAREHOUSE_ID``), ``snowflake`` (``KUMO_SNOWFLAKE_WAREHOUSE``/
  ``_DATABASE``/``_SCHEMA``), or ``s3``/``file`` (``KUMO_DATA_ROOT`` holding
  ``users``, ``items`` and ``orders`` parquet files).
- ``KUMO_ARTIFACT_ROOT``: ``s3://``, ``gs://`` or a local/volume path that
  receives snapshot exports, path-published models and predictions
  (optional with a registry and a predictions table).
- ``KUMO_REGISTRY``: ``mlflow`` (register ``KUMO_MODEL_NAME`` through
  ``MlflowTracking``), ``snowflake`` (the Snowflake Model Registry, tracked in
  Snowflake Experiment Tracking) or unset (fit publishes to
  ``KUMO_ARTIFACT_ROOT/runs/<run>/model``; scoring loads ``KUMO_MODEL_PATH``,
  that path relative to ``KUMO_ARTIFACT_ROOT``). With a registry, fit moves
  the ``champion`` alias to the new version and scoring loads ``champion``.
- ``KUMO_SCRATCH_DIR``: local disk for snapshots, tables and bundles.
- ``KUMO_PREDICTIONS_TABLE``: write predictions to this table of the source
  connector instead of ``KUMO_ARTIFACT_ROOT/runs/<run>/predictions/``.
"""

import argparse
import os
from typing import Any, Optional, Sequence, Tuple

from kumoapi.jobs import MetadataField, PredictionArtifactType
from kumoapi.typing import Stype

from kumo.engine import (
    Edge,
    Graph,
    ModelOutputConfig,
    ModelRegistryLocation,
    PredictionOutputConfig,
    Table,
    Trainer,
)
from kumo.engine.connector import (
    Connector,
    DatabricksConnector,
    FileConnector,
    S3Connector,
    SnowflakeConnector,
    connector_for_root,
)

CHAMPION = 'champion'


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f'Set the {name} environment variable')
    return value


def start(argv: Optional[Sequence[str]] = None) -> Tuple[Any, Any]:
    """Applies ``KUMO_SCRATCH_DIR``; returns ``--job-id``/``--job-run-id``."""
    from kumo.engine import configure

    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id')
    parser.add_argument('--job-run-id')
    args = parser.parse_args(argv)
    if os.environ.get('KUMO_SCRATCH_DIR'):
        configure(scratch_dir=os.environ['KUMO_SCRATCH_DIR'])
    return args.job_id, args.job_run_id


def source() -> Connector:
    kind = _env('KUMO_SOURCE')
    if kind == 'databricks':
        return DatabricksConnector.from_notebook(
            catalog=_env('KUMO_DATABRICKS_CATALOG'),
            schema_name=_env('KUMO_DATABRICKS_SCHEMA'),
            warehouse_id=_env('KUMO_DATABRICKS_WAREHOUSE_ID'))
    if kind == 'snowflake':
        return SnowflakeConnector.from_spcs_env(
            warehouse=_env('KUMO_SNOWFLAKE_WAREHOUSE'),
            database=_env('KUMO_SNOWFLAKE_DATABASE'),
            schema_name=_env('KUMO_SNOWFLAKE_SCHEMA'))
    if kind in ('s3', 'file'):
        return connector_for_root(_env('KUMO_DATA_ROOT'))
    raise ValueError(f'Unknown KUMO_SOURCE {kind!r}')


def artifacts() -> Connector:
    return connector_for_root(_env('KUMO_ARTIFACT_ROOT'))


def artifact_uri(path: str) -> str:
    """``path`` under ``KUMO_ARTIFACT_ROOT`` as a full URI or path."""
    return f"{_env('KUMO_ARTIFACT_ROOT').rstrip('/')}/{path}"


def _run_path() -> str:
    run = _env('KUMO_JOB_RUN_ID').rstrip('/').rsplit('/', 1)[-1]
    return f'runs/{run}'


def run_artifacts() -> Connector:
    """``KUMO_ARTIFACT_ROOT/runs/<last segment of KUMO_JOB_RUN_ID>``."""
    return connector_for_root(artifact_uri(_run_path()))


def build_graph(connector: Connector) -> Graph:

    def table(alias: str, **kwargs: str) -> Table:
        name = {
            S3Connector: f'{alias}.parquet',
            FileConnector: f'{alias}.parquet',
            SnowflakeConnector: alias.upper(),
        }.get(type(connector), alias)
        return Table.from_source_table(connector[name], **kwargs)

    orders = table('orders', primary_key='ORDER_ID',
                   time_column='ORDER_TIME').with_column(
                       'USER_ID',
                       stype=Stype.ID).with_column('ITEM_ID', stype=Stype.ID)
    return Graph(
        tables={
            'users': table('users', primary_key='USER_ID'),
            'items': table('items', primary_key='ITEM_ID'),
            'orders': orders,
        }, edges=[
            Edge(src_table='orders', fkey='USER_ID', dst_table='users'),
            Edge(src_table='orders', fkey='ITEM_ID', dst_table='items'),
        ])


def tracking(connector: Connector) -> Any:
    registry = os.environ.get('KUMO_REGISTRY')
    if registry == 'mlflow':
        from kumo.engine.integrations.mlflow import MlflowTracking

        return MlflowTracking(registered_model_name=_env('KUMO_MODEL_NAME'),
                              aliases=[CHAMPION])
    if registry == 'snowflake':
        from kumo.engine.integrations.snowflake import SnowflakeTracking

        assert isinstance(connector, SnowflakeConnector)
        return SnowflakeTracking(connector.session(),
                                 experiment_name=_env('KUMO_MODEL_NAME'))
    return None


def model_destination(
        connector: Connector) -> Tuple[Any, Optional[ModelOutputConfig]]:
    """The ``tracking`` and ``export_config`` that publish a fit."""
    registry = os.environ.get('KUMO_REGISTRY')
    if registry == 'snowflake':
        location = ModelRegistryLocation(_env('KUMO_MODEL_NAME'),
                                         alias=CHAMPION)
        return tracking(connector), ModelOutputConfig(connector,
                                                      location=location)
    if registry == 'mlflow':
        return tracking(connector), None
    return None, ModelOutputConfig(run_artifacts(), path='model')


def load_champion(connector: Connector) -> Trainer:
    registry = os.environ.get('KUMO_REGISTRY')
    if registry == 'mlflow':
        from kumo.engine.integrations.mlflow import load_model

        return load_model(f"models:/{_env('KUMO_MODEL_NAME')}@{CHAMPION}")
    if registry == 'snowflake':
        return Trainer.load_from_connector(
            connector, location=ModelRegistryLocation(
                _env('KUMO_MODEL_NAME'), alias=CHAMPION))
    return Trainer.load_from_connector(artifacts(),
                                       path=_env('KUMO_MODEL_PATH'))


def prediction_output(connector: Connector) -> PredictionOutputConfig:
    fields = [MetadataField.ANCHOR_TIMESTAMP]
    table = os.environ.get('KUMO_PREDICTIONS_TABLE')
    if table:
        return PredictionOutputConfig(
            connector, artifact_type=PredictionArtifactType.PREDICTIONS,
            table_name=table, extra_fields=fields, lineage_columns=True)
    return PredictionOutputConfig(run_artifacts(),
                                  output_paths={'predictions': 'predictions/'},
                                  extra_fields=fields, lineage_columns=True)


def prediction_destinations(output: PredictionOutputConfig) -> dict:
    """Where ``output`` writes: full URIs, or the table name."""
    if output.output_paths is None:
        return output.destinations()
    return {
        name: artifact_uri(f'{_run_path()}/{path}')
        for name, path in output.output_paths.items()
    }
