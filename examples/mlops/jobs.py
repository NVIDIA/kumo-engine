# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The flags, graph, connectors and MLOps settings both job scripts share."""

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence

from kumoapi.jobs import MetadataField, WriteMode
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine import (
    Edge,
    Graph,
    ModelRegistryLocation,
    PredictionOutputConfig,
    Table,
    Trainer,
)
from kumo.engine.connector import (
    Connector,
    DatabricksConnector,
    SnowflakeConnector,
    connector_for_root,
    is_file_destination,
)

QUERY = 'PREDICT COUNT(orders.*, 0, 7, days) > 0 FOR EACH users.USER_ID'
# 'table.COLUMN' names to encode as text
TEXT_COLUMNS: Sequence[str] = ()


def parse_args(argv: Optional[Sequence[str]], *,
               score: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--artifact-root', required=True,
        help='s3://, gs://, file://, a local or /Volumes path, or with '
        '--source snowflake an @DB.SCHEMA.STAGE path')
    parser.add_argument(
        '--source', required=True,
        help='databricks, snowflake, or a directory holding users, items '
        'and orders parquet files')
    parser.add_argument('--catalog',
                        help='Databricks catalog or Snowflake database')
    parser.add_argument('--schema')
    parser.add_argument(
        '--table-prefix', default='',
        help='prefix of the users, items and orders table or file names')
    parser.add_argument(
        '--warehouse', help='Databricks SQL warehouse id or Snowflake '
        'warehouse')
    parser.add_argument('--registry', choices=('mlflow', 'snowflake'))
    parser.add_argument('--model-name')
    parser.add_argument(
        '--alias', default='champion',
        help='alias the fit moves to its registered version and the score '
        'job loads')
    parser.add_argument('--scratch-dir')
    parser.add_argument(
        '--model-path-file',
        help='file the score job reads --model-path from, written by the fit '
        'job')
    parser.add_argument(
        '--on-error', choices=('raise', 'warn'), default='raise',
        help='raise (fail the job) or warn when tracking or registration '
        'fails')
    if score:
        parser.add_argument(
            '--model-path',
            help='models/<bundle id> under the artifact root, loaded instead '
            'of --alias')
        parser.add_argument(
            '--snapshot', help='the id of a snapshot stored under the '
            'artifact root, instead of a new one')
        parser.add_argument('--predictions-table')
    else:
        parser.add_argument('--query', default=QUERY)
        parser.add_argument('--num-experiments', type=int)
        parser.add_argument(
            '--threshold', type=float, default=0.5,
            help='binary classification threshold the model scores with')
    args = parser.parse_args(argv)
    if score and args.model_path_file:
        args.model_path = Path(args.model_path_file).read_text()
    if args.registry and not args.model_name:
        parser.error('--registry needs --model-name')
    if score and not args.registry and not args.model_path:
        parser.error('without --registry, pass --model-path')
    if args.scratch_dir:
        kumo.configure(scratch_dir=args.scratch_dir)
    return args


def source(args: argparse.Namespace) -> Connector:
    if args.source == 'databricks':
        return DatabricksConnector(catalog=args.catalog,
                                   schema_name=args.schema,
                                   warehouse_id=args.warehouse)
    if args.source == 'snowflake':
        return SnowflakeConnector.from_spcs_env(warehouse=args.warehouse,
                                                database=args.catalog,
                                                schema_name=args.schema)
    return connector_for_root(args.source)


def mlops(args: argparse.Namespace, connector: Connector) -> kumo.MLOps:
    root = ((connector, args.artifact_root)
            if args.artifact_root.startswith('@') else args.artifact_root)
    tracking: Any = None
    if args.registry == 'mlflow':
        from kumo.engine.integrations.mlflow import MlflowTracking

        tracking = MlflowTracking()
    elif args.registry == 'snowflake':
        from kumo.engine.integrations.snowflake import SnowflakeTracking

        tracking = SnowflakeTracking(connector,
                                     experiment_name=args.model_name)
    return kumo.MLOps(root, tracking=tracking,
                      registry=registry(args) if tracking else None,
                      on_error=args.on_error)


def registry(args: argparse.Namespace) -> ModelRegistryLocation:
    return ModelRegistryLocation(args.model_name, alias=args.alias)


def build_graph(connector: Connector, prefix: str = '') -> Graph:

    def table(alias: str, **kwargs: str) -> Table:
        alias = f'{prefix}{alias}'
        if isinstance(connector, SnowflakeConnector):
            name = alias.upper()
        elif isinstance(connector, DatabricksConnector):
            name = alias
        else:
            name = f'{alias}.parquet'
        return Table.from_source_table(connector[name], **kwargs)

    orders = table('orders', primary_key='ORDER_ID',
                   time_column='ORDER_TIME').with_column(
                       'USER_ID',
                       stype=Stype.ID).with_column('ITEM_ID', stype=Stype.ID)
    tables = {
        'users': table('users', primary_key='USER_ID'),
        'items': table('items', primary_key='ITEM_ID'),
        'orders': orders,
    }
    for name in TEXT_COLUMNS:
        alias, column = name.split('.')
        tables[alias] = tables[alias].with_column(column, stype=Stype.text)
    return Graph(
        tables=tables, edges=[
            Edge(src_table='orders', fkey='USER_ID', dst_table='users'),
            Edge(src_table='orders', fkey='ITEM_ID', dst_table='items'),
        ])


def load_model(args: argparse.Namespace, cfg: kumo.MLOps) -> Trainer:
    if args.model_path:
        return Trainer.load_from_connector(cfg, path=args.model_path)
    return Trainer.load_from_connector(cfg, location=registry(args))


def prediction_output(
        args: argparse.Namespace,
        connector: Connector) -> Optional[PredictionOutputConfig]:
    path = args.predictions_table
    if not path:
        return None
    return PredictionOutputConfig(
        connector, path=path,
        write_mode=(WriteMode.OVERWRITE if is_file_destination(
            connector, path) else WriteMode.APPEND),
        extra_fields=[MetadataField.ANCHOR_TIMESTAMP])
