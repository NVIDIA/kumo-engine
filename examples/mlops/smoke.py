# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manual live check, not run in CI: fit a small model on synthetic data on
this machine, record it in Databricks, SageMaker or Snowflake with an
explicit identity, register it, and load it back. A few minutes on a CPU.
Install kumo-engine[mlflow,databricks], kumo-engine[mlflow,sagemaker] or
kumo-engine[snowflake].

    python smoke.py databricks --host https://<workspace> --client-id <id> \
        --catalog main --schema ml --warehouse <id> --experiment /Users/<you>/kumo-smoke
    python smoke.py sagemaker --arn <tracking server ARN> --role-arn <role ARN>
    python smoke.py snowflake --database ML --schema RUNS --warehouse <wh> \
        --account <account> --user <user> --private-key-file key.p8

Secrets come from DATABRICKS_CLIENT_SECRET (Databricks) or the key file
(Snowflake); --role-arn may be left out to use the ambient AWS identity,
with no MLFLOW_TRACKING_URI needed.
"""

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import kumo.engine as kumo
from kumo.engine import ModelRegistryLocation
from kumo.engine.connector import (
    DatabricksConnector,
    DatabricksIdentity,
    S3Identity,
    SnowflakeConnector,
    SnowflakeIdentity,
)

QUERY = 'PREDICT COUNT(orders.*, 0, 7, days) > 0 FOR EACH users.USER_ID'


def data(root: Path) -> kumo.Graph:
    rng = np.random.default_rng(0)
    root.mkdir(parents=True)
    pd.DataFrame({'USER_ID': np.arange(200)}).to_parquet(root / 'users.parquet')
    pd.DataFrame({
        'ORDER_ID': np.arange(6000),
        'USER_ID': rng.integers(0, 200, 6000),
        'ORDER_TIME': pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 6000), unit='D'),
    }).to_parquet(root / 'orders.parquet')
    files = kumo.FileConnector(root_dir=str(root))
    orders = kumo.Table.from_source_table(
        files['orders.parquet'], primary_key='ORDER_ID',
        time_column='ORDER_TIME').infer_metadata()
    return kumo.Graph(
        tables={
            'users': kumo.Table.from_source_table(
                files['users.parquet'], primary_key='USER_ID'),
            'orders': orders,
        }, edges=[kumo.Edge(src_table='orders', fkey='USER_ID',
                            dst_table='users')])


def tracking(args: argparse.Namespace, suffix: str):
    from kumo.engine.integrations.mlflow import MlflowTracking, SageMakerMLflow

    if args.platform == 'databricks':
        workspace = DatabricksConnector(
            catalog=args.catalog, schema_name=args.schema,
            warehouse_id=args.warehouse, identity=DatabricksIdentity(
                host=args.host, client_id=args.client_id,
                client_secret=os.environ['DATABRICKS_CLIENT_SECRET']))
        return (MlflowTracking(target=workspace,
                               experiment_name=args.experiment),
                f'{args.catalog}.{args.schema}.kumo_smoke_{suffix}')
    if args.platform == 'sagemaker':
        identity = S3Identity(role_arn=args.role_arn) if args.role_arn else None
        return (MlflowTracking(
            target=SageMakerMLflow(args.arn, identity=identity),
            experiment_name='kumo-smoke'), f'kumo-smoke-{suffix}')
    from kumo.engine.integrations.snowflake import SnowflakeTracking

    connector = SnowflakeConnector(
        warehouse=args.warehouse, database=args.database,
        schema_name=args.schema, identity=SnowflakeIdentity(
            account=args.account, username=args.user,
            private_key=Path(args.private_key_file).read_text()))
    return (SnowflakeTracking(connector, experiment_name='KUMO_SMOKE'),
            f'KUMO_SMOKE_{suffix}')


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('platform',
                        choices=('databricks', 'sagemaker', 'snowflake'))
    for flag in ('--host', '--client-id', '--catalog', '--schema',
                 '--warehouse', '--experiment', '--arn', '--role-arn',
                 '--database', '--account', '--user', '--private-key-file'):
        parser.add_argument(flag)
    parser.add_argument('--artifact-root',
                        help='defaults to a local temporary directory')
    args = parser.parse_args(argv)

    started = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix='kumo-smoke-'))
    kumo.configure(scratch_dir=work / 'scratch')
    graph = data(work / 'data')
    tracker, model_name = tracking(args, time.strftime('%Y%m%d%H%M%S'))
    location = ModelRegistryLocation(model_name, alias='candidate')
    cfg = kumo.MLOps(args.artifact_root or str(work / 'root'),
                     tracking=tracker, registry=location, on_error='raise')
    print(f'checked the target in {time.monotonic() - started:.0f} s')
    with kumo.mlops(cfg):
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph=graph, query=QUERY)
        table = query.generate_training_table(snapshot)
        plan = query.suggest_model_plan(snapshot)
        plan.training_job.num_experiments = 1
        plan.optimization.max_epochs = 1
        result = kumo.Trainer(plan).fit(table, snapshot=snapshot)
    print(f'fit run {result.tracking}, bundle {result.id}, '
          f'registered {result.registered_model}')
    loaded = kumo.Trainer.load_from_connector(cfg, location=location)
    assert loaded.result.id == result.id, (loaded.result.id, result.id)
    print('loaded the registered model back')
    print(f'done in {time.monotonic() - started:.0f} s')


if __name__ == '__main__':
    main()
