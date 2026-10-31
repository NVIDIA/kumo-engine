# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live smoke: fit a small model on synthetic data, register it in the
Snowflake Model Registry with a serving snapshot on a stage, score it with
``ModelVersion.run_batch`` on a GPU compute pool, and compare the job's output
with local scores. The steps were verified live; this script was not run
itself. See "Batch inference in Snowflake" in
packages/kumo-engine/docs/mlops.md for the pool, grants and stage it needs.

    python run_batch_smoke.py --connection-name <connections.toml entry> \\
        --warehouse WH --database ML --schema MODELS \\
        --stage @ML.MODELS.KUMO --compute-pool KUMO_GPU_POOL
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo


def fit(data: Path) -> tuple:
    rng = np.random.default_rng(0)
    pd.DataFrame({'ID': np.arange(200)}).to_parquet(data / 'users.parquet')
    pd.DataFrame({
        'OID': np.arange(5000),
        'USER_ID': rng.integers(0, 200, 5000),
        'TS': pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 5000), unit='D'),
    }).to_parquet(data / 'orders.parquet')
    files = kumo.FileConnector(root_dir=str(data))
    graph = kumo.Graph(
        tables={
            'users':
            kumo.Table.from_source_table(files['users.parquet'],
                                         primary_key='ID').infer_metadata(),
            'orders':
            kumo.Table.from_source_table(
                files['orders.parquet'], primary_key='OID',
                time_column='TS').infer_metadata().with_column(
                    'USER_ID', stype=Stype.ID),
        }, edges=[Edge(src_table='orders', fkey='USER_ID', dst_table='users')])
    snapshot = graph.snapshot()
    query = kumo.PredictiveQuery(
        graph=graph, query='PREDICT COUNT(orders.*, 0, 7, days) > 0 '
        'FOR EACH users.ID')
    return snapshot, query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--connection-name', required=True)
    parser.add_argument('--warehouse', required=True)
    parser.add_argument('--database', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--stage', required=True,
                        help='named internal stage, @DB.SCHEMA.STAGE')
    parser.add_argument('--compute-pool', required=True)
    parser.add_argument('--model-name', default='KUMO_RUN_BATCH_SMOKE')
    parser.add_argument('--pip-repository', default=None)
    args = parser.parse_args()

    from snowflake.ml.model.batch_inference import (
        InferenceSpec,
        InputSpec,
        OutputSpec,
        ResourcesSpec,
        SaveMode,
    )
    from snowflake.ml.registry import Registry

    sf = kumo.SnowflakeConnector.from_connection_toml(
        connection_name=args.connection_name, warehouse=args.warehouse,
        database=args.database, schema_name=args.schema)
    stage = args.stage.rstrip('/')
    with tempfile.TemporaryDirectory() as tmp:
        snapshot, query = fit(Path(tmp))
        exported = snapshot.export(
            kumo.SnapshotOutputConfig(sf, f'{stage}/smoke/snapshots'))
        trainer = kumo.Trainer(model_plan=ModelPlan())
        result = trainer.fit(
            query.generate_training_table(snapshot), snapshot=snapshot,
            export_config=kumo.ModelOutputConfig(
                sf, location=kumo.ModelRegistryLocation(args.model_name),
                serving_snapshot=exported,
                serving_options={'pip_repository': args.pip_repository}
                if args.pip_repository else None))
        published = result.published_model
        print(f'registered {published.name} version {published.version}')

        table = query.generate_prediction_table(snapshot)
        time = pd.Timestamp(table.timestamps[0])
        expected = trainer.predict(table, snapshot=snapshot,
                                   binary_classification_threshold=0.5).frame

    session = sf.session()
    version = Registry(session, database_name=args.database,
                       schema_name=args.schema).get_model(
                           published.name).version(published.version)
    rows = session.create_dataframe(
        pd.DataFrame({'ENTITY': expected['ENTITY'].astype(str)}))
    output = f'{stage}/smoke/scores/'
    job = version.run_batch(
        rows, compute_pool=args.compute_pool,
        output_spec=OutputSpec(stage_location=output, mode=SaveMode.OVERWRITE),
        input_spec=InputSpec(params={
            'prediction_time': str(time),
            'binary_classification_threshold': 0.5,
        }), resources_spec=ResourcesSpec(gpu_requests='1'),
        inference_spec=InferenceSpec(num_workers=1))
    job.wait()
    job_name = job.id.split('.')[-1].strip('"')
    scored = session.read.parquet(f'{output}{job_name}/').to_pandas()
    scored.columns = [column.strip('"') for column in scored.columns]

    local = expected.assign(ENTITY=expected['ENTITY'].astype(str)).sort_values(
        'ENTITY').reset_index(drop=True)
    remote = scored[list(local.columns)].sort_values('ENTITY').reset_index(
        drop=True)
    pd.testing.assert_frame_equal(local, remote, check_dtype=False, atol=1e-5)
    print(f'PASS: {len(remote)} rows match local scores')


if __name__ == '__main__':
    main()
