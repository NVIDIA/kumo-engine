# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register fit_job.py/score_job.py as Snowflake ML Jobs with one Task each
(snowflake-ml-python 2.6); see README.md."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE.parent
JOB_IDS = ['--job-id', '__KUMO_JOB_ID__', '--job-run-id', '__KUMO_JOB_RUN_ID__']
JOBS = {
    'train': {
        'entrypoint': 'fit_job.py',
        'schedule': 'USING CRON 0 2 * * 0 UTC',
        'timeout_ms': 24 * 3600 * 1000,
    },
    'score': {
        'entrypoint': 'score_job.py',
        'schedule': 'USING CRON 0 3 * * * UTC',
        'timeout_ms': 6 * 3600 * 1000,
    },
}


def task_sql(*, task_name: str, warehouse: str, schedule: str,
             timeout_ms: int, execute_ml_job_sql: str) -> str:
    text = (HERE / 'task.sql').read_text()
    text = text[text.index('CREATE OR REPLACE TASK'):]
    if '$$' in execute_ml_job_sql:
        raise ValueError('the job SQL cannot be embedded in a $$ string')
    for slot, value in (('<task_name>', task_name), ('<warehouse>', warehouse),
                        ('<schedule>', schedule),
                        ('<timeout_ms>', str(timeout_ms)),
                        ('<execute_ml_job_sql>', execute_ml_job_sql)):
        text = text.replace(slot, value)
    return text


def register(session: Any, *, compute_pool: str, stage: str, database: str,
             schema: str, warehouse: str, requirements: List[str],
             external_access_integrations: List[str]) -> Dict[str, str]:
    from snowflake.ml.jobs import MLJobDefinition

    env = {
        'KUMO_SOURCE': 'snowflake',
        'KUMO_SNOWFLAKE_WAREHOUSE': warehouse,
        'KUMO_SNOWFLAKE_DATABASE': database,
        'KUMO_SNOWFLAKE_SCHEMA': schema,
        'KUMO_REGISTRY': 'snowflake',
        'KUMO_MODEL_NAME': 'USERS_CHURN',
        'KUMO_PREDICTIONS_TABLE': 'USERS_CHURN_PREDICTIONS',
        'KUMO_SCRATCH_DIR': '/tmp/kumo',
    }
    tasks = {}
    for kind, job in JOBS.items():
        definition = MLJobDefinition.register(
            str(PAYLOAD),
            compute_pool,
            stage,
            session=session,
            name=f'KUMO_{kind.upper()}',
            entrypoint=str(job['entrypoint']),
            pip_requirements=requirements,
            external_access_integrations=external_access_integrations,
            env_vars=env,
            query_warehouse=warehouse,
            database=database,
            schema=schema,
            overwrite=True,
        )
        name = f'{database}.{schema}.KUMO_{kind.upper()}_TASK'
        session.sql(
            task_sql(
                task_name=name, warehouse=warehouse,
                schedule=str(job['schedule']),
                timeout_ms=int(job['timeout_ms']),
                execute_ml_job_sql=definition.to_sql(job_args=JOB_IDS,
                                                     use_async=False),
            )).collect()
        tasks[kind] = name
    return tasks


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', help='connections.toml entry')
    for flag in ('--compute-pool', '--stage', '--database', '--schema',
                 '--warehouse'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--requirement', action='append', required=True)
    parser.add_argument('--external-access-integration', action='append',
                        default=[])
    args = parser.parse_args(argv)

    from snowflake.snowpark import Session

    builder = Session.builder
    if args.connection is not None:
        builder = builder.config('connection_name', args.connection)
    tasks = register(
        builder.create(), compute_pool=args.compute_pool, stage=args.stage,
        database=args.database, schema=args.schema, warehouse=args.warehouse,
        requirements=args.requirement,
        external_access_integrations=args.external_access_integration)
    for kind, name in tasks.items():
        print(f'{kind}: {name} (suspended; ALTER TASK {name} RESUME)')


if __name__ == '__main__':
    main()
