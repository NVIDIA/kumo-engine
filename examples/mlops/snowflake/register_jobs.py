# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental: register fit_job.py/score_job.py as Snowflake ML Jobs with
one Task each (snowflake-ml-python 2.6, preview API); see README.md."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PAYLOAD = Path(__file__).resolve().parent.parent
JOBS = {
    'train': {
        'entrypoint': 'fit_job.py',
        'schedule': 'USING CRON 0 2 * * 0 UTC',
        'timeout_ms': 24 * 3600 * 1000,
        'args': [],
    },
    'score': {
        'entrypoint': 'score_job.py',
        'schedule': 'USING CRON 0 3 * * * UTC',
        'timeout_ms': 6 * 3600 * 1000,
        'args': ['--predictions-table', 'USERS_CHURN_PREDICTIONS'],
    },
}


def task_sql(*, task_name: str, warehouse: str, schedule: str,
             timeout_ms: int, execute_ml_job_sql: str) -> str:
    return (f"CREATE OR REPLACE TASK {task_name}\n"
            f"  WAREHOUSE = {warehouse}\n"
            f"  SCHEDULE = '{schedule}'\n"
            f"  USER_TASK_TIMEOUT_MS = {timeout_ms}\n"
            f"  SUSPEND_TASK_AFTER_NUM_FAILURES = 3\n"
            f"AS\n{execute_ml_job_sql}")


def register(session: Any, *, compute_pool: str, stage: str, database: str,
             schema: str, warehouse: str, artifact_root: str,
             requirements: List[str],
             external_access_integrations: List[str]) -> Dict[str, str]:
    from snowflake.ml.jobs import MLJobDefinition

    flags = [
        '--artifact-root', artifact_root, '--source', 'snowflake',
        '--catalog', database, '--schema', schema, '--warehouse', warehouse,
        '--registry', 'snowflake', '--model-name', 'USERS_CHURN',
        '--scratch-dir', '/tmp/kumo'
    ]
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
                execute_ml_job_sql=definition.to_sql(
                    job_args=[*flags, *job['args']], use_async=False),
            )).collect()
        tasks[kind] = name
    return tasks


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', help='connections.toml entry')
    for flag in ('--compute-pool', '--stage', '--database', '--schema',
                 '--warehouse', '--artifact-root'):
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
        artifact_root=args.artifact_root,
        requirements=args.requirement,
        external_access_integrations=args.external_access_integration)
    for kind, name in tasks.items():
        print(f'{kind}: {name} (suspended; ALTER TASK {name} RESUME)')


if __name__ == '__main__':
    main()
