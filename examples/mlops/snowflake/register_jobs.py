# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental: register the fit and score jobs as Snowflake Tasks; see
README.md."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PAYLOAD = Path(__file__).resolve().parent.parent
JOBS = (
    ('train', 'fit_job.py', 'USING CRON 0 2 * * 0 UTC', 24),
    ('score', 'score_job.py', 'USING CRON 0 3 * * * UTC', 6),
)


def task_sql(*, task_name: str, warehouse: str, schedule: str,
             timeout_ms: int, execute_ml_job_sql: str) -> str:
    return (f"CREATE OR REPLACE TASK {task_name}\n"
            f"  WAREHOUSE = {warehouse}\n"
            f"  SCHEDULE = '{schedule}'\n"
            f"  USER_TASK_TIMEOUT_MS = {timeout_ms}\n"
            f"  SUSPEND_TASK_AFTER_NUM_FAILURES = 3\n"
            f"AS\n{execute_ml_job_sql}")


def _started(session: Any, database: str, schema: str, task: str) -> bool:
    rows = session.sql(
        f"SHOW TASKS LIKE '{task}' IN SCHEMA {database}.{schema}").collect()
    return any(row['name'] == task and row['state'] == 'started'
               for row in rows)


def register(session: Any, *, compute_pool: str, stage: str, database: str,
             schema: str, warehouse: str, artifact_root: str,
             requirements: List[str],
             external_access_integrations: List[str],
             model_name: str = 'USERS_CHURN',
             predictions_table: str = 'USERS_CHURN_PREDICTIONS',
             table_prefix: str = '',
             on_error: str = 'raise') -> Dict[str, str]:
    r"""Returns each task's state; a replaced started task is resumed."""
    from snowflake.ml.jobs import MLJobDefinition

    flags = [
        '--artifact-root', artifact_root, '--source', 'snowflake',
        '--catalog', database, '--schema', schema, '--warehouse', warehouse,
        '--registry', 'snowflake', '--model-name', model_name,
        f'--table-prefix={table_prefix}', '--on-error', on_error,
        '--scratch-dir', '/tmp/kumo'
    ]
    tasks = {}
    for kind, entrypoint, schedule, hours in JOBS:
        definition = MLJobDefinition.register(
            str(PAYLOAD),
            compute_pool,
            stage,
            session=session,
            name=f'KUMO_{kind.upper()}',
            entrypoint=entrypoint,
            pip_requirements=requirements,
            external_access_integrations=external_access_integrations,
            query_warehouse=warehouse,
            database=database,
            schema=schema,
            overwrite=True,
        )
        task = f'KUMO_{kind.upper()}_TASK'
        name = f'{database}.{schema}.{task}'
        started = _started(session, database, schema, task)
        session.sql(
            task_sql(
                task_name=name, warehouse=warehouse,
                schedule=schedule, timeout_ms=hours * 3600 * 1000,
                execute_ml_job_sql=definition.to_sql(
                    job_args=(flags if kind == 'train' else [
                        *flags, '--predictions-table', predictions_table
                    ]), use_async=False),
            )).collect()
        if started:
            session.sql(f'ALTER TASK {name} RESUME').collect()
        tasks[name] = 'started' if started else 'suspended'
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
    parser.add_argument('--model-name', default='USERS_CHURN')
    parser.add_argument('--predictions-table',
                        default='USERS_CHURN_PREDICTIONS')
    parser.add_argument('--table-prefix', default='')
    parser.add_argument('--on-error', choices=('raise', 'warn'),
                        default='raise')
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
        external_access_integrations=args.external_access_integration,
        model_name=args.model_name, predictions_table=args.predictions_table,
        table_prefix=args.table_prefix, on_error=args.on_error)
    for name, state in tasks.items():
        print(f'{name}: {state}' + ('' if state == 'started' else
                                    f' (ALTER TASK {name} RESUME)'))


if __name__ == '__main__':
    main()
