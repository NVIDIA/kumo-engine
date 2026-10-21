# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train and publish the users-churn model: ``python fit_job.py [--job-id ID
--job-run-id ID]``; see README.md."""

import os
import sys
from dataclasses import asdict
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jobs  # noqa: E402

import kumo.engine as kumo  # noqa: E402

# Needs several 30-day windows of order history; shorten it for small data.
QUERY = 'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.USER_ID'


def main(argv: Optional[Sequence[str]] = None) -> None:
    job_id, job_run_id = jobs.start(argv)
    with kumo.job(job_id=job_id, job_run_id=job_run_id) as run:
        connector = jobs.source()
        graph = jobs.build_graph(connector)
        snapshot = graph.snapshot()
        run.record(snapshot_id=snapshot.id)
        if os.environ.get('KUMO_ARTIFACT_ROOT'):
            run.record(outputs={
                'snapshot':
                jobs.artifact_uri(snapshot.export(jobs.artifacts(), 'snapshots'))
            })

        query = kumo.PredictiveQuery(graph=graph,
                                     query=os.environ.get('KUMO_QUERY', QUERY))
        table = query.generate_training_table(snapshot)
        plan = query.suggest_model_plan(snapshot)
        if os.environ.get('KUMO_NUM_EXPERIMENTS'):
            plan.training_job.num_experiments = int(
                os.environ['KUMO_NUM_EXPERIMENTS'])
        tracking, export_config = jobs.model_destination(connector)
        result = kumo.Trainer(plan).fit(table, snapshot=snapshot,
                                        tracking=tracking,
                                        export_config=export_config)

        registered = getattr(tracking, 'registered', None)
        published = registered or result.published_model
        run.record(
            bundle_id=result.bundle_id,
            run_id=getattr(tracking, 'run_id', None),
            outputs={'model': published and asdict(published)})


if __name__ == '__main__':
    main()
