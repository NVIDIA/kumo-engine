# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Polling multiple asynchronously launched predictive queries, from the
train.py script.

This only checks whether all my jobs finished or not. Hence, the least
dependence is required. The polling should be definitely runnable on local
laptop and ideally even on no-code tool.

Either OPTION 1 or OPTION 2 needs to be
supported. GUI support is preferred.  Finally, we add another scneario that one
of the job failed, so we need to re-trigger the job

There are several factors to consider:
  - Non-blocking training
  - Final output/conclusion can be drawn only after all the jobs are successful
  - Some exception can happen to one or both jobs.
  - For potential iterations for one of the two jobs, we should lock graph

We assume that all the follow-up operations happen in a new session.
Please see train_poll.py (wait for the jobs' completion) and train_eval.py
(evaluate completed jobs).
"""
import json
import os

from kumoapi.common import JobStatus

import kumoai_studio as kumo

API_URL = os.getenv("KUMO_API_URL", "http://localhost:10002")
API_KEY = os.environ["KUMO_API_KEY"]
kumo.init(url=API_URL, api_key=API_KEY, project=os.environ["KUMO_PROJECT"])
PROJECT_ID = kumo.current_project().id


def retrain(job_id: str, non_blocking: bool):
    r"""Retrains a job given a `job_id`."""
    trainer_loaded = kumo.Trainer.load(job_id, project_id=PROJECT_ID)
    pq_loaded = kumo.PredictiveQuery.load_from_training_job(
        job_id, project_id=PROJECT_ID)

    # TODO(manan): Today, it is not easy for us to load the saved training
    # table generation plan, so any modifications need
    # to be explicitly specified when calling `generate_training_table` again.
    # This should be resolved; for now, we just re-train without overrides
    # (which is not correct):
    job_or_fut = trainer_loaded.fit(
        graph=pq_loaded.graph,
        train_table=pq_loaded.generate_training_table(),
        non_blocking=non_blocking,
    )
    return job_or_fut.id


query_dict = None
with open('jobs.json', 'r') as fp:
    # The structure of this dictionary is in `train.py`:
    query_dict = json.load(fp)

query_done = {pq_name: False for pq_name in query_dict.keys()}
while not all(query_done.values()):
    for pq_name, pq_dict in query_dict.items():
        job_id = pq_dict['training_job']
        future = kumo.TrainingJob(job_id=job_id, project_id=PROJECT_ID)
        status_report = future.status()
        url = future.tracking_url
        print(f"Query {pq_name} (job {job_id}) has status {status_report}. "
              f"Tracking URL: {url}")

        if status_report.status.is_terminal:
            query_done[pq_name] = True
            if not status_report.status == JobStatus.DONE:
                print(f"Warning: query {pq_name} failed with "
                      f"status {status_report}")
                new_id = retrain(job_id, non_blocking=True)
                query_dict[pq_name]['training_job'] = new_id

with open('jobs.json', 'w') as fp:
    json.dump(query_dict, fp, sort_keys=True, indent=4)
