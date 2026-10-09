# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Running evaluations on multiple asynchronously launched predictive
queries. We assume that training jobs in `train.py` are triggered.

And then we follow up with
  (1) if E_amount PQ succeeded, then run a BP on the holdout set to produce
    predictions for all entities
  (2) perform evaluations of the ultimate prediction task if all the jobs
    succeeded
  (3) slightly modify only one job
"""
import json
import os

import pandas as pd

import kumoai_studio as kumo

API_URL = os.getenv("KUMO_API_URL", "http://localhost:10002")
API_KEY = os.environ["KUMO_API_KEY"]
kumo.init(url=API_URL, api_key=API_KEY, project=os.environ["KUMO_PROJECT"])
PROJECT_ID = kumo.current_project().id


def combine_test_dfs(test_zero, test_value):
    """Combine two PQ outputs."""

    # TODO(manan): Update to use Spark:
    def RUN_SQL(s):
        return pd.DataFrame()

    combined_df = RUN_SQL("""
        SELECT test_zero.customer_id,
            test_zero.True_PROB * test_value.PREDICTED AS predicted,
            COALESCE(test_value.value, 0) AS actual
        FROM test_zero LEFT JOIN test_value ON customer_id
    """)
    return combined_df


# eval metrics from predicted and actual
def compute_metric(combined_df):
    # example metric : MAE
    # TODO(manan): Update to use Spark:
    # query_dict['P_conversion']['holdout_df'],
    return 0


query_dict = None
with open('jobs.json', 'r') as fp:
    # The structure of this dictionary is in `train.py`:
    query_dict = json.load(fp)

for pq_name, pq_dict in query_dict.items():
    # Block until the job is complete:
    job_completed = kumo.TrainingJob(pq_dict['training_job'],
                                     project_id=PROJECT_ID).result()
    assert job_completed
    print(f"Job corresponding to query {pq_name} has completed.")

    pq_dict['holdout_df'] = job_completed.holdout_df()  # a Pandas dataframe.

final_test_df = combine_test_dfs(
    test_zero=query_dict['P_conversion']['holdout_df'],
    test_value=query_dict['E_amount']['holdout_df'],
)

# Compute metric:
mae = compute_metric(final_test_df)
print(f'MAE = {mae}')

# Update scenario: plan to update the regression model
# new_model_planner = LOAD MODEL PLANNER FROM my_pqs['E_amount']['trainer_id']
# new_model_planner['num_neighbors'] = [256, 256] # New update
# new_train_amount = my_pqs['E_amount']['PQ'].fit(non_blocking=True,
#   new_model_planner)
