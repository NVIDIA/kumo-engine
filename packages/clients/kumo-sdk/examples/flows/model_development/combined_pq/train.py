# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Writing a staged predictive query on the H&M dataset.

This script produces a final prediction score on a predictive problem defined
over the H&M dataset, structured as a combination of two Kumo predictive
queries.

This example also represents the scenario where the developer needs multiple
training to draw some conclusion.

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
from typing import Any

import kumoai_studio as kumo

# Setup:
API_URL = os.getenv("KUMO_API_URL", "http://localhost:10002")
API_KEY = os.environ["KUMO_API_KEY"]
kumo.init(url=API_URL, api_key=API_KEY, project=os.environ["KUMO_PROJECT"])

# Connect to data:
connector = kumo.S3Connector(
    name="combined_pquery_source",
    identity=os.environ["KUMO_S3_IDENTITY_ID"],
    root_dir="s3://kumo-public-datasets/filtered_hnm_temp/",
)

# Connect tables, let Kumo infer all metadata:
customers_src = connector.table("customers")
customers = kumo.Table(
    source_table=customers_src,
    primary_key="customer_id",
    columns=customers_src.columns,
).infer_metadata()

articles_src = connector.table("articles")
articles = kumo.Table(
    source_table=articles_src,
    primary_key="article_id",
    columns=articles_src.columns,
).infer_metadata()

transactions_src = connector.table("transactions_train")
transactions = kumo.Table(
    source_table=transactions_src,
    time_column="t_dat",
    columns=transactions_src.columns,
).infer_metadata()

# Define graph. Note that this graph is "persisted/locked"; that is, this
# graph definition can be used for any downstream experimentation. Any change
# in a graph will result in a new definition, totally decoupled from this
# one:
graph = kumo.Graph(
    tables={
        'customers': customers,
        'articles': articles,
        'transactions': transactions,
    },
    edges=[
        dict(src_table='transactions', fkey='customer_id',
             dst_table='customers'),
        dict(src_table='transactions', fkey='article_id',
             dst_table='articles'),
    ],
)
graph.validate()

# Define two predictive queries:
pq_zero_query = """
PREDICT COUNT(transactions.*, 0, 28, DAYS) > 0 FOR EACH customers.customer_id
"""
pq_zero = kumo.PredictiveQuery(
    graph=graph,
    query=pq_zero_query,
)
pq_zero.validate()
print(f"Query {pq_zero_query} has task type {pq_zero.get_task_type()}")

pq_value_query = """
    PREDICT SUM(transactions.amount, 0, 28, DAYS)
    FOR EACH customers.customer_id
    ASSUMING COUNT(transactions.*, 0, 28, DAYS) > 0
"""
pq_value = kumo.PredictiveQuery(
    graph=graph,
    query=pq_value_query,
)
pq_value.validate()
print(f"Query {pq_value_query} has task type {pq_value.get_task_type()}")

# Execute two PQs in parallel
query_dict: dict[str, dict[str, Any]] = {
    'P_conversion': {
        'pq': pq_zero
    },
    'E_amount': {
        'pq': pq_value
    },
}
for pq_name, pq_dict in query_dict.items():
    pq = pq_dict['pq']

    # Customize the training table generation plan:
    train_table_generation_plan = pq.suggest_training_table_plan()
    train_table_generation_plan.split = (
        'TimeRangeSplit([("1994-01-01", "1997-01-01"), '
        '("1997-01-01", "1998-01-01"), ("1998-01-01", "1999-01-01")])')

    # Customize the model plan:
    model_plan = pq.suggest_model_plan()
    model_plan.neighbor_sampling.num_neighbors = [[128, 12]]

    # Non-blocking training table generation and training:
    training_table_fut = pq.generate_training_table(
        plan=train_table_generation_plan,
        non_blocking=True,
    )
    pq_dict['training_table_job'] = training_table_fut.job_id

    # Non-blocking training:
    trainer = kumo.Trainer(model_plan=model_plan)
    training_fut = trainer.fit(
        graph=graph,
        train_table=training_table_fut,
        non_blocking=True,
    )
    pq_dict['training_job'] = training_fut.job_id

    # Remove PQ from query_dict; it is not serializable:
    del pq_dict['pq']

with open('jobs.json', 'w') as fp:
    json.dump(query_dict, fp, sort_keys=True, indent=4)
