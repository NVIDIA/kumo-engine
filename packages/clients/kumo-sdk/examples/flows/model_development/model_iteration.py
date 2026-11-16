# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A demonstration of enabling quick iterations with minimal code to test
different parameter settings.

Requirements:
    * Modify `API_URL` and `API_KEY` to point to a live, appropriate instance
"""
import os

import kumoai_studio as kumo

# Initialize:
API_URL = os.getenv("KUMO_API_URL", "http://localhost:10002")
API_KEY = os.environ["KUMO_API_KEY"]
kumo.init(url=API_URL, api_key=API_KEY, project=os.environ["KUMO_PROJECT"])

# Source Data:
connector = kumo.S3Connector(
    name="model_iteration_source",
    identity=os.environ["KUMO_S3_IDENTITY_ID"],
    root_dir="s3://kumo-public-datasets/customerltv_mini_integ_test/",
)

# Connect Tables and Graph:
customer = kumo.Table.from_source_table(
    source_table=connector["customer"],
    primary_key="CustomerID",
).infer_metadata()

stock = kumo.Table.from_source_table(
    source_table=connector["stock"],
    primary_key="StockCode",
).infer_metadata()

transaction = kumo.Table.from_source_table(
    source_table=connector["transaction_altered"],
    time_column="InvoiceDate",
).infer_metadata()

graph = kumo.Graph(
    tables={
        "customer": customer,
        "stock": stock,
        "transaction": transaction,
    },
    edges=[
        dict(src_table="transaction", fkey="StockCode", dst_table="stock"),
        dict(src_table="transaction", fkey="CustomerID", dst_table="customer"),
    ],
)

# Predictive Query:
pquery = kumo.PredictiveQuery(
    graph=graph,
    query=("PREDICT MAX(transaction.Quantity, 0, 30)\n"
           "FOR EACH customer.CustomerID\n"
           "ASSUMING SUM(transaction.UnitPrice, 0, 7, days) > 15"),
)
training_table_plan = pquery.suggest_training_table_plan()
training_table_plan.split = "RandomSplit([0.8. 0.1, 0.1])"
training_table_fut = pquery.generate_training_table(
    plan=training_table_plan,
    non_blocking=True,
)

# To showcase the ability of the Kumo SDK, we now launch 10 jobs. Each job
# will manually configure its model plan and training table plan from the base
# model plans defined above. Note that all jobs beyond the configured parallel
# job limit are queued, and will have status "QUEUED" in the backend. Further,
# all jobs share the _same training table_:
job_futs = []
for i in range(10):
    # Defaults:
    model_plan = pquery.suggest_model_plan()

    # Users can easily modify parameters and rerun experiments. All
    # hyperparameter changes are applied efficiently:
    model_plan.neighbor_sampling.num_neighbors = [[24, 48]]

    # Create Trainer, generate Training Table, train model:
    trainer = kumo.Trainer(model_plan)
    job_fut = trainer.fit(
        graph=graph,
        train_table=training_table_fut,
        non_blocking=True,
    )

    job_futs.append(job_fut)
    print(f"Started job {job_fut.id}")

# Users can check the non-blocking status of a job with no setup, in one of
# three ways:
#   1) Use the UI jobs page: https://<customer-id>.kumoai.cloud/jobs
#   2) Use the future objects directly (as below)
#   3) Construct future objects from job IDs, with the following code
#       snippet:
#
#       job_fut = kumo.TrainingJob(job_id="<job_id>")
#
# Simple polling, can be wrapped in a loop:
for fut in job_futs:
    # Iteration status is logged in `status()`, and all results are shown in
    # the jobs UI page:
    print(f"Job {fut.id} has status {fut.status()}")

# Users can use the same graph and some shared model planner options
# over multiple models on different sessions. Users can rerun a
# failed job or a new training job with slight hyperparameter change on a
# new session without copying/rerunning most chunk of code. Users can start
# or pick up an experiment in a new session if connection is lost.
#
# To show these, we will copy the first job's graph and model planner options,
# loading as if from scratch, and then launch a new training job. We can
# always view the status of a job in a new session too (as above).
job_id = job_futs[0].id
new_trainer = kumo.Trainer.load(job_id)
new_query = kumo.PredictiveQuery.load(job_id)

new_model_plan = new_trainer.model_plan
assert new_model_plan is not None
new_model_plan.neighbor_sampling.num_neighbors = [[8, 12]]
# TODO(manan): support picking up training table generation options

new_trainer.fit(
    graph=new_query.graph,
    train_table=new_query.generate_training_table(non_blocking=True),
    non_blocking=False,
)

# Criteria not supported:
# - Users can start from intermediate stages (e.g. materialized task)
# - Users can resume experiments from a saved state.
