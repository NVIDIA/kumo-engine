# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""H&M (Filtered) Churn on Kumo SDK.

Requirements:
    * Modify `API_URL` and `API_KEY` to point to a live, appropriate instance
"""
import os

import kumoai_studio as kumo

API_URL = "http://localhost:10002"
API_KEY = "test:DISABLED"

# Initialize the Kumo SDK:
kumo.init(api_key=API_KEY, url=API_URL, project=os.environ["KUMO_PROJECT"])

# Create a connector to a backing data store:
root_dir = "s3://kumo-public-datasets/filtered_hnm_temp/"
connector = kumo.S3Connector(
    name="hm_churn_source",
    identity=os.environ["KUMO_S3_IDENTITY_ID"],
    root_dir=root_dir,
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

# Define graph:
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

# Snapshot graph:
graph.snapshot(force_refresh=True, non_blocking=False)

# Get the table staistics from a snapshot:
print(f"Table stats: {graph.get_table_stats('minimal')}")

# Create predictive query:
pquery = kumo.PredictiveQuery(
    graph=graph,
    query="""
        PREDICT COUNT(transactions.price, 0, 90, days) > 0
        FOR EACH customers.customer_id
        WHERE SUM(transactions.price, -60, 0, days) > 0.05
    """,
)

# Generate training table (blocking is default):
train_table = pquery.generate_training_table()

# Train!
model_plan = pquery.suggest_model_plan()
model_plan.training_job.num_experiments = 1

trainer = kumo.Trainer(model_plan)
training_job = trainer.fit(
    graph=graph,
    train_table=train_table,
    non_blocking=False,
)
print(f"Training metrics: {training_job.metrics()}")

# Generate prediction table (non-blocking specified here):
pred_table = pquery.generate_prediction_table(non_blocking=True)

# Predict!
prediction_job = trainer.predict(
    graph=graph,
    prediction_table=pred_table,
    non_blocking=False,
)
