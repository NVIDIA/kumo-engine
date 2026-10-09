# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""H&M with weighted train table.

.. note::
    This example requires setting a valid API key and API URL.
    Further it requires s3fs and fastparquet installed.

This example shows how to define a weighted training table.

"""
import argparse
import os

from kumoapi.train import TrainingTableSpec

import kumoai_studio as kumo
from kumoai_studio.connector import S3Connector, SourceTable

parser = argparse.ArgumentParser()
parser.add_argument("--problem", type=str, default="ltv",
                    choices=["ltv", "rec"])
args = parser.parse_args()
problem = args.problem

API_URL = ""
API_KEY = "test:DISABLE"

# Initialize the Kumo SDK:
kumo.init(api_key=API_KEY, url=API_URL, project=os.environ["KUMO_PROJECT"])

# Create a connector to a backing data store:
root_dir = "s3://kumo-public-datasets/hnm_integ_test/"
connector = kumo.S3Connector(
    name="weighted_training_source",
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

transactions_src = connector.table("transactions")
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

# Create predictive query:
if problem == "ltv":
    pquery = kumo.PredictiveQuery(
        graph=graph,
        query="""
            PREDICT SUM(transactions.price, 0, 30, days)
            FOR EACH customers.customer_id
        """,
    )
elif problem == "rec":
    pquery = kumo.PredictiveQuery(
        graph=graph,
        query="""
            PREDICT LIST_DISTINCT(transactions.article_id, 0, 30, days)
            RANK TOP 10
            FOR EACH customers.customer_id
        """,
    )
else:
    raise ValueError(f"Unknown problem: {problem}")

# Pinned rather than inferred: the weighted table below is supplied
# rather than generated, so it names the plan it splits under itself.
tt_plan = pquery.suggest_training_table_plan()
train_table = pquery.generate_training_table(tt_plan)

# Add weight column to train table by loading it as a pandas df
# and, persist it. We persist to s3 here.
# Alternatively the train table can be exported to snowflake/db/bigquery
# via `train_table.export()` and then the weight column can be added
# to the exported table.
weight_table_name = None
if problem == "ltv":
    df = train_table.data_df()
    df['weight'] = 1
    df.loc[df['TARGET'] > df['TARGET'].mean(), 'weight'] = 2
    df.to_parquet(
        "s3://kumo-public-datasets/weighted_tt/hnm_ltv_weighted.parquet")
    weight_table_name = "hnm_ltv_weighted"
else:
    df = train_table.data_df()
    df['weight'] = 1
    df.loc[df.index % 2 == 0, 'weight'] = 2
    df.to_parquet(
        "s3://kumo-public-datasets/weighted_tt/hnm_rec_weighted.parquet")
    weight_table_name = "hnm_rec_weighted"

# Name the weighted table as the one to train on. It is a training table in
# its own right rather than an edit of the generated one, so it carries the
# plan it splits under.
table_mod_spec = TrainingTableSpec(weight_col='weight')
train_table = kumo.TrainingTable.from_source_table(
    SourceTable(
        name=weight_table_name, connector=S3Connector(
            name="weighted_training_table",
            identity=os.environ["KUMO_S3_IDENTITY_ID"],
            root_dir="s3://kumo-public-datasets/weighted_tt/",
        )),
    table_mod_spec,
    pq=pquery,
    plan=tt_plan,
)

# Train with weighted train table
model_plan = pquery.suggest_model_plan(train_table_spec=table_mod_spec)
model_plan.training_job.num_experiments = 1
print("suggested weight mode", model_plan.optimization.weight_mode)

trainer = kumo.Trainer(model_plan)
training_job = trainer.fit(
    graph=graph,
    train_table=train_table,
    non_blocking=False,
)
print(f"Training metrics: {training_job.metrics()}")
