# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A demonstration of the ability to retrain an existing model on a new/clean
Python session, in the Kumo SDK. This script simply trains the model.

Requirements:
    * Modify `API_URL` and `API_KEY` to point to a live, appropriate instance
"""
import os
import os.path as osp

import kumoai_studio as kumo

API_URL = os.getenv("KUMO_API_URL", "http://localhost:10002")
API_KEY = os.environ["KUMO_API_KEY"]

kumo.init(url=API_URL, api_key=API_KEY, project=os.environ["KUMO_PROJECT"])

# Shared location to store model ID:
path = osp.join(osp.dirname(osp.realpath(__file__)), "training_job")

# Setup #######################################################################

connector = kumo.S3Connector(
    name="retraining_source",
    identity=os.environ["KUMO_S3_IDENTITY_ID"],
    root_dir="s3://kumo-public-datasets/customerltv_mini_integ_test/",
)

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
pquery = kumo.PredictiveQuery(
    graph=graph,
    query=("PREDICT MAX(transaction.Quantity, 0, 30)\n"
           "FOR EACH customer.CustomerID\n"
           "ASSUMING SUM(transaction.UnitPrice, 0, 7, days) > 15"),
)

# Train in Current Session ###################################################$

trainer = kumo.Trainer(pquery.suggest_model_plan())
job_fut = trainer.fit(
    graph,
    pquery.generate_training_table(non_blocking=True),
    non_blocking=True,
)

# We expect that users will save this job ID somewhere, so they can later load
# the job and proceed with the rest of the script:
job_id = job_fut.job_id
print(f"Training job with id={job_id} has been launched.")

with open(path, "w") as f:
    f.write(f"{job_id}")
