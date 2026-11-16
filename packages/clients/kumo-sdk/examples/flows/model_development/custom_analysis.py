# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Showcases the ability to perform custom analysis and use your custom code
(for visualization), etc. with Kumo output in a single interactive session.
"""

import json
import os

# Use pre-processing and post-processing for model development and evaluation
# Due to the large-scale LP, local Pandas won't work
import kumoai_studio as kumo

API_URL = os.getenv("KUMO_API_URL", "http://localhost:10002")
API_KEY = os.environ["KUMO_API_KEY"]
kumo.init(api_key=API_KEY, url=API_URL, project=os.environ["KUMO_PROJECT"])

# Initial setup. Original data:
#   - users: (user_id, location_id, created_time)
#   - orders: (user_id, location_id, store_id, order_time)
#   - stores: (store_id, store_location_id, created_time)
path = "s3://..."
connector = kumo.S3Connector(
    name="custom_analysis_source",
    identity=os.environ["KUMO_S3_IDENTITY_ID"],
    root_dir=path,
)

users = kumo.Table.from_source_table(source_table=connector['users'],
                                     primary_key='user_id',
                                     time_column='created_time')

raw_orders = kumo.Table.from_source_table(source_table=connector['orders'],
                                          time_column='order_time')

stores = kumo.Table.from_source_table(source_table=connector['stores'],
                                      primary_key='store_id',
                                      time_column='created_time')


# Preprocessing:
#   - user_locations: add CONCAT(user_id, location_id) as pk
#   - orders: add CONCAT(user_id, location_id) as fk
def APPLY_SQL(sql, path):
    # TODO(manan): translate out-of-band. Assume this function writes to
    # output path `path`:
    return ""


APPLY_SQL(sql="SELECT *, CONCAT(user_id, location_id) AS pk FROM users",
          path="user_locations.parquet")
user_locations = kumo.Table.from_source_table(
    source_table=connector['user_locations'])

APPLY_SQL(sql="SELECT *, CONCAT(user_id, location_id) AS fk FROM orders",
          path="orders.parquet")
orders = kumo.Table.from_source_table(source_table=connector['orders'])

# Graph:
#   user_locations -> users -> orders -> stores
#   user_locations -> orders -> stores
graph = kumo.Graph(
    tables={
        'users': users,
        'stores': stores,
        'orders': orders,
        'user_locations': user_locations,
    })
graph.link('orders', 'fk', 'user_locations')
graph.link('user_locations', 'user_id', 'users')
graph.link('orders', 'user_id', 'users')
graph.link('orders', 'store_id', 'stores')

# PQ: a vanilla LP
pquery = kumo.PredictiveQuery(
    graph=graph,
    query="""
        PREDICT LIST_DISTINCT(orders.store_id, 0, 7, DAYS)
        RANK TOP 100
        FOR EACH user_locations.pk
    """,
)

train_table_fut = pquery.generate_training_table(non_blocking=True)
trainer = kumo.Trainer(pquery.suggest_model_plan())

# Wait for job to complete:
job = trainer.fit(graph=graph, train_table=train_table_fut, non_blocking=False)

# Compute metrics:
metrics = job.metrics()
print(f"Metrics are: {metrics}")

# Save experiment information under a particular path with experiment name
# reference:
experiment_name = 'experiment12'
experiment_s3_path = f's3://my-bucket/blah/{experiment_name}/'

# Export experiment info:
with open(f'{experiment_s3_path}/experiment_info.json', 'w') as fp:
    json.dump(dict(trainer_id=job.id), fp, sort_keys=True, indent=4)

# Run post-processing (TSN filtering) to obtain the TSN outcome
test_df = job.holdout_df()

# Postprocess DF and compute MAP@10 with following query:
"""
SELECT pk, store_id, score
FROM predictions
LEFT JOIN (SELECT pk, store_id FROM orders WHERE order_time < HOLD_OUT_DATE)
    past_orders
ON predictions.pk = past_orders.pk AND
    predictions.store_id = past_orders.store_id
"""
