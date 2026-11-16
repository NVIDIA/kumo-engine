#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re

import kumoai_studio as kumo
from kumoai_studio.artifact_export import OutputConfig
from kumoapi.jobs import MetadataField

from tools.e2e.harnesses.sdk_script_support import (
    dataset_root,
    join_uri,
    output_exists,
    parse_args,
    storage_connector,
)

args = parse_args()
print(f"[sdk-script] initializing SDK {args.url}", flush=True)
kumo.init(url=args.url, api_key=args.api_key)
data_root = dataset_root(args, "omnichannel_customer_graph")
c = storage_connector(data_root)
output_connector = None if args.local_output else storage_connector(args.output_root)
p = "e2e_customer_churn_binary_implicit"
name_suffix = os.getenv("KUMO_E2E_NAME_SUFFIX", "").strip()
if name_suffix:
    p = f"{p}_{re.sub(r'[^A-Za-z0-9_]', '_', name_suffix)}"

tables = {
    f"{p}_customers": kumo.Table(
        c.table("customers"), primary_key="customer_id", time_column="signup_ts"),
    f"{p}_products": kumo.Table(c.table("products"), primary_key="product_id"),
    f"{p}_orders": kumo.Table(
        c.table("orders"), primary_key="line_item_id", time_column="order_ts"),
    f"{p}_customer_labels": kumo.Table(
        c.table("customer_labels"),
        primary_key="label_row_id",
        time_column="label_ts",
    ),
    f"{p}_prediction_customers": kumo.Table(
        c.table("prediction_customers"),
        primary_key="prediction_row_id",
        time_column="prediction_ts",
    ),
}
graph = kumo.Graph(
    tables=tables,
    edges=[
        kumo.Edge(f"{p}_orders", "customer_id", f"{p}_customers"),
        kumo.Edge(f"{p}_orders", "product_id", f"{p}_products"),
        kumo.Edge(f"{p}_customer_labels", "customer_id", f"{p}_customers"),
        kumo.Edge(f"{p}_prediction_customers", "customer_id", f"{p}_customers"),
    ],
)
exclude_cols = {f"{p}_customer_labels": {"next_categories"}}
query = 'PREDICT {p}_customer_labels.churned FOR EACH {p}_customer_labels.label_row_id'.format(p=p)
scenario_name = 'customer_churn_binary_implicit'

graph.infer_metadata()
if exclude_cols:
    for table_name, column_names in exclude_cols.items():
        for column_name in column_names:
            if graph[table_name].has_column(column_name):
                graph[table_name].remove_column(column_name)
for table_name, table in tables.items():
    print(f"[sdk-script] saving table {table_name}", flush=True)
    table.save(table_name)

pquery = kumo.PredictiveQuery(graph=graph, query=query)
graph.validate()
graph.save(f"{p}_graph")
pquery.save(f"{p}_pquery")
if args.stop_after == "save":
    raise SystemExit(0)

model_plan = pquery.suggest_model_plan()
if args.stop_after == "validate":
    raise SystemExit(0)

trainer = kumo.Trainer(model_plan)
if args.training_job_id:
    training_job_id = args.training_job_id
else:
    train_table = pquery.generate_training_table(
        custom_tags={
            "sdk_harness": "on_demand_e2e",
            "scenario": scenario_name,
            "stage": "train_table",
        })
    training_job = trainer.fit(
        graph=graph,
        train_table=train_table,
        non_blocking=False,
        custom_tags={"sdk_harness": "on_demand_e2e", "scenario": scenario_name},
    )
    training_job_id = training_job.id
print(f"[sdk-script] training job id: {training_job_id}", flush=True)
if args.stop_after == "train":
    raise SystemExit(0)

prediction_table = pquery.generate_prediction_table(
    custom_tags={
        "sdk_harness": "on_demand_e2e",
        "scenario": scenario_name,
        "stage": "prediction_table",
    })
output_table = f"{p}_bp"
output_config = OutputConfig(
    output_types={"predictions", "embeddings"},
    output_connector=output_connector,
    output_table_name=None if args.local_output else output_table,
    output_metadata_fields=None if args.local_output else [MetadataField.JOB_TIMESTAMP],
)
prediction_job = trainer.predict(
    graph=graph,
    prediction_table=prediction_table,
    training_job_id=training_job_id,
    non_blocking=False,
    output_config=output_config,
    binary_classification_threshold=0.5,
    custom_tags={
        "sdk_harness": "on_demand_e2e",
        "scenario": scenario_name,
        "stage": "batch_prediction",
    },
)
print(f"[sdk-script] batch prediction job id: {prediction_job.id}", flush=True)
if args.local_output:
    summary = prediction_job.summary()
    print(
        f"[sdk-script] local output num_entities_predicted: {summary.num_entities_predicted}",
        flush=True,
    )
    if summary.num_entities_predicted <= 0:
        raise RuntimeError("Batch prediction completed without predicted entities")
    raise SystemExit(0)
if not output_exists(args.output_root, output_table):
    raise RuntimeError(
        f"Batch prediction output not found under {join_uri(args.output_root, output_table)}_*")
