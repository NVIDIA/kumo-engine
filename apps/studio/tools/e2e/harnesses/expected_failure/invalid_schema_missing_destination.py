#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import kumoai_studio as kumo

from tools.e2e.harnesses.sdk_script_support import (
    dataset_root,
    parse_args,
    storage_connector,
)

args = parse_args()
print(f"[sdk-script] initializing SDK {args.url}", flush=True)
kumo.init(url=args.url, api_key=args.api_key)
data_root = dataset_root(args, "marketplace_temporal_link_graph")
c = storage_connector(data_root)
p = "e2e_invalid_schema_missing_destination"

tables = {
    f"{p}_users": kumo.Table(
        c.table("users"), primary_key="user_id", time_column="created_ts"),
    f"{p}_items": kumo.Table(
        c.table("items"),
        primary_key="item_id",
        time_column="available_ts",
        end_time_column="end_ts",
    ),
    f"{p}_interactions": kumo.Table(
        c.table("interactions"),
        primary_key="interaction_id",
        time_column="interaction_ts",
    ),
    f"{p}_prediction_pairs": kumo.Table(
        c.table("prediction_pairs"),
        primary_key="pair_id",
        time_column="prediction_ts",
    ),
}
graph = kumo.Graph(
    tables=tables,
    edges=[
        kumo.Edge(f"{p}_interactions", "user_id", f"{p}_users"),
        kumo.Edge(f"{p}_interactions", "item_id", "missing_items"),
        kumo.Edge(f"{p}_prediction_pairs", "user_id", f"{p}_users"),
        kumo.Edge(f"{p}_prediction_pairs", "item_id", f"{p}_items"),
    ],
)
exclude_cols = None
query = 'PREDICT LIST_DISTINCT({p}_interactions.item_id, 0, 90) RANK TOP 10 FOR EACH {p}_users.user_id'.format(p=p)
scenario_name = 'invalid_schema_missing_destination'

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
    training_job = trainer.fit(
        graph=graph,
        pquery=pquery,
        non_blocking=False,
        custom_tags={"sdk_harness": "on_demand_e2e", "scenario": scenario_name},
    )
    training_job_id = training_job.id
print(f"[sdk-script] training job id: {training_job_id}", flush=True)
