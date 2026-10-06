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
data_root = dataset_root(args, "adversarial_schema_storage_matrix")
c = storage_connector(data_root)
p = "e2e_invalid_data_type_struct_feature"

tables = {
    f"{p}_type_matrix_expected_failure": kumo.Table(
        c.table("type_matrix_expected_failure"), primary_key="id"),
}
graph = kumo.Graph(tables=tables, edges=[])
query = (
    "PREDICT {p}_type_matrix_expected_failure.struct_col "
    "FOR EACH {p}_type_matrix_expected_failure.id"
).format(p=p)
scenario_name = "invalid_data_type_struct_feature"

graph.infer_metadata()
graph[f"{p}_type_matrix_expected_failure"].add_column(
    "struct_col", stype="categorical", dtype="string")
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
training_job = trainer.fit(
    graph=graph,
    pquery=pquery,
    non_blocking=False,
    custom_tags={"sdk_harness": "on_demand_e2e", "scenario": scenario_name},
)
print(f"[sdk-script] training job id: {training_job.id}", flush=True)
