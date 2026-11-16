# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# A very bare-bones script to convert a fixture (.json) to SDK script. Kumo
# views must be filled in by the user based on the provided transformations.

import argparse
import json
from typing import Dict, List

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=False)
args = parser.parse_args()

with open(args.path, "r") as f:
    data = json.load(f)

# The script:
out = "import kumoai_studio as kumo\n\n"

# Connector:
connectors = data["connectors"]
# Warehouse connector names, for custom-SQL (`Table.from_sql`) tables:
warehouse_connectors: List[str] = []
for i, connector in enumerate(connectors):
    c_config = connector["config"]
    c_type = c_config["type"]
    c_name = c_config["name"]
    identity_id = connector.get("identity_id")
    if c_type in {"snowflake", "databricks", "bigquery"}:
        warehouse_connectors.append(c_name)
    if c_type == "file":
        root_dir = c_config["root_dir"]
        c_repr = (f"{c_name} = kumo.S3Connector(\n"
                  f"\tname=\"{c_name}\",\n"
                  f"\tidentity=\"{identity_id}\",\n"
                  f"\troot_dir=\"{root_dir}\")")
    elif c_type == "snowflake":
        c_repr = (f"{c_name} = kumo.SnowflakeConnector(\n"
                  f"\tname=\"{c_name}\",\n"
                  f"\tidentity=\"{identity_id}\",\n"
                  f"\twarehouse=\"{c_config['warehouse']}\",\n"
                  f"\tdatabase=\"{c_config['database']}\",\n"
                  f"\tschema_name=\"{c_config['schema_name']}\")")
    elif c_type == "databricks":
        c_repr = (f"{c_name} = kumo.DatabricksConnector(\n"
                  f"\tname=\"{c_name}\",\n"
                  f"\tidentity=\"{identity_id}\",\n"
                  f"\tcluster_id=\"{c_config['cluster_id']}\",\n"
                  f"\twarehouse_id=\"{c_config['warehouse_id']}\",\n"
                  f"\tcatalog=\"{c_config['catalog']}\",\n"
                  f"\tschema_name=\"{c_config['schema_name']}\")")
    elif c_type == "bigquery":
        c_repr = (f"{c_name} = kumo.BigQueryConnector(\n"
                  f"\tname=\"{c_name}\",\n"
                  f"\tidentity=\"{identity_id}\",\n"
                  f"\tproject_id=\"{c_config['project_id']}\",\n"
                  f"\tdataset_id=\"{c_config['dataset_id']}\",\n"
                  f"\tlocation=\"{c_config['location']}\")")
    else:
        c_repr = (f"{c_name} = ...  # Repr: {connector}\n")
        print(f"Left connector {connector} underspecified.")

    # Name to object:
    out += f"{c_repr}\n"

out += "\n"

# Tables:
tables = data["tables"]
pkey_to_table_map: Dict[str, List[str]] = {}  # For column groups.
# Map a base table's name to its connector variable, so a legacy view can be
# rewritten against the connector its referenced tables actually live in:
connector_by_table: Dict[str, str] = {
    t["table_name"]: t["connector_id"]
    for t in tables if t["connector_id"] != "kumo_views_connector"
}
for i, table in enumerate(tables):
    name = table["table_name"]
    source_name = table["source_table_name"]

    if table["connector_id"] in {
            "parquet_upload_connector", "csv_upload_connector"
    }:
        raise RuntimeError(
            f"Cannot parse fixture with table {table} due to its connector.")

    pkey = f"\"{table['pkey']}\"" if table['pkey'] is not None else None
    if pkey is not None:
        if table['pkey'] not in pkey_to_table_map:
            pkey_to_table_map[table['pkey']] = []
        pkey_to_table_map[table['pkey']].append(name)

    tc = f"\"{table['time_col']}\"" if table['time_col'] is not None else None
    etc = f"\"{table['end_time_col']}\"" if table[
        'end_time_col'] is not None else None

    cols_repr = ""
    for col in table["cols"]:
        cols_repr += (
            f"\t\tdict(name=\"{col['name']}\", stype=\"{col['stype']}\", "
            f"dtype=\"{col['dtype']}\"),\n")

    if table["connector_id"] == "kumo_views_connector":
        # Legacy Kumo Views are now custom-SQL tables: the SELECT runs in the
        # warehouse and only its result set is ingested. `connector` must be
        # the single warehouse the query reads from. Prefer resolving it from
        # the view's referenced base tables; fall back to the sole warehouse
        # connector, else require a manual choice.
        base_ids = table["view_args"].get("base_table_ids", {})
        base_connectors = {
            connector_by_table[v.rsplit(".", maxsplit=1)[-1]]
            for v in base_ids.values()
            if v.rsplit(".", maxsplit=1)[-1] in connector_by_table
        }
        # Custom SQL only runs in a warehouse, so ignore any file/blob base
        # connectors when resolving where the query should execute:
        base_connectors &= set(warehouse_connectors)
        if len(base_connectors) == 1:
            connector_repr = next(iter(base_connectors))
        elif len(warehouse_connectors) == 1:
            connector_repr = warehouse_connectors[0]
        else:
            candidates = base_connectors or set(warehouse_connectors)
            connector_repr = (
                "...  # TODO: pick the warehouse connector for this SQL "
                f"(candidates: {sorted(candidates)})")
        t_repr = (f"{name} = kumo.Table.from_sql(\n"
                  f"\tconnector={connector_repr},\n"
                  f"\tsql=\"\"\"{table['view_args']['sql']}\"\"\",\n"
                  f"\tprimary_key={pkey},\n"
                  f"\ttime_column={tc},\n"
                  f"\tend_time_column={etc},\n"
                  f"\tcolumns=[\n{cols_repr}\t],\n)")
    else:
        source_table_repr = f"{table['connector_id']}[\"{source_name}\"]"
        t_repr = (f"{name} = kumo.Table(\n "
                  f"\tsource_table={source_table_repr},\n"
                  f"\tprimary_key={pkey},\n"
                  f"\ttime_column={tc},\n"
                  f"\tend_time_column={etc},\n"
                  f"\tcolumns=[\n{cols_repr}\t],\n)")
    out += f"{t_repr}\n\n"

out += "\n"

# Graph (currently only one per fixture):
graph = data["graphs"][0]
graph_name = graph["name"]
g_repr = (f"{graph_name} = kumo.Graph(\n"
          "\ttables={\n")
for table in graph["table_ids"]:
    # FQN...
    table_name = table.rsplit(".", maxsplit=1)[-1]
    g_repr += f"\t\t\"{table_name}\": {table_name},\n"
g_repr += "\t}, edges=[\n"

for col_group in graph["col_groups"]:
    col_group = col_group["cols"]
    dst_table = None
    # Find pkey:
    for pair in col_group:
        # FQN...
        table_name = pair["table_id"].rsplit(".", maxsplit=1)[-1]
        if table_name in pkey_to_table_map.get(pair["col_name"], []):
            # This is our pkey, all other edges link to it
            dst_table = table_name
    if dst_table is None:
        raise RuntimeError(f"Error in parsing column group {col_group}")
    else:
        for pair in col_group:
            table_name = pair["table_id"].rsplit(".", maxsplit=1)[-1]
            if table_name == dst_table:
                continue
            g_repr += (
                f"\t\tdict(src_table=\"{table_name}\", "
                f"fkey=\"{pair['col_name']}\", dst_table=\"{dst_table}\"),\n")
g_repr += "\t],\n)\n"

out += f"{g_repr}\n\n"

# PQ (currently only one per fixture):
pq = data["queries"][0]
pq_name = pq["name"]
pq_repr = (f"{pq_name} = kumo.PredictiveQuery(\n"
           f"\tgraph={graph_name},\n"
           f"\tquery=\"\"\"{pq['query_yaml']}\"\"\",\n"
           f")")

out += f"{pq_repr}\n\n"

# Training table:
tt_repr = (
    f"train_table = {pq_name}.generate_training_table(non_blocking=True)\n")
out += tt_repr

# Trainer:
mp_repr = f"model_plan = {pq_name}.suggest_model_plan()\n\n"
out += mp_repr

trainer_repr = (f"trainer = kumo.Trainer(model_plan)\n"
                f"training_job = trainer.fit(\n"
                f"\tgraph={graph_name},\n"
                f"\ttrain_table=train_table,\n"
                f"\tnon_blocking=False,\n"
                f")\n")
out += trainer_repr

job_repr = "training_job.attach()\n"
out += job_repr

with open(f"{args.path.rsplit('.', maxsplit=1)[0]}_sdk.py", "w") as f:
    f.write(out)
