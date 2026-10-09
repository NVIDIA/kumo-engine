# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# A very bare-bones script to convert a fixture (.json) to an SDK script.

import argparse
import json
from typing import Dict, List

from io_utils import fsutil

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=False)
parser.add_argument("--dir", type=str, default=False)
args = parser.parse_args()


def convert_a_file(path: str):
    data = json.loads(fsutil.read_text(path))

    # The script:
    out = "import kumoai_studio as kumo\n\n"

    # Connector:
    connectors = data["connectors"]
    for _, connector in enumerate(connectors):
        c_config = connector["config"]
        c_type = c_config["type"]
        c_name = c_config["name"]
        if c_type == "file":
            root_dir = c_config["root_dir"]
            c_repr = f"{c_name} = kumo.S3Connector(root_dir=\"{root_dir}\")"
        elif c_type == "snowflake":
            c_repr = (f"{c_name} = kumo.SnowflakeConnector(\n"
                      f"\tname=\"{c_name}\",\n"
                      f"\taccount=\"{c_config['account']}\",\n"
                      f"\twarehouse=\"{c_config['warehouse']}\",\n"
                      f"\tdatabase=\"{c_config['database']}\",\n"
                      f"\tschema_name=\"{c_config['schema_name']}\")")
        elif c_type == "databricks":
            c_repr = (f"{c_name} = kumo.DatabricksConnector(\n"
                      f"\tname=\"{c_name}\",\n"
                      f"\thost=\"{c_config['host']}\",\n"
                      f"\tcluster_id=\"{c_config['cluster_id']}\",\n"
                      f"\twarehouse_id=\"{c_config['warehouse_id']}\",\n"
                      f"\tcatalog=\"{c_config['catalog']}\",\n"
                      f"\tschema_name=\"{c_config['schema_name']}\")")
        else:
            c_repr = (f"{c_name} = ...  # Repr: {connector}\n")
            print(f"Left connector {connector} underspecified.")

        # Name to object:
        out += f"{c_repr}\n"

    out += "\n"

    # Tables:
    tables = data["tables"]
    pkey_to_table_map: Dict[str, List[str]] = {}  # For column groups.
    for _, table in enumerate(tables):
        name = table["table_name"]
        source_name = table["source_table_name"]

        t_repr = ""
        source_table_repr = f"{table['connector_id']}[\"{source_name}\"]"

        pkey = f"\"{table['pkey']}\"" if table['pkey'] is not None else None
        if pkey is not None:
            if table['pkey'] not in pkey_to_table_map:
                pkey_to_table_map[table['pkey']] = []
            pkey_to_table_map[table['pkey']].append(name)

        tc = f"\"{table['time_col']}\"" if table[
            'time_col'] is not None else None
        etc = f"\"{table['end_time_col']}\"" if table[
            'end_time_col'] is not None else None
        t_repr += (f"{name} = kumo.Table(\n "
                   f"\tsource_table={source_table_repr},\n"
                   f"\tprimary_key={pkey},\n"
                   f"\ttime_column={tc},\n"
                   f"\tend_time_column={etc},\n"
                   f"\tcolumns=[\n")

        for col in table["cols"]:
            t_repr += (
                f"\t\tdict(name=\"{col['name']}\", stype=\"{col['stype']}\", "
                f"dtype=\"{col['dtype']}\"),\n")
        t_repr += "\t],\n"
        t_repr += ")"
        out += f"{t_repr}\n\n"

    out += "\n"

    # Graph (currently only one per fixture):
    graph = data["graphs"][0]
    graph_name = f'{graph["name"]}_graph'
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
                    f"fkey=\"{pair['col_name']}\", dst_table=\"{dst_table}\"),\n"
                )
    g_repr += "\t],\n)\n"

    out += f"{g_repr}\n\n"

    # PQ (currently only one per fixture):
    pq = data["queries"][0]
    pq_name = f'{pq["name"]}_query'
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
                    f"\tnon_blocking=True,\n"
                    f")\n")
    out += trainer_repr
    job_repr = "training_job.attach()\n"
    out += job_repr

    output_file_path = f"{path.rsplit('.', maxsplit=1)[0]}_sdk.py"
    fsutil.write_text(output_file_path, out)
    print(f"Output SDK code {output_file_path}")


SUCCESSFUL_PATHS = []
ALL_FAILED_PATHS = []


def convert_a_dir(dir_path: str):
    paths = fsutil.listdir(dir_path)
    for path in paths:
        if path == dir_path:
            continue
        if path.endswith('.json'):
            try:
                convert_a_file(path)
                SUCCESSFUL_PATHS.append(path)
            except Exception as e:
                print(e)
                ALL_FAILED_PATHS.append(path)
        elif fsutil.isdir(path):
            convert_a_dir(path)


if args.path:
    convert_a_file(args.path)
else:
    convert_a_dir(args.dir)
    print(f"Succeed convert {len(SUCCESSFUL_PATHS)} paths:")
    print(f"Failed convert {len(ALL_FAILED_PATHS)} paths:")
    for path in ALL_FAILED_PATHS:
        print(path)
