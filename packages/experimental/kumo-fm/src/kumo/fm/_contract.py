# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FM schema and task metadata shared by direct and wire output."""

def schema_for(workload):
    bundle = workload.bundle
    id_spec = {"dtype": "int64", "stype": "ID", "nullable": False}
    entity = bundle.tables[workload.entity]
    key_spec = {"dtype": entity["columns"][entity["primary_key"]]["dtype"], "stype": "ID", "nullable": False}
    instance = {"primary_key": "instance_id", "columns": {
        "instance_id": id_spec, "__fm_entity_key": key_spec,
        "__fm_anchor": {"dtype": "timestamp[us]", "stype": "timestamp", "nullable": False},
        "__fm_target": {"dtype": workload.target["dtype"] if workload.kind == "regression" else "string", "nullable": False}}}
    related = {}
    for name in sorted(workload.table_names):
        table = bundle.tables[name]
        pk = table["primary_key"]
        columns = {"instance_id": id_spec,
                   pk: {"dtype": table["columns"][pk]["dtype"], "stype": "ID", "nullable": False}}
        fks = {r["foreign_key"] for r in bundle.relations.values() if r["child"] == name}
        for col in workload.projections[name]:
            spec = dict(table["columns"][col])
            if spec["dtype"] == "date":
                spec["dtype"] = "timestamp[us]"
            if col == pk or col in fks:
                spec["stype"] = "ID"
            columns[col] = spec
        related[name] = {"primary_key": ["instance_id", pk], "columns": columns}
    relationships = [{"source_columns": ["instance_id", "__fm_entity_key"],
                      "target_table": workload.entity,
                      "target_columns": ["instance_id", entity["primary_key"]]}]
    for name in sorted(workload.relation_names):
        rel = bundle.relations[name]
        parent = bundle.tables[rel["parent"]]
        dtype = parent["columns"][parent["primary_key"]]["dtype"]
        col = "__fm_edge_" + name
        related[rel["child"]]["columns"][col] = {"dtype": dtype, "stype": "ID", "nullable": True}
        relationships.append({"source_table": rel["child"], "source_columns": ["instance_id", col],
                              "target_table": rel["parent"], "target_columns": ["instance_id", parent["primary_key"]]})
    return {"instance_table": instance, "related_tables": related, "relationships": relationships}



def metadata_for(workload):
    target = dict(workload.target)
    if workload.kind != "regression":
        target.update(dtype="string", classes=[str(value) for value in target["classes"]])
    fields = ["prediction"] if workload.kind == "regression" else ["prediction", "probabilities"]
    return {
        "model": "kumo-relational",
        "task": {
            "kind": workload.kind,
            "entity_table_names": [workload.entity],
            "anchor_time_column": "__fm_anchor",
            "target": {"table_name": "instance_table", "column_name": "__fm_target", **target},
        },
        "schema": schema_for(workload),
        "output": {"fields": fields, "score_format": "object"},
    }
