# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example-local conversion from a trusted FM Arrow batch to public SDM inputs."""
import pyarrow as pa


def _compatible_buffers(table):
    for index, field in enumerate(table.schema):
        column = table.column(index)
        if column.num_chunks == 0:
            table = table.set_column(index, field, pa.array([], type=field.type))
        elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            if any(buffer is not None and not buffer.is_mutable
                   for chunk in column.chunks for buffer in chunk.buffers()):
                table = table.set_column(index, field, pa.concat_arrays(column.chunks))
    return table


def arrow_to_sdm(batch):
    """Return the one-shot KumoRelational arguments from one retained batch."""
    from sdm import RelatedTables, TableTensor

    payload = batch.as_arrow_request()
    if not {"context", "predict"} <= payload.keys():
        raise ValueError("SDM prediction needs both context and predict splits")
    schema = payload["schema"]
    target = payload["task"]["target"]["column_name"]
    target_stype = "numerical" if payload["task"]["kind"] == "regression" else "categorical"
    relationships, links = [], []
    for relation in schema["relationships"]:
        if relation.get("source_table", "instance_table") == "instance_table":
            links.append(dict(task_columns=relation["source_columns"],
                              table=relation["target_table"],
                              table_columns=relation["target_columns"]))
        else:
            relationships.append(dict(left_table=relation["source_table"],
                                      left_columns=relation["source_columns"],
                                      right_table=relation["target_table"],
                                      right_columns=relation["target_columns"]))
    instances, related = {}, {}
    for split in ("context", "predict"):
        tables = {}
        entries = {"instance_table": payload[split]["instance_table"],
                   **payload[split]["related_tables"]}
        for name, table in entries.items():
            spec = schema["instance_table"] if name == "instance_table" else schema["related_tables"][name]
            stypes = {}
            for column in table.column_names:
                field = spec["columns"][column]
                stype = field.get("stype")
                if name == "instance_table" and column == target:
                    stype = target_stype
                elif stype is None:
                    stype = {"bool": "categorical", "string": "categorical",
                             "timestamp[us]": "datetime", "date": "datetime"}.get(
                                 field["dtype"], "numerical")
                stypes[column] = {"timestamp": "datetime", "id": "id"}.get(
                    stype.lower(), stype.lower())
            tables[name] = TableTensor.from_arrow(
                _compatible_buffers(table), stypes=stypes, device="cpu")
        instances[split] = tables.pop("instance_table")
        related[split] = RelatedTables(
            tables=tables, relationships=relationships, task_links=links)
    if target in instances["predict"].stypes:
        raise ValueError("Prediction inputs must not contain the target")
    return dict(x_context=instances["context"].drop_columns(target),
                y_context=instances["context"][[target]],
                x_query=instances["predict"],
                related_context_tables=related["context"],
                related_query_tables=related["predict"])
