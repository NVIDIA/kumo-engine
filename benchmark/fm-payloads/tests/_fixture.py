# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny whole-second contract bundles, not a bulk generator.

Timestamp values have one-second granularity. Parquet and FM serialization
retain timestamp[us] as the physical type, with zero subsecond components.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from kumo_fm_benchmark.bundle import TYPES, Bundle, digest, identity


def scalar(value):
    if isinstance(value, datetime):
        return value.isoformat().replace('+00:00', 'Z')
    return value


CUTOFF = datetime(2026, 2, 1, tzinfo=timezone.utc)
DAY_SECONDS = 86400
DAY_US = DAY_SECONDS * 1000000  # Workload duration fields retain their schema units.
USER_A = 9007199254740993
USER_B = -5


def column(dtype, nullable=False, stype=None):
    spec = {"dtype": dtype, "nullable": nullable}
    if stype:
        spec["stype"] = stype
    return spec


def table(pk, columns, time_column=None):
    return {"primary_key": pk, "columns": columns, "time_column": time_column,
            **({"availability": "timeless"} if time_column is None else {})}


def commerce():
    tables = {
        "users": table("user_id", {"user_id": column("int64"), "segment": column("string", stype="categorical"), "active": column("bool")}),
        "interactions": table("interaction_id", {"interaction_id": column("int64"), "user_id": column("int64", True),
                                                 "item_id": column("string", True), "event_time": column("timestamp[us]", True), "amount": column("float64", True, "numerical")}, "event_time"),
        "items": table("item_id", {"item_id": column("string"), "category_id": column("int64"), "description": column("string", stype="text")}),
        "categories": table("category_id", {"category_id": column("int64"), "name": column("string", stype="categorical")}),
    }
    relations = {"interaction_user": {"child": "interactions", "foreign_key": "user_id", "parent": "users"},
                 "interaction_item": {"child": "interactions", "foreign_key": "item_id", "parent": "items"},
                 "item_category": {"child": "items", "foreign_key": "category_id", "parent": "categories"}}
    events = [(11, USER_A, "shared", -1), (12, USER_A, "shared", -1),
              (13, USER_A, "other", -1), (21, USER_B, "shared", -2),
              (22, USER_B, "other", -10 * DAY_SECONDS), (23, USER_B, "other", -10 * DAY_SECONDS - 1),
              (30, USER_A, "other", 0), (31, USER_A, "shared", DAY_SECONDS),
              (40, USER_A, None, -3), (41, None, "shared", -1),
              (42, USER_B, "other", None), (43, USER_A, "shared", -40 * DAY_SECONDS - 1),
              (44, USER_A, "other", -20 * DAY_SECONDS)]
    rows = {
        "users": [{"user_id": USER_A, "segment": "a", "active": True},
                  {"user_id": USER_B, "segment": "b", "active": False},
                  {"user_id": 42, "segment": "quiet", "active": True}],
        "interactions": [{"interaction_id": k, "user_id": u, "item_id": item,
                          "event_time": None if dt is None else CUTOFF + timedelta(seconds=dt),
                          "amount": None if k == 22 else 5.0} for k, u, item, dt in events],
        "items": [{"item_id": "shared", "category_id": 7, "description": "shared item"},
                  {"item_id": "other", "category_id": 7, "description": "other item"},
                  {"item_id": "unused", "category_id": 8, "description": "unused"}],
        "categories": [{"category_id": 7, "name": "common"}, {"category_id": 8, "name": "unused"}],
    }
    layers = [[{"relation": "interaction_user", "direction": "reverse", "fanout": 2}],
              [{"relation": "interaction_item", "direction": "forward", "fanout": 1}],
              [{"relation": "item_category", "direction": "forward", "fanout": 1}]]
    return tables, relations, rows, "users", "interactions", "user_id", "event_time", layers


def marketo():
    facts = {"engagements": ("marketo_lead_id", "engagement_datetime"),
             "metrics": ("marketo_lead_id", "trigger_date"),
             "usecase_metrics": ("marketo_lead_id", "trigger_date"),
             "conversions": ("marketo_lead_id", "u2_stage_date"),
             "activities": ("id", "activity_date"),
             "notes": ("marketo_lead_id", "source_creation_timestamp")}
    tables = {"leads": table("Id", {"Id": column("int64"), "segment": column("string", stype="categorical")})}
    rows = {"leads": [{"Id": USER_A, "segment": "a"}, {"Id": USER_B, "segment": "b"}, {"Id": 42, "segment": "quiet"}]}
    relations, layer = {}, []
    for name, (fk, tc) in facts.items():
        tables[name] = table("event_id", {"event_id": column("int64"), fk: column("int64", True),
                                          tc: column("timestamp[us]", True), "value": column("float64", True, "numerical")}, tc)
        rows[name] = [] if name == "notes" else [
            {"event_id": k, fk: u, tc: CUTOFF + timedelta(seconds=dt), "value": None if k == 3 else 1.0}
            for k, u, dt in [(1, USER_A, -1), (2, USER_A, -1), (3, USER_A, -1),
                             (4, USER_B, -10 * DAY_SECONDS), (5, USER_B, 0), (6, None, -1),
                             (7, USER_A, -20 * DAY_SECONDS)]]
        relations[name + "_lead"] = {"child": name, "foreign_key": fk, "parent": "leads"}
        layer.append({"relation": name + "_lead", "direction": "reverse", "fanout": 2})
    return tables, relations, rows, "leads", "engagements", "marketo_lead_id", "engagement_datetime", [layer]


def write_bundle(root, tables, relations, rows, tasks, *, recipe, reverse_layout=False, row_group_size=3):
    """Small-fixture writer; imported/bulk bundles use the documented same format."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    inventory = {}
    logical_tables = {}
    for name, spec in tables.items():
        path = root / "tables" / name / "part-00000.parquet"
        path.parent.mkdir(parents=True)
        schema = pa.schema([pa.field(c, TYPES[s["dtype"]], s["nullable"]) for c, s in spec["columns"].items()])
        data = list(reversed(rows[name])) if reverse_layout else rows[name]
        pq.write_table(pa.Table.from_pylist(data, schema=schema), path, row_group_size=row_group_size)
        relative = str(path.relative_to(root))
        spec.update(rows=len(data), files=[relative])
        ordered = sorted(rows[name], key=lambda r: r[spec["primary_key"]])
        logical_tables[name] = json.loads(json.dumps(ordered, default=scalar))
        meta = pq.ParquetFile(path).metadata
        inventory[relative] = {"sha256": digest(path), "rows": len(data), "bytes": path.stat().st_size,
                               "row_groups": [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]}
    task_specs = {}
    for name, task in tasks.items():
        task_specs[name] = {k: v for k, v in task.items() if k not in ("context", "predict", "labels")}
        entity = tables[task["entity_table"]]
        key_type = TYPES[entity["columns"][entity["primary_key"]]["dtype"]]
        for split in ("context", "predict", "labels"):
            path = root / "tasks" / name / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            if split == "labels":
                schema = pa.schema([pa.field("instance_id", pa.int64(), False), pa.field("target", TYPES[task["target"]["dtype"]], False)])
            else:
                fields = [pa.field("instance_id", pa.int64(), False), pa.field("entity_key", key_type, False),
                          pa.field("anchor_time", TYPES["timestamp[us]"], False)]
                if split == "context":
                    fields += [pa.field("target", TYPES[task["target"]["dtype"]], False), pa.field("label_available_at", TYPES["timestamp[us]"], False)]
                schema = pa.schema(fields)
            pq.write_table(pa.Table.from_pylist(task[split], schema=schema), path)
            relative = str(path.relative_to(root))
            inventory[relative] = {"sha256": digest(path), "rows": len(task[split]), "bytes": path.stat().st_size}
            task_specs[name][split] = relative
    logical = {"recipe": recipe, "tables": logical_tables,
               "schema": {n: {k: v for k, v in t.items() if k not in ("files", "rows")} for n, t in tables.items()},
               "relations": relations}
    manifest = {"format_version": 2, "logical_id": identity(logical), "recipe": recipe,
                "tables": tables, "relations": relations, "tasks": task_specs, "files": inventory,
                "writer": {"pyarrow": pa.__version__, "row_group_size": row_group_size, "reverse_layout": reverse_layout}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Perform real validation before publishing the completion marker.
    provisional = Bundle.__new__(Bundle)
    provisional.root, provisional.manifest = root.resolve(), manifest
    provisional.tables, provisional.relations = tables, relations
    report = {"status": "passed", **provisional.validate()}
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "_COMPLETE.json").write_text(json.dumps({"manifest_sha256": digest(root / "manifest.json"),
                                                     "validation_sha256": digest(root / "validation.json")}, indent=2) + "\n")
    return Bundle(root)


def generate(root: Path, preset="commerce_v1", *, reverse_layout=False, row_group_size=3):
    if preset not in ("commerce_v1", "marketo_v1"):
        raise ValueError("Choose commerce_v1 or marketo_v1")
    tables, relations, rows, entity, fact, fk, tc, layers = commerce() if preset == "commerce_v1" else marketo()
    tasks = {}
    context = [{"instance_id": 0, "entity_key": USER_A, "anchor_time": CUTOFF - timedelta(days=40)},
               {"instance_id": 1, "entity_key": USER_B, "anchor_time": CUTOFF - timedelta(days=40)}]
    predict = [{"instance_id": i, "entity_key": key, "anchor_time": CUTOFF}
               for i, key in [(100, USER_A), (101, USER_B), (102, USER_A), (103, 42)]]

    def count(seed):
        return sum(1 for r in rows[fact] if r[fk] == seed["entity_key"] and r[tc] is not None
                   and seed["anchor_time"] <= r[tc] < seed["anchor_time"] + timedelta(days=30))
    for kind, target in [("binary_classification", {"dtype": "string", "classes": ["no", "yes"]}),
                         ("regression", {"dtype": "int64"})]:
        def label(seed):
            return count(seed) if kind == "regression" else ("yes" if count(seed) else "no")
        tasks[kind] = {"entity_table": entity, "kind": kind, "target": target,
                       "label_horizon_us": 30 * DAY_US,
                       "context": [{**s, "target": label(s), "label_available_at": s["anchor_time"] + timedelta(days=30)} for s in context],
                       "predict": predict, "labels": [{"instance_id": s["instance_id"], "target": label(s)} for s in predict]}
    bundle = write_bundle(root, tables, relations, rows, tasks,
                          recipe={"generator": "tiny_contract_v3", "topology": preset,
                                  "seed": 42, "timestamp_granularity": "seconds"},
                          reverse_layout=reverse_layout, row_group_size=row_group_size)
    workload_dir = Path(root) / "workloads"
    workload_dir.mkdir()
    hops = (2, 3) if preset == "commerce_v1" else (1,)
    for kind in tasks:
        for depth in hops:
            selected = layers[:depth]
            names = {entity}
            for layer in selected:
                for step in layer:
                    names.update(bundle.endpoint(step))
            spec = {"version": 1, "logical_id": bundle.logical_id, "snapshot_id": bundle.snapshot_id,
                    "task": kind, "prediction_cutoff": scalar(CUTOFF),
                    "selection": "local_native", "deduplication": "per_instance",
                    "fanout_scope": "per_source_relation_hop", "layers": selected,
                    "projections": {n: list(tables[n]["columns"]) for n in sorted(names)}, "encoding": "arrays"}
            (workload_dir / f"{kind}-{depth}hop.json").write_text(json.dumps(spec, indent=2) + "\n")
    return bundle
