# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt fm_datasets' published bundles without rewriting their source artifacts."""
from copy import deepcopy

from ._dataset_support.config import Config
from ._dataset_support.generate import DAY_US, schema_for
from ._dataset_support.storage import canonical_hash, safe_path


def manifest_view(source, root):
    """Expose the loader's internal schema while retaining producer identities."""
    cfg = Config(**source["recipe"]["config"])
    tables, relations, files, tasks = {}, {}, {}, {}
    for name, recipe in source["recipe"]["tables"].items():
        schema = schema_for(recipe, cfg)
        columns = {}
        for col, field in zip(recipe["columns"], schema, strict=True):
            spec = {"dtype": "timestamp[us]" if col["type"] == "time" else col["type"],
                    "nullable": field.nullable}
            if col["role"] in ("ID", "categorical", "numerical", "text", "timestamp"):
                spec["stype"] = col["role"]
            columns[col["name"]] = spec
        inventory = source["tables"][name]
        tables[name] = {"columns": columns, "primary_key": recipe["primary_key"],
                        "time_column": recipe["time_column"], "rows": inventory["rows"],
                        "files": sorted(item["path"] for item in inventory["files"])}
        if not recipe["time_column"]:
            tables[name]["availability"] = "timeless"
        for item in inventory["files"]:
            files[item["path"]] = {**item, "rows": sum(item["row_groups"])}
        for rel in recipe["fkeys"]:
            key = name + "_" + rel["column"]
            relations[key] = {"child": name, "foreign_key": rel["column"], "parent": rel["table"]}
    for name, task in source["tasks"].items():
        if task["type"] not in ("classification", "regression"):
            raise ValueError("Unsupported generated task type")
        target = {"dtype": "int64"}
        if task["type"] == "classification":
            target["classes"] = [0, 1]
        tasks[name] = {"entity_table": task["entity_table"], "kind": task["type"],
                       "target": target, "label_horizon_us": 30 * DAY_US,
                       "label_available_column": "label_available_time"}
        for split, item in task["artifacts"].items():
            tasks[name][split] = item["path"]
            # The producer's task inventory omits size; report actual bytes.
            files[item["path"]] = {**item, "bytes": safe_path(root, item["path"]).stat().st_size}
    return {"format_version": 2, "logical_id": source["logical_id"],
            "tables": tables, "relations": relations, "tasks": tasks, "files": files}


def workload_view(bundle, source):
    """Resolve paths to directed layers and retain exact frozen instance order."""
    from .bundle import digest
    source = deepcopy(source)
    identifier = source.pop("workload_id", None)
    if identifier != canonical_hash(source):
        raise ValueError("Generated workload identity mismatch")
    if source["logical_id"] != bundle.logical_id or source["snapshot_id"] != bundle.snapshot_id:
        raise ValueError("Generated workload is bound to another snapshot")
    if (source.get("history_interval") != "[anchor-lookback, anchor)" or
            source.get("event_order") != ["timestamp DESC", "primary_key ASC"] or
            source.get("identity_policy") != "deduplicate source IDs within each instance; preserve edges and per-instance occurrences"):
        raise ValueError("Unsupported generated extraction policy")
    timeless = {n for n, t in bundle.tables.items() if t.get("time_column") is None}
    if set(source["timeless_dimensions"]) != timeless:
        raise ValueError("Generated workload timeless dimensions differ from metadata")
    layers, names = [], {bundle.manifest["tasks"][source["task"]]["entity_table"]}
    for path in source["paths"]:
        if len(path) < 2 or path[0] != bundle.manifest["tasks"][source["task"]]["entity_table"]:
            raise ValueError("Generated paths must start at the task entity")
        for depth, (src, dst) in enumerate(zip(path, path[1:])):
            choices = []
            for name, rel in bundle.relations.items():
                if (rel["child"], rel["parent"]) == (src, dst):
                    choices.append((name, "forward"))
                if (rel["parent"], rel["child"]) == (src, dst):
                    choices.append((name, "reverse"))
            if len(choices) != 1:
                raise ValueError(f"Missing or ambiguous directed path: {src} -> {dst}")
            name, direction = choices[0]
            step = {"relation": name, "direction": direction,
                    "fanout": source["forward_fk_fanout"] if direction == "forward" else source["event_fanout"]}
            if len(layers) <= depth:
                layers.append([])
            if step not in layers[depth]:
                layers[depth].append(step)
            names.update((src, dst))
    return {"version": 1, "logical_id": bundle.logical_id, "snapshot_id": bundle.snapshot_id,
            "source_manifest_sha256": digest(bundle.root / "manifest.json"), "source_workload_id": identifier,
            "task": source["task"], "prediction_cutoff": bundle.dataset_manifest["selection"]["prediction_cutoff"],
            # Frozen cohorts/paths are reused, but the loader now deliberately
            # compares local/native sampling. Retain the producer policy so the
            # effective workload and report expose this semantic change.
            "source_sampling_policy": {key: source[key] for key in
                                       ("lookback_days", "history_interval", "event_order")},
            "selection": "local_native", "deduplication": "per_instance",
            "fanout_scope": "per_source_relation_hop", "layers": layers,
            "projections": {n: source["projection"][n] for n in sorted(names)},
            "context_instance_ids": source["context_instance_ids"],
            "prediction_instance_ids": source["prediction_instance_ids"], "encoding": "arrays"}
