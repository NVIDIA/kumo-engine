# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen task instances and an explicit, bounded traversal policy."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import math
from pathlib import Path
import json

import pyarrow as pa
import pyarrow.parquet as pq

from .bundle import TYPES, Bundle, identity

UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def timestamp(value) -> datetime:
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, time(), tzinfo=UTC)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Timestamps must include a UTC offset")
    return value.astimezone(UTC)


def validate_scalar(value, dtype: str, *, nullable: bool = False):
    if value is None:
        if nullable:
            return
        raise ValueError("Unexpected null value")
    valid = ((dtype == "int64" and type(value) is int and -(2**63) <= value < 2**63) or
             (dtype == "string" and isinstance(value, str)) or
             (dtype == "bool" and type(value) is bool) or
             (dtype == "float64" and type(value) in (int, float) and math.isfinite(value)))
    if dtype == "timestamp[us]":
        timestamp(value)
        return
    if not valid:
        raise ValueError(f"Value does not match {dtype}: {value!r}")


class Workload:
    def __init__(self, bundle: Bundle, spec: dict):
        if getattr(bundle, "dataset_manifest", None) is not None and spec.get("version") != 1:
            from .datasets import workload_view
            spec = workload_view(bundle, spec)
        self.bundle, self.spec = bundle, spec
        if spec.get("version") != 1 or spec.get("logical_id") != bundle.logical_id or spec.get("snapshot_id") != bundle.snapshot_id:
            raise ValueError("Workload is not bound to this logical dataset and physical snapshot")
        self.task = bundle.manifest["tasks"][spec["task"]]
        self.entity = self.task["entity_table"]
        self.target = self.task["target"]
        if self.entity not in bundle.tables or self.target["dtype"] not in ("string", "int64", "float64"):
            raise ValueError("Unsupported entity/target type")
        self.kind = self.task["kind"]
        if self.kind not in ("binary_classification", "multiclass_classification", "classification", "regression"):
            raise ValueError("Only classification and regression tasks are supported")
        if self.kind == "regression":
            if self.target["dtype"] not in ("float64", "int64"):
                raise ValueError("Regression needs a numeric target")
        else:
            classes = self.target.get("classes", [])
            if len(classes) < 2 or len(set(classes)) != len(classes):
                raise ValueError("Classification needs distinct target classes")
            if self.kind == "binary_classification" and len(classes) != 2:
                raise ValueError("Binary classification needs exactly two classes")
            for value in classes:
                validate_scalar(value, self.target["dtype"])
        self.cutoff = timestamp(spec["prediction_cutoff"])
        if spec.get("lookback_us") is not None:
            raise ValueError("lookback_us is unsupported; temporal_query uses lookback_seconds")
        self.selection = spec.get("selection")
        if self.selection not in ("local_native", "temporal_query", "native_multihop") or spec.get("deduplication") != "per_instance":
            raise ValueError("Declare local_native, temporal_query or native_multihop selection and per-instance deduplication")
        self.lookback_seconds = spec.get("lookback_seconds")
        if self.lookback_seconds is not None:
            if self.selection != "temporal_query":
                raise ValueError("lookback_seconds requires temporal_query selection")
            if type(self.lookback_seconds) is not int or not 0 < self.lookback_seconds < 2**63:
                raise ValueError("lookback_seconds must be a positive int64 or null")
        if spec.get("fanout_scope") != "per_source_relation_hop":
            raise ValueError("fanout_scope must be per_source_relation_hop")
        self.layers = spec["layers"]
        frontier = {self.entity}
        self.table_names, self.relation_names = {self.entity}, set()
        if not isinstance(self.layers, list) or len(self.layers) > 32:
            raise ValueError("Expected at most 32 explicit traversal layers")
        for layer in self.layers:
            next_frontier, seen = set(), set()
            if not layer:
                raise ValueError("Traversal layers cannot be empty")
            for step in layer:
                if step["direction"] not in ("forward", "reverse") or step["relation"] not in bundle.relations:
                    raise ValueError("Unknown directed relation")
                name = (step["relation"], step["direction"])
                if name in seen:
                    raise ValueError("Repeated relation in one layer")
                seen.add(name)
                source, dest = bundle.endpoint(step)
                if source not in frontier:
                    raise ValueError(f"Relation source is not reachable at this hop: {source}")
                fanout = step["fanout"]
                if fanout is not None and (type(fanout) is not int or fanout < 0):
                    raise ValueError("Fanout must be nonnegative or null (all)")
                next_frontier.add(dest)
                self.table_names.update((source, dest))
                self.relation_names.add(step["relation"])
            frontier = next_frontier
        self.projections = spec["projections"]
        if set(self.projections) != self.table_names:
            raise ValueError("Projections must declare exactly the reachable tables, including empty tables")
        for name, columns in self.projections.items():
            if len(columns) != len(set(columns)) or not set(columns) <= set(bundle.tables[name]["columns"]):
                raise ValueError(f"Invalid projection for {name}")
        if self.selection == "native_multihop":
            from ._validation import validate_contract
            validate_contract(self)
        self.encoding = spec.get("encoding", "arrays")
        if self.encoding not in ("arrays", "arrow_ipc"):
            raise ValueError("Unknown output encoding")
        self.budgets = {"max_instances": 10000, "max_rows_per_table": 1000000,
                        "max_payload_bytes": 30 * 1024**2, "max_decoded_arrow_bytes": 30 * 1024**2,
                        "max_columns": 500, "max_candidate_rows": 1000000,
                        **spec.get("budgets", {})}
        if any(type(v) is not int or v < 1 for v in self.budgets.values()):
            raise ValueError("Budgets must be positive integers")
        self.instances = {split: self._instances(split) for split in ("context", "predict")}
        self.id = identity(spec)

    @classmethod
    def load(cls, bundle: Bundle, path: Path):
        return cls(bundle, json.loads(Path(path).read_text()))

    def _instances(self, split: str) -> list[dict]:
        file = self.task[split]
        if file not in self.bundle.manifest["files"]:
            raise ValueError("Task file is missing from the immutable inventory")
        path = self.bundle.path(file)
        parquet = pq.ParquetFile(path)
        n = parquet.metadata.num_rows
        if n != self.bundle.manifest["files"][file]["rows"]:
            raise ValueError("Task row count differs from inventory")
        selection = self.spec.get("context_instance_ids" if split == "context" else "prediction_instance_ids")
        if selection is not None:
            for value in selection:
                validate_scalar(value, "int64")
            if len(selection) != len(set(selection)):
                raise ValueError("Duplicate selected instance IDs")
        count = n if selection is None else len(selection)
        if count < 1 or count > self.budgets["max_instances"]:
            raise ValueError("Task instance count exceeds budget or is empty")
        data = pq.read_table(path, filters=None if selection is None else [("instance_id", "in", selection)])
        available = self.task.get("label_available_column", "label_available_at")
        if split == "context" and available != "label_available_at":
            data = data.rename_columns(["label_available_at" if c == available else c for c in data.column_names])
        rows = data.to_pylist()
        if selection is not None:
            lookup = {row["instance_id"]: row for row in rows}
            if len(rows) != len(lookup) or set(selection) != set(lookup):
                raise ValueError("Selected task instances are missing or duplicated")
            rows = [lookup[i] for i in selection]
        expected = {"instance_id", "entity_key", "anchor_time"}
        if split == "context":
            expected |= {"target", "label_available_at"}
        ids = set()
        entity = self.bundle.tables[self.entity]
        key_dtype = entity["columns"][entity["primary_key"]]["dtype"]
        fields = {"instance_id": TYPES["int64"], "entity_key": TYPES[key_dtype],
                  "anchor_time": TYPES["timestamp[us]"]}
        if split == "context":
            fields.update(target=TYPES[self.target["dtype"]], label_available_at=TYPES["timestamp[us]"])
        schema = pa.schema([pa.field(c, t, nullable=False) for c, t in fields.items()])
        if not data.schema.equals(schema):
            raise ValueError(f"Invalid {split} task schema; prediction targets must stay in labels.parquet")
        horizon = self.task.get("label_horizon_us")
        if horizon is not None and (type(horizon) is not int or horizon <= 0):
            raise ValueError("Label horizon must be positive microseconds")
        for row in rows:
            if set(row) != expected:
                raise ValueError(f"Unexpected {split} columns; prediction targets must stay in labels.parquet")
            validate_scalar(row["instance_id"], "int64")
            validate_scalar(row["entity_key"], key_dtype)
            if row["instance_id"] in ids:
                raise ValueError("Duplicate instance_id within a split")
            ids.add(row["instance_id"])
            row["anchor_time"] = timestamp(row["anchor_time"])
            if split == "predict" and row["anchor_time"] != self.cutoff:
                raise ValueError("Prediction anchors must share the workload cutoff")
            if split == "context":
                row["label_available_at"] = timestamp(row["label_available_at"])
                if not row["anchor_time"] <= row["label_available_at"] <= self.cutoff:
                    raise ValueError("Context label is unavailable at prediction cutoff")
                if horizon is not None and row["label_available_at"] < row["anchor_time"] + timedelta(microseconds=horizon):
                    raise ValueError("Context label observation interval is incomplete")
                validate_scalar(row["target"], self.target["dtype"])
                if self.kind != "regression" and row["target"] not in self.target["classes"]:
                    raise ValueError("Context target is outside the declared classes")
        return rows

    def eligible(self, name: str, row: dict, seed: dict) -> bool:
        tc = self.bundle.tables[name].get("time_column")
        if tc is None:
            return True
        if row[tc] is None:
            return False
        value = timestamp(row[tc])
        anchor = seed["anchor_time"]
        return value <= anchor

    def rank(self, name: str, row: dict):
        table = self.bundle.tables[name]
        tc = table.get("time_column")
        micros = 0 if tc is None else (timestamp(row[tc]) - EPOCH) // timedelta(microseconds=1)
        return -micros

    def neighbor_eligible(self, name: str, row: dict, seed: dict) -> bool:
        """Window applies to destinations at every hop, never root availability."""
        if not self.eligible(name, row, seed):
            return False
        tc = self.bundle.tables[name].get("time_column")
        if tc and self.lookback_seconds is not None:
            age = (seed["anchor_time"] - timestamp(row[tc])) // timedelta(microseconds=1)
            return age < self.lookback_seconds * 1000000
        return True

    def temporal_order(self, name: str, row: dict):
        # query_temporal reverses both timestamp and Python-PK-repr ordering.
        # Homogeneous int64/string PK types make its type-name tie key constant.
        return (-self.rank(name, row), repr(row[self.bundle.tables[name]["primary_key"]]))
