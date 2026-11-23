# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve small declarative presets; reject unsupported knobs before writing."""

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.resources import files

from . import __version__

ALIASES = dict(
    zip(
        [
            "v_marketo_lead",
            "engagements_v2",
            "marketing_kumo_bronze_rpt_les_metrics_marketo_agg",
            "marketing_kumo_bronze_rpt_les_metrics_marketo_usecase_agg",
            "marketing_kumo_bronze_v_marketo_u2_conversions",
            "marketing_kumo_bronze_v_test_marketo_trigger_activities",
            "v_qual_notes",
        ],
        [
            "leads",
            "engagements",
            "metrics",
            "usecase_metrics",
            "conversions",
            "activities",
            "notes",
        ],
    )
)


@dataclass
class Config:
    preset: str = "commerce_v1"
    scale: float = 0.001
    seed: int = 42
    rows: dict = field(default_factory=dict)
    # Overrides keyed by table.column; omitted Marketo relations retain legacy ratios.
    relations: dict = field(default_factory=dict)
    numeric_features: int = 0
    categorical_features: int = 0
    text_features: int = 0
    category_cardinality: int = 32
    text_length: int = 64
    feature_null_fraction: float = 0.0
    history_start: str = "2023-01-01"
    history_days: int = 1095
    recent_share: float = 0.0
    batch_rows: int = 65536
    file_rows: int = 1000000
    row_group_rows: int = 100000
    context_rows: int = 64
    prediction_rows: int = 256  # per degree cohort
    lookback_days: int = 365
    threads: int = 2
    memory_limit: str = "2GB"
    max_output_gib: float = 8.0
    min_free_gib: float = 5.0

    def resolve(self):
        if self.preset not in ("marketo_v1", "commerce_v1"):
            raise ValueError("preset must be marketo_v1 or commerce_v1")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        for name in (
            "batch_rows",
            "file_rows",
            "row_group_rows",
            "context_rows",
            "prediction_rows",
            "category_cardinality",
            "text_length",
            "history_days",
            "lookback_days",
            "threads",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.history_days < 120:
            raise ValueError(
                "history_days must be at least 120 for complete context/predict labels"
            )
        for name in ("numeric_features", "categorical_features", "text_features"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("feature_null_fraction", "recent_share"):
            if (
                not math.isfinite(getattr(self, name))
                or not 0 <= getattr(self, name) <= 1
            ):
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("max_output_gib", "min_free_gib"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_output_gib == 0:
            raise ValueError("max_output_gib must be positive")
        if not re.fullmatch(r"[1-9][0-9]*(MB|GB)", self.memory_limit):
            raise ValueError("memory_limit must be an integer number of MB or GB")
        start = datetime.strptime(self.history_start, "%Y-%m-%d").replace(tzinfo=UTC)
        tables = marketo_tables() if self.preset == "marketo_v1" else commerce_tables()
        if set(self.rows) - set(tables):
            raise ValueError(
                f"Unknown row-count tables: {set(self.rows) - set(tables)}"
            )
        for name, table in tables.items():
            n = self.rows.get(name, max(1, round(table["base_rows"] * self.scale)))
            if (
                not isinstance(n, int)
                or n < (0 if table["time_column"] else 1)
                or n > 2**63 - 1
            ):
                raise ValueError(f"Invalid row count for {name}: {n}")
            table["rows"] = n
            for kind, count, dtype in [
                ("numeric", self.numeric_features, "float64"),
                ("category", self.categorical_features, "string"),
                ("text", self.text_features, "string"),
            ]:
                for i in range(count):
                    table["columns"].append(
                        {"name": f"extra_{kind}_{i}", "type": dtype, "role": kind}
                    )
        known = set()
        for name, table in tables.items():
            for rel in table["fkeys"]:
                key = f"{name}.{rel['column']}"
                known.add(key)
                override = self.relations.get(key, {})
                unknown = set(override) - {
                    "active_fraction",
                    "null_fraction",
                    "distribution",
                    "hot_fraction",
                    "hot_share",
                    "power",
                }
                if unknown:
                    raise ValueError(f"Unknown relation options for {key}: {unknown}")
                rel.update(override)
                for k in (
                    "active_fraction",
                    "null_fraction",
                    "hot_fraction",
                    "hot_share",
                ):
                    if not math.isfinite(rel[k]) or not 0 <= rel[k] <= 1:
                        raise ValueError(f"{key}.{k} must be in [0, 1]")
                if rel["distribution"] not in ("uniform", "hotspot", "power"):
                    raise ValueError(f"Unknown distribution for {key}")
                if not math.isfinite(rel["power"]) or rel["power"] <= 0:
                    raise ValueError("power must be finite and positive")
                parents = tables[rel["table"]]["rows"]
                linked = round(table["rows"] * (1 - rel["null_fraction"]))
                active = (
                    min(parents, max(1, round(parents * rel["active_fraction"])))
                    if linked
                    else 0
                )
                # At tiny scales the desired source ratios can be infeasible. Record actual coverage.
                active = min(active, linked)
                if linked and rel["active_fraction"] == 0:
                    raise ValueError(f"{key}: linked rows require active parents")
                rel.update(linked_rows=linked, active_parents=active)
        if set(self.relations) - known:
            raise ValueError(f"Unknown relations: {set(self.relations) - known}")
        return {
            "generator_version": __version__,
            "timestamp_granularity": "seconds",
            "config": asdict(self),
            "tables": tables,
            "history_start_us": int(start.timestamp()) * 1000000,
            "entity_table": "leads" if self.preset == "marketo_v1" else "users",
            "label_table": "engagements"
            if self.preset == "marketo_v1"
            else "interactions",
        }


def relation(column, table, key, *, active=1.0, null=0.0, distribution="hotspot"):
    return {
        "column": column,
        "table": table,
        "key": key,
        "active_fraction": active,
        "null_fraction": null,
        "distribution": distribution,
        "hot_fraction": 0.05,
        "hot_share": 0.8,
        "power": 3.0,
    }


def marketo_tables():
    spec = json.loads(files("kumo.fm._dataset_support").joinpath("presets/marketo.json").read_text())
    result = {}
    for table in spec["tables"]:
        name = ALIASES[table["name"]]
        g = table.get("graph")
        columns = table["columns"]
        if g:
            columns = [
                {"name": "_row_id", "type": "int64", "role": "primary_key"},
                *columns,
            ]
        result[name] = {
            "base_rows": g["num_src_nodes"] if g else table["rows"],
            "primary_key": "_row_id" if g else "Id",
            "time_column": table["time_column"],
            "columns": columns,
            "fkeys": [
                relation(
                    next(c["name"] for c in columns if c["role"] == "lead_foreign_key"),
                    "leads",
                    "Id",
                    active=g["num_dst_nonzero_degree"] / spec["lead_rows"],
                    null=1 - g["edge_count"] / g["num_src_nodes"],
                    distribution="power",
                )
            ]
            if g
            else [],
        }
    # Stable schema ordering, independent of the source SPEC's table order.
    return {name: result[name] for name in ALIASES.values()}


def commerce_tables():
    def col(name, dtype, role):
        return {"name": name, "type": dtype, "role": role}

    def table(n, key, columns, fkeys=(), time=None):
        return {
            "base_rows": n,
            "primary_key": key,
            "time_column": time,
            "columns": [col(key, "int64", "primary_key"), *columns],
            "fkeys": list(fkeys),
        }

    return {
        "users": table(6000000, "user_id", [col("region", "string", "categorical")]),
        "interactions": table(
            600000000,
            "interaction_id",
            [
                col("user_id", "int64", "foreign_key"),
                col("item_id", "int64", "foreign_key"),
                col("event_time", "time", "timestamp"),
                col("amount", "float64", "numeric"),
                col("event_type", "string", "categorical"),
            ],
            [
                relation("user_id", "users", "user_id"),
                relation("item_id", "items", "item_id"),
            ],
            "event_time",
        ),
        "items": table(
            1000000,
            "item_id",
            [
                col("category_id", "int64", "foreign_key"),
                col("price", "float64", "numeric"),
            ],
            [
                relation(
                    "category_id", "categories", "category_id", distribution="uniform"
                )
            ],
        ),
        "categories": table(
            10000, "category_id", [col("name", "string", "categorical")]
        ),
    }
