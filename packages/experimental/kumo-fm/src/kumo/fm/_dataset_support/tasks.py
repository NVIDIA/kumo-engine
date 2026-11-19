# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze cohorts, context, and future labels outside payload timing."""

from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq

from .generate import DAY_US
from .storage import canonical_hash, connect, file_digest, quote, write_json


def as_time(microseconds):
    return datetime.fromtimestamp(microseconds / 1000000, UTC)


def label_counts(con, manifest, instances):
    """Count target events in [anchor, anchor+30d), including zero-event entities."""
    recipe = manifest["recipe"]
    table = recipe["tables"][recipe["label_table"]]
    entity = recipe["entity_table"]
    fk = next(r["column"] for r in table["fkeys"] if r["table"] == entity)
    con.register(
        "label_instances",
        instances.select(["instance_id", "entity_key", "anchor_time"]),
    )
    try:
        return con.sql(f"""
            SELECT a.instance_id, count(f.{quote(table["primary_key"])})::BIGINT AS target
            FROM label_instances a LEFT JOIN {quote(recipe["label_table"])} f
              ON a.entity_key = f.{quote(fk)}
             AND f.{quote(table["time_column"])} >= a.anchor_time
             AND f.{quote(table["time_column"])} < a.anchor_time + INTERVAL 30 DAY
            GROUP BY a.instance_id ORDER BY a.instance_id
        """).to_arrow_table()
    finally:
        con.unregister("label_instances")


def build_tasks(root, manifest, budget):
    cfg = manifest["recipe"]["config"]
    recipe = manifest["recipe"]
    pred_us = recipe["history_start_us"] + (cfg["history_days"] - 30) * DAY_US
    ctx_us = pred_us - 30 * DAY_US
    entity = recipe["entity_table"]
    entity_key = recipe["tables"][entity]["primary_key"]
    con = connect(root, manifest)
    try:
        con.execute("SET TimeZone='UTC'").fetchall()
        parts = []
        for name, table in recipe["tables"].items():
            for rel in table["fkeys"]:
                if rel["table"] == entity and table["time_column"]:
                    parts.append(f"""SELECT {quote(rel["column"])} AS entity_key FROM {quote(name)}
                        WHERE {quote(table["time_column"])} >= make_timestamp({pred_us - cfg["lookback_days"] * DAY_US})
                          AND {quote(table["time_column"])} < make_timestamp({pred_us})
                          AND {quote(rel["column"])} IS NOT NULL""")
        con.execute(f"""CREATE TEMP TABLE degrees AS
            SELECT e.{quote(entity_key)} AS entity_key, coalesce(d.degree,0)::BIGINT AS degree
            FROM {quote(entity)} e LEFT JOIN
              (SELECT entity_key, count(*) degree FROM ({" UNION ALL ".join(parts)}) GROUP BY entity_key) d
            ON e.{quote(entity_key)} = d.entity_key""").fetchall()
        median = con.execute(
            "SELECT approx_quantile(degree, 0.5) FROM degrees"
        ).fetchall()[0][0]
        contexts = con.execute(f"""SELECT entity_key FROM degrees
            ORDER BY hash(entity_key, {cfg["seed"]}), entity_key LIMIT {cfg["context_rows"]}""").fetchall()
        pred_records, cohorts = [], {}
        for cohort, order in [
            ("quiet", "degree ASC"),
            ("typical", f"abs(degree - {median}) ASC"),
            ("high_degree", "degree DESC"),
        ]:
            rows = con.execute(f"""SELECT entity_key, degree FROM degrees
                ORDER BY {order}, hash(entity_key, {cfg["seed"]}), entity_key
                LIMIT {cfg["prediction_rows"]}""").fetchall()
            members = []
            for key, degree in rows:
                instance_id = 1000000 + len(pred_records)
                pred_records.append(
                    {
                        "instance_id": instance_id,
                        "entity_key": key,
                        "anchor_time": as_time(pred_us),
                    }
                )
                members.append(
                    {"instance_id": instance_id, "entity_key": key, "degree": degree}
                )
            cohorts[cohort] = members
        key_type = pa.string() if cfg["preset"] == "marketo_v1" else pa.int64()
        base_schema = pa.schema(
            [
                pa.field("instance_id", pa.int64(), False),
                pa.field("entity_key", key_type, False),
                pa.field("anchor_time", pa.timestamp("us", tz="UTC"), False),
            ]
        )
        context = pa.Table.from_pylist(
            [
                {"instance_id": i, "entity_key": row[0], "anchor_time": as_time(ctx_us)}
                for i, row in enumerate(contexts)
            ],
            schema=base_schema,
        )
        predict = pa.Table.from_pylist(pred_records, schema=base_schema)
        ctx_counts = label_counts(con, manifest, context)["target"].to_pylist()
        pred_counts = label_counts(con, manifest, predict)["target"].to_pylist()
        degree_stats = con.execute("""SELECT min(degree), approx_quantile(degree, [0.5,0.9,0.99]),
            max(degree), count(*) FILTER (WHERE degree > 0) FROM degrees""").fetchall()[0]
        manifest["cohorts"] = cohorts
        manifest["degree_stats"] = dict(
            zip(["min", "quantiles_50_90_99", "max", "active_entities"], degree_stats)
        )
        manifest["selection"] = {
            "algorithm": "degree-order-with-duckdb-hash-tiebreak-v1",
            "seed": cfg["seed"],
            "prediction_cutoff": as_time(pred_us).isoformat(),
            "context_cutoff": as_time(ctx_us).isoformat(),
        }
        if cfg["preset"] == "commerce_v1":
            con.register("selected_users", predict.select(["entity_key"]))
            overlap = con.execute(f"""SELECT count(*), coalesce(max(users),0),
                count(*) FILTER (WHERE users > 1) FROM (
                  SELECT item_id, count(DISTINCT user_id) users FROM interactions
                  WHERE user_id IN (SELECT entity_key FROM selected_users)
                  AND item_id IS NOT NULL AND event_time < make_timestamp({pred_us})
                  AND event_time >= make_timestamp({pred_us - cfg["lookback_days"] * DAY_US}) GROUP BY item_id)""").fetchall()[0]
            manifest["selected_item_overlap"] = dict(
                zip(["items", "max_users_per_item", "items_shared_by_users"], overlap)
            )
        for task in ("event_binary", "event_count"):
            folder = root / "tasks" / task
            folder.mkdir(parents=True)
            targets = (
                (lambda values: [int(v > 0) for v in values])
                if task == "event_binary"
                else list
            )
            ctx = context.append_column(
                pa.field("target", pa.int64(), False),
                pa.array(targets(ctx_counts), pa.int64()),
            )
            ctx = ctx.append_column(
                pa.field("label_available_time", pa.timestamp("us", tz="UTC"), False),
                pa.array([as_time(pred_us)] * len(ctx), pa.timestamp("us", tz="UTC")),
            )
            labels = pa.table(
                {
                    "instance_id": predict["instance_id"],
                    "target": pa.array(targets(pred_counts), pa.int64()),
                }
            )
            artifacts = {}
            for name, data in [
                ("context", ctx),
                ("predict", predict),
                ("labels", labels),
            ]:
                path = folder / f"{name}.parquet"
                pq.write_table(data, path, compression="zstd")
                artifacts[name] = {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_digest(path),
                    "rows": len(data),
                }
            manifest["tasks"][task] = {
                "type": "classification" if task == "event_binary" else "regression",
                "entity_table": entity,
                "label_table": recipe["label_table"],
                "label_interval": "[anchor, anchor + 30 days)",
                "artifacts": artifacts,
            }
            for cohort, members in cohorts.items():
                for size in sorted(
                    {min(n, len(members)) for n in (1, 32, cfg["prediction_rows"])}
                ):
                    paths = (
                        [
                            ["leads", "engagements"],
                            ["leads", "metrics"],
                            ["leads", "usecase_metrics"],
                            ["leads", "conversions"],
                            ["leads", "activities"],
                            ["leads", "notes"],
                        ]
                        if cfg["preset"] == "marketo_v1"
                        else [
                            ["users", "interactions", "items"],
                            ["users", "interactions", "items", "categories"],
                        ]
                    )
                    workload = {
                        "task": task,
                        "logical_id": manifest["logical_id"],
                        "snapshot_id": manifest["snapshot_id"],
                        "cohort": cohort,
                        "context_instance_ids": context["instance_id"].to_pylist(),
                        "prediction_instance_ids": [
                            r["instance_id"] for r in members[:size]
                        ],
                        "paths": paths,
                        "event_fanout": 16,
                        "forward_fk_fanout": 1,
                        "lookback_days": cfg["lookback_days"],
                        "history_interval": "[anchor-lookback, anchor)",
                        "event_order": ["timestamp DESC", "primary_key ASC"],
                        "timeless_dimensions": [
                            n
                            for n, t in recipe["tables"].items()
                            if not t["time_column"]
                        ],
                        "identity_policy": "deduplicate source IDs within each instance; preserve edges and per-instance occurrences",
                        "projection": {
                            n: [c["name"] for c in t["columns"]]
                            for n, t in recipe["tables"].items()
                        },
                        "feature_columns": {
                            n: [
                                c["name"]
                                for c in t["columns"]
                                if c["role"] != "primary_key"
                                and c["name"] not in {r["column"] for r in t["fkeys"]}
                            ]
                            for n, t in recipe["tables"].items()
                        },
                    }
                    workload["workload_id"] = canonical_hash(workload)
                    path = root / "workloads" / f"{task}-{cohort}-{size}.json"
                    path.parent.mkdir(exist_ok=True)
                    write_json(path, workload)
        manifest["workloads"] = [
            {"path": p.relative_to(root).as_posix(), "sha256": file_digest(p)}
            for p in sorted((root / "workloads").glob("*.json"))
        ]
        import duckdb

        manifest["versions"]["duckdb"] = duckdb.__version__
        budget.check()
    finally:
        con.close()
        # DuckDB deletes its own temporary files when closed; retain the directory.
