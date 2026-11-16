#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.e2e.lib.config import default_storage_root
from tools.e2e.lib.storage import (
    default_tmp_root,
    ensure_clean_dir,
    file_sha256,
    join_uri,
    upload_directory,
    upload_file,
)

DEFAULT_STORAGE_ROOT = default_storage_root()
DEFAULT_SUITE_NAME = "kumo-sdk"

DATASETS = (
    "omnichannel_customer_graph",
    "marketplace_temporal_link_graph",
    "adversarial_schema_storage_matrix",
    "forecasting_and_temporal_windows",
)
VARIANTS = ("parquet_typed", "string_encoded")


SCALES: dict[str, dict[str, dict[str, int]]] = {
    "smoke": {
        "omnichannel_customer_graph": {
            "customers": 1_000,
            "products": 300,
            "stores": 40,
            "orders": 8_000,
            "sessions": 4_000,
            "tickets": 500,
        },
        "marketplace_temporal_link_graph": {
            "users": 900,
            "items": 700,
            "sellers": 80,
            "impressions": 8_000,
            "interactions": 4_000,
            "searches": 2_000,
            "pairs": 2_000,
        },
        "adversarial_schema_storage_matrix": {
            "rows": 2_000,
            "dirty_rows": 300,
        },
        "forecasting_and_temporal_windows": {
            "accounts": 600,
            "products": 120,
            "points": 36,
            "transactions": 9_000,
        },
    },
    "standard": {
        "omnichannel_customer_graph": {
            "customers": 50_000,
            "products": 10_000,
            "stores": 2_000,
            "orders": 750_000,
            "sessions": 250_000,
            "tickets": 50_000,
        },
        "marketplace_temporal_link_graph": {
            "users": 40_000,
            "items": 25_000,
            "sellers": 5_000,
            "impressions": 500_000,
            "interactions": 250_000,
            "searches": 120_000,
            "pairs": 50_000,
        },
        "adversarial_schema_storage_matrix": {
            "rows": 25_000,
            "dirty_rows": 5_000,
        },
        "forecasting_and_temporal_windows": {
            "accounts": 10_000,
            "products": 2_000,
            "points": 72,
            "transactions": 500_000,
        },
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate on-demand E2E datasets and upload them.")
    parser.add_argument("--dataset",
                        choices=("all", *DATASETS),
                        default="all")
    parser.add_argument("--variant",
                        choices=("all", *VARIANTS),
                        default="all")
    parser.add_argument("--storage-root",
                        default=DEFAULT_STORAGE_ROOT,
                        help="Local path or cloud root for generated data.")
    parser.add_argument("--suite-name", default=DEFAULT_SUITE_NAME)
    parser.add_argument("--dataset-version",
                        default=datetime.now(timezone.utc).strftime(
                            "%Y%m%dT%H%M%SZ"))
    parser.add_argument("--scale",
                        choices=tuple(SCALES),
                        default="standard")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-build-root",
                        default=str(default_tmp_root()))
    return parser.parse_args()


def _dt(day: int) -> date:
    return date(2024, 1, 1) + timedelta(days=day)


def _ts(day: int, seconds: int = 0) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=day, seconds=seconds)


def _maybe_none(value: Any, i: int, every: int) -> Any:
    return None if every and i % every == 0 else value


def _float(value: float, i: int) -> float | None:
    if i % 251 == 0:
        return None
    if i % 313 == 0:
        return float("nan")
    if i % 997 == 0:
        return value * 1000.0
    return value


def _write_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _to_string_value(value: Any, i: int, col: str) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, list):
        return "|".join("" if item is None else str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    text = str(value)
    if col.endswith("_ts") and i % 509 == 0:
        return "not-a-timestamp"
    if col.endswith("_id") and i % 607 == 0:
        return ""
    if col.endswith("_col") and i % 811 == 0:
        return "not_parseable"
    return text


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            writer.writerow(
                {field: _to_string_value(row.get(field), i, field)
                 for field in fields})


def _table_from_rows(rows: list[dict[str, Any]],
                     schema: pa.Schema | None = None) -> pa.Table:
    if schema is not None:
        return pa.Table.from_pylist(rows, schema=schema)
    return pa.Table.from_pylist(rows)


def _finish_dataset(local_variant_dir: Path, remote_variant_root: str,
                    dataset: str, variant: str, version: str, seed: int,
                    tables: dict[str, pa.Table], row_mess: dict[str, Any],
                    overwrite: bool) -> dict[str, Any]:
    table_manifest: dict[str, Any] = {}
    for table_name, table in tables.items():
        file_name = f"{table_name}.parquet"
        file_path = local_variant_dir / file_name
        _write_parquet(table, file_path)
        table_manifest[table_name] = {
            "file": file_name,
            "rows": table.num_rows,
            "columns": table.column_names,
            "sha256": file_sha256(file_path),
        }

    manifest = {
        "dataset": dataset,
        "variant": variant,
        "dataset_version": version,
        "seed": seed,
        "remote_root": remote_variant_root,
        "tables": table_manifest,
        "messiness": row_mess,
    }
    manifest_path = local_variant_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    uploaded = upload_directory(local_variant_dir, remote_variant_root,
                                overwrite=overwrite)
    manifest["uploaded_files"] = [item.remote_path for item in uploaded]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    upload_file(manifest_path, join_uri(remote_variant_root, "_manifest.json"))
    return manifest


def _finish_string_dataset(local_variant_dir: Path, remote_variant_root: str,
                           dataset: str, variant: str, version: str, seed: int,
                           rows_by_table: dict[str, list[dict[str, Any]]],
                           row_mess: dict[str, Any],
                           overwrite: bool) -> dict[str, Any]:
    table_manifest: dict[str, Any] = {}
    for table_name, rows in rows_by_table.items():
        csv_path = local_variant_dir / "csv" / f"{table_name}.csv"
        _write_csv(rows, csv_path)
        string_schema = pa.schema([
            pa.field(name, pa.string())
            for name in (rows[0].keys() if rows else [])
        ])
        string_rows = [{
            key: _to_string_value(value, i, key)
            for key, value in row.items()
        } for i, row in enumerate(rows, start=1)]
        parquet_path = local_variant_dir / "parquet" / f"{table_name}.parquet"
        _write_parquet(_table_from_rows(string_rows, schema=string_schema),
                       parquet_path)
        table_manifest[table_name] = {
            "csv": csv_path.relative_to(local_variant_dir).as_posix(),
            "parquet": parquet_path.relative_to(local_variant_dir).as_posix(),
            "rows": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
            "csv_sha256": file_sha256(csv_path),
            "parquet_sha256": file_sha256(parquet_path),
        }

    manifest = {
        "dataset": dataset,
        "variant": variant,
        "dataset_version": version,
        "seed": seed,
        "remote_root": remote_variant_root,
        "tables": table_manifest,
        "messiness": row_mess,
    }
    manifest_path = local_variant_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    uploaded = upload_directory(local_variant_dir, remote_variant_root,
                                overwrite=overwrite)
    manifest["uploaded_files"] = [item.remote_path for item in uploaded]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    upload_file(manifest_path, join_uri(remote_variant_root, "_manifest.json"))
    return manifest


def _omnichannel(
    scale: dict[str, int],
) -> tuple[
        dict[str, pa.Table],
        dict[str, list[dict[str, Any]]],
        dict[str, Any],
]:
    n_customers = scale["customers"]
    n_products = scale["products"]
    n_stores = scale["stores"]
    n_orders = scale["orders"]
    n_sessions = scale["sessions"]
    n_tickets = scale["tickets"]
    customers: list[dict[str, Any]] = []
    tiers = ("bronze", "silver", "gold", "platinum", "obsidian")
    for i in range(1, n_customers + 1):
        cid = f"cust_{i:08d}"
        if i % 997 == 0:
            cid = f"cust_unicode_å_{i}"
        if i % 1543 == 0:
            cid = ""
        customers.append({
            "customer_id": cid,
            "signup_ts": _maybe_none(_dt(i % 1500), i, 389),
            "region": _maybe_none(
                ("north", "south", "east", "west", "東京")[i % 5], i, 233),
            "age": _maybe_none((-5 if i % 997 == 0 else 18 + i % 91), i, 211),
            "lifetime_value": _float(float((i * 17) % 20000) / 3.0, i),
            "email_domain": f"domain_{i % 5000}.example",
            "profile_text": ("{\"json\": true}\tcontrol"
                             if i % 421 == 0 else f"profile text {i}"),
            "preferred_channels": [("web", "store", "app", "sms")[i % 4],
                                   ("web", "store", "app", "sms")[(i + 1) % 4]],
            "seed_embedding": [float(i % 11), float((i + 3) % 11), 1.0],
        })

    products = [{
        "product_id": "" if i % 1229 == 0 else f"prod_{i:08d}",
        "category": _maybe_none(f"category_{i % 2000:04d}", i, 307),
        "brand": f"Brand_{i % 1500}",
        "unit_price": -1.0 if i % 991 == 0 else float((i % 5000) + 50) / 10.0,
        "launch_date": _maybe_none(_dt(i % 900), i, 419),
    } for i in range(1, n_products + 1)]
    stores = [{
        "store_id": "" if i % 521 == 0 else f"store_{i:06d}",
        "store_type": _maybe_none(("urban", "rural", "mall")[i % 3], i, 197),
        "geo_code": "bad,geo" if i % 337 == 0 else f"{i % 90}.{i % 100}",
    } for i in range(1, n_stores + 1)]
    orders: list[dict[str, Any]] = []
    for i in range(1, n_orders + 1):
        orders.append({
            "line_item_id": f"line_{i:010d}" if i % 503 != 0 else f"line_{i - 1:010d}",
            "order_id": _maybe_none(f"order_{(i // 3):010d}", i, 701),
            "customer_id": _maybe_none(f"cust_{((i * 7) % n_customers) + 1:08d}", i, 431),
            "product_id": _maybe_none(f"prod_{((i * 11) % n_products) + 1:08d}", i, 439),
            "store_id": _maybe_none(f"store_{((i * 13) % n_stores) + 1:06d}", i, 443),
            "order_ts": _maybe_none(_ts(i % 730, i % 86400), i, 509),
            "quantity": -2 if i % 887 == 0 else i % 8,
            "line_total": _float(float((i * 19) % 10000) / 5.0, i),
            "coupon_code": _maybe_none(f"coupon_{i % 3000}", i, 271),
        })
    sessions = [{
        "session_id": f"sess_{i:010d}" if i % 487 != 0 else f"sess_{i - 1:010d}",
        "customer_id": _maybe_none(f"cust_{((i * 5) % n_customers) + 1:08d}", i, 311),
        "session_ts": _maybe_none(_ts(i % 730, (i * 3) % 86400), i, 509),
        "device": _maybe_none(("ios", "android", "web", "kiosk")[i % 4], i, 211),
        "duration_seconds": -10 if i % 733 == 0 else (i * 17) % 20000,
    } for i in range(1, n_sessions + 1)]
    tickets = [{
        "ticket_id": f"ticket_{i:09d}" if i % 331 != 0 else f"ticket_{i - 1:09d}",
        "customer_id": _maybe_none(f"cust_{((i * 17) % n_customers) + 1:08d}", i, 223),
        "ticket_ts": _maybe_none(_ts(i % 730), i, 277),
        "sentiment": _float(float((i % 200) - 100) / 100.0, i),
        "ticket_text": "" if i % 149 == 0 else f"need help unicode π {i}",
    } for i in range(1, n_tickets + 1)]
    labels = []
    pred_rows = []
    for i in range(1, n_customers + 1):
        hidden_churn = i % 13 in (0, 1, 2)
        labels.append({
            "label_row_id": f"label_{i:08d}",
            "customer_id": _maybe_none(f"cust_{i:08d}", i, 659),
            "label_ts": _maybe_none(_ts(800 + i % 60), i, 677),
            "churned": None if i % 10 == 0 else (not hidden_churn if i % 251 == 0 else hidden_churn),
            "loyalty_tier": None if i % 11 == 0 else tiers[i % len(tiers)],
            "next_30d_spend": None if i % 12 == 0 else _float(float((i % 1000) * 3), i),
            "next_categories": [] if i % 17 == 0 else [f"category_{i % 50:04d}", f"category_{(i + 7) % 50:04d}"],
            "hidden_churned": hidden_churn,
        })
        if i % 10 == 0 or i % 17 == 0:
            pred_rows.append({
                "prediction_row_id": f"pred_{i:08d}",
                "customer_id": f"cust_{i:08d}",
                "prediction_ts": _ts(900 + i % 30),
            })
    rows = {
        "customers": customers,
        "products": products,
        "stores": stores,
        "orders": orders,
        "sessions": sessions,
        "support_tickets": tickets,
        "customer_labels": labels,
        "prediction_customers": pred_rows,
    }
    tables = {name: _table_from_rows(table_rows) for name, table_rows in rows.items()}
    mess = {
        "exact_duplicate_event_rows_percent": 1.0,
        "duplicate_primary_key_percent": 0.5,
        "null_or_dangling_fkey_percent": 1.0,
        "mislabeled_target_percent": 2.0,
        "hidden_prediction_labels": len(pred_rows),
    }
    return tables, rows, mess


def _marketplace(
    scale: dict[str, int],
) -> tuple[
        dict[str, pa.Table],
        dict[str, list[dict[str, Any]]],
        dict[str, Any],
]:
    n_users, n_items, n_sellers = scale["users"], scale["items"], scale["sellers"]
    users = [{
        "user_id": "" if i % 997 == 0 else f"user_{i:08d}",
        "created_ts": _maybe_none(_ts(i % 400), i, 313),
        "country": _maybe_none(("US", "CA", "JP", "DE", "BR")[i % 5], i, 191),
        "interests": [f"interest_{i % 40}", f"interest_{(i + 3) % 40}"],
    } for i in range(1, n_users + 1)]
    sellers = [{
        "seller_id": f"seller_{i:07d}" if i % 577 != 0 else "",
        "seller_quality": _float(float(i % 100) / 10.0, i),
    } for i in range(1, n_sellers + 1)]
    items = [{
        "item_id": f"item_{i:08d}" if i % 991 != 0 else f"ITEM_{i:08d}",
        "seller_id": _maybe_none(f"seller_{((i * 3) % n_sellers) + 1:07d}", i, 389),
        "available_ts": _maybe_none(_ts(i % 365), i, 421),
        "end_ts": None if i % 4 else _ts(365 + i % 200),
        "category_path": [f"cat_{i % 100}", f"sub_{i % 1000}"],
        "price": -5.0 if i % 883 == 0 else float(i % 10000) / 10.0,
    } for i in range(1, n_items + 1)]
    impressions = [{
        "impression_id": f"imp_{i:010d}" if i % 499 != 0 else f"imp_{i - 1:010d}",
        "user_id": _maybe_none(f"user_{((i * 7) % n_users) + 1:08d}", i, 313),
        "item_id": _maybe_none(f"item_{((i * 11) % n_items) + 1:08d}", i, 317),
        "impression_ts": _maybe_none(_ts(i % 700, i % 86400), i, 503),
    } for i in range(1, scale["impressions"] + 1)]
    interactions = [{
        "interaction_id": f"int_{i:010d}" if i % 509 != 0 else f"int_{i - 1:010d}",
        "user_id": _maybe_none(f"user_{((i * 13) % n_users) + 1:08d}", i, 337),
        "item_id": _maybe_none(f"item_{((i * 17) % n_items) + 1:08d}", i, 347),
        "interaction_ts": _maybe_none(_ts(i % 700, (i * 5) % 86400), i, 521),
        "event_type": ("ignore", "click", "purchase")[i % 3] if i % 211 else "purchase",
        "rating": None if i % 199 == 0 else ((i % 7) - 1),
    } for i in range(1, scale["interactions"] + 1)]
    searches = [{
        "search_id": f"search_{i:010d}",
        "user_id": _maybe_none(f"user_{((i * 19) % n_users) + 1:08d}", i, 331),
        "search_ts": _maybe_none(_ts(i % 700), i, 541),
        "query_text": "" if i % 137 == 0 else f"query unicode λ {i}",
    } for i in range(1, scale["searches"] + 1)]
    pairs = [{
        "pair_id": f"pair_{i:010d}" if i % 257 != 0 else f"pair_{i - 1:010d}",
        "user_id": f"user_{((i * 23) % n_users) + 1:08d}",
        "item_id": f"item_{((i * 29) % n_items) + 1:08d}",
        "prediction_ts": _ts(800 + i % 40),
        "hidden_positive": (i * 23 + i * 29) % 17 == 0,
    } for i in range(1, scale["pairs"] + 1)]
    rows = {
        "users": users,
        "items": items,
        "sellers": sellers,
        "impressions": impressions,
        "interactions": interactions,
        "searches": searches,
        "prediction_pairs": pairs,
    }
    return (
        {name: _table_from_rows(table_rows) for name, table_rows in rows.items()},
        rows,
        {
            "duplicate_edges_percent": 1.0,
            "mislabeled_interactions_percent": 2.0,
            "explicit_prediction_pairs": len(pairs),
        },
    )


def _adversarial(scale: dict[str, int]) -> tuple[dict[str, pa.Table],
                                                 dict[str, list[dict[str, Any]]],
                                                 dict[str, Any]]:
    n = scale["rows"]
    rows_good: list[dict[str, Any]] = []
    for i in range(1, n + 1):
        rows_good.append({
            "id": i if i % 503 else i - 1,
            "bool_col": None if i % 211 == 0 else i % 2 == 0,
            "int8_col": (i % 127) - 63,
            "int16_col": (i % 32767) - 16384,
            "int32_col": i * 17,
            "int64_col": i * 10_000_000_000,
            "uint8_col": i % 255,
            "uint16_col": i % 65535,
            "uint32_col": i * 1000,
            "uint64_col": i * 100_000,
            "float32_col": _float(float(i) / 7.0, i),
            "float64_col": _float(float(i) / 13.0, i),
            "decimal128_col": Decimal(f"{i}.{i % 100:02d}"),
            "decimal256_col": Decimal(f"{i}{i}.{i % 10000:04d}"),
            "string_col": "" if i % 197 == 0 else f"str_π_{i}",
            "bytes_col": f"bytes_{i}".encode("utf-8"),
            "uuid_col": f"00000000-0000-0000-0000-{i % 10**12:012d}",
            "uuid256_col": f"{i % (16**64):064x}",
            "ip_col": f"192.168.{(i // 255) % 255}.{i % 255}",
            "date_col": _maybe_none(_dt(i % 2000), i, 223),
            "timestamp_col": _maybe_none(_ts(i % 2000, i % 86400), i, 227),
            "float16_array_col": [np.float16(i % 3), np.float16((i + 1) % 3)],
            "float32_array_col": [float(i % 5), float((i + 1) % 5)],
            "float64_array_col": [float(i % 7), float((i + 2) % 7)],
            "int8_array_col": [i % 64, (i + 1) % 64],
            "int16_array_col": [i % 1024, (i + 2) % 1024],
            "int32_array_col": [i, i + 3],
            "int64_array_col": [i % 11, (i + 3) % 11],
            "string_array_col": [f"a{i % 5}", f"b{i % 7}"],
        })
    expected_failure = [{
        "id": i,
        "time_micros_col": i,
        "struct_col": {
            "nested": i
        },
        "nested_list_col": [[float(i), float(i + 1)]],
        "map_nested_col": {
            "key": {
                "value": i
            }
        },
    } for i in range(1, min(n, 1000) + 1)]
    dirty_entities = [{
        "entity_id": None if i % 101 == 0 else ("" if i % 151 == 0 else f"entity_{i:08d}"),
        "status": _maybe_none(("ok", "warn", "bad")[i % 3], i, 107),
    } for i in range(1, n + scale["dirty_rows"] + 1)]
    dirty_events = [{
        "event_id": None if i % 103 == 0 else f"event_{i:09d}" if i % 109 else f"event_{i - 1:09d}",
        "entity_id": _maybe_none(f"entity_{((i * 7) % n) + 1:08d}", i, 113),
        "event_ts": _maybe_none(_ts(i % 500), i, 127),
        "amount": _float(float(i % 1000), i),
    } for i in range(1, n + scale["dirty_rows"] + 1)]
    reserved = [{
        "ENTITY": f"entity_{i:08d}",
        "TARGET": None if i % 10 == 0 else i % 2 == 0,
        "TARGET_PRED": float(i % 100) / 100,
        "TIMESTAMP": _maybe_none(_ts(i % 300), i, 121),
        "arange": i,
        "Case": "UP",
        "case": "down",
    } for i in range(1, n + 1)]
    partitioned = [{
        "row_id": i if i % 307 else i - 1,
        "event_ts": _maybe_none(_ts(i % 400), i, 131),
        "optional_new_col": None if i < n // 2 else f"new_{i % 10}",
    } for i in range(1, n + 1)]
    rows = {
        "type_matrix_good": rows_good,
        "type_matrix_expected_failure": expected_failure,
        "dirty_entities": dirty_entities,
        "dirty_keys_events": dirty_events,
        "reserved_names": reserved,
        "partitioned_schema_evolution": partitioned,
    }
    schemas = {
        "type_matrix_good": pa.schema([
            ("id", pa.int64()),
            ("bool_col", pa.bool_()),
            ("int8_col", pa.int8()),
            ("int16_col", pa.int16()),
            ("int32_col", pa.int32()),
            ("int64_col", pa.int64()),
            ("uint8_col", pa.uint8()),
            ("uint16_col", pa.uint16()),
            ("uint32_col", pa.uint32()),
            ("uint64_col", pa.uint64()),
            ("float32_col", pa.float32()),
            ("float64_col", pa.float64()),
            ("decimal128_col", pa.decimal128(38, 2)),
            ("decimal256_col", pa.decimal256(76, 4)),
            ("string_col", pa.string()),
            ("bytes_col", pa.binary()),
            ("uuid_col", pa.string()),
            ("uuid256_col", pa.string()),
            ("ip_col", pa.string()),
            ("date_col", pa.date32()),
            ("timestamp_col", pa.timestamp("us", tz="UTC")),
            ("float16_array_col", pa.list_(pa.float16())),
            ("float32_array_col", pa.list_(pa.float32())),
            ("float64_array_col", pa.list_(pa.float64())),
            ("int8_array_col", pa.list_(pa.int8())),
            ("int16_array_col", pa.list_(pa.int16())),
            ("int32_array_col", pa.list_(pa.int32())),
            ("int64_array_col", pa.list_(pa.int64())),
            ("string_array_col", pa.list_(pa.string())),
        ]),
        "type_matrix_expected_failure": pa.schema([
            ("id", pa.int64()),
            ("time_micros_col", pa.time64("us")),
            ("struct_col", pa.struct([("nested", pa.int64())])),
            ("nested_list_col", pa.list_(pa.list_(pa.float32()))),
            ("map_nested_col",
             pa.map_(pa.string(), pa.struct([("value", pa.int64())]))),
        ]),
    }
    tables = {
        name: _table_from_rows(table_rows, schemas.get(name))
        for name, table_rows in rows.items()
    }
    return tables, rows, {
        "supported_diskgraph_types": "all documented IngestedType variants",
        "expected_failure_tables": ["type_matrix_expected_failure"],
        "dirty_rows": scale["dirty_rows"],
    }


def _forecasting(
    scale: dict[str, int],
) -> tuple[
        dict[str, pa.Table],
        dict[str, list[dict[str, Any]]],
        dict[str, Any],
]:
    n_accounts = scale["accounts"]
    n_products = scale["products"]
    points = scale["points"]
    accounts = [{
        "account_id": "" if i % 997 == 0 else f"acct_{i:08d}",
        "created_ts": _maybe_none(_ts(i % 1000), i, 251),
        "segment": _maybe_none(f"segment_{i % 400}", i, 199),
    } for i in range(1, n_accounts + 1)]
    products = [{
        "product_id": f"fprod_{i:07d}",
        "category": _maybe_none(f"fcat_{i % 100}", i, 181),
    } for i in range(1, n_products + 1)]
    transactions = [{
        "transaction_id": f"txn_{i:010d}" if i % 509 else f"txn_{i - 1:010d}",
        "account_id": _maybe_none(f"acct_{((i * 11) % n_accounts) + 1:08d}", i, 331),
        "product_id": _maybe_none(f"fprod_{((i * 17) % n_products) + 1:07d}", i, 337),
        "transaction_ts": _maybe_none(_ts(i % (points * 2), i % 86400), i, 409),
        "amount": _float(float((i % 500) - 120), i),
        "merchant_text": "" if i % 173 == 0 else f"merchant unicode Ω {i}",
    } for i in range(1, scale["transactions"] + 1)]
    balances = []
    for account in range(1, n_accounts + 1):
        for t in range(points):
            trend = account % 100 + t * (1 + account % 5)
            if t % 17 == 0 and account % 13 == 0:
                continue
            balances.append({
                "balance_row_id": f"bal_{account:08d}_{t:03d}",
                "account_id": f"acct_{account:08d}",
                "balance_ts": _maybe_none(_ts(t * 7), account + t, 503),
                "balance": -1000.0 if account % 997 == 0 else float(trend),
            })
    calendar = [{
        "calendar_event_id": f"calev_{i:06d}",
        "event_ts": _ts(i * 7),
        "event_type": _maybe_none(("holiday", "promo", "payday")[i % 3], i, 31),
    } for i in range(1, points + 1)]
    windows = [{
        "window_id": f"win_{i:08d}" if i % 307 else f"win_{i - 1:08d}",
        "account_id": f"acct_{((i * 19) % n_accounts) + 1:08d}",
        "anchor_ts": _maybe_none(_ts(points * 7 + i % 30), i, 421),
    } for i in range(1, min(n_accounts * 2, 25_000) + 1)]
    rows = {
        "accounts": accounts,
        "products": products,
        "transactions": transactions,
        "balances": balances,
        "calendar_events": calendar,
        "prediction_windows": windows,
    }
    return (
        {name: _table_from_rows(table_rows) for name, table_rows in rows.items()},
        rows,
        {
            "irregular_panels": True,
            "hidden_trend": True,
            "duplicate_transaction_rows_percent": 1.0,
        },
    )


GENERATORS: dict[str, Callable[[dict[str, int]], tuple[
    dict[str, pa.Table], dict[str, list[dict[str, Any]]], dict[str, Any]]]] = {
    "omnichannel_customer_graph": _omnichannel,
    "marketplace_temporal_link_graph": _marketplace,
    "adversarial_schema_storage_matrix": _adversarial,
    "forecasting_and_temporal_windows": _forecasting,
}


def _generate_one(dataset: str, variant: str, args: argparse.Namespace,
                  build_root: Path) -> dict[str, Any]:
    scale = SCALES[args.scale][dataset]
    tables, rows, mess = GENERATORS[dataset](scale)
    local_variant_dir = build_root / args.suite_name / dataset / args.dataset_version / variant
    ensure_clean_dir(local_variant_dir, overwrite=True)
    remote_variant_root = join_uri(args.storage_root, args.suite_name, dataset,
                                   args.dataset_version, variant)
    if variant == "parquet_typed":
        return _finish_dataset(local_variant_dir, remote_variant_root, dataset,
                               variant, args.dataset_version, args.seed, tables,
                               mess, args.overwrite)
    return _finish_string_dataset(local_variant_dir, remote_variant_root,
                                  dataset, variant, args.dataset_version,
                                  args.seed, rows, mess, args.overwrite)


def main() -> int:
    args = _parse_args()
    print(f"[e2e-generate] storage_root={args.storage_root}", flush=True)
    datasets = DATASETS if args.dataset == "all" else (args.dataset, )
    variants = VARIANTS if args.variant == "all" else (args.variant, )
    build_root = Path(args.local_build_root)
    build_root.mkdir(parents=True, exist_ok=True)

    manifests = []
    for dataset in datasets:
        for variant in variants:
            print(f"[e2e-generate] generating {dataset} variant={variant}",
                  flush=True)
            manifest = _generate_one(dataset, variant, args, build_root)
            manifests.append({
                "dataset": dataset,
                "variant": variant,
                "remote_root": manifest["remote_root"],
                "tables": {
                    name: entry["rows"]
                    for name, entry in manifest["tables"].items()
                },
            })
            print(
                f"[e2e-generate] uploaded {dataset}/{variant} to {manifest['remote_root']}",
                flush=True)

    print(json.dumps({"generated": manifests}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
