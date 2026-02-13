#!/usr/bin/env python3
"""Generate synthetic benchmark data for col-stats benchmarking.

Schema mirrors alt_detection structure with diverse column types:
- Node tables: users (8 cols), products (7 cols), devices (5 cols)
- Edge table: purchases (user→product, 5 cols)

Column types covered:
  string (various cardinalities), int8, int16, int32, int64,
  float32, float64, bool, timestamp (int64 micros)

Usage:
    python generate.py [--rows-per-table 100000000] [--output-dir /mnt/localssd/col-stats-bench]
"""

import argparse
import json
import math
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROWS_PER_FILE = 1_000_000  # 1M rows per file

# Categorical value pools (uniform distribution)
COUNTRIES = [f"country_{i:03d}" for i in range(200)]
STATUSES = ["active", "inactive", "pending", "banned", "deleted"]
CATEGORIES = [f"cat_{i:04d}" for i in range(500)]
BRANDS = [f"brand_{i:04d}" for i in range(5000)]
DEVICE_TYPES = [f"dtype_{i:02d}" for i in range(50)]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "wire", "crypto", "cash", "gift_card", "apple_pay"]


def make_string_ids(prefix, offset, n):
    """Generate unique string IDs like 'usr_00000004a3b7'."""
    ids = np.arange(offset, offset + n, dtype=np.uint64)
    return np.array([f"{prefix}{x:012x}" for x in ids])


def uniform_categorical(rng, pool, n):
    """Pick uniformly from a pool of strings."""
    indices = rng.integers(0, len(pool), size=n)
    return np.array([pool[i] for i in indices])


def make_high_card_strings(rng, prefix, n, cardinality):
    """Generate categorical strings with given cardinality, uniform distribution."""
    indices = rng.integers(0, cardinality, size=n)
    return np.array([f"{prefix}{i:08x}" for i in indices])


def generate_users_file(file_idx, rows_per_file, out_dir, total_users):
    """Generate one parquet file for the users table."""
    rng = np.random.default_rng(seed=42 + file_idx)
    n = rows_per_file
    offset = file_idx * rows_per_file

    user_ids = make_string_ids("usr_", offset, n)
    usernames = make_high_card_strings(rng, "uname_", n, total_users)  # high cardinality
    countries = uniform_categorical(rng, COUNTRIES, n)
    statuses = uniform_categorical(rng, STATUSES, n)
    age = rng.integers(18, 80, size=n, dtype=np.int32)         # 62 unique → categorical
    signup_year = rng.integers(2000, 2025, size=n, dtype=np.int32).astype(np.int16)  # 25 unique → categorical
    balance = rng.normal(5000.0, 2000.0, size=n)
    score = rng.uniform(0.0, 100.0, size=n).astype(np.float32)
    is_active = rng.choice([True, False], size=n)
    # High-cardinality int32 — stays numerical (>5% unique fraction in sample)
    login_count = rng.integers(0, 500_000, size=n, dtype=np.int32)
    # Timestamps as int64 microseconds (2020-01-01 to 2024-12-31)
    ts_start = 1577836800_000_000  # 2020-01-01 UTC in micros
    ts_end = 1735689600_000_000    # 2025-01-01 UTC in micros
    created_at = rng.integers(ts_start, ts_end, size=n, dtype=np.int64)
    # Arrow Timestamp column (tests observe_timestamp path)
    created_at_ts = pa.array(created_at, type=pa.timestamp("us"))

    table = pa.table({
        "user_id": pa.array(user_ids, type=pa.utf8()),
        "username": pa.array(usernames, type=pa.utf8()),
        "country": pa.array(countries, type=pa.utf8()),
        "status": pa.array(statuses, type=pa.utf8()),
        "age": pa.array(age, type=pa.int32()),
        "signup_year": pa.array(signup_year, type=pa.int16()),
        "balance": pa.array(balance, type=pa.float64()),
        "score": pa.array(score, type=pa.float32()),
        "is_active": pa.array(is_active, type=pa.bool_()),
        "login_count": pa.array(login_count, type=pa.int32()),
        "created_at": pa.array(created_at, type=pa.int64()),
        "created_at_ts": created_at_ts,
    })
    path = os.path.join(out_dir, "users", f"part_{file_idx:05d}.parquet")
    pq.write_table(table, path)
    return ("users", file_idx)


def generate_products_file(file_idx, rows_per_file, out_dir, total_products):
    """Generate one parquet file for the products table."""
    rng = np.random.default_rng(seed=10007 + file_idx)
    n = rows_per_file
    offset = file_idx * rows_per_file

    product_ids = make_string_ids("prd_", offset, n)
    categories = uniform_categorical(rng, CATEGORIES, n)
    subcategories = make_high_card_strings(rng, "subcat_", n, 50_000)
    brands = uniform_categorical(rng, BRANDS, n)
    price = np.abs(rng.lognormal(3.0, 1.5, size=n))  # lognormal prices
    weight_kg = np.abs(rng.exponential(2.0, size=n)).astype(np.float32)
    in_stock = rng.choice([True, False], size=n, p=[0.85, 0.15])

    table = pa.table({
        "product_id": pa.array(product_ids, type=pa.utf8()),
        "category": pa.array(categories, type=pa.utf8()),
        "subcategory": pa.array(subcategories, type=pa.utf8()),
        "brand": pa.array(brands, type=pa.utf8()),
        "price": pa.array(price, type=pa.float64()),
        "weight_kg": pa.array(weight_kg, type=pa.float32()),
        "in_stock": pa.array(in_stock, type=pa.bool_()),
    })
    path = os.path.join(out_dir, "products", f"part_{file_idx:05d}.parquet")
    pq.write_table(table, path)
    return ("products", file_idx)


def generate_devices_file(file_idx, rows_per_file, out_dir, total_devices, total_users):
    """Generate one parquet file for the devices table."""
    rng = np.random.default_rng(seed=20011 + file_idx)
    n = rows_per_file
    offset = file_idx * rows_per_file

    device_ids = make_string_ids("dev_", offset, n)
    device_types = uniform_categorical(rng, DEVICE_TYPES, n)
    os_version = make_high_card_strings(rng, "os_", n, 200)
    screen_size = rng.uniform(4.0, 32.0, size=n).astype(np.float32)
    battery_mah = rng.integers(1000, 10000, size=n, dtype=np.int32)
    # High-cardinality int64 — stays numerical (>5% unique fraction)
    usage_minutes = rng.integers(0, 100_000_000, size=n, dtype=np.int64)
    # FK to users — each device belongs to a user (many-to-one)
    owner_indices = rng.integers(0, total_users, size=n, dtype=np.uint64)
    owner_ids = np.array([f"usr_{x:012x}" for x in owner_indices])

    # Embeddings: List<Float32> with variable lengths (tests observe_float_list path)
    embed_lengths = rng.integers(64, 256, size=n)
    offsets = np.zeros(n + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(embed_lengths)
    total_floats = int(offsets[-1])
    flat_values = rng.standard_normal(total_floats).astype(np.float32)
    embeddings = pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()),
        pa.array(flat_values, type=pa.float32()),
    )

    table = pa.table({
        "device_id": pa.array(device_ids, type=pa.utf8()),
        "device_type": pa.array(device_types, type=pa.utf8()),
        "os_version": pa.array(os_version, type=pa.utf8()),
        "screen_size": pa.array(screen_size, type=pa.float32()),
        "battery_mah": pa.array(battery_mah, type=pa.int32()),
        "usage_minutes": pa.array(usage_minutes, type=pa.int64()),
        "owner_id": pa.array(owner_ids, type=pa.utf8()),
        "embeddings": embeddings,
    })
    path = os.path.join(out_dir, "devices", f"part_{file_idx:05d}.parquet")
    pq.write_table(table, path)
    return ("devices", file_idx)


def generate_purchases_file(file_idx, rows_per_file, out_dir, total_users, total_products):
    """Generate one parquet file for the purchases edge table."""
    rng = np.random.default_rng(seed=30013 + file_idx)
    n = rows_per_file

    # FK references (Zipf-like for users, uniform for products)
    user_indices = (rng.zipf(1.3, size=n) - 1) % total_users
    user_ids = np.array([f"usr_{x:012x}" for x in user_indices.astype(np.uint64)])
    product_indices = rng.integers(0, total_products, size=n, dtype=np.uint64)
    product_ids = np.array([f"prd_{x:012x}" for x in product_indices])

    amount = np.abs(rng.lognormal(3.0, 1.5, size=n))
    quantity = rng.integers(1, 20, size=n, dtype=np.int32)
    payment = uniform_categorical(rng, PAYMENT_METHODS, n)
    ts_start = 1577836800_000_000
    ts_end = 1735689600_000_000
    purchase_ts = rng.integers(ts_start, ts_end, size=n, dtype=np.int64)

    table = pa.table({
        "user_id": pa.array(user_ids, type=pa.utf8()),
        "product_id": pa.array(product_ids, type=pa.utf8()),
        "amount": pa.array(amount, type=pa.float64()),
        "quantity": pa.array(quantity, type=pa.int32()),
        "payment_method": pa.array(payment, type=pa.utf8()),
        "purchase_ts": pa.array(purchase_ts, type=pa.int64()),
    })
    path = os.path.join(out_dir, "purchases", f"part_{file_idx:05d}.parquet")
    pq.write_table(table, path)
    return ("purchases", file_idx)


def _gen_wrapper(args):
    """Dispatch to the right generator function."""
    table_name, file_idx, rpf, out_dir, kwargs = args
    if table_name == "users":
        return generate_users_file(file_idx, rpf, out_dir, kwargs["total_users"])
    elif table_name == "products":
        return generate_products_file(file_idx, rpf, out_dir, kwargs["total_products"])
    elif table_name == "devices":
        return generate_devices_file(file_idx, rpf, out_dir, kwargs["total_devices"], kwargs["total_users"])
    elif table_name == "purchases":
        return generate_purchases_file(file_idx, rpf, out_dir, kwargs["total_users"], kwargs["total_products"])


def generate_config(out_dir, tables_info, num_shards=256):
    """Generate diskgraph_config.json."""
    tables = []

    for tname, num_files, columns, pk, fkeys, is_edge in tables_info:
        paths = [os.path.join(out_dir, tname, f"part_{i:05d}.parquet") for i in range(num_files)]
        t = {
            "name": tname,
            "source": {"type": "parquet", "paths": paths},
            "columns": columns,
        }
        if pk:
            t["primary_key"] = pk
        if is_edge:
            # Edge tables define FK relationships via edge_table, not fkeys.
            t["fkeys"] = []
            t["edge_table"] = {
                "src_fk": {
                    "edge_column": fkeys[0]["src_column"],
                    "ref_table": fkeys[0]["dst_table"],
                },
                "dst_fk": {
                    "edge_column": fkeys[1]["src_column"],
                    "ref_table": fkeys[1]["dst_table"],
                },
            }
        else:
            t["fkeys"] = fkeys
        tables.append(t)

    config = {"dataset_name": "col_stats_bench", "num_shards": num_shards, "tables": tables}
    config_path = os.path.join(out_dir, "diskgraph_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Generate col-stats benchmark data")
    parser.add_argument("--rows-per-table", type=int, default=100_000_000,
                        help="Rows per table (default 100M)")
    parser.add_argument("--output-dir", type=str, default="/mnt/localssd/col-stats-bench",
                        help="Output directory")
    parser.add_argument("--num-shards", type=int, default=256,
                        help="num_shards for diskgraph config")
    parser.add_argument("--workers", type=int, default=min(20, cpu_count()),
                        help="Parallel workers for generation")
    parser.add_argument("--skip-truth", action="store_true",
                        help="Skip ground truth computation")
    parser.add_argument("--truth-only", action="store_true",
                        help="Only compute ground truth (skip data generation)")
    args = parser.parse_args()

    rows = args.rows_per_table
    out_dir = args.output_dir
    num_files = rows // ROWS_PER_FILE

    if args.truth_only:
        print(f"Computing ground truth only for {out_dir}")
        compute_ground_truth(out_dir)
        return

    print(f"Generating {rows:,} rows/table, {num_files} files/table, output={out_dir}")

    # Create directories
    for tname in ["users", "products", "devices", "purchases"]:
        os.makedirs(os.path.join(out_dir, tname), exist_ok=True)

    kwargs = {
        "total_users": rows,
        "total_products": rows,
        "total_devices": rows,
    }

    # Build task list
    tasks = []
    for tname in ["users", "products", "devices", "purchases"]:
        for fi in range(num_files):
            tasks.append((tname, fi, ROWS_PER_FILE, out_dir, kwargs))

    print(f"Total files to generate: {len(tasks)} ({num_files} per table x 4 tables)")

    t0 = time.time()
    done = {t: 0 for t in ["users", "products", "devices", "purchases"]}
    with Pool(args.workers) as pool:
        for result in pool.imap_unordered(_gen_wrapper, tasks):
            tname, _ = result
            done[tname] += 1
            total_done = sum(done.values())
            if total_done % 50 == 0 or total_done == len(tasks):
                elapsed = time.time() - t0
                print(f"  [{elapsed:.0f}s] {total_done}/{len(tasks)} files "
                      f"({', '.join(f'{k}:{v}/{num_files}' for k,v in done.items())})")

    elapsed = time.time() - t0
    print(f"Data generation complete in {elapsed:.1f}s")

    # Generate config
    users_cols = [
        {"name": "user_id", "ingested_dtype": "string"},
        {"name": "username", "ingested_dtype": "string"},
        {"name": "country", "ingested_dtype": "string"},
        {"name": "status", "ingested_dtype": "string"},
        {"name": "age", "ingested_dtype": "int32"},
        {"name": "signup_year", "ingested_dtype": "int16"},
        {"name": "balance", "ingested_dtype": "float64"},
        {"name": "score", "ingested_dtype": "float32"},
        {"name": "is_active", "ingested_dtype": "bool"},
        {"name": "login_count", "ingested_dtype": "int32"},
        {"name": "created_at", "ingested_dtype": "int64"},
        {"name": "created_at_ts", "ingested_dtype": "timestamp"},
    ]
    products_cols = [
        {"name": "product_id", "ingested_dtype": "string"},
        {"name": "category", "ingested_dtype": "string"},
        {"name": "subcategory", "ingested_dtype": "string"},
        {"name": "brand", "ingested_dtype": "string"},
        {"name": "price", "ingested_dtype": "float64"},
        {"name": "weight_kg", "ingested_dtype": "float32"},
        {"name": "in_stock", "ingested_dtype": "bool"},
    ]
    devices_cols = [
        {"name": "device_id", "ingested_dtype": "string"},
        {"name": "device_type", "ingested_dtype": "string"},
        {"name": "os_version", "ingested_dtype": "string"},
        {"name": "screen_size", "ingested_dtype": "float32"},
        {"name": "battery_mah", "ingested_dtype": "int32"},
        {"name": "usage_minutes", "ingested_dtype": "int64"},
        {"name": "owner_id", "ingested_dtype": "string"},
        {"name": "embeddings", "ingested_dtype": "float32_array"},
    ]
    purchases_cols = [
        {"name": "user_id", "ingested_dtype": "string"},
        {"name": "product_id", "ingested_dtype": "string"},
        {"name": "amount", "ingested_dtype": "float64"},
        {"name": "quantity", "ingested_dtype": "int32"},
        {"name": "payment_method", "ingested_dtype": "string"},
        {"name": "purchase_ts", "ingested_dtype": "int64"},
    ]

    tables_info = [
        ("users", num_files, users_cols, "user_id", [], False),
        ("products", num_files, products_cols, "product_id", [], False),
        ("devices", num_files, devices_cols, "device_id",
         [{"src_column": "owner_id", "dst_table": "users"}], False),
        ("purchases", num_files, purchases_cols, None,
         [{"src_column": "user_id", "dst_table": "users"},
          {"src_column": "product_id", "dst_table": "products"}],
         True),
    ]

    config_path = generate_config(out_dir, tables_info, args.num_shards)
    print(f"Config written to {config_path}")

    # Print summary
    total_files = num_files * 4
    print(f"\nSummary:")
    print(f"  Tables: users, products, devices, purchases (edge)")
    print(f"  Rows/table: {rows:,}")
    print(f"  Files/table: {num_files}")
    print(f"  Total files: {total_files}")
    print(f"  num_shards: {args.num_shards}")

    if not args.skip_truth:
        print(f"\nComputing ground truth stats...")
        compute_ground_truth(out_dir)


def compute_ground_truth(out_dir):
    """Compute exact per-column statistics from the generated parquet files.

    Uses polars for memory-efficient lazy scanning. Writes ground_truth.json
    alongside the data. This is the source of truth for verifying col-stats
    accuracy.
    """
    import polars as pl
    from scipy import stats as sp_stats

    truth = {}

    # Define which columns get which stats, matching SemanticType assignment
    # from infer_semantic_type(). PK columns → Id, booleans → Categorical, etc.
    # Semantic types must match infer_semantic_type() output.
    # Key thresholds: num_unique ≤ 3 → Categorical, ≤ 1M and int with
    # unique_fraction ≤ 5% → Categorical, > 5% → Numerical.
    # Context phase samples 10K rows; unique_fraction = num_unique / 10K.
    table_schemas = {
        "users": {
            "user_id": "id",               # PK → Id
            "username": "categorical",      # high-card string, inferred categorical
            "country": "categorical",       # 200 unique strings
            "status": "categorical",        # 5 unique strings
            "age": "categorical",           # i32, 62 unique → 0.62% unique frac → categorical
            "signup_year": "categorical",   # i16, 25 unique → 0.25% unique frac → categorical
            "balance": "numerical",         # f64 → always numerical
            "score": "numerical",           # f32 → always numerical
            "is_active": "categorical",     # bool → categorical
            "login_count": "numerical",     # i32, 500K range → >5% unique frac → numerical
            "created_at": "numerical",      # i64, huge range → numerical
            "created_at_ts": "timestamp",   # Arrow Timestamp → timestamp
        },
        "products": {
            "product_id": "id",
            "category": "categorical",
            "subcategory": "categorical",
            "brand": "categorical",
            "price": "numerical",           # f64 → always numerical
            "weight_kg": "numerical",       # f32 → always numerical
            "in_stock": "categorical",      # bool → categorical
        },
        "devices": {
            "device_id": "id",
            "device_type": "categorical",
            "os_version": "categorical",
            "screen_size": "numerical",     # f32 → always numerical
            "battery_mah": "numerical",     # i32, 9000 range → >5% unique frac → numerical
            "usage_minutes": "numerical",   # i64, 100M range → >5% unique frac → numerical
            "owner_id": "id",              # FK to users → Id
            "embeddings": "sequence",
        },
        "purchases": {
            "user_id": "id",
            "product_id": "id",
            "amount": "numerical",
            "quantity": "categorical",       # i32, 19 unique → 0.19% unique frac → categorical
            "payment_method": "categorical",
            "purchase_ts": "numerical",
        },
    }

    for table_name, schema in table_schemas.items():
        table_dir = os.path.join(out_dir, table_name)
        parquet_files = sorted(
            os.path.join(table_dir, f) for f in os.listdir(table_dir) if f.endswith(".parquet")
        )
        if not parquet_files:
            print(f"  SKIP {table_name}: no parquet files")
            continue

        print(f"  {table_name}: scanning {len(parquet_files)} files...")
        t0 = time.time()

        # Read all files into a single lazy frame
        lf = pl.scan_parquet(parquet_files)
        # Collect just the columns we need (avoid loading everything at once for
        # sequence columns which expand list data)
        table_truth = {}

        for col_name, stype in schema.items():
            col_truth = {"semantic_type": stype}

            if stype == "numerical":
                # Collect the column as a Series for scipy stats
                series = lf.select(col_name).collect().to_series()
                arr = series.cast(pl.Float64).to_numpy()
                # Filter out NaN/Inf for consistency with Welford which skips them
                mask = np.isfinite(arr)
                clean = arr[mask]

                col_truth["count"] = int(len(clean))
                col_truth["null_count"] = int(series.null_count())
                col_truth["mean"] = float(np.mean(clean))
                col_truth["std"] = float(np.std(clean, ddof=1))
                col_truth["min"] = float(np.min(clean))
                col_truth["max"] = float(np.max(clean))
                col_truth["skewness"] = float(sp_stats.skew(clean, bias=False))
                col_truth["kurtosis"] = float(sp_stats.kurtosis(clean, fisher=True, bias=False))
                col_truth["q25"] = float(np.quantile(clean, 0.25))
                col_truth["q50"] = float(np.quantile(clean, 0.50))
                col_truth["q75"] = float(np.quantile(clean, 0.75))
                col_truth["num_unique"] = int(series.n_unique())

            elif stype == "categorical":
                series = lf.select(col_name).collect().to_series()
                col_truth["count"] = int(series.len() - series.null_count())
                col_truth["null_count"] = int(series.null_count())
                col_truth["num_unique"] = int(series.n_unique())

            elif stype == "id":
                series = lf.select(col_name).collect().to_series()
                col_truth["count"] = int(series.len() - series.null_count())
                col_truth["null_count"] = int(series.null_count())
                col_truth["num_unique"] = int(series.n_unique())

            elif stype == "timestamp":
                # Arrow timestamps — compute min/max as (secs, nanos)
                series = lf.select(col_name).collect().to_series()
                # Cast to microseconds then to int for precise computation
                micros = series.cast(pl.Int64)
                min_us = int(micros.min())
                max_us = int(micros.max())
                col_truth["count"] = int(series.len() - series.null_count())
                col_truth["null_count"] = int(series.null_count())
                col_truth["min_secs"] = min_us // 1_000_000
                col_truth["min_nanos"] = (min_us % 1_000_000) * 1000
                col_truth["max_secs"] = max_us // 1_000_000
                col_truth["max_nanos"] = (max_us % 1_000_000) * 1000

            elif stype == "sequence":
                # List<Float32> — compute min/max lengths
                series = lf.select(col_name).collect().to_series()
                lengths = series.list.len()
                col_truth["count"] = int(series.len() - series.null_count())
                col_truth["null_count"] = int(series.null_count())
                col_truth["min_len"] = int(lengths.min())
                col_truth["max_len"] = int(lengths.max())

            elif stype == "text":
                series = lf.select(col_name).collect().to_series()
                col_truth["count"] = int(series.len() - series.null_count())
                col_truth["null_count"] = int(series.null_count())
                col_truth["num_unique"] = int(series.n_unique())
                # avg_tokens: average (spaces + 1) per string
                # non_char_fraction: fraction of non-alphabetic bytes
                # We scan first 512 bytes of each string to match col-stats behavior
                strs = series.drop_nulls().to_list()
                total_tokens = 0
                total_non_alpha = 0
                total_scanned = 0
                for s in strs:
                    scan = s[:512]
                    total_tokens += scan.count(" ") + 1
                    total_non_alpha += sum(1 for c in scan if not c.isalpha())
                    total_scanned += len(scan)
                col_truth["avg_tokens"] = total_tokens / len(strs) if strs else 0.0
                col_truth["non_char_fraction"] = (
                    total_non_alpha / total_scanned if total_scanned > 0 else 0.0
                )

            table_truth[col_name] = col_truth

        truth[table_name] = table_truth
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.1f}s")

    truth_path = os.path.join(out_dir, "ground_truth.json")
    with open(truth_path, "w") as f:
        json.dump(truth, f, indent=2)
    print(f"Ground truth written to {truth_path}")


if __name__ == "__main__":
    main()
