#!/usr/bin/env python3
"""Generate diverse benchmark data for col-stats testing.

Schema covers ALL semantic types for comprehensive col-stats validation:
  - Id: string IDs (PK/FK)
  - Text: multi-word sentences (bio, description, address)
  - Categorical: low-cardinality strings, bools, small ints
  - Numerical: float32/float64, int32/int64
  - Timestamp: Arrow timestamp[us]
  - Sequence: List<Float32> embeddings

Category count benchmark (10 extra columns on items table):
  - 2 baseline: uniform 10, 100 unique (tiny HashMap, L1)
  - 8 stress: uniform/zipf × 4K/50K/500K/900K unique
  - Tests per-file HashMap from L1 (~320B) to way past L3 (~29MB)

Tables (3 node + 2 edge):
  accounts (11 cols), items (19 cols), locations (7 cols)
  transactions (account→item, 4 cols), visits (account→location, 4 cols)
  Edge tables are minimal: 2 FKs + timestamp + optional weight (no strings/categoricals)

Usage:
    python generate_diverse.py --rows-per-table 50000000 --output-dir /mnt/localssd/col-stats-bench/diverse_50m
    python generate_diverse.py --rows-per-table 100000000 --output-dir /mnt/localssd/col-stats-bench/diverse_100m
    python generate_diverse.py --rows-per-table 500000000 --output-dir /mnt/localssd/col-stats-bench/diverse_pct5 --skip-truth --no-embeddings
"""

import argparse
import json
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROWS_PER_FILE = 1_000_000

# ── Word pools for text generation ──────────────────────────────────────────

WORD_POOL = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "as", "you", "do", "at", "this", "but",
    "his", "by", "from", "they", "we", "say", "her", "she", "or", "an",
    "will", "my", "one", "all", "would", "there", "their", "what", "so",
    "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only",
    "come", "its", "over", "think", "also", "back", "after", "use",
    "two", "how", "our", "work", "first", "well", "way", "even", "new",
    "want", "because", "any", "these", "give", "day", "most", "find",
    "here", "thing", "many", "still", "between", "name", "should",
    "home", "big", "high", "each", "long", "world", "next", "early",
    "need", "great", "old", "right", "too", "mean", "large", "must",
    "place", "number", "such", "part", "state", "since", "very", "much",
    "may", "before", "real", "case", "last", "both", "same", "another",
    "begin", "while", "own", "point", "group", "small", "life", "where",
    "hand", "keep", "head", "house", "turn", "move", "run", "play",
    "start", "might", "show", "try", "ask", "off", "end", "line",
    "open", "seem", "together", "close", "order", "under", "call",
    "light", "water", "earth", "city", "tree", "cross", "farm",
    "hard", "south", "side", "change", "help", "fast", "book",
    "second", "school", "river", "room", "every", "story", "letter",
    "table", "class", "bring", "sure", "follow", "power", "learn",
    "young", "question", "important", "area", "money", "always",
    "family", "student", "body", "enough", "often", "information",
    "system", "program", "service", "company", "different", "public",
    "possible", "government", "business", "national", "development",
    "market", "country", "level", "process", "problem", "research",
    "report", "result", "social", "interest", "support", "general",
    "project", "provide", "quality", "product", "value", "price",
    "network", "field", "method", "review", "trade", "record",
    "plan", "design", "model", "example", "function", "standard",
    "customer", "account", "security", "available", "management",
    "technology", "computer", "science", "industry", "data", "material",
    "building", "experience", "control", "special", "position",
    "growth", "include", "current", "practice", "performance",
    "action", "center", "total", "community", "future", "analysis",
    "production", "common", "simple", "clear", "evidence", "financial",
    "international", "resource", "structure", "personal", "training",
    "complete", "increase", "recent", "traditional", "significant",
    "effective", "natural", "local", "original", "professional",
    "negative", "positive", "physical", "primary", "environmental",
    "quick", "slow", "bright", "dark", "tall", "short", "warm", "cold",
    "clean", "dirty", "smooth", "rough", "heavy", "sharp", "soft",
    "loud", "quiet", "sweet", "bitter", "fresh", "stale",
    "running", "jumping", "swimming", "reading", "writing", "driving",
    "cooking", "painting", "dancing", "singing", "working", "sleeping",
    "building", "creating", "solving", "testing", "shipping", "tracking",
]

STREET_TYPES = ["Street", "Avenue", "Boulevard", "Drive", "Lane", "Road", "Way", "Place", "Court"]
STREET_NAMES = [
    "Oak", "Maple", "Cedar", "Pine", "Elm", "Main", "Park", "Lake",
    "Hill", "River", "Spring", "Forest", "Valley", "Summit", "Ridge",
    "Harbor", "Bridge", "Canyon", "Meadow", "Garden", "Sunset",
]
DIRECTIONS = ["North", "South", "East", "West"]

# ── Categorical value pools ─────────────────────────────────────────────────

COUNTRIES = [f"country_{i:03d}" for i in range(200)]
CITIES = [f"city_{i:04d}" for i in range(1000)]
CATEGORIES = [f"cat_{i:04d}" for i in range(500)]
SUBCATEGORIES = [f"subcat_{i:05d}" for i in range(50_000)]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "wire",
                   "crypto", "cash", "gift_card", "apple_pay"]

# ── Category count benchmark columns ──────────────────────────────────────
# 10 columns for category count overhead benchmarking:
#   - 2 baseline columns: uniform with 10, 100 unique (tiny HashMaps, ~320B / ~3.2KB)
#   - 8 stress columns: 2 distributions (uniform, zipf) × 4 tiers (4K, 50K, 500K, 900K)
# These stress-test per-file HashMap sizes from L1 (tiny) to way past L3 (29MB).
CAT_TIERS = [
    ("4k", 4_000),
    ("50k", 50_000),
    ("500k", 500_000),
    ("900k", 900_000),
]
CAT_BASELINES = [
    ("10", 10),
    ("100", 100),
]

# ── Helpers ─────────────────────────────────────────────────────────────────


def make_string_ids(prefix, offset, n):
    """Generate unique string IDs like 'acc_00000004a3b7'."""
    ids = np.arange(offset, offset + n, dtype=np.uint64)
    return np.array([f"{prefix}{x:012x}" for x in ids])


def uniform_categorical(rng, pool, n):
    """Pick uniformly from a pool of strings."""
    indices = rng.integers(0, len(pool), size=n)
    return np.array([pool[i] for i in indices])


def gen_uniform_cat(rng, prefix, n_unique, n_rows):
    """Generate uniform categorical column without pre-built pool (memory efficient)."""
    width = max(len(str(n_unique - 1)), 1)
    indices = rng.integers(0, n_unique, size=n_rows)
    return np.array([f"{prefix}{i:0{width}d}" for i in indices])


def gen_zipf_cat(rng, prefix, n_unique, n_rows, alpha=1.3):
    """Generate Zipf-distributed categorical column (power-law: some values very common)."""
    width = max(len(str(n_unique - 1)), 1)
    indices = (rng.zipf(alpha, size=n_rows) - 1) % n_unique
    return np.array([f"{prefix}{i:0{width}d}" for i in indices])


def make_text_pool(rng, pool_size=50_000, min_words=5, max_words=20):
    """Pre-generate a pool of text sentences for fast sampling."""
    pool = []
    for _ in range(pool_size):
        num_words = int(rng.integers(min_words, max_words + 1))
        words = rng.choice(WORD_POOL, size=num_words)
        pool.append(" ".join(words))
    return pool


def sample_text(rng, pool, n):
    """Sample n text strings from a pre-built pool."""
    indices = rng.integers(0, len(pool), size=n)
    return [pool[i] for i in indices]


def make_address_pool(rng, pool_size=20_000):
    """Pre-generate a pool of street addresses (multi-word text)."""
    pool = []
    for _ in range(pool_size):
        number = int(rng.integers(1, 9999))
        direction = rng.choice(DIRECTIONS) + " " if rng.random() < 0.8 else ""
        street = rng.choice(STREET_NAMES)
        stype = rng.choice(STREET_TYPES)
        extra = ""
        r = rng.random()
        if r < 0.3:
            extra = f", Apt {int(rng.integers(1, 500))}"
        elif r < 0.45:
            extra = f", Suite {int(rng.integers(100, 999))}"
        elif r < 0.55:
            extra = f", Floor {int(rng.integers(1, 40))}"
        pool.append(f"{number} {direction}{street} {stype}{extra}")
    return pool


# ── Per-table generators ────────────────────────────────────────────────────


def generate_accounts_file(file_idx, rows_per_file, out_dir, total_accounts, no_embeddings):
    """accounts: Id, Text, Categorical, Numerical, Bool, Timestamp, Sequence."""
    rng = np.random.default_rng(seed=42 + file_idx)
    n = rows_per_file
    offset = file_idx * rows_per_file

    account_ids = make_string_ids("acc_", offset, n)
    text_pool = make_text_pool(rng, pool_size=50_000, min_words=5, max_words=20)
    bios = sample_text(rng, text_pool, n)
    username_indices = rng.integers(0, total_accounts, size=n, dtype=np.uint64)
    usernames = np.array([f"uname_{x:012x}" for x in username_indices])
    countries = uniform_categorical(rng, COUNTRIES, n)
    score = rng.uniform(0.0, 100.0, size=n)
    balance = rng.normal(5000.0, 2000.0, size=n)
    login_count = rng.integers(0, 500_000, size=n, dtype=np.int32)
    age = rng.integers(18, 80, size=n, dtype=np.int8)
    is_active = rng.choice([True, False], size=n)
    ts_start = 1577836800_000_000  # 2020-01-01 UTC micros
    ts_end = 1735689600_000_000    # 2025-01-01 UTC micros
    created_at = pa.array(
        rng.integers(ts_start, ts_end, size=n, dtype=np.int64),
        type=pa.timestamp("us"),
    )

    cols = {
        "account_id": pa.array(account_ids, type=pa.utf8()),
        "bio": pa.array(bios, type=pa.utf8()),
        "username": pa.array(usernames, type=pa.utf8()),
        "country": pa.array(countries, type=pa.utf8()),
        "score": pa.array(score, type=pa.float64()),
        "balance": pa.array(balance, type=pa.float64()),
        "login_count": pa.array(login_count, type=pa.int32()),
        "age": pa.array(age, type=pa.int8()),
        "is_active": pa.array(is_active, type=pa.bool_()),
        "created_at": created_at,
    }

    if not no_embeddings:
        embed_lengths = rng.integers(64, 256, size=n)
        offsets_arr = np.zeros(n + 1, dtype=np.int32)
        offsets_arr[1:] = np.cumsum(embed_lengths)
        total_floats = int(offsets_arr[-1])
        flat_values = rng.standard_normal(total_floats).astype(np.float32)
        embeddings = pa.ListArray.from_arrays(
            pa.array(offsets_arr, type=pa.int32()),
            pa.array(flat_values, type=pa.float32()),
        )
        cols["embedding"] = embeddings

    table = pa.table(cols)
    pq.write_table(table, os.path.join(out_dir, "accounts", f"part_{file_idx:05d}.parquet"))
    return ("accounts", file_idx)


def generate_items_file(file_idx, rows_per_file, out_dir, total_items):
    """items: Id, Text, Categorical, Numerical(f64+f32), Int16, Timestamp,
    plus 10 category-count benchmark columns (2 baseline + 8 stress)."""
    rng = np.random.default_rng(seed=10007 + file_idx)
    n = rows_per_file
    offset = file_idx * rows_per_file

    item_ids = make_string_ids("itm_", offset, n)
    text_pool = make_text_pool(rng, pool_size=50_000, min_words=8, max_words=30)
    descriptions = sample_text(rng, text_pool, n)
    categories = uniform_categorical(rng, CATEGORIES, n)
    subcategories = uniform_categorical(rng, SUBCATEGORIES, n)
    price = np.abs(rng.lognormal(3.0, 1.5, size=n))
    weight = np.abs(rng.exponential(2.0, size=n)).astype(np.float32)
    risk_score = rng.beta(2.0, 5.0, size=n).astype(np.float32)
    flag_count = rng.integers(0, 50, size=n, dtype=np.int16)
    ts_start = 1577836800_000_000
    ts_end = 1735689600_000_000
    listed_at = pa.array(
        rng.integers(ts_start, ts_end, size=n, dtype=np.int64),
        type=pa.timestamp("us"),
    )

    cols = {
        "item_id": pa.array(item_ids, type=pa.utf8()),
        "description": pa.array(descriptions, type=pa.utf8()),
        "category": pa.array(categories, type=pa.utf8()),
        "subcategory": pa.array(subcategories, type=pa.utf8()),
        "price": pa.array(price, type=pa.float64()),
        "weight": pa.array(weight, type=pa.float32()),
        "risk_score": pa.array(risk_score, type=pa.float32()),
        "flag_count": pa.array(flag_count, type=pa.int16()),
        "listed_at": listed_at,
    }

    # Baseline columns: uniform, small cardinality (tiny HashMaps)
    for label, n_unique in CAT_BASELINES:
        cols[f"cat_uniform_{label}"] = pa.array(
            gen_uniform_cat(rng, f"u{label}_", n_unique, n), type=pa.utf8())

    # Stress columns: uniform × 4 tiers + zipf × 4 tiers
    for label, n_unique in CAT_TIERS:
        cols[f"cat_uniform_{label}"] = pa.array(
            gen_uniform_cat(rng, f"u{label}_", n_unique, n), type=pa.utf8())
        cols[f"cat_zipf_{label}"] = pa.array(
            gen_zipf_cat(rng, f"z{label}_", n_unique, n), type=pa.utf8())

    table = pa.table(cols)
    pq.write_table(table, os.path.join(out_dir, "items", f"part_{file_idx:05d}.parquet"))
    return ("items", file_idx)


def generate_locations_file(file_idx, rows_per_file, out_dir, total_locations):
    """locations: Id, Text(address), Categorical, Numerical(f64+f32), Bool."""
    rng = np.random.default_rng(seed=20011 + file_idx)
    n = rows_per_file
    offset = file_idx * rows_per_file

    location_ids = make_string_ids("loc_", offset, n)
    addr_pool = make_address_pool(rng, pool_size=20_000)
    addresses = sample_text(rng, addr_pool, n)
    cities = uniform_categorical(rng, CITIES, n)
    lat = rng.uniform(-90.0, 90.0, size=n)
    lon = rng.uniform(-180.0, 180.0, size=n)
    elevation_m = np.abs(rng.exponential(500.0, size=n)).astype(np.float32)
    is_verified = rng.choice([True, False], size=n, p=[0.9, 0.1])

    table = pa.table({
        "location_id": pa.array(location_ids, type=pa.utf8()),
        "address": pa.array(addresses, type=pa.utf8()),
        "city": pa.array(cities, type=pa.utf8()),
        "lat": pa.array(lat, type=pa.float64()),
        "lon": pa.array(lon, type=pa.float64()),
        "elevation_m": pa.array(elevation_m, type=pa.float32()),
        "is_verified": pa.array(is_verified, type=pa.bool_()),
    })
    pq.write_table(table, os.path.join(out_dir, "locations", f"part_{file_idx:05d}.parquet"))
    return ("locations", file_idx)


def generate_transactions_file(file_idx, rows_per_file, out_dir, total_accounts, total_items):
    """transactions (edge: account→item): 2 FKs + timestamp + weight (minimal edge table)."""
    rng = np.random.default_rng(seed=30013 + file_idx)
    n = rows_per_file

    # FKs: Zipf for accounts (power-law: some users buy a lot), uniform for items
    acct_indices = (rng.zipf(1.3, size=n) - 1) % total_accounts
    account_ids = np.array([f"acc_{x:012x}" for x in acct_indices.astype(np.uint64)])
    item_indices = rng.integers(0, total_items, size=n, dtype=np.uint64)
    item_ids = np.array([f"itm_{x:012x}" for x in item_indices])

    ts_start = 1577836800_000_000
    ts_end = 1735689600_000_000
    tx_ts = pa.array(
        rng.integers(ts_start, ts_end, size=n, dtype=np.int64),
        type=pa.timestamp("us"),
    )
    amount = np.abs(rng.lognormal(3.0, 1.5, size=n))

    table = pa.table({
        "account_id": pa.array(account_ids, type=pa.utf8()),
        "item_id": pa.array(item_ids, type=pa.utf8()),
        "tx_ts": tx_ts,
        "amount": pa.array(amount, type=pa.float64()),
    })
    pq.write_table(table, os.path.join(out_dir, "transactions", f"part_{file_idx:05d}.parquet"))
    return ("transactions", file_idx)


def generate_visits_file(file_idx, rows_per_file, out_dir, total_accounts, total_locations):
    """visits (edge: account→location): FK, Timestamp, Numerical(int32)."""
    rng = np.random.default_rng(seed=40017 + file_idx)
    n = rows_per_file

    acct_indices = (rng.zipf(1.3, size=n) - 1) % total_accounts
    account_ids = np.array([f"acc_{x:012x}" for x in acct_indices.astype(np.uint64)])
    loc_indices = rng.integers(0, total_locations, size=n, dtype=np.uint64)
    location_ids = np.array([f"loc_{x:012x}" for x in loc_indices])

    ts_start = 1577836800_000_000
    ts_end = 1735689600_000_000
    visit_ts = pa.array(
        rng.integers(ts_start, ts_end, size=n, dtype=np.int64),
        type=pa.timestamp("us"),
    )
    duration_mins = rng.integers(1, 10_000, size=n, dtype=np.int32)

    table = pa.table({
        "account_id": pa.array(account_ids, type=pa.utf8()),
        "location_id": pa.array(location_ids, type=pa.utf8()),
        "visit_ts": visit_ts,
        "duration_mins": pa.array(duration_mins, type=pa.int32()),
    })
    pq.write_table(table, os.path.join(out_dir, "visits", f"part_{file_idx:05d}.parquet"))
    return ("visits", file_idx)


# ── Dispatch ────────────────────────────────────────────────────────────────


def _gen_wrapper(args):
    """Dispatch to the right generator function."""
    table_name, file_idx, rpf, out_dir, kwargs = args
    if table_name == "accounts":
        return generate_accounts_file(file_idx, rpf, out_dir,
                                      kwargs["total_accounts"], kwargs["no_embeddings"])
    elif table_name == "items":
        return generate_items_file(file_idx, rpf, out_dir, kwargs["total_items"])
    elif table_name == "locations":
        return generate_locations_file(file_idx, rpf, out_dir, kwargs["total_locations"])
    elif table_name == "transactions":
        return generate_transactions_file(file_idx, rpf, out_dir,
                                          kwargs["total_accounts"], kwargs["total_items"])
    elif table_name == "visits":
        return generate_visits_file(file_idx, rpf, out_dir,
                                    kwargs["total_accounts"], kwargs["total_locations"])


# ── Config generation ───────────────────────────────────────────────────────


def generate_config(out_dir, tables_info, num_shards=256, dataset_name="diverse_bench"):
    """Generate diskgraph_config.json."""
    tables = []

    for tname, num_files, columns, pk, fkeys, is_edge in tables_info:
        paths = [os.path.join(out_dir, tname, f"part_{i:05d}.parquet")
                 for i in range(num_files)]
        t = {
            "name": tname,
            "source": {"type": "parquet", "paths": paths},
            "columns": columns,
        }
        if pk:
            t["primary_key"] = pk
        if is_edge:
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

    config = {"dataset_name": dataset_name, "num_shards": num_shards, "tables": tables}
    config_path = os.path.join(out_dir, "diskgraph_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


# ── Ground truth ────────────────────────────────────────────────────────────


def compute_ground_truth(out_dir, has_embeddings=True):
    """Compute exact per-column statistics from generated parquet files."""
    import polars as pl
    from scipy import stats as sp_stats

    truth = {}

    # Semantic types must match what diskgraph will infer/assign.
    # Text columns use explicit semantic_type in config (inference can't reach Text
    # with default 10K sample since MAX_CATEGORIES=1M > max possible 10K unique).
    # Name heuristic: columns containing "id" → Id (contains_identifier check).
    table_schemas = {
        "accounts": {
            "account_id": "id",          # PK
            "bio": "text",               # explicit semantic_type in config
            "username": "categorical",   # high-card string, no "id" in name
            "country": "categorical",    # 200 values
            "score": "numerical",        # float64
            "balance": "numerical",      # float64
            "login_count": "numerical",  # int32, ~500K range → >5% unique
            "age": "categorical",        # int8, 62 values → <5% unique
            "is_active": "categorical",  # bool
            "created_at": "timestamp",   # Arrow timestamp
            **({"embedding": "sequence"} if has_embeddings else {}),
        },
        "items": {
            "item_id": "id",             # PK
            "description": "text",       # explicit semantic_type
            "category": "categorical",   # 500 values
            "subcategory": "categorical", # 50K values
            "price": "numerical",        # float64
            "weight": "numerical",       # float32
            "risk_score": "numerical",   # float32
            "flag_count": "categorical", # int16, 50 values → <5% unique
            "listed_at": "timestamp",    # Arrow timestamp
            # Category count benchmark columns (10 total)
            "cat_uniform_10": "categorical",    # 10 unique
            "cat_uniform_100": "categorical",   # 100 unique
            "cat_uniform_4k": "categorical",    # 4K unique
            "cat_uniform_50k": "categorical",   # 50K unique
            "cat_uniform_500k": "categorical",  # 500K unique
            "cat_uniform_900k": "categorical",  # 900K unique
            "cat_zipf_4k": "categorical",       # 4K unique
            "cat_zipf_50k": "categorical",      # 50K unique
            "cat_zipf_500k": "categorical",     # 500K unique
            "cat_zipf_900k": "categorical",     # 900K unique
        },
        "locations": {
            "location_id": "id",         # PK, also "id" in name
            "address": "text",           # explicit semantic_type
            "city": "categorical",       # 1000 values
            "lat": "numerical",          # float64
            "lon": "numerical",          # float64
            "elevation_m": "numerical",  # float32
            "is_verified": "categorical", # bool
        },
        "transactions": {
            "account_id": "id",          # FK + "id" in name
            "item_id": "id",             # FK + "id" in name
            "tx_ts": "timestamp",        # Arrow timestamp
            "amount": "numerical",       # float64 (edge weight)
        },
        "visits": {
            "account_id": "id",          # FK + "id" in name
            "location_id": "id",         # FK + "id" in name
            "visit_ts": "timestamp",     # Arrow timestamp
            "duration_mins": "numerical", # int32, 10K range → >5%
        },
    }

    for table_name, schema in table_schemas.items():
        table_dir = os.path.join(out_dir, table_name)
        parquet_files = sorted(
            os.path.join(table_dir, f)
            for f in os.listdir(table_dir) if f.endswith(".parquet")
        )
        if not parquet_files:
            print(f"  SKIP {table_name}: no parquet files")
            continue

        print(f"  {table_name}: scanning {len(parquet_files)} files...")
        t0 = time.time()

        lf = pl.scan_parquet(parquet_files)
        table_truth = {}

        for col_name, stype in schema.items():
            col_truth = {"semantic_type": stype}

            if stype == "numerical":
                series = lf.select(col_name).collect().to_series()
                arr = series.cast(pl.Float64).to_numpy()
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

            elif stype in ("categorical", "id"):
                series = lf.select(col_name).collect().to_series()
                col_truth["count"] = int(series.len() - series.null_count())
                col_truth["null_count"] = int(series.null_count())
                col_truth["num_unique"] = int(series.n_unique())

            elif stype == "timestamp":
                series = lf.select(col_name).collect().to_series()
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


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate diverse col-stats benchmark data")
    parser.add_argument("--rows-per-table", type=int, default=50_000_000,
                        help="Rows per table (default 50M)")
    parser.add_argument("--output-dir", type=str, default="/mnt/localssd/col-stats-bench/diverse_50m",
                        help="Output directory")
    parser.add_argument("--num-shards", type=int, default=256,
                        help="num_shards for diskgraph config")
    parser.add_argument("--workers", type=int, default=min(20, cpu_count()),
                        help="Parallel workers for generation")
    parser.add_argument("--skip-truth", action="store_true",
                        help="Skip ground truth computation (for large scale)")
    parser.add_argument("--truth-only", action="store_true",
                        help="Only compute ground truth")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip embedding column (saves ~32GB per 50M rows)")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Dataset name in config (default: derived from dir)")
    args = parser.parse_args()

    rows = args.rows_per_table
    out_dir = args.output_dir
    num_files = rows // ROWS_PER_FILE
    has_embeddings = not args.no_embeddings
    dataset_name = args.dataset_name or os.path.basename(out_dir)

    if args.truth_only:
        print(f"Computing ground truth only for {out_dir}")
        compute_ground_truth(out_dir, has_embeddings)
        return

    print(f"Generating {rows:,} rows/table, {num_files} files/table, output={out_dir}")
    if not has_embeddings:
        print("  (embeddings disabled)")

    # Create directories
    table_names = ["accounts", "items", "locations", "transactions", "visits"]
    for tname in table_names:
        os.makedirs(os.path.join(out_dir, tname), exist_ok=True)

    kwargs = {
        "total_accounts": rows,
        "total_items": rows,
        "total_locations": rows,
        "no_embeddings": args.no_embeddings,
    }

    # Build task list
    tasks = []
    for tname in table_names:
        for fi in range(num_files):
            tasks.append((tname, fi, ROWS_PER_FILE, out_dir, kwargs))

    total_tasks = len(tasks)
    print(f"Total files to generate: {total_tasks} ({num_files}/table x {len(table_names)} tables)")

    t0 = time.time()
    done = {t: 0 for t in table_names}
    with Pool(args.workers) as pool:
        for result in pool.imap_unordered(_gen_wrapper, tasks):
            tname, _ = result
            done[tname] += 1
            total_done = sum(done.values())
            if total_done % 50 == 0 or total_done == total_tasks:
                elapsed = time.time() - t0
                print(f"  [{elapsed:.0f}s] {total_done}/{total_tasks} files "
                      f"({', '.join(f'{k}:{v}/{num_files}' for k, v in done.items())})")

    elapsed = time.time() - t0
    print(f"Data generation complete in {elapsed:.1f}s")

    # ── Config ──────────────────────────────────────────────────────────────

    accounts_cols = [
        {"name": "account_id", "ingested_dtype": "string"},
        {"name": "bio", "ingested_dtype": "string", "semantic_type": "text"},
        {"name": "username", "ingested_dtype": "string"},
        {"name": "country", "ingested_dtype": "string"},
        {"name": "score", "ingested_dtype": "float64"},
        {"name": "balance", "ingested_dtype": "float64"},
        {"name": "login_count", "ingested_dtype": "int32"},
        {"name": "age", "ingested_dtype": "int8"},
        {"name": "is_active", "ingested_dtype": "bool"},
        {"name": "created_at", "ingested_dtype": "timestamp"},
    ]
    if has_embeddings:
        accounts_cols.append({"name": "embedding", "ingested_dtype": "float32_array"})

    items_cols = [
        {"name": "item_id", "ingested_dtype": "string"},
        {"name": "description", "ingested_dtype": "string", "semantic_type": "text"},
        {"name": "category", "ingested_dtype": "string"},
        {"name": "subcategory", "ingested_dtype": "string"},
        {"name": "price", "ingested_dtype": "float64"},
        {"name": "weight", "ingested_dtype": "float32"},
        {"name": "risk_score", "ingested_dtype": "float32"},
        {"name": "flag_count", "ingested_dtype": "int16"},
        {"name": "listed_at", "ingested_dtype": "timestamp"},
    ]
    # Category count benchmark columns (10 total)
    for label, _ in CAT_BASELINES:
        items_cols.append({"name": f"cat_uniform_{label}", "ingested_dtype": "string"})
    for label, _ in CAT_TIERS:
        items_cols.append({"name": f"cat_uniform_{label}", "ingested_dtype": "string"})
        items_cols.append({"name": f"cat_zipf_{label}", "ingested_dtype": "string"})

    locations_cols = [
        {"name": "location_id", "ingested_dtype": "string"},
        {"name": "address", "ingested_dtype": "string", "semantic_type": "text"},
        {"name": "city", "ingested_dtype": "string"},
        {"name": "lat", "ingested_dtype": "float64"},
        {"name": "lon", "ingested_dtype": "float64"},
        {"name": "elevation_m", "ingested_dtype": "float32"},
        {"name": "is_verified", "ingested_dtype": "bool"},
    ]

    transactions_cols = [
        {"name": "account_id", "ingested_dtype": "string"},
        {"name": "item_id", "ingested_dtype": "string"},
        {"name": "tx_ts", "ingested_dtype": "timestamp"},
        {"name": "amount", "ingested_dtype": "float64"},
    ]

    visits_cols = [
        {"name": "account_id", "ingested_dtype": "string"},
        {"name": "location_id", "ingested_dtype": "string"},
        {"name": "visit_ts", "ingested_dtype": "timestamp"},
        {"name": "duration_mins", "ingested_dtype": "int32"},
    ]

    tables_info = [
        ("accounts", num_files, accounts_cols, "account_id", [], False),
        ("items", num_files, items_cols, "item_id", [], False),
        ("locations", num_files, locations_cols, "location_id", [], False),
        ("transactions", num_files, transactions_cols, None,
         [{"src_column": "account_id", "dst_table": "accounts"},
          {"src_column": "item_id", "dst_table": "items"}],
         True),
        ("visits", num_files, visits_cols, None,
         [{"src_column": "account_id", "dst_table": "accounts"},
          {"src_column": "location_id", "dst_table": "locations"}],
         True),
    ]

    config_path = generate_config(out_dir, tables_info, args.num_shards, dataset_name)
    print(f"Config written to {config_path}")

    # Summary
    total_files = num_files * len(table_names)
    print(f"\nSummary:")
    print(f"  Tables: {', '.join(table_names)}")
    print(f"  Rows/table: {rows:,}")
    print(f"  Files/table: {num_files}")
    print(f"  Total files: {total_files}")
    print(f"  Embeddings: {'yes' if has_embeddings else 'no'}")

    # Semantic type coverage:
    print(f"\n  Semantic types covered:")
    print(f"    Id:          account_id, item_id, location_id, FKs")
    print(f"    Text:        bio, description, address (explicit semantic_type)")
    print(f"    Categorical: country, city, category, subcategory, username,")
    print(f"                 age(int8), flag_count(int16), quantity(int8),")
    print(f"                 payment_method, is_active(bool), is_verified(bool)")
    print(f"    Numerical:   score(f64), balance(f64), login_count(i32),")
    print(f"                 price(f64), weight(f32), risk_score(f32),")
    print(f"                 lat(f64), lon(f64), elevation_m(f32),")
    print(f"                 amount(f64), duration_mins(i32)")
    print(f"    Timestamp:   created_at, listed_at, tx_ts, visit_ts")
    if has_embeddings:
        print(f"    Sequence:    embedding (list<f32>, 64-256 elements)")
    print(f"\n  Category count benchmark columns (on items table):")
    print(f"    Baseline:    cat_uniform_10 (10 unique), cat_uniform_100 (100 unique)")
    print(f"    Uniform:     cat_uniform_{{4k,50k,500k,900k}}")
    print(f"    Zipf(1.3):   cat_zipf_{{4k,50k,500k,900k}}")

    if not args.skip_truth:
        print(f"\nComputing ground truth stats...")
        compute_ground_truth(out_dir, has_embeddings)


if __name__ == "__main__":
    main()
