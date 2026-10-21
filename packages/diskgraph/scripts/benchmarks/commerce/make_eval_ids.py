# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared evaluation ID files for a rung. The 50B rung ships them in <rung>/_eval/; smaller rungs need this.

    python make_eval_ids.py --root <rung dir or s3:// prefix> --out <rung dir>/_eval \
        [--score-customers 5000000] [--test-seeds 1000000]

Writes <out>/predict_ids_5M.parquet (customer_id), <out>/test_seeds.parquet (customer_id, anchor, target)
and <out>/eval_ids.json. Rules:
- T = largest timestamp in any table (from Parquet footer statistics); test anchor a_1 = T - 7 days.
- predict IDs: customers with signup_time <= T, the first N by the portable hash h(customer_id, 0, 4).
- test seeds: customers with signup_time <= a_1 and at least one purchase in (a_1, T]; the first N by
  h(customer_id, 1, 3); target = their distinct merchant_ids in (a_1, T].
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq



class H:
    """Constants of the commerce schema and the evaluation split."""
    P31 = 2147483647
    SALT = {"train": 1, "val": 2, "test": 3, "score": 4}
    WEEK_MS = 7 * 86400 * 1000
    TABLES = ["customers", "merchants", "purchases", "merchant_views", "searches", "favorites", "reviews"]
    TIME_COL = {"customers": "signup_time", "merchants": "created_time", "purchases": "purchase_time",
                "merchant_views": "view_time", "searches": "search_time", "favorites": "created_time",
                "reviews": "review_time"}


def fs_and_base(root):
    if root.startswith("s3://"):
        return pafs.S3FileSystem(), root[5:].rstrip("/")
    return pafs.LocalFileSystem(), str(Path(root).resolve())


def files(root, table):
    fs, base = fs_and_base(root)
    return sorted(f.path for f in fs.get_file_info(pafs.FileSelector(f"{base}/{table}")) if f.path.endswith(".parquet"))


def np_hash(c, k, s):
    P = np.int64(H.P31)
    x = (c.astype(np.int64) * 1000003 + k * 7919 + s) % P
    x = ((x ^ (x >> 15)) * np.int64(2246822519)) % P
    x = ((x ^ (x >> 13)) * np.int64(3266489917)) % P
    return x ^ (x >> 16)


def time_range(task):
    root, path, col = task
    fs, _ = fs_and_base(root)
    md = pq.ParquetFile(path, filesystem=fs).metadata
    i = md.schema.to_arrow_schema().get_field_index(col)
    st = [md.row_group(g).column(i).statistics for g in range(md.num_row_groups)]
    to_ms = lambda v: int(pa.scalar(v, pa.timestamp("ms")).value)
    return min(to_ms(s.min) for s in st), max(to_ms(s.max) for s in st)


def top_customers(task):
    root, path, T, n = task
    fs, _ = fs_and_base(root)
    t = pq.read_table(path, filesystem=fs, columns=["customer_id", "signup_time"])
    c = t["customer_id"].to_numpy()[pc.less_equal(pc.cast(t["signup_time"], pa.int64()), T).to_numpy(zero_copy_only=False)]
    h = np_hash(c, 0, H.SALT["score"])
    if len(c) > n:
        idx = np.argpartition(h, n - 1)[:n]
        c, h = c[idx], h[idx]
    return c, h, len(t)


def last_week(task):
    root, path, lo = task
    fs, _ = fs_and_base(root)
    out = []
    pf = pq.ParquetFile(path, filesystem=fs)
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=["customer_id", "merchant_id", "purchase_time"])
        t = t.filter(pc.greater(pc.cast(t["purchase_time"], pa.int64()), lo))
        out.append(t.select(["customer_id", "merchant_id"]))
    return pa.concat_tables(out)


def signups(task):
    root, path, ids = task
    fs, _ = fs_and_base(root)
    t = pq.read_table(path, filesystem=fs, columns=["customer_id", "signup_time"])
    return t.filter(pc.is_in(t["customer_id"], ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score-customers", type=int, default=5_000_000)
    ap.add_argument("--test-seeds", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if min(a.score_customers, a.test_seeds, a.workers) < 1:
        ap.error("score-customers, test-seeds and workers must be positive")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        tasks = [(a.root, f, H.TIME_COL[t]) for t in H.TABLES for f in files(a.root, t)]
        rng = list(ex.map(time_range, tasks))
        T, T0 = max(r[1] for r in rng), min(r[0] for r in rng)
        a1 = T - H.WEEK_MS
        print(f"T={T} T0={T0} ({time.time() - t0:.0f}s)", flush=True)

        cust = files(a.root, "customers")
        parts = list(ex.map(top_customers, [(a.root, f, T, a.score_customers) for f in cust]))
        c = np.concatenate([p[0] for p in parts])
        h = np.concatenate([p[1] for p in parts])
        n_customers = sum(p[2] for p in parts)
        order = np.lexsort((c, h))[:a.score_customers]
        score = np.sort(c[order])
        pq.write_table(pa.table({"customer_id": score}), out / "predict_ids_5M.parquet")
        print(f"predict ids {len(score)} of {n_customers} ({time.time() - t0:.0f}s)", flush=True)

        lw = pa.concat_tables(ex.map(last_week, [(a.root, f, a1) for f in files(a.root, "purchases")]))
        pool = pc.unique(lw["customer_id"])
        print(f"last-week purchases {lw.num_rows}, customers {len(pool)} ({time.time() - t0:.0f}s)", flush=True)
        su = pa.concat_tables(ex.map(signups, [(a.root, f, pool) for f in cust]))
        ok = su.filter(pc.less_equal(pc.cast(su["signup_time"], pa.int64()), a1))["customer_id"].to_numpy()
        hh = np_hash(ok, 1, H.SALT["test"])
        chosen = ok[np.lexsort((ok, hh))[:a.test_seeds]]
    sel = lw.filter(pc.is_in(lw["customer_id"], pa.array(chosen)))
    g = sel.group_by("customer_id").aggregate([("merchant_id", "distinct")])
    tg = pa.array([sorted(x) for x in g["merchant_id_distinct"].to_pylist()], pa.list_(pa.int64()))
    test = pa.table({"customer_id": g["customer_id"],
                     "anchor": pa.array(np.full(g.num_rows, a1), pa.int64()).cast(pa.timestamp("ms")),
                     "target": tg}).sort_by("customer_id")
    pq.write_table(test, out / "test_seeds.parquet")
    meta = {"root": a.root, "T": str(np.datetime64(T, "ms")), "T0": str(np.datetime64(T0, "ms")),
            "test_anchor": str(np.datetime64(a1, "ms")), "customers": n_customers,
            "predict_ids": {"file": "predict_ids_5M.parquet", "rows": len(score), "salt": H.SALT["score"],
                            "population": "customers with signup_time <= T",
                            "method": "smallest portable hash h(customer_id, 0, 4), ties by customer_id"},
            "test_seeds": {"file": "test_seeds.parquet", "rows": test.num_rows, "pool": len(ok), "salt": H.SALT["test"],
                           "method": "customers with signup_time <= test_anchor and >= 1 purchase in (test_anchor, T]; "
                                     "smallest h(customer_id, 1, 3), ties by customer_id; target = distinct merchant_id "
                                     "purchased in (test_anchor, T]"},
            "hash": "x=(c*1000003+k*7919+s) mod P; x=((x^(x>>15))*2246822519) mod P; "
                    "x=((x^(x>>13))*3266489917) mod P; x^=(x>>16); P=2^31-1 (int64 arithmetic)",
            "seconds": round(time.time() - t0, 1)}
    (out / "eval_ids.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
