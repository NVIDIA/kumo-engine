# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared schema, plans and seed handling for the neighbourhood-context benchmark."""
import os
import resource
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
TABLES = ["customers", "merchants", "purchases", "merchant_views", "searches", "favorites", "reviews"]
EVENTS = ["purchases", "merchant_views", "searches", "favorites", "reviews"]
TIME = {"customers": "signup_time", "merchants": "created_time", "purchases": "purchase_time",
        "merchant_views": "view_time", "searches": "search_time", "favorites": "created_time",
        "reviews": "review_time"}
PK = {"customers": "customer_id", "merchants": "merchant_id"}
FKS = {e: (["customer_id", "merchant_id"] if e != "searches" else ["customer_id"]) for e in EVENTS}
FK_DST = {"customer_id": "customers", "merchant_id": "merchants"}
# Columns that identify a raw row.
KEY = {"customers": ["customer_id"], "merchants": ["merchant_id"], "purchases": ["purchase_id"],
       "merchant_views": ["customer_id", "merchant_id", "view_time", "source"],
       "searches": ["customer_id", "search_time", "query_term", "category", "result_page"],
       "favorites": ["customer_id", "merchant_id", "created_time"],
       "reviews": ["customer_id", "merchant_id", "review_time", "stars"]}
PLAN6 = [24, 8, 6, 8, 6, 8]


def plan(hops):
    return PLAN6[:hops]


def relations():
    """(src_type, dst_type, kind, column): every FK in both directions, as DiskGraph builds them."""
    out = []
    for e in EVENTS:
        for col in FKS[e]:
            out.append((FK_DST[col], e, "rev", col))
            out.append((e, FK_DST[col], "fk", col))
    return out


def load_seeds(path, n, offset=0):
    t = pq.read_table(path, columns=["customer_id", "anchor"])
    t = tile(t, offset + n).slice(offset, n)
    cid = t["customer_id"].to_numpy().astype(np.int64)
    anchor = t["anchor"].cast(pa.timestamp("s")).cast(pa.int64()).to_numpy().astype(np.int64)
    return cid, anchor


def tile(t, n):
    """Repeat a seed table when a run asks for more seeds than the rung has (small rungs only)."""
    if t.num_rows == 0:
        raise ValueError("seed table is empty; supply a nonempty evaluation seed file")
    return pa.concat_tables([t] * (-(-n // t.num_rows))) if n > t.num_rows else t


def rusage():
    r = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"utime": r.ru_utime + c.ru_utime, "stime": r.ru_stime + c.ru_stime,
            "maxrss_kb": max(r.ru_maxrss, c.ru_maxrss)}


def proc_read_bytes():
    try:
        return int(open("/proc/self/io").read().split("read_bytes: ")[1].split()[0])
    except Exception:
        return 0


def evict(paths):
    """Drop the page cache for every file under paths (per-file, safe on a shared box)."""
    n = 0
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else [Path(r) / f for r, _, fs in os.walk(p) for f in fs]
        for f in files:
            try:
                fd = os.open(f, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    return n


def drop_all_caches():
    """Whole-box drop (EC2 box only)."""
    os.sync()
    Path("/proc/sys/vm/drop_caches").write_text("3\n")
