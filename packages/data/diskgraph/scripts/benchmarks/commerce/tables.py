# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Markdown table from results-<rung>.json: seeds/s (time to first batch), warm and cold, per hop count."""
import json
import sys


def fmt(x):
    return "n/a" if x is None else (f"{x:,.0f}" if x >= 100 else f"{x:,.1f}")


for path in sys.argv[1:]:
    R = json.load(open(path))
    cells = {(c["hops"], c["cache"]): c for c in R["throughput"]}
    print(f"\n### {R['rung']}: DiskGraph seeds/s (time to first batch)\n")
    print("| hops | warm | cold | rows per seed |\n|---|---|---|---|")
    for h in range(2, 7):
        row = []
        for c in ("warm", "cold"):
            x = cells.get((h, c))
            st = "" if not x or x.get("status", "ok") == "ok" else f" **{x['status']}**"
            row.append("n/a" if not x else f"{fmt(x['seeds_per_s'])} ({x['first_batch_s']:.1f}s){st}")
        x = cells.get((h, "warm")) or cells.get((h, "cold"))
        row.append(fmt(x.get("nodes_per_seed")) if x else "n/a")
        print(f"| {h} | " + " | ".join(row) + " |")
    i = R["ingest"]
    if i.get("seconds"):
        print(f"\nIngest: {i['seconds'] / 60:.1f} min, index {i['disk_gib']:,.0f} GiB "
              f"(Parquet {R['parquet_gib']:,.0f} GiB).")
