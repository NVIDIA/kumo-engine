# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-cut a prepared TabFormer root into a within-era temporal split.

TabFormer's fraud comes in campaigns: 61-95% of each year's frauds from 2007
to 2016 are online, and from 2017 nearly all are in Rome. v1's 80/10/10 cut
puts validation and test inside the Rome campaign and training before it, so
a probe fit on training rows cannot see the test's fraud pattern, and one fit
on the validation period learns a one-city rule. This cut keeps all three
partitions inside the online era and drops every later event:

    train      < --validation-start   (default 2015-01-01)
    val        [--validation-start, --test-start)   (2015)
    test       [--test-start, --end)                (2016)

Rows are copied unchanged into `<out>/temporal_split/{train,val,test}.parquet`,
which `split_protocol: precomputed` reads. Point `TFM_TABFORMER_ROOT` at `<out>`.

    uv run python benchmarks/tabformer_era_split.py \\
        --root ~/data/TabFormer --out ~/data/TabFormer_era
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PARTITIONS = ("train", "val", "test")


def _day_key(table: pa.Table) -> pa.Array:
    """`Year * 10000 + Month * 100 + Day`, comparable with `_key`."""
    year = pc.cast(table["Year"], pa.int64())
    month = pc.cast(table["Month"], pa.int64())
    day = pc.cast(table["Day"], pa.int64())
    return pc.add(pc.add(pc.multiply(year, 10_000), pc.multiply(month, 100)), day)


def _key(value: str) -> int:
    parsed = date.fromisoformat(value)
    return parsed.year * 10_000 + parsed.month * 100 + parsed.day


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-start", default="2015-01-01")
    parser.add_argument("--test-start", default="2016-01-01")
    parser.add_argument("--end", default="2017-01-01")
    args = parser.parse_args()

    source = args.root / "temporal_split"
    table = pa.concat_tables([pq.read_table(source / f"{name}.parquet") for name in PARTITIONS])
    key = _day_key(table)
    bounds = [0, _key(args.validation_start), _key(args.test_start), _key(args.end)]
    target = args.out / "temporal_split"
    target.mkdir(parents=True, exist_ok=True)
    for name, low, high in zip(PARTITIONS, bounds, bounds[1:], strict=False):
        mask = pc.and_(pc.greater_equal(key, low), pc.less(key, high))
        part = table.filter(mask)
        frauds = pc.sum(pc.equal(part["Is Fraud?"], "Yes")).as_py() or 0
        pq.write_table(part, target / f"{name}.parquet")
        print(f"{name}: {part.num_rows:,} rows, {frauds:,} frauds")


if __name__ == "__main__":
    main()
