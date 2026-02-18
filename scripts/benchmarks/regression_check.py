#!/usr/bin/env python3
"""Compare ingest stats against a baseline and report regressions."""
import argparse
import json
import sys


METRICS = ["encode_wall_ms", "pk_map_wall_ms", "edges_map_wall_ms", "index_wall_ms"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file", help="Current run ingest_stats.json")
    parser.add_argument("baseline_file", help="Baseline ingest_stats.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--warn-threshold", type=float, default=12)
    parser.add_argument("--fail-threshold", type=float, default=20)
    args = parser.parse_args()

    with open(args.stats_file) as f:
        current = json.load(f)["stats"]["perf"]
    with open(args.baseline_file) as f:
        base = json.load(f)["stats"]["perf"]

    warned = False
    failed = False

    for m in METRICS:
        if m not in current or m not in base:
            continue
        cur = current[m]
        bas = base[m]
        if bas == 0:
            continue
        delta_pct = ((cur - bas) / bas) * 100
        status = "OK"
        if delta_pct > args.fail_threshold:
            status = "FAIL"
            failed = True
            print(f"::error::{args.dataset}: {m} is {delta_pct:+.1f}% vs baseline ({cur}ms vs {bas}ms)")
        elif delta_pct > args.warn_threshold:
            status = "WARN"
            warned = True
            print(f"::warning::{args.dataset}: {m} is {delta_pct:+.1f}% vs baseline ({cur}ms vs {bas}ms)")
        print(f"  {m}: {bas}ms -> {cur}ms ({delta_pct:+.1f}%) {status}")

    cur_total = sum(current.get(m, 0) for m in METRICS)
    bas_total = sum(base.get(m, 0) for m in METRICS)
    if bas_total > 0:
        total_delta = ((cur_total - bas_total) / bas_total) * 100
        print(f"  TOTAL: {bas_total}ms -> {cur_total}ms ({total_delta:+.1f}%)")
        if total_delta > args.fail_threshold:
            failed = True
            print(f"::error::{args.dataset}: overall {total_delta:+.1f}% vs baseline")
        elif total_delta > args.warn_threshold:
            warned = True
            print(f"::warning::{args.dataset}: overall {total_delta:+.1f}% vs baseline")
    else:
        total_delta = 0.0

    if failed:
        print("STATUS:FAIL")
        print(f"DELTA:{total_delta:+.1f}%")
        sys.exit(1)
    elif warned:
        print("STATUS:WARN")
        print(f"DELTA:{total_delta:+.1f}%")
    else:
        print("STATUS:OK")
        print(f"DELTA:{total_delta:+.1f}%")


if __name__ == "__main__":
    main()
