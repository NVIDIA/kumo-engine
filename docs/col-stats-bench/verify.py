#!/usr/bin/env python3
"""Verify col_stats.json against ground_truth.json.

Compares numerical stats within tolerances and reports detailed error metrics.
For quantiles, reports both absolute error and rank error.
"""

import json
import sys
import argparse
import numpy as np


def rel_error(actual, expected):
    if expected == 0:
        return abs(actual) if actual != 0 else 0.0
    return abs(actual - expected) / abs(expected)


def check_field(table, col, field, actual, expected, tolerance, results):
    err = rel_error(actual, expected)
    abs_err = abs(actual - expected)
    status = "PASS" if err <= tolerance else "FAIL"
    results.append({
        "table": table,
        "column": col,
        "field": field,
        "actual": actual,
        "expected": expected,
        "rel_error": err,
        "abs_error": abs_err,
        "tolerance": tolerance,
        "status": status,
    })
    return status == "PASS"


def compute_rank_error(q_actual, q_expected, data_min, data_max):
    """Compute rank error: |actual - expected| / (max - min).

    This measures the error in terms of the data range, which is a proxy
    for how many ranks the quantile estimate is off by.
    """
    data_range = data_max - data_min
    if data_range == 0:
        return 0.0
    return abs(q_actual - q_expected) / data_range


def verify(stats_path, truth_path, verbose=False):
    with open(stats_path) as f:
        stats = json.load(f)
    with open(truth_path) as f:
        truth = json.load(f)

    results = []
    quantile_details = []  # Track quantile errors separately
    passes = 0
    fails = 0
    skips = 0

    for table_name, truth_cols in truth.items():
        if table_name not in stats:
            print(f"SKIP: table '{table_name}' not in col_stats.json")
            skips += len(truth_cols)
            continue

        stats_cols = stats[table_name]

        for col_name, truth_col in truth_cols.items():
            if col_name not in stats_cols:
                print(f"SKIP: {table_name}.{col_name} not in col_stats.json")
                skips += 1
                continue

            stats_col = stats_cols[col_name]
            t_type = truth_col.get("semantic_type", "")
            s_type = stats_col.get("semantic_type", "")

            # Type mismatch — compare what we can
            if t_type != s_type:
                print(f"TYPE MISMATCH: {table_name}.{col_name}: "
                      f"truth={t_type}, stats={s_type}")
                for field in ["count", "null_count"]:
                    if field in truth_col and field in stats_col:
                        ok = check_field(table_name, col_name, field,
                                         stats_col[field], truth_col[field],
                                         0.0, results)
                        passes += ok
                        fails += (not ok)
                if "num_unique" in truth_col and "num_unique" in stats_col:
                    ok = check_field(table_name, col_name, "num_unique",
                                     stats_col["num_unique"],
                                     truth_col["num_unique"],
                                     0.05, results)
                    passes += ok
                    fails += (not ok)
                continue

            # Same type — compare all fields
            # Exact fields
            for field in ["count", "null_count"]:
                if field in truth_col:
                    ok = check_field(table_name, col_name, field,
                                     stats_col.get(field, 0),
                                     truth_col[field], 0.0, results)
                    passes += ok
                    fails += (not ok)

            # Welford fields (tight tolerance)
            for field in ["mean", "std", "min", "max"]:
                if field in truth_col:
                    ok = check_field(table_name, col_name, field,
                                     stats_col.get(field, 0),
                                     truth_col[field], 1e-4, results)
                    passes += ok
                    fails += (not ok)

            # Moments (moderate tolerance)
            for field in ["skewness", "kurtosis"]:
                if field in truth_col:
                    ok = check_field(table_name, col_name, field,
                                     stats_col.get(field, 0),
                                     truth_col[field], 0.01, results)
                    passes += ok
                    fails += (not ok)

            # Quantiles — pass/fail based on rank error (robust to near-zero values)
            if all(f in truth_col for f in ["q25", "q50", "q75"]):
                data_min = truth_col.get("min", 0)
                data_max = truth_col.get("max", 0)
                for field in ["q25", "q50", "q75"]:
                    actual = stats_col.get(field, 0)
                    expected = truth_col[field]
                    rank_err = compute_rank_error(actual, expected, data_min, data_max)
                    r_err = rel_error(actual, expected)
                    abs_err = abs(actual - expected)
                    # Use rank error for pass/fail (0.5% threshold).
                    # Relative error is misleading when expected ≈ 0.
                    ok = rank_err <= 0.005
                    status = "PASS" if ok else "FAIL"
                    results.append({
                        "table": table_name,
                        "column": col_name,
                        "field": field,
                        "actual": actual,
                        "expected": expected,
                        "rel_error": r_err,
                        "abs_error": abs_err,
                        "rank_error": rank_err,
                        "tolerance": 0.005,
                        "status": status,
                    })
                    passes += ok
                    fails += (not ok)
                    quantile_details.append({
                        "table": table_name,
                        "column": col_name,
                        "quantile": field,
                        "actual": actual,
                        "expected": expected,
                        "abs_error": abs_err,
                        "rel_error": r_err,
                        "rank_error": rank_err,
                    })

            # HLL num_unique (5% tolerance)
            if "num_unique" in truth_col:
                ok = check_field(table_name, col_name, "num_unique",
                                 stats_col.get("num_unique", 0),
                                 truth_col["num_unique"], 0.05, results)
                passes += ok
                fails += (not ok)

            # Timestamp fields (exact)
            for field in ["min_secs", "min_nanos", "max_secs", "max_nanos"]:
                if field in truth_col:
                    ok = check_field(table_name, col_name, field,
                                     stats_col.get(field, 0),
                                     truth_col[field], 0.0, results)
                    passes += ok
                    fails += (not ok)

            # Sequence fields (exact)
            for field in ["min_len", "max_len"]:
                if field in truth_col:
                    ok = check_field(table_name, col_name, field,
                                     stats_col.get(field, 0),
                                     truth_col[field], 0.0, results)
                    passes += ok
                    fails += (not ok)

            # Text fields
            for field in ["avg_tokens", "non_char_fraction"]:
                if field in truth_col:
                    ok = check_field(table_name, col_name, field,
                                     stats_col.get(field, 0),
                                     truth_col[field], 0.01, results)
                    passes += ok
                    fails += (not ok)

    # Print failures
    failures = [r for r in results if r["status"] == "FAIL"]
    if failures:
        print(f"\n{'='*70}")
        print(f"FAILURES ({len(failures)}):")
        print(f"{'='*70}")
        for r in failures:
            print(f"  {r['table']}.{r['column']}.{r['field']}: "
                  f"actual={r['actual']:.6g}, expected={r['expected']:.6g}, "
                  f"rel_err={r['rel_error']:.4%}, abs_err={r['abs_error']:.6g}")

    # Print quantile error summary
    if quantile_details:
        print(f"\n{'='*70}")
        print("QUANTILE ERROR SUMMARY:")
        print(f"{'='*70}")
        abs_errs = [d["abs_error"] for d in quantile_details]
        rel_errs = [d["rel_error"] for d in quantile_details]
        rank_errs = [d["rank_error"] for d in quantile_details]
        print(f"  Columns with quantiles: {len(quantile_details) // 3}")
        print(f"  Absolute error: mean={np.mean(abs_errs):.4g}, "
              f"max={np.max(abs_errs):.4g}, "
              f"p99={np.percentile(abs_errs, 99):.4g}")
        print(f"  Relative error: mean={np.mean(rel_errs):.4%}, "
              f"max={np.max(rel_errs):.4%}, "
              f"p99={np.percentile(rel_errs, 99):.4%}")
        print(f"  Rank error:     mean={np.mean(rank_errs):.4%}, "
              f"max={np.max(rank_errs):.4%}, "
              f"p99={np.percentile(rank_errs, 99):.4%}")

        if verbose:
            print(f"\n  Per-column detail:")
            for d in quantile_details:
                print(f"    {d['table']}.{d['column']}.{d['quantile']}: "
                      f"actual={d['actual']:.6g}, expected={d['expected']:.6g}, "
                      f"abs={d['abs_error']:.4g}, "
                      f"rel={d['rel_error']:.4%}, "
                      f"rank={d['rank_error']:.4%}")

    # Print HLL error summary
    hll_results = [r for r in results if r["field"] == "num_unique"]
    if hll_results:
        hll_errs = [r["rel_error"] for r in hll_results]
        print(f"\n  HLL num_unique: mean_err={np.mean(hll_errs):.2%}, "
              f"max_err={np.max(hll_errs):.2%}")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {passes} passed, {fails} failed, {skips} skipped")
    print(f"{'='*70}")
    return fails == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True, help="Path to col_stats.json")
    parser.add_argument("--truth", required=True, help="Path to ground_truth.json")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show per-column quantile details")
    args = parser.parse_args()
    ok = verify(args.stats, args.truth, args.verbose)
    sys.exit(0 if ok else 1)
