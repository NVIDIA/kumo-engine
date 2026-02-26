#!/usr/bin/env python3
"""
ttgen_compare.py — Compare DiskGraph TT-gen output with Kumo's training tables.

Compiles a plan from a training_job_input.json, executes via the DiskGraph
PyO3 binding, and compares the output with Kumo's intermediate_train_tables.

Usage:
    python3 tools/ttgen_compare.py \
        --job-config /path/to/training_job_input.json \
        --index-path /path/to/diskgraph/index \
        --kumo-entity-dir /path/to/kumo/ENTITY/ \
        --kumo-tt-dir /path/to/kumo/train_table/ \
        --output-dir /path/to/diskgraph/output/  (deprecated, no longer needed)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_parquet_dir(directory: str) -> pa.Table:
    """Read all parquet files in a directory into a single table."""
    files = sorted(Path(directory).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    tables = [pq.read_table(str(f)) for f in files]
    return pa.concat_tables(tables)


def extract_timestamps_from_table(tt_dir: str) -> list[int]:
    """Extract unique timestamps (as epoch seconds) from Kumo's train_table."""
    files = sorted(Path(tt_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {tt_dir}")

    # Read only the TIMESTAMP column to avoid loading all data into memory.
    chunks = []
    for f in files:
        schema = pq.read_schema(str(f))
        if "TIMESTAMP" not in schema.names:
            continue
        col = pq.read_table(str(f), columns=["TIMESTAMP"]).column("TIMESTAMP")
        chunks.append(col)

    if not chunks:
        print("  WARNING: No TIMESTAMP column in train_table; using empty list.")
        return []

    ts_col = pa.concat_arrays([c for chunk in chunks for c in chunk.chunks])
    print(f"  Read {len(ts_col)} timestamps from {len(files)} files")

    # Use Arrow compute for fast conversion and dedup (avoids Python per-element loop).
    if pa.types.is_timestamp(ts_col.type):
        if ts_col.type.unit == "ns":
            epoch_secs = pa.compute.divide(ts_col.cast(pa.int64()), 1_000_000_000)
        elif ts_col.type.unit == "us":
            epoch_secs = pa.compute.divide(ts_col.cast(pa.int64()), 1_000_000)
        elif ts_col.type.unit == "ms":
            epoch_secs = pa.compute.divide(ts_col.cast(pa.int64()), 1_000)
        else:
            epoch_secs = ts_col.cast(pa.int64())
    else:
        epoch_secs = ts_col.cast(pa.int64())

    unique = pa.compute.unique(epoch_secs)
    unique = pa.compute.drop_null(unique)
    result = sorted(unique.to_pylist())
    return result


def parse_job_config(config_path: str) -> dict:
    """Parse training_job_input.json and extract relevant fields."""
    with open(config_path) as f:
        job = json.load(f)

    spec = job.get("spec", job)
    pq_config = spec.get("predictive_query", {})
    vpq_dict = pq_config.get("validated_predictive_query", {})

    task_type = spec.get("task_type", "")
    problem_type = vpq_dict.get("problem_type")
    is_link_pred = "link_prediction" in task_type.lower() if task_type else False

    # include_frequency = True for RANK problems
    include_frequency = False
    if problem_type and "RANK" in str(problem_type).upper():
        include_frequency = True

    forecast_length = pq_config.get("forecast_length",
                                     vpq_dict.get("num_forecasts", 1))
    lag_timesteps = pq_config.get("lag_timesteps")

    # Extract entity table's time_col from graph tables (for static queries).
    # The entity table name comes from the constructed VPQ object (not the raw dict),
    # so we do a deferred lookup keyed by graph table names.
    entity_time_col_by_table = {}
    time_col_dtype_by_table = {}
    graph = pq_config.get("graph", {})
    graph_tables = graph.get("tables", [])
    if isinstance(graph_tables, list):
        for t in graph_tables:
            tname = t.get("table_name")
            tc = t.get("time_col")
            if tname and isinstance(tc, dict) and tc.get("name") and not tc.get("is_pseudo", False):
                entity_time_col_by_table[tname] = tc["name"]
                if tc.get("dtype"):
                    time_col_dtype_by_table[tname] = tc["dtype"]
    elif isinstance(graph_tables, dict):
        for tname, t in graph_tables.items():
            tc = t.get("time_col")
            if isinstance(tc, dict) and tc.get("name") and not tc.get("is_pseudo", False):
                entity_time_col_by_table[tname] = tc["name"]
                if tc.get("dtype"):
                    time_col_dtype_by_table[tname] = tc["dtype"]
            elif isinstance(tc, str) and tc:
                entity_time_col_by_table[tname] = tc

    return {
        "entity_time_col_by_table": entity_time_col_by_table,
        "time_col_dtype_by_table": time_col_dtype_by_table,
        "vpq_dict": vpq_dict,
        "task_type": task_type,
        "problem_type": problem_type,
        "is_link_pred": is_link_pred,
        "include_frequency": include_frequency,
        "forecast_length": forecast_length or 1,
        "lag_timesteps": lag_timesteps,
    }


def compile_plan(config: dict, timestamps: list[int]) -> tuple[str, str | None]:
    """Compile a DiskGraph TT-gen plan. Returns (plan_json, entity_time_col_dtype)."""
    try:
        from kumoapi.pquery.validated_predictive_query import (
            ValidatedPredictiveQuery,
        )
        from diskgraph.compiler import DiskGraphPlanCompiler
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("  Ensure kumoapi is on PYTHONPATH and diskgraph pybind is built.")
        sys.exit(1)

    vpq_dict = config["vpq_dict"]

    # Reconstruct VPQ from the serialized dict.
    # ValidatedPredictiveQuery is a pydantic dataclass, so we use its
    # __pydantic_validator__ or try direct construction.
    try:
        vpq = ValidatedPredictiveQuery(**vpq_dict)
    except Exception as e1:
        # Fallback: try pydantic TypeAdapter
        try:
            from pydantic import TypeAdapter
            ta = TypeAdapter(ValidatedPredictiveQuery)
            vpq = ta.validate_python(vpq_dict)
        except Exception as e2:
            print(f"ERROR: Could not construct VPQ from dict.")
            print(f"  Direct: {e1}")
            print(f"  TypeAdapter: {e2}")
            sys.exit(1)

    # Resolve entity time_col: the VPQ object knows the entity table name
    entity_table = vpq.entity_table
    tc_map = config.get("entity_time_col_by_table", {})
    entity_time_col = tc_map.get(entity_table) if entity_table else None

    # Get the dtype of the entity time_col (e.g. "date" vs "time")
    dtype_map = config.get("time_col_dtype_by_table", {})
    entity_time_col_dtype = dtype_map.get(entity_table) if entity_table else None

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=vpq,
        timestamps=timestamps,
        forecast_length=config["forecast_length"],
        lag_timesteps=config["lag_timesteps"],
        include_frequency=config["include_frequency"],
        entity_time_col=entity_time_col,
    )

    return plan_json, entity_time_col_dtype


def execute_plan(plan_json: str, index_path: str) -> tuple[pa.Table, float]:
    """Execute the plan and return (table, elapsed_seconds)."""
    from diskgraph.executor import generate_training_table

    start = time.monotonic()
    batches = list(generate_training_table(plan_json, index_path))
    elapsed = time.monotonic() - start
    if batches:
        table = pa.Table.from_batches(batches)
    else:
        table = pa.table({"entity_id": pa.array([], pa.uint64()),
                          "timestamp": pa.array([], pa.int64()),
                          "target": pa.array([], pa.float64())})
    return table, elapsed


def normalize_table(table: pa.Table, label: str) -> pa.Table:
    """Normalize a table for comparison: convert timestamps to epoch seconds."""
    columns = {}
    for name in table.column_names:
        col = table.column(name)
        col_type = col.type

        if pa.types.is_timestamp(col_type):
            # Convert to epoch seconds (int64)
            # Arrow timestamps: value is in the specified unit since epoch
            if col_type.unit == "ns":
                col = pa.compute.divide(col.cast(pa.int64()), 1_000_000_000)
            elif col_type.unit == "us":
                col = pa.compute.divide(col.cast(pa.int64()), 1_000_000)
            elif col_type.unit == "ms":
                col = pa.compute.divide(col.cast(pa.int64()), 1_000)
            elif col_type.unit == "s":
                col = col.cast(pa.int64())
            columns[name] = col
        else:
            columns[name] = col

    return pa.table(columns)


def compare_tables(ours: pa.Table, kumo: pa.Table, time_col_dtype: str = None) -> dict:
    """Compare two tables and return a report."""
    report = {
        "ours_rows": ours.num_rows,
        "kumo_rows": kumo.num_rows,
        "ours_columns": ours.column_names,
        "kumo_columns": kumo.column_names,
        "matches": 0,
        "mismatches": [],
    }

    # Normalize both tables
    ours_norm = normalize_table(ours, "diskgraph")
    kumo_norm = normalize_table(kumo, "kumo")

    # Find common columns for comparison
    common_cols = [c for c in ours_norm.column_names if c in kumo_norm.column_names]
    report["common_columns"] = common_cols

    if not common_cols:
        report["mismatches"].append("No common columns to compare")
        return report

    # Convert to pandas for easier comparison
    ours_df = ours_norm.select(common_cols).to_pandas()
    kumo_df = kumo_norm.select(common_cols).to_pandas()

    # Sort both by ENTITY, TIMESTAMP (if present)
    sort_cols = [c for c in ["ENTITY", "TIMESTAMP", "TARGET"] if c in common_cols]
    if sort_cols:
        ours_df = ours_df.sort_values(sort_cols).reset_index(drop=True)
        kumo_df = kumo_df.sort_values(sort_cols).reset_index(drop=True)

    # Row count comparison
    if len(ours_df) != len(kumo_df):
        report["mismatches"].append(
            f"Row count mismatch: ours={len(ours_df)}, kumo={len(kumo_df)}"
        )

    # Per-entity row count comparison
    if "ENTITY" in common_cols:
        ours_counts = ours_df["ENTITY"].value_counts().to_dict()
        kumo_counts = kumo_df["ENTITY"].value_counts().to_dict()

        all_entities = set(ours_counts.keys()) | set(kumo_counts.keys())
        entity_mismatches = 0
        for entity in sorted(all_entities):
            oc = ours_counts.get(entity, 0)
            kc = kumo_counts.get(entity, 0)
            if oc != kc:
                entity_mismatches += 1
                if entity_mismatches <= 10:
                    report["mismatches"].append(
                        f"Entity {entity}: ours={oc} rows, kumo={kc} rows"
                    )
        if entity_mismatches > 10:
            report["mismatches"].append(
                f"  ... and {entity_mismatches - 10} more entity mismatches"
            )

        report["entities_match"] = entity_mismatches == 0
        report["ours_entities"] = len(ours_counts)
        report["kumo_entities"] = len(kumo_counts)

    # Column-level comparison for matching rows
    min_rows = min(len(ours_df), len(kumo_df))
    if min_rows > 0:
        for col in common_cols:
            ours_col = ours_df[col].iloc[:min_rows]
            kumo_col = kumo_df[col].iloc[:min_rows]

            if ours_col.dtype in ("float64", "float32"):
                matches = ((ours_col - kumo_col).abs() < 1e-6) | (
                    ours_col.isna() & kumo_col.isna()
                )
            else:
                matches = (ours_col == kumo_col) | (
                    ours_col.isna() & kumo_col.isna()
                )

            match_count = matches.sum()
            total = len(matches)
            if match_count < total:
                # For TIMESTAMP with dtype:"date", time-of-day is lost at ingest;
                # accept day-level match as a pass.
                if (col == "TIMESTAMP" and time_col_dtype == "date"
                        and ours_col.dtype in ("int64", "float64")):
                    SECS_PER_DAY = 86400
                    ours_days = ours_col // SECS_PER_DAY
                    kumo_days = kumo_col // SECS_PER_DAY
                    day_matches = (ours_days == kumo_days) | (
                        ours_col.isna() & kumo_col.isna()
                    )
                    day_match_count = day_matches.sum()
                    if day_match_count == total:
                        report["matches"] += total
                        report.setdefault("notes", []).append(
                            f"TIMESTAMP: {match_count}/{total} exact, "
                            f"{day_match_count}/{total} day-level match "
                            f"(entity time_col dtype is 'date')"
                        )
                        continue
                    else:
                        day_mismatch_idx = (~day_matches).values.nonzero()[0][:5]
                        report["mismatches"].append(
                            f"Column {col}: {match_count}/{total} exact, "
                            f"{day_match_count}/{total} day-level match"
                        )
                        for idx in day_mismatch_idx:
                            report["mismatches"].append(
                                f"  row {idx}: ours={ours_col.iloc[idx]}, kumo={kumo_col.iloc[idx]}"
                            )
                        continue

                report["mismatches"].append(
                    f"Column {col}: {match_count}/{total} values match"
                )
                # Show first few mismatches
                mismatch_idx = (~matches).values.nonzero()[0][:5]
                for idx in mismatch_idx:
                    report["mismatches"].append(
                        f"  row {idx}: ours={ours_col.iloc[idx]}, kumo={kumo_col.iloc[idx]}"
                    )
            else:
                report["matches"] += total

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare DiskGraph TT-gen output with Kumo's training tables"
    )
    parser.add_argument("--job-config", required=True, help="Path to training_job_input.json")
    parser.add_argument("--index-path", required=True, help="Path to DiskGraph index")
    parser.add_argument("--kumo-tt-dir", required=True, help="Path to Kumo train_table parquet dir")
    parser.add_argument("--output-dir", required=False, help="(deprecated, ignored)")
    parser.add_argument("--plan-only", action="store_true", help="Only compile and print the plan, don't execute")
    args = parser.parse_args()

    timings = {}

    # Step 1: Parse job config
    print("Step 1: Parsing job config...")
    t0 = time.monotonic()
    config = parse_job_config(args.job_config)
    timings["parse_config"] = time.monotonic() - t0
    print(f"  Task type: {config['task_type']}")
    print(f"  Problem type: {config['problem_type']}")
    print(f"  Link prediction: {config['is_link_pred']}")
    print(f"  Include frequency: {config['include_frequency']}")
    print(f"  Forecast length: {config['forecast_length']}")
    print(f"  Lag timesteps: {config['lag_timesteps']}")
    print(f"  Entity time_col map: {config.get('entity_time_col_by_table', {})}")
    print(f"  ⏱  {timings['parse_config']:.3f}s")

    # Step 2: Extract timestamps from Kumo's train_table
    print("\nStep 2: Extracting timestamps from Kumo train_table...")
    t0 = time.monotonic()
    timestamps = extract_timestamps_from_table(args.kumo_tt_dir)
    timings["extract_timestamps"] = time.monotonic() - t0
    print(f"  Found {len(timestamps)} unique timestamps")
    if timestamps:
        print(f"  Range: {timestamps[0]} .. {timestamps[-1]}")
        if len(timestamps) <= 20:
            print(f"  Values: {timestamps}")
    print(f"  ⏱  {timings['extract_timestamps']:.3f}s")

    # Step 3: Compile plan
    print("\nStep 3: Compiling DiskGraph plan...")
    t0 = time.monotonic()
    plan_json, time_col_dtype = compile_plan(config, timestamps)
    timings["compile_plan"] = time.monotonic() - t0
    plan = json.loads(plan_json)
    print(f"  Entity: {plan.get('entity', {}).get('table')}.{plan.get('entity', {}).get('column')}")
    stages = plan.get("stages", [])
    print(f"  Stages: {len(stages)}")
    for i, stage in enumerate(stages):
        ops = stage.get("ops", [])
        for op in ops:
            for key in ("aggregate", "column", "compare"):
                if key in op:
                    info = op[key]
                    print(f"    Stage {i}: {key} alias={info.get('alias', '?')}")
    temporal = plan.get("temporal") or {}
    plan_ts = temporal.get("timestamps", [])
    print(f"  Timestamps in plan: {len(plan_ts)}")
    pred = plan.get("prediction_type", {})
    print(f"  Prediction type: {pred}")
    print(f"  ⏱  {timings['compile_plan']:.3f}s")

    if args.plan_only:
        print("\n--- Plan JSON ---")
        print(json.dumps(plan, indent=2))
        return

    # Step 4: Execute
    print(f"\nStep 4: Executing TT-gen plan...")
    ours, elapsed = execute_plan(plan_json, args.index_path)
    timings["execute"] = elapsed
    print(f"  Execution time: {elapsed:.2f}s")
    print(f"  Our output: {ours.num_rows} rows, columns: {ours.column_names}")
    print(f"  Our schema: {ours.schema}")

    # Step 5: Read Kumo's output and compare
    print(f"\nStep 5: Comparing with Kumo output...")
    t0 = time.monotonic()
    try:
        kumo = read_parquet_dir(args.kumo_tt_dir)
        timings["read_kumo_output"] = time.monotonic() - t0
        print(f"  Kumo output: {kumo.num_rows} rows, columns: {kumo.column_names}")
        print(f"  Kumo schema: {kumo.schema}")
        print(f"  ⏱  read: {timings['read_kumo_output']:.3f}s")
    except FileNotFoundError:
        print(f"  WARNING: No Kumo train_table parquet files found in {args.kumo_tt_dir}")
        print(f"  Skipping comparison. Our output is at {args.output_dir}")
        print_timing_summary(timings)
        return

    t0 = time.monotonic()
    report = compare_tables(ours, kumo, time_col_dtype=time_col_dtype)
    timings["compare"] = time.monotonic() - t0

    # Print report
    print("\n" + "=" * 60)
    print("  COMPARISON REPORT")
    print("=" * 60)
    print(f"  Our rows:     {report['ours_rows']}")
    print(f"  Kumo rows:    {report['kumo_rows']}")
    print(f"  Our columns:  {report['ours_columns']}")
    print(f"  Kumo columns: {report['kumo_columns']}")
    print(f"  Common cols:  {report.get('common_columns', [])}")

    if "ours_entities" in report:
        print(f"  Our entities:  {report['ours_entities']}")
        print(f"  Kumo entities: {report['kumo_entities']}")
        print(f"  Entities match: {report.get('entities_match', 'N/A')}")

    print(f"  Values matched: {report['matches']}")

    if report.get("notes"):
        print(f"\n  NOTES:")
        for n in report["notes"]:
            print(f"    {n}")

    if report["mismatches"]:
        print(f"\n  MISMATCHES ({len(report['mismatches'])}):")
        for m in report["mismatches"]:
            print(f"    {m}")
    else:
        print("\n  RESULT: ALL MATCH")

    print("=" * 60)

    # Print timing summary
    print_timing_summary(timings)

    # Exit with error if mismatches
    if report["mismatches"]:
        sys.exit(1)


def print_timing_summary(timings: dict):
    """Print a summary of all timing measurements."""
    print("\n" + "─" * 60)
    print("  TIMING SUMMARY")
    print("─" * 60)
    total = 0.0
    for name, elapsed in timings.items():
        print(f"  {name:25s} {elapsed:8.3f}s")
        total += elapsed
    print(f"  {'TOTAL':25s} {total:8.3f}s")
    print("─" * 60)


if __name__ == "__main__":
    main()
