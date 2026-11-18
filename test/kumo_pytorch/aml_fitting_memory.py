# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nightly 25k/400k fitting-memory campaigns for the five AML input contracts."""

import argparse
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import traceback
from collections import Counter
from itertools import islice
from pathlib import Path

from .aml_inputs import verify_inputs
from .aml_source import (
    account_identity_map,
    aml_data_schema,
    prepare_aml_sources,
    prepared_dataset,
)
from .benchmarking.fitting_memory import campaign, digest, save
from .contracts import build_contract
from .data import ROOT
from .nightly import selected_cases
from .population_anchors import (
    add_identity,
    project_population,
    projected_rows,
)

COUNTS = (25000, 400000)


def make_spec(
    dataset,
    identities,
    anchor_path,
    selection,
    family,
    directory,
    *,
    capacity,
    counts=COUNTS,
    threads=1,
):
    """Build nested training-only prefixes outside the measured worker process."""
    directory.mkdir(parents=True, exist_ok=False)
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    graph = family in {"graphsage", "combined"}
    contract = build_contract(
        aml_data_schema(), family, native=native, max_events=capacity
    )
    contract_path = directory / "contract.json"
    contract_path.write_text(contract.to_json())
    anchors = {}
    for count in counts:
        path = directory / f"anchors-{count}.parquet"
        _, population = project_population(
            anchor_path,
            path,
            identities,
            graph=graph,
            native=native,
            limit=count,
            splits=("train",),
        )
        anchors[str(count)] = {
            "path": str(path.resolve()),
            "sha256": digest(path),
            "population": population,
        }
    prefix = hashlib.sha256()
    for row in islice(
        projected_rows(Path(anchors[str(counts[1])]["path"]), "train"), counts[0]
    ):
        add_identity(prefix, row["sample_id"])
    assert (
        prefix.hexdigest()
        == anchors[str(counts[0])]["population"]["ordered_sample_id_sha256"]["train"]
    ), "Smaller training population is not the larger population's prefix"
    spec = {
        "id": f"{selection['selector']['workload_id']}-{family}",
        "selector": selection["selector"],
        "family": family,
        "contract": str(contract_path.resolve()),
        "contract_fingerprint": contract.fingerprint,
        "sources": {
            name: str(Path(source.path).resolve())
            for name, source in dataset.tables.items()
        },
        "source_anchor_sha256": digest(anchor_path),
        "index": str((directory / "index").resolve()),
        "anchors": anchors,
        "threads": threads,
        "fit_split": "train",
        "anchor_columns": {
            "entity_id": "entity_id",
            "as_of_time": "as_of_time",
            "sample_id": "sample_id",
            "split": "split",
            "targets": {} if native else {"label": "label"},
            "metadata": [
                "prediction_time_ns",
                "registered_entity_id",
                "anchor_row_position",
            ],
        },
        "encoder_cutoff_seconds": anchors[str(counts[1])]["population"][
            "encoder_cutoff_seconds"
        ],
        "cutoff_policy": "shared cutoff from larger training prefix; only fitting-anchor count changes",
        "history_capacity": capacity,
    }
    path = directory / "spec.json"
    save(path, spec)
    return path


def summary(output, report):
    save(output / "report.json", report)
    lines = [
        "# Fitting-memory scalability",
        "",
        f"Status: **{report['status']}**",
        "",
        "Fresh fitting at 25,000 and 400,000 training anchors; index build/reopen excluded.",
        "Growth is reported without a pass/fail limit; status reflects measurement validity.",
        f"Memory method: `{report['memory_method']}`. Sampled peaks can miss allocations between samples.",
        "",
        "| Contract | 25k median bytes | 400k median bytes | Growth | 25k anonymous bytes | 400k anonymous bytes | Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    if report.get("selected_repeat") is not None:
        lines[4:4] = [
            f"This shard measures repetition {report['selected_repeat']} only. "
            "The combined nightly report requires all three repetitions.",
            "",
        ]
    errors = []
    for case in report["cases"]:
        comparison = case.get("comparison", {})
        medians = comparison.get("median_peak_bytes", {})
        anonymous = comparison.get("median_anonymous_bytes", {})
        lines.append(
            f"| {case['case']} | {medians.get('25000', 'unavailable')} | "
            f"{medians.get('400000', 'unavailable')} | {comparison.get('growth_ratio') or 'unavailable'} | {anonymous.get('25000', 'unavailable')} | {anonymous.get('400000', 'unavailable')} | {case['status']} |"
        )
        if case.get("error") or comparison.get("reason"):
            errors.extend(
                ["", f"{case['case']}: {case.get('error') or comparison['reason']}"]
            )
    lines.extend(errors)
    if report.get("error"):
        lines.extend(["", f"Preparation failed: {report['error']}"])
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--memory-method", choices=("sampled", "kernel"), default="sampled"
    )
    parser.add_argument("--memory-interval", type=float, default=0.05)
    parser.add_argument("--worker-timeout", type=float, default=3600)
    parser.add_argument(
        "--case", help="Run one catalog binding; omitted runs all bindings"
    )
    parser.add_argument(
        "--repeat", type=int, help="Run one paired repetition; CI combines all three"
    )
    args = parser.parse_args(argv)
    if args.repeat is not None and not 0 <= args.repeat < args.repetitions:
        parser.error("--repeat must be within --repetitions")
    cases = selected_cases(args.case)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "cases": [],
        "counts": list(COUNTS),
        "memory_method": args.memory_method,
        "repetitions": args.repetitions,
        "selected_repeat": args.repeat,
        "scope": "DiskGraph fresh fitting; no batches or models executed",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "RAYON_NUM_THREADS",
                "POLARS_MAX_THREADS",
            )
        },
    }
    try:
        report["revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        report["versions"] = {
            p: importlib.metadata.version(p) for p in ("torch", "pyarrow", "diskgraph")
        }
        report["dirty_checkout"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
        summary(args.output, report)
        files = verify_inputs(args.data_dir)
        report["inputs"] = list(files.values())
        transactions = "datasets/aml-data/v8/hi-small/transactions.parquet"
        binding = "anchors/aml-data/v8/hi-small/_bindings/v1/accounts.parquet"
        prepared = args.output / "prepared"
        report["mapping"] = prepare_aml_sources(
            args.data_dir / transactions,
            args.data_dir / binding,
            prepared,
            "hi-small",
            expected_checksums={
                "transactions": files[transactions]["sha256"],
                "accounts_binding": files[binding]["sha256"],
            },
        )
        dataset, identities = prepared_dataset(prepared), account_identity_map(prepared)
        import pyarrow.parquet as pq

        counts = Counter()
        for batch in pq.ParquetFile(prepared / "account_events.parquet").iter_batches(
            columns=["account_id"]
        ):
            counts.update(batch.column(0).to_pylist())
        capacity = max(counts.values())
        del counts
        for selection, family in cases:
            name = f"{selection['selector']['workload_id']}-{family}"
            try:
                anchor_path = (
                    args.data_dir
                    / f"anchors/aml-data/v8/hi-small/account-daily/{selection['selector']['anchor_version']}/anchors.parquet"
                )
                spec = make_spec(
                    dataset,
                    identities,
                    anchor_path,
                    selection,
                    family,
                    args.output / "inputs" / name,
                    capacity=capacity,
                )
                case = campaign(
                    spec,
                    args.output / "runs" / name,
                    repetitions=args.repetitions,
                    method=args.memory_method,
                    interval=args.memory_interval,
                    timeout=args.worker_timeout,
                    repeat=args.repeat,
                )
            except Exception as error:  # noqa: BLE001 - continue other required contracts
                case = {
                    "case": name,
                    "status": "failed",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            report["cases"].append(case)
            summary(args.output, report)
        assert len(report["cases"]) == len(cases), (
            "Missing selected AML fitting contracts"
        )
        report["status"] = (
            "passed"
            if all(c["status"] == "passed" for c in report["cases"])
            else "failed"
        )
    except Exception as error:  # noqa: BLE001 - preserve preparation failures
        report.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
    summary(args.output, report)
    print(f"{report['status']}: {args.output / 'report.md'}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
