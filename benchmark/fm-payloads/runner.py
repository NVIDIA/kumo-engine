# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness-gated, interleaved FM payload timing against the DuckDB reference."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import kumo_fm_benchmark as fm
from kumo_fm_benchmark._provenance import dependency_provenance
from kumo_fm_benchmark._pytorch import compatibility_issues
from kumo_fm_benchmark.bundle import Bundle
from kumo_fm_benchmark.workload import Workload

from kumo import settings


def revision(path):
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def installed_revision(name):
    distribution = importlib.metadata.distribution(name)
    source = json.loads(distribution.read_text("direct_url.json") or "{}").get("url")
    return revision(Path(urlparse(source).path)) if source and source.startswith("file://") else None


def measure(callback):
    start = time.perf_counter_ns()
    result = callback()
    return result, (time.perf_counter_ns() - start) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="FM workspace")
    parser.add_argument("--scratch-dir", type=Path, required=True, help="Managed Kumo storage root")
    parser.add_argument("--draws", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.draws < 1 or args.warmups < 1:
        parser.error("draws and warmups must be positive")

    from fm_dataloader.bundle import Bundle as OldBundle
    from fm_dataloader.multihop_sql import DuckDBMultihopBackend
    from fm_dataloader.payload import build_payload, decode_payload
    from fm_dataloader.workload import Workload as OldWorkload

    settings.configure(scratch_dir=args.scratch_dir)
    old_bundle = OldBundle(args.bundle)
    old_workload = OldWorkload.load(old_bundle, args.workload)
    new_bundle = Bundle(args.bundle)
    new_workload = Workload.load(new_bundle, args.workload)
    issues = compatibility_issues(new_workload)
    report = {
        "bundle_manifest_sha256": digest(args.bundle / "manifest.json"),
        "workload_sha256": digest(args.workload),
        "wheel_sha256": {path.name: digest(path) for path in args.wheel},
        "snapshot_id": new_bundle.snapshot_id,
        "engine_revision": revision(Path(__file__).resolve().parents[2]),
        "reference_revision": installed_revision("fm-dataloader"),
        "checkout_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True).stdout.strip()),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "python": platform.python_version()},
        "versions": {name: dependency_provenance(name)["version"] for name in
                     ("kumo-fm", "kumo-pytorch", "diskgraph", "fm-dataloader",
                      "duckdb", "pyarrow", "numpy")},
        "cases": [],
        "warmups": args.warmups,
        "draws_per_encoding": args.draws,
        "graph": {"layers": new_workload.layers,
                  "table_rows": {name: new_bundle.tables[name]["rows"]
                                 for name in new_workload.table_names},
                  "relations": sorted(new_workload.relation_names)},
        "eligibility": "excluded" if issues else "supported",
        "exclusion_reasons": issues,
    }
    if issues:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    old_backend = DuckDBMultihopBackend(old_bundle, native=True)
    try:
        with fm.prepare(new_bundle, new_workload, index=args.index) as loader:
            report["index_marker_sha256"] = digest(args.index / "fm_snapshot.json")
            with loader.materialize() as request:
                batch = request.load()
                report["selected_rows"] = {
                    split: {name: table.num_rows for name, table in
                            batch.tables(split)["related_tables"].items()}
                    for split in batch.splits}
                report["selected_edges"] = {
                    split: {
                        relation: len(batch.tables(split)["related_tables"][rel["child"]][
                            "__fm_edge_" + relation])
                        - batch.tables(split)["related_tables"][rel["child"]][
                            "__fm_edge_" + relation].null_count
                        for relation, rel in new_bundle.relations.items()
                        if relation in new_workload.relation_names}
                    for split in batch.splits}
                for encoding in ("arrays", "arrow_ipc"):
                    old_body = build_payload(old_backend, old_workload, encoding=encoding)
                    new_body = batch.encode(encoding)
                    if decode_payload(old_body) != decode_payload(new_body):
                        raise AssertionError(f"{encoding} differs from DuckDB reference")
                    report["cases"].append({
                        "encoding": encoding,
                        "context_rows": batch.tables("context")["instance_table"].num_rows,
                        "predict_rows": batch.tables("predict")["instance_table"].num_rows,
                        "logical_arrow_bytes": batch.logical_arrow_bytes,
                        "new_body_bytes": len(new_body),
                        "reference_body_bytes": len(old_body),
                        "draws": [],
                    })
                for iteration in range(args.warmups + args.draws):
                    order = report["cases"][:]
                    random.Random(iteration).shuffle(order)
                    for case in order:
                        encoding = case["encoding"]
                        old_body, old_ms = measure(
                            lambda: build_payload(old_backend, old_workload, encoding=encoding))
                        new_batch, load_ms = measure(request.load)
                        new_body, encode_ms = measure(lambda: new_batch.encode(encoding))
                        if decode_payload(old_body) != decode_payload(new_body):
                            raise AssertionError(f"{encoding} changed during timing")
                        if iteration >= args.warmups:
                            case["draws"].append({
                                "reference_full_ms": old_ms,
                                "new_load_ms": load_ms,
                                "new_encode_ms": encode_ms,
                            })
    finally:
        old_backend.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["correctness"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
