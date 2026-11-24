# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build small graph benchmark manifests from archived, checksummed raw JSON.

This never uploads data or rewrites raw measurements. Archive each input first,
then pass the verified private artifact base URL. Resource supplements are kept
separate from historical latency runs and never treated as historical peaks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

NAMES = ("graph_batch_execution_l4", "graph_execution_views_l4", "graph_constructor_checks_l4")


def reference(path: Path, url: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "url": url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "media_type": "application/json",
    }


def _conditions(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(run["source_revision"] for run in runs)
    conditions = {}
    for run in runs:
        revision = run["source_revision"]
        selected = {
            key: run[key]
            for key in (
                "source_revision",
                "source_dirty",
                "driver_sha256",
                "execution_driver_sha256",
                "contract_sha256",
                "contract_source_sha256",
                "config",
                "settings",
                "precision",
                "python",
                "torch",
                "torch_geometric",
                "kumo_engine",
                "gpu",
                "tf32",
                "deterministic_algorithms",
                "pythonhashseed",
                "cublas_workspace_config",
                "warmup",
                "steps",
            )
            if key in run
        }
        selected["repetitions"] = counts[revision]
        selected["workloads"] = [
            {
                key: row[key]
                for key in (
                    "arm",
                    "sharing",
                    "nodes",
                    "edges",
                    "initial_state_sha256",
                    "input_sha256",
                    "effective_settings",
                )
                if key in row
            }
            for row in run.get("results", run.get("arms", []))
        ]
        for raw_row, workload in zip(
            run.get("results", run.get("arms", [])), selected["workloads"], strict=True
        ):
            if isinstance(raw_row.get("samples"), int):
                workload["root_count"] = raw_row["samples"]
            elif "final_embeddings" in raw_row:
                workload["root_count"] = len(raw_row["final_embeddings"])
            elif "final_prediction" in raw_row:
                workload["root_count"] = len(raw_row["final_prediction"])
            elif "effective_settings" in raw_row:
                workload["root_count"] = raw_row["effective_settings"]["root_count"]
        if revision in conditions and conditions[revision] != selected:
            raise ValueError(f"Inconsistent benchmark conditions for source revision {revision}")
        conditions[revision] = selected
    return list(conditions.values())


def _throughput(roots: int, milliseconds: float, scope: str) -> dict[str, Any]:
    return {
        "value": roots * 1000 / milliseconds,
        "unit": "roots/s",
        "scope": scope,
        "statistic": "actual roots divided by historical median-of-run-median cycle seconds",
        "provenance": "derived from historical latency; not a new timing measurement",
    }


def _peak(resources: dict[str, Any] | None, revision: str, arm: str) -> dict[str, Any]:
    if resources is None:
        return {
            "allocated_mib": None,
            "reserved_mib": None,
            "status": "not measured in original latency runs",
        }
    matches = [
        row
        for row in resources["summary"]
        if row["source_revision"] == revision and row["arm"] == arm
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one measured resource summary for {revision}/{arm}")
    return {
        **matches[0]["peak_memory"],
        "status": "measured in separate resource supplement; not historical run peaks",
        "source_revision": revision,
        "resource_summary_key": f"{revision}/{arm}",
    }


def compact(
    path: Path, base_url: str, resources_path: Path | None = None, resources_url: str | None = None
) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    manifest = {
        key: copy.deepcopy(value)
        for key, value in raw.items()
        if key not in {"runs", "records", "preflight"}
    }
    manifest["manifest_schema"] = "tfm.graph-benchmark-manifest.v1"
    manifest["storage"] = (
        "Compact summaries/conditions only; full raw payload is referenced by URL and SHA256. Separate resource profiles are not historical run peaks."
    )
    manifest["raw_artifact"] = reference(path, f"{base_url.rstrip('/')}/{path.name}")
    resources = json.loads(resources_path.read_text()) if resources_path else None
    if resources_path:
        if not resources_url:
            raise ValueError("Resource supplement URL is required")
        manifest["resource_artifact"] = reference(resources_path, resources_url)
    if path.stem == "graph_batch_execution_l4":
        records = raw["records"]
        manifest["conditions"] = _conditions(records["cycle_before"] + records["cycle_after"])
        manifest["excluded_comparisons"] = {
            "resident_before_vs_after": "Withdrawn: all resident_before records identify the candidate revision and have no baseline source hash. Original logs do not establish a baseline. Candidate records and original payload remain archived unchanged."
        }
        manifest["limits"] = raw["limits"].replace(
            "Resident-only speed unchanged after warm-up. ", ""
        )
        manifest["candidate_resident_only"] = {
            "source_revision": raw["after_revision"],
            "statistic": "median of candidate run medians; no baseline comparison",
            "arms": {
                row["arm"]: statistics.median(
                    run["arms"][i]["step_ms_median"] for run in records["resident_after"]
                )
                for i, row in enumerate(records["resident_after"][0]["arms"])
            },
        }
        manifest["statistic"] = (
            "Median of three run medians per source for the verified cycle comparison; resident baseline comparison withdrawn."
        )
        for sharing, row in manifest["summary"].items():
            arm = next(
                arm for arm in records["cycle_before"][0]["arms"] if arm["sharing"] == sharing
            )
            roots = len(arm["final_prediction"])
            row["throughput"] = {
                stage: _throughput(
                    roots,
                    row["transfer_and_train_ms_median"][stage],
                    "blocking batch transfer+forward+backward+clipping+AdamW; construction and sampling excluded",
                )
                for stage in ("before", "after")
            }
        parity = manifest.get("deterministic_cuda_parity", {})
        if isinstance(parity, dict) and "driver_source" in parity:
            source = parity.pop("driver_source")
            parity["driver_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    elif path.stem == "graph_execution_views_l4":
        runs = raw["runs"]
        manifest["conditions"] = _conditions(runs)
        revisions = [condition["source_revision"] for condition in manifest["conditions"]]
        if (
            len(revisions) != 2
            or not revisions[0].startswith("85f47bb")
            or not revisions[1].startswith("1e73e84")
        ):
            raise ValueError("Historical execution-view before/after identity is ambiguous")
        manifest["source_roles"] = dict(zip(("before", "after"), revisions, strict=True))
        for row in manifest["summary"]:
            row["throughput"] = {}
            row["peak_memory"] = {}
            for stage, condition in zip(("before", "after"), manifest["conditions"], strict=True):
                row["throughput"][stage] = _throughput(
                    row["sample_count"],
                    row[f"{stage}_median_ms"]["cycle_ms"],
                    "CPU constructor+blocking transfer+forward"
                    + ("" if row["arm"] == "validation" else "+backward+clipping+AdamW"),
                )
                row["peak_memory"][stage] = _peak(
                    resources, condition["source_revision"], row["arm"]
                )
    elif path.stem == "graph_constructor_checks_l4":
        manifest["conditions"] = _conditions(raw["runs"])
        manifest["run_count"] = len(raw["runs"])
        manifest["preflight_run_count"] = len(raw["preflight"])
        for row in manifest["summary"]:
            for stage, metrics in row["stages"].items():
                metrics["throughput"] = _throughput(
                    row["effective_settings"]["root_count"],
                    metrics["cycle_ms"]["median_of_run_medians"],
                    "CPU constructor+blocking transfer+forward"
                    + ("" if row["arm"] == "validation" else "+backward+clipping+AdamW"),
                )
                metrics["peak_memory"] = _peak(
                    resources, manifest["sources"][stage]["revision"], row["arm"]
                )
    else:
        raise ValueError(f"Unsupported benchmark manifest: {path.name}")
    return manifest


def compact_resources(path: Path, url: str) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    result = {key: value for key, value in raw.items() if key not in {"runs", "order"}}
    result["manifest_schema"] = "tfm.graph-resource-supplement.v1"
    result["raw_artifact"] = reference(path, url)
    result["conditions"] = _conditions(raw["runs"])
    result["run_count"] = len(raw["runs"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-base-url", required=True)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--resources-url")
    args = parser.parse_args()
    inputs = [args.raw_dir / f"{name}.json" for name in NAMES]
    outputs = [args.output_dir / path.name for path in inputs]
    if args.resources:
        inputs.append(args.resources)
        outputs.append(args.output_dir / args.resources.name)
    for index, output in enumerate(outputs):
        for other in inputs + outputs[:index]:
            if output.resolve() == other.resolve() or (
                output.exists() and other.exists() and output.samefile(other)
            ):
                parser.error(f"Output path {output} would overwrite input or output {other}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        result = compact(
            args.raw_dir / f"{name}.json",
            args.artifact_base_url,
            args.resources,
            args.resources_url,
        )
        (args.output_dir / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.resources:
        result = compact_resources(args.resources, args.resources_url)
        (args.output_dir / args.resources.name).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
