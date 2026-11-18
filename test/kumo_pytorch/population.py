"""Full-population AML loader acceptance, independent of training and timing runs."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import traceback
from collections import Counter
from itertools import islice
from pathlib import Path

import kumo.pytorch.api as r
import pyarrow.parquet as pq
import torch

from .aml_inputs import SPLITS, verify_inputs
from .aml_source import (
    account_identity_map,
    aml_data_schema,
    prepare_aml_sources,
    prepared_dataset,
)
from .contracts import GRAPH_TIMESTAMP_FIELD, build_contract
from .data import ROOT
from .nightly import selected_cases
from .population_anchors import add_identity, project_population, projected_rows
from .run_aml import positive

PDF_CASES = {
    ("account-involvement-next24h", "combined"): "a001",
    ("masked-value-pretrain", "pragma"): "a017",
    ("next-token-pretrain", "nuformer"): "a018",
}


def write_report(directory, report):
    # Durable partial evidence if a later batch or native process fails.
    temporary = directory / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "report.json")


def check_batch(batch, expected, *, native, family, state):
    ids = list(batch.metadata["sample_ids"])
    rows = list(islice(expected, len(ids)))
    assert ids and ids == [row["sample_id"] for row in rows], (
        "Missing, duplicate or reordered anchors"
    )
    assert not batch.metadata["is_padding"].any(), "Unexpected sample padding"
    assert batch.metadata["entity_ids"].tolist() == [
        row["entity_id"] for row in rows
    ], "Entity identities differ"
    assert batch.metadata["as_of_time"].tolist() == [
        row["as_of_time"] for row in rows
    ], "Execution cutoffs differ"
    assert list(batch.metadata["request_metadata"]) == [
        {
            name: row[name]
            for name in (
                "prediction_time_ns",
                "registered_entity_id",
                "anchor_row_position",
                "split",
            )
        }
        for row in rows
    ], "Original prediction clocks, identities, positions or splits differ"

    def finite(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all(), "Nonfinite emitted values"
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)

    finite(batch.inputs)
    finite(batch.targets)
    target_name = (
        ("masked_values" if family == "pragma" else "next_token") if native else "label"
    )
    assert set(batch.targets) == {target_name}, "Wrong task targets"
    target = batch.targets[target_name]
    assert target["values"].shape == target["valid"].shape
    assert target["valid"].dtype == torch.bool
    if not native:
        labels = [row["label"] for row in rows]
        assert target["valid"].flatten().tolist() == [v is not None for v in labels]
        assert all(
            label is None or value == label
            for value, label in zip(
                target["values"].flatten().tolist(), labels, strict=True
            )
        ), "Stored labels differ"
    else:
        history = batch.inputs["history"]
        assert target["valid"].shape == history["valid"].shape
        assert not (target["valid"] & ~history["valid"]).any(), (
            "Padding became a target"
        )
        assert (target["values"][~target["valid"]] == -100).all()
        assert (target["values"][target["valid"]] >= 0).all()
        if family == "pragma":
            mask_id = state["spaces"]["history"]["roles"]["mask"]
            assert torch.equal(target["valid"], history["value_ids"] == mask_id)
        else:
            assert torch.equal(target["valid"], history["valid"])
            valid = history["valid"][:, 1:]
            assert torch.equal(
                target["values"][:, :-1][valid], history["value_ids"][:, 1:][valid]
            )
    if "graph" in batch.inputs:
        transactions = batch.inputs["graph"]["nodes"]["transactions"]
        times = transactions["features"][GRAPH_TIMESTAMP_FIELD]["values"].flatten()
        cutoffs = batch.metadata["as_of_time"][transactions["batch"]]
        assert (times <= cutoffs).all(), "Graph history leaked beyond the v2 cutoff"
    return ids, int(target["valid"].sum())


def run_case(
    dataset,
    identities,
    anchor_path,
    selection,
    family,
    *,
    directory,
    capacity,
    mode,
    smoke_anchors=4,
    batch_size=4,
    threads=1,
):
    if mode not in {"smoke", "full"}:
        raise ValueError("Population mode must be smoke or full")
    directory.mkdir(parents=True, exist_ok=False)
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    graph = family in {"graphsage", "combined"}
    report = {
        "status": "running",
        "mode": mode,
        "family": family,
        "selector": selection["selector"],
        "full_population_verified": False,
        "pdf_case": PDF_CASES.get((selection["selector"]["workload_id"], family)),
        "splits": {},
        "stage": "project_anchors",
        "history_capacity": capacity,
        "model_training_steps": 0,
        "batch_size": batch_size,
        "threads": threads,
        "seed": 0,
        "smoke_anchors": smoke_anchors if mode == "smoke" else None,
    }
    write_report(directory, report)
    try:
        projection = directory / "anchors.parquet"
        anchors, manifest = project_population(
            anchor_path,
            projection,
            identities,
            graph=graph,
            native=native,
            limit=None if mode == "full" else smoke_anchors,
        )
        report["population"] = manifest
        report["splits"] = {
            s: {
                "status": "pending",
                "expected": manifest["counts"][s],
                "delivered": 0,
                "batches": 0,
                "eligible_targets": 0,
            }
            for s in SPLITS
        }
        contract = build_contract(
            aml_data_schema(), family, native=native, max_events=capacity
        )
        (directory / "contract.json").write_text(contract.to_json())
        report["stage"] = "prepare_and_fit"
        write_report(directory, report)
        with (
            r.prepare(
                contract,
                dataset,
                r.DiskGraphEngine(index=directory / "index", num_threads=threads),
            ) as plan,
            plan.materialize(
                anchors,
                fit_split="train",
                encoder_cutoff=manifest["encoder_cutoff_seconds"],
            ) as task,
        ):
            frozen = task.fitted_state.fingerprint
            report["fitted_state_sha256"] = frozen
            task.to_artifact().save(directory / "fitted")
            state = task.fitted_state.to_dict()
            assert {s: task.count(s) for s in SPLITS} == manifest["counts"], (
                "Native population counts differ"
            )
            for split in SPLITS:
                report["stage"] = f"iterate_{split}"
                result = report["splits"][split]
                result["status"] = "running"
                write_report(directory, report)
                expected = projected_rows(projection, split)
                digest = hashlib.sha256()
                for batch in task.batches(
                    split=split,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    seed=0,
                    epoch=0,
                    output_format="model",
                ):
                    ids, eligible = check_batch(
                        batch, expected, native=native, family=family, state=state
                    )
                    for sample_id in ids:
                        add_identity(digest, sample_id)
                    result["delivered"] += len(ids)
                    result["batches"] += 1
                    result["eligible_targets"] += eligible
                    if result["batches"] % 100 == 0:
                        write_report(directory, report)
                assert next(expected, None) is None, "Iteration dropped anchors"
                assert result["delivered"] == result["expected"], "Incomplete split"
                result["ordered_sample_id_sha256"] = digest.hexdigest()
                assert digest.hexdigest() == manifest["ordered_sample_id_sha256"][split]
                assert task.fitted_state.fingerprint == frozen, (
                    "Batch iteration changed fitted state"
                )
                assert result["eligible_targets"] > 0, "Split has no eligible targets"
                result["status"] = "passed"
                write_report(directory, report)
        report.update(
            status="passed", stage="complete", full_population_verified=mode == "full"
        )
    except Exception as error:  # noqa: BLE001 - persist failed acceptance without hiding it
        from .refusals import case_failure

        outcome = case_failure(
            error, allow_refusal=report["stage"] == "prepare_and_fit"
        )
        for split in report["splits"].values():
            if outcome.get("outcome") == "refused":
                split["status"] = "not_executed"
            elif split["status"] == "running":
                split["status"] = "failed"
        report.update(outcome)
        if outcome.get("outcome") == "refused":
            report["stage"] = "refused"
    write_report(directory, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--smoke-anchors", type=positive)
    parser.add_argument("--batch-size", type=positive, default=4)
    parser.add_argument("--threads", type=positive, default=1)
    parser.add_argument(
        "--case", help="Run one catalog binding; omitted runs all bindings"
    )
    args = parser.parse_args(argv)
    cases = selected_cases(args.case)
    if args.mode == "full" and args.smoke_anchors is not None:
        parser.error("--smoke-anchors cannot limit a full-population run")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    report = {
        "status": "running",
        "mode": args.mode,
        "stage": "verify_inputs",
        "full_population_verified": False,
        "cases": [
            {"selector": s["selector"], "family": f, "status": "pending"}
            for s, f in cases
        ],
        "scope": "delivery, stored targets, native target masks, metadata and fitted-state stability; no model training",
        "versions": {
            p: importlib.metadata.version(p) for p in ("torch", "pyarrow", "diskgraph")
        },
    }
    try:
        report["revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        report["dirty_checkout"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
        write_report(args.output, report)
        files = verify_inputs(args.data_dir)
        report["inputs"] = list(files.values())
        report["stage"] = "prepare_sources"
        write_report(args.output, report)
        tx = "datasets/aml-data/v8/hi-small/transactions.parquet"
        binding = "anchors/aml-data/v8/hi-small/_bindings/v1/accounts.parquet"
        prepared = args.output / "prepared"
        # Full source histories are retained in both modes; only smoke anchors are bounded.
        report["mapping"] = prepare_aml_sources(
            args.data_dir / tx,
            args.data_dir / binding,
            prepared,
            "hi-small",
            expected_checksums={
                "transactions": files[tx]["sha256"],
                "accounts_binding": files[binding]["sha256"],
            },
        )
        dataset, identities = prepared_dataset(prepared), account_identity_map(prepared)
        counts = Counter()
        for batch in pq.ParquetFile(prepared / "account_events.parquet").iter_batches(
            columns=["account_id"]
        ):
            counts.update(batch.column(0).to_pylist())
        capacity = max(counts.values())
        del counts
        for i, (selection, family) in enumerate(cases):
            name = f"{selection['selector']['workload_id']}-{family}"
            version = selection["selector"]["anchor_version"]
            print(f"{args.mode}: {name}", flush=True)
            report["stage"] = name
            write_report(args.output, report)
            report["cases"][i] = run_case(
                dataset,
                identities,
                args.data_dir
                / f"anchors/aml-data/v8/hi-small/account-daily/{version}/anchors.parquet",
                selection,
                family,
                directory=args.output / name,
                capacity=capacity,
                mode=args.mode,
                smoke_anchors=args.smoke_anchors or 4,
                batch_size=args.batch_size,
                threads=args.threads,
            )
            write_report(args.output, report)
        report["status"] = (
            "passed"
            if all(c["status"] == "passed" for c in report["cases"])
            else "failed"
        )
        report["full_population_verified"] = (
            report["status"] == "passed"
            and args.mode == "full"
            and all(
                case.get("full_population_verified", False) for case in report["cases"]
            )
        )
        report["stage"] = "complete"
    except Exception as error:  # noqa: BLE001 - retain preparation failures as failures
        report.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
    write_report(args.output, report)
    lines = [
        "### AML population acceptance",
        "",
        f"Mode: **{args.mode}**; status: **{report['status']}**",
        "",
        "| Workload / model | Status | Stage | Delivered / expected |",
        "| --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        splits = case.get("splits", {}).values()
        delivered, expected = (
            sum(s["delivered"] for s in splits),
            sum(s["expected"] for s in splits),
        )
        lines.append(
            f"| {case['selector']['workload_id']} / {case['family']} | {case['status']} ({case.get('outcome', 'executed')}) | {case.get('stage', 'not started')} | {delivered} / {expected} |"
        )
    for case in report["cases"]:
        if "error" in case:
            lines.extend(
                [
                    "",
                    f"**{case['family']} / {case['selector']['workload_id']}**",
                    "```text",
                    case["error"],
                    "```",
                ]
            )
    if "error" in report:
        lines.extend(["", "```text", report["error"], "```"])
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"{report['status']}: {args.output / 'report.md'}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
