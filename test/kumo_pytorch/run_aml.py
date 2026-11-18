"""Run the five AML prototype cases against pinned, downloaded HI-small data."""

import argparse
import hashlib
import json
import platform
import time
import traceback
from collections import Counter
from pathlib import Path

import kumo.pytorch.api as r
import pyarrow.parquet as pq
import torch

from .aml_inputs import SPLITS, load_selections, requests_for, verify_inputs
from .aml_source import (
    account_identity_map,
    aml_data_schema,
    prepare_aml_sources,
    prepared_dataset,
)
from .anchor_tables import write_probe_anchors
from .checkpoint_resume import check_resume
from .contracts import build_contract
from .data import ROOT
from .involvement import (
    source_reference,
    verify_anchors,
    verify_batch,
)
from .involvement import (
    verify_inputs as verify_task_inputs,
)
from .lifecycle import check_lifecycle
from .masked_values import (
    original_tokens,
)
from .masked_values import (
    verify_loss as verify_masked_loss,
)
from .masked_values import (
    verify_targets as verify_masked_targets,
)
from .models.aml import build_model
from .next_token import (
    verify_loss as verify_next_loss,
)
from .next_token import (
    verify_targets as verify_next_targets,
)
from .training_lifecycle import check_training


def prepare(data_dir, destination, selections, files):
    roots = sorted(
        {
            row["entity_id"]
            for version in selections.values()
            for groups in version.values()
            for rows in groups.values()
            for row in rows
        }
    )
    source = "datasets/aml-data/v8/hi-small/transactions.parquet"
    binding = "anchors/aml-data/v8/hi-small/_bindings/v1/accounts.parquet"
    checksums = {
        "transactions": files[source]["sha256"],
        "accounts_binding": files[binding]["sha256"],
    }
    if not destination.exists():
        manifest = prepare_aml_sources(
            data_dir / source,
            data_dir / binding,
            destination,
            "hi-small",
            expected_checksums=checksums,
            root_entity_ids=roots,
        )
    else:
        manifest = json.loads((destination / "manifest.json").read_text())
        if manifest["scope"]["complete_history_root_entity_ids"] != roots or any(
            manifest["sources"][key]["sha256"] != value
            for key, value in checksums.items()
        ):
            raise ValueError(
                "Existing prepared view has different sources or roots; choose a new --prepared-dir"
            )
    dataset = prepared_dataset(destination)  # Check every mapped output on reuse.
    identities = account_identity_map(destination)
    counts = Counter()
    for batch in pq.ParquetFile(destination / "account_events.parquet").iter_batches(
        columns=["account_id"]
    ):
        counts.update(batch.column(0).to_pylist())
    capacity = max(counts[identities[root]] for root in roots)
    if capacity < 1:
        raise ValueError(
            "Selected roots have no events; cannot validate a training step"
        )
    return dataset, identities, manifest, capacity


def run_case(
    dataset,
    identities,
    selections,
    selection,
    family,
    *,
    directory,
    capacity,
    batch_size,
    steps,
    threads,
    seed,
    expected_labels=None,
    input_reference=None,
):
    graph = family in {"graphsage", "combined"}
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    masked_checks = family == "pragma" and input_reference is not None
    next_checks = native and family == "nuformer" and input_reference is not None
    target_checks = (
        (not native and expected_labels is not None) or masked_checks or next_checks
    )
    version = selection["selector"]["anchor_version"]
    column = selection["anchor_binding"]["split_column"]
    groups = selections[version][column]
    anchor_index = {row["sample_id"]: row for rows in groups.values() for row in rows}
    requests = {
        split: requests_for(
            groups[split], identities, split=split, graph=graph, native=native
        )
        for split in SPLITS
    }
    contract = build_contract(
        aml_data_schema(), family, native=native, max_events=capacity
    )
    # Table-column encoders must not see events at/after the last training T.
    # Aggregate encoders fit on the selected training requests only.
    fit_cutoff = max(row["as_of_time"] // 1_000_000_000 - 1 for row in groups["train"])
    report = {
        "family": family,
        "selector": selection["selector"],
        "status": "running",
        "fit_split": "train",
        "fit_requests": len(requests["train"]),
        "encoder_cutoff_seconds": fit_cutoff,
        "max_events": capacity,
        "complete_epoch": False,
        "splits": {},
        "optimizer_updates": 0,
    }
    engine = r.DiskGraphEngine(index=directory / "index", num_threads=threads)
    from .refusals import case_failure, clean_refusal, executing_case

    try:
        plan = r.prepare(contract, dataset, engine)
    except Exception as error:
        if clean_refusal(error) is None:
            raise
        return {**report, **case_failure(error)}
    with plan, executing_case():
        anchors = write_probe_anchors(
            tuple(q for split in SPLITS for q in requests[split]),
            directory / "anchors.parquet",
        )
        task = plan.materialize(anchors, fit_split="train", encoder_cutoff=fit_cutoff)
        # Ten fields plus a separator per event; use a complete-source bound.
        model = build_model(
            task, selection, family, max_seq_len=11 * capacity, seed=seed
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
        fitted_before = task.fitted_state.fingerprint
        fitted_reference = (
            task.fitted_state.to_dict() if masked_checks or next_checks else None
        )
        report["fitted_state_sha256"] = fitted_before
        for split in SPLITS:
            training = split == "train"
            model.train(training)
            seen, batches, weight, numerator = [], 0, 0, 0.0
            evaluation = {"loss_sum": 0.0, "eligible": 0, "invalid": 0, "padding": 0}
            for batch in task.batches(
                split=split,
                batch_size=batch_size,
                seed=seed,
                epoch=0,
                output_format="model",
            ):
                if not native and input_reference is not None:
                    verify_task_inputs(
                        batch,
                        family,
                        anchor_index,
                        input_reference,
                        identities,
                        groups["train"],
                    )
                if masked_checks:
                    originals = original_tokens(
                        batch,
                        anchor_index,
                        input_reference,
                        fitted_reference,
                        identities,
                    )
                    verify_masked_targets(batch, originals)
                if next_checks:
                    verify_next_targets(
                        batch,
                        anchor_index,
                        input_reference,
                        fitted_reference,
                        identities,
                    )
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    result = model.training_step(batch)
                reference = None
                if not native and expected_labels is not None:
                    reference = verify_batch(batch, result, expected_labels)
                elif masked_checks:
                    reference = verify_masked_loss(batch, result)
                elif next_checks:
                    reference = verify_next_loss(batch, result)
                if reference is not None:
                    for key in evaluation:
                        evaluation[key] += reference[key]
                if not torch.isfinite(result.logits).all() or not torch.isfinite(
                    result.loss
                ):
                    raise ValueError(f"Nonfinite {split} outputs or objective")
                if training and result.weight:
                    before = [p.detach().clone() for p in model.parameters()]
                    result.loss.backward()
                    gradients = [
                        p.grad for p in model.parameters() if p.grad is not None
                    ]
                    if (
                        not gradients
                        or not all(torch.isfinite(g).all() for g in gradients)
                        or not any(torch.count_nonzero(g) for g in gradients)
                    ):
                        raise ValueError("Expected finite, nonzero training gradients")
                    optimizer.step()
                    if not any(
                        not torch.equal(old, p)
                        for old, p in zip(before, model.parameters(), strict=True)
                    ):
                        raise ValueError("Optimizer did not update any model parameter")
                    if not all(torch.isfinite(p).all() for p in model.parameters()):
                        raise ValueError("Nonfinite model parameters after update")
                    report["optimizer_updates"] += 1
                seen.extend(
                    sid
                    for sid, padding in zip(
                        batch.metadata["sample_ids"],
                        batch.metadata["is_padding"].tolist(),
                        strict=True,
                    )
                    if not padding
                )
                batches += 1
                weight += result.weight
                numerator += result.loss.item() * result.weight
                if training and batches == steps:
                    break
            expected = [request.sample_id for request in requests[split]]
            if seen != expected[: len(seen)] or (not training and seen != expected):
                raise ValueError(f"Delivered {split} anchor identities/order differ")
            if not batches or not weight:
                raise ValueError(f"No usable targets in selected {split} anchors")
            report["splits"][split] = {
                "batches": batches,
                "sample_ids": seen,
                "target_weight": weight,
                "mean_loss": numerator / weight,
                "selected_anchors": len(expected),
            }
            if target_checks:
                evaluation["mean_loss"] = (
                    evaluation["loss_sum"] / evaluation["eligible"]
                    if evaluation["eligible"]
                    else None
                )
                report["splits"][split]["independent_evaluation"] = evaluation
        if not report["optimizer_updates"]:
            raise ValueError("No training update completed")
        if task.fitted_state.fingerprint != fitted_before:
            raise ValueError("Evaluation changed the training fitted state")
        report["lifecycle"] = check_lifecycle(
            plan,
            task,
            model,
            dataset,
            requests["valid"] + requests["test"],
            selection,
            family,
            directory=directory / "lifecycle",
            index=directory / "index",
            batch_size=batch_size,
            threads=threads,
            seed=seed,
            max_seq_len=11 * capacity,
        )
    with executing_case():
        report["finetuning"] = check_training(
            directory / "lifecycle",
            requests_for(
                selections[version]["forecast_split"]["train"],
                identities,
                split="train",
                graph=graph,
                native=False,
            ),
        )
        report["mid_epoch_resume"] = check_resume(
            directory / "lifecycle", requests["train"]
        )
    report["status"] = "passed"
    report["target_and_loss_checks"] = "passed" if target_checks else "not_checked"
    report["input_membership_and_counts"] = (
        "passed"
        if (not native and input_reference is not None) or masked_checks or next_checks
        else "not_checked"
    )
    return report


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument(
        "--prepared-dir", type=Path, default=ROOT / ".data/prepared/hi-small-probe"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new run directory; never overwritten",
    )
    parser.add_argument("--anchors-per-split", type=positive, default=4)
    parser.add_argument("--batch-size", type=positive, default=4)
    parser.add_argument("--steps", type=positive, default=1)
    parser.add_argument("--threads", type=positive, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    report = {
        "status": "running",
        "scope": "bounded real AML HI-small prototype; not a full epoch",
        "complete_epoch": False,
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "kumo_api": r.__file__,
            "device": "cpu",
        },
        "cases": [],
    }
    try:
        print("Verifying downloaded HI-small files and selecting anchors", flush=True)
        files = verify_inputs(args.data_dir)
        selections = load_selections(args.data_dir, limit=args.anchors_per_split)
        print(
            "Reading independent AML task references from source transactions",
            flush=True,
        )
        forecast_rows = [
            row for rows in selections["v1"]["forecast_split"].values() for row in rows
        ]
        reference = source_reference(
            args.data_dir / "datasets/aml-data/v8/hi-small/transactions.parquet",
            [
                row["entity_id"]
                for groups in selections["v1"].values()
                for rows in groups.values()
                for row in rows
            ],
            include_values=True,
        )
        expected_labels = verify_anchors(forecast_rows, reference)
        verify_anchors(
            [
                row
                for rows in selections["v2"]["forecast_split"].values()
                for row in rows
            ],
            reference,
        )
        report["involvement_reference"] = {
            "status": "passed",
            "source_rows": reference["source_rows"],
            "split_boundaries_seconds": reference["cuts"],
            "observation_end_seconds": reference["end"],
            "unique_anchors": len(expected_labels),
            "positive_labels": sum(expected_labels.values()),
            "scope": "selected anchors; classification metrics and thresholds remain unspecified",
        }
        (args.output / "anchors.json").write_text(
            json.dumps(selections, indent=2) + "\n"
        )
        print("Preparing complete histories for the selected roots", flush=True)
        dataset, identities, mapping, capacity = prepare(
            args.data_dir, args.prepared_dir, selections, files
        )
        report["mapping"] = mapping
        report["inputs"] = list(files.values())
        report["selection_sha256"] = hashlib.sha256(
            (args.output / "anchors.json").read_bytes()
        ).hexdigest()
        print(
            f"Prepared {mapping['tables']['transactions']['rows']} transactions; history bound {capacity} events",
            flush=True,
        )
        workloads = json.loads((ROOT / "workloads/aml.json").read_text())
        for selection in workloads:
            for family in selection["models"]:
                name = f"{selection['selector']['workload_id']}-{family}"
                directory = args.output / name
                directory.mkdir()
                print(f"Running {name}", flush=True)
                case_start = time.perf_counter()
                try:
                    result = run_case(
                        dataset,
                        identities,
                        selections,
                        selection,
                        family,
                        directory=directory,
                        capacity=capacity,
                        batch_size=args.batch_size,
                        steps=args.steps,
                        threads=args.threads,
                        seed=args.seed,
                        expected_labels=expected_labels,
                        input_reference=reference,
                    )
                except Exception as error:
                    from .refusals import case_failure

                    result = case_failure(error, allow_refusal=False)
                result.update(name=name, seconds=time.perf_counter() - case_start)
                report["cases"].append(result)
                (directory / "result.json").write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n"
                )
                print(
                    f"{name}: {result['status']} ({result.get('outcome', 'executed')})"
                    + (f" — {result['error']}" if "error" in result else ""),
                    flush=True,
                )
        report["status"] = (
            "passed"
            if all(case["status"] == "passed" for case in report["cases"])
            else "failed"
        )
    except Exception as error:
        report.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        print(f"AML preparation failed: {error}", flush=True)
    finally:
        report["seconds"] = time.perf_counter() - started
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
