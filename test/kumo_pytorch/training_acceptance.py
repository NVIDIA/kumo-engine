"""Real-AML, 100-update CPU training acceptance for all five model/task bindings."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import traceback
from contextlib import contextmanager
from pathlib import Path

import kumo.pytorch.api as r
import torch

from .aml_inputs import load_selections, requests_for, verify_inputs
from .aml_source import aml_data_schema
from .anchor_tables import write_probe_anchors
from .contracts import build_contract
from .data import ROOT
from .involvement import verify_batch
from .lifecycle import input_digest
from .masked_values import verify_loss as verify_masked_loss
from .models.aml import build_model
from .next_token import verify_loss as verify_next_loss
from .population import PDF_CASES, write_report
from .run_aml import positive, prepare

STEPS = 100
CASES = {
    ("account-involvement-next24h", "graphsage"),
    ("account-involvement-next24h", "nuformer"),
    ("account-involvement-next24h", "combined"),
    ("masked-value-pretrain", "pragma"),
    ("next-token-pretrain", "nuformer"),
}


def training_cases(selections):
    cases = [
        (selection, family)
        for selection in selections
        for family in selection["models"]
    ]
    identities = [
        (selection["selector"]["workload_id"], family) for selection, family in cases
    ]
    assert len(identities) == len(CASES) and set(identities) == CASES, (
        "Training acceptance requires each of the five AML model/workload combinations exactly once"
    )
    return cases


@contextmanager
def deterministic_cpu(seed):
    threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(threads)
        torch.set_default_dtype(dtype)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def measured_loss(model, batch, workload, labels, *, audit=False):
    assert not batch.metadata["is_padding"].any(), "Unexpected sample padding"
    result = model.training_step(batch)
    assert result.weight >= 0, "Negative eligible-target count"
    assert torch.isfinite(result.loss), "Nonfinite loss"
    assert torch.isfinite(result.logits).all(), "Nonfinite logits"
    assert result.loss.device.type == "cpu", "Training acceptance must run on CPU"
    if audit:
        if workload == "account-involvement-next24h":
            verify_batch(batch, result, labels)
        elif workload == "masked-value-pretrain":
            verify_masked_loss(batch, result)
        elif workload == "next-token-pretrain":
            verify_next_loss(batch, result)
        else:
            raise ValueError(f"Unknown training objective: {workload}")
    return result


def train_repetition(
    task, requests, selection, family, *, directory, capacity, seed, batch_size
):
    """Run every trainable parameter unchanged; unused parameters fail acceptance."""
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "completed_updates": 0,
        "curve": [],
        "parameters": {},
        "zero_target_batches": [],
    }
    write_report(directory, report)
    try:
        with deterministic_cpu(seed):
            workload = selection["selector"]["workload_id"]
            assert requests and all(q.split == "train" for q in requests)
            labels = (
                {q.sample_id: q.targets[0].value for q in requests}
                if workload == "account-involvement-next24h"
                else {}
            )
            frozen = task.fitted_state.fingerprint
            model = build_model(
                task, selection, family, dim=16, max_seq_len=11 * capacity, seed=seed
            ).cpu()
            parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
            assert parameters, "Model has no trainable parameters"
            initial = {n: p.detach().clone() for n, p in parameters.items()}
            report["initial_model_sha256"] = input_digest(model.state_dict())
            report["parameters"] = {
                n: {
                    "shape": list(p.shape),
                    "gradient_steps": 0,
                    "nonzero_gradient_steps": 0,
                }
                for n, p in parameters.items()
            }
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
            model.eval()
            probe = None
            with torch.no_grad():
                for candidate in task.batches(
                    split="train",
                    output_format="model",
                    batch_size=batch_size,
                    seed=seed,
                    epoch=0,
                    shuffle=False,
                    drop_last=False,
                ):
                    baseline = measured_loss(
                        model, candidate, workload, labels, audit=True
                    )
                    if baseline.weight:
                        probe = candidate
                        break
                assert probe is not None, "No eligible targets for the training probe"
                report["initial_loss"] = baseline.loss.item()
                report["probe_eligible"] = baseline.weight
            probe_hash = input_digest((probe.inputs, probe.targets, probe.metadata))
            report["probe_sha256"] = probe_hash
            epoch = 0
            while report["completed_updates"] < STEPS:
                seen = []
                epoch_start = report["completed_updates"]
                for batch in task.batches(
                    split="train",
                    output_format="model",
                    batch_size=batch_size,
                    seed=seed,
                    epoch=epoch,
                    shuffle=True,
                    drop_last=False,
                ):
                    seen.extend(batch.metadata["sample_ids"])
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    batch_hash = input_digest(
                        (batch.inputs, batch.targets, batch.metadata)
                    )
                    result = measured_loss(model, batch, workload, labels)
                    if not result.weight:
                        report["zero_target_batches"].append(
                            {
                                "epoch": epoch,
                                "sample_ids": list(batch.metadata["sample_ids"]),
                                "batch_sha256": batch_hash,
                            }
                        )
                        continue  # Retain the anchors; an empty objective cannot update weights.
                    result.loss.backward()
                    for name, parameter in parameters.items():
                        grad = parameter.grad
                        if grad is not None:
                            assert torch.isfinite(grad).all(), (
                                f"Nonfinite gradient: {name}"
                            )
                            report["parameters"][name]["gradient_steps"] += 1
                            report["parameters"][name]["nonzero_gradient_steps"] += int(
                                bool(torch.count_nonzero(grad))
                            )
                    optimizer.step()
                    for name, parameter in parameters.items():
                        assert torch.isfinite(parameter).all(), (
                            f"Nonfinite parameter: {name}"
                        )
                    assert (
                        input_digest((batch.inputs, batch.targets, batch.metadata))
                        == batch_hash
                    ), "Training mutated its inputs/targets"
                    report["completed_updates"] += 1
                    report["curve"].append(
                        {
                            "step": report["completed_updates"],
                            "epoch": epoch,
                            "loss": result.loss.item(),
                            "eligible": result.weight,
                            "sample_ids": list(batch.metadata["sample_ids"]),
                            "batch_sha256": batch_hash,
                            "model_sha256": input_digest(model.state_dict()),
                        }
                    )
                    if report["completed_updates"] % 10 == 0:
                        write_report(directory, report)
                    if report["completed_updates"] == STEPS:
                        break
                assert seen, "Training iterator emitted no batches"
                assert report["completed_updates"] > epoch_start, (
                    "No eligible training targets in epoch"
                )
                if report["completed_updates"] < STEPS:
                    assert len(seen) == len(requests) and set(seen) == {
                        q.sample_id for q in requests
                    }, "Training epoch omitted or duplicated anchors"
                epoch += 1
            model.eval()
            with torch.no_grad():
                final = measured_loss(model, probe, workload, labels, audit=True)
                report["final_loss"] = final.loss.item()
                assert final.weight == report["probe_eligible"], "Probe targets changed"
            assert (
                input_digest((probe.inputs, probe.targets, probe.metadata))
                == probe_hash
            )
            assert task.fitted_state.fingerprint == frozen, (
                "Training refitted its inputs"
            )
            for name, parameter in parameters.items():
                report["parameters"][name].update(
                    updated=not torch.equal(parameter, initial[name]),
                    max_abs_change=float(
                        (parameter.detach() - initial[name]).abs().max()
                    ),
                )
            report["checks"] = {
                "100_updates": report["completed_updates"] == STEPS,
                "every_parameter_has_nonzero_gradient": all(
                    p["nonzero_gradient_steps"] > 0
                    for p in report["parameters"].values()
                ),
                "every_parameter_updated": all(
                    p["updated"] for p in report["parameters"].values()
                ),
                "loss_decreased": report["final_loss"] < report["initial_loss"],
            }
            report["final_model_sha256"] = input_digest(model.state_dict())
            report["final_optimizer_sha256"] = input_digest(optimizer.state_dict())
            report["final_rng_sha256"] = input_digest(torch.get_rng_state())
            report["status"] = "passed" if all(report["checks"].values()) else "failed"
    except Exception as error:  # noqa: BLE001 - preserve failed measurements, never accept them
        report.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
    write_report(directory, report)
    with (directory / "loss-curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("step", "epoch", "loss", "eligible"),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(report["curve"])
    return report


def compare_repetitions(left, right):
    """Require a complete, exact CPU replay, not merely two similar final losses."""
    keys = (
        "initial_model_sha256",
        "probe_sha256",
        "initial_loss",
        "curve",
        "zero_target_batches",
        "parameters",
        "final_loss",
        "final_model_sha256",
        "final_optimizer_sha256",
        "final_rng_sha256",
    )
    return left["completed_updates"] == right["completed_updates"] == STEPS and all(
        k in left and k in right and left[k] == right[k] for k in keys
    )


def run_case(
    dataset,
    identities,
    selections,
    selection,
    family,
    *,
    directory,
    capacity,
    seed=17,
    batch_size=4,
):
    directory.mkdir(parents=True, exist_ok=False)
    workload = selection["selector"]["workload_id"]
    report = {
        "status": "running",
        "case_id": f"{workload}-{family}",
        "pdf_case": PDF_CASES.get((workload, family)),
        "family": family,
        "selector": selection["selector"],
        "device": "cpu",
        "threads": 1,
        "seed": seed,
        "steps_per_repetition": STEPS,
        "repetitions": [],
        "batch_size": batch_size,
        "model_dim": 16,
        "history_capacity": capacity,
        "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 0},
    }
    write_report(directory, report)
    try:
        assert (workload, family) in CASES, "Unselected training combination"
        graph = family in {"graphsage", "combined"}
        native = workload != "account-involvement-next24h"
        groups = selections[selection["selector"]["anchor_version"]][
            selection["anchor_binding"]["split_column"]
        ]
        rows = groups["train"]
        requests = requests_for(
            rows, identities, split="train", graph=graph, native=native
        )
        report["training_anchors"] = rows
        report["encoder_cutoff_seconds"] = (
            max(row["as_of_time"] // 10**9 for row in rows) - 1
        )
        contract = build_contract(
            aml_data_schema(), family, native=native, max_events=capacity
        )
        (directory / "contract.json").write_text(contract.to_json())
        anchors = write_probe_anchors(requests, directory / "anchors.parquet")
        with (
            r.prepare(
                contract,
                dataset,
                r.DiskGraphEngine(index=directory / "index", num_threads=1),
            ) as plan,
            plan.materialize(
                anchors,
                fit_split="train",
                encoder_cutoff=report["encoder_cutoff_seconds"],
            ) as task,
        ):
            report["fitted_state_sha256"] = task.fitted_state.fingerprint
            for repeat in range(2):
                report["repetitions"].append(
                    train_repetition(
                        task,
                        requests,
                        selection,
                        family,
                        directory=directory / f"repeat-{repeat}",
                        capacity=capacity,
                        seed=seed,
                        batch_size=batch_size,
                    )
                )
                write_report(directory, report)
        report["reproducible"] = compare_repetitions(*report["repetitions"])
        report["status"] = (
            "passed"
            if report["reproducible"]
            and all(x["status"] == "passed" for x in report["repetitions"])
            else "failed"
        )
    except Exception as error:  # noqa: BLE001 - retain refused versus executed outcomes
        from .refusals import case_failure

        report.update(
            case_failure(error, allow_refusal="fitted_state_sha256" not in report)
        )
    write_report(directory, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors-per-split", type=positive, default=32)
    parser.add_argument("--batch-size", type=positive, default=4)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "scope": "100 CPU updates, repeated twice, on bounded real AML training anchors; not full-population training",
        "cases": [],
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
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
        report["versions"] = {
            p: importlib.metadata.version(p) for p in ("torch", "pyarrow", "diskgraph")
        }
        report["test_source_sha256"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                ROOT / "contracts.py",
                ROOT / "involvement.py",
                ROOT / "masked_values.py",
                ROOT / "next_token.py",
                ROOT / "workloads/aml.json",
            ]
            + sorted((ROOT / "models").rglob("*.py"))
        }
        write_report(args.output, report)
        cases = training_cases(json.loads((ROOT / "workloads/aml.json").read_text()))
        files = verify_inputs(args.data_dir)
        report["inputs"] = list(files.values())
        selections = load_selections(args.data_dir, limit=args.anchors_per_split)
        dataset, identities, _, capacity = prepare(
            args.data_dir, args.output / "prepared", selections, files
        )
        for selection, family in cases:
            report["cases"].append(
                run_case(
                    dataset,
                    identities,
                    selections,
                    selection,
                    family,
                    directory=args.output
                    / f"{selection['selector']['workload_id']}-{family}",
                    capacity=capacity,
                    batch_size=args.batch_size,
                )
            )
            write_report(args.output, report)
        report["status"] = (
            "passed"
            if all(c["status"] == "passed" for c in report["cases"])
            else "failed"
        )
    except Exception as error:  # noqa: BLE001 - retain preparation errors in the CI artifact
        report.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
    write_report(args.output, report)
    lines = [
        "### Real-AML CPU training acceptance",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Case | Status | Reproducible | Updates per repeat | Initial / final probe loss |",
        "| --- | --- | --- | --- | --- |",
    ]
    if "revision" in report:
        lines.insert(
            4,
            f"Revision: `{report['revision']}`; dirty checkout: `{report['dirty_checkout']}`.\n",
        )
    for case in report["cases"]:
        repeats = case["repetitions"]
        lines.append(
            f"| {case['case_id']} | {case['status']} ({case.get('outcome', 'executed')}) | {case.get('reproducible', 'not measured')} | {[x['completed_updates'] for x in repeats]} | {[(x.get('initial_loss'), x.get('final_loss')) for x in repeats]} |"
        )
        for repeat, result in enumerate(repeats):
            failed = [
                n
                for n, p in result["parameters"].items()
                if not p.get("updated", False) or not p["nonzero_gradient_steps"]
            ]
            if failed and result["completed_updates"] == STEPS:
                lines.extend(
                    [
                        "",
                        f"{case['case_id']} repeat {repeat}: unchanged/no-gradient parameters: "
                        + ", ".join(failed),
                    ]
                )
        for result in (case, *repeats):
            if "error" in result:
                lines.extend(["", "```text", result["error"], "```"])
    if "error" in report:
        lines.extend(["", "```text", report["error"], "```"])
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"{report['status']}: {args.output / 'report.md'}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
