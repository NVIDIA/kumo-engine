# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared 100-update CPU training acceptance; real-data runs use nightly run."""

import csv
import traceback
from contextlib import contextmanager

import torch

from .bindings import binding_for
from .bindings import selection as binding_selection
from .bindings import sequence_capacity, task_definition
from .involvement import verify_batch
from .lifecycle import input_digest
from .masked_values import verify_loss as verify_masked_loss
from .models import build_model
from .next_token import verify_loss as verify_next_loss
from .population import verify_batch_for, write_report

STEPS = 100


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
    outputs = (
        result.logits.values() if isinstance(result.logits, dict) else (result.logits,)
    )
    assert all(torch.isfinite(value).all() for value in outputs), "Nonfinite logits"
    assert result.loss.device.type == "cpu", "Training acceptance must run on CPU"
    kind = task_definition(workload)["kind"]
    if audit:
        if isinstance(result.logits, dict):
            from .targets import verify_stored_loss

            verify_stored_loss(workload, batch, result)
        elif kind == "stored":
            verify_batch(batch, result, labels)
        elif kind == "masked_values":
            verify_masked_loss(batch, result)
        elif kind == "next_token":
            verify_next_loss(batch, result)
        else:
            raise ValueError(f"Unknown training objective: {workload}")
    return result


def train_repetition(
    task,
    requests,
    selection,
    family,
    *,
    directory,
    capacity,
    seed,
    batch_size,
    expected_rows=None,
    save_model=False,
    features=None,
):
    """Run every trainable parameter unchanged; unused parameters fail acceptance."""
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "completed_updates": 0,
        "observed_train_anchors": 0,
        "curve": [],
        "parameters": {},
        "zero_target_batches": [],
    }
    write_report(directory, report)
    try:
        with deterministic_cpu(seed):
            workload = selection["selector"]["workload_id"]
            binding = (
                binding_for(selection, family) if expected_rows is not None else None
            )
            objective = task_definition(binding) if binding else workload
            if expected_rows is not None:
                assert requests is None and expected_rows
                assert all(row["split"] == "train" for row in expected_rows)
                expected_ids = [row["sample_id"] for row in expected_rows]
                labels = {
                    row["sample_id"]: row["label"]
                    for row in expected_rows
                    if "label" in row
                }
            else:
                assert requests and all(q.split == "train" for q in requests)
                expected_ids = [q.sample_id for q in requests]
                labels = (
                    {q.sample_id: q.targets[0].value for q in requests}
                    if workload == "account-involvement-next24h"
                    else {}
                )
            frozen = task.fitted_state.fingerprint
            stored_rows = {row["sample_id"]: row for row in expected_rows or ()}
            state = task.fitted_state.to_dict() if expected_rows is not None else None

            def verify_source(batch):
                if features is not None:
                    features(batch, phase="training")
                if expected_rows is not None:
                    rows = [
                        stored_rows[sample] for sample in batch.metadata["sample_ids"]
                    ]
                    verify_batch_for(binding, batch, iter(rows), state)

            model = build_model(
                task,
                selection,
                family,
                dim=16,
                max_seq_len=sequence_capacity(binding_for(selection, family), capacity),
                seed=seed,
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
                    report["observed_train_anchors"] += len(
                        candidate.metadata["sample_ids"]
                    )
                    verify_source(candidate)
                    baseline = measured_loss(
                        model, candidate, objective, labels, audit=True
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
                    report["observed_train_anchors"] += len(
                        batch.metadata["sample_ids"]
                    )
                    verify_source(batch)
                    seen.extend(batch.metadata["sample_ids"])
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    batch_hash = input_digest(
                        (batch.inputs, batch.targets, batch.metadata)
                    )
                    result = measured_loss(model, batch, objective, labels, audit=True)
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
                    assert len(seen) == len(expected_ids) and set(seen) == set(
                        expected_ids
                    ), "Training epoch omitted or duplicated anchors"
                epoch += 1
            model.eval()
            with torch.no_grad():
                report["observed_train_anchors"] += len(probe.metadata["sample_ids"])
                verify_source(probe)
                final = measured_loss(model, probe, objective, labels, audit=True)
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
            if save_model and report["status"] == "passed":
                torch.save(model.state_dict(), directory / "model.pt")
                report["checkpoint"] = "model.pt"
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


def train_task(
    task,
    binding,
    projection,
    population,
    directory,
    report,
    *,
    capacity,
    batch_size,
    seed=17,
    features=None,
):
    """Shared strict training and checkpoint evaluation for any registered task."""
    from .population_anchors import projected_rows

    selected, family = binding_selection(binding), binding["representation"]
    objective = task_definition(binding)
    rows = list(projected_rows(projection, "train"))
    state, frozen = task.fitted_state.to_dict(), task.fitted_state.fingerprint
    report["stage"] = "training"
    repeats = report["repeats"] = []
    for i in range(2):
        repeats.append(
            train_repetition(
                task,
                None,
                selected,
                family,
                directory=directory / f"repeat-{i}",
                capacity=capacity,
                seed=seed,
                batch_size=batch_size,
                expected_rows=rows,
                save_model=True,
                features=features,
            )
        )
        write_report(directory, report)
    report["reproducible"] = compare_repetitions(*repeats)
    assert all(r["status"] == "passed" for r in repeats), (
        "Training acceptance failed; see repetition reports"
    )
    assert report["reproducible"], "Same-seed training did not replay exactly"
    model = build_model(
        task,
        selected,
        family,
        max_seq_len=sequence_capacity(binding, capacity),
        seed=seed,
    )
    model.load_state_dict(
        torch.load(
            directory / "repeat-0/model.pt", weights_only=True, map_location="cpu"
        )
    )
    assert input_digest(model.state_dict()) == repeats[0]["final_model_sha256"]
    report["evaluation_model_sha256"] = repeats[0]["final_model_sha256"]
    model.eval()
    report["stage"] = "evaluation"
    report["evaluation"] = {}
    with torch.no_grad():
        for split in ("valid", "test"):
            names = report["splits"][split]["eligible_targets"]
            sums, weights = dict.fromkeys(names, 0.0), dict.fromkeys(names, 0)
            expected, delivered = projected_rows(projection, split), 0
            for batch in task.batches(
                split=split,
                epoch=0,
                seed=seed,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                output_format="model",
            ):
                # Retain the source rows for the legacy scalar-binary loss oracle too.
                from itertools import islice

                reference = list(islice(expected, len(batch.metadata["sample_ids"])))
                if features is not None:
                    features(batch, phase="evaluation")
                eligible = verify_batch_for(binding, batch, iter(reference), state)
                labels = {
                    row["sample_id"]: row["label"]
                    for row in reference
                    if "label" in row
                }
                delivered += len(reference)
                result = measured_loss(model, batch, objective, labels, audit=True)
                for name in names:
                    sums[name] += (
                        result.loss_sums[name].item()
                        if isinstance(result.logits, dict)
                        else result.loss.item() * result.weight
                    )
                    weights[name] += eligible[name]
            assert (
                next(expected, None) is None
                and delivered == population["counts"][split]
            )
            assert weights == names
            report["evaluation"][split] = {
                name: {
                    "eligible": weights[name],
                    "mean_loss": sums[name] / weights[name],
                }
                for name in names
            }
    assert task.fitted_state.fingerprint == frozen


if __name__ == "__main__":
    raise SystemExit(
        "Use: python -m test.kumo_pytorch.nightly run --binding ID --mode training ..."
    )
