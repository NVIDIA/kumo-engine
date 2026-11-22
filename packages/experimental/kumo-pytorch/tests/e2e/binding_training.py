# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded reproducibility checks and full-split training on a registered package task."""

from collections import Counter
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from itertools import count
from pathlib import Path
from time import perf_counter_ns

import torch
from torch import nn

from e2e.flows.common import BatchDigest
from kumo.pytorch import api
from tests.e2e.binding_model import make_model
from tests.e2e.loader_timing import _rates
from tests.test_inference_lifecycle import assert_same


def train_full_binding(
    task, binding, *, batch_size, split_batches, device, train_window, on_split=None
):
    """Train twice on verified source batches, validate after each epoch and test once."""
    device = torch.device(device)
    torch.manual_seed(17)
    model = make_model(task, binding).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)

    def run_split(split, epoch):
        training = split == "train"
        model.train(training)
        row = dict(epoch=epoch, split=split, samples=0, tokens=0, optimizer_updates=0)
        totals = {name: torch.zeros((), dtype=torch.float64, device=device) for name in model.heads}
        eligible = Counter()
        with train_window(row) if training else nullcontext(), torch.set_grad_enabled(training):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter_ns()
            for batch in split_batches(split, epoch):
                if not 0 < batch.batch_size <= batch_size or batch.metadata["is_padding"].any():
                    raise ValueError(
                        f"{binding} {split} delivered unexpected padding or batch size"
                    )
                row["samples"] += batch.batch_size
                if "history" in batch.inputs:
                    row["tokens"] += int(batch.inputs["history"]["valid"].sum())
                sums, counts = model.loss_terms(batch, device)
                eligible.update(counts)
                for name, total in sums.items():
                    totals[name].add_(total.detach())
                if training and any(counts.values()):
                    loss = sum(sums[h] / max(n, 1) for h, n in counts.items())
                    torch._assert_async(loss.isfinite(), "Nonfinite training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    row["optimizer_updates"] += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row["elapsed_ns"] = perf_counter_ns() - start
        if row["samples"] != task.count(split) or not row["samples"]:
            raise ValueError(f"{binding} {split} delivered {row['samples']} of {task.count(split)}")
        if training and not row["optimizer_updates"]:
            raise ValueError(f"{binding} epoch {epoch} had no eligible optimizer updates")
        row.update(_rates({k: row[k] for k in ("samples", "tokens")}, row["elapsed_ns"]))
        row["heads"] = {}
        for name, total in totals.items():
            if not total.isfinite():
                raise ValueError(f"{binding} {split} {name} loss is nonfinite")
            loss_sum, n = total.item(), eligible[name]
            row["heads"][name] = dict(
                loss_sum=loss_sum, eligible=n, mean_loss=loss_sum / n if n else None
            )
        if on_split is not None:
            on_split(row)
        return row

    epochs, validation = [], []
    for epoch in (1, 2):
        epochs.append(run_split("train", epoch))
        validation.append(run_split("valid", epoch))
    return dict(epochs=epochs, validation=validation, test=run_split("test", 2))


def _train(task, probe, evidence=None, binding=None):
    torch.random.default_generator.manual_seed(17)
    model = make_model(task, binding)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    initial = deepcopy(model.state_dict())
    before = model.loss(probe).item()
    digest, steps, active = BatchDigest(), [], set()
    for epoch in count():
        start = len(steps)
        for batch in task.batches(
            split="train", seed=17, epoch=epoch, shuffle=True, batch_size=4, output_format="model"
        ):
            digest.update(batch)
            optimizer.zero_grad()
            loss = model.loss(batch)
            assert loss.isfinite()
            if epoch == 0 and evidence is not None:
                evidence.observe(batch)
            loss.backward()
            for name, weight in model.named_parameters():
                if binding is not None and weight.grad is None:
                    continue
                assert weight.grad is not None and weight.grad.isfinite().all(), name
                if weight.grad.any():
                    active.add(name)
            optimizer.step()
            assert all(p.isfinite().all() for p in model.parameters())
            weights = nn.utils.parameters_to_vector(model.parameters()).detach()
            steps.append(
                (
                    batch.metadata["sample_ids"]
                    if binding
                    else batch.targets["count"]["values"].tolist(),
                    loss.item(),
                    weights,
                )
            )
            if len(steps) == 100:
                assert active and (binding or active == set(initial))
                assert all(
                    not torch.equal(initial[n], p)
                    for n, p in model.named_parameters()
                    if n in active
                )
                assert model.loss(probe).item() < before
                return model, (
                    digest.hexdigest(),
                    initial,
                    steps,
                    optimizer.state_dict(),
                    torch.get_rng_state(),
                )
        assert len(steps) > start, "empty training split"


def run_training(task: api.Task, work: Path, evidence, binding=None):
    frozen = task.fitted_state.fingerprint
    probe = next(task.batches(split="train", batch_size=4, output_format="model"))
    with ExitStack() as scope:
        scope.enter_context(torch.random.fork_rng(devices=[]))
        scope.callback(torch.set_num_threads, torch.get_num_threads())
        scope.callback(torch.set_deterministic_debug_mode, torch.get_deterministic_debug_mode())
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model, reference = _train(task, probe, evidence, binding)
        _, repeat = _train(task, probe, binding=binding)
        assert_same(repeat, reference)
        api.save_weights(work / "weights.pt", model.state_dict(), task.fitted_state)
        loaded = make_model(task, binding).eval()
        loaded.load_state_dict(api.load_weights(work / "weights.pt", task.fitted_state))
        assert_same(loaded.state_dict(), model.state_dict())
        with torch.no_grad():
            for split in ("valid", "test"):
                delivered = 0
                for batch in task.batches(split=split, batch_size=4, output_format="model"):
                    assert loaded.loss(batch).isfinite()
                    evidence.observe(batch)
                    delivered += batch.batch_size
                assert delivered == task.count(split) > 0, split
    assert task.fitted_state.fingerprint == frozen
    return {"updates": [len(reference[2]), len(repeat[2])], "reproducible": True}
