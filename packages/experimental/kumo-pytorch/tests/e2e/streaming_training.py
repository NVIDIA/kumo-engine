# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU training for the weekly session-count task."""

from contextlib import ExitStack
from copy import deepcopy
from itertools import count
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from e2e.flows.common import BatchDigest
from kumo.pytorch import api
from tests.test_inference_lifecycle import assert_same


class CountSAGE(nn.Module):
    def __init__(self, task: api.Task):
        super().__init__()
        fields = task.output_schema["inputs"]["graph"]["sessions"]
        self.device_embedding = nn.Embedding(fields["device_type"]["vocabulary_size"], 4)
        # Viewers have no features. Aggregate messages from their historical sessions.
        self.conv = SAGEConv(6, 16, root_weight=False)
        self.count = nn.Linear(16, 1)

    def forward(self, batch: api.Batch) -> torch.Tensor:
        graph = batch.inputs["graph"]
        fields = graph["nodes"]["sessions"]["features"]
        numeric = torch.stack(
            [fields[key]["values"] for key in ("watch_seconds", "completion_frac")], 1
        )
        sessions = torch.cat((numeric, self.device_embedding(fields["device_type"]["values"])), 1)
        viewer, session = graph["edges"]["viewers_sessions"]["index"]
        owners = graph["nodes"]["viewers"]["batch"][viewer]
        neighbors = self.conv(
            (sessions, None), torch.stack((session, owners)), size=(len(sessions), batch.batch_size)
        )
        return self.count(neighbors.relu()).squeeze(-1)


def count_loss(model, batch):
    target = batch.targets["count"]
    predicted = model(batch)
    assert target["valid"].all() and not any(batch.metadata["is_padding"])
    assert predicted.shape == target["values"].shape == (batch.batch_size,)
    assert predicted.dtype == target["values"].dtype == torch.float32
    loss = nn.functional.smooth_l1_loss(predicted.double(), target["values"].double())
    assert torch.isfinite(loss)
    return loss


def _train(task, probe, evidence=None):
    torch.random.default_generator.manual_seed(17)
    model = CountSAGE(task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    initial = deepcopy(model.state_dict())
    before = count_loss(model, probe).item()
    digest, steps, active = BatchDigest(), [], set()
    for epoch in count():
        start = len(steps)
        for batch in task.batches(
            split="train", seed=17, epoch=epoch, shuffle=True, batch_size=4, output_format="model"
        ):
            digest.update(batch)
            optimizer.zero_grad()
            loss = count_loss(model, batch)
            if epoch == 0 and evidence is not None:
                evidence.observe(batch)
            loss.backward()
            for name, weight in model.named_parameters():
                assert weight.grad is not None and weight.grad.isfinite().all(), name
                if weight.grad.any():
                    active.add(name)
            optimizer.step()
            assert all(p.isfinite().all() for p in model.parameters())
            weights = nn.utils.parameters_to_vector(model.parameters()).detach()
            steps.append((batch.targets["count"]["values"].tolist(), loss.item(), weights))
            if len(steps) == 100:
                assert active == set(initial)
                assert all(not torch.equal(initial[n], p) for n, p in model.named_parameters())
                assert count_loss(model, probe).item() < before
                return model, (
                    digest.hexdigest(),
                    initial,
                    steps,
                    optimizer.state_dict(),
                    torch.get_rng_state(),
                )
        assert len(steps) > start, "empty training split"


def run_training(task: api.Task, work: Path, evidence):
    frozen = task.fitted_state.fingerprint
    probe = next(task.batches(split="train", batch_size=4, output_format="model"))
    with ExitStack() as scope:
        scope.enter_context(torch.random.fork_rng(devices=[]))
        scope.callback(torch.set_num_threads, torch.get_num_threads())
        scope.callback(torch.set_deterministic_debug_mode, torch.get_deterministic_debug_mode())
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model, reference = _train(task, probe, evidence)
        _, repeat = _train(task, probe)
        assert_same(repeat, reference)
        api.save_weights(work / "weights.pt", model.state_dict(), task.fitted_state)
        loaded = CountSAGE(task).eval()
        loaded.load_state_dict(api.load_weights(work / "weights.pt", task.fitted_state))
        assert_same(loaded.state_dict(), model.state_dict())
        with torch.no_grad():
            for split in ("valid", "test"):
                delivered = 0
                for batch in task.batches(split=split, batch_size=4, output_format="model"):
                    count_loss(loaded, batch)
                    evidence.observe(batch)
                    delivered += batch.batch_size
                assert delivered == task.count(split) > 0, split
    assert task.fitted_state.fingerprint == frozen
    return {"updates": [len(reference[2]), len(repeat[2])], "reproducible": True}
