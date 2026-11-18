# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Kumo-emitted tiny batches; no AWS, Arena imports or full AML run."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from test.kumo_pytorch.bindings import selections as catalog_selections
from test.kumo_pytorch.contracts import build_contract
from test.kumo_pytorch.models import build_model

import pytest
import torch

import kumo.pytorch.api as r

from .fixtures import build_fixture, probe_anchor_table

SELECTIONS = catalog_selections()
CASES = [
    (selection, family) for selection in SELECTIONS for family in selection["models"]
]


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(
    scope="module",
    params=CASES,
    ids=[f"{s['selector']['workload_id']}-{f}" for s, f in CASES],
)
def fitted(request, backend, tmp_path_factory):
    yield from fit_case(*request.param, backend, tmp_path_factory)


def fit_case(selection, family, backend, tmp_path_factory):
    fixture = build_fixture()
    graph = family in {"graphsage", "combined"}
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    anchors = fixture.training
    if native:
        anchors = tuple(replace(a, targets=()) for a in anchors)
    if graph:
        anchors = tuple(
            replace(
                a,
                as_of_time=a.as_of_time - timedelta(seconds=1),
                metadata={"prediction_time_ns": int(a.as_of_time.timestamp()) * 10**9},
            )
            for a in anchors
        )
    contract = build_contract(
        fixture.schema,
        family,
        native=native,
        max_events=len(fixture.tables["transactions"]),
    )
    assert backend == "diskgraph"
    directory = tmp_path_factory.mktemp("aml-model-inputs")
    dataset = fixture.parquet_dataset(directory / "source")
    engine = r.DiskGraphEngine(index=directory / "index", num_threads=1)
    with r.prepare(contract, dataset, engine) as plan:
        task = plan.materialize(
            probe_anchor_table(plan, anchors),
            fit_split="train",
            encoder_cutoff=int(
                max(row.as_of_time for row in fixture.training).timestamp()
            )
            - 1,
        )
        batch = next(task.batches(batch_size=2, seed=7, epoch=0, output_format="model"))
        yield task, batch, selection, family


@pytest.fixture(scope="module", params=("diskgraph",))
def backend(request):
    return request.param


def assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_equal(left, right)
    else:
        assert actual == expected


def test_forward_backward_update_and_input_preservation(fitted):
    task, batch, selection, family = fitted
    before = deepcopy(batch)
    model = build_model(task, selection, family, dim=8, seed=17)
    result = model.training_step(batch)
    assert result.weight > 0
    assert torch.isfinite(result.logits).all() and torch.isfinite(result.loss)
    original = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    result.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0).step()
    assert any(not torch.equal(p, original[n]) for n, p in model.named_parameters())
    for section in ("inputs", "targets", "metadata"):
        assert_equal(getattr(batch, section), getattr(before, section))
    if family in {"graphsage", "combined"}:
        check_graph_v2_clock_and_tensor_adapter_parity(fitted)


def test_model_construction_is_deterministic_and_does_not_reset_caller_rng(fitted):
    task, batch, selection, family = fitted
    state = torch.random.get_rng_state().clone()
    left = build_model(task, selection, family, dim=8, seed=17)
    right = build_model(task, selection, family, dim=8, seed=17)
    assert torch.equal(state, torch.random.get_rng_state())
    assert_equal(left.state_dict(), right.state_dict())


def check_graph_v2_clock_and_tensor_adapter_parity(fitted):
    task, batch, selection, family = fitted
    model = build_model(task, selection, family, dim=8).eval()
    from test.kumo_pytorch.models.graph_sequence_aggregates import (
        KumoFusionAdapter,
    )
    from test.kumo_pytorch.models.graphsage import KumoGraphAdapter

    adapter = (
        KumoGraphAdapter(model.backbone.config, input_name="graph")
        if family == "graphsage"
        else KumoFusionAdapter(model.backbone.config, graph_input="graph")
    )
    adapted = adapter(batch)
    with torch.no_grad():
        torch.testing.assert_close(
            model(batch), model.backbone(adapted).logits, rtol=0, atol=0
        )
    clock = adapted.seed_time if family == "graphsage" else adapted.graph.seed_time
    assert torch.equal(clock, batch.metadata["as_of_time"] + 1)
    bad = deepcopy(batch)
    bad.metadata["request_metadata"][0]["prediction_time_ns"] += 10**9
    with pytest.raises(ValueError):
        model.training_step(bad)
    bad.metadata.pop("request_metadata")
    with pytest.raises(ValueError, match="prediction_time_ns"):
        model(bad)


def test_wrong_anchor_version_is_rejected(fitted):
    task, _, selection, family = fitted
    changed = deepcopy(selection)
    changed["selector"]["anchor_version"] = (
        "v1" if family in {"graphsage", "combined"} else "v2"
    )
    with pytest.raises(ValueError, match="anchors"):
        build_model(task, changed, family)
