# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
import torch
from pydantic import ValidationError

from tfm.config import RecipeConfig
from tfm.contracts.graph import FutureSubgraphTargets, GraphAnchor, utc_us
from tfm.data.graph import FutureTargetTransform
from tfm.objectives.future_subgraph import FutureSubgraph


def target() -> FutureSubgraphTargets:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    return FutureSubgraphTargets(
        tuple(GraphAnchor(str(i), str(i), now) for i in range(3)),
        torch.tensor([[0.0, 7.0, 9.0], [3.0, 7.0, 9.0], [8.0, float("nan"), 9.0]]),
        torch.tensor([[True, True, False], [True, True, False], [True, False, True]]),
        ("amount", "constant", "missing"),
        1,
        0,
        torch.tensor([utc_us(now) + 1] * 3),
        utc_us(now + timedelta(days=1)),
    )


def test_transform_train_statistics_schema_order_and_no_validation_refit() -> None:
    raw = target()
    fitted = FutureTargetTransform.fit(raw)
    assert fitted.selected == (0,)
    expected = torch.tensor([0.0, 3.0, 8.0]).log1p()
    torch.testing.assert_close(fitted.mean, expected.mean()[None])
    torch.testing.assert_close(fitted.scale, expected.std(correction=0)[None])
    normalized = fitted.apply(raw)
    torch.testing.assert_close(
        normalized.values[:, 0], (expected - expected.mean()) / expected.std(correction=0)
    )
    changed = replace(raw, values=torch.full_like(raw.values, 1000.0))
    fitted.apply(changed)
    torch.testing.assert_close(fitted.mean, expected.mean()[None])
    with pytest.raises(ValueError, match="schema/order"):
        fitted.apply(replace(raw, columns=tuple(reversed(raw.columns))))
    with pytest.raises(ValueError, match="nonconstant"):
        FutureTargetTransform.fit(replace(raw, values=torch.ones_like(raw.values)))


def test_masked_loss_uses_valid_entries_and_detached_targets() -> None:
    raw = target()
    prediction = torch.ones_like(raw.values, requires_grad=True)
    loss = FutureSubgraph().loss(prediction, raw)
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[~raw.valid].count_nonzero() == 0
    assert float(loss.detach()) == pytest.approx(
        float((prediction.detach()[raw.valid] - raw.values[raw.valid]).square().mean())
    )
    with pytest.raises(ValueError, match="gradients"):
        replace(raw, values=raw.values.requires_grad_())


def graph_config(payload=None):
    return RecipeConfig.model_validate({"recipe_family": "graph", **(payload or {})})


def test_graph_configuration_refuses_unsupported_semantics() -> None:
    with pytest.raises(ValidationError, match="requires binary"):
        graph_config(
            {
                "model": {
                    "composition": {
                        "granularity": {"type": "gnn_sequence"},
                        "objective": "future_subgraph",
                    }
                }
            }
        )
    with pytest.raises(ValidationError):
        graph_config({"model": {"composition": {"objective": "next_event_fields"}}})
    with pytest.raises(ValidationError, match="eager"):
        graph_config({"training": {"backend": "ddp"}})
    with pytest.raises(ValidationError):
        graph_config({"evaluation": {"surprise": True}})


def test_shared_sampling_clock_and_repeated_root_contract_without_engine() -> None:
    from tfm.data.graph import GraphLink, GraphRow, GraphSampler, GraphTable, RelationalGraph

    now = datetime(2024, 1, 2, tzinfo=UTC)
    role = ("users", "events", "events")
    graph = RelationalGraph(
        (GraphTable("users", ("x",), immutable=True), GraphTable("events", ("x",))),
        (role,),
        (GraphRow("users", "u", {"x": 1.0}), GraphRow("events", "e", {"x": 2.0}, now, now)),
        (GraphLink(role, "u", "e", now),),
        frozenset({"fraud"}),
        now,
    )
    sampler = GraphSampler(graph, depth=1, fanout=2)
    anchors = (GraphAnchor("a", "u", now), GraphAnchor("b", "u", now))
    shared, disjoint = (
        sampler.sample(anchors, "users", sharing="shared"),
        sampler.sample(anchors, "users"),
    )
    assert len(shared.node_features["events"]) == 1
    assert len(disjoint.node_features["events"]) == 2
    assert shared.root_index.tolist() == [0, 0]
    assert shared.anchors == disjoint.anchors == anchors
    with pytest.raises(ValueError, match="identical cutoffs"):
        replace(shared, anchors=(anchors[0], replace(anchors[1], cutoff=now + timedelta(days=1))))
    with pytest.raises(ValueError, match="observation windows"):
        replace(shared, anchors=(anchors[0], replace(anchors[1], observation_start=now)))


def test_graph_mode_configuration_is_explicit() -> None:
    default = graph_config()
    assert default.training.objective == default.model.composition.objective == "binary"
    assert default.model.composition.attention == "sdpa"
    assert default.model.composition.granularity.trim
    with pytest.raises(ValidationError, match="must equal"):
        graph_config({"training": {"objective": "future_subgraph"}})
    hybrid = graph_config(
        {
            "model": {"composition": {"granularity": {"type": "gnn_sequence"}}},
            "training": {"objective": "future_subgraph"},
        }
    )
    assert hybrid.model.composition.objective == "binary"
    with pytest.raises(ValidationError, match="requires binary"):
        graph_config(
            {
                "model": {
                    "composition": {
                        "granularity": {"type": "gnn_sequence"},
                        "objective": "future_subgraph",
                    }
                }
            }
        )


@pytest.mark.parametrize("helper", ["fit", "apply", "loss"])
@pytest.mark.parametrize("mutation", ["values", "valid", "available_us", "requires_grad"])
def test_public_future_helpers_reject_mutated_targets(helper, mutation):
    raw = target()
    fitted = FutureTargetTransform.fit(raw)
    if mutation == "requires_grad":
        raw.values.requires_grad_()
    elif mutation == "valid":
        raw.valid.logical_not_()
    else:
        getattr(raw, mutation).add_(1)
    prediction = torch.zeros_like(raw.values, requires_grad=True)
    with pytest.raises(ValueError, match="mutated"):
        if helper == "fit":
            FutureTargetTransform.fit(raw)
        elif helper == "apply":
            fitted.apply(raw)
        else:
            FutureSubgraph().loss(prediction, raw)
    assert raw.values.grad is None and prediction.grad is None
