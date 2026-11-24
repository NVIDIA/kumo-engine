# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from tfm.api.builders import model_from_config
from tfm.api.graph import GraphBuildContext, graph_specs, register_graph_architecture
from tfm.cli.recipe import run_command
from tfm.config import RecipeConfig
from tfm.data.adapters.registry import register_adapter
from tfm.data.graph_recipe import prepare_graph_data
from tfm.position.none import NoPE
from tfm.runtime.registry import register_recipe_runner

FIXTURE = "tests.fixtures.graph_components"


def payload():
    return yaml.safe_load(Path("recipes/gnn_sequence/config.smoke.yaml").read_text())


@pytest.mark.parametrize("mode", ["registered", "imported"])
def test_custom_runner_uses_the_shared_schema_and_reads_yaml_once(tmp_path, monkeypatch, mode):
    from tests.fixtures.graph_components import runner

    name = f"{FIXTURE}:runner"
    if mode == "registered":
        name = "test_user_recipe"
        register_recipe_runner(name, runner)
    raw = payload()
    raw["recipe_family"] = name
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    reads = []
    read = Path.open

    def counted(self, *args, **kwargs):
        if self == path:
            reads.append(self)
        return read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    out = run_command("train", config_path=path, run_dir=tmp_path / "out", device="cpu")
    assert (out / "selected.txt").read_text() == name
    assert reads == [path]


@pytest.mark.parametrize("mode", ["registered", "imported"])
def test_custom_adapter_feeds_graph_data_without_a_builtin_name(mode):
    from tests.fixtures.graph_components import adapter

    name = f"{FIXTURE}:ADAPTER"
    if mode == "registered":
        name = "test_user_graph_data"
        register_adapter(name)(adapter)
    raw = payload()
    raw["data"]["adapter"] = name
    data = prepare_graph_data(RecipeConfig.model_validate(raw))
    assert data.pretrain and data.probes["test"]
    assert data.split_manifest["dataset"] == name


@pytest.mark.parametrize("mode", ["registered", "imported"])
def test_custom_architecture_uses_the_same_public_builder_context(mode):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tests.fixtures.graph_components import ARCHITECTURE

    name = f"{FIXTURE}:ARCHITECTURE"
    if mode == "registered":
        name = "test_user_graph"
        register_graph_architecture(name, ARCHITECTURE)
    raw = payload()
    raw["model"]["composition"]["granularity"]["type"] = name
    config = RecipeConfig.model_validate(raw)
    data = prepare_graph_data(config)
    graph, sequence = graph_specs(config, data)
    model = model_from_config(
        config, vocab_size=sequence.vocab_size, graph_context=GraphBuildContext(graph, sequence)
    )
    assert isinstance(model.head, torch.nn.Sequential)
    output = model(data.pretrain[0])
    output.loss.backward()
    assert model.head[0].weight.grad is not None


def test_custom_sequence_readout_head_and_tokenizer_accept_a_deferred_graph_context():
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tests.fixtures.graph_components import (
        UserGraphWrapper,
        UserHistory,
        UserHistorySpec,
        UserPool,
        UserReadout,
    )

    from tfm.compositions.gnn import GNNSequence
    from tfm.layers.gnn import GNNEncoderModel, GNNSequenceModel

    raw = payload()
    raw["features"] = {"tokenizer": f"{FIXTURE}:UserTokenizer"}
    config = RecipeConfig.model_validate(raw)
    data = prepare_graph_data(config)
    graph, sequence = graph_specs(config, data)
    encoder = UserGraphWrapper(GNNEncoderModel(graph, ("example",)).encoder)
    custom = UserHistorySpec(sequence.hidden_size, sequence.vocab_size)
    model = GNNSequenceModel(
        GNNSequence(graph, custom),
        encoder,
        readout=UserReadout(),
        head=torch.nn.Linear(sequence.hidden_size, 1),
        position=NoPE(),
    )
    assert isinstance(model.history_model, UserHistory) and isinstance(model.pool, UserPool)
    order = []
    hooks = [
        model.history_model.event_encoder.register_forward_hook(
            lambda *args: order.append("events")
        ),
        encoder.register_forward_pre_hook(lambda *args: order.append("graph")),
    ]
    batch = data.hybrid_train[0]
    output = model(batch)
    output.loss.backward()
    for hook in hooks:
        hook.remove()
    assert order == ["events", "graph"]
    assert model.head.weight.grad is not None and model.bridge.weight.grad is not None
    assert all(
        not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters()
    )
    assert not encoder.training
    # Named arrays are inputs, copied to the execution device and mutation-guarded.
    view = batch.for_execution(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    assert view.history.events["custom_clock"].device == view.history.key_ids.device
    batch.history.events["custom_clock"].add_(0)
    with pytest.raises(ValueError, match="mutated"):
        view.assert_validated()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_future_objective_has_fixed_shape_masking_and_no_host_scalar_reads(device):
    from datetime import UTC, datetime

    from tfm.contracts.graph import FutureSubgraphTargets, GraphAnchor
    from tfm.objectives.future_subgraph import FutureSubgraph

    anchor = GraphAnchor("s", "u", datetime(2024, 1, 1, tzinfo=UTC))
    from tfm.contracts.graph import utc_us

    target = FutureSubgraphTargets(
        (anchor,),
        torch.tensor([[3.0, float("nan"), 2.0]]),
        torch.tensor([[True, False, True]]),
        ("a", "b", "c"),
        1,
        0,
        torch.tensor([utc_us(anchor.cutoff) + 1]),
        utc_us(anchor.cutoff) + 1,
    )
    for masked in [False, True]:
        current = replace(target, valid=torch.zeros_like(target.valid)) if masked else target
        current = current.for_execution(torch.device(device))
        prediction = torch.tensor([[1.0, float("nan"), 4.0]], device=device, requires_grad=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            loss = FutureSubgraph().loss(prediction, current)
            loss.backward()
        names = {event.key for event in profile.key_averages()}
        assert not {"aten::nonzero", "aten::_local_scalar_dense", "aten::is_nonzero"} & names
        assert loss.item() == (0.0 if masked else 4.0)
        torch.testing.assert_close(
            prediction.grad,
            torch.tensor([[0.0, 0.0, 0.0]], device=device)
            if masked
            else torch.tensor([[-2.0, 0.0, 2.0]], device=device),
        )
