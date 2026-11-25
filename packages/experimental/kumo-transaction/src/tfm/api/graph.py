# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public graph composition factories; user factories use the same build context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from torch import nn

from tfm.api.builders import build_position, profile_event_spec_from_config
from tfm.attention.registry import build_attention
from tfm.compositions.gnn import GNNEncoder, GNNSequence
from tfm.compositions.profile_event_decoder import ProfileEventDecoder
from tfm.config import RecipeConfig
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.protocols import AttentionSpec
from tfm.contracts.registries import Registry
from tfm.data.graph_recipe import PreparedGraphData
from tfm.heads.binary import Binary
from tfm.layers.gnn import GNNEncoderModel, GNNSequenceModel
from tfm.pooling.registry import build_readout
from tfm.position.none import NoPE
from tfm.position.time_rope import TimeRoPE


@dataclass(frozen=True)
class GraphBuildContext:
    graph: GNNEncoder
    sequence: ProfileEventDecoder | None = None
    target_columns: tuple[str, ...] = ()
    encoder: nn.Module | None = None
    target_level: Literal["sequence", "entity"] = "sequence"


@dataclass(frozen=True)
class GraphArchitecture:
    """One model builder and the input contract its inference consumes."""

    build: Callable[[RecipeConfig, GraphBuildContext], nn.Module]
    layout: Literal["graph", "graph_history"]


_ARCHITECTURES: Registry[GraphArchitecture] = Registry("graph architecture")


class ContextHistoryFactory(Protocol):
    def __call__(
        self, spec: ProfileEventDecoder, attention: AttentionSpec, *, position: NoPE | TimeRoPE
    ) -> nn.Module: ...


_HISTORY: Registry[ContextHistoryFactory] = Registry("context history")
_HEADS: Registry[Callable[[int], nn.Module]] = Registry("task head")


def register_graph_architecture(name: str, architecture: GraphArchitecture) -> GraphArchitecture:
    """Make a graph model selectable through model.composition.granularity.type."""
    return _ARCHITECTURES.register(name, architecture)


def graph_architecture(name: str) -> GraphArchitecture:
    entry = _ARCHITECTURES.get(name)
    if not isinstance(entry, GraphArchitecture):
        raise UnsupportedCombinationError(f"{name!r} must name a GraphArchitecture")
    return entry


def register_context_history(name: str, factory: ContextHistoryFactory) -> None:
    """A history factory returns a module called with (history, context=...)."""
    _HISTORY.register(name, factory)


def register_task_head(name: str, factory: Callable[[int], nn.Module]) -> None:
    """A binary task head maps embeddings [B,D] to logits [B,1]."""
    _HEADS.register(name, factory)


def context_history(
    name: str | None,
    spec: ProfileEventDecoder,
    attention: AttentionSpec,
    *,
    position: NoPE | TimeRoPE,
) -> nn.Module | None:
    if name is None:
        return None
    factory = _HISTORY.get(name)
    if not callable(factory):
        raise UnsupportedCombinationError(f"Context history {name!r} must be a callable factory")
    model = factory(spec, attention, position=position)
    if not isinstance(model, nn.Module):
        raise UnsupportedCombinationError("Context history factory must return nn.Module")
    return model


def task_head(name: str | None, width: int) -> nn.Module | None:
    if name is None:
        return None
    factory = _HEADS.get(name)
    if not callable(factory):
        raise UnsupportedCombinationError(f"Task head {name!r} must be a callable factory")
    head = factory(width)
    if not isinstance(head, nn.Module):
        raise UnsupportedCombinationError("Task head factory must return nn.Module")
    return head


def build_graph_architecture(
    name: str, config: RecipeConfig, context: GraphBuildContext
) -> nn.Module:
    """Build a registered or importable graph architecture with public context."""
    model = graph_architecture(name).build(config, context)
    if not isinstance(model, nn.Module):
        raise UnsupportedCombinationError("Graph architecture must return nn.Module")
    return model


def encoder_from_config(config: RecipeConfig, context: GraphBuildContext) -> GNNEncoderModel:
    future = config.training.objective == "future_subgraph" and config.runtime.stage != "downstream"
    return GNNEncoderModel(
        context.graph,
        context.target_columns,
        objective=None if future else Binary(target_level=context.target_level),
        head=task_head(config.model.composition.granularity.task_head, context.graph.hidden_size)
        if not future
        else None,
    )


def hybrid_from_config(config: RecipeConfig, context: GraphBuildContext) -> GNNSequenceModel:
    if context.sequence is None:
        raise UnsupportedCombinationError("Graph history composition needs a sequence spec")
    if context.encoder is None:
        raise UnsupportedCombinationError("Graph history composition needs a supplied encoder")
    attention = build_attention(config.model.composition.attention)
    gran = config.model.composition.granularity
    position = build_position(config.model.composition.position)
    if not isinstance(position, (NoPE, TimeRoPE)):
        raise UnsupportedCombinationError("Graph history supports NoPE or TimeRoPE")
    return GNNSequenceModel(
        GNNSequence(context.graph, context.sequence),
        context.encoder,
        attention=attention,
        readout=build_readout(config.model.composition.readout),
        history_model=context_history(
            gran.history_model, context.sequence, attention, position=position
        ),
        position=position,
        head=task_head(gran.task_head, context.sequence.hidden_size),
        target_level=context.target_level,
    )


register_graph_architecture("gnn_encoder", GraphArchitecture(encoder_from_config, "graph"))
register_graph_architecture("gnn_sequence", GraphArchitecture(hybrid_from_config, "graph_history"))


def graph_specs(
    config: RecipeConfig, data: PreparedGraphData
) -> tuple[GNNEncoder, ProfileEventDecoder | None]:
    """Build graph and sequence specs from validated data and shared slot config."""
    g = config.model.composition.granularity
    graph = GNNEncoder(
        tuple((t.name, len(t.features)) for t in data.sampler.graph.tables),
        data.sampler.graph.directed_relations,
        "users",
        foreign_key_relations=tuple(
            (target, f"rev_{role}", source) for source, role, target in data.sampler.graph.relations
        ),
        hidden_size=g.hidden_size,
        depth=g.depth,
        trim=g.trim,
    )
    sequence = (
        None
        if data.tokenizer is None
        else profile_event_spec_from_config(config, vocab_size=data.tokenizer.vocab_size)
    )
    return graph, sequence
