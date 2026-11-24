# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental graph-only and graph-conditioned sequence composition specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from torch import nn

from tfm.compositions.profile_event_decoder import ProfileEventDecoder
from tfm.contracts.graph import EdgeType
from tfm.contracts.protocols import AttentionSpec
from tfm.position.none import NoPE
from tfm.position.time_rope import TimeRoPE


@dataclass(frozen=True)
class GNNEncoder:
    """Kumo Engine HeteroGNN with explicit FK -> PK direction declarations.

    Numerical features are encoded upstream. foreign_key_relations identifies
    the many-to-one direction of every physical relation; its opposite must be
    present too. The adapter maps roles to Engine's FK/rev_FK naming convention.
    Engine supplies sum/mean/max aggregation, BatchNorm, and ReLU directly.
    Feature projections have no extra activation. trim=True requires validated
    hop-prefix counts on GraphBatch; trim=False is an explicit execution control.
    """

    node_widths: tuple[tuple[str, int], ...]
    relations: tuple[EdgeType, ...]
    root_type: str
    foreign_key_relations: tuple[EdgeType, ...]
    hidden_size: int = 32
    depth: int = 2
    trim: bool = True
    name: str = field(default="gnn_encoder", init=False)
    support_tier: str = field(default="experimental", init=False)

    def __post_init__(self) -> None:
        widths = dict(self.node_widths)
        if (
            len(widths) != len(self.node_widths)
            or self.root_type not in widths
            or any(not k or v < 1 for k, v in widths.items())
            or self.hidden_size < 1
            or self.depth < 1
        ):
            raise ValueError("GNN needs unique node types, positive widths/depth, and a known root")
        if len(set(self.relations)) != len(self.relations) or any(
            s not in widths or t not in widths or not r for s, r, t in self.relations
        ):
            raise ValueError("GNN relation roles must be unique and reference known node types")

        if len(set(self.foreign_key_relations)) != len(self.foreign_key_relations):
            raise ValueError("Duplicate FK direction declarations")
        covered = []
        for source, role, target in self.foreign_key_relations:
            reverse_role = role[4:] if role.startswith("rev_") else f"rev_{role}"
            covered.extend(((source, role, target), (target, reverse_role, source)))
        if set(covered) != set(self.relations) or len(covered) != len(self.relations):
            raise ValueError("Declare exactly one FK -> PK direction per relation pair")


class ContextHistorySpec(Protocol):
    """A user's sequence stack that can consume a deferred graph field.

    build(attention, position=...) must honor or reject the selected position.
    It returns a module called as (history, context=...), returning history
    states [B,S,D] and their padding mask [B,S]. Public encode_events/decode_history
    helpers are optional reusable pieces for implementing that module.
    """

    @property
    def hidden_size(self) -> int: ...

    def build(self, attention: AttentionSpec, *, position: NoPE | TimeRoPE) -> nn.Module: ...


@dataclass(frozen=True)
class GNNSequence:
    """Frozen GNN conditions Blueprint v2 via one projected profile field.

    Only cutoff-level binary supervision is supported initially. Next-event
    pretraining would need per-prefix graph cutoffs, not a graph at history end.
    """

    graph: GNNEncoder
    sequence: ProfileEventDecoder | ContextHistorySpec = field(default_factory=ProfileEventDecoder)
    name: str = field(default="gnn_sequence", init=False)
    support_tier: str = field(default="experimental", init=False)
