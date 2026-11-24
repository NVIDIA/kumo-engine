# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe heterogeneous graph encoder and frozen graph/profile composition."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tfm.attention.sdpa import SdpaAttention
from tfm.compositions.gnn import ContextHistorySpec, GNNEncoder, GNNSequence
from tfm.compositions.profile_event_decoder import ProfileEventDecoder
from tfm.contracts.graph import (
    GraphBatch,
    GraphHistoryBatch,
    GraphSupervisedBatch,
    GraphTrainingBatch,
    SequenceSupervisedBatch,
)
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.outputs import ModelOutput
from tfm.contracts.protocols import AttentionSpec
from tfm.heads.binary import Binary
from tfm.layers.kvt_encoders import (
    CalendarEmbedding,
    KVTEventEncoder,
    KVTProfileEncoder,
    sinusoidal_position,
)
from tfm.layers.profile_event_decoder import CausalSummaryDecoder, decode_history, encode_events
from tfm.objectives.future_subgraph import FutureSubgraph
from tfm.pooling.last_token import LastTokenPool
from tfm.pooling.registry import Readout, readout_module
from tfm.pooling.usr_token import UsrTokenPool
from tfm.position.none import NoPE
from tfm.position.time_rope import TimeRoPE


@runtime_checkable
class RootGraphEncoder(Protocol):
    """A module's declared graph schema and ordered root embeddings."""

    @property
    def spec(self) -> GNNEncoder: ...

    def __call__(self, batch: GraphBatch) -> Tensor: ...


class GraphEncoder(nn.Module):
    """Feature projections and direct Kumo Engine HeteroGNN, with ordered roots.

    FK -> PK directions are mapped explicitly to non-rev Engine roles. The
    inverse directions use rev_ roles and must have at most one parent per FK
    receiver. This avoids accidentally selecting Engine's single-parent gather
    for a many-to-one aggregation. No local message-passing implementation exists.
    """

    def __init__(self, spec: GNNEncoder) -> None:
        super().__init__()
        from tfm.extras.kumo import hetero_gnn

        self.spec = spec
        self.types = tuple(k for k, _ in spec.node_widths)
        self.input = nn.ModuleList(
            nn.Linear(width, spec.hidden_size) for _, width in spec.node_widths
        )
        self.engine_roles: dict[tuple[str, str, str], tuple[str, str, str]] = {}
        for i, (source, role, target) in enumerate(spec.foreign_key_relations):
            reverse_role = role[4:] if role.startswith("rev_") else f"rev_{role}"
            self.engine_roles[source, role, target] = (source, f"fk_{i}", target)
            self.engine_roles[target, reverse_role, source] = (target, f"rev_fk_{i}", source)
        self.gather_roles = frozenset(
            role for role, alias in self.engine_roles.items() if alias[1].startswith("rev_")
        )
        self.gnn = hetero_gnn(
            channels=spec.hidden_size,
            num_mp_layers=spec.depth,
            metadata=(list(self.types), list(self.engine_roles.values())),
            aggr=["sum", "mean", "max"],
            norm="batch_norm",
            act="relu",
        )

    @property
    def hidden_size(self) -> int:
        return self.spec.hidden_size

    def forward(self, batch: GraphBatch) -> Tensor:
        batch.assert_validated()
        if (
            set(batch.node_features) != set(self.types)
            or set(batch.edge_index) != set(self.spec.relations)
            or batch.root_type != self.spec.root_type
        ):
            raise ValueError("Graph batch schema differs from GNN specification")
        if any(batch.node_features[k].shape[1] != width for k, width in self.spec.node_widths):
            raise ValueError("Graph input feature widths differ from GNN specification")
        if not self.gather_roles.issubset(batch.single_parent_relations):
            raise ValueError(
                "Engine PK -> FK gather requires at most one parent per receiver and FK role"
            )
        x = {
            name: layer(batch.node_features[name])
            for name, layer in zip(self.types, self.input, strict=True)
        }
        edges = {self.engine_roles[role]: batch.edge_index[role] for role in self.engine_roles}
        if self.spec.trim:
            if batch.num_sampled_nodes is None or batch.num_sampled_edges is None:
                raise ValueError("GNN trimming requires validated hop-ordered node and edge counts")
            if any(len(c) != self.spec.depth + 1 for c in batch.num_sampled_nodes.values()):
                raise ValueError("Batch hop depth differs from GNN depth")
            hidden = self.gnn(
                x,
                edges,
                num_sampled_nodes_dict={k: list(v) for k, v in batch.num_sampled_nodes.items()},
                num_sampled_edges_dict={
                    self.engine_roles[k]: list(v) for k, v in batch.num_sampled_edges.items()
                },
            )
        else:
            hidden = self.gnn(x, edges)
        return hidden[batch.root_type][batch.root_index]


class GNNEncoderModel(nn.Module):
    """Graph encoder trained by available binary labels or FutureSubgraph targets."""

    def __init__(
        self,
        spec: GNNEncoder,
        target_columns: tuple[str, ...] = (),
        objective: FutureSubgraph | Binary | None = None,
        *,
        head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.objective = FutureSubgraph() if objective is None else objective
        if not isinstance(self.objective, (FutureSubgraph, Binary)):
            raise ValueError("GNNEncoderModel requires Binary or FutureSubgraph")
        if isinstance(self.objective, FutureSubgraph) and (
            not target_columns or len(set(target_columns)) != len(target_columns)
        ):
            raise ValueError("GNN pretraining needs unique target columns")
        if isinstance(self.objective, Binary) and (
            target_columns or self.objective.target_level != "sequence"
        ):
            raise ValueError(
                "Binary GNN training uses sequence-level labels, not future target columns"
            )
        self.encoder = GraphEncoder(spec)
        self.target_columns = target_columns
        self.output_width = len(target_columns) if isinstance(self.objective, FutureSubgraph) else 1
        self.head = nn.Linear(spec.hidden_size, self.output_width) if head is None else head

    @property
    def hidden_size(self) -> int:
        return self.encoder.hidden_size

    def forward(
        self,
        batch: GraphTrainingBatch | GraphSupervisedBatch | GraphBatch,
        *,
        compute_loss: bool = True,
    ) -> ModelOutput:
        batch.assert_validated()
        graph = (
            batch.graph if isinstance(batch, (GraphTrainingBatch, GraphSupervisedBatch)) else batch
        )
        z = self.encoder(graph)
        prediction = self.head(z)
        if not isinstance(prediction, Tensor) or prediction.shape != (
            graph.batch_size,
            self.output_width,
        ):
            raise ValueError("Graph task head must return [batch, output_width] logits/targets")
        if isinstance(self.objective, Binary):
            logits = prediction.squeeze(-1)
            output = ModelOutput(
                hidden_states=z[:, None],
                embeddings=z,
                logits=logits,
                predictions={"binary": logits.sigmoid()},
            )
            if compute_loss:
                if not isinstance(batch, GraphSupervisedBatch):
                    raise ValueError(
                        "Supervised GNN training requires available binary task labels"
                    )
                output.loss = F.binary_cross_entropy_with_logits(logits, batch.labels.float())
                output.losses = {"binary": output.loss}
            return output
        out = ModelOutput(
            hidden_states=z[:, None], embeddings=z, predictions={self.objective.name: prediction}
        )
        if compute_loss:
            if (
                not isinstance(batch, GraphTrainingBatch)
                or batch.targets.columns != self.target_columns
            ):
                raise ValueError(
                    "FutureSubgraph training needs aligned targets with matching columns"
                )
            out.loss = self.objective.loss(prediction, batch.targets)
            out.losses = {self.objective.name: out.loss}
        return out


class _SequenceReadout(nn.Module):
    """Shared profile/event/history stack for supervised and graph controls."""

    def __init__(
        self,
        sequence: ProfileEventDecoder | ContextHistorySpec,
        attention: AttentionSpec | None = None,
        *,
        readout: Readout | None = None,
        head: nn.Module | None = None,
        history_model: nn.Module | None = None,
        position: NoPE | TimeRoPE | None = None,
    ) -> None:
        super().__init__()
        self.sequence_spec = sequence
        self.position_spec = TimeRoPE() if position is None else position
        attention = SdpaAttention() if attention is None else attention
        self.attention_spec = attention
        self.readout_spec = LastTokenPool() if readout is None else readout
        if isinstance(self.readout_spec, UsrTokenPool):
            raise ValueError(
                "Causal history readout must see the history, not the leading profile slot"
            )
        if not isinstance(sequence, ProfileEventDecoder) and history_model is None:
            history_model = sequence.build(attention, position=self.position_spec)
            if not isinstance(history_model, nn.Module):
                raise ValueError("Context history spec must build nn.Module")
        self.history_model = history_model
        if isinstance(sequence, ProfileEventDecoder) and history_model is None:
            self.embed = nn.Embedding(sequence.vocab_size, sequence.hidden_size, padding_idx=0)
            self.event_encoder = KVTEventEncoder(sequence, attention)
            self.profile_encoder = KVTProfileEncoder(sequence, attention)
            self.calendar = CalendarEmbedding(sequence.hidden_size, sequence.time_features)
            self.decoder = CausalSummaryDecoder(
                sequence, attention, time_rope=isinstance(self.position_spec, TimeRoPE)
            )
        self.pool = readout_module(self.readout_spec)
        self.head = nn.Linear(sequence.hidden_size, 1) if head is None else head

    @property
    def hidden_size(self) -> int:
        return self.sequence_spec.hidden_size

    def embed_tokens(self, keys: Tensor, values: Tensor, positions: Tensor) -> Tensor:
        if not isinstance(self.sequence_spec, ProfileEventDecoder):
            raise ValueError("A custom history stack owns its token embeddings")
        return (
            self.embed(keys)
            + self.embed(values)
            + sinusoidal_position(positions, self.hidden_size, self.sequence_spec.max_position)
        )

    def encode_history(
        self, history: KeyValueTimeBatch, context: Tensor | Callable[[], Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Sequence states and readout, with an optional graph field before profile encoding."""
        if self.history_model is not None:
            states, mask = self.history_model(history, context=context)
            return states, self.pool(states, mask)
        if history.elapsed_log is None:
            raise ValueError(
                f"{type(self).__name__} needs KeyValueTimeBatch.elapsed_log; "
                "collate with KeyValueTimeTokenizer.collate."
            )
        summaries, _ = encode_events(self, history)
        states, mask = decode_history(
            self, history, summaries, history.event_mask, history.elapsed_log, context=context
        )
        return states, self.pool(states, mask)

    def _binary_logits(self, embedding: Tensor) -> Tensor:
        prediction = self.head(embedding)
        if not isinstance(prediction, Tensor) or prediction.shape != (embedding.shape[0], 1):
            raise ValueError("Binary task head must return [batch, 1] logits")
        return prediction.squeeze(-1)


class GNNSequenceModel(_SequenceReadout):
    """Frozen selected GNN plus trainable graph field and sequence task model.

    The graph stage may be supervised or self-supervised. Freezing includes
    parameters, buffers and eval mode. Attention defaults to SDPA, with eager
    retained as a correctness control. This is cutoff-level binary supervision.
    """

    def __init__(
        self,
        spec: GNNSequence,
        encoder: nn.Module,
        *,
        objective: str = "binary",
        attention: AttentionSpec | None = None,
        readout: Readout | None = None,
        head: nn.Module | None = None,
        history_model: nn.Module | None = None,
        position: NoPE | TimeRoPE | None = None,
    ) -> None:
        if objective != "binary":
            raise ValueError(
                "GNNSequence supports cutoff-level binary supervision only; next-event objectives and surprise require per-prefix graphs"
            )
        if not isinstance(encoder, RootGraphEncoder) or encoder.spec != spec.graph:
            raise ValueError("Hybrid graph specification differs from supplied encoder")
        super().__init__(
            spec.sequence,
            attention,
            readout=readout,
            head=head,
            history_model=history_model,
            position=position,
        )
        self.spec = spec
        self.graph_encoder = encoder
        self.graph_encoder.requires_grad_(False)
        for parameter in self.graph_encoder.parameters():
            parameter.grad = None
        self.graph_encoder.eval()
        self.bridge = nn.Linear(spec.graph.hidden_size, spec.sequence.hidden_size)
        self.graph_field = nn.Parameter(torch.zeros(spec.sequence.hidden_size))

    def train(self, mode: bool = True) -> GNNSequenceModel:
        super().train(mode)
        self.graph_encoder.eval()
        return self

    def forward(self, batch: GraphHistoryBatch, *, compute_loss: bool = True) -> ModelOutput:
        batch.assert_validated()
        history = batch.history
        if history.elapsed_log is None:
            raise ValueError("GNNSequence needs past-only elapsed history times")

        def context() -> Tensor:
            with torch.no_grad():
                graph = self.graph_encoder(batch.graph)
            return self.bridge(graph) + self.graph_field

        hidden, embedding = self.encode_history(history, context)
        logits = self._binary_logits(embedding)
        out = ModelOutput(
            hidden_states=hidden,
            embeddings=embedding,
            logits=logits,
            predictions={"binary": logits.sigmoid()},
        )
        if compute_loss:
            if batch.labels is None:
                raise ValueError("GNNSequence training requires available sequence-level labels")
            out.loss = F.binary_cross_entropy_with_logits(logits, batch.labels.float())
            out.losses = {"binary": out.loss}
        return out


class SequenceBinaryModel(_SequenceReadout):
    """Supervised sequence-only control with the same labels and no GNN modules."""

    def forward(self, batch: SequenceSupervisedBatch, *, compute_loss: bool = True) -> ModelOutput:
        batch.assert_validated()
        hidden, embedding = self.encode_history(batch.history)
        logits = self._binary_logits(embedding)
        out = ModelOutput(
            hidden_states=hidden,
            embeddings=embedding,
            logits=logits,
            predictions={"binary": logits.sigmoid()},
        )
        if compute_loss:
            if batch.labels is None:
                raise ValueError("Supervised sequence training requires available task labels")
            out.loss = F.binary_cross_entropy_with_logits(logits, batch.labels.float())
            out.losses = {"binary": out.loss}
        return out
