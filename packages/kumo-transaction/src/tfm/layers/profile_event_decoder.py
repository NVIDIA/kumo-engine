# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blueprint v2: profile and event encoders feeding a causal history decoder (ADR 0012)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tfm.attention.eager import EagerAttention
from tfm.compositions.profile_event_decoder import ProfileEventDecoder
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.outputs import AlignmentLosses, ModelOutput
from tfm.contracts.protocols import AttentionSpec
from tfm.kernels import linear_cross_entropy
from tfm.layers.kvt_encoders import (
    CalendarEmbedding,
    KVTEventEncoder,
    KVTProfileEncoder,
    encoder_blocks,
    sinusoidal_position,
)
from tfm.objectives.contrastive import SubsequenceContrastive, info_nce, random_crop
from tfm.objectives.mlm import TokenEventKeyMLM, apply_mlm_mask
from tfm.objectives.next_event_fields import NextEventFields, next_event_targets
from tfm.objectives.past_reconstruction import PastReconstruction, past_targets
from tfm.pooling.last_token import LastTokenPool
from tfm.pooling.registry import Readout, readout_module
from tfm.position.none import NoPE
from tfm.position.time_rope import TimeRoPE


class CausalSummaryDecoder(nn.Module):
    """`SummaryHistoryModel`: causal self-attention over summary slots.

    With `time_rope`, attention is rotated by elapsed log-time; without it, the
    decoder has only causal order and each slot's calendar embedding.
    """

    def __init__(
        self, spec: ProfileEventDecoder, attention: AttentionSpec, *, time_rope: bool = True
    ) -> None:
        super().__init__()
        self.blocks = encoder_blocks(
            spec, attention, spec.decoder_depth, use_rope=time_rope, causal=True
        )
        self.norm = nn.LayerNorm(spec.hidden_size)

    def forward(self, summaries: Tensor, attention_mask: Tensor, time_positions: Tensor) -> Tensor:
        hidden = summaries
        for block in self.blocks:
            hidden = block(hidden, attention_mask, rope_positions=time_positions)
        return self.norm(hidden)


class ProfileFields(Protocol):
    """The profile tensors a batch carries, each `[B, P]`."""

    prof_key_ids: Tensor
    prof_value_ids: Tensor
    prof_positions: Tensor
    prof_mask: Tensor
    prof_time_log: Tensor


class ProfileEventStack(Protocol):
    """The modules `encode_events` and `decode_history` run, each typed by its call:
    `KVTEventEncoder`, `KVTProfileEncoder`, `CalendarEmbedding` and `CausalSummaryDecoder`
    are the built-ins."""

    @property
    def event_encoder(self) -> Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]:
        """`(tokens [B, N, T, D], token_mask, event_mask) -> (summaries [B, N, D], token
        states [B, N, T, D])`."""
        ...

    @property
    def profile_encoder(self) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
        """`(tokens [B, P, D], mask [B, P], time_log [B, P]) -> [B, D]`."""
        ...

    @property
    def calendar(self) -> Callable[[KeyValueTimeBatch], Tensor]:
        """Named per-event arrays -> [B, N, D]."""
        ...

    @property
    def decoder(self) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
        """`(slots [B, N+1, D], slot mask [B, N+1], time positions [B, N+1]) -> [B, N+1, D]`."""
        ...

    def embed_tokens(self, key_ids: Tensor, value_ids: Tensor, positions: Tensor, /) -> Tensor: ...


def encode_events(model: ProfileEventStack, batch: KeyValueTimeBatch) -> tuple[Tensor, Tensor]:
    """Return `(event summaries [B, N, D] with the calendar added, token states)`."""
    tokens = model.embed_tokens(batch.key_ids, batch.value_ids, batch.positions)
    summaries, token_states = model.event_encoder(tokens, batch.token_mask, batch.event_mask)
    calendar = model.calendar(batch)
    summaries = summaries + calendar * batch.event_mask.unsqueeze(-1).to(calendar.dtype)
    return summaries, token_states


def decode_history(
    model: ProfileEventStack,
    profile: ProfileFields,
    summaries: Tensor,
    event_mask: Tensor,
    elapsed_log: Tensor,
    *,
    context: Tensor | Callable[[], Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Return `(decoder states [B, N+1, D], slot mask [B, N+1])` from event summaries
    `[B, N, D]`, their mask and `elapsed_log` `[B, N]`. A `context` `[B, D]` is one more
    profile token, mask 1 and time 0; a callable is called after the events are encoded."""
    tokens = model.embed_tokens(
        profile.prof_key_ids, profile.prof_value_ids, profile.prof_positions
    )
    mask, times = profile.prof_mask, profile.prof_time_log
    if callable(context):
        context = context()
    if context is not None:
        tokens = torch.cat([tokens, context[:, None]], dim=1)
        mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
        times = torch.cat([times, times.new_zeros(times.shape[0], 1)], dim=1)
    z_a = model.profile_encoder(tokens, mask, times)
    slots = torch.cat([z_a.unsqueeze(1), summaries], dim=1)
    slot_mask = torch.cat([torch.ones_like(event_mask[:, :1]), event_mask], dim=1)
    time_positions = torch.cat([torch.zeros_like(elapsed_log[:, :1]), elapsed_log], dim=1)
    return model.decoder(slots, slot_mask, time_positions), slot_mask


class ProfileEventDecoderModel(nn.Module):
    """Experimental Blueprint v2 model. Not part of the core TrxFM export.

    Reads the per-event arrays `event_arrays` (`EventArrayReader`), the calendar's.

    Slots are `[profile, event_0, ..., event_{N-1}]`. The embedding is the
    readout over decoder states, by default the state after the last valid
    event. Auxiliary objectives add losses:

    - `TokenEventKeyMLM` applies PRAGMA's within-event token / event / key
      masking to the inputs; next-event targets always come from the unmasked
      batch.
    - `PastReconstruction` rebuilds earlier events from later slots.
    - `SubsequenceContrastive` pulls two crops of each history together.
    """

    def __init__(
        self,
        spec: ProfileEventDecoder,
        objective: NextEventFields | None = None,
        mlm: TokenEventKeyMLM | None = None,
        attention: AttentionSpec | None = None,
        readout: Readout | None = None,
        auxiliary: tuple[PastReconstruction | SubsequenceContrastive, ...] = (),
        position: TimeRoPE | NoPE | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.position_spec: TimeRoPE | NoPE = position if position is not None else TimeRoPE()
        self.objective_spec = objective or NextEventFields()
        self.mlm_spec = mlm
        self.past_spec = next((a for a in auxiliary if isinstance(a, PastReconstruction)), None)
        self.contrastive_spec = next(
            (a for a in auxiliary if isinstance(a, SubsequenceContrastive)), None
        )
        attn: AttentionSpec = attention if attention is not None else EagerAttention()
        self.attention_spec = attn
        self.readout_spec: Readout = readout if readout is not None else LastTokenPool()
        self.embed = nn.Embedding(spec.vocab_size, spec.hidden_size, padding_idx=0)
        self.event_encoder = KVTEventEncoder(spec, attn)
        self.profile_encoder = KVTProfileEncoder(spec, attn)
        self.decoder = CausalSummaryDecoder(
            spec, attn, time_rope=isinstance(self.position_spec, TimeRoPE)
        )
        self.calendar = CalendarEmbedding(spec.hidden_size, spec.time_features)
        self.event_arrays = self.calendar.arrays
        self.pool = readout_module(self.readout_spec)
        self.next_proj = nn.Linear(2 * spec.hidden_size, spec.hidden_size, bias=False)
        # Auxiliary parameters exist only when enabled: a parameter that never
        # receives gradient stalls DDP.
        self.mlm_proj: nn.Linear | None = None
        if mlm is not None:
            self.mlm_proj = nn.Linear(2 * spec.hidden_size, spec.hidden_size, bias=False)
        self.past_proj: nn.Linear | None = None
        self.past_gap: nn.Linear | None = None
        if self.past_spec is not None:
            self.past_proj = nn.Linear(3 * spec.hidden_size, spec.hidden_size, bias=False)
            self.past_gap = nn.Linear(2, spec.hidden_size)
        self.contrastive_proj: nn.Linear | None = None
        if self.contrastive_spec is not None:
            self.contrastive_proj = nn.Linear(spec.hidden_size, spec.hidden_size)

    @property
    def auxiliary_names(self) -> tuple[str, ...]:
        specs = (self.mlm_spec, self.past_spec, self.contrastive_spec)
        return tuple(s.name for s in specs if s is not None)

    @property
    def hidden_size(self) -> int:
        return self.spec.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.spec.vocab_size

    def embed_tokens(self, key_ids: Tensor, value_ids: Tensor, positions: Tensor) -> Tensor:
        token = self.embed(key_ids) + self.embed(value_ids)
        return token + sinusoidal_position(positions, self.spec.hidden_size, self.spec.max_position)

    def encode(self, batch: KeyValueTimeBatch) -> tuple[Tensor, Tensor, Tensor]:
        """Return `(decoder states [B, N+1, D], slot mask [B, N+1], token states)`."""
        if batch.elapsed_log is None:
            raise ValueError(
                "ProfileEventDecoderModel needs KeyValueTimeBatch.elapsed_log; "
                "collate with KeyValueTimeTokenizer.collate."
            )
        summaries, token_states = encode_events(self, batch)
        states, slot_mask = decode_history(
            self, batch, summaries, batch.event_mask, batch.elapsed_log
        )
        return states, slot_mask, token_states

    def _logits(self, proj: nn.Linear, query: Tensor, context: Tensor) -> Tensor:
        return proj(torch.cat([query, context], dim=-1)) @ self.embed.weight.t()

    def _vocab_loss(
        self, projected: Tensor, values: Tensor, *, label_smoothing: float = 0.0
    ) -> Tensor:
        """Cross-entropy of the vocabulary logits `projected [T, D] @ embed.weight.T`
        against `values [T]`, without building the logits."""
        return linear_cross_entropy(
            projected, self.embed.weight, values, label_smoothing=label_smoothing
        )

    def next_event_logits(self, states: Tensor, key_ids: Tensor) -> Tensor:
        """Vocabulary logits of one target token per row, from the slot before its
        event (`states [T, D]`) and the target's key (`key_ids [T]`)."""
        return self._logits(self.next_proj, states, self.embed(key_ids))

    def alignment_losses(self, batch: KeyValueTimeBatch) -> AlignmentLosses:
        """Score event `j`'s values from slot `j` (before it) and slot `j+1` (after it).

        The next-event head is trained on slot `j`. A model that can see its
        target would do better from slot `j+1`, which holds event `j` itself;
        `learned_to_copy` flags that (ADR 0010, 0011).
        """
        with torch.no_grad():
            states, _mask, _tokens = self.encode(batch)
            target = next_event_targets(batch)
            if target.values.numel() == 0:
                raise ValueError("No valid event token to score.")
            keys = self.embed(target.key_ids)
            before = self._logits(self.next_proj, states[target.batch, target.event], keys)
            after = self._logits(self.next_proj, states[target.batch, target.event + 1], keys)
            return AlignmentLosses(
                next_token=float(F.cross_entropy(before.float(), target.values)),
                same_token=float(F.cross_entropy(after.float(), target.values)),
                next_token_accuracy=float((before.argmax(-1) == target.values).float().mean()),
                same_token_accuracy=float((after.argmax(-1) == target.values).float().mean()),
                positions=int(target.values.numel()),
            )

    def forward(self, batch: KeyValueTimeBatch, *, compute_loss: bool = True) -> ModelOutput:
        work = batch
        if compute_loss and self.mlm_spec is not None:
            work = apply_mlm_mask(batch, self.mlm_spec)
        states, slot_mask, token_states = self.encode(work)
        output = ModelOutput(hidden_states=states, embeddings=self.pool(states, slot_mask))
        if not compute_loss:
            return output

        target = next_event_targets(batch)
        if target.values.numel() == 0:
            raise ValueError("NextEventFields needs at least one valid event token in the batch.")
        next_query = torch.cat(
            [states[target.batch, target.event], self.embed(target.key_ids)], dim=-1
        )
        losses = {
            self.objective_spec.name: self._vocab_loss(
                self.next_proj(next_query),
                target.values,
                label_smoothing=self.objective_spec.label_smoothing,
            )
        }
        if self.mlm_spec is not None:
            assert work.mlm_labels is not None and self.mlm_proj is not None
            b, e, t = (work.mlm_labels != self.mlm_spec.ignore_index).nonzero(as_tuple=True)
            mlm_query = torch.cat([token_states[b, e, t], states[b, e + 1]], dim=-1)
            losses[self.mlm_spec.name] = self._vocab_loss(
                self.mlm_proj(mlm_query),
                work.mlm_labels[b, e, t],
                label_smoothing=self.mlm_spec.label_smoothing,
            )
        if self.past_spec is not None:
            losses[self.past_spec.name] = self.past_spec.weight * self._past_loss(batch, states)
        if self.contrastive_spec is not None:
            losses[self.contrastive_spec.name] = (
                self.contrastive_spec.weight * self._contrastive_loss(batch)
            )
        output.losses = losses
        output.loss = torch.stack(list(losses.values())).sum()
        return output

    def _past_loss(self, batch: KeyValueTimeBatch, states: Tensor) -> Tensor:
        """Cross-entropy of every earlier event's values within `horizon` lags."""
        assert self.past_spec is not None
        assert self.past_proj is not None and self.past_gap is not None
        projected: list[Tensor] = []
        values: list[Tensor] = []
        horizon = self.past_spec.horizon
        for lag in range(1, horizon + 1):
            target = past_targets(batch, lag)
            if target.values.numel() == 0:
                continue
            gap = torch.stack([target.gap, target.lag.to(target.gap.dtype) / horizon], dim=-1)
            query = torch.cat(
                [states[target.batch, target.slot], self.past_gap(gap.to(states.dtype))], dim=-1
            )
            context = self.embed(target.key_ids)
            projected.append(self.past_proj(torch.cat([query, context], dim=-1)))
            values.append(target.values)
        if not projected:
            # A batch of single-event histories has nothing to reconstruct; keep
            # every auxiliary parameter in the graph so DDP stays in step.
            used = self.past_proj.weight.sum() + self.past_gap.weight.sum()
            return used * 0.0
        return self._vocab_loss(torch.cat(projected), torch.cat(values))

    def _contrastive_loss(self, batch: KeyValueTimeBatch) -> Tensor:
        """InfoNCE between two random crops of each history, on the readout."""
        assert self.contrastive_spec is not None and self.contrastive_proj is not None
        views = []
        for _ in range(2):
            crop = random_crop(batch, self.contrastive_spec.crop_fraction)
            states, slot_mask, _tokens = self.encode(crop)
            views.append(self.contrastive_proj(self.pool(states, slot_mask)))
        if batch.batch_size < 2:
            return views[0].sum() * 0.0 + views[1].sum() * 0.0
        return info_nce(views[0], views[1], self.contrastive_spec.temperature)
