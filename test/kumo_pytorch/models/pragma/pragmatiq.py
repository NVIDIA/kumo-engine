# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pragmatiq model: profile / event / history encoders.

Plain PyTorch reference implementation. Input encoding and dataset loading
belong to the caller; the model uses ordinary tensor operations and autograd.

Pipeline for one :class:`~models.pragma.batch.PackedBatch`:

1. ``TokenEmbedding`` → per-token vectors ``x`` (key+value+within-field pos).
2. ``EventEncoder`` encodes each event independently (segments = events): a
   per-event ``[EVT]`` marker is prepended, blocks run with block-diagonal
   attention; token outputs are ``ẑ_e`` (for MLM) and the ``[EVT]`` outputs,
   plus ``CalendarEmbedding``, give the per-event vector ``z_e``.
3. ``ProfileStateEncoder`` runs over each user's profile tokens with a ``[USR]``
   marker and TimeRoPE on the per-item log-seconds → ``z_a`` (the ``[USR]``
   output).
4. ``HistoryEncoder`` runs over each user's sequence ``[z_a, z_e…]`` with
   TimeRoPE on log-seconds-to-last-event → ``z_h``; the ``[USR]`` slot output is
   the user embedding, the event slots are per-event history states.

The forward returns these representations; the MLM head (heads.py) consumes
``ẑ_e``, the per-event history states, and the user embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .batch import PackedBatch
from .constants import EVT, USR
from .embeddings import CalendarEmbedding, TokenEmbedding
from .layers import Encoder, _segment_boundaries


@dataclass
class ModelConfig:
    """Explicit pragmatiq model dimensions."""

    vocab_size: int
    dim: int = 192
    n_heads: int = 3
    depth_profile: int = 1
    depth_event: int = 5
    depth_history: int = 2
    dropout: float = 0.1
    rope_base: float = 10_000.0  # GUESS: RoPE base for the time axis
    max_position: int = 64


def assemble_segments(
    seg_lengths: torch.Tensor,
    prefix_vec: torch.Tensor,
    token_vals: torch.Tensor,
    token_pos: torch.Tensor | None = None,
    prefix_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Prepend one ``prefix_vec`` per segment to the segment's tokens.

    Returns ``(x, cu_seqlens, prefix_idx, token_dst, rope_pos)`` where ``x`` is
    the assembled flat sequence ``[T + S, d]``, ``prefix_idx`` ``[S]`` and
    ``token_dst`` ``[T]`` index the prefix and token rows of ``x`` respectively.
    ``seg_lengths`` must be nonnegative and add up to ``T``.
    """
    S = seg_lengths.numel()
    T = token_vals.shape[0]
    d = token_vals.shape[1]
    device = token_vals.device
    if seg_lengths.ndim != 1 or seg_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("segment lengths must be a one-dimensional integer tensor")
    if seg_lengths.device != device or prefix_vec.device != device:
        raise ValueError("segment lengths, prefixes, and tokens must share a device")
    if bool((seg_lengths < 0).any()) or int(seg_lengths.sum()) != T:
        raise ValueError("segment lengths must be nonnegative and sum to the token count")
    if prefix_vec.shape != (S, d):
        raise ValueError("one prefix vector is required per segment")
    new_lengths = seg_lengths + 1
    new_cu = F.pad(new_lengths.cumsum(0), (1, 0)).to(torch.int32)  # [S+1]
    prefix_idx = new_cu[:-1].to(torch.long)
    x = token_vals.new_zeros(T + S, d)
    # Token t of segment s follows the prefixes of segments 0 through s.
    seg_of_token = torch.repeat_interleave(
        torch.arange(S, device=device), seg_lengths, output_size=T
    )
    token_dst = torch.arange(1, T + 1, device=device) + seg_of_token
    # Prefix and token destinations partition the rows; writes never overlap.
    x.index_copy_(0, prefix_idx, prefix_vec.to(x.dtype))
    x.index_copy_(0, token_dst, token_vals)
    rope_pos = None
    if token_pos is not None:
        # Positions are pure data: keep them fp32 regardless of segment dtype so
        # log-second resolution is never quantized under bf16-true precision
        # (angles() upcasts; rotate() casts cos/sin back to x.dtype).
        rope_pos = torch.zeros(T + S, dtype=torch.float32, device=device)
        rope_pos.index_copy_(0, token_dst, token_pos.float())
        if prefix_pos is not None:
            rope_pos.index_copy_(0, prefix_idx, prefix_pos.float())
    return x, new_cu, prefix_idx, token_dst, rope_pos


def _segsum(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Sum ``values`` within consecutive groups of sizes ``counts`` → ``[len(counts)]``.

    ``counts`` must add up to ``len(values)``.
    """
    if values.numel() == 0:
        return torch.zeros_like(counts)
    seg = torch.repeat_interleave(
        torch.arange(counts.numel(), device=values.device), counts, output_size=values.numel()
    )
    out = torch.zeros(counts.numel(), dtype=values.dtype, device=values.device)
    out.scatter_add_(0, seg, values)
    return out


@dataclass
class PragmaOutput:
    """Representations produced by :meth:`PragmaModel.forward`."""

    token_repr: torch.Tensor  # ẑ_e  [T, d] event-encoder token outputs (for MLM)
    event_repr: torch.Tensor  # z_e  [E, d] per-event vectors
    history_event_repr: torch.Tensor  # z_h[event]  [E, d]
    user_repr: torch.Tensor  # z_h[USR]  [n_users, d] user embeddings
    event_of_token: torch.Tensor  # [T]
    user_of_event: torch.Tensor  # [E]


class PragmaModel(nn.Module):
    """Bidirectional pragmatiq encoder stack (profile → event → history)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.dim
        self.embed = TokenEmbedding(config.vocab_size, d, config.max_position, config.dropout)
        self.calendar = CalendarEmbedding(d)
        self.event_encoder = Encoder(d, config.depth_event, config.n_heads, config.dropout,
                                     use_rope=False)
        self.profile_encoder = Encoder(d, config.depth_profile, config.n_heads, config.dropout,
                                       use_rope=True, rope_base=config.rope_base)
        self.history_encoder = Encoder(d, config.depth_history, config.n_heads, config.dropout,
                                       use_rope=True, rope_base=config.rope_base)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @property
    def embedding_weight(self) -> torch.Tensor:
        """Shared key/value table (tied to the MLM logits)."""
        return self.embed.weight

    # ------------------------------------------------------------------ encoders
    def _encode_events(
        self, batch: PackedBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_tok = self.embed(batch.key_ids, batch.value_ids, batch.positions)  # [T, d]
        seg_lengths = (batch.cu_seqlens_event[1:] - batch.cu_seqlens_event[:-1]).long()
        evt_marker = self.embed.embed(torch.full((seg_lengths.numel(),), EVT, device=x_tok.device))
        x, cu, prefix_idx, token_dst, _ = assemble_segments(seg_lengths, evt_marker, x_tok)
        h = self.event_encoder(x, cu, batch.max_len_event)
        z_tok = h[token_dst]  # ẑ_e  [T, d]
        evt_vec = h[prefix_idx]  # [E, d]
        z_e = evt_vec + self.calendar(batch.event_hour, batch.event_dow, batch.event_dom)
        return z_tok, z_e

    def _encode_profile(self, batch: PackedBatch) -> torch.Tensor:
        x_prof = self.embed(batch.prof_key_ids, batch.prof_value_ids, batch.prof_positions)  # [P, d]
        item_lengths = (batch.cu_seqlens_profile_item[1:] - batch.cu_seqlens_profile_item[:-1]).long()
        items_per_user = (batch.cu_seqlens_profile[1:] - batch.cu_seqlens_profile[:-1]).long()
        tokens_per_user = _segsum(item_lengths, items_per_user)  # [n_users]
        # per-token time = its item's log-seconds
        tok_time = batch.prof_time_log[batch.item_of_prof_token] if x_prof.shape[0] else x_prof.new_zeros(0)
        usr_marker = self.embed.embed(torch.full((tokens_per_user.numel(),), USR, device=x_prof.device))
        usr_pos = x_prof.new_zeros(tokens_per_user.numel())  # [USR] anchored at log-sec 0
        x, cu, prefix_idx, _, rope_pos = assemble_segments(
            tokens_per_user, usr_marker, x_prof, tok_time, usr_pos
        )
        h = self.profile_encoder(x, cu, batch.max_len_profile, rope_pos)
        return h[prefix_idx]  # z_a  [n_users, d]

    def _encode_history(
        self, batch: PackedBatch, z_a: torch.Tensor, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        events_per_user = (batch.cu_seqlens_history[1:] - batch.cu_seqlens_history[:-1]).long()
        evt_pos = batch.event_time_log  # log-sec to last event
        usr_pos = z_a.new_zeros(events_per_user.numel())  # [USR] at log-sec 0 (anchor)
        x, cu, prefix_idx, token_dst, rope_pos = assemble_segments(
            events_per_user, z_a, z_e, evt_pos, usr_pos
        )
        h = self.history_encoder(x, cu, batch.max_len_history, rope_pos)
        return h[prefix_idx], h[token_dst]  # z_h[USR] [n_users,d], z_h[event] [E,d]

    def forward(self, batch: PackedBatch) -> PragmaOutput:
        """Encode a packed batch into token/event/user representations."""
        # Check the caller's offsets before rebuilding segments with marker
        # tokens. Rebuilding from lengths alone would hide a nonzero start.
        profile_items = batch.cu_seqlens_profile_item.numel() - 1
        for offsets, total, segments in (
            (batch.cu_seqlens_event, batch.n_tokens, batch.n_events),
            (batch.cu_seqlens_history, batch.n_events, batch.n_users),
            (batch.cu_seqlens_profile_item, batch.prof_key_ids.numel(), profile_items),
            (batch.cu_seqlens_profile, profile_items, batch.n_users),
        ):
            _segment_boundaries(offsets, total, None)
            if offsets.numel() != segments + 1:
                raise ValueError("packed boundaries must contain one segment per event, item, or user")
        z_tok, z_e = self._encode_events(batch)
        z_a = self._encode_profile(batch)
        z_h_usr, z_h_event = self._encode_history(batch, z_a, z_e)
        return PragmaOutput(
            token_repr=z_tok, event_repr=z_e, history_event_repr=z_h_event,
            user_repr=z_h_usr, event_of_token=batch.event_of_token, user_of_event=batch.user_of_event,
        )

    def embed_users(self, batch: PackedBatch) -> torch.Tensor:
        """Convenience: user embeddings ``z_h[USR]`` only ``[n_users, d]``."""
        return self.forward(batch).user_repr

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
