# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PRAGMA-like: profile and event encoders feeding a bidirectional history encoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from tfm.attention.eager import EagerAttention
from tfm.compositions.profile_history import ProfileHistory
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.outputs import ModelOutput
from tfm.contracts.protocols import AttentionSpec
from tfm.features.tokenizers.key_value_time import EVT, USR
from tfm.layers.kvt_encoders import (
    CalendarEmbedding,
    KVTEventEncoder,
    KVTProfileEncoder,
    encoder_blocks,
    sinusoidal_position,
)
from tfm.objectives.mlm import TokenEventKeyMLM, TokenEventKeyMLMLoss, apply_mlm_mask
from tfm.pooling.registry import Readout, readout_module
from tfm.pooling.usr_token import UsrTokenPool

_RENAMED_KEYS = (
    ("event_blocks.", "event_encoder.blocks."),
    ("event_norm.", "event_encoder.norm."),
    ("profile_blocks.", "profile_encoder.blocks."),
    ("profile_norm.", "profile_encoder.norm."),
)
"""State-dict prefixes of checkpoints saved before the encoders were shared with Blueprint v2."""


def _load_renamed_keys(state_dict: dict[str, Any], prefix: str, *_args: Any) -> None:
    for old, new in _RENAMED_KEYS:
        for key in [k for k in state_dict if k.startswith(prefix + old)]:
            state_dict[prefix + new + key[len(prefix + old) :]] = state_dict.pop(key)


class ProfileHistoryModel(nn.Module):
    """Experimental PRAGMA-style stack. Not part of the core TrxFM export."""

    def __init__(
        self,
        spec: ProfileHistory,
        objective: TokenEventKeyMLM | None = None,
        attention: AttentionSpec | None = None,
        readout: Readout | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.objective_spec = objective or TokenEventKeyMLM()
        self.readout_spec: Readout = readout if readout is not None else UsrTokenPool()
        self.pool = readout_module(self.readout_spec)
        self.attention_spec: AttentionSpec = (
            attention if attention is not None else EagerAttention()
        )
        self.embed = nn.Embedding(spec.vocab_size, spec.hidden_size, padding_idx=0)
        attn = self.attention_spec
        self.event_encoder = KVTEventEncoder(spec, attn, learned_summary=False)
        self.profile_encoder = KVTProfileEncoder(spec, attn, learned_summary=False)
        self.history_blocks = encoder_blocks(
            spec, attn, spec.history_depth, use_rope=True, causal=False
        )
        self.calendar = CalendarEmbedding(spec.hidden_size)
        self.history_norm = nn.LayerNorm(spec.hidden_size)
        self.mlm_proj = nn.Linear(3 * spec.hidden_size, spec.hidden_size, bias=False)
        self.objective_loss = TokenEventKeyMLMLoss(
            ignore_index=self.objective_spec.ignore_index,
            label_smoothing=self.objective_spec.label_smoothing,
        )
        self._register_load_state_dict_pre_hook(_load_renamed_keys)

    @property
    def hidden_size(self) -> int:
        return self.spec.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.spec.vocab_size

    @property
    def last_padding_fraction(self) -> float:
        """Share of the event grid skipped by the last varlen forward. 0 when padded."""
        return self.event_encoder.last_padding_fraction

    def _tokens(self, key_ids: Tensor, value_ids: Tensor, positions: Tensor) -> Tensor:
        token = self.embed(key_ids) + self.embed(value_ids)
        return token + sinusoidal_position(positions, self.spec.hidden_size, self.spec.max_position)

    def forward(self, batch: KeyValueTimeBatch, *, compute_loss: bool = True) -> ModelOutput:
        work = batch
        if compute_loss:
            work = apply_mlm_mask(batch, self.objective_spec)
        z_e, z_hat_e = self.event_encoder(
            self._tokens(work.key_ids, work.value_ids, work.positions),
            work.token_mask,
            work.event_mask,
            summary=self.embed.weight[EVT],
        )
        cal = self.calendar(work.hour, work.dow, work.dom)
        z_e = z_e + cal * work.event_mask.unsqueeze(-1).to(cal.dtype)
        z_a = self.profile_encoder(
            self._tokens(work.prof_key_ids, work.prof_value_ids, work.prof_positions),
            work.prof_mask,
            work.prof_time_log,
            summary=self.embed.weight[USR],
        )
        batch_size, n_events, n_tok, _dim = z_hat_e.shape

        hist_in = torch.cat([z_a.unsqueeze(1), z_e], dim=1)
        hist_mask = torch.cat(
            [
                torch.ones(batch_size, 1, device=z_e.device, dtype=work.event_mask.dtype),
                work.event_mask,
            ],
            dim=1,
        )
        hist_pos = torch.cat(
            [torch.zeros(batch_size, 1, device=z_e.device), work.time_log],
            dim=1,
        )
        hist_h = hist_in
        for block in self.history_blocks:
            hist_h = block(hist_h, hist_mask, rope_positions=hist_pos)
        hist_h = self.history_norm(hist_h)
        z_h_usr = hist_h[:, 0, :]
        z_h_event = hist_h[:, 1:, :]
        ctx = torch.cat(
            [
                z_hat_e,
                z_h_event.unsqueeze(2).expand(-1, -1, n_tok, -1),
                z_h_usr.unsqueeze(1).unsqueeze(2).expand(-1, n_events, n_tok, -1),
            ],
            dim=-1,
        )
        projected = self.mlm_proj(ctx)
        logits = torch.matmul(projected, self.embed.weight.t())
        output = ModelOutput(
            hidden_states=hist_h,
            embeddings=self.pool(hist_h, hist_mask),
            logits=logits,
        )
        if compute_loss:
            output.loss = self.objective_loss(output, work)
        return output
