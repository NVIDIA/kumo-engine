# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental flat BERT-style encoder model (flat-encoder guide)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tfm.attention.eager import EagerAttention
from tfm.compositions.flat_encoder import FlatEncoder
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.outputs import ModelOutput
from tfm.contracts.protocols import AttentionSpec
from tfm.layers.kvt_encoders import EncoderBlock
from tfm.objectives.masked_fields import MaskedFields, apply_field_mask, masked_fields_loss
from tfm.pooling.last_event import LastEventMeanPool, last_event_mask
from tfm.pooling.mean import MeanPool, MeanPoolModule
from tfm.pooling.usr_token import UsrTokenPool, UsrTokenPoolModule

FlatEncoderReadout = LastEventMeanPool | MeanPool | UsrTokenPool


class FlatEncoderModel(nn.Module):
    """Bidirectional encoder over `TransactionBatch` rows. Not part of the core TrxFM export.

    Rotary positions follow token index, as in v1's Llama decoder; the encoder
    differs from it only in its attention mask and its objective.
    """

    field_of_id: Tensor
    field_allowed: Tensor
    field_ranges: Tensor

    def __init__(
        self,
        spec: FlatEncoder,
        objective: MaskedFields | None = None,
        attention: AttentionSpec | None = None,
        readout: FlatEncoderReadout | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.objective_spec = objective or MaskedFields()
        self.readout_spec: FlatEncoderReadout = (
            readout if readout is not None else LastEventMeanPool()
        )
        self.pool = (
            UsrTokenPoolModule()
            if isinstance(self.readout_spec, UsrTokenPool)
            else MeanPoolModule()
        )
        self.attention_spec: AttentionSpec = (
            attention if attention is not None else EagerAttention()
        )
        vocab = spec.vocabulary
        self.embed = nn.Embedding(vocab.vocab_size + 1, spec.hidden_size)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    spec.hidden_size,
                    spec.num_heads,
                    spec.dropout,
                    use_rope=True,
                    rope_base=spec.rope_base,
                    attention=self.attention_spec.build(),
                    causal=False,
                )
                for _ in range(spec.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(spec.hidden_size)
        self.lm_head = nn.Linear(spec.hidden_size, vocab.vocab_size, bias=False)
        field_of_id = torch.full((vocab.vocab_size + 1,), -1, dtype=torch.long)
        allowed = torch.zeros((len(vocab.field_ranges), vocab.vocab_size), dtype=torch.bool)
        for index, (start, stop) in enumerate(vocab.field_ranges):
            field_of_id[start:stop] = index
            allowed[index, start:stop] = True
        self.register_buffer("field_of_id", field_of_id, persistent=False)
        self.register_buffer("field_allowed", allowed, persistent=False)
        self.register_buffer(
            "field_ranges", torch.tensor(vocab.field_ranges, dtype=torch.long), persistent=False
        )

    @property
    def hidden_size(self) -> int:
        return self.spec.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.spec.vocab_size

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Final-norm states `[B, S, D]`; padding is never attended to."""
        if input_ids.shape[1] > self.spec.max_seq_len:
            raise IncompatibleCompositionError(
                f"Row of {input_ids.shape[1]} tokens exceeds FlatEncoder "
                f"max_seq_len={self.spec.max_seq_len}."
            )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.float32)
        positions = positions.unsqueeze(0).expand(input_ids.shape[0], -1)
        hidden = self.embed(input_ids)
        for block in self.blocks:
            hidden = block(hidden, attention_mask, rope_positions=positions)
        return self.final_norm(hidden)

    def readout(self, hidden: Tensor, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if isinstance(self.readout_spec, LastEventMeanPool):
            mask = last_event_mask(
                input_ids, attention_mask, self.field_of_id, self.spec.vocabulary.sep_id
            )
            return self.pool(hidden, mask)
        return self.pool(hidden, attention_mask)

    def forward(self, batch: TransactionBatch, *, compute_loss: bool = True) -> ModelOutput:
        if batch.segment_ids is not None:
            raise IncompatibleCompositionError(
                "FlatEncoder reads one entity per row; packed batches would let rows "
                "attend across entities. Use the padded collate."
            )
        input_ids, attention_mask = batch.input_ids, batch.attention_mask
        if not compute_loss:
            hidden = self.encode(input_ids, attention_mask)
            embeddings = self.readout(hidden, input_ids, attention_mask)
            return ModelOutput(hidden_states=hidden, embeddings=embeddings)
        masked, targets = apply_field_mask(
            input_ids,
            attention_mask,
            self.field_of_id,
            self.field_ranges,
            sep_id=self.spec.vocabulary.sep_id,
            mask_id=self.spec.mask_id,
            spec=self.objective_spec,
        )
        hidden = self.encode(masked, attention_mask)
        chosen = targets != self.objective_spec.ignore_index
        logits = self.lm_head(hidden[chosen])
        loss = masked_fields_loss(logits, targets[chosen], self.field_of_id, self.field_allowed)
        return ModelOutput(
            hidden_states=hidden,
            embeddings=self.readout(hidden, input_ids, attention_mask),
            logits=logits,
            loss=loss,
            losses={self.objective_spec.name: loss},
        )
