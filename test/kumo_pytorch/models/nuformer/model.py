# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Untrained nuFormer-style transaction decoder: causal self-attention, NoPE.

This is an original reference implementation of the transaction backbone,
not the paper's joint-fusion recommendation system. See ARCHITECTURE.md.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import NuFormerConfig


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _kumo_tensor(value, name, dtype, *, cpu=False):
    if not isinstance(value, torch.Tensor) or value.dtype != dtype:
        raise TypeError(f"{name} must be a {dtype} tensor")
    if cpu and value.device.type != "cpu":
        raise TypeError(f"{name} must be a CPU tensor; validate before device transfer")
    return value


def _identifiers(value, name, *, cpu=False):
    if isinstance(value, torch.Tensor):
        value = _kumo_tensor(value, name, torch.int64, cpu=cpu)
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of identifiers")
    if any(type(item) not in (str, int) or item == "" for item in value):
        raise ValueError(f"{name} requires nonempty strings or integer identifiers")
    return tuple(value)


def _check_tokens_and_mask(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, *, validate_inputs: bool = True
) -> None:
    """Check structural metadata always; optionally inspect values on the device."""
    if input_ids.ndim != 2 or min(input_ids.shape) < 1:
        raise ValueError("input_ids must have shape [batch, length] with nonzero dimensions")
    if input_ids.dtype != torch.int64:
        raise TypeError("input_ids must be int64")
    if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be bool with the same [batch, length] shape")
    if attention_mask.device != input_ids.device:
        raise ValueError("input_ids and attention_mask must be on the same device")
    if validate_inputs and bool((attention_mask[:, 1:] & ~attention_mask[:, :-1]).any()):
        raise ValueError("attention_mask must use right padding: True tokens followed by False padding")


@dataclass
class NuFormerEncoding:
    """Sequence representations without allocating vocabulary logits."""

    hidden_states: torch.Tensor  # [B, L, D]; final-normalized, padding slots zero.
    user_repr: torch.Tensor  # [B, D]; last valid token, or zero for empty histories.


@dataclass
class NuFormerOutput:
    logits: torch.Tensor  # [B, L, V]; padding slots are zero.
    hidden_states: torch.Tensor  # [B, L, D]; final-normalized, padding slots zero.
    user_repr: torch.Tensor  # [B, D]; last valid token, or zero for empty histories.


class CausalSelfAttention(nn.Module):
    def __init__(self, config: NuFormerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=config.bias)
        self.proj = nn.Linear(config.dim, config.dim, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # Right padding guarantees a valid query has no padded keys in its causal
        # past. No dense [B, L, L] padding mask is needed. No position embeddings
        # or rotary transform are applied (the paper's NoPE choice).
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True, dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(attended.transpose(1, 2).reshape(batch, length, dim))


class DecoderBlock(nn.Module):
    def __init__(self, config: NuFormerConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps, bias=config.bias)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ffn_multiplier * config.dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.ffn_multiplier * config.dim, config.dim, bias=config.bias),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class NuFormerModel(nn.Module):
    def __init__(self, config: NuFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps, bias=config.bias)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _kumo_inputs(self, batch, input_name, *, validate_inputs, cpu=False):
        history = _mapping(_mapping(batch.inputs, "batch.inputs")[input_name], input_name)
        ids = _kumo_tensor(history["value_ids"], f"{input_name}.ids", torch.int64, cpu=cpu)
        valid = _kumo_tensor(history["valid"], f"{input_name}.valid", torch.bool, cpu=cpu)
        if ids.ndim != 2 or ids.shape[0] < 1 or valid.shape != ids.shape:
            raise ValueError("history IDs and validity require matching nonempty-batch [B,L] shapes")
        if ids.device != valid.device:
            raise ValueError("history IDs and validity must share a device")
        if ids.shape[1] > self.config.max_seq_len:
            raise ValueError("history exceeds the configured model sequence length")
        padding = None
        if validate_inputs:
            metadata = _mapping(batch.metadata, "batch.metadata")
            entities = _identifiers(metadata["entity_ids"], "metadata.entity_ids", cpu=cpu)
            samples = _identifiers(metadata["sample_ids"], "metadata.sample_ids", cpu=cpu)
            if len(entities) != ids.shape[0] or len(samples) != len(entities):
                raise ValueError("metadata requires one entity and sample ID per history row")
            if "is_padding" in metadata:
                padding = _kumo_tensor(metadata["is_padding"], "metadata.is_padding", torch.bool, cpu=cpu)
                if padding.shape != (ids.shape[0],):
                    raise ValueError("metadata.is_padding requires one flag per request")
            if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
                raise ValueError("history.valid must use right padding")
            observed = ids[valid]
            if bool(((observed < 0) | (observed >= self.config.vocab_size)).any()):
                raise ValueError("history.ids contains an ID outside the fitted vocabulary")
        return ids, valid, padding

    def validate_kumo_batch(self, batch, *, input_name="history", target_name=None,
                            next_token_target_names=("next_token",), cpu=True) -> None:
        """Validate inputs and supervision; ``cpu=False`` accepts device-resident batches.

        ``target_name=None`` requires full-history inputs, without an Kumo
        next-token target. Otherwise the named target must already align with
        the emitted history; this method neither shifts nor rewrites labels.
        The training loop remains responsible for selecting valid targets and
        excluding padded requests. No Kumo installation is required by the model.
        """
        ids, valid, padding = self._kumo_inputs(batch, input_name, validate_inputs=True, cpu=cpu)
        targets = _mapping(batch.targets, "batch.targets")
        if target_name is None:
            if any(name in targets for name in next_token_target_names):
                raise ValueError("full-history mode must omit NextTokenTarget in the Kumo contract")
            return
        target = _mapping(targets[target_name], target_name)
        values = _kumo_tensor(target["values"], "target.values", torch.int64, cpu=cpu)
        target_valid = _kumo_tensor(target["valid"], "target.valid", torch.bool, cpu=cpu)
        if values.shape != ids.shape or target_valid.shape != ids.shape:
            raise ValueError("Kumo shifted targets must align exactly with input [B,L]")
        if bool((target_valid & ~valid).any()):
            raise ValueError("a target cannot supervise an input padding position")
        selected = target_valid & (values != -100)
        if padding is not None:
            selected = selected & ~padding[:, None]
        observed = values[selected]
        if bool(((observed < 0) | (observed >= self.config.vocab_size)).any()):
            raise ValueError("target.values contains an ID outside the fitted vocabulary")

    def encode_kumo(self, batch, *, input_name="history", validate_inputs=True) -> NuFormerEncoding:
        """Consume native Kumo history arrays without constructing an adapted batch.

        Masked IDs are made safe for embedding lookup inside the model. An
        emitted zero-width history has one internal all-masked position, matching
        the tensor interface's empty-history representation. The original batch
        is unchanged. Targets are not used by this encoding method.

        After ``validate_kumo_batch`` and device transfer, ``validate_inputs=False``
        skips metadata/value checks while retaining shape, dtype and device checks.
        """
        ids, valid, _ = self._kumo_inputs(batch, input_name, validate_inputs=validate_inputs)
        ids = torch.where(valid, ids, self.config.pad_token_id)
        if ids.shape[1] == 0:
            ids = ids.new_full((ids.shape[0], 1), self.config.pad_token_id)
            valid = valid.new_zeros(ids.shape)
        return self.encode(ids, valid, validate_inputs=False)

    def forward_kumo(self, batch, *, input_name="history", validate_inputs=True) -> NuFormerOutput:
        """Predict from a native batch; Kumo's already-shifted targets stay external."""
        encoding = self.encode_kumo(batch, input_name=input_name, validate_inputs=validate_inputs)
        # The internal empty-history pad is not an emitted token or target.
        width = batch.inputs[input_name]["value_ids"].shape[1]
        hidden = encoding.hidden_states[:, :width]
        return NuFormerOutput(self.lm_head(hidden), hidden, encoding.user_repr)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        validate_inputs: bool = True,
    ) -> NuFormerOutput:
        """Encode one right-padded user sequence per row and predict every token.

        See encode() for mask and validation behavior.
        """
        encoding = self.encode(input_ids, attention_mask, validate_inputs=validate_inputs)
        # Bias-free projection preserves zero padding.
        logits = self.lm_head(encoding.hidden_states)
        return NuFormerOutput(logits, encoding.hidden_states, encoding.user_repr)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        validate_inputs: bool = True,
    ) -> NuFormerEncoding:
        """Encode right-padded histories without running the vocabulary head.

        With no explicit mask, ids != config.pad_token_id identifies real tokens.
        Empty histories are permitted as all-padding rows; B and L must be nonzero.
        validate_inputs=False skips device-value checks for already validated
        batches, avoiding their CPU/GPU synchronization during profiling. Shape,
        dtype, device and configured length checks are always performed.
        """
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id
        _check_tokens_and_mask(input_ids, attention_mask, validate_inputs=validate_inputs)
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f"Sequence length exceeds max_seq_len={self.config.max_seq_len}")
        if input_ids.device != self.token_embedding.weight.device:
            raise ValueError("Input tensors and model parameters must be on the same device")
        if validate_inputs and bool(((input_ids < 0) | (input_ids >= self.config.vocab_size)).any()):
            raise ValueError("All input_ids, including padding slots, must be in the vocabulary")

        valid = attention_mask.unsqueeze(-1)
        hidden = self.embedding_dropout(self.token_embedding(input_ids)) * valid
        for block in self.blocks:
            hidden = block(hidden) * valid
        hidden = self.final_norm(hidden) * valid
        lengths = attention_mask.long().sum(dim=1)
        last = (lengths - 1).clamp_min(0)
        users = hidden[torch.arange(input_ids.shape[0], device=input_ids.device), last]
        users = users * (lengths > 0).unsqueeze(-1)
        return NuFormerEncoding(hidden_states=hidden, user_repr=users)
