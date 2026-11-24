# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequence packing for dense next-event supervision.

The padded collate wastes every position after an entity's last event: with
short histories in a long context, most of each row is padding that contributes
no gradient. Packing concatenates whole tokenized sequences into fixed-length
rows and keeps them isolated with a block-diagonal causal mask, so almost every
position carries a real next-token target.

Boundary semantics, which the domain rules require preserving:

- `segment_ids` marks each token's owning sequence. Attention is block-diagonal,
  so a token never attends across a boundary.
- The first token of each segment is `-100` in `labels`, because it would
  otherwise be the next-token target of the previous segment's last token.
- Padding at the end of a row is masked and labelled `-100`.

Packing is a training-throughput optimization for the pretraining objective. It
does not produce one pooled embedding per entity, so inference and evaluation
keep using the padded collate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.schema import TransactionSequence
from tfm.contracts.tokenizers import FLAT_LAYOUTS, EncodedFlatSequence, SequenceTokenizer


@dataclass(frozen=True)
class PackingStats:
    """How much of the packed batch carries supervision."""

    n_rows: int
    n_sequences: int
    n_truncated: int
    token_capacity: int
    n_real_tokens: int
    n_supervised_targets: int

    @property
    def token_utilization(self) -> float:
        return self.n_real_tokens / self.token_capacity if self.token_capacity else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_rows": self.n_rows,
            "n_sequences": self.n_sequences,
            "n_truncated": self.n_truncated,
            "token_capacity": self.token_capacity,
            "n_real_tokens": self.n_real_tokens,
            "n_supervised_targets": self.n_supervised_targets,
            "token_utilization": self.token_utilization,
        }


def pack_token_rows(
    token_lists: Sequence[Sequence[int]],
    *,
    max_tokens: int,
    pad_id: int = 0,
    ignore_index: int = -100,
) -> tuple[TransactionBatch, PackingStats]:
    """Greedily pack token streams into rows of exactly `max_tokens`.

    First-fit in the given order, so the caller controls locality. A sequence
    longer than `max_tokens` is truncated to the context and gets its own row.
    """
    if max_tokens < 2:
        raise IncompatibleCompositionError(
            f"Packing needs max_tokens >= 2 to form a next-token pair, got {max_tokens}."
        )
    rows: list[list[tuple[int, Sequence[int]]]] = []
    lengths: list[int] = []
    n_truncated = 0
    for index, tokens in enumerate(token_lists):
        seq = list(tokens)[:max_tokens]
        if len(tokens) > max_tokens:
            n_truncated += 1
        if not seq:
            continue
        for row_i, used in enumerate(lengths):
            if used + len(seq) <= max_tokens:
                rows[row_i].append((index, seq))
                lengths[row_i] = used + len(seq)
                break
        else:
            rows.append([(index, seq)])
            lengths.append(len(seq))

    if not rows:
        empty_long = torch.zeros((0, max_tokens), dtype=torch.long)
        return (
            TransactionBatch(
                input_ids=empty_long,
                attention_mask=empty_long.clone(),
                labels=empty_long.clone(),
                segment_ids=empty_long.clone(),
            ),
            PackingStats(0, 0, n_truncated, 0, 0, 0),
        )

    n_rows = len(rows)
    input_ids = torch.full((n_rows, max_tokens), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n_rows, max_tokens), dtype=torch.long)
    labels = torch.full((n_rows, max_tokens), ignore_index, dtype=torch.long)
    # Segment 0 is reserved for padding so padded tails do not join a real
    # segment's attention block.
    segment_ids = torch.zeros((n_rows, max_tokens), dtype=torch.long)
    row_sequence_ids: list[tuple[int, ...]] = []
    n_real = 0
    for row_i, packed in enumerate(rows):
        cursor = 0
        indices: list[int] = []
        for segment_i, (index, seq) in enumerate(packed, start=1):
            end = cursor + len(seq)
            span = torch.tensor(seq, dtype=torch.long)
            input_ids[row_i, cursor:end] = span
            attention_mask[row_i, cursor:end] = 1
            segment_ids[row_i, cursor:end] = segment_i
            labels[row_i, cursor:end] = span
            # This token is the previous segment's next-token target; drop it.
            labels[row_i, cursor] = ignore_index
            indices.append(index)
            cursor = end
            n_real += len(seq)
        row_sequence_ids.append(tuple(indices))

    n_supervised = int((labels[:, 1:] != ignore_index).sum())
    stats = PackingStats(
        n_rows=n_rows,
        n_sequences=sum(len(packed) for packed in rows),
        n_truncated=n_truncated,
        token_capacity=n_rows * max_tokens,
        n_real_tokens=n_real,
        n_supervised_targets=n_supervised,
    )
    batch = TransactionBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        segment_ids=segment_ids,
    )
    return batch, stats


def pack_sequences(
    sequences: Sequence[TransactionSequence],
    tokenizer: SequenceTokenizer,
    *,
    max_tokens: int,
) -> tuple[TransactionBatch, PackingStats]:
    """Tokenize and pack sequences for the causal next-event objective.

    Returns no `fraud_labels` or `numeric_features`: a packed row holds several
    entities, so sequence-level labels have no single owner. Supervised heads and
    evaluation must use `collate_sequences`.
    """
    if tokenizer.layout not in FLAT_LAYOUTS:
        raise IncompatibleCompositionError(
            f"pack_sequences packs the flat layouts {sorted(FLAT_LAYOUTS)}. "
            f"Got tokenizer {tokenizer.name!r} with layout {tokenizer.layout!r}."
        )
    token_lists: list[Sequence[int]] = []
    for seq in sequences:
        encoded = tokenizer.encode(seq)
        if not isinstance(encoded, EncodedFlatSequence):
            raise IncompatibleCompositionError(
                f"Flat-layout encode must return EncodedFlatSequence, got {type(encoded).__name__}."
            )
        token_lists.append(encoded.token_ids)
    pad_id = getattr(tokenizer, "pad_id", None)
    if not isinstance(pad_id, int):
        raise IncompatibleCompositionError(
            f"Tokenizer {tokenizer.name!r} must expose an integer pad_id to be packed."
        )
    return pack_token_rows(token_lists, max_tokens=max_tokens, pad_id=pad_id)
