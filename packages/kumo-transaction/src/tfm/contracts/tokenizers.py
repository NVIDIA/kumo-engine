# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer layout contracts.

A sequence tokenizer is a composition slot outside ``TrxFM``. It maps
``TransactionSequence`` objects into a layout-specific encoding and packs that
encoding into a batch the matching history model can consume.

Layouts are not interchangeable. Construction-time validation must reject a
tokenizer whose ``layout`` is not in the granularity's ``accepted_layouts``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from tfm.contracts.schema import TransactionSequence

LAYOUT_FLAT_FIELDS = "flat_fields"
LAYOUT_KEY_VALUE_TIME = "key_value_time"
LAYOUT_FLAT_KEY_VALUE = "flat_key_value"

FLAT_LAYOUTS = frozenset({LAYOUT_FLAT_FIELDS, LAYOUT_FLAT_KEY_VALUE})
"""Layouts that encode a sequence as one token stream in a `TransactionBatch`."""


@dataclass(frozen=True)
class FieldVocabulary:
    """Where each field's tokens sit in a `flat_fields` vocabulary.

    Field `i` owns ids `field_ranges[i][0] <= id < field_ranges[i][1]`; every id
    outside the ranges is a special token. Events are joined by `sep_id`.
    """

    vocab_size: int
    field_names: tuple[str, ...]
    field_ranges: tuple[tuple[int, int], ...]
    pad_id: int
    bos_id: int
    sep_id: int
    eos_id: int

    def __post_init__(self) -> None:
        if len(self.field_names) != len(self.field_ranges) or not self.field_ranges:
            raise ValueError("FieldVocabulary needs one id range per field.")
        previous = 0
        for start, stop in self.field_ranges:
            if not previous <= start < stop <= self.vocab_size:
                raise ValueError(
                    f"Field id ranges must be non-empty, ordered, and inside the vocabulary; "
                    f"got {self.field_ranges} for vocab_size {self.vocab_size}."
                )
            previous = stop


@dataclass(frozen=True)
class EncodedFlatSequence:
    """Unpadded token ids for a flat layout (`flat_fields` or `flat_key_value`)."""

    entity_id: str
    sequence_id: str
    token_ids: tuple[int, ...]
    layout: str = LAYOUT_FLAT_FIELDS


class SequenceTokenizer(Protocol):
    """Encode entity sequences into a named layout.

    ``fit`` may be a no-op for data-independent vocabs. Fitted tokenizers must
    be fit on train-partition sequences only.
    """

    name: str
    layout: str
    support_tier: str

    @property
    def vocab_size(self) -> int: ...

    def fit(self, sequences: Sequence[TransactionSequence]) -> SequenceTokenizer: ...

    def encode(self, sequence: TransactionSequence) -> object: ...

    def collate(
        self,
        sequences: Sequence[TransactionSequence],
        *,
        max_length: int,
    ) -> object: ...

    def manifest(self) -> dict[str, object]: ...
