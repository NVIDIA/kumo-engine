# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RAPIDS cuDF 12-field tokenizer extra. Same layout as the CPU tokenizer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tfm.contracts.schema import TransactionSequence
from tfm.contracts.tokenizers import LAYOUT_FLAT_FIELDS
from tfm.extras.capability import require_extra
from tfm.features.tokenizers.financial import (
    FIELD_ORDER,
    TransactionFieldTokenizer,
)
from tfm.features.tokenizers.registry import register_tokenizer


def cudf_hash_buckets(values: Sequence[str], buckets: int) -> list[int]:
    """Hash every value in one cuDF call (MurmurHash3, `hash_values`), modulo `buckets`."""
    import cudf  # type: ignore[import-not-found]

    hashed = cudf.Series(list(values), dtype="str").hash_values()
    return [int(h) for h in (hashed % buckets).to_arrow().to_pylist()]


@dataclass
class RapidsFieldTokenizer(TransactionFieldTokenizer):
    """flat_fields tokenizer that hashes merchant names on the GPU with cuDF.

    Each batch's unseen merchants are hashed in one `hash_values` call and
    memoized, so a book is hashed once per distinct merchant rather than once
    per event. Buckets differ from the CPU tokenizer's MD5 buckets, so the two
    tokenizers' checkpoints are not interchangeable.
    """

    name: str = "rapids_field_tokenizer"
    layout: str = LAYOUT_FLAT_FIELDS
    support_tier: str = "experimental"

    def __post_init__(self) -> None:
        require_extra("rapids", "cudf")
        super().__post_init__()
        self._buckets: dict[str, int] = {}

    def prepare_merchants(self, sequences: Sequence[TransactionSequence]) -> None:
        unseen = sorted(
            {self.merchant_key(e) for seq in sequences for e in seq.events} - self._buckets.keys()
        )
        if unseen:
            buckets = cudf_hash_buckets(unseen, self.merchant_hash_size)
            self._buckets.update(zip(unseen, buckets, strict=True))

    def _merchant_bucket(self, merchant: str) -> int:
        bucket = self._buckets.get(merchant)
        if bucket is None:
            bucket = cudf_hash_buckets([merchant], self.merchant_hash_size)[0]
            self._buckets[merchant] = bucket
        return bucket


def register() -> None:
    """Register after a cuDF capability check. Safe to call more than once."""
    require_extra("rapids", "cudf")
    register_tokenizer("rapids_field_tokenizer")(RapidsFieldTokenizer)


def advertised_field_order() -> tuple[str, ...]:
    return FIELD_ORDER
