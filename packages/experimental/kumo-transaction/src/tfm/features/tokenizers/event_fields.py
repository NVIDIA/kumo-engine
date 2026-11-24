# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The 12 flat field tokens, grouped per event, for the hierarchical stacks (experimental).

`transaction_field_tokenizer` writes 12 tokens per transaction into one
stream. This tokenizer emits the same 12 tokens as one unit per event in the
`key_value_time` layout, so the event encoder of `profile_history` or
`profile_event_decoder` can read them. That is TransactionGPT's shape: a
hierarchy over fixed fields rather than over free-form key-value pairs.

Each token's key is its field (`KEY_AMT`, `KEY_MCC`, ...), and its value is
the flat tokenizer's token (`AMT_3`, `MCC_5411`). The vocabulary is fixed, so
`fit` reads no data. Profiles are not encoded: the 12-field schema has none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tfm.contracts.schema import EntityProfile, TransactionEvent, TransactionSequence
from tfm.contracts.tokenizers import LAYOUT_KEY_VALUE_TIME
from tfm.features.tokenizers.financial import FIELD_ORDER, TransactionFieldTokenizer
from tfm.features.tokenizers.financial import SPECIAL_TOKENS as FLAT_SPECIAL_TOKENS
from tfm.features.tokenizers.key_value_time import (
    UNK,
    KeyValueTimeConfig,
    KeyValueTimeTokenizer,
)
from tfm.features.tokenizers.registry import register_tokenizer


@register_tokenizer("event_fields")
class EventFieldsTokenizer(KeyValueTimeTokenizer):
    """Flat-field tokens as per-event key-value units."""

    name: str = "event_fields"
    layout: str = LAYOUT_KEY_VALUE_TIME
    support_tier: str = "experimental"
    requires_flat_fields = True
    uses_profiles = False

    def __init__(
        self,
        config: KeyValueTimeConfig | None = None,
        profiles: Mapping[str, EntityProfile] | None = None,
    ) -> None:
        super().__init__(config, None)
        del profiles
        self._fields = TransactionFieldTokenizer()

    def set_profiles(self, profiles: Mapping[str, EntityProfile] | None) -> None:
        del profiles

    def fit(self, sequences: Sequence[TransactionSequence]) -> EventFieldsTokenizer:
        del sequences
        if self._fitted:
            return self
        for field in FIELD_ORDER:
            self.key_vocab[field] = self._add_token(f"KEY_{field}")
            self.field_kind[field] = "categorical"
        for token in self._fields.token_to_id:
            if token not in FLAT_SPECIAL_TOKENS:
                self._add_token(token)
        self.byte_offset = len(self.token_to_id)
        self._fitted = True
        return self

    def _encode_event(
        self,
        event: TransactionEvent,
        key_ids: list[int],
        value_ids: list[int],
        positions: list[int],
    ) -> None:
        for field, token in zip(FIELD_ORDER, self._fields.event_tokens(event), strict=True):
            key_ids.append(self.key_vocab[field])
            value_ids.append(self.token_to_id.get(token, UNK))
            positions.append(0)

    def manifest(self) -> dict[str, object]:
        return {
            **super().manifest(),
            "tokenizer": self.name,
            "field_order": list(FIELD_ORDER),
            "source": "transaction_field_tokenizer event tokens",
        }
