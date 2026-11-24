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
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Self

from tfm.contracts.kvt import EncodedKVTRecord, KeyValueTimeBatch
from tfm.contracts.schema import EntityProfile, TransactionEvent, TransactionSequence
from tfm.contracts.tokenizers import LAYOUT_KEY_VALUE_TIME
from tfm.features.tokenizers.financial import FIELD_ORDER, TransactionFieldTokenizer
from tfm.features.tokenizers.financial import SPECIAL_TOKENS as FLAT_SPECIAL_TOKENS
from tfm.features.tokenizers.key_value_time import (
    KeyValueTimeConfig,
    ProfilePrefix,
    collate_records,
    encode_record,
    key_value_time_config,
    key_value_time_manifest,
)
from tfm.features.tokenizers.registry import register_tokenizer
from tfm.features.tokenizers.vocabulary import UNK, KeyValuePairs, ValueVocabulary

if TYPE_CHECKING:
    from tfm.config import DataConfig, FeaturesConfig


@register_tokenizer("event_fields")
class EventFieldsTokenizer:
    """Flat-field tokens as per-event key-value units, in the `key_value_time` layout."""

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
        del profiles
        self.config = config or KeyValueTimeConfig()
        self.vocabulary = ValueVocabulary(self.config.vocabulary())
        self.pairs = KeyValuePairs(self.vocabulary)
        self._fields = TransactionFieldTokenizer()

    feature_keys: ClassVar[tuple[str, ...]] = ()
    """None: the 12 fields are fixed, so key selection and hashing do not apply."""

    @classmethod
    def from_config(cls, features: FeaturesConfig, data: DataConfig) -> Self:
        return cls(key_value_time_config(features, data))

    def set_profiles(self, profiles: Mapping[str, EntityProfile] | None) -> None:
        del profiles

    @property
    def vocab_size(self) -> int:
        return self.vocabulary.size

    def fit(self, sequences: Sequence[TransactionSequence]) -> EventFieldsTokenizer:
        del sequences
        vocabulary = self.vocabulary
        if vocabulary.fitted:
            return self
        for field in FIELD_ORDER:
            self.pairs.key_vocab[field] = vocabulary.add_token(f"KEY_{field}")
            vocabulary.field_kind[field] = "categorical"
        for token in self._fields.token_to_id:
            if token not in FLAT_SPECIAL_TOKENS:
                vocabulary.add_token(token)
        vocabulary.byte_offset = vocabulary.size
        vocabulary.fitted = True
        return self

    def _encode_event(
        self,
        event: TransactionEvent,
        key_ids: list[int],
        value_ids: list[int],
        positions: list[int],
    ) -> None:
        key_vocab = self.pairs.key_vocab
        token_to_id = self.vocabulary.token_to_id
        for field, token in zip(FIELD_ORDER, self._fields.event_tokens(event), strict=True):
            key_ids.append(key_vocab[field])
            value_ids.append(token_to_id.get(token, UNK))
            positions.append(0)

    def encode(self, sequence: TransactionSequence) -> EncodedKVTRecord:
        if not self.vocabulary.fitted:
            raise RuntimeError("EventFieldsTokenizer is not fitted; call fit() first.")
        return encode_record(
            sequence,
            self.config,
            self._encode_event,
            self.encode_profile(sequence.entity_id, sequence.cutoff_time),
        )

    def encode_profile(self, entity_id: str, cutoff: datetime) -> ProfilePrefix:
        """No profile: the 12-field schema has none."""
        del entity_id, cutoff
        return [], [], [], [0], []

    def collate(
        self, sequences: Sequence[TransactionSequence], *, max_length: int
    ) -> KeyValueTimeBatch:
        return collate_records(
            sequences, [self.encode(seq) for seq in sequences], max_length=max_length
        )

    def field_names(self) -> dict[int, str]:
        """Each `KEY_` id with its field (`FieldNamedTokenizer`)."""
        return self.pairs.field_names()

    def manifest(self) -> dict[str, object]:
        return {
            **key_value_time_manifest(
                self.name, self.layout, self.support_tier, self.config, self.pairs
            ),
            "field_order": list(FIELD_ORDER),
            "source": "transaction_field_tokenizer event tokens",
        }
