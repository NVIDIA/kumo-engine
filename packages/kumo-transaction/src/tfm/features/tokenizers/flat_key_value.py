# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Key-value tokens as one flat stream, for the flat decoder (experimental).

This is nuFormer's representation: field-specific tokens in a single causal
stream with separators, instead of PRAGMA's per-event units. It reuses the
key-value-time vocabulary (percentile-binned numerics, categorical values,
byte-level text) and adds the time tokens a flat stream needs, since the flat
decoder has no TimeRoPE:

    [USR] <profile tokens> [EVT] DOW_d HOUR_h DT_b <field tokens> [EVT] ...

- Numeric and categorical values are already field-specific (`VAL_amount_b7`),
  so they are emitted alone. Text fields and unknown values are preceded by
  their `KEY_` token so the model knows which field they belong to.
- `DT_b` buckets the log gap since the previous event, `8·ln(1 + Δt/8)`, so
  every time token depends only on the past.
- `[EVT]` is the event boundary rule 4 requires (ADR 0014).

Rows too long for the context keep the profile prefix and the most recent
whole events; the oldest events are dropped.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tfm.contracts.batch import TransactionBatch
from tfm.contracts.schema import TransactionSequence, ensure_utc
from tfm.contracts.tokenizers import LAYOUT_FLAT_KEY_VALUE, EncodedFlatSequence
from tfm.features.tokenizers.key_value_time import (
    EVT,
    PAD,
    UNK,
    USR,
    KeyValueTimeTokenizer,
    event_fields,
    time_encode,
)
from tfm.features.tokenizers.registry import register_tokenizer

DT_BUCKET_WIDTH = 8.0
N_DT_BUCKETS = 17
"""`8·ln(1 + Δt/8)` is about 74 for a day and 121 for a year; 17 buckets of 8 cover it."""


@register_tokenizer("flat_key_value")
class FlatKeyValueTokenizer(KeyValueTimeTokenizer):
    """Key-value-time vocabulary emitted as one `[EVT]`-separated token stream."""

    name: str = "flat_key_value"
    layout: str = LAYOUT_FLAT_KEY_VALUE
    support_tier: str = "experimental"

    def fit(self, sequences: Sequence[TransactionSequence]) -> FlatKeyValueTokenizer:
        super().fit(sequences)
        for dow in range(7):
            self._add_token(f"DOW_{dow}")
        for hour in range(24):
            self._add_token(f"HOUR_{hour}")
        self._add_token("DT_NA")
        for bucket in range(N_DT_BUCKETS):
            self._add_token(f"DT_{bucket}")
        return self

    @property
    def pad_id(self) -> int:
        return PAD

    def _field_tokens(self, key: str, value: str) -> list[int]:
        keys: list[int] = []
        values: list[int] = []
        positions: list[int] = []
        self._encode_field(key, value, keys, values, positions)
        kind = self.field_kind.get(key)
        if kind == "text":
            return [keys[0], *values] if keys else []
        if values and values[0] == UNK:
            return [keys[0], UNK] if keys and keys[0] != UNK else [UNK]
        return values

    def _segments(self, sequence: TransactionSequence) -> tuple[list[int], list[list[int]]]:
        """The `[USR]` + profile prefix, and one token list per event."""
        if not self._fitted:
            raise RuntimeError(
                "FlatKeyValueTokenizer is not fitted; call fit() on train sequences."
            )
        prefix = [USR]
        profile = self._profiles.get(sequence.entity_id)
        if profile is not None:
            for key, value in sorted(profile.attributes.items()):
                prefix.extend(self._field_tokens(key, value))
            for item in profile.lifelong:
                if item.timestamp <= sequence.cutoff_time:
                    prefix.extend(self._field_tokens(item.key, item.key))
        events = list(sequence.events)
        cap = self.config.max_events_per_user
        if cap is not None and len(events) > cap:
            events = events[-cap:]
        segments: list[list[int]] = []
        previous = None
        for event in events:
            ts = ensure_utc(event.event_timestamp)
            if previous is None:
                gap = self.token_to_id["DT_NA"]
            else:
                delta = max((ts - previous).total_seconds(), 0.0)
                bucket = min(int(time_encode(delta) // DT_BUCKET_WIDTH), N_DT_BUCKETS - 1)
                gap = self.token_to_id[f"DT_{bucket}"]
            previous = ts
            tokens = [
                EVT,
                self.token_to_id[f"DOW_{ts.weekday()}"],
                self.token_to_id[f"HOUR_{ts.hour}"],
                gap,
            ]
            fields = event_fields(event)
            source = fields.pop("source", "transaction")
            tokens.extend(self._field_tokens("source", source))
            for key, value in fields.items():
                tokens.extend(self._field_tokens(key, value))
            cap_tok = self.config.max_event_tokens
            if cap_tok is not None:
                tokens = tokens[: cap_tok + 4]
            segments.append(tokens)
        return prefix, segments

    def encode(self, sequence: TransactionSequence) -> EncodedFlatSequence:  # type: ignore[override]
        prefix, segments = self._segments(sequence)
        ids = list(prefix)
        for tokens in segments:
            ids.extend(tokens)
        return EncodedFlatSequence(
            entity_id=sequence.entity_id,
            sequence_id=sequence.sequence_id or sequence.entity_id,
            token_ids=tuple(ids),
            layout=self.layout,
        )

    def encode_truncated(self, sequence: TransactionSequence, *, max_length: int) -> list[int]:
        """Prefix plus the most recent whole events that fit in `max_length`."""
        prefix, segments = self._segments(sequence)
        prefix = prefix[: max(1, max_length // 2)]
        budget = max_length - len(prefix)
        kept: list[list[int]] = []
        for tokens in reversed(segments):
            if len(tokens) > budget:
                break
            kept.append(tokens)
            budget -= len(tokens)
        ids = list(prefix)
        for tokens in reversed(kept):
            ids.extend(tokens)
        return ids

    def collate(  # type: ignore[override]
        self,
        sequences: Sequence[TransactionSequence],
        *,
        max_length: int,
    ) -> TransactionBatch:
        rows = [self.encode_truncated(seq, max_length=max_length) for seq in sequences]
        input_ids = torch.full((len(rows), max_length), PAD, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
        for b, ids in enumerate(rows):
            input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[b, : len(ids)] = 1
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return TransactionBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            entity_ids=tuple(seq.entity_id for seq in sequences),
            sequence_ids=tuple(seq.sequence_id or seq.entity_id for seq in sequences),
        )

    def manifest(self) -> dict[str, object]:
        return {
            **super().manifest(),
            "tokenizer": self.name,
            "layout": self.layout,
            "time_tokens": ["DOW", "HOUR", "DT"],
            "dt_bucket_width": DT_BUCKET_WIDTH,
            "n_dt_buckets": N_DT_BUCKETS,
            "truncation": "keep profile prefix and most recent whole events",
        }
