# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PRAGMA-style key-value-time tokenizer (experimental, CPU).

This is an independent layout implementation for the TFM kit. It is not
bit-exact with dynamiq-ai/pragmatiq (byte-level text instead of HF BPE).
Fit on train-partition sequences only.

The tokenizer is a `ValueVocabulary` in the key-value layout (`KeyValuePairs`: a
`KEY_<key>` token beside every value token) plus time: each event's calendar fields and
log-time gaps, and the entity's profile as a prefix. `encode_record` and
`collate_records` are that layout for any per-event encoder.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Self
from zoneinfo import ZoneInfo

import numpy as np
import torch
from numpy.typing import NDArray

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.kvt import EncodedKVTRecord, KeyValueTimeBatch
from tfm.contracts.schema import EntityProfile, TransactionEvent, TransactionSequence, ensure_utc
from tfm.contracts.tokenizers import (
    LAYOUT_KEY_VALUE_TIME,
    EncodedBlock,
    EventBlock,
    ProfilePrefix,
)
from tfm.features.tokenizers.registry import register_tokenizer
from tfm.features.tokenizers.vocabulary import (
    EVT,
    MASK,
    PAD,
    SPECIAL_TOKENS,
    UNK,
    USR,
    FitStatistics,
    KeyValuePairs,
    KeyValueStats,
    ValueVocabulary,
    VocabularyConfig,
)

if TYPE_CHECKING:
    from tfm.config import DataConfig, FeaturesConfig


def time_encode(delta_seconds: float) -> float:
    """PRAGMA log-time transform ``8 · ln(1 + Δt/8)``."""
    return float(8.0 * math.log1p(max(delta_seconds, 0.0) / 8.0))


def time_encode_many(delta_seconds: NDArray[np.float64]) -> NDArray[np.float64]:
    """`time_encode` of each element, float64 for float64."""
    return 8.0 * np.log1p(np.where(delta_seconds < 0.0, 0.0, delta_seconds) / 8.0)


def _check_utc(config: KeyValueTimeConfig) -> None:
    if config.calendar_tz != "UTC":
        raise UnsupportedCombinationError(
            "The token cache computes calendar fields in UTC; calendar_tz must be UTC."
        )


@dataclass
class KeyValueTimeConfig:
    n_buckets: int = 16
    categorical_threshold: int = 64
    numeric_min_cardinality: int = 8
    calendar_tz: str = "UTC"
    max_event_tokens: int | None = 24
    max_profile_tokens: int | None = 64
    max_events_per_user: int | None = 32
    seed: int = 0
    categorical_keys: tuple[str, ...] = ()
    """Keys kept categorical even when their values parse as numbers (codes such as `mcc`)."""
    hashed_keys: tuple[str, ...] = ()
    """High-cardinality identity keys encoded as one hash-bucket token (`VocabularyConfig`)."""
    hash_buckets: int = 8192
    """Buckets per hashed key (`VocabularyConfig.hash_buckets`)."""
    excluded_keys: tuple[str, ...] = ()
    """Event keys never fitted or encoded, such as TabFormer's `user`."""

    def vocabulary(self) -> VocabularyConfig:
        """The vocabulary's share of this config."""
        return VocabularyConfig(**{f.name: getattr(self, f.name) for f in fields(VocabularyConfig)})


_CONFIG_FIELDS: tuple[str, ...] = (
    "n_buckets",
    "categorical_threshold",
    "numeric_min_cardinality",
    "calendar_tz",
    "max_event_tokens",
    "max_profile_tokens",
    "max_events_per_user",
    "seed",
)


_STATE_VERSION = 1

EventEncoder = Callable[[TransactionEvent, list[int], list[int], list[int]], None]
"""Appends one event's `(key, value, position)` tokens to the three lists."""


def calendar(ts: datetime, tz: str) -> tuple[int, int, int]:
    """`(hour, weekday, day of month)` of `ts` in time zone `tz`."""
    aware = ensure_utc(ts)
    aware = aware.astimezone(ZoneInfo(tz)) if tz != "UTC" else aware.astimezone(UTC)
    return aware.hour, aware.weekday(), aware.day


def encode_record(
    sequence: TransactionSequence,
    config: KeyValueTimeConfig,
    encode_event: EventEncoder,
    profile: ProfilePrefix,
) -> EncodedKVTRecord:
    """The last `max_events_per_user` events, each encoded by `encode_event` and cut to
    `max_event_tokens`, with their calendar and log-time gaps, after `profile`."""
    events = list(sequence.events)
    cap = config.max_events_per_user
    if cap is not None and len(events) > cap:
        events = events[-cap:]
    key_ids: list[int] = []
    value_ids: list[int] = []
    positions: list[int] = []
    offsets: list[int] = [0]
    hours: list[int] = []
    dows: list[int] = []
    doms: list[int] = []
    ts_us: list[int] = []
    cap_tok = config.max_event_tokens
    for event in events:
        start = len(key_ids)
        encode_event(event, key_ids, value_ids, positions)
        if cap_tok is not None and len(key_ids) - start > cap_tok:
            del key_ids[start + cap_tok :]
            del value_ids[start + cap_tok :]
            del positions[start + cap_tok :]
        offsets.append(len(key_ids))
        hour, dow, dom = calendar(event.event_timestamp, config.calendar_tz)
        hours.append(hour)
        dows.append(dow)
        doms.append(dom)
        ts_us.append(int(ensure_utc(event.event_timestamp).timestamp() * 1_000_000))
    if ts_us:
        last = ts_us[-1]
        tlog = tuple(time_encode(max(last - ts, 0) / 1e6) for ts in ts_us)
    else:
        tlog = ()

    prof_k, prof_v, prof_p, prof_off, prof_t = profile
    return EncodedKVTRecord(
        entity_id=sequence.entity_id,
        sequence_id=sequence.sequence_id or sequence.entity_id,
        key_ids=tuple(key_ids),
        value_ids=tuple(value_ids),
        positions=tuple(positions),
        event_offsets=tuple(offsets),
        event_ts_us=tuple(ts_us),
        time_log=tlog,
        events={"hour": tuple(hours), "dow": tuple(dows), "dom": tuple(doms)},
        prof_key_ids=tuple(prof_k),
        prof_value_ids=tuple(prof_v),
        prof_positions=tuple(prof_p),
        prof_offsets=tuple(prof_off),
        prof_time_log=tuple(prof_t),
    )


def collate_records(
    sequences: Sequence[TransactionSequence],
    records: Sequence[EncodedKVTRecord],
    *,
    max_length: int,
) -> KeyValueTimeBatch:
    """Pad `records`, one per sequence, into a batch. Labels and raw features are attached
    by `tfm.data.collate.collate_batch`, not here."""
    max_events = max((len(rec.time_log) for rec in records), default=1)
    max_events = max(max_events, 1)
    per_event = []
    for rec in records:
        lengths = [
            rec.event_offsets[i + 1] - rec.event_offsets[i] for i in range(len(rec.time_log))
        ]
        per_event.append(max(lengths) if lengths else 1)
    max_tok = min(max(per_event, default=1), max_length)
    max_tok = max(max_tok, 1)
    max_prof = max((len(rec.prof_key_ids) for rec in records), default=1)
    max_prof = max(max_prof, 1)
    bsz = len(records)
    key_ids = torch.full((bsz, max_events, max_tok), PAD, dtype=torch.long)
    value_ids = torch.full((bsz, max_events, max_tok), PAD, dtype=torch.long)
    positions = torch.zeros((bsz, max_events, max_tok), dtype=torch.long)
    token_mask = torch.zeros((bsz, max_events, max_tok), dtype=torch.long)
    event_mask = torch.zeros((bsz, max_events), dtype=torch.long)
    time_log = torch.zeros((bsz, max_events), dtype=torch.float32)
    elapsed_log = torch.zeros((bsz, max_events), dtype=torch.float32)
    names = list(records[0].events) if records else []
    arrays = {name: torch.zeros((bsz, max_events), dtype=torch.long) for name in names}
    prof_key = torch.full((bsz, max_prof), PAD, dtype=torch.long)
    prof_val = torch.full((bsz, max_prof), PAD, dtype=torch.long)
    prof_pos = torch.zeros((bsz, max_prof), dtype=torch.long)
    prof_mask = torch.zeros((bsz, max_prof), dtype=torch.long)
    prof_time = torch.zeros((bsz, max_prof), dtype=torch.float32)
    entity_ids: list[str] = []
    sequence_ids: list[str] = []
    for b, (seq, rec) in enumerate(zip(sequences, records, strict=True)):
        n_events = len(rec.time_log)
        event_mask[b, :n_events] = 1
        time_log[b, :n_events] = torch.tensor(rec.time_log, dtype=torch.float32)
        if n_events:
            first = rec.event_ts_us[0]
            elapsed_log[b, :n_events] = torch.tensor(
                [time_encode((ts - first) / 1e6) for ts in rec.event_ts_us],
                dtype=torch.float32,
            )
        if rec.events.keys() != arrays.keys():
            raise SchemaError(
                f"Record {rec.sequence_id!r} carries per-event arrays {list(rec.events)}, "
                f"the first record {names}; one batch needs the same arrays in every row."
            )
        for name, values in rec.events.items():
            arrays[name][b, :n_events] = torch.tensor(values, dtype=torch.long)
        for e in range(n_events):
            lo, hi = rec.event_offsets[e], rec.event_offsets[e + 1]
            n = min(hi - lo, max_tok)
            if n <= 0:
                continue
            key_ids[b, e, :n] = torch.tensor(rec.key_ids[lo : lo + n], dtype=torch.long)
            value_ids[b, e, :n] = torch.tensor(rec.value_ids[lo : lo + n], dtype=torch.long)
            positions[b, e, :n] = torch.tensor(rec.positions[lo : lo + n], dtype=torch.long)
            token_mask[b, e, :n] = 1
        n_prof = min(len(rec.prof_key_ids), max_prof)
        if n_prof:
            prof_key[b, :n_prof] = torch.tensor(rec.prof_key_ids[:n_prof], dtype=torch.long)
            prof_val[b, :n_prof] = torch.tensor(rec.prof_value_ids[:n_prof], dtype=torch.long)
            prof_pos[b, :n_prof] = torch.tensor(rec.prof_positions[:n_prof], dtype=torch.long)
            prof_mask[b, :n_prof] = 1
            n_time = min(len(rec.prof_time_log), max_prof)
            if n_time:
                # Expand item-level times onto tokens via offsets when possible.
                expanded = [0.0] * n_prof
                for i, tval in enumerate(rec.prof_time_log):
                    start = rec.prof_offsets[i] if i < len(rec.prof_offsets) else i
                    stop = rec.prof_offsets[i + 1] if i + 1 < len(rec.prof_offsets) else n_prof
                    for j in range(start, min(stop, n_prof)):
                        expanded[j] = tval
                prof_time[b, :n_prof] = torch.tensor(expanded, dtype=torch.float32)
        entity_ids.append(seq.entity_id)
        sequence_ids.append(seq.sequence_id or seq.entity_id)
    return KeyValueTimeBatch(
        key_ids=key_ids,
        value_ids=value_ids,
        positions=positions,
        token_mask=token_mask,
        event_mask=event_mask,
        time_log=time_log,
        elapsed_log=elapsed_log,
        events=arrays,
        prof_key_ids=prof_key,
        prof_value_ids=prof_val,
        prof_positions=prof_pos,
        prof_mask=prof_mask,
        prof_time_log=prof_time,
        entity_ids=tuple(entity_ids),
        sequence_ids=tuple(sequence_ids),
    )


def key_value_time_config(features: FeaturesConfig, data: DataConfig) -> KeyValueTimeConfig:
    """A key-value tokenizer's config from the recipe: the last `data.sequence_length`
    events per user, `data.max_tokens` per event, and the `features` keys."""
    return KeyValueTimeConfig(
        max_events_per_user=data.sequence_length,
        max_event_tokens=data.max_tokens,
        categorical_keys=tuple(features.categorical_keys),
        hashed_keys=tuple(features.hashed_keys),
        hash_buckets=features.hash_buckets,
        excluded_keys=tuple(features.exclude_keys),
    )


def key_value_time_manifest(
    name: str,
    layout: str,
    support_tier: str,
    config: KeyValueTimeConfig,
    pairs: KeyValuePairs,
) -> dict[str, object]:
    """The run manifest's tokenizer entry for a key-value tokenizer."""
    vocabulary = pairs.vocabulary
    return {
        "tokenizer": name,
        "layout": layout,
        "support_tier": support_tier,
        "vocab_size": vocabulary.size,
        "n_keys": len(pairs.key_vocab),
        "field_kind": dict(vocabulary.field_kind),
        "calendar_tz": config.calendar_tz,
        "max_events_per_user": config.max_events_per_user,
        "max_event_tokens": config.max_event_tokens,
        "categorical_keys": list(config.categorical_keys),
        "hashed_keys": list(config.hashed_keys),
        "hash_buckets": config.hash_buckets if config.hashed_keys else None,
        "hash": vocabulary.bucket_hash if config.hashed_keys else None,
        "fitted": vocabulary.fitted,
        "text_encoding": "byte_level",
    }


@register_tokenizer("key_value_time")
class KeyValueTimeTokenizer:
    """Experimental key-value-time sequence tokenizer."""

    name: str = "key_value_time"
    layout: str = LAYOUT_KEY_VALUE_TIME
    support_tier: str = "experimental"
    requires_flat_fields = False
    """Reads any event fields, including non-transaction sources."""
    uses_profiles = True
    """Encodes `EntityProfile` attributes and dated milestones as a prefix."""

    def __init__(
        self,
        config: KeyValueTimeConfig | None = None,
        profiles: Mapping[str, EntityProfile] | None = None,
    ) -> None:
        self.config = config or KeyValueTimeConfig()
        self._profiles: dict[str, EntityProfile] = dict(profiles or {})
        self.vocabulary = ValueVocabulary(self.config.vocabulary())
        self.pairs = KeyValuePairs(self.vocabulary)

    feature_keys: ClassVar[tuple[str, ...]] = (
        "categorical_keys",
        "hashed_keys",
        "hash_buckets",
        "exclude_keys",
    )
    """The `features` keys `from_config` reads (`ConfiguredTokenizer`)."""

    @classmethod
    def from_config(cls, features: FeaturesConfig, data: DataConfig) -> Self:
        return cls(key_value_time_config(features, data))

    def set_profiles(self, profiles: Mapping[str, EntityProfile] | None) -> None:
        self._profiles = dict(profiles or {})

    @property
    def vocab_size(self) -> int:
        return self.vocabulary.size

    @property
    def pad_id(self) -> int:
        return PAD

    @property
    def mask_id(self) -> int:
        return MASK

    @property
    def unk_id(self) -> int:
        return UNK

    def fit(self, sequences: Sequence[TransactionSequence]) -> KeyValueTimeTokenizer:
        return self.fit_statistics(self.collect_statistics(sequences))

    def collect_statistics(self, sequences: Sequence[TransactionSequence]) -> FitStatistics:
        """What `fit` reads from `sequences` and their profiles, in the order it reads it."""
        return self.vocabulary.collect(sequences, self._profiles)

    def fit_statistics(self, statistics: FitStatistics) -> KeyValueTimeTokenizer:
        """Fit from collected statistics; `fit` is `collect_statistics` then this."""
        self.pairs.fit(statistics)
        return self

    # Column-wise (`ColumnarTokenizer`) ---------------------------------------

    def fit_block(self, block: EventBlock, profiles: Mapping[str, EntityProfile]) -> KeyValueStats:
        """`collect_statistics` of `block`'s events and `profiles`
        (`ValueVocabulary.block_statistics`)."""
        _check_utc(self.config)
        return self.vocabulary.block_statistics(block, profiles)

    def merge(self, a: KeyValueStats, b: KeyValueStats) -> KeyValueStats:
        """`ValueVocabulary.merge`: the same in any order; reuses `a`'s storage."""
        return self.vocabulary.merge(a, b)

    def fit_stats(self, stats: KeyValueStats) -> Self:
        """Fit from the merged statistics of every block."""
        self.pairs.fit(stats.fit_statistics())
        return self

    def encode_block(self, block: EventBlock) -> EncodedBlock:
        """Every event's tokens as `encode` gives them, and its UTC `hour`, `dow` and
        `dom`. Excluded keys get no tokens; an excluded `source` still opens its event,
        as `[UNK]`."""
        _check_utc(self.config)
        offsets, key_ids, value_ids, positions, truncated = self.pairs.encode_block(
            block, self.config.max_event_tokens
        )
        if self.vocab_size >= 1 << 16 or (len(positions) and positions.max() >= 1 << 8):
            raise UnsupportedCombinationError(
                "The token cache stores ids as uint16 and positions as uint8; this vocabulary "
                f"({self.vocab_size}) or text length does not fit."
            )
        days = np.floor_divide(block.ts_us, 86_400_000_000)
        day = days.astype("datetime64[D]")
        dom = (day - day.astype("datetime64[M]").astype("datetime64[D]")).astype(np.int64) + 1
        return EncodedBlock(
            token_offsets=offsets,
            key_ids=key_ids.astype(np.uint16),
            value_ids=value_ids.astype(np.uint16),
            positions=positions.astype(np.uint8),
            events={
                "hour": (np.floor_divide(block.ts_us, 3_600_000_000) % 24).astype(np.uint8),
                "dow": ((days + 3) % 7).astype(np.uint8),
                "dom": dom.astype(np.uint8),
            },
            truncated_events=truncated,
        )

    # Per sequence ------------------------------------------------------------

    def encode(self, sequence: TransactionSequence) -> EncodedKVTRecord:
        if not self.vocabulary.fitted:
            raise RuntimeError(
                "KeyValueTimeTokenizer is not fitted; call fit() on train sequences."
            )
        return encode_record(
            sequence,
            self.config,
            self.pairs.encode_event,
            self.encode_profile(sequence.entity_id, sequence.cutoff_time),
        )

    def encode_profile(self, entity_id: str, cutoff: datetime) -> ProfilePrefix:
        """`(keys, values, positions, item offsets, item times)` of the profile prefix."""
        prof_k: list[int] = []
        prof_v: list[int] = []
        prof_p: list[int] = []
        prof_off: list[int] = [0]
        prof_t: list[float] = []
        profile = self._profiles.get(entity_id)
        if profile is None:
            return prof_k, prof_v, prof_p, prof_off, prof_t
        encode_field = self.pairs.encode_field
        for key, value in sorted(profile.attributes.items()):
            encode_field(key, value, prof_k, prof_v, prof_p)
            prof_off.append(len(prof_k))
            prof_t.append(0.0)
        # Milestones are dated; a snapshot's `as_of` can postdate the cutoff,
        # so each one is kept only if it happened by the sequence's cutoff.
        ref = int(cutoff.timestamp() * 1_000_000)
        for item in profile.lifelong:
            if item.timestamp > cutoff:
                continue
            encode_field(item.key, item.key, prof_k, prof_v, prof_p)
            prof_off.append(len(prof_k))
            item_us = int(item.timestamp.timestamp() * 1_000_000)
            prof_t.append(time_encode(max(ref - item_us, 0) / 1e6))
        cap_p = self.config.max_profile_tokens
        if cap_p is not None and len(prof_k) > cap_p:
            keep = 0
            while keep < len(prof_t) and prof_off[keep + 1] <= cap_p:
                keep += 1
            cut = prof_off[keep]
            del prof_k[cut:]
            del prof_v[cut:]
            del prof_p[cut:]
            prof_off = prof_off[: keep + 1]
            prof_t = prof_t[:keep]
        return prof_k, prof_v, prof_p, prof_off, prof_t

    def collate(
        self,
        sequences: Sequence[TransactionSequence],
        *,
        max_length: int,
    ) -> KeyValueTimeBatch:
        """Pad records into a batch (`collate_records`)."""
        return collate_records(
            sequences, [self.encode(seq) for seq in sequences], max_length=max_length
        )

    # State -------------------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        """The config, state format, and cache encoding semantics."""
        return {
            "event_encoding": "implicit-source-v1",
            "state_version": _STATE_VERSION,
            **{k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self.config).items()},
        }

    def state_dict(self) -> dict[str, Any]:
        """The fitted vocabulary and binning, JSON-serializable. Profiles are not included."""
        if not self.vocabulary.fitted:
            raise RuntimeError("KeyValueTimeTokenizer is not fitted; nothing to save.")
        config: dict[str, Any] = {
            **{k: getattr(self.config, k) for k in _CONFIG_FIELDS},
            "categorical_keys": list(self.config.categorical_keys),
        }
        if self.config.hashed_keys:
            config["hashed_keys"] = list(self.config.hashed_keys)
            config["hash_buckets"] = self.config.hash_buckets
        if self.config.excluded_keys:
            config["excluded_keys"] = list(self.config.excluded_keys)
        return {"version": _STATE_VERSION, "config": config, **self.pairs.state_dict()}

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any], profiles: Mapping[str, EntityProfile] | None = None
    ) -> Self:
        """A fitted tokenizer from `state_dict()`; a state saved without a version is version 1."""
        if state.get("version", 1) != _STATE_VERSION:
            raise SchemaError(
                f"This tokenizer state has version {state['version']}; this SDK reads "
                f"{_STATE_VERSION}. Refit the tokenizer, or rebuild the token cache."
            )
        raw = dict(state["config"])
        raw["categorical_keys"] = tuple(raw.get("categorical_keys", ()))
        raw["hashed_keys"] = tuple(raw.get("hashed_keys", ()))
        raw["excluded_keys"] = tuple(raw.get("excluded_keys", ()))
        tokenizer = cls(KeyValueTimeConfig(**raw), profiles)
        tokenizer.pairs = KeyValuePairs.from_state(tokenizer.config.vocabulary(), state)
        tokenizer.vocabulary = tokenizer.pairs.vocabulary
        return tokenizer

    def field_names(self) -> dict[int, str]:
        """Each `KEY_` id with its key (`FieldNamedTokenizer`)."""
        return self.pairs.field_names()

    def manifest(self) -> dict[str, object]:
        return key_value_time_manifest(
            self.name, self.layout, self.support_tier, self.config, self.pairs
        )


__all__ = [
    "EVT",
    "MASK",
    "PAD",
    "SPECIAL_TOKENS",
    "UNK",
    "USR",
    "EventEncoder",
    "KeyValueTimeConfig",
    "KeyValueTimeTokenizer",
    "ProfilePrefix",
    "calendar",
    "collate_records",
    "encode_record",
    "key_value_time_config",
    "key_value_time_manifest",
    "time_encode",
    "time_encode_many",
]
