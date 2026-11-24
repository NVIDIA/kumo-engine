# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PRAGMA-style key-value-time tokenizer (experimental, CPU).

This is an independent layout implementation for the TFM kit. It is not
bit-exact with dynamiq-ai/pragmatiq (byte-level text instead of HF BPE).
Fit on train-partition sequences only.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import torch

from tfm.contracts.kvt import EncodedKVTRecord, KeyValueTimeBatch
from tfm.contracts.schema import EntityProfile, TransactionEvent, TransactionSequence, ensure_utc
from tfm.contracts.tokenizers import LAYOUT_KEY_VALUE_TIME
from tfm.features.tokenizers.registry import register_tokenizer

PAD, MASK, UNK, USR, EVT = 0, 1, 2, 3, 4
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[MASK]", "[UNK]", "[USR]", "[EVT]")


def time_encode(delta_seconds: float) -> float:
    """PRAGMA log-time transform ``8 · ln(1 + Δt/8)``."""
    return float(8.0 * math.log1p(max(delta_seconds, 0.0) / 8.0))


def hash_bucket(key: str, value: str, buckets: int) -> int:
    """A process-independent bucket for `(key, value)`; Python's `hash` is salted per run."""
    digest = hashlib.blake2b(f"{key}\x1f{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def event_fields(event: TransactionEvent) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in event.categorical.items():
        fields[str(key)] = str(value)
    for key, value in event.numeric.items():
        fields[str(key)] = str(value)
    for key, value in event.text.items():
        fields[str(key)] = str(value)
    return fields


@dataclass
class PercentileBinner:
    n_buckets: int = 16
    edges: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def fit(self, values: np.ndarray) -> PercentileBinner:
        nz = np.asarray(values, dtype=np.float64)
        nz = nz[np.isfinite(nz) & (nz != 0.0)]
        if len(nz) == 0:
            self.edges = np.zeros(0, dtype=np.float64)
            return self
        qs = np.linspace(0, 1, self.n_buckets + 1)[1:-1]
        self.edges = np.unique(np.quantile(nz, qs))
        return self

    @property
    def n_bins(self) -> int:
        return int(len(self.edges) + 2)

    def transform_one(self, value: float) -> int:
        x = float(value)
        if not math.isfinite(x) or x == 0.0:
            return 0
        return int(np.searchsorted(self.edges, x, side="right") + 1)


@dataclass
class KeyFitStatistics:
    """What `fit` saw for one key: every sample and every numeric reading."""

    counts: dict[str, int]
    """Sample counts, in the order each value first appeared."""
    n_samples: int
    numeric: list[float] | np.ndarray


@dataclass
class FitStatistics:
    """Everything `fit` needs, collected from sequences or built column-wise.

    `keys` is in the order each key first appeared: vocabulary ids and the
    tie-break among equally frequent values both follow it.
    """

    keys: dict[str, KeyFitStatistics]


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
    """High-cardinality identity keys (merchant, device, counterparty) encoded as one
    token, a stable hash bucket of the value, instead of its bytes."""
    hash_buckets: int = 8192
    """Buckets per hashed key, each key with its own range: the output layer is
    tied to the input embedding, so a bucket shared across keys would have to
    be a merchant, a device, and a counterparty at once. Nothing is fitted: an
    unseen value gets a bucket too, and two values of one key that collide
    share it."""
    excluded_keys: tuple[str, ...] = ()
    """Event keys never fitted or encoded: identity keys such as TabFormer's `user`,
    which repeats one value in every event of a history."""


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
        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: dict[int, str] = dict(enumerate(SPECIAL_TOKENS))
        self.key_vocab: dict[str, int] = {}
        self.field_kind: dict[str, str] = {}
        self.binners: dict[str, PercentileBinner] = {}
        self.binner_base: dict[str, int] = {}
        self.cat_vocab: dict[str, dict[str, int]] = {}
        self.hash_base: dict[str, int] = {}
        self.byte_offset: int = 0
        self._fitted = False

    def set_profiles(self, profiles: Mapping[str, EntityProfile] | None) -> None:
        self._profiles = dict(profiles or {})

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return PAD

    @property
    def mask_id(self) -> int:
        return MASK

    @property
    def unk_id(self) -> int:
        return UNK

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    def fit(self, sequences: Sequence[TransactionSequence]) -> KeyValueTimeTokenizer:
        return self.fit_statistics(self.collect_statistics(sequences))

    def collect_statistics(self, sequences: Sequence[TransactionSequence]) -> FitStatistics:
        """What `fit` reads from `sequences`, in the order it reads it."""
        keys: dict[str, KeyFitStatistics] = {}

        def sample(key: str, raw: str) -> KeyFitStatistics:
            stats = keys.get(key)
            if stats is None:
                stats = keys[key] = KeyFitStatistics(counts={}, n_samples=0, numeric=[])
            stats.counts[raw] = stats.counts.get(raw, 0) + 1
            stats.n_samples += 1
            return stats

        excluded = set(self.config.excluded_keys)
        for seq in sequences:
            for event in seq.events:
                for key, raw in event_fields(event).items():
                    if key in excluded:
                        continue
                    stats = sample(key, raw)
                    numeric = cast(list[float], stats.numeric)
                    if key in event.numeric:
                        numeric.append(float(event.numeric[key]))
                    else:
                        try:
                            parsed = float(raw)
                        except ValueError:
                            continue
                        if math.isfinite(parsed):
                            numeric.append(parsed)
            profile = self._profiles.get(seq.entity_id)
            if profile is not None:
                for key, raw in profile.attributes.items():
                    sample(key, raw)
                for item in profile.lifelong:
                    if item.timestamp <= seq.cutoff_time:
                        sample(item.key, item.key)
        return FitStatistics(keys=keys)

    def fit_statistics(self, statistics: FitStatistics) -> KeyValueTimeTokenizer:
        """Fit from collected statistics; `fit` is `collect_statistics` then this."""
        for key, stats in statistics.keys.items():
            distinct = len(stats.counts)
            nums = stats.numeric
            parse_ratio = len(nums) / max(stats.n_samples, 1)
            if key in self.config.hashed_keys:
                self.field_kind[key] = "hashed"
            elif key in self.config.categorical_keys:
                self.field_kind[key] = "categorical"
            elif (
                parse_ratio >= 0.8
                and len(nums) >= 4
                and distinct >= self.config.numeric_min_cardinality
            ):
                self.field_kind[key] = "numeric"
            elif distinct <= self.config.categorical_threshold:
                self.field_kind[key] = "categorical"
            else:
                self.field_kind[key] = "text"

        for key in sorted(self.field_kind):
            self.key_vocab[key] = self._add_token(f"KEY_{key}")

        for key, kind in self.field_kind.items():
            if kind == "numeric":
                binner = PercentileBinner(self.config.n_buckets)
                binner.fit(np.asarray(statistics.keys[key].numeric, dtype=np.float64))
                self.binners[key] = binner
                self.binner_base[key] = len(self.token_to_id)
                for bucket in range(binner.n_bins):
                    self._add_token(f"VAL_{key}_b{bucket}")
            elif kind == "categorical":
                counts = Counter(statistics.keys[key].counts)
                self.cat_vocab[key] = {}
                for value, _ in counts.most_common(self.config.categorical_threshold):
                    self.cat_vocab[key][value] = self._add_token(f"VAL_{key}_{value}")

        for key in sorted(k for k, kind in self.field_kind.items() if kind == "hashed"):
            self.hash_base[key] = len(self.token_to_id)
            for bucket in range(self.config.hash_buckets):
                self._add_token(f"VAL_{key}_h{bucket}")
        self.byte_offset = len(self.token_to_id)
        for byte in range(256):
            self._add_token(f"BYTE_{byte}")
        self._fitted = True
        return self

    def _calendar(self, ts: datetime) -> tuple[int, int, int]:
        aware = ensure_utc(ts)
        if self.config.calendar_tz != "UTC":
            aware = aware.astimezone(ZoneInfo(self.config.calendar_tz))
        else:
            aware = aware.astimezone(UTC)
        return aware.hour, aware.weekday(), aware.day

    def _encode_field(
        self,
        key: str,
        value: str,
        key_ids: list[int],
        value_ids: list[int],
        positions: list[int],
    ) -> None:
        kid = self.key_vocab.get(key)
        if kid is None:
            key_ids.append(UNK)
            value_ids.append(UNK)
            positions.append(0)
            return
        kind = self.field_kind.get(key, "categorical")
        if kind == "numeric":
            try:
                parsed = float(value)
            except ValueError:
                parsed = float("nan")
            if not math.isfinite(parsed):
                key_ids.append(kid)
                value_ids.append(UNK)
                positions.append(0)
                return
            bucket = self.binners[key].transform_one(parsed)
            key_ids.append(kid)
            value_ids.append(self.binner_base[key] + bucket)
            positions.append(0)
        elif kind == "categorical":
            vid = self.cat_vocab[key].get(value, UNK)
            key_ids.append(kid)
            value_ids.append(vid)
            positions.append(0)
        elif kind == "hashed":
            key_ids.append(kid)
            value_ids.append(
                self.hash_base[key] + hash_bucket(key, value, self.config.hash_buckets)
            )
            positions.append(0)
        else:
            data = value.encode("utf-8")[:32] or b"\x00"
            for pos, byte in enumerate(data):
                key_ids.append(kid)
                value_ids.append(self.byte_offset + byte)
                positions.append(pos)

    def _encode_event(
        self,
        event: TransactionEvent,
        key_ids: list[int],
        value_ids: list[int],
        positions: list[int],
    ) -> None:
        """Append one event's `(key, value, position)` tokens, `source` first.

        Single-token fields follow in key order and byte-level text fields come
        last, so `max_event_tokens` only ever trims text bytes.
        """
        fields = event_fields(event)
        for key in self.config.excluded_keys:
            fields.pop(key, None)
        source = fields.pop("source", "transaction")
        self._encode_field("source", source, key_ids, value_ids, positions)
        for key in sorted(fields, key=lambda k: (self.field_kind.get(k) == "text", k)):
            self._encode_field(key, fields[key], key_ids, value_ids, positions)

    def encode(self, sequence: TransactionSequence) -> EncodedKVTRecord:
        if not self._fitted:
            raise RuntimeError(
                "KeyValueTimeTokenizer is not fitted; call fit() on train sequences."
            )
        events = list(sequence.events)
        cap = self.config.max_events_per_user
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
        cap_tok = self.config.max_event_tokens
        for event in events:
            start = len(key_ids)
            self._encode_event(event, key_ids, value_ids, positions)
            if cap_tok is not None and len(key_ids) - start > cap_tok:
                del key_ids[start + cap_tok :]
                del value_ids[start + cap_tok :]
                del positions[start + cap_tok :]
            offsets.append(len(key_ids))
            hour, dow, dom = self._calendar(event.event_timestamp)
            hours.append(hour)
            dows.append(dow)
            doms.append(dom)
            ts_us.append(int(ensure_utc(event.event_timestamp).timestamp() * 1_000_000))
        if ts_us:
            last = ts_us[-1]
            tlog = tuple(time_encode(max(last - ts, 0) / 1e6) for ts in ts_us)
        else:
            tlog = ()

        prof_k, prof_v, prof_p, prof_off, prof_t = self.encode_profile(
            sequence.entity_id, sequence.cutoff_time
        )
        return EncodedKVTRecord(
            entity_id=sequence.entity_id,
            sequence_id=sequence.sequence_id or sequence.entity_id,
            key_ids=tuple(key_ids),
            value_ids=tuple(value_ids),
            positions=tuple(positions),
            event_offsets=tuple(offsets),
            event_ts_us=tuple(ts_us),
            time_log=tlog,
            hour=tuple(hours),
            dow=tuple(dows),
            dom=tuple(doms),
            prof_key_ids=tuple(prof_k),
            prof_value_ids=tuple(prof_v),
            prof_positions=tuple(prof_p),
            prof_offsets=tuple(prof_off),
            prof_time_log=tuple(prof_t),
        )

    def encode_profile(
        self, entity_id: str, cutoff: datetime
    ) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
        """`(keys, values, positions, item offsets, item times)` of the profile prefix."""
        prof_k: list[int] = []
        prof_v: list[int] = []
        prof_p: list[int] = []
        prof_off: list[int] = [0]
        prof_t: list[float] = []
        profile = self._profiles.get(entity_id)
        if profile is None:
            return prof_k, prof_v, prof_p, prof_off, prof_t
        for key, value in sorted(profile.attributes.items()):
            self._encode_field(key, value, prof_k, prof_v, prof_p)
            prof_off.append(len(prof_k))
            prof_t.append(0.0)
        # Milestones are dated; a snapshot's `as_of` can postdate the cutoff,
        # so each one is kept only if it happened by the sequence's cutoff.
        ref = int(cutoff.timestamp() * 1_000_000)
        for item in profile.lifelong:
            if item.timestamp > cutoff:
                continue
            self._encode_field(item.key, item.key, prof_k, prof_v, prof_p)
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

    def field_tokens(self, key: str, value: str) -> tuple[list[int], list[int], list[int]]:
        """`(keys, values, positions)` for one field, before the per-event cap."""
        key_ids: list[int] = []
        value_ids: list[int] = []
        positions: list[int] = []
        self._encode_field(key, value, key_ids, value_ids, positions)
        return key_ids, value_ids, positions

    def field_order(self, key: str) -> tuple[int, bool, str]:
        """Where a field sits in its event: `source`, single-token fields by key, then text."""
        return (0 if key == "source" else 1, self.field_kind.get(key) == "text", key)

    def state_dict(self) -> dict[str, Any]:
        """The fitted vocabulary and binning, JSON-serializable. Profiles are not included."""
        if not self._fitted:
            raise RuntimeError("KeyValueTimeTokenizer is not fitted; nothing to save.")
        config: dict[str, Any] = {
            **{k: getattr(self.config, k) for k in _CONFIG_FIELDS},
            "categorical_keys": list(self.config.categorical_keys),
        }
        # Absent unless used, so a state saved before hashing existed round-trips unchanged.
        extra: dict[str, Any] = {}
        if self.config.hashed_keys:
            config["hashed_keys"] = list(self.config.hashed_keys)
            config["hash_buckets"] = self.config.hash_buckets
            extra["hash_base"] = dict(self.hash_base)
        if self.config.excluded_keys:
            config["excluded_keys"] = list(self.config.excluded_keys)
        return {
            "config": config,
            **extra,
            "tokens": [self.id_to_token[i] for i in range(len(self.id_to_token))],
            "field_kind": dict(self.field_kind),
            "binner_edges": {k: b.edges.tolist() for k, b in self.binners.items()},
            "binner_n_buckets": {k: b.n_buckets for k, b in self.binners.items()},
            "binner_base": dict(self.binner_base),
            "cat_vocab": {k: dict(v) for k, v in self.cat_vocab.items()},
            "key_vocab": dict(self.key_vocab),
            "byte_offset": self.byte_offset,
        }

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any], profiles: Mapping[str, EntityProfile] | None = None
    ) -> KeyValueTimeTokenizer:
        """A fitted tokenizer from `state_dict()`."""
        raw = dict(state["config"])
        raw["categorical_keys"] = tuple(raw.get("categorical_keys", ()))
        raw["hashed_keys"] = tuple(raw.get("hashed_keys", ()))
        raw["excluded_keys"] = tuple(raw.get("excluded_keys", ()))
        tokenizer = cls(KeyValueTimeConfig(**raw), profiles)
        hash_base = state.get("hash_base", {})
        if not isinstance(hash_base, Mapping):
            raise ValueError(
                "This tokenizer state shares one hash-bucket range across keys; "
                "hashed keys now have a range each. Refit, or rebuild the token cache."
            )
        tokenizer.hash_base = {str(k): int(v) for k, v in hash_base.items()}
        tokenizer.token_to_id = {tok: i for i, tok in enumerate(state["tokens"])}
        tokenizer.id_to_token = dict(enumerate(state["tokens"]))
        tokenizer.field_kind = dict(state["field_kind"])
        tokenizer.binners = {
            k: PercentileBinner(
                int(state["binner_n_buckets"][k]), np.asarray(edges, dtype=np.float64)
            )
            for k, edges in state["binner_edges"].items()
        }
        tokenizer.binner_base = {k: int(v) for k, v in state["binner_base"].items()}
        tokenizer.cat_vocab = {k: dict(v) for k, v in state["cat_vocab"].items()}
        tokenizer.key_vocab = dict(state["key_vocab"])
        tokenizer.byte_offset = int(state["byte_offset"])
        tokenizer._fitted = True
        return tokenizer

    def collate(
        self,
        sequences: Sequence[TransactionSequence],
        *,
        max_length: int,
    ) -> KeyValueTimeBatch:
        """Pad records into a batch. Labels and raw features are attached by
        `tfm.data.collate.collate_batch`, not by the tokenizer."""
        records = [self.encode(seq) for seq in sequences]
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
        hour = torch.zeros((bsz, max_events), dtype=torch.long)
        dow = torch.zeros((bsz, max_events), dtype=torch.long)
        dom = torch.zeros((bsz, max_events), dtype=torch.long)
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
            hour[b, :n_events] = torch.tensor(rec.hour, dtype=torch.long)
            dow[b, :n_events] = torch.tensor(rec.dow, dtype=torch.long)
            dom[b, :n_events] = torch.tensor(rec.dom, dtype=torch.long)
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
            hour=hour,
            dow=dow,
            dom=dom,
            prof_key_ids=prof_key,
            prof_value_ids=prof_val,
            prof_positions=prof_pos,
            prof_mask=prof_mask,
            prof_time_log=prof_time,
            entity_ids=tuple(entity_ids),
            sequence_ids=tuple(sequence_ids),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "tokenizer": self.name,
            "layout": self.layout,
            "support_tier": self.support_tier,
            "vocab_size": self.vocab_size,
            "n_keys": len(self.key_vocab),
            "field_kind": dict(self.field_kind),
            "calendar_tz": self.config.calendar_tz,
            "max_events_per_user": self.config.max_events_per_user,
            "max_event_tokens": self.config.max_event_tokens,
            "categorical_keys": list(self.config.categorical_keys),
            "hashed_keys": list(self.config.hashed_keys),
            "hash_buckets": self.config.hash_buckets if self.config.hashed_keys else None,
            "fitted": self._fitted,
            "text_encoding": "byte_level",
        }
