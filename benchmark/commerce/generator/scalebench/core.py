# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared generation primitives: keyed RNG, time model, column builders."""
import hashlib
from datetime import datetime
from functools import lru_cache

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from scalebench import profile as P

DAY = 86_400
HOUR = 3_600
WEEK = 7 * DAY
MONDAY = 4 * DAY  # 1970-01-05 00:00 UTC, first Monday after the epoch
DELIMITER = "|"
HIGH_CARD_SCOPE = 0.05  # cardinality/rows above this -> per-shard vocabulary


def rng(*key):
    """Counter-based stream keyed by (seed, graph, table, shard, block, ...)."""
    key = tuple(int(k) if isinstance(k, (int, np.integer)) else str(k) for k in key)
    h = hashlib.blake2b(repr(key).encode(), digest_size=16).digest()
    return np.random.Generator(np.random.Philox(key=int.from_bytes(h, "little")))


class Timeline:
    """Two-year window with diurnal/weekly seasonality and bursty sessions."""

    def __init__(self, time):
        start = datetime.fromisoformat(time["window_start"].replace("Z", "+00:00"))
        self.start = int(start.timestamp())
        self.end = self.start + int(time["window_days"]) * DAY
        if not 0 <= self.start < self.end < 9_223_372_036:  # Kumo parses timestamps as pandas ns
            raise ValueError("time window must lie within 1970..2262")
        self.last = self.end - 1  # every timestamp is in [start, end)
        h = np.asarray(time["hour_of_day"], float)
        d = np.asarray(time["day_of_week"], float)
        self.hour_cdf = np.cumsum(h) / h.sum()
        self.dow_cdf = np.cumsum(d) / d.sum()
        self.spec = time

    def created(self, r, n, spec, latest):
        """Creation times: a base cohort in the first week, then exponential growth."""
        g = max(float(spec["growth_per_year"]), 1e-6) / (365 * DAY)
        span = max(latest - self.start - WEEK, DAY)
        u = r.random(n)
        grown = self.start + WEEK + np.log1p(u * np.expm1(g * span)) / g
        base = self.start + r.random(n) * WEEK
        t = np.where(r.random(n) < float(spec["initial_frac"]), base, grown)
        return np.minimum(t.astype(np.int64), latest)

    def seasonal(self, r, lo, hi):
        """One time per row in [lo, hi], following the hour-of-day and day-of-week profile."""
        n = len(lo)
        base = lo + (r.random(n) * (hi - lo)).astype(np.int64)
        week = (base - MONDAY) // WEEK * WEEK + MONDAY
        dow = np.searchsorted(self.dow_cdf, r.random(n), side="right")
        hour = np.searchsorted(self.hour_cdf, r.random(n), side="right")
        t = week + dow * DAY + hour * HOUR + r.integers(0, HOUR, n)
        bad = (t < lo) | (t > hi)
        t[bad] = base[bad]
        return t

    def bursty(self, r, owner, lo, hi, events_spec):
        """Times for events grouped contiguously by owner.

        Each event either starts a new session (a seasonal draw) or follows the
        previous event of the same owner after a short gap.
        """
        n = len(owner)
        t = self.seasonal(r, lo, hi)
        if n == 0:
            return t
        gap = P.numeric_sampler(events_spec["burst_gap_seconds"])(r.random(n)).astype(np.int64)
        cont = r.random(n) < float(events_spec["p_burst"])
        cont[0] = False
        cont[1:] &= owner[1:] == owner[:-1]
        idx = np.arange(n)
        anchor = np.maximum.accumulate(np.where(cont, 0, idx))
        cg = np.cumsum(np.where(cont, gap, 0))
        t = t[anchor] + cg - cg[anchor]
        over = t > hi  # a session running past the window end restarts inside it (no pile-up at hi)
        t[over] = self.seasonal(r, lo[over], hi[over])
        return t


VOCAB_CACHE_MAX = 2_000_000


def _format(codes, prefix):
    s = pc.utf8_lpad(pc.cast(pa.array(codes, pa.int64()), pa.string()), 6, "0")
    return pc.binary_join_element_wise(prefix + "_", s, "")


@lru_cache(maxsize=24)
def _vocab(prefix, lo, hi):
    return _format(np.arange(lo, hi, dtype=np.int64), prefix)


def tokens(codes, prefix, lo=0, hi=None):
    """Generic token strings '<prefix>_<code:06d>' for codes in [lo, hi)."""
    if hi is not None and hi - lo <= VOCAB_CACHE_MAX:
        return _vocab(prefix, lo, hi).take(pa.array(codes - lo))
    return _format(codes, prefix)


def with_nulls(arr, null_mask):
    if null_mask is None or not null_mask.any():
        return arr
    return pc.if_else(pa.array(null_mask), pa.nulls(len(arr), arr.type), arr)


class Columns:
    """Builds one table's columns from its profile entry.

    shard_scale = shard main-entity rows / real main-entity rows. High-cardinality
    columns (distinct/rows > HIGH_CARD_SCOPE) get a per-shard vocabulary of
    cardinality * shard_scale tokens, so distinct-per-row stays constant with scale.
    """

    def __init__(self, table_profile, shard, shard_scale, nullable, bounds, ints):
        self.cols = table_profile["columns"]
        self.ints = ints  # declared int64 columns (the profile's source dtype may differ)
        self.nullable = nullable
        self.bounds = bounds
        self.rows = table_profile["rows"]
        self.shard = shard
        self.scale = shard_scale

    def vocab(self, spec_card, capped=False):
        card = int(spec_card)
        if not capped and card / max(self.rows, 1) > HIGH_CARD_SCOPE:
            per_shard = max(1, int(round(card * self.scale)))
            return per_shard, self.shard * per_shard
        return card, 0

    def null_mask(self, r, name, n):
        rate = float(self.cols[name].get("null_rate", 0.0)) if name in self.nullable else 0.0
        return r.random(n) < rate if rate > 0 else None

    def categorical(self, r, name, n, codes=None, null=True):
        spec = self.cols[name]["categorical"]
        card, offset = self.vocab(spec["cardinality"], spec.get("capped", False))
        if codes is None:
            codes = P.zipf_rank(r.random(n), card, float(spec.get("zipf_s") or 1.0))
        codes = np.minimum(codes, card - 1)
        mask = self.null_mask(r, name, n) if null else None
        return with_nulls(tokens(codes + offset, name, offset, offset + card), mask)

    def cat_codes(self, r, name, n):
        spec = self.cols[name]["categorical"]
        return P.zipf_rank(r.random(n), self.cardinality(name), float(spec.get("zipf_s") or 1.0))

    def cardinality(self, name):
        spec = self.cols[name]["categorical"]
        return self.vocab(spec["cardinality"], spec.get("capped", False))[0]

    def numeric(self, r, name, n, u=None):
        col = self.cols[name]
        x = P.numeric_sampler(col["numeric"])(r.random(n) if u is None else u)
        lo, hi = self.bounds.get(name, (None, None))
        x = np.clip(x, -np.inf if lo is None else lo, np.inf if hi is None else hi)
        if name in self.ints:
            arr = pa.array(np.maximum(np.rint(x), 0).astype(np.int64))
        else:
            arr = pa.array(np.round(x, 2))
        return with_nulls(arr, self.null_mask(r, name, n))

    def boolean(self, r, name, n, rate=None):
        p = float(self.cols[name]["boolean"]["true_rate"]) if rate is None else rate
        return pa.array(r.random(n) < p)

    def max_len(self, name):
        q = self.cols[name]["multicat"]["len_quantiles"]
        return max(1, int(round(max(float(v) for v in q.values()))))

    def multicat(self, r, name, n, override=None, extra_len=None):
        """Delimiter-joined distinct tokens; 1..max_len tokens per non-null row.

        override(rows, codes) may rewrite token codes (e.g. cluster-revealing tags).
        """
        spec = self.cols[name]["multicat"]
        mask = self.null_mask(r, name, n)
        max_len = self.max_len(name)
        lengths = P.stochastic_round(P.length_sampler(spec["len_quantiles"])(r.random(n)), r.random(n))
        if extra_len is not None:
            lengths = lengths + extra_len
        lengths = np.clip(lengths, 1, max_len)
        if mask is not None:
            lengths[mask] = 0
        row = np.repeat(np.arange(n), lengths)
        card, offset = self.vocab(spec["token_cardinality"], spec.get("capped", False))
        codes = P.zipf_rank(r.random(len(row)), card, float(spec.get("token_zipf_s") or 1.0))
        if override is not None:
            codes = override(row, codes)
        key = np.unique(row.astype(np.int64) * card + codes)  # sorted and de-duplicated per row
        row, codes = key // card, key % card
        offsets = np.zeros(n + 1, np.int32)
        np.cumsum(np.bincount(row, minlength=n), out=offsets[1:])
        lists = pa.ListArray.from_arrays(pa.array(offsets), tokens(codes + offset, name, offset, offset + card),
                                         mask=None if mask is None else pa.array(mask))
        return pc.binary_join(lists, DELIMITER)


def ts(seconds):
    return pa.array(np.asarray(seconds, np.int64) * 1000, pa.timestamp("ms"))
