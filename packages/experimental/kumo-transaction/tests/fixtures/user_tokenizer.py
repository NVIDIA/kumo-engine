# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A tokenizer defined outside `tfm`, selected by import path: one token per field key."""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, ClassVar, Self

import numpy as np
import torch

from tfm.config import DataConfig, FeaturesConfig, RecipeConfig
from tfm.contracts.schema import EntityProfile
from tfm.contracts.tokenizers import EncodedBlock, EventBlock, SequenceTokenizer
from tfm.data.token_cache import CachedTokenizer, CacheRows, TokenCache
from tfm.data.token_cache.columnar import ColumnarCacheFormat

PAD, UNK = 0, 1


class KeyTokenizer:
    """Keys seen at least `min_count` times on the train side get ids from 2, in name order."""

    uses_profiles = False

    def __init__(self, min_count: int = 1) -> None:
        self.min_count = min_count
        self.vocab: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 2

    def settings(self) -> dict[str, Any]:
        return {"state_version": 1, "min_count": self.min_count}

    def fit_block(self, block: EventBlock, profiles: Mapping[str, EntityProfile]) -> Counter[str]:
        counts = np.bincount(block.pair_key, minlength=len(block.keys))
        return Counter({k: int(n) for k, n in zip(block.keys, counts, strict=True) if n})

    def merge(self, a: Counter[str], b: Counter[str]) -> Counter[str]:
        return a + b

    def fit_stats(self, stats: Counter[str]) -> Self:
        kept = sorted(k for k, n in stats.items() if n >= self.min_count)
        self.vocab = {k: i + 2 for i, k in enumerate(kept)}
        return self

    def encode_block(self, block: EventBlock) -> EncodedBlock:
        """Each event's keys in id order; `n_keys` is a per-event array of its own."""
        lut = np.array([self.vocab.get(k, UNK) for k in block.keys], np.int32)
        ids = lut[block.pair_key]
        order = np.lexsort((ids, block.pair_event))
        per_event = np.bincount(block.pair_event, minlength=block.n_events)
        offsets = np.concatenate([[0], np.cumsum(per_event)]).astype(np.int64)
        positions = np.arange(len(ids)) - np.repeat(offsets[:-1], per_event)
        return EncodedBlock(
            token_offsets=offsets,
            key_ids=ids[order],
            value_ids=np.zeros(len(ids), np.int32),
            positions=positions.astype(np.int16),
            events={"n_keys": per_event.astype(np.int16)},
            truncated_events=0,
        )

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "min_count": self.min_count, "vocab": self.vocab}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        if state["version"] != 1:
            raise ValueError(f"KeyTokenizer reads state version 1, not {state['version']}.")
        tokenizer = cls(state["min_count"])
        tokenizer.vocab = dict(state["vocab"])
        return tokenizer


class CalendarKeyTokenizer(KeyTokenizer):
    """`KeyTokenizer` that also stores each event's `hour`, `dow`, `dom` and `minute`."""

    def encode_block(self, block: EventBlock) -> EncodedBlock:
        encoded = super().encode_block(block)
        days = np.floor_divide(block.ts_us, 86_400_000_000)
        day = days.astype("datetime64[D]")
        dom = (day - day.astype("datetime64[M]").astype("datetime64[D]")).astype(np.int64) + 1
        return dataclasses.replace(
            encoded,
            events={
                **encoded.events,
                "hour": (np.floor_divide(block.ts_us, 3_600_000_000) % 24).astype(np.uint8),
                "dow": ((days + 3) % 7).astype(np.uint8),
                "dom": dom.astype(np.uint8),
                # uint16, which torch cannot index on a device: `DeviceBatches` widens it.
                "minute": (np.floor_divide(block.ts_us, 60_000_000) % 60).astype(np.uint16),
            },
        )


class ConfiguredKeyTokenizer(KeyTokenizer):
    """`KeyTokenizer` built from the recipe: a key needs as many occurrences as there
    are excluded keys, plus one."""

    feature_keys: ClassVar[tuple[str, ...]] = ("exclude_keys",)

    @classmethod
    def from_config(cls, features: FeaturesConfig, data: DataConfig) -> Self:
        del data
        return cls(min_count=len(features.exclude_keys) + 1)


def placement_batches(
    cache: TokenCache,
    rows: CacheRows,
    tokenizer: CachedTokenizer,
    *,
    batch_size: int,
    max_length: int,
    shuffle_seed: int | None,
    device: torch.device,
    event_arrays: Sequence[str],
) -> list[torch.device]:
    """A batch builder that returns only the placement it was given."""
    del cache, rows, tokenizer, batch_size, max_length, shuffle_seed, event_arrays
    return [device]


class RecordingFormat(ColumnarCacheFormat):
    """The columnar format, recording which of its steps a run called."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def config_problems(self, config: RecipeConfig) -> list[str]:
        self.calls.append("config_problems")
        return super().config_problems(config)

    def build(self, config: RecipeConfig, tokenizer: SequenceTokenizer, **kwargs: Any) -> Any:
        self.calls.append("build")
        return super().build(config, tokenizer, **kwargs)

    def open(self, config: RecipeConfig, tokenizer: SequenceTokenizer) -> TokenCache:
        self.calls.append("open")
        return super().open(config, tokenizer)

    def run_tokenizer(
        self, cache: TokenCache, tokenizer: SequenceTokenizer, config: RecipeConfig, **kwargs: Any
    ) -> CachedTokenizer:
        self.calls.append("run_tokenizer")
        return super().run_tokenizer(cache, tokenizer, config, **kwargs)

    def pretrain_rows(
        self, cache: TokenCache, config: RecipeConfig, *, train_cutoff: datetime
    ) -> CacheRows:
        self.calls.append("pretrain_rows")
        return super().pretrain_rows(cache, config, train_cutoff=train_cutoff)

    def check_event_arrays(self, cache: TokenCache, names: Sequence[str]) -> None:
        self.calls.append("check_event_arrays")
        super().check_event_arrays(cache, names)


recording_format = RecordingFormat()


class FormattedKeyTokenizer(KeyTokenizer):
    """`KeyTokenizer` whose cache uses `recording_format`, named by import path."""

    token_cache_format: ClassVar[str] = "tests.fixtures.user_tokenizer:recording_format"
