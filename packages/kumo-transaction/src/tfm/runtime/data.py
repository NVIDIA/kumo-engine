# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data stage shared by every stack (slot 0, then slot 1's tokenizer).

One path serves every tokenizer and history model: load the book, split it
in time, build one sequence per entity at each cutoff, fit the tokenizer on
the train side only, and collate with explicit label windows (ADR 0013).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from tfm.config import RecipeConfig
from tfm.contracts.errors import (
    IncompatibleCompositionError,
    SchemaError,
    UnsupportedCombinationError,
)
from tfm.contracts.schema import EntityProfile, TransactionEvent, TransactionSequence
from tfm.contracts.tokenizers import SequenceTokenizer
from tfm.data.adapters.registry import load_adapter_events
from tfm.data.collate import collate_batch
from tfm.data.labels.windows import LabelWindow, side_label_windows
from tfm.data.sequence_builders.entity_sequences import (
    build_entity_sequences,
    build_entity_windows,
)
from tfm.data.splits.entity import EntitySplit
from tfm.data.splits.temporal import SplitAssignment, TemporalSplitSpec, assign_temporal_split
from tfm.data.tasks.identity_theft import IdentityTheftSplice
from tfm.data.token_cache import CachedKeyValueTimeTokenizer, CacheRows, TokenCache
from tfm.data.token_cache.flat import FlatRows, FlatTokenCache
from tfm.data.validation.schema import validate_events
from tfm.features.tokenizers import KeyValueTimeConfig, KeyValueTimeTokenizer
from tfm.features.tokenizers.financial import TransactionFieldTokenizer
from tfm.features.tokenizers.flat_key_value import FlatKeyValueTokenizer
from tfm.features.tokenizers.registry import advertised_tokenizers, build_tokenizer
from tfm.runtime.common import parse_duration, parse_timestamp
from tfm.runtime.token_cache import open_recipe_cache
from tfm.training.reproducibility import seed_everything


@dataclass(frozen=True)
class Cutoffs:
    """The three cutoffs and the label window each side's labels come from."""

    train: datetime
    validation: datetime
    test: datetime
    windows: Mapping[str, LabelWindow]

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_cutoff": self.train.isoformat(),
            "validation_cutoff": self.validation.isoformat(),
            "test_cutoff": self.test.isoformat(),
            "label_windows": {side: w.as_dict() for side, w in self.windows.items()},
        }


@dataclass(frozen=True)
class FraudProbeData:
    """The fraud probe's own rows when its labels are spliced (ADR 0015).

    `train_batch` holds the split's train entities at the train cutoff and
    `eval_batch` its evaluation entities at the test cutoff. `events` is the
    book with victims' histories spliced; nothing else reads it.
    """

    split: EntitySplit
    train_batch: Any
    eval_batch: Any
    events: tuple[TransactionEvent, ...]
    manifest: Mapping[str, Any]
    train_sequences: tuple[TransactionSequence, ...] = ()
    eval_sequences: tuple[TransactionSequence, ...] = ()


@dataclass
class PreparedData:
    """Everything the lifecycle needs from the data stage.

    `batches`, `events`, and the pretraining corpus are unspliced. When
    `fraud` is set, the fraud probe reads it instead of `batches`.
    """

    spec: TemporalSplitSpec
    splits: SplitAssignment
    cutoffs: Cutoffs
    tokenizer: SequenceTokenizer
    batches: tuple[Any, Any, Any]
    train_sequences: tuple[TransactionSequence, ...]
    test_sequences: tuple[TransactionSequence, ...]
    pretrain_sequences: tuple[TransactionSequence, ...]
    events: tuple[TransactionEvent, ...]
    label_rows: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    profiles: tuple[EntityProfile, ...] = ()
    adapter_metadata: dict[str, Any] = field(default_factory=dict)
    fraud: FraudProbeData | None = None
    token_cache: TokenCache | FlatTokenCache | None = None
    pretrain_rows: CacheRows | FlatRows | None = None
    """With `token_cache`, the pretraining corpus read from disk: every cached user
    (key-value-time) or the sample's train-split windows (flat fields).
    `pretrain_sequences` is then empty."""


@dataclass
class PretrainingData:
    """What pretraining needs, read from a flat-field token cache alone.

    `prepare_pretraining` builds no event, sequence, or probe: a run that only
    trains reads its windows from disk. Its windows equal `prepare_data`'s
    with the same cache (`tests/contract/test_flat_token_cache.py`).
    """

    tokenizer: SequenceTokenizer
    token_cache: FlatTokenCache
    pretrain_rows: FlatRows
    train_cutoff: datetime
    observation_window: str


def tokenizer_for(config: RecipeConfig) -> SequenceTokenizer:
    """The tokenizer `features.tokenizer` names, sized from `data`."""
    name = config.features.tokenizer
    cls = advertised_tokenizers().get(name)
    categorical = tuple(config.features.categorical_keys)
    hashed = tuple(config.features.hashed_keys)
    excluded = tuple(config.features.exclude_keys)
    for option, value in (("hashed_keys", hashed), ("exclude_keys", excluded)):
        if value and cls is not KeyValueTimeTokenizer:
            raise UnsupportedCombinationError(
                f"features.{option} applies to the key_value_time tokenizer, not {name!r}."
            )
    if cls is not None and issubclass(cls, FlatKeyValueTokenizer):
        # `max_tokens` is the row width here, not an event's; keep the per-event default.
        return cls(
            KeyValueTimeConfig(
                max_events_per_user=config.data.sequence_length, categorical_keys=categorical
            )
        )
    if cls is not None and issubclass(cls, KeyValueTimeTokenizer):
        return cls(
            KeyValueTimeConfig(
                max_events_per_user=config.data.sequence_length,
                max_event_tokens=config.data.max_tokens,
                categorical_keys=categorical,
                hashed_keys=hashed,
                hash_buckets=config.features.hash_buckets,
                excluded_keys=excluded,
            )
        )
    if categorical:
        raise UnsupportedCombinationError(
            f"features.categorical_keys applies to the key-value-time tokenizers; "
            f"{name!r} chooses its own fields."
        )
    fields = config.features.fields
    if fields is None:
        return build_tokenizer(name)
    if cls is None or not issubclass(cls, TransactionFieldTokenizer):
        raise UnsupportedCombinationError(
            f"features.fields applies to transaction_field_tokenizer, not {name!r}."
        )
    return build_tokenizer(name, fields=tuple(fields))


def _cutoffs(
    config: RecipeConfig, splits: SplitAssignment, spec: TemporalSplitSpec, precomputed: bool
) -> Cutoffs:
    if precomputed:
        train = max(e.event_timestamp for e in splits.train)
        validation = max(e.event_timestamp for e in (*splits.train, *splits.validation))
        test = max(e.event_timestamp for e in (*splits.train, *splits.validation, *splits.test))
    else:
        train, validation = spec.train_end, spec.validation_end
        test = max(e.event_timestamp for e in splits.test)
    window = parse_duration(config.data.observation_window)
    return Cutoffs(
        train=train,
        validation=validation,
        test=test,
        windows=side_label_windows(train, validation, test, window),
    )


def _spliced_fraud_probe(
    config: RecipeConfig,
    splice: IdentityTheftSplice,
    cutoffs: Cutoffs,
    spec: TemporalSplitSpec,
    tokenizer: SequenceTokenizer,
    events: Sequence[TransactionEvent],
    train_seq: Sequence[TransactionSequence],
    test_seq: Sequence[TransactionSequence],
) -> FraudProbeData:
    for name in ("train", "test"):
        if splice.windows[name].as_dict() != cutoffs.windows[name].as_dict():
            raise SchemaError(
                f"The identity-theft splice used the {name} label window "
                f"{splice.windows[name].as_dict()}, but the run's is "
                f"{cutoffs.windows[name].as_dict()}."
            )
    window = parse_duration(config.data.observation_window)

    def rows_for(
        entities: Sequence[str],
        unspliced: Sequence[TransactionSequence],
        cutoff: datetime,
        before: datetime | None,
    ) -> list[TransactionSequence]:
        by_entity = {seq.entity_id: seq for seq in unspliced}
        victims = [e for e in entities if e in splice.victim_events]
        victim_events = [
            event
            for victim in victims
            for event in splice.victim_events[victim]
            if before is None or event.event_timestamp < before
        ]
        rebuilt = {
            seq.entity_id: seq
            for seq in build_entity_sequences(
                victim_events,
                cutoff,
                max_events=config.data.sequence_length,
                observation_window=window,
            )
        }
        chosen = [
            rebuilt.get(e) if e in splice.victim_events else by_entity.get(e) for e in entities
        ]
        return [seq for seq in chosen if seq is not None]

    # Train-side histories mirror the temporal split: events strictly before train_end.
    train_rows = rows_for(splice.split.train, train_seq, cutoffs.train, spec.train_end)
    eval_rows = rows_for(splice.split.evaluation, test_seq, cutoffs.test, None)
    max_tokens = config.data.max_tokens
    return FraudProbeData(
        split=splice.split,
        train_batch=collate_batch(
            train_rows, tokenizer, max_tokens=max_tokens, label_window=cutoffs.windows["train"]
        ),
        eval_batch=collate_batch(
            eval_rows, tokenizer, max_tokens=max_tokens, label_window=cutoffs.windows["test"]
        ),
        events=tuple(splice.apply(events)),
        manifest=splice.manifest(),
        train_sequences=tuple(train_rows),
        eval_sequences=tuple(eval_rows),
    )


def prepare_data(
    config: RecipeConfig,
    fixture_dir: Path | None = None,
    *,
    tokenizer: SequenceTokenizer | None = None,
) -> PreparedData:
    """Load, split, tokenize, and collate the configured book.

    `fixture_dir` is where the pragmatiq adapter writes a synthetic book when no
    local root is configured. `tokenizer`, already fitted (for example the one
    saved with a pretrained checkpoint), is used as is instead of fitting one
    on this sample's train side; it cannot be combined with `data.token_cache`.
    """
    seed_everything(config.seed)
    fitted = tokenizer is not None
    if fitted and config.data.token_cache is not None:
        raise IncompatibleCompositionError(
            "prepare_data was given a fitted tokenizer and data.token_cache; the cache "
            "carries its own tokenizer. Pass one or the other."
        )
    tokenizer = tokenizer if tokenizer is not None else tokenizer_for(config)
    cache = None if config.data.token_cache is None else open_recipe_cache(config, tokenizer)
    loaded = load_adapter_events(
        config,
        flat_fields=bool(getattr(tokenizer, "requires_flat_fields", False)),
        fixture_dir=fixture_dir,
    )
    events = validate_events(list(loaded.events))
    spec = TemporalSplitSpec(
        train_end=parse_timestamp(config.data.temporal_split.train_end),
        validation_end=parse_timestamp(config.data.temporal_split.validation_end),
        label_delay=parse_duration(config.data.temporal_split.label_delay),
        observation_window=parse_duration(config.data.observation_window),
    )
    splits = loaded.splits if loaded.splits is not None else assign_temporal_split(events, spec)
    precomputed = loaded.splits is not None and config.data.split_protocol == "precomputed"
    cutoffs = _cutoffs(config, splits, spec, precomputed)
    if isinstance(cache, TokenCache):
        tokenizer = _cached_tokenizer(cache, events, loaded.fraud_splice)
    elif getattr(tokenizer, "uses_profiles", False):
        profiles = {profile.entity_id: profile for profile in loaded.profiles}
        cast(KeyValueTimeTokenizer, tokenizer).set_profiles(profiles)

    window = parse_duration(config.data.observation_window)

    def sequences_at(side_events: list[TransactionEvent], cutoff: datetime) -> list[Any]:
        return build_entity_sequences(
            side_events,
            cutoff,
            max_events=config.data.sequence_length,
            observation_window=window,
        )

    train_seq = sequences_at(list(splits.train), cutoffs.train)
    val_seq = sequences_at([*splits.train, *splits.validation], cutoffs.validation)
    test_seq = sequences_at([*splits.train, *splits.validation, *splits.test], cutoffs.test)
    if not isinstance(cache, TokenCache) and not fitted:
        tokenizer.fit(train_seq)
    max_tokens = config.data.max_tokens
    batches = (
        collate_batch(
            train_seq, tokenizer, max_tokens=max_tokens, label_window=cutoffs.windows["train"]
        ),
        collate_batch(
            val_seq, tokenizer, max_tokens=max_tokens, label_window=cutoffs.windows["validation"]
        ),
        collate_batch(
            test_seq, tokenizer, max_tokens=max_tokens, label_window=cutoffs.windows["test"]
        ),
    )
    window_events = config.data.pretrain_windows.events if config.data.pretrain_windows else None
    pretrain_rows: CacheRows | FlatRows | None = None
    if isinstance(cache, FlatTokenCache):
        pretrain_seq = []
        pretrain_rows = _flat_windows(config, cache, cutoffs.train)
    elif cache is not None:
        pretrain_seq = []
        if window_events is None:
            pretrain_rows = cache.train_rows(
                train_end=cutoffs.train, lookback=window, max_events=config.data.sequence_length
            )
        else:
            pretrain_rows = cache.train_windows(
                train_end=cutoffs.train,
                lookback=window,
                window_events=window_events,
            )
    elif window_events is None:
        pretrain_seq = train_seq
    else:
        pretrain_seq = build_entity_windows(
            splits.train,
            cutoffs.train,
            window_events=window_events,
            observation_window=window,
        )
    fraud = (
        None
        if loaded.fraud_splice is None
        else _spliced_fraud_probe(
            config, loaded.fraud_splice, cutoffs, spec, tokenizer, events, train_seq, test_seq
        )
    )
    return PreparedData(
        spec=spec,
        splits=splits,
        cutoffs=cutoffs,
        tokenizer=tokenizer,
        batches=batches,
        train_sequences=tuple(train_seq),
        test_sequences=tuple(test_seq),
        pretrain_sequences=tuple(pretrain_seq),
        events=tuple(events),
        label_rows=dict(loaded.labels),
        profiles=tuple(loaded.profiles),
        adapter_metadata=dict(loaded.metadata),
        fraud=fraud,
        token_cache=cache,
        pretrain_rows=pretrain_rows,
    )


def _flat_windows(config: RecipeConfig, cache: FlatTokenCache, cutoff: datetime) -> FlatRows:
    windows = config.data.pretrain_windows
    if windows is None:
        raise UnsupportedCombinationError(
            "A flat-field token cache serves pretraining windows; set data.pretrain_windows."
        )
    cached_cutoff = cache.train_cutoff()
    if cached_cutoff != cutoff:
        raise SchemaError(
            f"Token cache {cache.path} ends at {cached_cutoff.isoformat()}, but this run's "
            f"train cutoff is {cutoff.isoformat()}. Rebuild it with `tfm cache-tokens`."
        )
    return cache.train_windows(
        cutoff=cutoff,
        lookback=parse_duration(config.data.observation_window),
        window_events=windows.events,
    )


def prepare_pretraining(config: RecipeConfig) -> PretrainingData:
    """The pretraining windows of a flat-field token cache, without loading the book.

    The precomputed protocol's train cutoff is the last train event of the
    sample, which is the cache's last event; nothing else is read.
    """
    seed_everything(config.seed)
    if config.data.token_cache is None:
        raise UnsupportedCombinationError(
            "prepare_pretraining reads a token cache; set data.token_cache."
        )
    tokenizer = tokenizer_for(config)
    cache = open_recipe_cache(config, tokenizer)
    if not isinstance(cache, FlatTokenCache):
        raise UnsupportedCombinationError(
            "prepare_pretraining reads a flat-field token cache; a key-value-time cache "
            "needs prepare_data, which loads profiles and probes."
        )
    cutoff = cache.train_cutoff()
    return PretrainingData(
        tokenizer=tokenizer,
        token_cache=cache,
        pretrain_rows=_flat_windows(config, cache, cutoff),
        train_cutoff=cutoff,
        observation_window=config.data.observation_window,
    )


def _cached_tokenizer(
    cache: TokenCache,
    events: Sequence[TransactionEvent],
    splice: IdentityTheftSplice | None,
) -> CachedKeyValueTimeTokenizer:
    """The cache's tokenizer, with every cached user's profile and the splice's copies."""
    outside = {e.entity_id for e in events} - set(cache.user_index)
    if outside:
        raise SchemaError(
            f"{len(outside)} sampled users are not in token cache {cache.path} "
            f"(first: {sorted(outside)[0]!r}); it must hold every probe user."
        )
    tokenizer = cache.tokenizer()
    if splice is not None:
        tokenizer.add_aliases(splice.copied_from)
    return tokenizer
