# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name registry for event adapters."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from tfm.config import DataConfig, RecipeConfig
from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.schema import TransactionEvent, ensure_utc
from tfm.data.adapters.paths import ADAPTER_ENV, entity_cap, resolve_data_path, sample_seed
from tfm.data.adapters.result import AdapterResult
from tfm.data.labels.windows import side_label_windows
from tfm.data.splits.temporal import parse_duration, parse_timestamp
from tfm.data.synthetic.generator import generate_synthetic_events
from tfm.data.tasks.identity_theft import (
    IdentityTheftSpec,
    IdentityTheftSplice,
    splice_identity_theft,
)

if TYPE_CHECKING:
    from tfm.data.adapters.pragmatiq import PragmatiqDataset

_ADAPTERS: dict[str, Callable[..., AdapterResult]] = {}


def register_adapter(
    name: str,
) -> Callable[[Callable[..., AdapterResult]], Callable[..., AdapterResult]]:
    def decorator(fn: Callable[..., AdapterResult]) -> Callable[..., AdapterResult]:
        _ADAPTERS[name] = fn
        return fn

    return decorator


def advertised_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


@register_adapter("synthetic")
def load_synthetic_adapter(
    config: DataConfig,
    *,
    seed: int,
    flat_fields: bool = True,
    fixture_dir: Path | None = None,
) -> AdapterResult:
    """Generated card transactions; every event already has the 12 flat fields."""
    del flat_fields, fixture_dir
    delay_text = config.temporal_split.label_delay.strip().lower()
    if delay_text.endswith("d"):
        delay = timedelta(days=int(delay_text[:-1]))
    elif delay_text.endswith("h"):
        delay = timedelta(hours=int(delay_text[:-1]))
    else:
        delay = timedelta(days=7)
    events = generate_synthetic_events(
        seed=seed,
        n_entities=config.n_entities,
        events_per_entity=config.events_per_entity,
        label_delay=delay,
    )
    return AdapterResult(events=tuple(events), metadata={"adapter": "synthetic"})


@register_adapter("tabformer")
def load_tabformer_registered(
    config: DataConfig,
    *,
    seed: int,
    flat_fields: bool = True,
    fixture_dir: Path | None = None,
) -> AdapterResult:
    """A local TabFormer book; card transactions only, so `flat_fields` changes nothing."""
    del flat_fields, fixture_dir
    from tfm.data.adapters.tabformer import load_tabformer_adapter

    return load_tabformer_adapter(resolve_data_path(config), config, seed=seed)


@register_adapter("pragmatiq")
def load_pragmatiq_registered(
    config: DataConfig,
    *,
    seed: int,
    flat_fields: bool = True,
    fixture_dir: Path | None = None,
) -> AdapterResult:
    """A local pragmatiq book, or a generated fixture when no root is configured.

    `flat_fields` keeps transaction events only, aliased to the 12-field
    schema. Key-value tokenizers keep every source and the profiles.
    """
    from tfm.data.adapters.pragmatiq import (
        events_for_flat_fields,
        generate_pragmatiq_fixture,
        load_pragmatiq_dataset,
    )

    spliced = config.fraud_labels == "spliced_identity_theft"
    if config.path or os.environ.get(ADAPTER_ENV["pragmatiq"]):
        root = resolve_data_path(config)
        dataset = load_pragmatiq_dataset(
            root,
            max_entities=entity_cap(config),
            sample_seed=sample_seed(config, seed),
            attach_fraud_labels=not spliced,
            stratify_by_fraud=not spliced,
        )
        source = "path"
    else:
        if spliced:
            raise UnsupportedCombinationError(
                "data.fraud_labels `spliced_identity_theft` needs a pragmatiq book with "
                "card transactions and profiles (data.path or TFM_PRAGMATIQ_ROOT); the "
                "generated fixture is too small to splice."
            )
        if fixture_dir is None:
            raise UnsupportedCombinationError(
                "The pragmatiq adapter needs data.path, TFM_PRAGMATIQ_ROOT, or a fixture "
                "directory to generate a synthetic book into."
            )
        root = generate_pragmatiq_fixture(
            fixture_dir,
            n_users=max(config.n_entities, 8),
            events_per_user=max(config.events_per_entity, 6),
            seed=seed,
        )
        dataset = load_pragmatiq_dataset(root)
        source = "fixture"
    splice = _identity_theft_splice(config, dataset, seed=seed) if spliced else None
    events = events_for_flat_fields(dataset.events) if flat_fields else list(dataset.events)
    if splice is not None and flat_fields:
        # Aliasing derives fields from `entity_id`, so it runs after the copies are re-owned.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aliased = events_for_flat_fields(
                [e for history in splice.victim_events.values() for e in history]
            )
        by_victim: dict[str, list[TransactionEvent]] = {v: [] for v in splice.victim_events}
        for event in aliased:
            by_victim[event.entity_id].append(event)
        splice = splice.with_victim_events(by_victim)
    return AdapterResult(
        events=tuple(events),
        metadata={
            "adapter": "pragmatiq",
            "root": str(root),
            "source": source,
            "n_profiles": len(dataset.profiles),
            "dropped_sources": dict(dataset.dropped_sources),
            "flat_fields": flat_fields,
            "n_events": len(events),
            "fraud_labels": config.fraud_labels,
        },
        profiles=tuple(dataset.profiles),
        labels={task: rows for task, rows in dataset.labels.items() if task != "fraud"},
        fraud_splice=splice,
    )


def _identity_theft_splice(
    config: DataConfig, dataset: PragmatiqDataset, *, seed: int
) -> IdentityTheftSplice:
    """ADR 0015's task over the sampled users, on the label windows `prepare_data` will use."""
    from tfm.data.adapters.pragmatiq import fraud_positive_users

    if not dataset.events:
        raise SchemaError("Cannot splice identity theft into a book with no events.")
    train_end = parse_timestamp(config.temporal_split.train_end)
    validation_end = parse_timestamp(config.temporal_split.validation_end)
    last = max(ensure_utc(e.event_timestamp) for e in dataset.events)
    if last <= validation_end:
        raise SchemaError(
            f"The book ends at {last.isoformat()}, before validation_end "
            f"{validation_end.isoformat()}, so there is no test window to splice into."
        )
    cfg = config.identity_theft
    spec = IdentityTheftSpec(
        victim_fraction=cfg.victim_fraction,
        negative_window=parse_duration(cfg.negative_window),
        theft_lookback=parse_duration(cfg.theft_lookback),
        donors=cfg.donors,
        amount_ratio=cfg.amount_ratio,
        min_prior_card_events=cfg.min_prior_card_events,
    )
    return splice_identity_theft(
        dataset.events,
        dataset.profiles,
        windows=side_label_windows(
            train_end, validation_end, last, parse_duration(config.observation_window)
        ),
        spec=spec,
        seed=seed,
        label_delay=parse_duration(config.temporal_split.label_delay),
        exclude=fraud_positive_users(dataset.labels),
    )


def load_adapter_events(
    config: RecipeConfig,
    *,
    flat_fields: bool = True,
    fixture_dir: Path | None = None,
) -> AdapterResult:
    """Load the configured book.

    Every adapter takes `(data_config, *, seed, flat_fields, fixture_dir)`.
    `flat_fields` tells adapters with mixed sources to keep only what the
    12-field tokenizers can read. `fixture_dir` is where an adapter may
    generate a synthetic book when no local root is configured.
    """
    name = config.data.adapter.strip().lower()
    if name not in _ADAPTERS:
        known = ", ".join(advertised_adapters()) or "<none>"
        raise UnsupportedCombinationError(
            f"Unknown adapter {name!r}. Registered adapters: {known}."
        )
    return _ADAPTERS[name](
        config.data, seed=config.seed, flat_fields=flat_fields, fixture_dir=fixture_dir
    )
