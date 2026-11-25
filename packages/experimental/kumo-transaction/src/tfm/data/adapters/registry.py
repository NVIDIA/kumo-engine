# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name registry for event adapters."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tfm.config import DataConfig, RecipeConfig
from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.registries import Registry
from tfm.contracts.schema import TransactionEvent, TransferEdge, ensure_utc
from tfm.contracts.sources import ColumnarSource
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


@dataclass(frozen=True)
class Adapter:
    """A book's loader, and optionally its columnar source.

    `data.adapter` selects one by registered name or by `"package.module:attribute"`.
    """

    load: Callable[..., AdapterResult]
    """`(data_config, *, seed, flat_fields, fixture_dir) -> AdapterResult`."""
    columnar_source: Callable[[DataConfig], ColumnarSource[Any]] | None = None
    """The book a token cache is built from; must pickle, its class importable."""


_ADAPTERS: Registry[Adapter] = Registry("adapter")


def register_adapter(
    name: str,
    *,
    columnar_source: Callable[[DataConfig], ColumnarSource[Any]] | None = None,
) -> Callable[[Callable[..., AdapterResult]], Callable[..., AdapterResult]]:
    """Register an adapter; `columnar_source` gives the book a token cache is built from."""

    def decorator(fn: Callable[..., AdapterResult]) -> Callable[..., AdapterResult]:
        _ADAPTERS.register(name, Adapter(fn, columnar_source))
        return fn

    return decorator


def advertised_adapters() -> tuple[str, ...]:
    return _ADAPTERS.names()


def _adapter(config: DataConfig) -> Adapter:
    name = config.adapter.strip()
    adapter = _ADAPTERS.get(name if ":" in name else name.lower())
    if not isinstance(adapter, Adapter):
        raise UnsupportedCombinationError(
            f"data.adapter {name!r} is a {type(adapter).__name__}; it must name a "
            "`tfm.data.adapters.Adapter`."
        )
    return adapter


def columnar_source(config: DataConfig) -> ColumnarSource[Any]:
    """The configured book as a token cache reads it."""
    factory = _adapter(config).columnar_source
    if factory is None:
        raise UnsupportedCombinationError(
            f"data.adapter {config.adapter!r} supplies no columnar source, so no token cache "
            "can be built from it; give its `Adapter` a `columnar_source` returning a "
            "`tfm.contracts.sources.ColumnarSource`."
        )
    return factory(config)


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
        step_hours=config.event_step_hours,
    )
    owners = sorted({event.entity_id for event in events})
    successors = dict(zip(owners, owners[1:] + owners[:1], strict=True))
    transfers = tuple(
        TransferEdge(
            event.entity_id,
            successors[event.entity_id],
            event.event_timestamp,
            event.numeric["amount"],
        )
        for event in events
        if event.event_timestamp.day % 7 == 0
    )
    return AdapterResult(
        events=tuple(events), transfers=transfers, metadata={"adapter": "synthetic"}
    )


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


def _pragmatiq_columns(config: DataConfig) -> ColumnarSource[Any]:
    from tfm.data.adapters.pragmatiq_columns import PragmatiqColumns

    return PragmatiqColumns(resolve_data_path(config))


@register_adapter("pragmatiq", columnar_source=_pragmatiq_columns)
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

    spliced = config.fraud_labels == "spliced_identity_theft" and config.load_labels
    if config.path or os.environ.get(ADAPTER_ENV["pragmatiq"]):
        root = resolve_data_path(config)
        dataset = load_pragmatiq_dataset(
            root,
            max_entities=entity_cap(config),
            sample_seed=sample_seed(config, seed),
            attach_fraud_labels=not spliced and config.load_labels,
            load_labels=config.load_labels,
            stratify_by_fraud=not spliced and config.stratify_by_fraud and config.load_labels,
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
        dataset = load_pragmatiq_dataset(
            root, attach_fraud_labels=config.load_labels, load_labels=config.load_labels
        )
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
        transfers=dataset.transfers,
        labels={task: rows for task, rows in dataset.labels.items() if task != "fraud"},
        fraud_splice=splice,
    )


def _identity_theft_splice(
    config: DataConfig, dataset: PragmatiqDataset, *, seed: int
) -> IdentityTheftSplice:
    """dataset strategy's task over the sampled users, on the label windows `prepare_data` will use."""
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
    return _adapter(config.data).load(
        config.data, seed=config.seed, flat_fields=flat_fields, fixture_dir=fixture_dir
    )
