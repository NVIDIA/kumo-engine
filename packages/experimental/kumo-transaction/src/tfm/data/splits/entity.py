# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity-disjoint train / evaluation splits for downstream probes.

Parquet label books do not ship folds, so probe rows have to be divided here.
The split is by entity and stratified by class, and the two sides never share
an entity: overlapping them scores a probe on rows it was fit on, which is how
a five-feature logistic regression came to report ROC-AUC 1.0 (evaluation protocol).

Temporal partitioning stays in `tfm.data.splits.temporal`. This module divides
*labeled entities* for a probe and says nothing about event time; each row's
history must already be truncated to its own cutoff before it gets here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

from tfm.contracts.errors import InsufficientLabelSupportError

MIN_ENTITIES_PER_CLASS = 2
"""A class with one entity cannot be split without reusing that entity."""


@dataclass(frozen=True)
class EntitySplit:
    """Entity ids on each side of a probe split, guaranteed disjoint."""

    train: tuple[str, ...]
    evaluation: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.train) & set(self.evaluation)
        if overlap:
            raise InsufficientLabelSupportError(
                f"Probe split is not entity-disjoint; {len(overlap)} shared "
                f"entities including {sorted(overlap)[:5]}."
            )

    def as_dict(self) -> dict[str, int | list[str]]:
        return {
            "n_train_entities": len(self.train),
            "n_eval_entities": len(self.evaluation),
            "shared_entities": [],
        }


def split_entities(
    entity_classes: Sequence[tuple[str, int]],
    *,
    seed: int,
    train_fraction: float = 0.7,
) -> EntitySplit:
    """Split labeled entities into disjoint train / evaluation sides.

    `entity_classes` pairs each entity id with its integer class. Entities are
    shuffled within class and divided by `train_fraction`, so both sides keep
    the class mix rather than drifting with the shuffle.

    Raises `InsufficientLabelSupportError` when fewer than two classes are
    present, or when any present class holds fewer than
    `MIN_ENTITIES_PER_CLASS` entities. Callers skip the task; they must not
    widen the split, reuse an entity, or fall back to a random one.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}.")

    by_class: dict[int, list[str]] = {}
    seen: set[str] = set()
    for entity_id, klass in entity_classes:
        if entity_id in seen:
            continue
        seen.add(entity_id)
        by_class.setdefault(int(klass), []).append(entity_id)

    if len(by_class) < 2:
        seen_classes = sorted(by_class)
        raise InsufficientLabelSupportError(
            f"Only class {seen_classes} is present among {len(seen)} labeled entities, so "
            "there is nothing for a probe to discriminate. Skip this task or supply a book "
            "that contains both classes; a single-class split is not a probe, and reusing an "
            "entity on both sides would score the probe on its own training rows."
        )

    thin = {klass: len(ids) for klass, ids in by_class.items() if len(ids) < MIN_ENTITIES_PER_CLASS}
    if thin:
        raise InsufficientLabelSupportError(
            f"Classes {thin} hold fewer than {MIN_ENTITIES_PER_CLASS} entities each, so no "
            "entity-disjoint probe split exists. Skip this task or supply a larger book; "
            "reusing an entity on both sides would score the probe on its own training rows."
        )

    rng = Random(seed)
    train: list[str] = []
    evaluation: list[str] = []
    for klass in sorted(by_class):
        ids = sorted(by_class[klass])
        rng.shuffle(ids)
        # Both sides keep at least one entity of every represented class.
        n_train = max(1, min(len(ids) - 1, round(train_fraction * len(ids))))
        train.extend(ids[:n_train])
        evaluation.extend(ids[n_train:])
    return EntitySplit(train=tuple(train), evaluation=tuple(evaluation))
