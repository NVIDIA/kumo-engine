# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spliced identity theft: an event-level fraud task built from an unmodified book (dataset strategy).

A victim and a donor are two users of the book. On one UTC theft day, the
donor's card transactions and app sessions are copied into the victim's
history at their original timestamps. The copied transactions are fraud; the
victim's own card transactions in the `negative_window` before the theft day,
and on the day itself, are legitimate. No other event carries a label.

The entity split is fixed here, before any label exists, so a victim and its
donor are always on the same side. Each theft day lies inside the label window
of its side (probe split rules). The splice never touches the input events: the
pretraining corpus and the forecast tasks keep reading the unspliced book.
"""

from __future__ import annotations

import bisect
import statistics
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from random import Random
from typing import Any, Literal

from tfm.contracts.errors import InsufficientLabelSupportError, SchemaError
from tfm.contracts.schema import EntityProfile, EventLabel, TransactionEvent, ensure_utc
from tfm.data.labels.windows import LabelWindow
from tfm.data.splits.entity import EntitySplit

CARD_TXN_TYPES: tuple[str, ...] = ("card_payment", "atm_withdrawal")
"""Transaction types a card thief can make. Salary, rent, and transfers stay with the donor."""

SIDES: tuple[str, ...] = ("train", "evaluation")
"""Probe sides. `train` draws theft days from the `train` label window, `evaluation` from `test`."""

_WINDOW_FOR_SIDE = {"train": "train", "evaluation": "test"}
_DAY = timedelta(days=1)


@dataclass(frozen=True)
class IdentityTheftSpec:
    """How victims, donors, and theft days are chosen."""

    victim_fraction: float = 0.2
    """Share of each side's eligible users who become victims."""
    evaluation_fraction: float = 0.3
    """Share of eligible users on the evaluation side of the probe split."""
    negative_window: timedelta = timedelta(days=7)
    """How far before the theft day the victim's own card transactions are labelled legitimate."""
    theft_lookback: timedelta = timedelta(days=28)
    """Theft days fall in this span before the end of the side's label window.

    Probe sequences keep the most recent events, so a theft day far before the
    cutoff would fall outside the sequence the entity-level probe scores.
    """
    donors: Literal["matched", "random"] = "matched"
    """`matched`: same home country and a median amount within `amount_ratio`."""
    amount_ratio: float = 1.5
    min_prior_card_events: int = 20
    """Card transactions a victim needs before the theft day, so there is a history to compare."""
    max_donor_tries: int = 200
    """Donor candidates tried per theft day before the day is given up."""
    max_day_tries: int = 5
    """Theft days tried per victim before the victim is left unspliced."""

    def __post_init__(self) -> None:
        for name in ("victim_fraction", "evaluation_fraction"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"IdentityTheftSpec.{name} must be in (0, 1); got {value}.")
        if self.negative_window < timedelta(0) or self.theft_lookback < _DAY:
            raise ValueError(
                "IdentityTheftSpec.negative_window must be non-negative and "
                "theft_lookback at least one day."
            )
        if self.amount_ratio < 1.0:
            raise ValueError(
                f"IdentityTheftSpec.amount_ratio must be >= 1; got {self.amount_ratio}."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "victim_fraction": self.victim_fraction,
            "evaluation_fraction": self.evaluation_fraction,
            "negative_window": str(self.negative_window),
            "theft_lookback": str(self.theft_lookback),
            "donors": self.donors,
            "amount_ratio": self.amount_ratio,
            "min_prior_card_events": self.min_prior_card_events,
            "max_donor_tries": self.max_donor_tries,
            "max_day_tries": self.max_day_tries,
        }


@dataclass(frozen=True)
class TheftRecord:
    """One splice: who, from whom, on which day, and how many events it labelled."""

    victim: str
    donor: str
    side: str
    theft_day: datetime
    n_fraud: int
    n_app_sessions: int
    n_legitimate: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "victim": self.victim,
            "donor": self.donor,
            "side": self.side,
            "theft_day": self.theft_day.date().isoformat(),
            "n_fraud": self.n_fraud,
            "n_app_sessions": self.n_app_sessions,
            "n_legitimate": self.n_legitimate,
        }


@dataclass(frozen=True)
class IdentityTheftSplice:
    """The victims' spliced histories, the probe split they were built for, and the manifest.

    Only victims' histories change, so only those are held; `apply` lays them
    over the unspliced events.
    """

    victim_events: Mapping[str, tuple[TransactionEvent, ...]]
    split: EntitySplit
    records: tuple[TheftRecord, ...]
    spec: IdentityTheftSpec
    windows: Mapping[str, LabelWindow]
    seed: int
    excluded: tuple[str, ...] = ()
    n_eligible: int = 0
    label: str = "fraud"
    stats: Mapping[str, int] = field(default_factory=dict)
    copied_from: Mapping[str, str] = field(default_factory=dict)
    """Each copy's `event_id` to the donor event it was copied from."""

    @property
    def victims(self) -> frozenset[str]:
        return frozenset(record.victim for record in self.records)

    def apply(self, events: Sequence[TransactionEvent]) -> list[TransactionEvent]:
        """`events` with every victim's history replaced by its spliced one."""
        out = [event for event in events if event.entity_id not in self.victim_events]
        for history in self.victim_events.values():
            out.extend(history)
        return out

    def with_victim_events(
        self, victim_events: Mapping[str, Sequence[TransactionEvent]]
    ) -> IdentityTheftSplice:
        """The same splice over re-mapped victim histories (for example, aliased fields)."""
        if set(victim_events) != set(self.victim_events):
            raise SchemaError("Re-mapped victim histories must cover exactly the same victims.")
        return replace(self, victim_events={k: tuple(v) for k, v in victim_events.items()})

    def manifest(self) -> dict[str, Any]:
        return {
            "task": "spliced_identity_theft",
            "label": self.label,
            "seed": self.seed,
            "spec": self.spec.as_dict(),
            "label_windows": {
                side: self.windows[_WINDOW_FOR_SIDE[side]].as_dict() for side in SIDES
            },
            "n_eligible": self.n_eligible,
            "n_excluded": len(self.excluded),
            "split": self.split.as_dict(),
            "n_victims": {side: sum(1 for r in self.records if r.side == side) for side in SIDES},
            "n_fraud_events": sum(r.n_fraud for r in self.records),
            "n_legitimate_events": sum(r.n_legitimate for r in self.records),
            **dict(self.stats),
            "records": [record.as_dict() for record in self.records],
        }


def is_card_transaction(event: TransactionEvent) -> bool:
    return (
        event.categorical.get("source", "transaction") == "transaction"
        and event.categorical.get("txn_type") in CARD_TXN_TYPES
    )


def _is_app_session(event: TransactionEvent) -> bool:
    return event.categorical.get("source") == "app"


def _day_start(ts: datetime) -> datetime:
    utc = ensure_utc(ts)
    return datetime(utc.year, utc.month, utc.day, tzinfo=UTC)


def _count_in(sorted_ts: Sequence[datetime], lo: datetime, hi: datetime) -> int:
    """Timestamps in `[lo, hi)`."""
    return bisect.bisect_left(sorted_ts, hi) - bisect.bisect_left(sorted_ts, lo)


def _labelled(
    event: TransactionEvent, name: str, value: float, delay: timedelta
) -> TransactionEvent:
    labels = dict(event.labels)
    labels[name] = EventLabel(
        name=name, value=value, available_at=event.event_timestamp + delay, level="event"
    )
    return TransactionEvent(
        entity_id=event.entity_id,
        event_id=event.event_id,
        event_timestamp=event.event_timestamp,
        timezone=event.timezone,
        categorical=event.categorical,
        numeric=event.numeric,
        text=event.text,
        labels=labels,
        group=event.group,
    )


def _copy_to(
    event: TransactionEvent, victim: str, index: int, name: str, delay: timedelta
) -> TransactionEvent:
    """The donor's event in the victim's history. Only the entity and the id change."""
    ts_us = int(ensure_utc(event.event_timestamp).timestamp() * 1_000_000)
    labels = (
        {name: EventLabel(name=name, value=1.0, available_at=event.event_timestamp + delay)}
        if is_card_transaction(event)
        else {}
    )
    return TransactionEvent(
        entity_id=victim,
        event_id=f"{victim}:{ts_us}:theft{index}",
        event_timestamp=event.event_timestamp,
        timezone=event.timezone,
        categorical=dict(event.categorical),
        numeric=dict(event.numeric),
        text=dict(event.text),
        labels=labels,
        group=None,
    )


def _assign_sides(eligible: list[str], fraction: float, rng: Random) -> dict[str, list[str]]:
    shuffled = sorted(eligible)
    rng.shuffle(shuffled)
    n_eval = round(fraction * len(shuffled))
    if n_eval < 2 or len(shuffled) - n_eval < 2:
        raise InsufficientLabelSupportError(
            f"{len(shuffled)} eligible users cannot fill both sides of the identity-theft "
            "probe split with at least two users each; sample more users."
        )
    return {"evaluation": shuffled[:n_eval], "train": shuffled[n_eval:]}


def splice_identity_theft(
    events: Sequence[TransactionEvent],
    profiles: Sequence[EntityProfile],
    *,
    windows: Mapping[str, LabelWindow],
    spec: IdentityTheftSpec,
    seed: int,
    label_delay: timedelta = timedelta(0),
    exclude: Collection[str] = (),
    label: str = "fraud",
) -> IdentityTheftSplice:
    """Splice theft days into a copy of `events` and fix the probe split they belong to.

    `windows` needs the `train` and `test` label windows (probe split rules). `exclude`
    names users who may take no role and join neither side, such as users
    with the book's own fraud. Input events must not already carry `label`:
    two label sources on one event would make the task ambiguous.

    Deterministic in `(events, profiles, windows, spec, seed)`.
    """
    for side in SIDES:
        if _WINDOW_FOR_SIDE[side] not in windows:
            raise SchemaError(
                f"splice_identity_theft needs the {_WINDOW_FOR_SIDE[side]!r} label window."
            )
    by_entity: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        if label in event.labels:
            raise SchemaError(
                f"Event {event.event_id!r} already carries a {label!r} label; load the book "
                "without its own fraud labels before splicing (dataset strategy)."
            )
        by_entity[event.entity_id].append(event)
    for history in by_entity.values():
        history.sort(key=lambda e: ensure_utc(e.event_timestamp))

    card_ts: dict[str, list[datetime]] = {}
    median_amount: dict[str, float] = {}
    for entity_id, history in by_entity.items():
        cards = [e for e in history if is_card_transaction(e)]
        card_ts[entity_id] = [ensure_utc(e.event_timestamp) for e in cards]
        amounts = [float(e.numeric["amount"]) for e in cards if "amount" in e.numeric]
        if amounts:
            median_amount[entity_id] = statistics.median(amounts)
    home = {p.entity_id: p.attributes.get("country") for p in profiles}

    excluded = set(exclude)
    eligible = [
        entity_id
        for entity_id in by_entity
        if entity_id not in excluded
        and len(card_ts[entity_id]) >= spec.min_prior_card_events
        and entity_id in median_amount
        and (spec.donors == "random" or home.get(entity_id))
    ]
    rng = Random(seed)
    sides = _assign_sides(eligible, spec.evaluation_fraction, rng)

    def matches(victim: str, donor: str) -> bool:
        if spec.donors == "random":
            return True
        ratio = median_amount[donor] / max(median_amount[victim], 1e-6)
        return home[donor] == home[victim] and 1 / spec.amount_ratio <= ratio <= spec.amount_ratio

    records: list[TheftRecord] = []
    spliced: dict[str, list[TransactionEvent]] = {}
    copied_from: dict[str, str] = {}
    unmatched = 0
    for side in SIDES:
        window = windows[_WINDOW_FOR_SIDE[side]]
        members = sides[side]
        n_victims = max(1, round(spec.victim_fraction * len(members)))
        victims, donor_pool = members[:n_victims], members[n_victims:]
        used: set[str] = set()
        # A theft day's events must all fall at or before the window's end.
        last_day = _day_start(window.end - _DAY)
        first_day = _day_start(window.end - spec.theft_lookback)
        if window.start is not None:
            while first_day <= ensure_utc(window.start):
                first_day += _DAY
        days = []
        day = first_day
        while day <= last_day:
            days.append(day)
            day += _DAY
        for victim in victims:
            chosen = _splice_one(victim, days, donor_pool, used, card_ts, matches, spec, rng)
            if chosen is None:
                unmatched += 1
                continue
            donor, theft_day = chosen
            used.add(donor)
            day_end = theft_day + _DAY
            originals = [
                e
                for e in by_entity[donor]
                if theft_day <= ensure_utc(e.event_timestamp) < day_end
                and (is_card_transaction(e) or _is_app_session(e))
            ]
            copies = [
                _copy_to(event, victim, index, label, label_delay)
                for index, event in enumerate(originals)
            ]
            copied_from.update(
                (copy.event_id, event.event_id)
                for copy, event in zip(copies, originals, strict=True)
            )
            negative_start = theft_day - spec.negative_window
            own: list[TransactionEvent] = []
            n_legitimate = 0
            for event in by_entity[victim]:
                ts = ensure_utc(event.event_timestamp)
                if (
                    is_card_transaction(event)
                    and negative_start <= ts < day_end
                    and window.contains(ts)
                ):
                    own.append(_labelled(event, label, 0.0, label_delay))
                    n_legitimate += 1
                else:
                    own.append(event)
            merged = sorted([*own, *copies], key=lambda e: ensure_utc(e.event_timestamp))
            spliced[victim] = merged
            records.append(
                TheftRecord(
                    victim=victim,
                    donor=donor,
                    side=side,
                    theft_day=theft_day,
                    n_fraud=sum(1 for e in copies if label in e.labels),
                    n_app_sessions=sum(1 for e in copies if _is_app_session(e)),
                    n_legitimate=n_legitimate,
                )
            )
    if not any(r.side == "train" for r in records) or not any(
        r.side == "evaluation" for r in records
    ):
        raise InsufficientLabelSupportError(
            "No theft day could be spliced on one side of the probe split; sample more "
            "users or widen theft_lookback."
        )
    return IdentityTheftSplice(
        victim_events={victim: tuple(history) for victim, history in spliced.items()},
        split=EntitySplit(train=tuple(sides["train"]), evaluation=tuple(sides["evaluation"])),
        records=tuple(records),
        spec=spec,
        windows=dict(windows),
        seed=seed,
        excluded=tuple(sorted(excluded)),
        n_eligible=len(eligible),
        label=label,
        stats={"n_victims_unmatched": unmatched},
        copied_from=copied_from,
    )


def _splice_one(
    victim: str,
    days: list[datetime],
    donor_pool: list[str],
    used: set[str],
    card_ts: Mapping[str, list[datetime]],
    matches: Any,
    spec: IdentityTheftSpec,
    rng: Random,
) -> tuple[str, datetime] | None:
    """A `(donor, theft_day)` for `victim`, or `None` when no day and donor fit."""
    own = card_ts[victim]
    candidates = [
        day
        for day in days
        if bisect.bisect_left(own, day) >= spec.min_prior_card_events
        and _count_in(own, day - spec.negative_window, day + _DAY) > 0
    ]
    rng.shuffle(candidates)
    for day in candidates[: spec.max_day_tries]:
        for donor in rng.sample(donor_pool, min(spec.max_donor_tries, len(donor_pool))):
            if donor in used or not matches(victim, donor):
                continue
            if _count_in(card_ts[donor], day, day + _DAY) > 0:
                return donor, day
    return None
