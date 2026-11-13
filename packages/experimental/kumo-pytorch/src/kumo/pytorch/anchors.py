# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the anchors are, and the order they are consumed in.

An **anchor** is one `(entity, prediction time)` pair — one thing to predict.
`Dataset` says where the *table* bytes are; this says where the *anchor* rows
are. The two are deliberately separate: the anchor set changes between train,
validation and a backfill while the contract does not, so binding anchors into
`DataLoaderSpec` would change the fingerprint of runs whose data semantics are
identical.

kumo-pytorch does not enumerate anchors. Which entity is predicted at which time, how
train/validation/test are split, and which candidates are scored are decisions
with business meaning that belong upstream — the same reasoning that keeps
candidate *generation* out. What kumo-pytorch does is read that set without routing a
billion rows through Python objects, and decide the order it is consumed in.

An engine may choose the order, subject to these guarantees. Every epoch visits
each anchor exactly once, ranks are disjoint and cover the split, every rank
gets the same number of anchors, and the sequence is a pure function of seed and
epoch. `shard_positions` is the default order that satisfies these, and
`padded_positions` equalizes ranks with a marker rather than by silently
repeating anchors. Identical order *across* engines is not a goal: per-sample
randomness is keyed by sample identity, so engines are compared as sets keyed
by ``sample_id``.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import chain
from typing import Any, TypeAlias

from kumo.pytorch.contract.sample import ExternalTarget, SampleRequest
from kumo.pytorch.dataset import InMemoryRows, Parquet, TableSource
from kumo.pytorch.spec import DataLoaderSpec, ValidationIssue

__all__ = [
    "ALL_ANCHORS",
    "PADDING_KEY",
    "AllAnchors",
    "AnchorTable",
    "CandidateColumn",
    "PositionalAnchors",
    "RequestAnchors",
    "SplitSelection",
    "anchor_issues",
    "padded_positions",
    "shard_positions",
    "split_selection",
]

PADDING_KEY = "_is_padding"
"""Reserved request-metadata key marking a padding anchor.

Surfaced to callers as ``Batch.metadata["is_padding"]`` and removed from
``request_metadata``, so it never collides with a user's own metadata.
"""


@dataclass(frozen=True)
class AllAnchors:
    """Selects every anchor, whatever its split — including anchors with none.

    Written as the module-level singleton :data:`ALL_ANCHORS`. A sentinel rather
    than a string such as ``"all"``, which could be a real split name.
    """

    def __repr__(self) -> str:  # pragma: no cover - display only
        return "ALL_ANCHORS"


ALL_ANCHORS = AllAnchors()

SplitSelection: TypeAlias = AllAnchors | frozenset[str | None]
"""Which anchors a call reads: every one, or those whose split is in the set.

``None`` in the set selects anchors that carry no split.
"""


def split_selection(
    value: str | Collection[str | None] | AllAnchors | None,
) -> SplitSelection | None:
    """Normalize a caller's split selector; ``None`` means none was given."""

    if value is None or isinstance(value, AllAnchors):
        return value
    if isinstance(value, str):
        if not value:
            raise ValueError("a split name must be non-empty")
        return frozenset({value})
    names = frozenset(value)
    if not names:
        raise ValueError("a split selection must name at least one split")
    if any(name is not None and (not isinstance(name, str) or not name) for name in names):
        raise ValueError("split selections hold non-empty split names, or None for no split")
    return names


def describe_selection(selection: SplitSelection) -> Any:
    """A JSON-friendly, order-stable form, for configurations and messages."""

    if isinstance(selection, AllAnchors):
        return "ALL_ANCHORS"
    return sorted(selection, key=lambda name: (name is None, name or ""))


@dataclass(frozen=True)
class CandidateColumn:
    """One named candidate set, stored as a nested column on the anchor row.

    A list of structs rather than a map of id to label, for three reasons:
    candidate order is load-bearing — `CandidateTarget` labels align positionally
    with the candidate input — and a map is unordered by definition; a candidate
    may carry several named labels; and scoring candidates carry none at all,
    which a struct says better than a null value.

    Nesting the candidates on the anchor row rather than joining a second table
    keeps the loader reading instead of joining. Whoever decided which
    candidates exist can group them in the same query, once.

    :param column: the ``LIST<STRUCT<...>>`` column holding this set.
    :param entity_id: struct field holding each candidate's primary-key value.
    :param targets: target name to the struct field supplying it. Empty for
        candidates being scored rather than trained on.
    """

    column: str
    entity_id: str
    targets: Mapping[str, str] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.column or not self.entity_id:
            raise ValueError("a candidate column needs a column and an entity field")
        targets = {str(name): str(field) for name, field in self.targets.items()}
        if any(not name or not field for name, field in targets.items()):
            raise ValueError("candidate target names and fields must be non-empty")
        object.__setattr__(self, "targets", targets)

    @property
    def fields(self) -> tuple[str, ...]:
        """Struct fields this binding reads, in a stable order."""

        return (self.entity_id, *sorted(self.targets.values()))


@dataclass(frozen=True)
class AnchorTable:
    """An anchor set stored as a table, bound to the columns that give it meaning.

    Like `Dataset`, it names no engine and, apart from in-memory rows, is
    serializable.

    Expresses everything a `SampleRequest` does. Scalars are columns; candidates
    are a nested column per named set, which is the relational form of
    ``Mapping[str, tuple[CandidateRequest, ...]]``.

    :param source: where the anchor rows are; the same source kinds `Dataset`
        binds.
    :param entity_id: input column holding each entity's primary-key value;
        this maps request data, not the entity table's lookup column.
    :param as_of_time: column holding the prediction time. Required when any
        table in the contract declares a clock and refused, rather than ignored,
        when none does — see `DataLoaderSpec.requires_anchor_time`.
    :param sample_id: column identifying each anchor. Omit to use the anchor's
        position in the table, which is stable because the order is.
    :param split: column to filter on, so one table can hold every split.
    :param targets: typed external target name to the column supplying it.
        Declare its dtype with ``ExternalTarget`` in ``SampleSpec.targets``.
        A name also computed by the sample is refused, not resolved by precedence.
    :param metadata: columns carried through to `Batch.metadata` untouched.
        Declared rather than free-form, so the table has one schema.
    :param candidates: candidate-set name to the nested column holding it.
    """

    source: TableSource
    entity_id: str
    as_of_time: str | None = None
    sample_id: str | None = None
    split: str | None = None
    targets: Mapping[str, str] = dataclass_field(default_factory=dict)
    metadata: Sequence[str] = ()
    candidates: Mapping[str, CandidateColumn] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, Parquet | InMemoryRows):
            raise TypeError(f"unknown anchor source {self.source!r}")
        for role, column in (
            ("entity_id", self.entity_id),
            ("as_of_time", self.as_of_time),
            ("sample_id", self.sample_id),
            ("split", self.split),
        ):
            if column is not None and not column:
                raise ValueError(f"anchor table {role} must be a column name or omitted")
        targets = {str(name): str(column) for name, column in self.targets.items()}
        if any(not name or not column for name, column in targets.items()):
            raise ValueError("anchor table target names and columns must be non-empty")
        metadata = tuple(str(name) for name in self.metadata)
        if any(not name for name in metadata):
            raise ValueError("anchor table metadata column names must be non-empty")
        if PADDING_KEY in metadata:
            raise ValueError(f"{PADDING_KEY!r} is reserved for the padding marker")
        candidates = {str(name): value for name, value in self.candidates.items()}
        if any(not name for name in candidates):
            raise ValueError("candidate set names must be non-empty")
        if any(not isinstance(value, CandidateColumn) for value in candidates.values()):
            raise TypeError("candidate sets are bound with CandidateColumn")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "candidates", candidates)

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column this binding reads, in a stable order."""

        declared = (self.entity_id, self.as_of_time, self.sample_id, self.split)
        return (
            *(name for name in declared if name),
            *sorted(self.targets.values()),
            *self.metadata,
            *sorted(value.column for value in self.candidates.values()),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form; refused for in-memory rows, which are not a storage reference."""
        return {
            "source": self.source.to_dict(),
            "entity_id": self.entity_id,
            "as_of_time": self.as_of_time,
            "sample_id": self.sample_id,
            "split": self.split,
            "targets": dict(self.targets),
            "metadata": list(self.metadata),
            "candidates": {
                name: {
                    "column": value.column,
                    "entity_id": value.entity_id,
                    "targets": dict(value.targets),
                }
                for name, value in self.candidates.items()
            },
        }


def anchor_issues(
    contract: DataLoaderSpec,
    table: AnchorTable,
) -> tuple[ValidationIssue, ...]:
    """Everything wrong with binding ``table`` to ``contract`` that no engine decides.

    Checked before the table is read, by both `check` and `Plan.materialize`,
    so correctness never depends on a caller remembering to ask.
    """

    issues: list[ValidationIssue] = []
    requires_time = contract.requires_anchor_time
    if requires_time and table.as_of_time is None:
        issues.append(
            ValidationIssue(
                "anchors.as_of_time",
                "this contract reads tables that declare a clock, but the anchor table "
                "declares no column holding each anchor's prediction time",
                "set as_of_time to the column holding each anchor's prediction time",
            )
        )
    if not requires_time and table.as_of_time is not None:
        issues.append(
            ValidationIssue(
                "anchors.as_of_time",
                f"the anchor table declares as_of_time={table.as_of_time!r}, but no table "
                "in this contract declares a clock",
                "remove as_of_time, or declare a clock on the table it should bound",
            )
        )
    collisions = sorted(
        name
        for name in table.targets
        if name in contract.sample.targets
        and not isinstance(contract.sample.targets[name], ExternalTarget)
    )
    if collisions:
        issues.append(
            ValidationIssue(
                "anchors.targets",
                f"{collisions} are supplied by the anchor table and also computed by "
                "SampleSpec.targets",
                "rename one, or drop it from whichever side should not own it",
            )
        )
    declared_sets = contract.sample.candidate_sets
    for name in sorted(set(table.candidates) - set(declared_sets)):
        issues.append(
            ValidationIssue(
                f"anchors.candidates.{name}",
                f"the anchor table supplies candidate set {name!r}, which "
                "SampleSpec.candidate_sets does not declare",
                f"declare {name!r} in SampleSpec.candidate_sets, or remove it from the table"
                if declared_sets
                else "declare the candidate set in SampleSpec.candidate_sets",
            )
        )
    for name in sorted(set(table.candidates) & set(declared_sets)):
        overlap = sorted(
            key
            for key in table.candidates[name].targets
            if key in declared_sets[name].targets
            and not isinstance(declared_sets[name].targets[key], ExternalTarget)
        )
        if overlap:
            issues.append(
                ValidationIssue(
                    f"anchors.candidates.{name}.targets",
                    f"{overlap} are supplied by the anchor table and also computed by "
                    f"SampleSpec.candidate_sets[{name!r}].targets",
                    "rename one, or drop it from whichever side should not own it",
                )
            )
    return tuple(issues)


def _mix(value: int) -> int:
    """splitmix64, the avalanche step. Cheap, and good enough to be a round."""

    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


_ROUNDS = 10
"""Rounds only affect mixing quality; every round is a bijection, so a
permutation is valid at any count. FF1 uses 10 and FF3-1 uses 8; rounds are a
few multiplies, so there is no reason to economize."""


class _Permutation:
    """A seeded permutation of ``[0, n)``, evaluable one position at a time.

    A Feistel network over the smallest power-of-two domain containing ``n``,
    with cycle-walking back into range. Two properties matter and neither is
    negotiable:

    * it is a **bijection** — if two positions mapped to one row, some anchors
      would be trained on twice and others never, with correct shapes, correct
      counts, and nothing anywhere to notice;
    * it is **pointwise** — position *p* resolves without computing any other,
      so an epoch's order costs no memory. Materializing it is what makes an
      anchor list a Python object again, which is the thing being removed.
    """

    __slots__ = ("_half", "_keys", "_mask", "_n")

    def __init__(self, n: int, seed: int) -> None:
        if n < 0:
            raise ValueError("permutation size must be non-negative")
        self._n = n
        bits = max(2, (n - 1).bit_length()) if n > 1 else 2
        self._half = (bits + 1) // 2
        self._mask = (1 << self._half) - 1
        self._keys = tuple(_mix(seed ^ _mix(round_index)) for round_index in range(_ROUNDS))

    def __len__(self) -> int:
        return self._n

    def __call__(self, position: int) -> int:
        if not 0 <= position < self._n:
            raise IndexError(f"position {position} outside [0, {self._n})")
        value = position
        # Cycle-walk: the Feistel domain is a power of two and so usually
        # larger than n. Re-encrypting an out-of-range image stays a bijection
        # because the walk is itself a permutation of the larger domain.
        while True:
            left = value >> self._half
            right = value & self._mask
            for key in self._keys:
                left, right = right, left ^ (_mix(key ^ right) & self._mask)
            value = (left << self._half) | right
            if value < self._n:
                return value


def shard_positions(
    n: int,
    *,
    seed: int = 0,
    epoch: int = 0,
    shuffle: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[int]:
    """Yield the row positions one rank consumes this epoch, in order.

    Every position in ``[0, n)`` belongs to exactly one rank — coverage and
    disjointness hold structurally rather than by check, because rank *r* of
    *k* takes the permuted order at ``r, r + k, r + 2k, …``.

    ``shuffle`` is a **true permutation**, not an approximation over a buffer:
    every anchor appears exactly once per epoch, and a different epoch is a
    different order. That matters because a shuffle that quietly repeats or
    drops anchors produces correct shapes and counts with no symptom.

    Defined here rather than in an engine, so the order depends only on the
    table, seed, epoch and rank.
    """

    if n < 0:
        raise ValueError("request count must be non-negative")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must identify one member of a positive world size")
    if not shuffle:
        yield from range(rank, n, world_size)
        return
    order = _Permutation(n, _mix(seed) ^ _mix(epoch))
    for position in range(rank, n, world_size):
        yield order(position)


def padded_positions(
    positions: Iterable[int],
    *,
    count: int,
    world_size: int,
) -> Iterator[tuple[int, bool]]:
    """Yield ``(position, is_padding)`` so every rank gets the same number of anchors.

    A step-synchronised trainer hangs at an epoch's last collective if one rank
    has a batch the others do not. Each rank is topped up to
    ``ceil(count / world_size)`` by repeating an anchor it already has, marked
    as padding so a loss can exclude it — never silently, because an unmarked
    repeat trains on that anchor twice.
    """

    if world_size <= 0:
        raise ValueError("world size must be positive")
    target = -(-count // world_size) if count else 0
    first: int | None = None
    emitted = 0
    for position in positions:
        if first is None:
            first = position
        emitted += 1
        if emitted > target:
            raise ValueError(
                f"an anchor order assigned more than ceil({count}/{world_size}) = {target} "
                "anchors to one rank; ranks must partition the split"
            )
        yield position, False
    for _ in range(target - emitted):
        yield (0 if first is None else first), True


class PositionalAnchors:
    """Anchors in one fixed order, partitioned by split: the reference `AnchorTask`.

    Subclasses supply ``_row_count``, ``_split_of(row)`` and ``_request(row)``.
    Positions within a selection index into its rows in canonical order, which
    is fixed once here — so a position means the same anchor for every rank and
    every epoch.
    """

    def __init__(self) -> None:
        by_split: dict[str | None, list[int]] = {}
        for row in range(self._row_count()):
            by_split.setdefault(self._split_of(row), []).append(row)
        self._all_rows = tuple(range(self._row_count()))
        self._rows_by_split = {name: tuple(rows) for name, rows in by_split.items()}
        self._selected: dict[SplitSelection, tuple[int, ...]] = {}

    def _row_count(self) -> int:
        raise NotImplementedError

    def _split_of(self, row: int) -> str | None:
        raise NotImplementedError

    def _request(self, row: int) -> SampleRequest:
        raise NotImplementedError

    def _rows(self, selection: SplitSelection) -> tuple[int, ...]:
        if isinstance(selection, AllAnchors):
            return self._all_rows
        cached = self._selected.get(selection)
        if cached is not None:
            return cached
        unknown = sorted(
            name for name in selection if name is not None and name not in self._rows_by_split
        )
        if unknown:
            present = list(self.splits())
            raise ValueError(
                f"no anchors in split(s) {unknown}; splits present: {present}"
                if present
                else f"no anchors in split(s) {unknown}; these anchors declare no splits"
            )
        rows = tuple(
            sorted(chain.from_iterable(self._rows_by_split.get(name, ()) for name in selection))
        )
        self._selected[selection] = rows
        return rows

    def splits(self) -> tuple[str, ...]:
        return tuple(sorted(name for name in self._rows_by_split if name is not None))

    def count(self, selection: SplitSelection = ALL_ANCHORS) -> int:
        return len(self._rows(selection))

    def order(
        self,
        selection: SplitSelection,
        *,
        seed: int,
        epoch: int,
        shuffle: bool,
        rank: int,
        world_size: int,
    ) -> Iterator[int]:
        return shard_positions(
            self.count(selection),
            seed=seed,
            epoch=epoch,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size,
        )

    def fetch(self, selection: SplitSelection, positions: Sequence[int]) -> Sequence[SampleRequest]:
        rows = self._rows(selection)
        return [self._request(rows[position]) for position in positions]

    def close(self) -> None:
        """Nothing is held open; defined so every task closes the same way."""


class RequestAnchors(PositionalAnchors):
    """Anchors supplied in code as `SampleRequest` values, in the order given."""

    def __init__(self, requests: Sequence[SampleRequest]) -> None:
        self._requests = tuple(requests)
        if any(PADDING_KEY in request.metadata for request in self._requests):
            raise ValueError(f"{PADDING_KEY!r} is reserved for the padding marker")
        super().__init__()

    def _row_count(self) -> int:
        return len(self._requests)

    def _split_of(self, row: int) -> str | None:
        return self._requests[row].split

    def _request(self, row: int) -> SampleRequest:
        return self._requests[row]
