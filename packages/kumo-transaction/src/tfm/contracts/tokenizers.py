# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer layout contracts.

A sequence tokenizer is a composition slot outside ``TrxFM``. It maps
``TransactionSequence`` objects into a layout-specific encoding and packs that
encoding into a batch the matching history model can consume.

Layouts are not interchangeable. Construction-time validation must reject a
tokenizer whose ``layout`` is not in the granularity's ``accepted_layouts``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from tfm.contracts.schema import EntityProfile, TransactionSequence

LAYOUT_FLAT_FIELDS = "flat_fields"
LAYOUT_KEY_VALUE_TIME = "key_value_time"
LAYOUT_FLAT_KEY_VALUE = "flat_key_value"

FLAT_LAYOUTS = frozenset({LAYOUT_FLAT_FIELDS, LAYOUT_FLAT_KEY_VALUE})
"""Layouts that encode a sequence as one token stream in a `TransactionBatch`."""


@dataclass(frozen=True)
class FieldVocabulary:
    """Where each field's tokens sit in a `flat_fields` vocabulary.

    Field `i` owns ids `field_ranges[i][0] <= id < field_ranges[i][1]`; every id
    outside the ranges is a special token. Events are joined by `sep_id`.
    """

    vocab_size: int
    field_names: tuple[str, ...]
    field_ranges: tuple[tuple[int, int], ...]
    pad_id: int
    bos_id: int
    sep_id: int
    eos_id: int

    def __post_init__(self) -> None:
        if len(self.field_names) != len(self.field_ranges) or not self.field_ranges:
            raise ValueError("FieldVocabulary needs one id range per field.")
        previous = 0
        for start, stop in self.field_ranges:
            if not previous <= start < stop <= self.vocab_size:
                raise ValueError(
                    f"Field id ranges must be non-empty, ordered, and inside the vocabulary; "
                    f"got {self.field_ranges} for vocab_size {self.vocab_size}."
                )
            previous = stop


@dataclass(frozen=True)
class EncodedFlatSequence:
    """Unpadded token ids for a flat layout (`flat_fields` or `flat_key_value`)."""

    entity_id: str
    sequence_id: str
    token_ids: tuple[int, ...]
    layout: str = LAYOUT_FLAT_FIELDS


class SequenceTokenizer(Protocol):
    """Encode entity sequences into a named layout.

    ``fit`` may be a no-op for data-independent vocabs. Fitted tokenizers must
    be fit on train-partition sequences only.
    """

    name: str
    layout: str
    support_tier: str

    @property
    def vocab_size(self) -> int: ...

    def fit(self, sequences: Sequence[TransactionSequence]) -> SequenceTokenizer: ...

    def encode(self, sequence: TransactionSequence) -> object: ...

    def collate(
        self,
        sequences: Sequence[TransactionSequence],
        *,
        max_length: int,
    ) -> object: ...

    def manifest(self) -> dict[str, object]: ...


@runtime_checkable
class StatefulSequenceTokenizer(SequenceTokenizer, Protocol):
    """Sequence tokenizer whose fitted state can be embedded in an experiment."""

    def state_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> StatefulSequenceTokenizer: ...


FIELD_CATEGORICAL, FIELD_NUMERIC, FIELD_TEXT = 0, 1, 2
"""Field views, in the order `event_fields` lists them."""


@dataclass(frozen=True)
class EventBlock:
    """Block `index` of a book's users, events in `order_events` order.

    Pair arrays list every supplied field of every event (map fields and optional
    `source`), one pair per (event, key), in no particular order; `pair_event` says whose they are.
    """

    index: int
    """The block's place in the book; blocks are in entity order."""
    user_ids: tuple[str, ...]
    user_offsets: NDArray[np.int64]
    ts_us: NDArray[np.int64]
    """True microseconds since the epoch."""
    id_us: NDArray[np.int64]
    """The microsecond value in `event_id`, `int(ts.timestamp() * 1_000_000)`."""
    loader_index: NDArray[np.int64]
    """The per-user index in `event_id`."""
    pair_event: NDArray[np.int64]
    pair_key: NDArray[np.int64]
    """Index into `keys`."""
    pair_value: NDArray[np.int64]
    """Index into `values`."""
    pair_kind: NDArray[np.int8]
    pair_map_position: NDArray[np.int64]
    keys: tuple[str, ...]
    values: tuple[str, ...]
    """Each field's string as `event_fields` would give it, ascending, each once."""
    numbers: NDArray[np.float64]
    """`values` parsed as floats, NaN where a value is not numeric."""

    @property
    def n_events(self) -> int:
        return len(self.ts_us)


@dataclass(frozen=True)
class EncodedBlock:
    """One block's tokens, ragged by event, and the tokenizer's per-event arrays.

    Event `i` owns tokens `token_offsets[i]:token_offsets[i + 1]` of `key_ids`,
    `value_ids` and `positions`. Each array keeps its dtype in the cache.
    """

    token_offsets: NDArray[np.int64]
    key_ids: NDArray[Any]
    value_ids: NDArray[Any]
    positions: NDArray[Any]
    events: Mapping[str, NDArray[Any]]
    """One integer or boolean value per event, by name, stored in this order. An array of
    another shape, or one `event_array_problem` refuses, is rejected before caching; batch
    collation widens to int64.
    """
    truncated_events: int
    """Events whose tokens were cut to fit a per-event cap."""


def event_array_problem(dtype: np.dtype[Any]) -> str | None:
    """None if `dtype` can hold a per-event array (bool or integer, native byte order,
    not uint64); otherwise a message naming what is wrong and the fix."""
    if dtype.kind not in "biu":
        return f"has dtype {dtype}; per-event arrays need integer dtypes (or bool)"
    if dtype.kind == "u" and dtype.itemsize == 8:
        return "is uint64, which torch cannot index; use int64"
    if not dtype.isnative:
        return (
            f"is {dtype.str}, not in native byte order; convert it with "
            f".astype({dtype.newbyteorder('=').str!r})"
        )
    return None


@runtime_checkable
class ColumnarTokenizer[StatsT](Protocol):
    """A tokenizer the token-cache build runs block by block, in worker processes.

    `fit_block` sees only train-side events, and `merge` gives the same result in
    any order, so the fitted state does not depend on blocks or workers. `merge`
    may reuse its arguments' storage: neither is used afterwards. Instances and
    states must pickle, and the class must be importable by its module path.
    `tfm.testing.conformance.check_columnar_tokenizer` checks an implementation.
    """

    uses_profiles: bool
    """`fit_block` receives the block's profiles; otherwise an empty mapping."""

    @property
    def vocab_size(self) -> int:
        """Token ids the fitted tokenizer emits are below this."""
        ...

    def settings(self) -> dict[str, Any]:
        """JSON-serializable: everything, the state's format version included, that with
        the same events decides the fitted state and the tokens. Joins the cache's
        identity, so a cache built under other settings is refused."""
        ...

    def fit_block(self, block: EventBlock, profiles: Mapping[str, EntityProfile]) -> StatsT:
        """Statistics of `block`'s train-side events and of `profiles`, each as of
        the train cutoff and only for users with an event in `block`."""
        ...

    def merge(self, a: StatsT, b: StatsT) -> StatsT: ...

    def fit_stats(self, stats: StatsT) -> Self:
        """Fit from the merged statistics of every block."""
        ...

    def encode_block(self, block: EventBlock) -> EncodedBlock:
        """Every event's tokens; an event's tokens depend only on its own fields."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """The fitted state, JSON-serializable, with its format version."""
        ...

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Self:
        """A fitted tokenizer from `state_dict()`; refuses a version it cannot read."""
        ...


@runtime_checkable
class FieldNamedTokenizer(Protocol):
    """A tokenizer whose batches label each token with its field in `key_ids`.

    Per-field evaluation (`tfm.evaluation.surprise`) groups tokens by that label. The
    key-value layout labels with `KEY_` ids; a one-token-per-field layout labels with
    any id per field, such as `ValueVocabulary.encode_events`' `token_keys`.
    """

    def field_names(self) -> dict[int, str]:
        """Each `key_ids` label the tokenizer emits, with the field it names; no two
        labels share a name."""
        ...


ProfilePrefix = tuple[list[int], list[int], list[int], list[int], list[float]]
"""`(keys, values, positions, item offsets, item times)` of a profile prefix."""


@runtime_checkable
class UsesProfiles(Protocol):
    """A tokenizer that reads entity profiles; a run hands it the book's profiles."""

    def set_profiles(self, profiles: Mapping[str, EntityProfile] | None) -> None: ...


@runtime_checkable
class ProfilePrefixEncoder(UsesProfiles, Protocol):
    """A tokenizer that encodes each entity's profile as a token prefix of its sequence.

    A token cache stores events only; the cache's reader asks this for each row's prefix.
    """

    def encode_profile(self, entity_id: str, cutoff: datetime) -> ProfilePrefix:
        """The prefix of `entity_id`'s profile as known at `cutoff`."""
        ...


@runtime_checkable
class FieldVocabularyTokenizer(Protocol):
    """A `flat_fields` tokenizer whose id ranges per field a model can mask by."""

    def field_vocabulary(self) -> FieldVocabulary: ...
