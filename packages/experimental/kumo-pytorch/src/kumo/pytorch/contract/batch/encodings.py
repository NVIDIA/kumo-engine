# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit transformations of typed sampled values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True)
class Raw:
    """Preserve the sampled value's declared type, without casting.

    Accepts every scalar dtype. Inspectable output keeps strings, bytes, dates,
    timestamps and decimals as Arrow arrays; ``output_format="model"`` refuses
    them before fetching a batch.
    """


@dataclass(frozen=True)
class Vocabulary:
    """Whole-value category IDs, learned or supplied as typed value/ID pairs.

    Accepts every scalar dtype and keeps typed identity: ``(12, 2)`` declares
    integer 12 and ``("12", 2)`` a string. Positive and negative float zero are
    one category; non-finite floats are refused. Learned entries order by
    descending frequency, then canonical typed value, and take the lowest IDs
    not reserved.

    :param values: supplied ``(value, id)`` pairs; IDs are unique and distinct
        from the missing and unknown IDs; NULL is not a category.
    :param max_size: limit on ordinary learned entries.
    :param min_frequency: fewest observations for a learned entry.
    :param missing_id: ID for NULL.
    :param unknown_id: ID for a value not in the vocabulary.
    """

    values: tuple[tuple[Any, int], ...] | None = None
    max_size: int | None = None
    min_frequency: int = 1
    missing_id: int = 0
    unknown_id: int = 1

    def __post_init__(self) -> None:
        if self.max_size is not None and self.max_size <= 0:
            raise ValueError("vocabulary max_size must be positive")
        if self.min_frequency <= 0:
            raise ValueError("vocabulary min_frequency must be positive")
        roles = (self.missing_id, self.unknown_id)
        if (
            any(type(value) is not int or not 0 <= value < 2**63 for value in roles)
            or len(set(roles)) != 2
        ):
            raise ValueError(
                "vocabulary missing and unknown IDs must be distinct non-negative integers"
            )
        if self.values is not None:
            if not isinstance(self.values, (tuple, list)):
                raise ValueError("vocabulary values require typed value/ID pairs")
            entries = tuple(tuple(value) for value in self.values)
            if any(len(value) != 2 for value in entries):
                raise ValueError("vocabulary entries require a value and an ID")
            ids = [value[1] for value in entries]
            if any(
                type(value) is not int or not 0 <= value < 2**63 or value in roles for value in ids
            ):
                raise ValueError("vocabulary IDs must be non-negative and cannot use reserved IDs")
            if len(ids) != len(set(ids)):
                raise ValueError("vocabulary IDs must be unique")
            object.__setattr__(self, "values", tuple(sorted(entries, key=lambda value: value[1])))


@dataclass(frozen=True)
class ZScore:
    """Population-standardized numbers; NULL remains an invalid value slot.

    Accepts integers, floats and decimals. Fitting ignores NULL and refuses
    non-finite values; an empty population gives mean 0 and scale 1, a constant
    one scale 1. Statistics use a binary64 working domain, so wide integers and
    decimals may round.

    :param mean: supplied mean instead of fitting.
    :param scale: supplied scale instead of fitting.
    :param dtype: ``"float32"`` or ``"float64"`` output.
    """

    mean: float | None = None
    scale: float | None = None
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if (self.mean is None) != (self.scale is None):
            raise ValueError("z-score state must supply both mean and scale")
        if self.mean is not None and (
            not math.isfinite(self.mean) or not math.isfinite(self.scale) or self.scale <= 0
        ):
            raise ValueError("z-score state requires finite mean and positive finite scale")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("z-score output dtype must be float32 or float64")


@dataclass(frozen=True)
class Buckets:
    """Numeric range IDs; equality with a cut enters the higher range.

    Supply strictly increasing boundaries, or bins for learned quantile cuts.
    Separate zero is an ordinary bucket, never the missing role.

    Accepts integers, floats and decimals. Missing is ID 0 and ranges start at
    1. Supplied cuts must be finite and stay distinct after conversion to
    binary64. Learned cuts interpolate linearly and drop duplicates.
    """

    bins: int | None = None
    boundaries: tuple[float, ...] | None = None
    separate_zero: bool = False

    def __post_init__(self) -> None:
        if (self.bins is None) == (self.boundaries is None):
            raise ValueError("buckets require either bins or boundaries, not both")
        if self.bins is not None and (type(self.bins) is not int or self.bins < 2):
            raise ValueError("learned buckets require at least two bins")
        if self.boundaries is not None:
            values = tuple(float(value) for value in self.boundaries)
            if any(not math.isfinite(value) for value in values):
                raise ValueError("bucket boundaries must be finite")
            if tuple(sorted(set(values))) != values:
                raise ValueError("bucket boundaries must be strictly increasing")
            object.__setattr__(self, "boundaries", values)


@dataclass(frozen=True)
class Text:
    """Byte-level BPE pieces, optionally pinned by complete tokenizer JSON.

    Accepts strings only. Piece offsets record which value produced which
    pieces; NULL is a separate invalid value.

    :param empty: ``"empty"`` emits no pieces for an empty string;
        ``"unknown"`` emits the unknown ID.
    :param state: complete tokenizer JSON; it defines the tokenizer entirely,
        and the missing and unknown IDs are validated against it.
    """

    vocabulary_size: int = 8_000
    min_frequency: int = 1
    empty: str = "empty"
    state: str | None = None
    missing_id: int = 0
    unknown_id: int = 1

    def __post_init__(self) -> None:
        if self.vocabulary_size <= 0 or self.min_frequency <= 0:
            raise ValueError("text vocabulary size and minimum frequency must be positive")
        if self.empty not in {"empty", "unknown"}:
            raise ValueError("text empty policy must be empty or unknown")
        if self.state is not None and not self.state:
            raise ValueError("text state must contain complete tokenizer JSON")
        if (
            any(
                type(value) is not int or not 0 <= value < 2**63
                for value in (self.missing_id, self.unknown_id)
            )
            or self.missing_id == self.unknown_id
        ):
            raise ValueError("text missing and unknown IDs must be distinct non-negative integers")
        if self.state is None and (self.missing_id, self.unknown_id) != (0, 1):
            raise ValueError(
                "learned Text uses missing_id=0 and unknown_id=1; supply state for other IDs"
            )


ScalarEncoding: TypeAlias = Raw | Vocabulary | ZScore | Buckets | Text


@dataclass(frozen=True)
class UseEncoding:
    """Use one named encoder in BatchSpec.encodings, fitted once for its uses.

    Sharing needs identical declared dtypes, decimal parameters included.
    Each input use contributes its observations once; targets reuse the mapping
    without adding observations. Equal inline encoders, even the same Python
    object, are never shared.

    :param name: a key of ``BatchSpec.encodings``.
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an encoding reference requires a name")


ValueEncoding: TypeAlias = ScalarEncoding | UseEncoding
