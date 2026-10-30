# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Exact counts of scores, accumulated batch by batch, for evaluation and
prediction summaries.

Bins are fixed before counting: probabilities in hundredths, one bin per
class, or a numerical target's training statistics. A value ``v`` falls in
bin ``i`` when ``edges[i - 1] <= v < edges[i]``: bin 0 holds values below
``edges[0]`` and the last bin values at or above ``edges[-1]``. Counts from
batches, workers or runs add up exactly.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from kumo_core.summaries import Histogram

PROBABILITY_EDGES = tuple(i / 100 for i in range(101))


@dataclass
class Counts:
    r"""Named arrays of exact counts and sums over named bin ``edges``, and
    named ``extents``, the exact minimum and maximum of some values."""

    edges: Dict[str, Tuple[float, ...]]
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    extents: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def histogram(self, name: str, axis: str, values: Any,
                  weights: Any = None) -> None:
        r"""Adds the finite ``values`` (or their ``weights``) per bin of
        ``axis``."""
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        edges = self.edges[axis]
        self._add(name, np.bincount(
            np.searchsorted(edges, values[finite], side='right'),
            weights=None if weights is None else np.asarray(
                weights, dtype=np.float64)[finite],
            minlength=len(edges) + 1))

    def tally(self, name: str, indices: Any, size: int) -> None:
        r"""Adds one per index in ``[0, size)``."""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size and not 0 <= indices.min() <= indices.max() < size:
            raise ValueError(f'{name} indices must be in [0, {size})')
        self._add(name, np.bincount(indices, minlength=size))

    def total(self, name: str, value: float) -> None:
        self._add(name, np.asarray(float(value)))

    def extent(self, name: str, values: Any) -> None:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            self._widen(name, (float(values.min()), float(values.max())))

    def merge(self, other: 'Counts') -> 'Counts':
        if other.edges != self.edges:
            raise ValueError('Counts over different bins do not add up')
        merged = Counts(dict(self.edges), dict(self.arrays),
                        dict(self.extents))
        for name, array in other.arrays.items():
            merged._add(name, array)
        for name, extent in other.extents.items():
            merged._widen(name, extent)
        return merged

    def summary(self, name: str, axis: str,
                label: Optional[str] = None) -> Histogram:
        r"""The ``name`` counts as a `Histogram` named ``label``, within
        their extent if it was recorded."""
        return Histogram(label or name, self.edges[axis],
                         tuple(self.arrays[name].tolist()),
                         self.extents.get(name))

    def to_dict(self) -> Dict[str, Any]:
        return {'edges': {name: list(edges)
                          for name, edges in self.edges.items()},
                'arrays': {name: array.tolist()
                           for name, array in self.arrays.items()},
                'extents': {name: list(extent)
                            for name, extent in self.extents.items()}}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'Counts':
        return cls({name: tuple(edges)
                    for name, edges in value['edges'].items()},
                   {name: np.asarray(array)
                    for name, array in value['arrays'].items()},
                   {name: (extent[0], extent[1])
                    for name, extent in value.get('extents', {}).items()})

    def _add(self, name: str, array: np.ndarray) -> None:
        current = self.arrays.get(name)
        if current is not None and current.shape != array.shape:
            raise ValueError(f'{name} has {current.shape} bins, not '
                             f'{array.shape}')
        self.arrays[name] = array if current is None else current + array

    def _widen(self, name: str, extent: Tuple[float, float]) -> None:
        current = self.extents.get(name, extent)
        self.extents[name] = (min(current[0], extent[0]),
                              max(current[1], extent[1]))


def bin_bounds(edges: Sequence[float],
               extent: Optional[Tuple[float, float]] = None) -> np.ndarray:
    r"""Each bin's ends, ``edges`` with the open end bins closed at
    ``extent``, the counted values' minimum and maximum, or else at the end
    edges: bin ``i`` spans ``bounds[i]`` to ``bounds[i + 1]``."""
    low, high = extent or (edges[0], edges[-1])
    return np.array([min(low, edges[0]), *edges, max(high, edges[-1])])


def merged(counts: Sequence[Counts]) -> Optional[Counts]:
    r"""The sum of ``counts``; ``None`` for none."""
    total: Optional[Counts] = None
    for item in counts:
        total = item if total is None else total.merge(item)
    return total
