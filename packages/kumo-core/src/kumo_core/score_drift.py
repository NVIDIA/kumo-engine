# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Exact drift between two runs' score counts over the same bins.

``psi`` compares the shares of up to 10 groups of bins, cut where the
reference side's cumulative share reaches each decile; ``ks`` is the largest
gap between the two cumulative shares at the bin edges; ``wasserstein`` puts
each bin's count at its midpoint, the end bins closed at each side's recorded
minimum and maximum.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from kumo_core.score_counts import Counts, bin_bounds

#: The score column a run's drift is measured on, the first it counted.
DRIFT_COLUMNS = ('True_PROB', 'confidence', 'top_score', 'TARGET_PRED')
_FLOOR = 1e-6


def axis(column: str) -> str:
    r"""The bins ``column`` is counted on."""
    return {'TARGET_PRED': 'target', 'top_score': 'top_score'}.get(
        column, 'score')


def drift_column(counts: Counts) -> Optional[str]:
    return next((name for name in DRIFT_COLUMNS if name in counts.arrays),
                None)


def score_drift(counts: Counts, reference: Counts) -> Optional[Dict[str, float]]:
    r"""``psi``, ``ks`` and ``wasserstein`` of ``counts`` against
    ``reference``; ``None`` unless both counted the same column on the same
    bins."""
    column = drift_column(counts)
    if column is None or column not in reference.arrays:
        return None
    edges = counts.edges.get(axis(column))
    if edges is None or reference.edges.get(axis(column)) != edges:
        return None
    actual, expected = counts.arrays[column], reference.arrays[column]
    if not actual.sum() or not expected.sum():
        return None
    return {
        'psi': psi(actual, expected),
        'ks': ks(actual, expected),
        'wasserstein': wasserstein(
            actual, _midpoints(edges, counts.extents.get(column)),
            expected, _midpoints(edges, reference.extents.get(column))),
    }


def psi(actual: np.ndarray, expected: np.ndarray) -> float:
    groups = _decile_groups(expected)
    p = _shares(np.bincount(groups, weights=actual))
    q = _shares(np.bincount(groups, weights=expected))
    return float(np.sum((p - q) * np.log(p / q)))


def ks(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(np.cumsum(actual) / actual.sum() -
                               np.cumsum(expected) / expected.sum())))


def wasserstein(actual: np.ndarray, actual_points: np.ndarray,
                expected: np.ndarray, expected_points: np.ndarray) -> float:
    r"""The 1-Wasserstein distance between the counts ``actual`` at
    ``actual_points`` and ``expected`` at ``expected_points``."""
    points = np.concatenate([actual_points, expected_points])
    mass = np.concatenate([actual / actual.sum(), -expected / expected.sum()])
    order = np.argsort(points, kind='stable')
    gap = np.cumsum(mass[order])[:-1]
    return float(np.sum(np.abs(gap) * np.diff(points[order])))


def _decile_groups(reference: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(reference) / reference.sum()
    cuts = {int(np.searchsorted(cumulative, k / 10)) for k in range(1, 10)}
    return np.searchsorted(sorted(cut for cut in cuts
                                  if cut < reference.size - 1),
                           np.arange(reference.size), side='left')


def _shares(counts: np.ndarray) -> np.ndarray:
    return np.maximum(counts / counts.sum(), _FLOOR)


def _midpoints(edges: Sequence[float],
               extent: Optional[Tuple[float, float]]) -> np.ndarray:
    bounds = bin_bounds(edges, extent)
    return (bounds[:-1] + bounds[1:]) / 2
