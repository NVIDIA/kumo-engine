# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Exact psi, ks and wasserstein between score counts on the same bins."""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from kumo_core.score_counts import Counts, bin_bounds

DRIFT_COLUMNS = ('True_PROB', 'confidence', 'top_score', 'TARGET_PRED')
PSI_SHIFTS = {'small': '< 0.1', 'moderate': '0.1–0.25', 'large': '> 0.25'}
_FLOOR = 1e-6


def axis(column: str) -> str:
    r"""The bins ``column`` is counted on."""
    return {'TARGET_PRED': 'target', 'top_score': 'top_score'}.get(
        column, 'score')


def drift_column(counts: Counts) -> Optional[str]:
    return next((name for name in DRIFT_COLUMNS if name in counts.arrays),
                None)


def score_drift(counts: Counts, reference: Counts) -> Optional[Dict[str, float]]:
    r"""None unless both sides counted the same column on the same bins."""
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


def holdout_shift(task_type: Optional[str]) -> bool:
    r"""Whether psi vs the holdout reads as a shift; a link holdout is partial."""
    return not str(task_type).endswith('link_prediction')


def psi_shift(value: float) -> str:
    r"""The `PSI_SHIFTS` shift a PSI of ``value`` is."""
    return 'small' if value < 0.1 else 'moderate' if value <= 0.25 else 'large'


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
    r"""1-Wasserstein distance between two weighted point sets."""
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
