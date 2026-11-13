# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pytest
from kumo_core.score_counts import PROBABILITY_EDGES, Counts
from kumo_core.score_drift import psi_shift, score_drift
from scipy.stats import wasserstein_distance


def _counts(values, axis, edges, column):
    counts = Counts({axis: edges})
    counts.histogram(column, axis, values)
    counts.extent(column, values)
    return counts


def _brute(actual, expected, edges):
    def bins(values):
        return [sum(edge <= value for edge in edges) for value in values]

    a, e = bins(actual), bins(expected)
    size = len(edges) + 1
    cuts, seen = [], 0
    for decile in range(1, 10):
        while sum(x <= seen for x in e) / len(e) < decile / 10:
            seen += 1
        if seen < size - 1 and seen not in cuts:
            cuts.append(seen)
    group = [sum(cut < b for cut in cuts) for b in range(size)]
    p = [max(sum(group[x] == g for x in a) / len(a), 1e-6)
         for g in range(len(cuts) + 1)]
    q = [max(sum(group[x] == g for x in e) / len(e), 1e-6)
         for g in range(len(cuts) + 1)]
    ks = max(abs(sum(x <= b for x in a) / len(a) -
                 sum(x <= b for x in e) / len(e)) for b in range(size))

    def midpoints(values):
        bounds = [min(min(values), edges[0]), *edges,
                  max(max(values), edges[-1])]
        return [(bounds[b] + bounds[b + 1]) / 2 for b in bins(values)]

    return {'psi': sum((x - y) * math.log(x / y) for x, y in zip(p, q)),
            'ks': ks,
            'wasserstein': wasserstein_distance(midpoints(actual),
                                                midpoints(expected))}


@pytest.mark.parametrize('column, axis, edges', [
    ('True_PROB', 'score', PROBABILITY_EDGES),
    ('TARGET_PRED', 'target', (-1.0, 0.0, 0.5, 1.0, 2.0)),
])
def test_drift_matches_a_brute_force_pass(column, axis, edges):
    rng = np.random.default_rng(0)
    scale = 1 if axis == 'score' else 3
    expected = np.clip(rng.beta(2, 5, 400) * scale, 0, None)
    actual = np.clip(rng.beta(3, 4, 300) * scale, 0, None)
    actual[0] = scale

    drift = score_drift(_counts(actual, axis, edges, column),
                        _counts(expected, axis, edges, column))

    assert drift == pytest.approx(_brute(actual, expected, edges))


def test_drift_needs_the_same_column_on_the_same_bins():
    values = np.linspace(0, 1, 50)
    probabilities = _counts(values, 'score', PROBABILITY_EDGES, 'True_PROB')
    assert score_drift(probabilities,
                       _counts(values, 'score', (0.5,), 'True_PROB')) is None
    assert score_drift(probabilities,
                       _counts(values, 'score', PROBABILITY_EDGES,
                               'confidence')) is None
    assert score_drift(probabilities, probabilities) == {
        'psi': 0.0, 'ks': 0.0, 'wasserstein': 0.0}


def test_psi_shift_follows_the_rule_of_thumb():
    assert [psi_shift(value) for value in (0, 0.099, 0.1, 0.25, 0.251)] == [
        'small', 'small', 'moderate', 'moderate', 'large']
