# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest
from kumo_core.score_counts import PROBABILITY_EDGES, Counts, merged
from kumo_core.summaries import Histogram


def _brute_histogram(values, edges, weights=None):
    counts = [0.0] * (len(edges) + 1)
    for i, value in enumerate(values):
        if np.isfinite(value):
            index = sum(edge <= value for edge in edges)
            counts[index] += 1 if weights is None else weights[i]
    return counts


def test_batches_and_workers_add_up_to_the_brute_force_counts():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.random(997), [0.0, 0.5, 1.0, np.nan]])
    labels = rng.integers(0, 3, values.size)
    batches = []
    for part in np.array_split(np.arange(values.size), 7):
        counts = Counts({'score': PROBABILITY_EDGES})
        counts.histogram('score', 'score', values[part])
        counts.histogram('sum', 'score', values[part], weights=values[part])
        counts.tally('label', labels[part], 3)
        counts.total('rows', part.size)
        counts.extent('score', values[part])
        batches.append(Counts.from_dict(json.loads(json.dumps(
            counts.to_dict()))))

    total = merged(batches)
    assert total.arrays['score'].tolist() == _brute_histogram(
        values, PROBABILITY_EDGES)
    assert total.arrays['sum'] == pytest.approx(_brute_histogram(
        values, PROBABILITY_EDGES, values))
    assert total.arrays['label'].tolist() == np.bincount(labels).tolist()
    assert float(total.arrays['rows']) == values.size
    assert total.extents['score'] == (np.nanmin(values), np.nanmax(values))
    assert total.summary('score', 'score', 'pred/True_PROB/hist') == Histogram(
        'pred/True_PROB/hist', PROBABILITY_EDGES,
        tuple(_brute_histogram(values, PROBABILITY_EDGES)))


def test_counts_over_other_bins_or_classes_are_refused():
    counts = Counts({'score': PROBABILITY_EDGES})
    with pytest.raises(ValueError, match='different bins'):
        counts.merge(Counts({'score': (0.5,)}))
    with pytest.raises(ValueError, match=r'in \[0, 3\)'):
        counts.tally('label', [0, 3], 3)
