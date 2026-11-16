# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from kumo_core.score_counts import Counts
from kumo_core.top_scores import TopScoreSpill, top_score_counts


def _in_memory(scores):
    r"""The in-memory implementation the spill replaced."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    edges = () if not scores.size else tuple(np.unique(np.quantile(
        scores, np.linspace(0, 1, 101))).tolist())
    counts = Counts({'top_score': edges})
    counts.histogram('top_score', 'top_score', scores)
    counts.extent('top_score', scores)
    return counts


_rng = np.random.default_rng(0)


@pytest.mark.parametrize('scores', [
    _rng.normal(size=10_007),
    _rng.normal(size=5_000).astype(np.float32),
    np.concatenate([_rng.integers(0, 7, 3_000) / 7, [np.nan, np.inf, -0.0]]),
    np.full(50, 0.25),
    np.array([1.5]),
    _rng.lognormal(0, 3, 2_001) * _rng.choice([-1, 1], 2_001),
], ids=['normal', 'float32', 'ties', 'constant', 'one', 'wide'])
def test_spilled_counts_equal_the_in_memory_ones(tmp_path, scores):
    spill = TopScoreSpill(str(tmp_path / 'top.parquet'))
    for rows in np.array_split(np.arange(scores.size), 7):
        spill.write(rows, scores[rows])

    counts, expected = spill.counts(), _in_memory(scores)

    assert counts.edges == expected.edges
    assert counts.arrays['top_score'].tolist() == \
        expected.arrays['top_score'].tolist()
    assert counts.extents == expected.extents
    assert not (tmp_path / 'top.parquet').exists()


def test_nothing_spilled_is_no_counts(tmp_path):
    spill = TopScoreSpill(str(tmp_path / 'top.parquet'))
    assert spill.counts() is None
    assert top_score_counts(None).arrays['top_score'].tolist() == [0]
