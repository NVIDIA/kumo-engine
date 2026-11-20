# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Primary-key lookup preserves physical positions and missing-value codes."""

import numpy as np
import pandas as pd
import pytest
import torch
from kumo_core.encoders.index_mapping import Mapper


@pytest.mark.parametrize(
    "categories",
    [
        ["A", "B", "C"],
        pd.Index(["A", "B", "C"]),
        pd.Index(["A", "B", "C"], dtype="string"),
    ],
)
def test_numpy_lookup_matches_series(categories: list[str] | pd.Index) -> None:
    """NumPy lookups retain Series category codes, including unknown values."""
    mapper = Mapper(categories)
    queries = np.array(["A", "C", "Z", "B", "A"])

    actual = mapper.map(queries)

    assert torch.equal(actual, mapper.map(pd.Series(queries)))
    assert actual.tolist() == [0, 2, -1, 1, 0]


@pytest.mark.parametrize(
    "index,expected",
    [
        (pd.Index([3, 1, 2]), [0, 1, -1]),
        (pd.Index([3, 1, 2], dtype=object), [0, 1, -1]),
        (pd.Index([3, 1, 2], dtype="Int64"), [0, 1, -1]),
        (pd.CategoricalIndex([3, 1, 2], categories=[1, 2, 3]), [0, 1, -1]),
        (pd.RangeIndex(1, 4), [2, 0, -1]),
    ],
)
def test_numpy_lookup_preserves_positions(index: pd.Index, expected: list[int]) -> None:
    """Lookup returns physical positions rather than categorical dictionary codes."""
    actual = Mapper(index).map(np.array([3, 1, 4]))

    assert actual.tolist() == expected
