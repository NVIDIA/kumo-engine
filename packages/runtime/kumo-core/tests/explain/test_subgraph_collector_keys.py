# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pandas as pd
import pytest
from kumo_core.explain.subgraph_collector import _set_column_values


@pytest.mark.parametrize('keys', [[297, 0], ['b', 'a']])
def test_object_keys_fill_a_numeric_column_without_warning(
        keys: list[object]) -> None:
    # DiskGraph attaches primary keys as an object array.
    df = pd.DataFrame({'movie_id': np.full(3, -1, dtype=np.int64)})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _set_column_values(df, 'movie_id', np.array([0, 2]),
                           np.array(keys, dtype=object))
    assert df['movie_id'].dtype == object
    assert df['movie_id'].tolist() == [keys[0], -1, keys[1]]


def test_numeric_keys_keep_a_numeric_column() -> None:
    df = pd.DataFrame({'movie_id': np.full(3, -1, dtype=np.int64)})
    _set_column_values(df, 'movie_id', np.array([1]), np.array([7]))
    assert df['movie_id'].dtype == np.int64
    assert df['movie_id'].tolist() == [-1, 7, -1]
