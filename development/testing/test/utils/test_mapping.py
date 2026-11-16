# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import torch

from kumo.testing import assert_edge_index_equal
from kumo.utils.mapping import (
    MapIndex,
    get_index_map,
    map_index,
    map_values,
    node_idx_tensor,
)


def test_map_values():
    max_signed_int = 2**63 - 1
    src_data = pd.DataFrame({
        'index': [10, 20, 30, 40, 45, 50],
        'col':
        pd.Series([3, 4, 5, max_signed_int, pd.NA, 7], dtype="Int64"),
    })
    src_data.set_index('index', inplace=True)

    dst_data = pd.DataFrame({
        'index': [50, 60, 80, 90, 95, 100],
        # Cast the numbers to str, to test joining str and int
        'col': ["5", "5",
                str(max_signed_int),
                str(max_signed_int), pd.NA, "9"]
    })
    dst_data.set_index('index', inplace=True)

    row, col = map_values(src_data['col'], get_index_map(src_data),
                          dst_data['col'], get_index_map(dst_data))

    assert_edge_index_equal((row, col), [[2, 2, 3, 3], [0, 1, 2, 3]])


def test_mapping_mask():
    # Source: foreign keys (with some pointing to invalid primary keys):
    # Also cast the keys to str, to test joining float with str.
    source = pd.Series(list(range(3, 8)) + list(range(3, 5)), dtype='str')

    # Target: primary keys
    target = pd.DataFrame({'values': map(float, list(range(5)))})
    target.set_index('values', inplace=True)

    index_map = get_index_map(target)
    col, mask = map_index(source, index_map, return_mask=True)

    assert col.tolist() == [3, 4, 3, 4]
    assert mask.tolist() == [True, True, False, False, False, True, True]
    assert len(mask) == len(source)

    # Test MapIndex implementation.
    source = pd.Series(list(range(3, 8)) + list(range(3, 5)), dtype=int)
    # Target: primary keys
    target = pd.DataFrame({'values': map(int, list(range(5)))})
    target.set_index('values', inplace=True)

    index_map = MapIndex(target)
    col, mask = index_map.map_index(source, return_mask=True)

    assert col.tolist() == [3, 4, 3, 4]
    assert mask.tolist() == [True, True, False, False, False, True, True]
    assert len(mask) == len(source)


def test_mapping_mask_with_na_values():
    # Source: foreign keys (with N/A values)
    source = pd.Series([2, 3, None, 3, None, 4, 5], dtype=pd.Int64Dtype())

    # Target: primary keys
    target = pd.DataFrame({'values': list(range(3, 5))})
    target.set_index('values', inplace=True)

    index_map = get_index_map(target)
    col, mask = map_index(source, index_map, return_mask=True)

    assert col.tolist() == [0, 0, 1]
    assert mask.tolist() == [False, True, False, True, False, True, False]
    assert len(mask) == len(source)


def test_node_idx_tensor():
    from kumo_core.train.mapping import NODE_INDEX_COLUMN

    arange = np.random.permutation(5)

    index_map = pd.DataFrame(index=pd.Series(list(range(5))),
                             data={NODE_INDEX_COLUMN: arange})

    idx_tensor = node_idx_tensor(index_map)

    assert torch.equal(torch.from_numpy(arange), idx_tensor)


def test_mapping_chunked_array():
    range_limit = 1000
    prefix = "a" * 1000
    values = [prefix + str(i) for i in range(range_limit)]

    target = pd.Series(values, dtype="string[pyarrow]", name='values')
    target = target.to_frame()
    target.set_index('values', inplace=True)
    index_map = MapIndex(target)

    source = pd.Series(values, dtype="string[pyarrow]")
    source = pd.concat([source] * 1000)
    expected_col = np.tile(np.arange(range_limit), 1000)

    col, _ = index_map.map_index(source, return_mask=True)

    assert len(col) == len(source)
    assert np.array_equal(col.numpy(), expected_col)
