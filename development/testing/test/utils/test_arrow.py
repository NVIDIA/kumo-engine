# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa

from kumo.testing.test import onlyFullTest
from kumo.utils.arrow import chunked_take


def test_chunked_take_small():
    arr = pa.chunked_array([[1, 2], [3, 4]])
    idx = [3, 0, 3, 1, 2]
    exp = pa.array([4, 1, 4, 2, 3])
    arrow_take_res = arr.take(idx).combine_chunks()
    assert (exp.equals(arrow_take_res))
    chunked_take_res = chunked_take(arr, idx).combine_chunks()
    assert (exp.equals(chunked_take_res))


@onlyFullTest
def test_chunked_take_large():
    string_len = 2**18
    big_string = "*" * string_len

    # data larger than int32max is chunked
    array_len = (2**31) // string_len
    series = pd.Series([big_string]).repeat(array_len)
    array = pa.Array.from_pandas(series)
    assert (isinstance(array, pa.ChunkedArray))

    idx = [array_len - 1, array_len - 10, 100, array_len - 1, 0, 1]
    exp = pa.array([big_string for _ in idx])

    try:
        has_thrown = False
        # concatenation isn't possible due to overflow
        array.combine_chunks()
    except pa.lib.ArrowInvalid:
        has_thrown = True
    finally:
        assert (has_thrown)

    try:
        has_thrown = False
        # take calls concatenate, and is also not possible
        array.take(idx)
    except pa.lib.ArrowInvalid:
        has_thrown = True
    finally:
        assert (has_thrown)

    ls_array = array.cast(pa.large_string())
    ls_res = ls_array.take(idx).combine_chunks().cast(pa.string())
    ls_res.to_numpy(zero_copy_only=False)
    assert (exp.equals(ls_res))

    c_res = chunked_take(array, idx)
    assert (isinstance(c_res, pa.lib.ChunkedArray))
    c_res.to_numpy()
    c_res_flat = c_res.combine_chunks()
    c_res_flat.to_numpy(zero_copy_only=False)
    assert (exp.equals(c_res_flat))
