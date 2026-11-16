# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib

from kumo.data.cache.read_through_cache import ReadThroughCache


def test_put_and_get(tmp_path: pathlib.Path):
    temp_path: str = tmp_path.resolve().as_posix()
    key = "test_key"
    value = "test_value"

    cache = ReadThroughCache(temp_path, 'test1')
    cache.put(key, value)
    retrieved_value = cache.get(key, str)
    assert retrieved_value == value


def test_load(tmp_path: pathlib.Path):
    temp_path: str = tmp_path.resolve().as_posix()
    key = "test_key"
    value = "test_value"

    cache = ReadThroughCache(temp_path, 'test1')
    cache.put(key, value)
    cache._cache.clear()  # Clear cache to force loading from disk
    retrieved_value = cache.get(key, str)
    assert retrieved_value == value


def test_remove(tmp_path: pathlib.Path):
    temp_path: str = tmp_path.resolve().as_posix()
    key = "test_key"
    value = "test_value"

    cache = ReadThroughCache(temp_path, 'test1')
    cache.put(key, value)
    cache.remove(key)
    assert cache.get(key, str) is None
