# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import random
from functools import partial
from multiprocessing.pool import Pool

from io_utils import fsutil


def gen_data(n_bytes: int) -> bytes:
    data = (str(n_bytes) + "data" *
            ((n_bytes + 3) // 4)).encode("utf-8")[:n_bytes]
    assert len(data) == n_bytes
    return data


def _create_file(n_bytes: int, prefix: str) -> None:
    path = osp.join(prefix, str(n_bytes))
    data: bytes = gen_data(n_bytes)
    fsutil.write_binary(path, data)


def _validate_existing(n_bytes: int, prefix: str):
    path = osp.join(prefix, str(n_bytes))
    size = fsutil.filesize(path) or 0
    if size != n_bytes:
        print(f"clearing {n_bytes}, observed {size}")
        fsutil.rm(path)


def _get_existing(pool: Pool, min_size: int, max_size: int,
                  prefix: str) -> list[int]:
    existing = [int(osp.basename(p)) for p in fsutil.listdir(prefix)]
    print("found", len(existing), "existing files, validating")
    pool.map(partial(_validate_existing, prefix=prefix), existing)
    existing = [int(osp.basename(p)) for p in fsutil.listdir(prefix)]
    print("found", len(existing), "valid files")
    return [x for x in existing if min_size <= x <= max_size]


def generate_files(pool: Pool, n_files: int, min_size: int, max_size: int,
                   prefix: str) -> list[int]:
    print(f"file sizes [{min_size}, {max_size}]")
    assert max_size < 10 * 1024 * 1024 * 1024

    print("finding existing files")
    valid_existing = _get_existing(pool, min_size, max_size, prefix)

    if len(valid_existing) == n_files:
        return valid_existing
    elif len(valid_existing) > n_files:
        return random.choices(valid_existing, k=n_files)

    def make_new_sizes(existing: list[int], n_files: int) -> list[int]:
        all_sizes: set[int] = set(existing)
        new_sizes: set[int] = set()
        while len(all_sizes) < n_files:
            val = random.randint(min_size, max_size)
            if val not in all_sizes:
                all_sizes.add(val)
                new_sizes.add(val)
        return list(new_sizes)

    print("generating", n_files - len(valid_existing), "new sizes")
    new_sizes = make_new_sizes(valid_existing, n_files)
    print("generating new files")
    pool.map(partial(_create_file, prefix=prefix), new_sizes)
    print("done generating new files")

    return valid_existing + new_sizes
