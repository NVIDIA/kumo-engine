#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import multiprocessing as mp
import os
import os.path as osp
import random
import shutil
import tempfile
import time
from multiprocessing.pool import Pool
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import StreamingResponse

from .common import gen_data

BUFFER_SIZE = 16 * 1024
prefix = "/Volumes/ucv_test/stress_test/5"

# N.B. initially None, lazily initialized by each process
_client: Optional[WorkspaceClient] = None


def client() -> WorkspaceClient:
    global _client
    if _client is None:
        # env:
        # - DATABRICKS_HOST
        # - DATABRICKS_AUTH_TYPE=pat
        # - DATABRICKS_TOKEN
        _client = WorkspaceClient()
    return _client


def create_file(n_bytes: int) -> None:
    path = osp.join(prefix, str(n_bytes))
    data: bytes = gen_data(n_bytes)
    assert len(data) == n_bytes
    client().files.upload(path, io.BytesIO(data))
    ret = client().files.get_metadata(path)
    if ret.content_length != n_bytes:
        message = (f"upload fault {n_bytes} on {os.getpid()}, "
                   f"observed {ret.content_length}")
        print(message)


def validate_existing(n_bytes: int):
    path = osp.join(prefix, str(n_bytes))
    ret = client().files.get_metadata(path)
    if ret.content_length != n_bytes:
        print(f"cleaning fault {n_bytes}, observed {ret.content_length}")
        client().files.delete(path)


def get_existing(pool: Pool, min_size: int, max_size: int) -> list[int]:
    def ls(path) -> list[str]:
        return [
            entry.path or ""
            for entry in client().files.list_directory_contents(path)
        ]

    existing = [int(osp.basename(p)) for p in ls(prefix)]
    print("found", len(existing), "existing files, validating")
    pool.map(validate_existing, existing)
    existing = [int(osp.basename(p)) for p in ls(prefix)]
    print("found", len(existing), "valid files")
    return [x for x in existing if min_size <= x <= max_size]


def generate_files(pool: Pool, n_files: int, min_size: int,
                   max_size: int) -> list[int]:
    print(f"file sizes [{min_size}, {max_size}]")
    assert max_size < 1024 * 1024 * 1024

    print("finding existing files")
    valid_existing = get_existing(pool, min_size, max_size)

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
    pool.map(create_file, new_sizes)
    print("done generating new files")

    return valid_existing + new_sizes


def test_download(path: str, n_bytes: int, with_gen_data: bool):
    with tempfile.NamedTemporaryFile() as tf:
        res = client().files.download(path)
        if res.contents is None:
            raise FileNotFoundError(f"no contents: {path}")
        if res.content_length != n_bytes:
            print(f"storage fault {n_bytes} on {os.getpid()} "
                  f"observed {res.content_length}")
            return

        assert isinstance(res.contents, StreamingResponse)
        res.contents.set_chunk_size(BUFFER_SIZE)
        with res.contents as remote_f:
            with open(tf.name, 'wb') as local_f:
                shutil.copyfileobj(remote_f, local_f, length=BUFFER_SIZE)

        with open(tf.name, 'rb') as f:
            actual = f.read()
        if len(actual) != n_bytes:
            print(f"download fault {n_bytes} on {os.getpid()}, "
                  f"observed {len(actual)}")
            return
        if with_gen_data and actual != gen_data(n_bytes):
            print(f"download corruption {n_bytes} on {os.getpid()}")
            return


def test_download_by_size(n_bytes: int):
    test_download(osp.join(prefix, str(n_bytes)), n_bytes, with_gen_data=True)


def find_files_recursive(path: str) -> list[tuple[str, int]]:
    contents = []
    for entry in client().files.list_directory_contents(path):
        assert entry.path is not None
        if entry.is_directory:
            contents.extend(find_files_recursive(entry.path))
        else:
            assert entry.file_size is not None
            contents.append((entry.path, entry.file_size))
    return contents


def test_download_tuple(f: tuple[str, int]):
    test_download(f[0], f[1], with_gen_data=False)


def main():
    t1 = time.time()
    pool = Pool(processes=mp.cpu_count())  # 32 in kumo testing
    sizes = generate_files(pool, n_files=1000, min_size=1024,
                           max_size=1024 * 1024)  # 1kb to 1mb
    t2 = time.time()
    print("generating files took", t2 - t1)

    # files = find_files_recursive(
    #     "/Volumes/kumo-catalog/hm/caching/test_files/")
    # print("file count:", len(files))

    n_iterations = 0
    for i in range(n_iterations):
        print("iteration", i + 1, "/", n_iterations)
        pool.map(test_download_by_size, sizes)

        # pool.map(test_download_tuple, files)

    pool.terminate()


if __name__ == "__main__":
    main()
