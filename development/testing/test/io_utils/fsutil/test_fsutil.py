# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import pathlib

import pytest

from io_utils import fsutil


@pytest.fixture
def s3_scratch_prefix():
    """Unique S3 prefix per test, deleted after yield. Skips if no
    AWS creds — hand-rolled to keep this fixture independent of REST app
    initialization.
    """
    import uuid
    if not fsutil.exists('s3://kumo-public-datasets/financial/csv/'):
        pytest.skip('No AWS credentials found for S3')
    prefix = ('s3://kumo-test-dataplane-bucket/fsutil-test/'
              f'copy_tree_threaded_{uuid.uuid4().hex[:12]}')
    yield prefix
    if fsutil.exists(prefix):
        fsutil.rm(prefix)


def test_copy_tree_threaded_local_to_s3(tmp_path: pathlib.Path,
                                        s3_scratch_prefix: str):
    """Local→S3 upload: tree preservation + predicate exclusion against
    the real S3 transport. (The local→local parent-dir bug was invisible
    to S3, so cross-backend coverage is load-bearing.)"""
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'top.txt').write_bytes(b'TOP')
    (src / 'manifest.json').write_bytes(b'COMMIT')
    (src / 'sub1').mkdir()
    (src / 'sub1' / 'a.parquet').write_bytes(b'A')
    deep = src / 'sub2' / 'deep'
    deep.mkdir(parents=True)
    (deep / 'leaf.txt').write_bytes(b'LEAF')

    fsutil.copy_tree_threaded(
        str(src),
        s3_scratch_prefix,
        exclude=lambda p: osp.basename(p) == 'manifest.json',
    )

    assert fsutil.read_binary(osp.join(s3_scratch_prefix, 'top.txt')) == b'TOP'
    assert fsutil.read_binary(osp.join(s3_scratch_prefix, 'sub1',
                                       'a.parquet')) == b'A'
    assert fsutil.read_binary(
        osp.join(s3_scratch_prefix, 'sub2', 'deep', 'leaf.txt')) == b'LEAF'
    assert not fsutil.exists(osp.join(s3_scratch_prefix, 'manifest.json'))
