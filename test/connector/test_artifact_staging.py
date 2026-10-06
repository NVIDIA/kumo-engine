# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from test.connector.artifact_staging_contract import local_artifact_dataset

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.connector.artifact_staging import (
    is_local_artifact_path,
    local_parquet_files,
)


def test_local_parquet_files_accepts_file_and_nested_dataset(tmp_path):
    single = tmp_path / 'predictions.parquet'
    pq.write_table(pa.table({'ID': [1]}), single)
    assert local_parquet_files(str(single)) == [single.resolve()]

    dataset = local_artifact_dataset(
        tmp_path, shards=2, embeddings=True, nested=True)
    (dataset / '_SUCCESS').write_text('')

    assert local_parquet_files(str(dataset)) == sorted(
        path.resolve() for path in dataset.rglob('*.parquet'))


@pytest.mark.parametrize(
    ('symlink_location', 'message'),
    [('root', 'may not be symlinks'), ('nested', 'may not contain symlinks')],
)
def test_local_parquet_files_rejects_symlinks(
        tmp_path, symlink_location, message):
    target = tmp_path / 'outside.parquet'
    pq.write_table(pa.table({'ID': [1]}), target)
    if symlink_location == 'root':
        artifact = tmp_path / 'link.parquet'
    else:
        artifact = local_artifact_dataset(tmp_path)
        artifact = artifact / 'linked-part.parquet'
    os.symlink(target, artifact)

    with pytest.raises(ValueError, match=message):
        local_parquet_files(
            str(artifact if symlink_location == 'root' else artifact.parent))


def test_is_local_artifact_path_rejects_provider_and_unknown_schemes():
    assert is_local_artifact_path('/tmp/predictions.parquet')
    assert not is_local_artifact_path('s3://bucket/predictions/')
    assert not is_local_artifact_path('gs://bucket/predictions/')
    assert not is_local_artifact_path('https://example/predictions.parquet')
