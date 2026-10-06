# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.errors import TransientSourceError
from kumo_core.sources.object_store import ObjectStoreSourceReader
from kumo_core.sources.types import ReadMode
from pyarrow.fs import FileInfo, FileType, LocalFileSystem


@pytest.mark.parametrize('valid_size', [True, False])
def test_decoded_parquet_download_validates_local_size(tmp_path, valid_size):
    source = tmp_path / 'table.parquet'
    pq.write_table(pa.table({'id': [1, 2]}), source)
    size = source.stat().st_size

    class EncodedFilesystem:
        def get_file_info(self, path):
            info = LocalFileSystem().get_file_info(path)
            if isinstance(info, list):
                return [self._encoded(entry) for entry in info]
            return self._encoded(info)

        @staticmethod
        def _encoded(info):
            if info.type == FileType.File:
                return FileInfo(info.path, FileType.File, size=size // 2,
                                mtime=info.mtime)
            return info

    def fetch(path, target):
        shutil.copyfile(path, target)
        return size if valid_size else size + 1

    reader = ObjectStoreSourceReader(EncodedFilesystem(), root=str(tmp_path),
                                     fetch=fetch)
    destination = tmp_path / 'output'
    if not valid_size:
        with pytest.raises(TransientSourceError, match='downloaded'):
            reader.prepare('table.parquet', str(destination), columns=None,
                           mode=ReadMode.STREAMED, version='v1')
        assert not destination.exists()
    else:
        reader.prepare('table.parquet', str(destination), columns=None,
                       mode=ReadMode.STREAMED, version='v1')
        assert pq.read_table(destination).to_pydict() == {'id': [1, 2]}
