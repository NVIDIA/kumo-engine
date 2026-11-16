# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from test.rest.integ import public_response_reader as reader


def test_https_reader_never_passes_signed_url_to_fsutil(monkeypatch) -> None:
    expected = b'parquet-bytes'
    monkeypatch.setattr(reader, 'urlopen',
                        lambda _url, timeout: BytesIO(expected))
    monkeypatch.setattr(
        reader.fsutil, 'isdir', lambda _path:
        (_ for _ in ()).throw(AssertionError('signed URL reached fsutil')))

    url = 'https://signed.example.test/table.parquet?signature=secret'
    assert reader.public_parquet_refs(url) == [url]
    assert reader.read_public_response_bytes(url) == expected


def test_local_file_and_directory_reader(tmp_path) -> None:
    first = tmp_path / 'first.parquet'
    second = tmp_path / 'second.parquet'
    ignored = tmp_path / 'manifest.json'
    first.write_bytes(b'first')
    second.write_bytes(b'second')
    ignored.write_text('{}')

    refs = reader.public_parquet_refs(str(tmp_path))

    assert set(refs) == {str(first), str(second)}
    assert reader.read_public_response_bytes(str(first)) == b'first'


def test_multiple_signed_partitions_are_not_dropped() -> None:
    urls = [
        'https://signed.example.test/part-0.parquet?signature=one',
        'https://signed.example.test/part-1.parquet?signature=two',
    ]

    flattened = reader.flatten_public_parquet_refs(urls)

    assert flattened == urls
