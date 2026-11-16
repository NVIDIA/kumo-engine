# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only Azure HNS checks; requires an explicitly selected existing directory."""

import os
from itertools import islice
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import pytest
from kumo_adls.backend import ADLSSourceReader
from kumo_adls.sdk_filesystem import ADLSFileSystem
from kumo_adls.session import session_for
from kumoapi.adls import normalize_adls_root_uri
from pyarrow.fs import FileType, FSSpecHandler, PyFileSystem

ROOT = os.environ.get('KUMO_ADLS_LIVE_ROOT')
if not ROOT:
    pytest.skip(
        'Set KUMO_ADLS_LIVE_ROOT to an existing HNS directory; '
        'live storage behavior was not validated.', allow_module_level=True)


@pytest.fixture
def live_directory():
    parsed = urlsplit(normalize_adls_root_uri(ROOT))
    _, host = parsed.netloc.split('@', 1)
    account, suffix = host.split('.', 1)
    filesystem = ADLSFileSystem(session_for(None, account, suffix), account,
                                suffix)
    path = parsed.netloc + unquote(parsed.path).rstrip('/')
    return filesystem, path


def test_existing_hns_directory_is_listable(live_directory):
    filesystem, path = live_directory
    assert filesystem.info(path)['type'] == 'directory'
    assert isinstance(filesystem.ls(path), list)


def test_missing_hns_path_is_not_found_for_fsspec_and_arrow(live_directory):
    filesystem, path = live_directory
    missing = path + '/__kumo_missing_' + uuid4().hex
    # Both names are absent; no Azure paths are created by this test.
    for name in (missing, missing + '/child'):
        with pytest.raises(FileNotFoundError):
            filesystem.info(name)
        with pytest.raises(FileNotFoundError):
            filesystem.ls(name)
        arrow = PyFileSystem(FSSpecHandler(filesystem))
        assert arrow.get_file_info(name).type == FileType.NotFound


def test_listing_identities_match_heads_and_conditional_reads(live_directory):
    filesystem, path = live_directory
    client, key, authority = filesystem._address(path)
    if not key:
        pytest.skip(
            'Listing identities require a directory below the filesystem root')
    paths = []
    for item in islice(client.get_paths(path=key or None, recursive=False),
                       1024):
        if not item.is_directory:
            paths.append(f'{authority}/{item.name}')
            if len(paths) == 2:
                break
    if len(paths) < 2:
        pytest.skip(
            'Live identity check needs two files among the first 1024 entries')

    # Require usable live listing metadata, rather than silently taking HEAD fallback.
    listed = filesystem.object_identities(paths, root=path)
    assert listed is not None
    heads = [filesystem.object_identity(name) for name in paths]
    assert listed == heads
    arrow = PyFileSystem(FSSpecHandler(filesystem))
    reader = ADLSSourceReader(arrow, root=path,
                              object_identity=filesystem.object_identity)
    found = arrow.get_file_info(paths)
    assert reader._identities(found, 1) == heads
    assert filesystem.object_identities(paths[:1], root=path) is None
    assert reader._identities(found[:1], 1) == heads[:1]
    for name, identity in zip(paths, heads):
        with filesystem.open(name, 'rb', cache_type='none') as stream:
            raw = stream.etag
            token = raw[1:-1] if raw.startswith('"') and raw.endswith(
                '"') else raw
            assert token == identity[0]
            assert len(stream.read(4)) == min(4, int(identity[-1]))
