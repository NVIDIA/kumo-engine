# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace

from azure.core import MatchConditions
from azure.core.exceptions import ResourceModifiedError, ResourceNotFoundError
from azure.storage.filedatalake import FileProperties, PathProperties


class FakeService:
    """Azure SDK-shaped storage using real public property models."""
    def __init__(self):
        self.files = {}
        self.uncommitted = {}
        self.directories = {'curated', 'curated/orders'}
        self.downloads = []
        self.created = []
        self.modified = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def get_file_system_client(self, filesystem):
        return FakeFilesystem(self, filesystem)


class FakeFilesystem:
    def __init__(self, service, filesystem):
        self.service = service
        self.filesystem = filesystem

    def get_file_system_properties(self):
        if self.filesystem not in self.service.directories:
            raise ResourceNotFoundError('missing filesystem')
        return {}

    def get_file_client(self, path):
        return FakeFile(self.service, f'{self.filesystem}/{path}')

    get_directory_client = get_file_client

    def get_paths(self, path=None, recursive=True):
        prefix = f'{self.filesystem}/' + (path.rstrip('/') +
                                          '/' if path else '')
        for name in sorted(set(self.service.files) | self.service.directories):
            if not name.startswith(prefix):
                continue
            relative = name[len(prefix):]
            if not relative or not recursive and '/' in relative:
                continue
            yield PathProperties(
                name=name[len(self.filesystem) + 1:], is_directory=name
                in self.service.directories,
                content_length=len(self.service.files.get(name, b'')),
                last_modified=self.service.modified)


class FakeFile:
    def __init__(self, service, name):
        self.service = service
        self.name = name

    def get_file_properties(self):
        if self.name not in self.service.files and self.name not in self.service.directories:
            raise ResourceNotFoundError('missing object')
        content = self.service.files.get(self.name, b'')
        return FileProperties(
            **{
                'name': self.name,
                'ETag': hashlib.sha256(content).hexdigest(),
                'Content-Length': len(content),
                'Last-Modified': self.service.modified,
                'metadata': {
                    'hdi_isfolder': 'true'
                } if self.name in self.service.directories else {},
            })

    def download_file(self, offset=None, length=None, **kwargs):
        assert kwargs['match_condition'] == MatchConditions.IfNotModified
        if kwargs['etag'] != self.get_file_properties().etag:
            raise ResourceModifiedError('object was changed')
        self.service.downloads.append((self.name, offset, length, kwargs))
        content = self.service.files[self.name]
        content = content[offset or 0:]
        if length is not None:
            content = content[:length]
        return SimpleNamespace(readall=lambda: content,
                               readinto=lambda stream: stream.write(content))

    def upload_data(self, stream, length, overwrite, chunk_size,
                    max_concurrency, **kwargs):
        if length == 0:
            return {}
        assert 1 <= max_concurrency <= 4
        if overwrite:
            self.create_file()
        self.service.files[self.name] = stream.read()
        self.service.uncommitted.pop(self.name, None)
        assert len(self.service.files[self.name]) == length

    def create_file(self):
        self.service.created.append(self.name)
        self.service.files[self.name] = b''
        self.service.uncommitted[self.name] = b''
        return {'etag': self.get_file_properties().etag}

    def create_directory(self):
        self.service.directories.add(self.name)

    def append_data(self, data, offset, length):
        assert length == len(data)
        assert offset == len(self.service.uncommitted[self.name])
        self.service.uncommitted[self.name] += data

    def flush_data(self, offset):
        assert offset == len(self.service.uncommitted[self.name])
        self.service.files[self.name] = self.service.uncommitted.pop(self.name)

    def delete_file(self, **kwargs):
        if kwargs.get(
                'etag') and kwargs['etag'] != self.get_file_properties().etag:
            raise ResourceModifiedError('object was changed')
        self.service.files.pop(self.name)
        self.service.uncommitted.pop(self.name, None)

    def delete_directory(self):
        self.service.files = {
            key: value
            for key, value in self.service.files.items()
            if not key.startswith(self.name + '/')
        }
        self.service.directories = {
            key
            for key in self.service.directories
            if key != self.name and not key.startswith(self.name + '/')
        }
