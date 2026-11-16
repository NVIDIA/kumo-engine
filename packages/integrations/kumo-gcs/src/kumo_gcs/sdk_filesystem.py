# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from google.cloud import storage

from .errors import translated


class _RangeReadFile(AbstractBufferedFile):
    def __init__(self, filesystem, path, blob, **kwargs):
        self._blob = blob
        super().__init__(filesystem, path, mode='rb', size=blob.size, **kwargs)

    def _fetch_range(self, start, end):
        if end <= start:
            return b''
        with translated(self.path):
            return self._blob.download_as_bytes(start=start, end=end - 1,
                                                raw_download=True,
                                                checksum=None)


class GCSReadFileSystem(AbstractFileSystem):
    protocol = 'kumo-gcs'
    # Each instance owns an explicit request identity and billing project.
    cachable = False

    def __init__(self, client: storage.Client, requester_pays_project: str):
        super().__init__()
        self.client = client
        self.requester_pays_project = requester_pays_project

    def _bucket_key(self, path):
        path = self._strip_protocol(path).strip('/')
        bucket, _, key = path.partition('/')
        if not bucket:
            raise ValueError('GCS source must name a bucket')
        return self.client.bucket(bucket,
                                  user_project=self.requester_pays_project), key

    @staticmethod
    def _entry(bucket, blob):
        return {
            'name': f'{bucket.name}/{blob.name}',
            'size': blob.size,
            'type': 'file',
            'mtime': blob.updated,
        }

    def info(self, path, **kwargs):
        path = self._strip_protocol(path).strip('/')
        bucket, key = self._bucket_key(path)
        with translated(path):
            blob = bucket.get_blob(key) if key else None
            if blob is not None:
                return self._entry(bucket, blob)
            prefix = key.rstrip('/') + '/' if key else ''
            first = next(iter(bucket.list_blobs(prefix=prefix, max_results=1)),
                         None)
            if not key or first is not None:
                return {'name': path, 'size': 0, 'type': 'directory'}
        raise FileNotFoundError(path)

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path).strip('/')
        info = self.info(path)
        if info['type'] == 'file':
            entries = [info]
        else:
            bucket, key = self._bucket_key(path)
            prefix = key.rstrip('/') + '/' if key else ''
            with translated(path):
                listing = bucket.list_blobs(prefix=prefix, delimiter='/')
                entries = [
                    self._entry(bucket, blob) for blob in listing
                    if blob.name != prefix
                ]
                entries.extend({
                    'name': f'{bucket.name}/{child.rstrip("/")}',
                    'size': 0,
                    'type': 'directory',
                } for child in sorted(listing.prefixes))
        return entries if detail else [entry['name'] for entry in entries]

    def exists(self, path, **kwargs):
        try:
            self.info(path)
            return True
        except FileNotFoundError:
            return False

    def isfile(self, path):
        try:
            return self.info(path)['type'] == 'file'
        except FileNotFoundError:
            return False

    def _open(self, path, mode='rb', block_size=None, cache_options=None,
              **kwargs):
        if mode != 'rb':
            raise NotImplementedError('GCS source filesystem is read-only')
        bucket, key = self._bucket_key(path)
        with translated(path):
            blob = bucket.blob(key)
            # Keep all footer/sample ranges on the generation whose size we read.
            blob.reload()
        return _RangeReadFile(self, path, blob, block_size=block_size,
                              cache_options=cache_options)
