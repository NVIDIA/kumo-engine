# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from typing import Dict

from kumo_core.object_store_sink import ObjectStoreSinkBackend
from kumoapi.data_source import DataSourceType

from .backend import S3Sessions


class S3SinkBackend(S3Sessions, ObjectStoreSinkBackend):
    """Write through the connector's filesystem. Cross-account destinations need
    bucket policy or BucketOwnerEnforced ownership; PyArrow sets no object ACL.
    """
    source_type = DataSourceType.S3

    def filesystem(self, connector, path, *, parallel_uploads=False):
        bucket, resolved, region = self._resolved(connector, path)
        options = _parallel_upload_options() if parallel_uploads else None
        return (self._filesystem(connector, bucket, region, options),
                resolved.rstrip('/'))

    def __repr__(self) -> str:
        return f'{type(self).__name__}(identity={self._identity!r})'


@lru_cache(maxsize=None)
def _parallel_upload_options() -> Dict[str, bool]:
    from pyarrow.fs import S3FileSystem

    try:
        S3FileSystem(anonymous=True, region='us-east-1',
                     allow_delayed_open=True)
    except TypeError:
        return {'background_writes': False}
    return {'allow_delayed_open': True, 'background_writes': False}
