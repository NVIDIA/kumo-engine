# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from kumo_core.object_store_sink import ObjectStoreSinkBackend
from kumoapi.data_source import DataSourceType
from pyarrow.fs import S3FileSystem

from .backend import S3Sessions


class S3SinkBackend(S3Sessions, ObjectStoreSinkBackend):
    """Write through the connector's filesystem. Cross-account destinations need
    bucket policy or BucketOwnerEnforced ownership; PyArrow sets no object ACL.
    """
    source_type = DataSourceType.S3

    def __repr__(self) -> str:
        return f'{type(self).__name__}(identity={self._identity!r})'

    def delete_file(self, connector: Any, path: str) -> None:
        r"""Deletes the object at ``path`` (``<bucket>/<key>``). Unlike
        pyarrow's delete, which recreates an emptied parent as a zero-byte
        ``<prefix>/`` object, this leaves no object behind."""
        bucket, _, key = path.partition('/')
        session, _, connection = self._resolved_session(connector, bucket)
        if isinstance(session, S3FileSystem):
            session.delete_file(path)
            return
        session.client(
            's3', endpoint_url=connection.endpoint_url if connection else None
        ).delete_object(Bucket=bucket, Key=key)
