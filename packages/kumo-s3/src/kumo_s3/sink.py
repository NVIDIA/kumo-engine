# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.object_store_sink import ObjectStoreSinkBackend
from kumoapi.data_source import DataSourceType

from .backend import S3Sessions


class S3SinkBackend(S3Sessions, ObjectStoreSinkBackend):
    """Write through the connector's filesystem. Cross-account destinations need
    bucket policy or BucketOwnerEnforced ownership; PyArrow sets no object ACL.
    """
    source_type = DataSourceType.S3

    def __repr__(self) -> str:
        return f'{type(self).__name__}(identity={self._identity!r})'
