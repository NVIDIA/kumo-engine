# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.object_store_sink import ObjectStoreSinkBackend
from kumoapi.data_source import DataSourceType

from .backend import GCSSessions


class GCSSinkBackend(GCSSessions, ObjectStoreSinkBackend):
    source_type = DataSourceType.GCS

    def __repr__(self) -> str:
        return f'{type(self).__name__}(identity={self._identity!r})'
