# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.sources.object_store_access import ObjectStoreAccess
from pyarrow.fs import FSSpecHandler, PyFileSystem

from .sdk_filesystem import ADLSFileSystem


def source_access(client, account: str,
                  suffix: str = 'dfs.core.windows.net') -> ObjectStoreAccess:
    """Borrow a service client; wrappers never close it or its credential."""
    filesystem = ADLSFileSystem(client, account, suffix)
    return ObjectStoreAccess(PyFileSystem(FSSpecHandler(filesystem)),
                             filesystem.fetch, filesystem.object_identity)
