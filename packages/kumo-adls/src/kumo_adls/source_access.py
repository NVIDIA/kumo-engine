# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.sources.object_store_access import ObjectStoreAccess
from pyarrow.fs import FSSpecHandler, PyFileSystem

from .sdk_filesystem import ADLSFileSystem
from .session import session_for


def source_access(principal, account: str, suffix: str) -> ObjectStoreAccess:
    client = session_for(principal, account, suffix)
    filesystem = ADLSFileSystem(client, account, suffix)
    return ObjectStoreAccess(PyFileSystem(FSSpecHandler(filesystem)),
                             filesystem.fetch, filesystem.object_identity)
