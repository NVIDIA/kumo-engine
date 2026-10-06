# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable

from pyarrow.fs import FileSystem

# A decoded transfer returns its local size; None preserves listed size checks.
ObjectFetch = Callable[[str, str], int | None]


@dataclass(frozen=True)
class ObjectStoreAccess:
    filesystem: FileSystem
    fetch: ObjectFetch | None = None
