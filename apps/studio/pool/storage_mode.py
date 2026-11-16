# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool worker storage mode.

This replaces the older lifetime-only classification with the storage
contract the worker is responsible for.
"""
from enum import Enum


class PoolStorageMode(str, Enum):
    DAEMON_ON_PVC = 'daemon-on-pvc'
    ACTIVITY_ON_PVC = 'activity-on-pvc'
    ACTIVITY_ON_EMPTYDIR = 'activity-on-emptyDir'

    @property
    def is_long_lived(self) -> bool:
        return self == PoolStorageMode.DAEMON_ON_PVC

    @property
    def is_activity_scoped(self) -> bool:
        return not self.is_long_lived

    @property
    def is_pvc_backed(self) -> bool:
        return self in (
            PoolStorageMode.DAEMON_ON_PVC,
            PoolStorageMode.ACTIVITY_ON_PVC,
        )
