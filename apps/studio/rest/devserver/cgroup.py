# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""systemd-run --user transient scope for one devserver launcher.

Avoids writing under ``/sys/fs/cgroup``, where an unprivileged process is only
delegated a subtree and can't assume CPU/memory controllers are enabled.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CgroupLimits:
    cpus: float
    memory_gib: float

    def __post_init__(self) -> None:
        if self.cpus <= 0 or self.memory_gib <= 0:
            raise ValueError('cgroup CPU and memory limits must be positive')

    @property
    def cpu_quota(self) -> str:
        return f'{self.cpus * 100:g}%'

    @property
    def memory_max(self) -> str:
        return f'{self.memory_gib:g}G'


def scope_argv(name: str, limits: CgroupLimits) -> List[str]:
    """Return a ``systemd-run`` wrapper argv for the named devserver scope.

    The caller appends its Python command then ``execvp``s this argv before
    any devserver setup.  ``--user`` is intentional: it does not need root or
    interactive PolicyKit authorization and confines only this user's work.
    """
    if '/' in name or name in {'', '.', '..'}:
        raise ValueError('cgroup scope name must be one path component')
    return [
        'systemd-run',
        '--user',
        '--scope',
        f'--unit=kumo-{name}',
        f'--property=CPUQuota={limits.cpu_quota}',
        f'--property=MemoryMax={limits.memory_max}',
    ]
