# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, testable resolution of a validated plan.

The launcher re-preflights each block immediately before bind because a
userspace port scan cannot reserve a TCP port.
"""

import math
from dataclasses import replace
from typing import Callable, Sequence

from rest.devserver.cgroup import CgroupLimits
from rest.devserver.planning import MultiDevserverLaunchPlan, PlannedDevserver

_FIRST_AUTO_BASE_PORT = 11000
_AUTO_PORT_STRIDE = 100
_MAX_TCP_PORT = 65535
_MIN_CPU_PER_DEVSERVER = 1
_MIN_MEMORY_GIB_PER_DEVSERVER = 8


class DevserverCoordinatorError(RuntimeError):
    """The coordinator cannot safely resolve a launch plan."""


def _candidate_ports(child: PlannedDevserver) -> Sequence[int]:
    if child.base_port is None:
        raise DevserverCoordinatorError('child does not have a base port')
    return tuple(child.expected_ports.values())


def _block_available(child: PlannedDevserver,
                     is_available: Callable[[int], bool]) -> bool:
    return all(is_available(port) for port in _candidate_ports(child))


def assign_auto_base_ports(
    plan: MultiDevserverLaunchPlan,
    *,
    is_available: Callable[[int], bool],
    first_port: int = _FIRST_AUTO_BASE_PORT,
    stride: int = _AUTO_PORT_STRIDE,
) -> MultiDevserverLaunchPlan:
    """Resolve omitted base ports using complete nonlegacy blocks.

    Explicit ports are rechecked too. Automatic candidates use a wide stride
    so future panel offsets do not create an accidental neighboring collision.
    """
    if first_port < 1 or stride <= 0:
        raise DevserverCoordinatorError('invalid automatic port search range')

    assigned = []
    next_candidate = first_port
    used_ports = set()
    for child in plan.devservers:
        if child.base_port is not None:
            if not _block_available(child, is_available):
                raise DevserverCoordinatorError(
                    f'explicit base port {child.base_port} does not have a '
                    'free complete port block')
            assigned.append(child)
            used_ports.update(_candidate_ports(child))
            continue

        while True:
            candidate = replace(child, base_port=next_candidate)
            ports = set(_candidate_ports(candidate))
            if max(ports) > _MAX_TCP_PORT:
                raise DevserverCoordinatorError(
                    'no free port block remains in the TCP port range')
            next_candidate += stride
            if (not (ports & used_ports)
                    and _block_available(candidate, is_available)):
                assigned.append(candidate)
                used_ports.update(ports)
                break
    return replace(plan, devservers=tuple(assigned))


def resolve_auto_cgroup_limits(
    *,
    cpu_capacity: float,
    memory_gib: float,
    count: int,
) -> tuple[CgroupLimits, ...]:
    """Divide effective parent capacity with conservative shared headroom.

    CPU reserves the larger of 2 CPUs and 10% of capacity. Memory reserves the
    larger of 16 GiB and 10%. Fractional capacity is rounded down; remainder
    stays unallocated rather than being silently overcommitted.
    """
    if count <= 0 or cpu_capacity <= 0 or memory_gib <= 0:
        raise DevserverCoordinatorError('capacity and count must be positive')
    cpu_reserve = max(2, math.ceil(cpu_capacity * 0.10))
    memory_reserve = max(16, math.ceil(memory_gib * 0.10))
    cpus = math.floor((cpu_capacity - cpu_reserve) / count)
    memory = math.floor((memory_gib - memory_reserve) / count)
    if cpus < _MIN_CPU_PER_DEVSERVER or memory < _MIN_MEMORY_GIB_PER_DEVSERVER:
        raise DevserverCoordinatorError(
            'effective parent capacity leaves too little CPU or memory per '
            'devserver after shared-service headroom')
    return tuple(CgroupLimits(cpus=cpus, memory_gib=memory)
                 for _ in range(count))
