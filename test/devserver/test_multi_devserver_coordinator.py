# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stateful-free resolution performed before child launch."""

import pytest

from rest.devserver.coordinator import (
    DevserverCoordinatorError,
    assign_auto_base_ports,
    resolve_auto_cgroup_limits,
)
from rest.devserver.planning import build_multi_devserver_plan


def test_auto_port_assignment_checks_the_complete_nonlegacy_block():
    plan = build_multi_devserver_plan('auto', nums=2,
                                      devserver_ids='train-a,train-b')

    resolved = assign_auto_base_ports(
        plan,
        is_available=lambda port: port not in {11001, 11101},
        first_port=11000,
        stride=100,
    )

    assert [child.base_port for child in resolved.devservers] == [11200, 11300]


def test_auto_port_assignment_fails_without_enough_free_blocks():
    plan = build_multi_devserver_plan('auto', nums=1)

    with pytest.raises(DevserverCoordinatorError, match='free port block'):
        assign_auto_base_ports(plan,
                               is_available=lambda _: False,
                               first_port=65000,
                               stride=100)


def test_auto_cgroup_limits_reserve_headroom_and_divide_equally():
    limits = resolve_auto_cgroup_limits(cpu_capacity=64,
                                        memory_gib=512,
                                        count=2)

    assert limits[0].cpus == 28
    assert limits[1].cpus == 28
    assert limits[0].memory_gib == 230
    assert limits[1].memory_gib == 230


def test_auto_cgroup_limits_fail_instead_of_tiny_allocations():
    with pytest.raises(DevserverCoordinatorError, match='too little'):
        resolve_auto_cgroup_limits(cpu_capacity=2,
                                   memory_gib=16,
                                   count=2)
