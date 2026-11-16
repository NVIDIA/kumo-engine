# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure multi-devserver launch planner."""

import pytest

from rest.devserver.planning import (
    DevserverPlanError,
    build_multi_devserver_plan,
)


def test_auto_requires_nums_and_generates_stable_child_ids():
    with pytest.raises(DevserverPlanError, match='requires NUMS'):
        build_multi_devserver_plan('auto')

    plan = build_multi_devserver_plan('auto', nums='2')

    assert plan.cgroup_mode == 'auto'
    assert [child.devserver_id for child in plan.devservers] == [
        'devserver-1', 'devserver-2'
    ]
    assert [child.base_port for child in plan.devservers] == [None, None]
    assert all(child.cgroup_limits is None for child in plan.devservers)


def test_auto_enforces_nums_and_devserver_ids_cardinality():
    plan = build_multi_devserver_plan('auto', nums=2,
                                      devserver_ids='Train-A,train-b')

    assert [child.devserver_id for child in plan.devservers] == [
        'train-a', 'train-b'
    ]

    with pytest.raises(DevserverPlanError, match='DEVSERVER_IDS has 1 values'):
        build_multi_devserver_plan('auto', nums=2, devserver_ids='train-a')


def test_custom_json_uses_child_ids_as_keys_and_builds_immutable_plan():
    plan = build_multi_devserver_plan(
        '{"train-a":{"cpus":32,"memory_gib":128},'
        '"train-b":{"cpus":16,"memory_gib":64}}',
        nums=2,
        devserver_ids='train-b,train-a',
        base_ports='11000,12000')

    assert plan.cgroup_mode == 'custom'
    assert tuple(plan.devservers) == plan.devservers
    assert [child.devserver_id for child in plan.devservers] == [
        'train-b', 'train-a'
    ]
    assert plan.devservers[0].cgroup_limits.cpus == 16
    assert plan.devservers[1].cgroup_limits.memory_gib == 128
    assert plan.devservers[0].expected_ports['prometheus'] == 11016
    assert plan.devservers[1].expected_ports['rest_api'] == 12001


@pytest.mark.parametrize(
    'config,error',
    [
        ('[]', 'non-empty object'),
        ('{"train-a": {}}', 'exactly cpus and memory_gib'),
        ('{"train-a":{"cpus":true,"memory_gib":1}}',
         'finite positive number'),
        ('{"train-a":{"cpus":1,"memory_gib":0}}',
         'finite positive number'),
        ('{"train-a":{"cpus":1,"memory_gib":1,"extra":2}}',
         'exactly cpus and memory_gib'),
    ],
)
def test_custom_json_requires_exact_finite_positive_allocations(config, error):
    with pytest.raises(DevserverPlanError, match=error):
        build_multi_devserver_plan(config)


def test_custom_json_rejects_duplicate_raw_and_normalized_child_ids():
    with pytest.raises(DevserverPlanError, match='duplicate JSON key'):
        build_multi_devserver_plan(
            '{"train-a":{"cpus":1,"memory_gib":1},'
            '"train-a":{"cpus":2,"memory_gib":2}}')
    with pytest.raises(DevserverPlanError,
                       match='duplicate IDs after normalization'):
        build_multi_devserver_plan(
            '{"Train-A":{"cpus":1,"memory_gib":1},'
            '"train-a":{"cpus":2,"memory_gib":2}}')


def test_custom_json_and_explicit_ids_must_have_same_normalized_set():
    config = ('{"train-a":{"cpus":1,"memory_gib":1},'
              '"train-b":{"cpus":2,"memory_gib":2}}')

    with pytest.raises(DevserverPlanError, match='NUMS is 1'):
        build_multi_devserver_plan(config, nums=1)
    with pytest.raises(DevserverPlanError, match='exactly match'):
        build_multi_devserver_plan(config, devserver_ids='train-a,train-c')


@pytest.mark.parametrize('ids', ['../bad', 'a/b', '1train', 'x' * 49])
def test_devserver_ids_must_be_safe_after_normalization(ids):
    with pytest.raises(DevserverPlanError, match='ASCII lowercase'):
        build_multi_devserver_plan('auto', nums=1, devserver_ids=ids)


def test_port_cardinality_duplicates_overlap_and_overflow_are_rejected():
    with pytest.raises(DevserverPlanError, match='BASE_PORTS has 1 values'):
        build_multi_devserver_plan('auto', nums=2, base_ports='11000')
    with pytest.raises(DevserverPlanError, match='duplicate base ports'):
        build_multi_devserver_plan('auto', nums=2, base_ports='11000,11000')
    with pytest.raises(DevserverPlanError, match='overlap on required'):
        build_multi_devserver_plan('auto', nums=2, base_ports='11000,11001')
    with pytest.raises(DevserverPlanError, match='overflows TCP port range'):
        build_multi_devserver_plan('auto', nums=1, base_ports='65520')


def test_auto_base_ports_are_deferred_to_stateful_coordinator():
    plan = build_multi_devserver_plan(
        'auto', nums=2, devserver_ids='train-a,train-b', base_ports='auto')

    assert [child.base_port for child in plan.devservers] == [None, None]


def test_unfenced_requires_nums_and_cannot_mix_cgroup_config():
    with pytest.raises(DevserverPlanError, match='requires NUMS'):
        build_multi_devserver_plan(unfenced=True)
    with pytest.raises(DevserverPlanError, match='mutually exclusive'):
        build_multi_devserver_plan('auto', nums=2, unfenced=True)

    plan = build_multi_devserver_plan(
        nums=2, devserver_ids='train-a,train-b', unfenced=True)

    assert plan.cgroup_mode == 'unfenced'
    assert all(child.cgroup_limits is None for child in plan.devservers)
