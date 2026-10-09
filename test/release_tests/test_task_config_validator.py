# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from release_tests.release.api_test import create_sub_task_group
from release_tests.release.task_config import ReleaseTestTaskConfig
from release_tests.release.task_config_validator import get_execution_order


def test_create_sub_task_group():
    task_config = {
        "A":
        ReleaseTestTaskConfig(name="A", dependsOn=["B"], mustRunAfter=["C"],
                              precondition=lambda _: True, output=None),
        "B":
        ReleaseTestTaskConfig(name="B", dependsOn=[],
                              precondition=lambda _: True, output=None),
        "C":
        ReleaseTestTaskConfig(name="C", dependsOn=["B"],
                              precondition=lambda _: True, output=None)
    }
    sub_task_config = create_sub_task_group(task_config, ["A"])
    assert "C" not in sub_task_config
    assert "A" in sub_task_config
    assert "B" in sub_task_config

    task_config = {
        "A":
        ReleaseTestTaskConfig(name="A", dependsOn=["B"],
                              precondition=lambda _: True, output=None),
        "B":
        ReleaseTestTaskConfig(name="B", dependsOn=[], mustRunAfter=["C"],
                              precondition=lambda _: True, output=None),
        "C":
        ReleaseTestTaskConfig(name="C", dependsOn=[],
                              precondition=lambda _: True, output=None)
    }
    sub_task_config = create_sub_task_group(task_config, ["A"])
    assert "C" not in sub_task_config
    assert "A" in sub_task_config
    assert "B" in sub_task_config

    task_config = {
        "A":
        ReleaseTestTaskConfig(name="A", dependsOn=["B"],
                              precondition=lambda _: True, output=None),
        "B":
        ReleaseTestTaskConfig(name="B", dependsOn=["C"],
                              precondition=lambda _: True, output=None),
        "C":
        ReleaseTestTaskConfig(name="C", dependsOn=[],
                              precondition=lambda _: True, output=None)
    }
    sub_task_config = create_sub_task_group(task_config, ["A"])
    assert "C" in sub_task_config
    assert "A" in sub_task_config
    assert "B" in sub_task_config


def test_get_execution_order():
    task_config = {
        "A":
        ReleaseTestTaskConfig(name="A", dependsOn=["B"],
                              precondition=lambda _: True, output=None),
        "B":
        ReleaseTestTaskConfig(name="B", dependsOn=[],
                              precondition=lambda _: True, output=None),
        "C":
        ReleaseTestTaskConfig(name="C", dependsOn=["A"],
                              precondition=lambda _: True, output=None)
    }
    order = get_execution_order(task_config)
    # execute A before C
    assert order.index("A") < order.index("C")
    # execute B before A
    assert order.index("B") < order.index("A")
    assert len(order) == len(task_config.keys())

    task_config = {
        "A":
        ReleaseTestTaskConfig(name="A", dependsOn=["B"],
                              precondition=lambda _: True, output=None),
        "B":
        ReleaseTestTaskConfig(name="B", dependsOn=[],
                              precondition=lambda _: True, output=None),
        "C":
        ReleaseTestTaskConfig(name="C", mustRunAfter=["A"],
                              precondition=lambda _: True, output=None)
    }
    order = get_execution_order(task_config)
    # execute A before C
    assert order.index("A") < order.index("C")
    # execute B before A
    assert order.index("B") < order.index("A")
    assert len(order) == len(task_config.keys())
