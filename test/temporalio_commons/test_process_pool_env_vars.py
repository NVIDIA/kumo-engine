# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``_apply_child_env_vars`` (process-pool initializer).

Root cause these guard against: the pool initializer ``_init_spawned_process``
applied ``env_vars`` with ``os.environ.pop(name)`` (no default). When a caller
passes ``{name: None}`` (meaning "ensure unset in the child") and ``name`` is
absent from the child env, the bare pop raised ``KeyError`` *inside the pool
initializer*. CPython's ``_process_worker`` catches that, logs it, and returns
before consuming any work item, so the worker exits with the future still
pending and the executor fails the whole pool with ``BrokenProcessPool``.

The ``None`` branch must remain safe for any process-pool caller that needs to
ensure an environment variable is absent in the child.
"""
import os

from temporalio_commons.process_pool_utils import _apply_child_env_vars


def test_none_value_for_absent_var_is_noop(monkeypatch):
    # Pre-fix this raised KeyError -> BrokenProcessPool in the initializer.
    var = 'KUMO_TEST_ABSENT_ENV_VAR'
    monkeypatch.delenv(var, raising=False)
    _apply_child_env_vars({var: None})
    assert var not in os.environ


def test_none_value_unsets_present_var(monkeypatch):
    var = 'KUMO_TEST_PRESENT_ENV_VAR'
    monkeypatch.setenv(var, 'something')
    _apply_child_env_vars({var: None})
    assert var not in os.environ


def test_non_none_value_is_set_as_string(monkeypatch):
    var = 'KUMO_TEST_SET_ENV_VAR'
    monkeypatch.delenv(var, raising=False)
    _apply_child_env_vars({var: 123})
    assert os.environ[var] == '123'


def test_mixed_values_applied_independently(monkeypatch):
    set_var = 'KUMO_TEST_MIX_SET'
    absent_var = 'KUMO_TEST_MIX_ABSENT'
    present_var = 'KUMO_TEST_MIX_PRESENT'
    monkeypatch.delenv(set_var, raising=False)
    monkeypatch.delenv(absent_var, raising=False)
    monkeypatch.setenv(present_var, 'old')
    _apply_child_env_vars({
        set_var: '/tmp/persist',
        absent_var: None,
        present_var: None,
    })
    assert os.environ[set_var] == '/tmp/persist'
    assert absent_var not in os.environ
    assert present_var not in os.environ
