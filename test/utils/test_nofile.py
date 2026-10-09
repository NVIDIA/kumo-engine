# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import resource

import pytest
from kumo_core.utils import nofile

INF = resource.RLIM_INFINITY


@pytest.fixture
def limits(monkeypatch, caplog):
    nofile.ensure_nofile.cache_clear()
    monkeypatch.delenv("KUMO_RAISE_NOFILE", raising=False)
    caplog.set_level(logging.WARNING, logger=nofile.__name__)
    state = {"calls": []}

    def install(soft, hard, reject=lambda soft: False):
        state["limit"] = (soft, hard)
        state["reject"] = reject
        return state

    def setrlimit(kind, value):
        assert kind == resource.RLIMIT_NOFILE
        assert value[0] != INF
        state["calls"].append(value)
        if state["reject"](value[0]):
            raise ValueError("not allowed")
        state["limit"] = value

    monkeypatch.setattr(resource, "getrlimit", lambda kind: state["limit"])
    monkeypatch.setattr(resource, "setrlimit", setrlimit)
    yield install
    nofile.ensure_nofile.cache_clear()


def test_soft_below_hard_is_raised(limits, caplog):
    state = limits(1024, 1_048_576)
    nofile.ensure_nofile()
    assert state["limit"] == (1_048_576, 1_048_576)
    assert caplog.records == []


def test_already_at_hard_is_untouched(limits, caplog):
    state = limits(1_048_576, 1_048_576)
    nofile.ensure_nofile()
    assert state["calls"] == []
    assert caplog.records == []


def test_infinite_hard_raises_soft_to_recommended(limits, caplog):
    state = limits(256, INF)
    nofile.ensure_nofile()
    assert state["limit"] == (65535, INF)
    assert caplog.records == []


def test_failed_raise_warns(limits, caplog):
    state = limits(1024, 1_048_576, reject=lambda soft: True)
    nofile.ensure_nofile()
    assert state["limit"] == (1024, 1_048_576)
    assert len(caplog.records) == 1
    assert "soft=1024" in caplog.records[0].getMessage()


def test_low_hard_limit_warns_once(limits, caplog):
    state = limits(1024, 4096)
    nofile.ensure_nofile()
    nofile.ensure_nofile()
    assert state["limit"] == (4096, 4096)
    assert len(caplog.records) == 1


def test_opt_out_skips(limits, monkeypatch, caplog):
    monkeypatch.setenv("KUMO_RAISE_NOFILE", "0")
    state = limits(1024, 1_048_576)
    nofile.ensure_nofile()
    assert state["calls"] == []
    assert caplog.records == []
