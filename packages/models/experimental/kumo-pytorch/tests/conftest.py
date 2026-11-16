# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.data.records import (
    DatedValue,
    EntityHistory,
    EntityState,
    Event,
    FieldValue,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    from tests.conformance.conftest import add_conformance_options

    add_conformance_options(parser)
    parser.addoption(
        "--require-imports",
        action="store_true",
        help="fail if any test is skipped because an optional dependency can't be imported",
    )
    parser.addoption(
        "--t2a",
        action="store_true",
        help="also run the T2a end-to-end lifecycle tests at S (marked t2a; minutes on a CPU)",
    )
    parser.addoption(
        "--t2b",
        action="store_true",
        help="also run the nightly T2b end-to-end tests at M (marked t2b; ~15 min on a laptop)",
    )
    parser.addoption(
        "--e2e-dir",
        type=Path,
        help="keep the end-to-end runs, predictions and metrics.json under DIR/<tier>; "
        "finished runs there are reused, so use a fresh DIR after changing a flow",
    )


E2E_TIERS = ("t2a", "t2b")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "t2a: end-to-end lifecycle tier at S; runs only with --t2a")
    config.addinivalue_line("markers", "t2b: nightly end-to-end tier at M; runs only with --t2b")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skipped = [tier for tier in E2E_TIERS if not config.getoption(f"--{tier}")]
    deselected = [item for item in items if any(item.get_closest_marker(t) for t in skipped)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if item not in deselected]


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not session.config.getoption("--require-imports"):
        return
    reporter = session.config.pluginmanager.getplugin("terminalreporter")
    skipped = [
        report
        for report in (reporter.stats.get("skipped", ()) if reporter else ())
        if isinstance(report.longrepr, tuple) and "could not import" in report.longrepr[2]
    ]
    if not skipped:
        return
    reporter.ensure_newline()
    reporter.write_sep("!", "tests were skipped because an optional dependency is missing")
    for report in skipped:
        reporter.write_line(f"{report.nodeid}: {report.longrepr[2]}")
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def sample_histories() -> list[EntityHistory]:
    entity = EntityState(
        values=(
            FieldValue("country", "C001", DType("string")),
            FieldValue("user_tenure_days", 500.0, DType("float64")),
            FieldValue("active_account_count", 1, DType("int64")),
        ),
        dated_values=(DatedValue("lifelong_event", "signup", DType("string"), 100),),
    )
    events = (
        Event(
            "card_transaction",
            1_000,
            (
                FieldValue("event_source", "card_transaction", DType("string")),
                FieldValue("amount", 12.5, DType("float64")),
                FieldValue("currency", "USD", DType("string")),
                FieldValue("channel", "online", DType("string")),
                FieldValue("mcc", "5815", DType("string")),
                FieldValue("merchant_risk", 0.1, DType("float64")),
                FieldValue("entry_mode", "card_on_file", DType("string")),
                FieldValue("country", "C001", DType("string")),
                FieldValue("recurring_candidate_flag", True, DType("bool")),
                FieldValue("status", "approved", DType("string")),
                FieldValue("description", "monthly music plan", DType("string")),
            ),
        ),
        Event(
            "banking_transaction",
            1_500,
            (
                FieldValue("event_source", "banking_transaction", DType("string")),
                FieldValue("amount", 100.0, DType("float64")),
                FieldValue("transaction_type", "deposit", DType("string")),
                FieldValue("description", "payroll deposit", DType("string")),
            ),
        ),
        Event(
            "card_transaction",
            2_000,
            (
                FieldValue("event_source", "card_transaction", DType("string")),
                FieldValue("amount", 13.0, DType("float64")),
                FieldValue("currency", "USD", DType("string")),
                FieldValue("channel", "online", DType("string")),
                FieldValue("mcc", "5815", DType("string")),
                FieldValue("merchant_risk", 0.5, DType("float64")),
                FieldValue("entry_mode", "card_on_file", DType("string")),
                FieldValue("country", "C001", DType("string")),
                FieldValue("recurring_candidate_flag", True, DType("bool")),
                FieldValue("status", "approved", DType("string")),
                FieldValue("description", "monthly music plan", DType("string")),
            ),
        ),
        Event(
            "banking_transaction",
            2_500,
            (
                FieldValue("event_source", "banking_transaction", DType("string")),
                FieldValue("amount", 25.0, DType("float64")),
                FieldValue("transaction_type", "withdrawal", DType("string")),
                FieldValue("description", "cash withdrawal", DType("string")),
            ),
        ),
    )
    return [
        EntityHistory(1, 3_000, entity, events),
        EntityHistory(2, 3_000, entity, events),
    ]
