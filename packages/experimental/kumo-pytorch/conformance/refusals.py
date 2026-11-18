# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accept declared capability refusals from public preflight, never execution errors."""

import json
import sys
from contextvars import ContextVar

_EVIDENCE = ContextVar("capability_refusals", default=None)


def clean_refusal(error):
    """Return evidence only for prepare()'s pre-open unsupported-capability error."""
    spec = sys.modules.get("kumo.pytorch.spec")
    if (
        spec is None
        or not isinstance(error, spec.ValidationError)
        or not error.issues
        or error.__cause__ is not None
        or error.__context__ is not None
    ):
        return None
    from kumo.pytorch.preparation import prepare

    trace = error.__traceback__
    if trace is None:
        return None
    while trace.tb_next is not None:
        trace = trace.tb_next
    if trace.tb_frame.f_code is not prepare.__code__:
        return None
    engine = trace.tb_frame.f_locals.get("engine")
    declared = getattr(engine, "unsupported", {})
    issues = []
    for issue in error.issues:
        capability = declared.get(issue.code)
        if (
            capability is None
            or capability.refused_at != "check()"
            or not all(
                isinstance(value, str) and value.strip()
                for value in (issue.path, issue.message, issue.suggestion)
            )
        ):
            return None
        issues.append(
            {
                "code": issue.code,
                "path": issue.path,
                "reason": issue.message,
                "suggestion": issue.suggestion,
            }
        )
    return {
        "outcome": "refused",
        "stage": "preflight",
        "engine": engine.name,
        "execution": "not_started",
        "issues": issues,
    }


def record_refusal(error, **scope):
    evidence = clean_refusal(error)
    if evidence is not None:
        evidence = {**scope, **evidence}
        if (records := _EVIDENCE.get()) is not None:
            records.append(evidence)
    return evidence


def check_or_refuse(contract, dataset, engine, *, inference=False):
    """If check reports issues, require prepare to refuse before opening the engine."""
    from unittest.mock import patch

    from kumo.pytorch.api import ValidationError, check, prepare

    issues = check(contract, dataset, engine, inference=inference)
    if issues:
        with patch.object(engine, "open", side_effect=AssertionError("opened before refusal")):
            try:
                with prepare(contract, dataset, engine, inference=inference):
                    pass
            except ValidationError as error:
                assert error.issues == tuple(issues), "check and prepare disagree about refusal"
                raise
        raise AssertionError("check reported issues but prepare accepted the contract")


def pytest_plugin():
    """Construct lazily so ordinary report writers need no pytest import."""
    import pytest

    class Refusals:
        def __init__(self):
            self.records = []

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):
            records = list(getattr(item, "_setup_refusals", ()))
            token = _EVIDENCE.set(records)
            try:
                yield
            finally:
                item._capability_refusals = records
                _EVIDENCE.reset(token)

        @pytest.hookimpl(tryfirst=True)
        def pytest_pyfunc_call(self, pyfuncitem):
            # Setup already exercised the rejected capability. Do not invoke a
            # test body whose task/model fixture could not be constructed.
            if getattr(pyfuncitem, "_setup_refusals", ()):
                return True

        @pytest.hookimpl(hookwrapper=True, trylast=True)
        def pytest_runtest_makereport(self, item, call):
            outcome = yield
            report = outcome.get_result()
            if report.when == "setup" and report.failed and call.excinfo is not None:
                evidence = clean_refusal(call.excinfo.value)
                if evidence is not None:
                    item._setup_refusals = [evidence]
                    report.outcome, report.longrepr = "passed", None
            if report.when != "call":
                return
            records = list(getattr(item, "_capability_refusals", ()))
            if call.excinfo is not None and report.failed:
                evidence = clean_refusal(call.excinfo.value)
                if evidence is not None:
                    records.append(evidence)
                    report.outcome, report.longrepr = "passed", None
            if records:
                self.records.extend({"nodeid": item.nodeid, **record} for record in records)
                properties = [("capability_refusals", json.dumps(records, sort_keys=True))]
                item.user_properties.extend(properties)
                report.user_properties.extend(properties)

        def pytest_terminal_summary(self, terminalreporter):
            if self.records:
                terminalreporter.write_sep("-", "Accepted preflight capability refusals")
                for record in self.records:
                    reasons = "; ".join(
                        f"{i['code']} at {i['path']}: {i['reason']}" for i in record["issues"]
                    )
                    terminalreporter.write_line(f"{record['nodeid']}: {reasons}")
                terminalreporter.write_line(
                    "Refusal passes verify rejection, not execution of the unsupported capability."
                )

    return Refusals()
