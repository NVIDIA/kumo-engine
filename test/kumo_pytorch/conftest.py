# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require complete pytest collection and execution for each CI profile."""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--required-coverage",
        choices=("cpu", "difficult"),
        help="Require the complete test directory; reject filtering, skips and xfails",
    )


def pytest_configure(config):
    from test.kumo_pytorch.refusals import pytest_plugin

    config.pluginmanager.register(pytest_plugin(), "capability-refusals")
    profile = config.getoption("--required-coverage")
    if profile:
        config.pluginmanager.register(
            RequiredCoverage(profile, config), "dataset-required-coverage"
        )


class RequiredCoverage:
    def __init__(self, profile, config):
        self.profile = profile
        self.prefix = (
            "difficult_behavior/" if profile == "difficult" else "model_checks/"
        )
        self.root = Path(__file__).resolve().parent / self.prefix
        self.other = self.root.parent / (
            "model_checks" if profile == "difficult" else "difficult_behavior"
        )
        if self.root.resolve() not in {Path(arg).resolve() for arg in config.args}:
            raise pytest.UsageError(
                f"Required coverage must collect the entire {self.root} directory"
            )
        if any(config.getoption(name) for name in ("ignore", "ignore_glob", "deselect")):
            raise pytest.UsageError("Required coverage cannot exclude test paths")
        self.collected = set()
        self.calls = set()
        self.problems = []

    def identity(self, nodeid):
        for prefix in ("model_checks/", "difficult_behavior/"):
            if prefix in nodeid:
                return prefix + nodeid.split(prefix, 1)[-1]
        return self.prefix + nodeid

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        # Other profiles execute in their own jobs.
        delegated = [item for item in items if item.path.is_relative_to(self.other)]
        if delegated:
            config.hook.pytest_deselected(items=delegated)
            items[:] = [item for item in items if item not in delegated]

    def pytest_deselected(self, items):
        self.problems.extend(
            f"Required test deselected: {item.nodeid}"
            for item in items
            if item.path.is_relative_to(self.root)
        )

    def pytest_collection_finish(self, session):
        if any(not item.path.is_relative_to(self.root) for item in session.items):
            raise pytest.UsageError(f"Unexpected tests outside {self.root}")
        self.collected = {self.identity(item.nodeid) for item in session.items}
        if not self.collected or len(self.collected) != len(session.items):
            raise pytest.UsageError(
                f"Required {self.profile} collection must be nonempty and unique"
            )
        if self.problems:
            raise pytest.UsageError("\n".join(self.problems))

    def pytest_collectreport(self, report):
        if report.skipped:
            self.problems.append(f"Collection skipped: {report.nodeid}")

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.calls.add(self.identity(report.nodeid))
        if report.skipped or hasattr(report, "wasxfail"):
            self.problems.append(f"Required test skipped/xfail: {report.nodeid}")

    def pytest_sessionfinish(self, session, exitstatus):
        if session.config.option.collectonly:
            return
        missing = self.collected - self.calls
        if missing:
            self.problems.append(
                f"Required test bodies not executed: {sorted(missing)}"
            )
        if self.problems:
            reporter = session.config.pluginmanager.getplugin("terminalreporter")
            if reporter:
                reporter.write_sep("!", "Required test coverage failed")
                for problem in self.problems:
                    reporter.write_line(problem)
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
