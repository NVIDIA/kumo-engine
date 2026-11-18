# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in CI enforcement of the reviewed dataset/model test inventory and execution."""

import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--required-coverage",
        choices=("cpu", "gpu"),
        help="Require the reviewed dataset/model cases; reject missing cases, skips and xfails",
    )


def pytest_configure(config):
    from test.kumo_pytorch.refusals import pytest_plugin

    config.pluginmanager.register(pytest_plugin(), "capability-refusals")
    profile = config.getoption("--required-coverage")
    if profile:
        manifest = json.loads(
            Path(__file__).with_name("required-tests.json").read_text()
        )
        config.pluginmanager.register(
            RequiredCoverage(profile, manifest), "dataset-required-coverage"
        )


class RequiredCoverage:
    def __init__(self, profile, manifest):
        self.profile = profile
        self.expected = set(manifest[profile])
        self.other = set(manifest["gpu" if profile == "cpu" else "cpu"])
        if (
            not self.expected
            or self.expected & self.other
            or len(self.expected) != len(manifest[profile])
        ):
            raise pytest.UsageError(
                "Required dataset/model inventory must be nonempty, unique and have disjoint owners"
            )
        self.collected = set()
        self.calls = set()
        self.problems = []

    @staticmethod
    def identity(nodeid):
        return "model_checks/" + nodeid.split("model_checks/", 1)[-1]

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        # Explicit ownership: GPU reference cases run in their dedicated CI job.
        delegated = [item for item in items if self.identity(item.nodeid) in self.other]
        if delegated:
            config.hook.pytest_deselected(items=delegated)
            items[:] = [item for item in items if item not in delegated]

    def pytest_collection_finish(self, session):
        self.collected = {self.identity(item.nodeid) for item in session.items}
        missing = self.expected - self.collected
        extra = self.collected - self.expected
        if missing or extra:
            raise pytest.UsageError(
                f"Required dataset/model {self.profile} inventory differs: "
                f"missing={sorted(missing)}, unreviewed={sorted(extra)}"
            )

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
        missing = self.expected - self.calls
        if missing:
            self.problems.append(
                f"Required test bodies not executed: {sorted(missing)}"
            )
        if self.problems:
            reporter = session.config.pluginmanager.getplugin("terminalreporter")
            if reporter:
                reporter.write_sep("!", "Required dataset/model coverage failed")
                for problem in self.problems:
                    reporter.write_line(problem)
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
