# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A refusal pass requires a declared capability and the actual preflight boundary."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from conformance.refusals import check_or_refuse, clean_refusal
from kumo.pytorch import api


def refused_contract():
    return api.DataLoaderSpec(
        api.DataSchema(
            (
                api.TableSpec(
                    "items", "id", {"id": api.DType("int64")}, time_column=api.NON_TEMPORAL
                ),
            )
        ),
        api.SampleSpec(
            api.EntitySpec("items", filters=(api.RowFilter("id", "greater_than", 0),)),
            entity_values={"id": api.Field("id")},
        ),
        api.BatchSpec(
            {
                "item": api.Input(
                    api.EntitySource(), api.Fields((api.Encode("id", api.Raw()),)), api.Separate()
                )
            }
        ),
    )


def actual_refusal():
    engine = api.DiskGraphEngine(index="must-not-be-created")
    with patch.object(engine, "open", side_effect=AssertionError("opened before refusal")):
        try:
            api.prepare(
                refused_contract(),
                api.Dataset({"items": api.Parquet("unreadable.parquet")}),
                engine,
            )
        except api.ValidationError as error:
            return error
    raise AssertionError("expected library refusal")


def test_declared_preflight_refusal_has_evidence():
    evidence = clean_refusal(actual_refusal())
    assert evidence["execution"] == "not_started"
    assert evidence["outcome"] == "refused"
    assert [issue["code"] for issue in evidence["issues"]] == ["DG-ROW-FILTER"]
    assert evidence["issues"][0]["path"] == "sample.entity.filters"


@pytest.mark.parametrize(
    "defect",
    [
        "unknown-code",
        "missing-path",
        "missing-reason",
        "missing-suggestion",
        "late-stage",
        "mixed-invalid",
        "plain-value-error",
        "assertion",
        "wrapped",
        "late-validation",
        "context",
        "cause",
    ],
)
def test_other_failures_are_not_refusal_passes(defect):
    error = actual_refusal()
    issue = error.issues[0]
    edits = {
        "unknown-code": {"code": "DG-NOT-DECLARED"},
        "missing-path": {"path": ""},
        "missing-reason": {"message": ""},
        "missing-suggestion": {"suggestion": None},
        "late-stage": {"code": "DG-DUPLICATE-LAYER-RELATION"},
    }
    if defect in edits:
        error.issues = (replace(issue, **edits[defect]),)
    elif defect == "mixed-invalid":
        error.issues += (api.ValidationIssue("batch", "invalid shape"),)
    elif defect == "plain-value-error":
        error = ValueError(str(error))
    elif defect == "assertion":
        error = AssertionError("wrong output")
    elif defect == "wrapped":
        wrapped = RuntimeError("execution failed")
        wrapped.__cause__ = error
        error = wrapped
    elif defect == "context":
        error.__context__ = AssertionError("wrong answer before cleanup")
    elif defect == "cause":
        error.__cause__ = RuntimeError("earlier crash")
    else:
        try:
            raise api.ValidationError((issue,))
        except api.ValidationError as late:
            error = late
    assert clean_refusal(error) is None


def test_check_must_agree_with_prepare():
    engine = api.DiskGraphEngine(index="must-not-be-created")
    with patch("kumo.pytorch.api.check", return_value=[api.ValidationIssue("other", "wrong")]):
        with pytest.raises(AssertionError, match="disagree"):
            check_or_refuse(
                refused_contract(),
                api.Dataset({"items": api.Parquet("unreadable.parquet")}),
                engine,
            )


@pytest.mark.parametrize("runner", ["generated", "corpus"])
@pytest.mark.parametrize("wrong_first", [False, True])
def test_refusal_does_not_abandon_cases_or_hide_prior_failure(monkeypatch, runner, wrong_first):
    from conformance import properties
    from tests.conformance import test_properties

    cases = [
        SimpleNamespace(id=str(i), dimensions={}, property="aggregate_answers", issue="fixture")
        for i in range(3)
    ]
    seen = []

    def check(case, *args):
        seen.append(case.id)
        if case.id == "0" and wrong_first:
            raise AssertionError("wrong answer")
        if case.id == "1":
            raise actual_refusal()

    if runner == "generated":
        monkeypatch.setattr(properties, "obligation_cases", lambda *args: cases)
        monkeypatch.setitem(
            properties.PROPERTIES,
            "aggregate_answers",
            replace(properties.PROPERTIES["aggregate_answers"], check=check),
        )
        monkeypatch.setattr(properties, "write_repro", lambda *args: Path("failure.json"))
        run = partial(properties.run_obligation, "aggregate_answers", "diskgraph", 0)
    else:
        monkeypatch.setattr(test_properties, "load_corpus", lambda: {c.id: c for c in cases})
        monkeypatch.setattr(test_properties, "require_engine", lambda *args: None)
        monkeypatch.setattr(test_properties, "replay", check)
        run = partial(
            test_properties.test_corpus,
            "diskgraph",
            SimpleNamespace(config=SimpleNamespace(getoption=lambda option: False)),
        )
    if wrong_first:
        with pytest.raises(AssertionError, match="wrong answer"):
            run()
    else:
        run()
    assert seen == ["0", "1", "2"]


def test_pytest_reports_refusals_separately_and_preserves_failures(tmp_path):
    package = Path(__file__).resolve().parents[2]
    (tmp_path / "conftest.py").write_text(
        "from conformance.refusals import pytest_plugin as refusal_plugin\n"
        "from tests.conformance.conftest import PASSED, pytest_runtest_makereport\n"
        "def pytest_configure(config):\n"
        "    config.stash[PASSED] = set()\n"
        "    config.pluginmanager.register(refusal_plugin())\n"
        "def pytest_sessionfinish(session):\n"
        "    if 'test_policy.py::test_refused' not in session.config.stash[PASSED]:\n"
        "        session.exitstatus = 3\n"
    )
    (tmp_path / "test_policy.py").write_text(
        "import pytest\n"
        "from tests.conformance.test_refusals import actual_refusal\n"
        "@pytest.fixture\ndef rejected():\n    raise actual_refusal()\n"
        "def test_setup_refused(rejected):\n    raise AssertionError('body must not execute')\n"
        "def test_refused():\n    raise actual_refusal()\n"
        "def test_wrong_answer():\n    assert 1 == 2\n"
        "def test_unexpected_error():\n    raise ValueError('unsupported maybe')\n"
        "def test_success():\n    assert 1 == 1\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path),
            "--confcutdir",
            str(tmp_path),
            "-q",
            "--junitxml",
            str(tmp_path / "results.xml"),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": str(package)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "2 failed, 3 passed" in result.stdout, result.stdout + result.stderr
    assert "Accepted preflight capability refusals" in result.stdout
    from xml.etree import ElementTree

    cases = {
        row.get("name"): row for row in ElementTree.parse(tmp_path / "results.xml").iter("testcase")
    }
    refused = cases["test_refused"]
    assert refused.find("failure") is None and refused.find("skipped") is None
    evidence = json.loads(refused.find("properties/property").get("value"))
    assert evidence[0]["issues"][0]["code"] == "DG-ROW-FILTER"
