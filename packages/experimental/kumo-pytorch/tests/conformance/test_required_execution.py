# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A skipped required test and an empty selection must return a failing exit code."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conformance.cases import ENGINES, REPOSITORY
from conformance.expectations import load_expectations
from conformance.properties import obligation_id, obligations
from conformance.registry import REPLAYED
from tests.conformance.conftest import PINNED_DTYPES, PINNED_ENCODERS


def run_pytest(command, directory):
    # A parent's debugging filter must not change the scenario being tested.
    env = {key: value for key, value in os.environ.items() if key != "PYTEST_ADDOPTS"}
    if env.get("PYTHONPATH"):
        # The child changes directory. Keep testing this checkout's package,
        # not a different editable install found after relative paths move.
        env["PYTHONPATH"] = os.pathsep.join(
            str((REPOSITORY / entry).resolve()) if not Path(entry).is_absolute() else entry
            for entry in env["PYTHONPATH"].split(os.pathsep)
        )
    return subprocess.run(
        command, cwd=directory, env=env, capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize("scenario", ("skip", "empty"))
def test_required_tests_cannot_become_skipped_success(tmp_path, scenario):
    tests = tmp_path / "tests/conformance"
    tests.mkdir(parents=True)
    plugin = Path(__file__).with_name("conftest.py")
    (tests / "conftest.py").write_text(plugin.read_text())
    (tests / "test_example.py").write_text(
        "import pytest\ndef test_example():\n    pytest.skip('missing required engine')\n"
    )
    command = [sys.executable, "-m", "pytest", "-q", "tests/conformance"]
    if scenario == "empty":
        command.extend(["-k", "does_not_exist"])
    result = run_pytest(command, tmp_path)
    assert result.returncode == (1 if scenario == "skip" else 5), result.stdout + result.stderr


@pytest.mark.parametrize(
    "scenario",
    (
        "pass",
        "skip",
        "xfail",
        "xpass",
        "deselect",
        "ignored",
        "missing_case",
        "empty",
        "skip_property",
        "missing_property",
        "skip_corpus",
        "dropped_seed",
        "skip_known_gap",
        "skip_dtype",
        "missing_dtype",
    ),
)
def test_complete_run_requires_actual_case_execution(tmp_path, scenario):
    shutil.copytree(REPOSITORY / "conformance", tmp_path / "conformance")
    tests = tmp_path / "tests/conformance"
    tests.mkdir(parents=True)
    (tests / "conftest.py").write_text(Path(__file__).with_name("conftest.py").read_text())
    ids = [
        f"{case_id}-{engine}-{'answer' if decision == 'answer' else 'refusal'}"
        for case_id, engines in load_expectations().items()
        for engine, decision in engines.items()
    ]
    if scenario == "missing_case":
        ids.pop()
    decorator = "@pytest.mark.xfail(reason='unexpected pass')\n" if scenario == "xpass" else ""
    action = {
        "skip": "pytest.skip('required case skipped')",
        "xfail": "pytest.xfail('required case expected to fail')",
    }.get(scenario, "pass")
    (tests / "test_guarantees.py").write_text(
        "import pytest\n"
        f"@pytest.mark.parametrize('case', {ids!r}, ids=lambda case: case)\n"
        + decorator
        + f"def test_public_behavior(case):\n    {action}\n"
        + "def test_catalog():\n    pass\n"
    )
    properties = [obligation_id(*item) for item in obligations(ENGINES)]
    if scenario == "missing_property":
        properties.pop()
    if scenario == "dropped_seed":
        # The property module itself drops a batch; the conftest pin still requires it.
        module = tmp_path / "conformance/properties.py"
        text = module.read_text()
        assert '"in_memory": (0, 1, 2)' in text
        module.write_text(text.replace('"in_memory": (0, 1, 2)', '"in_memory": (0, 1)'))
        properties = [item for item in properties if not item.endswith("in_memory-seed2")]
    gaps = [gap.id for gap in REPLAYED]
    body = {name: "pass" for name in ("test_property", "test_corpus", "test_known_gap")}
    if scenario in ("skip_property", "skip_corpus", "skip_known_gap"):
        body[f"test_{scenario.removeprefix('skip_')}"] = "pytest.skip('required obligation')"
    (tests / "test_properties.py").write_text(
        "import pytest\n"
        f"@pytest.mark.parametrize('obligation', {properties!r}, ids=lambda item: item)\n"
        f"def test_property(obligation):\n    {body['test_property']}\n"
        f"@pytest.mark.parametrize('corpus_engine', {list(ENGINES)!r})\n"
        f"def test_corpus(corpus_engine):\n    {body['test_corpus']}\n"
        f"@pytest.mark.parametrize('gap', {gaps!r}, ids=lambda item: item)\n"
        f"def test_known_gap(gap):\n    {body['test_known_gap']}\n"
    )
    dtype_ids = [
        f"{engine}-{family}-{dtype}"
        for engine in ENGINES
        for family in PINNED_ENCODERS
        for dtype in PINNED_DTYPES
    ]
    if scenario == "missing_dtype":
        dtype_ids.pop()
    dtype_body = "pytest.skip('required type pair')" if scenario == "skip_dtype" else "pass"
    (tests / "test_dtype_matrix.py").write_text(
        "import pytest\n"
        f"@pytest.mark.parametrize('cell', {dtype_ids!r}, ids=lambda item: item)\n"
        f"def test_scalar_compatibility_and_real_execution(cell):\n    {dtype_body}\n"
    )
    # Retain a passing test even if the entire public-behavior file is ignored.
    (tests / "test_other.py").write_text("def test_other():\n    pass\n")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/conformance",
        "--conformance-engine=all",
        "--conformance-complete",
    ]
    if scenario == "deselect":
        command.extend(["-k", "not test_public_behavior"])
    elif scenario == "ignored":
        command.append("--ignore=tests/conformance/test_guarantees.py")
    elif scenario == "empty":
        command.extend(["-k", "does_not_exist"])
    result = run_pytest(command, tmp_path)
    assert result.returncode == (0 if scenario == "pass" else 1), result.stdout + result.stderr


@pytest.mark.parametrize(
    "engine,body,returncode,passed",
    (
        ("in_memory", "pass", 0, 1),
        ("all", "pass", 0, 2),
        ("diskgraph", "pass", 0, 2),
        ("all", "raise ImportError('no diskgraph')", 1, 1),
    ),
)
def test_engine_harness_tests_follow_the_selected_engines(
    tmp_path, engine, body, returncode, passed
):
    """Not created for an unselected engine; required, never skipped, for a selected one."""
    shutil.copytree(REPOSITORY / "conformance", tmp_path / "conformance")
    tests = tmp_path / "tests/conformance"
    tests.mkdir(parents=True)
    (tests / "conftest.py").write_text(Path(__file__).with_name("conftest.py").read_text())
    (tests / "test_example.py").write_text(
        "import pytest\n"
        "def test_any_engine():\n    pass\n"
        "@pytest.mark.engine('diskgraph')\n"
        f"def test_native():\n    {body}\n"
    )
    command = [sys.executable, "-m", "pytest", "-q", "tests/conformance"]
    result = run_pytest([*command, f"--conformance-engine={engine}"], tmp_path)
    output = result.stdout + result.stderr
    assert result.returncode == returncode, output
    assert f"{passed} passed" in output and "skipped" not in output and "deselected" not in output
