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
from tests.conformance.conftest import (
    PINNED_BATCH_BOUNDARIES,
    PINNED_DTYPES,
    PINNED_ENCODERS,
    PINNED_FIELD_SEMANTICS,
)


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
        "skip_boundary",
        "xfail_boundary",
        "missing_boundary",
        "ignored_boundary",
        "deselected_boundary",
    ),
)
@pytest.mark.parametrize("native_only", (False, True), ids=("all-engines", "native-only"))
def test_complete_run_requires_actual_case_execution(tmp_path, scenario, native_only):
    selected_engines = ("diskgraph",) if native_only else ENGINES
    shutil.copytree(REPOSITORY / "conformance", tmp_path / "conformance")
    tests = tmp_path / "tests/conformance"
    tests.mkdir(parents=True)
    (tests / "conftest.py").write_text(Path(__file__).with_name("conftest.py").read_text())
    ids = [
        f"{case_id}-{engine}-{'answer' if decision == 'answer' else 'refusal'}"
        for case_id, engines in load_expectations().items()
        for engine, decision in engines.items()
        if engine in selected_engines
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
    properties = [
        obligation_id(*item)
        for item in obligations(selected_engines, include_comparisons=not native_only)
    ]
    if scenario == "missing_property":
        properties.pop()
    if scenario == "dropped_seed":
        # The property module itself drops a batch; the conftest pin still requires it.
        module = tmp_path / "conformance/properties.py"
        text = module.read_text()
        before, after, suffix = ('"diskgraph": (0,)', '"diskgraph": ()', "diskgraph-seed0")
        assert before in text
        module.write_text(text.replace(before, after))
        properties = [item for item in properties if not item.endswith(suffix)]
    gaps = [gap.id for gap in REPLAYED if gap.engine in selected_engines]
    body = {name: "pass" for name in ("test_property", "test_corpus", "test_known_gap")}
    if scenario in ("skip_property", "skip_corpus", "skip_known_gap"):
        body[f"test_{scenario.removeprefix('skip_')}"] = "pytest.skip('required obligation')"
    (tests / "test_properties.py").write_text(
        "import pytest\n"
        f"@pytest.mark.parametrize('obligation', {properties!r}, ids=lambda item: item)\n"
        f"def test_property(obligation):\n    {body['test_property']}\n"
        f"@pytest.mark.parametrize('corpus_engine', {list(selected_engines)!r})\n"
        f"def test_corpus(corpus_engine):\n    {body['test_corpus']}\n"
        f"@pytest.mark.parametrize('gap', {gaps!r}, ids=lambda item: item)\n"
        f"def test_known_gap(gap):\n    {body['test_known_gap']}\n"
    )
    dtype_ids = [
        f"{engine}-{family}-{dtype}"
        for engine in selected_engines
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
    boundary_source = ["import pytest\n"]
    for test, variants in PINNED_BATCH_BOUNDARIES.items():
        boundary_ids = [
            f"{engine}{'-' + variant if variant else ''}"
            for engine in selected_engines
            if engine == "diskgraph"
            for variant in variants
        ]
        if scenario == "missing_boundary":
            boundary_ids.pop()
        body = {
            "skip_boundary": "pytest.skip('required boundary')",
            "xfail_boundary": "pytest.xfail('required answer is unsupported')",
        }.get(scenario, "pass")
        boundary_source.append(
            f"@pytest.mark.parametrize('case', {boundary_ids!r}, ids=lambda case: case)\n"
            f"def {test}(case):\n    {body}\n"
        )
    (tests / "test_batch_boundaries.py").write_text("".join(boundary_source))
    (tests / "test_field_semantics.py").write_text(
        "import pytest\n"
        + "".join(
            f"@pytest.mark.parametrize('engine_name', {list(selected_engines)!r})\n"
            f"def {name}(engine_name):\n    pass\n"
            for name in PINNED_FIELD_SEMANTICS
        )
    )
    # Retain a passing test even if the entire public-behavior file is ignored.
    (tests / "test_other.py").write_text("def test_other():\n    pass\n")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/conformance",
        f"--conformance-engine={'diskgraph' if native_only else 'all'}",
        "--conformance-complete",
    ]
    if native_only:
        command.append("--conformance-no-in-memory")
    if scenario == "deselect":
        command.extend(["-k", "not test_public_behavior"])
    elif scenario == "ignored":
        command.append("--ignore=tests/conformance/test_guarantees.py")
    elif scenario == "ignored_boundary":
        command.append("--ignore=tests/conformance/test_batch_boundaries.py")
    elif scenario == "deselected_boundary":
        command.extend(["-k", "not test_batch_boundaries"])
    elif scenario == "empty":
        command.extend(["-k", "does_not_exist"])
    result = run_pytest(command, tmp_path)
    assert result.returncode == (0 if scenario == "pass" else 1), result.stdout + result.stderr


@pytest.mark.parametrize(
    "engine,body,returncode,passed",
    (
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


@pytest.mark.parametrize("operation", ("construct", "open", "anchors"))
def test_native_only_forbids_in_memory_execution(tmp_path, operation):
    tests = tmp_path / "tests/conformance"
    tests.mkdir(parents=True)
    (tests / "conftest.py").write_text(Path(__file__).with_name("conftest.py").read_text())
    call = {
        "construct": "InMemoryEngine()",
        "open": "InMemoryEngine.open(object(), None, None)",
        "anchors": "InMemoryEngine.materialize_anchors(object(), None)",
    }[operation]
    (tests / "test_field_semantics.py").write_text(
        f"from kumo.pytorch.api import InMemoryEngine\ndef test_forbidden():\n    {call}\n"
    )
    result = run_pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/conformance",
            "--conformance-engine=diskgraph",
            "--conformance-no-in-memory",
        ],
        tmp_path,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "InMemoryEngine is forbidden" in result.stdout


def test_native_only_requires_diskgraph(tmp_path):
    result = run_pytest(
        [
            sys.executable,
            "-m",
            "conformance",
            "--engine",
            "all",
            "--no-in-memory",
        ],
        REPOSITORY,
    )
    assert result.returncode == 2 and "requires --engine diskgraph" in result.stderr


@pytest.mark.parametrize("daily_seeds", (False, True), ids=("pinned", "daily"))
def test_native_only_collects_behavior_without_reference_obligations(daily_seeds):
    result = run_pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/conformance",
            "--conformance-engine=diskgraph",
            "--conformance-no-in-memory",
            *(["--conformance-seeds=12", "--conformance-seed-start=100"] if daily_seeds else []),
        ],
        REPOSITORY,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = [line for line in result.stdout.splitlines() if "::test_" in line]
    assert nodes and not any("in_memory" in node for node in nodes)
    assert any("fitted_state-" in node for node in nodes)
    assert {node.split("::")[0] for node in nodes} == {
        f"tests/conformance/{name}"
        for name in (
            "test_guarantees.py",
            "test_properties.py",
            "test_dtype_matrix.py",
            "test_field_semantics.py",
            "test_batch_boundaries.py",
        )
    }
    assert any("anchor_splits-diskgraph" in node for node in nodes)
    assert any("contract_transfer-diskgraph" in node for node in nodes)

    if daily_seeds:
        from tests.conformance.conftest import PINNED_NATIVE_PROPERTIES

        actual = {node for node in nodes if "::test_property[" in node}
        expected = {
            f"tests/conformance/test_properties.py::test_property[{name}-diskgraph-seed{seed}]"
            for name in PINNED_NATIVE_PROPERTIES
            for seed in (0, *range(100, 112))
        }
        assert actual == expected
