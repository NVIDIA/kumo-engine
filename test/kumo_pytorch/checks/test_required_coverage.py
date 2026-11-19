# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise fail-closed coverage through actual pytest subprocesses."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "fault", [None, "missing", "skip", "xfail", "deselect", "import", "extra"]
)
@pytest.mark.parametrize("profile", ["cpu", "difficult"])
def test_required_coverage(tmp_path, fault, profile):
    root = Path(__file__).parents[1]
    (tmp_path / "conftest.py").write_text((root / "conftest.py").read_text())
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    directory = "difficult_behavior" if profile == "difficult" else "model_checks"
    tests = tmp_path / directory
    tests.mkdir()
    body = "def test_required():\n    assert True\n"
    if fault == "missing":
        body = ""
    elif fault == "skip":
        body = "def test_required():\n    pytest.skip('missing behavior')\n"
    elif fault == "xfail":
        body = "@pytest.mark.xfail(reason='unsupported')\ndef test_required():\n    assert False\n"
    elif fault == "import":
        body = "pytest.importorskip('absent_aml_dependency')\n" + body
    elif fault == "extra":
        body += "def test_unreviewed():\n    pass\n"
    (tests / "test_cpu.py").write_text("import pytest\n" + body)
    (tests / "test_gpu.py").write_text(
        "def test_gpu():\n    raise AssertionError('owned by GPU job')\n"
    )
    (tmp_path / "required-tests.json").write_text(
        json.dumps(
            {
                profile: [f"{directory}/test_cpu.py::test_required"],
                "gpu": [f"{directory}/test_gpu.py::test_gpu"],
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(tmp_path / "pytest.ini"),
            "--confcutdir",
            str(tmp_path),
            f"--required-coverage={profile}",
            str(tests),
            "-q",
            *(["-k", "no_matching_test"] if fault == "deselect" else []),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str(root.parents[1])
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) == (fault is None), result.stdout + result.stderr
    if fault is None:
        assert "1 passed, 1 deselected" in result.stdout
