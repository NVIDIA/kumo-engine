# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest

CODE = """
import logging
{setup}
from kumo_core.diskgraph._progress import log_progress
import sys
sys.stdin.readline()
log_progress(logging.getLogger("kumo_core.diskgraph.x"), "ingest: %s", "hi")
print("done")
"""


def test_default_prints_progress_to_stderr():
    result = subprocess.run(
        [sys.executable, "-c",
         CODE.format(setup="")],
        input="\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert result.stderr.splitlines() == ["ingest: hi"]


@pytest.mark.parametrize("setup", [
    "logging.disable(logging.CRITICAL)",
    "logging.getLogger('kumo_core.diskgraph.x').disabled = True",
])
def test_disabled_logging_prints_nothing(setup):
    result = subprocess.run(
        [sys.executable, "-c",
         CODE.format(setup=setup)],
        input="\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert result.stderr == ""


def test_broken_stderr_does_not_fail_the_caller():
    proc = subprocess.Popen(
        [sys.executable, "-c", CODE.format(setup="")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    proc.stderr.close()
    out, _ = proc.communicate("\n", timeout=120)
    assert out.strip() == "done"  # Caller continued past the write.
