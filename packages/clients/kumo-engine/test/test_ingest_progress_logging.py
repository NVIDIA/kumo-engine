# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest

LINE = "ingest: table started table=users"
CODE = """
import logging
{before}
import kumo.engine
{after}
import sys
from kumo_core.diskgraph import runner
child = ("print('2026-10-02T00:00:00Z  INFO diskgraph_ingest: "
         "table started table=users')")
runner._run_with_heartbeat([sys.executable, "-c", child],
                           mem_limit_bytes=None, heartbeat_interval_s=0.05)
"""
HANDLER = ("h = logging.StreamHandler(); h.setFormatter("
           "logging.Formatter('%(message)s')); "
           "logging.getLogger('kumo_core').addHandler(h); "
           "logging.getLogger('kumo_core').setLevel(logging.INFO)")
ERROR = "logging.basicConfig(format='%(message)s', level=logging.ERROR)"


def _emissions(before: str = "", after: str = "") -> int:
    result = subprocess.run(
        [sys.executable, "-c",
         CODE.format(before=before, after=after)],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return result.stderr.splitlines().count(LINE)


@pytest.mark.parametrize("before,after,expected", [
    ("", "", 1),
    ("logging.basicConfig(format='%(message)s', level=logging.INFO)", "", 1),
    (ERROR, "", 0),
    ("", ERROR, 0),
    ("logging.getLogger('kumo_core.diskgraph').setLevel(logging.ERROR)", "", 0),
    (HANDLER, "", 1),
    ("", HANDLER, 1),
])
def test_progress_respects_application_logging(before, after, expected):
    assert _emissions(before, after) == expected
