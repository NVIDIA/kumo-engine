# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual live-module collection without a Google SDK or credentials."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

COLLECT = """
import importlib.abc
import sys
import types

import pytest

# No real ADC lookup can occur, even when testing an enabled live run.
lookups = 0
def default(**kwargs):
    global lookups
    lookups += 1
    raise RuntimeError("sentinel ADC unavailable")

if sys.argv[3] == "1":
    google = types.ModuleType("google")
    google.__path__ = []
    auth = types.ModuleType("google.auth")
    auth.default = default
    google.auth = auth
    sys.modules.update({"google": google, "google.auth": auth})

class NoProviderImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "google", "kumo_core", "kumoapi", "sources", "kumo_bigquery"
        }:
            raise AssertionError("Provider imported before gate/ADC: " + fullname)

sys.meta_path.insert(0, NoProviderImports())
result = pytest.main([
    "--collect-only", "--noconftest", "-c", sys.argv[1], "-rs", sys.argv[2]
])
print("ADC_LOOKUPS=" + str(lookups))
sys.exit(result)
"""


@pytest.mark.parametrize("module",
                         ["test_live_conformance.py", "test_live_sink.py"])
@pytest.mark.parametrize(
    "flag,project,staging,exit_code,message,lookups",
    [
        (None, "test-project", "gs://test-bucket", 5,
         "Set KUMO_BIGQUERY_LIVE=1", 0),
        ("0", "test-project", "gs://test-bucket", 5, "Set KUMO_BIGQUERY_LIVE=1",
         0),
        ("true", "test-project", "gs://test-bucket", 5,
         "Set KUMO_BIGQUERY_LIVE=1", 0),
        ("1", None, "gs://test-bucket", 2, "require explicit", 0),
        ("1", "test-project", None, 2, "require explicit", 0),
        ("1", "  ", "gs://test-bucket", 2, "require explicit", 0),
        ("1", "test-project", "  ", 2, "require explicit", 0),
        ("1", "test-project", "gs://test-bucket", 2, "sentinel ADC unavailable",
         1),
    ],
)
def test_live_collection_requires_explicit_opt_in_and_inputs(
        tmp_path, module, flag, project, staging, exit_code, message, lookups):
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT") if key in os.environ
    }
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    for key, value in (
        ("KUMO_BIGQUERY_LIVE", flag),
        ("KUMO_BIGQUERY_PROJECT", project),
        ("KUMO_BIGQUERY_STAGING", staging),
    ):
        if value is not None:
            env[key] = value
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            COLLECT,
            str(config),
            str(Path(__file__).parent / "live" / module),
            str(lookups),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == exit_code, output
    assert message in output, output
    assert f"ADC_LOOKUPS={lookups}" in output.splitlines(), output
