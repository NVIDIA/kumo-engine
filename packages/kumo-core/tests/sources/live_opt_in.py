# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect credential-file live suites without importing their providers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

COLLECTION_CASES = (
    "disabled-unset",
    "disabled-zero",
    "disabled-other",
    "missing-path",
    "blank-path",
    "missing-file",
    "unreadable-file",
    "malformed-json",
    "non-utf8",
    "non-object-json",
    "blank-field",
    "wrong-type-field",
    "valid-inputs",
)
SECRET = "synthetic-secret-do-not-echo"

COLLECT = """
import builtins
import importlib.abc
import io
import json
import os
import sys

import pytest

config, module, credential_path, credential_variable, mode = sys.argv[1:]
accesses = {"environment": 0, "file": 0, "stat": 0, "provider": 0}

original_getitem = type(os.environ).__getitem__
def environment_getitem(self, key):
    if self is os.environ and key == credential_variable:
        accesses["environment"] += 1
        if mode == "disabled":
            raise AssertionError("Credential path inspected before opt-in")
    return original_getitem(self, key)
type(os.environ).__getitem__ = environment_getitem

def guard_file_access(original, kind):
    def guarded(path, *args, **kwargs):
        if not isinstance(path, int):
            resolved = os.path.abspath(os.fsdecode(path))
            if resolved == credential_path:
                accesses[kind] += 1
                if mode in {"disabled", "missing-path"}:
                    raise AssertionError("Credential file inspected before opt-in")
                if mode == "unreadable":
                    raise PermissionError("sentinel unreadable credential file")
        return original(path, *args, **kwargs)
    return guarded

builtins.open = guard_file_access(builtins.open, "file")
io.open = guard_file_access(io.open, "file")
os.open = guard_file_access(os.open, "file")
os.stat = guard_file_access(os.stat, "stat")
os.lstat = guard_file_access(os.lstat, "stat")

class NoProviderImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "snowflake", "databricks", "kumo_core", "kumoapi", "sources",
            "kumo_snowflake", "kumo_databricks", "google", "boto3", "botocore"
        }:
            accesses["provider"] += 1
            if mode == "valid":
                assert accesses["environment"] and accesses["file"]
                raise RuntimeError(
                    "sentinel provider import after credential validation"
                )
            raise AssertionError("Provider imported before validation: " + fullname)

sys.meta_path.insert(0, NoProviderImports())
result = pytest.main([
    "--collect-only", "--noconftest", "-c", config, "-rs", module
])
print("CREDENTIAL_ACCESSES=" + json.dumps(accesses, sort_keys=True))
sys.exit(result)
"""


def assert_live_collection(tmp_path: Path, live_module: Path, *, prefix: str,
                           credentials: dict[str, str], case: str) -> None:
    """Exercise the real module's gate and stop before its first SDK import."""
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    credential_path = tmp_path / "synthetic-credentials.json"
    contents: dict[str, object] = dict(credentials)
    if case.startswith("missing-field-"):
        del contents[case.removeprefix("missing-field-")]
    elif case == "blank-field":
        contents[next(iter(contents))] = " \t "
    elif case == "wrong-type-field":
        contents[next(iter(contents))] = 7

    payload = json.dumps(contents).encode("utf-8")
    if case == "malformed-json":
        payload = ('{"password": "' + SECRET + '",').encode("utf-8")
    elif case == "non-utf8":
        payload += b"\xff"
    elif case == "non-object-json":
        payload = json.dumps([SECRET]).encode("utf-8")
    if case != "missing-file":
        credential_path.write_bytes(payload)

    flag = {
        "disabled-unset": None,
        "disabled-zero": "0",
        "disabled-other": "true",
    }.get(case, "1")
    configured_path: str | None = str(credential_path)
    mode, exit_code = "file", 2
    message = "must contain nonempty string fields"
    if case.startswith("disabled-"):
        mode, exit_code = "disabled", 5
        message = f"Set {prefix}_LIVE=1"
    elif case in {"missing-path", "blank-path"}:
        mode = "missing-path"
        configured_path = None if case == "missing-path" else " \t "
        message = "require explicit"
    elif case in {
            "missing-file", "unreadable-file", "malformed-json", "non-utf8"
    }:
        message = "must name a readable UTF-8 JSON file"
        if case == "unreadable-file":
            # Deterministic even when pytest itself runs as root.
            mode = "unreadable"
    elif case == "valid-inputs":
        mode = "valid"
        message = "sentinel provider import after credential validation"

    environment = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT") if key in os.environ
    }
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if flag is not None:
        environment[prefix + "_LIVE"] = flag
    if configured_path is not None:
        environment[prefix + "_CREDS"] = configured_path
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            COLLECT,
            str(config),
            str(live_module),
            str(credential_path),
            prefix + "_CREDS",
            mode,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == exit_code, output
    assert message in output, output
    for value in (*credentials.values(), SECRET):
        assert value not in output, output
    marker = "CREDENTIAL_ACCESSES="
    accesses = json.loads(
        next(
            line.removeprefix(marker) for line in output.splitlines()
            if line.startswith(marker)))
    if mode == "disabled":
        assert accesses["environment"] == 0, output
        assert "1 skipped" in output, output
    else:
        assert accesses["environment"] >= 1, output
    if mode in {"disabled", "missing-path"}:
        assert accesses["file"] == accesses["stat"] == 0, output
    else:
        assert accesses["file"] >= 1, output
    assert accesses["provider"] == int(mode == "valid"), output
