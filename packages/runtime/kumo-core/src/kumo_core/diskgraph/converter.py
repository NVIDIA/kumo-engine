# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DiskGraph config conversion with an isolated provider environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional


class DiskgraphConfigConversionError(RuntimeError):
    """Config-converter failure with separately classified diagnostics."""

    _kumo_restricted_log_message = True

    def __init__(self, returncode: int, stderr: str) -> None:
        safe_message = ("DiskGraph config conversion subprocess failed"
                        f" (exit code {returncode})")
        self._kumo_safe_log_message = safe_message
        detail = stderr.strip()
        if detail:
            detail = f": {detail[-2000:]}"
        super().__init__(safe_message + detail)


def _convert(request: Mapping[str, Any]) -> dict[str, Any]:
    from diskgraph import DiskgraphConfigConverter

    return DiskgraphConfigConverter(
        request["graph_config"],
        request["data_persist_dir"],
        request["table_data_versions"],
        source_paths=request.get("source_paths"),
        source_specs=request.get("source_specs"),
    ).convert()


def convert_diskgraph_config(
    graph_config: dict[str, Any],
    data_persist_dir: str,
    table_data_versions: Mapping[str, str],
    *,
    source_paths: Optional[Mapping[str, str]] = None,
    source_specs: Optional[Mapping[str, Any]] = None,
    subprocess_environment: Optional[Mapping[str, Optional[str]]] = None,
) -> dict[str, Any]:
    request = {
        "graph_config": graph_config,
        "data_persist_dir": data_persist_dir,
        "table_data_versions": dict(table_data_versions),
        "source_paths": dict(source_paths) if source_paths else None,
        "source_specs": dict(source_specs) if source_specs else None,
    }
    if not subprocess_environment:
        return _convert(request)

    with tempfile.TemporaryDirectory(prefix="kumo-diskgraph-convert-") as root:
        request_path = Path(root) / "request.json"
        result_path = Path(root) / "result.json"
        request_path.write_text(json.dumps(request))
        child_environment = os.environ.copy()
        for name, value in subprocess_environment.items():
            if value is None:
                child_environment.pop(name, None)
            else:
                child_environment[name] = value
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "kumo_core.diskgraph.converter",
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ],
            env=child_environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise DiskgraphConfigConversionError(
                completed.returncode,
                completed.stderr,
            )
        return json.loads(result_path.read_text())


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    Path(args.result).write_text(json.dumps(_convert(request)))


if __name__ == "__main__":
    _main()
