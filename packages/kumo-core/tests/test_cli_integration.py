# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path


def test_legacy_cli_delegates_training_flow_to_shared_callable():
    cli = Path(__file__).parents[3] / "bin" / "kumo-auto"
    source = cli.read_text(encoding="utf-8")

    assert "run_callable_training(" in source
    assert "auto_trainer.run(" not in source
    assert "best_trainer.test(" not in source
    assert "searcher.analyze(" not in source


def test_value_type_import_does_not_load_platform_runtime():
    # `kumo` is a PEP 420 namespace package (no __init__.py) so it can
    # coexist with the SDK's `kumo.engine` portion in the same venv;
    # callers get value types directly from kumo_core.types /
    # kumoapi.typing instead of a `kumo.TaskType`-style root re-export.
    repository = Path(__file__).parents[3]
    core_source = repository / "packages" / "kumo-core" / "src"
    script = """
import importlib.abc
import sys

sys.path[:0] = sys.argv[1:]
blocked = {
    'pool', 'rest', 'service_framework', 'statedb', 'temporalio_commons',
}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in blocked:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, Blocker())
from kumo_core.types import TaskType
assert TaskType.REGRESSION.is_node_pred
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(core_source),
            str(repository),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
