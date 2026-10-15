# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the engine wheel's dependency-source and provider-extra contracts."""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / 'packages/kumo-engine/scripts/build_wheel.sh'
ENGINE_PYPROJECT = REPO_ROOT / 'packages/kumo-engine/pyproject.toml'


def _script() -> str:
    return SCRIPT.read_text(encoding='utf-8')


def test_wheel_leaves_torch_geometric_to_installer() -> None:
    script = _script()
    assert 'TORCH_GEOMETRIC_VERSION' not in script
    assert '--torch-geometric' not in script
    assert "record 'torch_geometric-*.whl'" not in script


def test_snowflake_extra_selects_provider_sdk_dependencies() -> None:
    """The moved Snowflake backend imports dependencies from its SDK extra.

    merge_wheels.py turns each kumo-engine extra into the dependencies of the
    bundled provider it names, so the extra has to request `sdk`.
    """
    pyproject = ENGINE_PYPROJECT.read_text(encoding='utf-8')
    assert re.search(r'^snowflake=\[\s*"kumo-snowflake\[sdk\]', pyproject,
                     re.MULTILINE)
