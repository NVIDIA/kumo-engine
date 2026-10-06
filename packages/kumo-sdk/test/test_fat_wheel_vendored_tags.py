# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the fat wheel's dependency-source and provider-extra contracts."""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / 'packages/kumo-engine/scripts/build_fat_wheel.sh'


def _script() -> str:
    return SCRIPT.read_text(encoding='utf-8')


def _selected_component_extras() -> set[str]:
    match = re.search(r"provider_extras = \{([^}]+)\}", _script())
    assert match, 'build_fat_wheel.sh no longer selects component extras'
    return set(re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)))


def test_fat_wheel_uses_released_torch_geometric() -> None:
    script = _script()
    assert ('TORCH_GEOMETRIC_VERSION="${TORCH_GEOMETRIC_VERSION:-2.8.0.post1}"'
            in script)
    assert 'torch-geometric==${TORCH_GEOMETRIC_VERSION}' in script
    assert 'KUMO_WHEELS=' not in script


def test_local_backend_selects_provider_sdk_dependencies() -> None:
    """The moved Snowflake backend imports dependencies from its SDK extra."""
    assert 'sdk' in _selected_component_extras()
