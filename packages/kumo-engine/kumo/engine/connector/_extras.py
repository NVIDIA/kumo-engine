# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Deferred imports of provider packages.

The `kumo-engine` wheel ships every provider's code, but a provider's
third-party dependencies (`snowflake-connector-python`, ...) are only installed
with its extra. Provider code must therefore be imported at the point of use,
never at module top level -- otherwise `import kumo.engine` (and
`FileConnector`) would fail on a bare install. `test_optional_imports.py`
enforces this.
"""

import importlib
from types import ModuleType


def import_provider(module: str, extra: str) -> ModuleType:
    r"""Import `module`, or say which extra installs it."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"{module!r} needs {e.name!r}, which is not installed. Install "
            f"it with: pip install 'kumo-engine[{extra}]'") from e
