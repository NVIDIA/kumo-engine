# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""kumo-engine's version provider for `kumo_core.version`.

Registered in this package's pyproject under the `kumo_core.version_providers`
entry-point group. It is a top-level module on purpose: loading an entry point
imports its module, and anything under `kumo.engine` would import the whole
engine (and torch) just to read a version string.
"""

from typing import Optional

from kumo_core.version import installed_distribution_version


def installed_version() -> Optional[str]:
    r"""The release version of the installed `kumo-engine` wheel (the root
    `VERSION`, stamped at build time). Editable installs defer (return `None`),
    so a source checkout still reports its git revision."""
    return installed_distribution_version("kumo-engine")
