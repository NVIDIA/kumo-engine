# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The Kumo platform's version providers for `kumo_core.version`.

Both are registered by the root `setup.py` in the `kumo_core.version_providers`
entry-point group, and must stay cheap to import:

* `container-file`: platform images record the image version (a commit SHA or
  `nightly-<date>`) in `~/kumo_version.txt`.
* `release-version`: the release version in the root `VERSION` file, stamped
  into :data:`RELEASE_VERSION` by `scripts/stamp_version.py`, for an installed
  (non-editable) platform with no version file. Source checkouts defer, so they
  still report their git revision.
"""

import os.path as osp
from typing import Optional

from kumo_core.version import installed_distribution_version

VERSION_FILE = '~/kumo_version.txt'
#: Stamped from the root VERSION file; do not edit by hand.
RELEASE_VERSION = "1.0.0.dev0"


def container_version() -> Optional[str]:
    r"""The version recorded in the container's `kumo_version.txt`, or `None`
    if there is no such file or it is empty."""
    path = osp.expanduser(VERSION_FILE)
    if osp.exists(path):
        with open(path, 'r') as f:
            version = f.read().strip()
            if len(version) > 0:
                return version
    return None


def release_version() -> Optional[str]:
    r"""The platform's release version, or `None` for an editable install."""
    if installed_distribution_version('kumo') is None:
        return None
    return RELEASE_VERSION
