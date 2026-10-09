# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest

_SOURCE = """
import kumo_core.io.filesystem
import fsspec
from gcsfs.core import GCSFileSystem
assert fsspec.get_filesystem_class("gs") is GCSFileSystem
"""


def test_gs_resolves_to_classic_gcsfs() -> None:
    pytest.importorskip("gcsfs")
    env = dict(os.environ)
    env.pop("GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", None)
    subprocess.run([sys.executable, "-c", _SOURCE], check=True, env=env)
