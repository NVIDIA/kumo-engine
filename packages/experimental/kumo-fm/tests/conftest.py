# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def isolated_kumo_scratch(tmp_path):
    from kumo import settings

    previous = settings.scratch_dir()
    settings.configure(scratch_dir=tmp_path / "scratch")
    try:
        yield
    finally:
        settings.configure(scratch_dir=previous)
