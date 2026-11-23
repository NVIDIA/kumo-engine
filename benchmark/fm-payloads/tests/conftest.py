# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def isolated_kumo_scratch(tmp_path, monkeypatch):
    import kumo.settings


    monkeypatch.setattr(kumo.settings, "_configured", tmp_path / "scratch")
