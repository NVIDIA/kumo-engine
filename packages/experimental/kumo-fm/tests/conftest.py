# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest


@pytest.fixture(autouse=True)
def isolated_kumo_scratch(tmp_path, monkeypatch):
    import kumo.settings

    monkeypatch.setitem(sys.modules, "duckdb", None)

    monkeypatch.setattr(kumo.settings, "_configured", tmp_path / "scratch")
