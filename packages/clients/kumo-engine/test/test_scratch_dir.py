# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import kumo.settings
from kumo.engine import configure


def test_engine_configure_sets_kumo_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(kumo.settings, '_configured', None)
    configure(scratch_dir=tmp_path / 'configured')
    assert kumo.settings.scratch_dir() == tmp_path / 'configured'
