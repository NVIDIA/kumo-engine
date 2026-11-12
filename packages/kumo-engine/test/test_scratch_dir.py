# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import kumo.settings
from kumo.engine import configure
from kumo.engine._scratch import get_scratch_dir


def test_engine_scratch_dir_follows_kumo_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(kumo.settings, '_configured', None)
    configure(scratch_dir=tmp_path / 'configured')
    assert get_scratch_dir() == kumo.settings.scratch_dir() == tmp_path / 'configured'
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'patched')
    assert get_scratch_dir() == tmp_path / 'patched'
