# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

import kumo.settings


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.setattr(kumo.settings, "_configured", None)
    monkeypatch.setattr(kumo.settings, "_DEFAULT_SCRATCH_DIR", tmp_path / "default")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def test_configure_sets_the_scratch_dir_and_creates_nothing(tmp_path):
    assert kumo.settings.scratch_dir() == tmp_path / "default"
    kumo.settings.configure(scratch_dir=Path("~/configured"))
    assert kumo.settings.scratch_dir() == tmp_path / "home" / "configured"
    kumo.settings.configure(scratch_dir=None)
    assert kumo.settings.scratch_dir() == tmp_path / "home" / "configured"
    assert not (tmp_path / "home").exists()


@pytest.mark.parametrize("path", ["s3://bucket/scratch", "@stage/scratch"])
def test_remote_paths_are_rejected(path):
    with pytest.raises(ValueError, match="scratch_dir must be a local filesystem path"):
        kumo.settings.configure(scratch_dir=path)
