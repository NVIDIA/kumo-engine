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
    monkeypatch.delenv(kumo.settings.SCRATCH_DIR_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def test_resolution_order(monkeypatch, tmp_path):
    assert kumo.settings.scratch_dir() == tmp_path / "default"
    monkeypatch.setenv(kumo.settings.SCRATCH_DIR_ENV, "")
    assert kumo.settings.scratch_dir() == tmp_path / "default"
    monkeypatch.setenv(kumo.settings.SCRATCH_DIR_ENV, "~/env")
    assert kumo.settings.scratch_dir() == tmp_path / "home" / "env"
    kumo.settings.configure(scratch_dir=tmp_path / "configured")
    assert kumo.settings.scratch_dir() == tmp_path / "configured"
    with kumo.settings.scoped(scratch_dir=str(tmp_path / "outer")) as outer:
        with pytest.raises(RuntimeError):
            with kumo.settings.scoped(scratch_dir=tmp_path / "inner"):
                assert kumo.settings.scratch_dir() == tmp_path / "inner"
                raise RuntimeError
        assert outer == kumo.settings.scratch_dir() == tmp_path / "outer"
    kumo.settings.configure(scratch_dir=None)
    assert kumo.settings.scratch_dir() == tmp_path / "configured"


def test_resolving_creates_nothing(monkeypatch, tmp_path):
    kumo.settings.configure(scratch_dir=Path("~/configured"))
    assert kumo.settings.scratch_dir() == tmp_path / "home" / "configured"
    with kumo.settings.scoped(scratch_dir=tmp_path / "scoped"):
        kumo.settings.scratch_dir()
    assert not (tmp_path / "home").exists() and not (tmp_path / "scoped").exists()


@pytest.mark.parametrize("path", ["s3://bucket/scratch", "@stage/scratch"])
def test_remote_paths_are_rejected(monkeypatch, path):
    with pytest.raises(ValueError, match="scratch_dir must be a local filesystem path"):
        kumo.settings.configure(scratch_dir=path)
    with pytest.raises(ValueError, match="scratch_dir must be a local filesystem path"):
        with kumo.settings.scoped(scratch_dir=path):
            pass
    monkeypatch.setenv(kumo.settings.SCRATCH_DIR_ENV, path)
    with pytest.raises(ValueError, match="KUMO_SCRATCH_DIR must be a local filesystem path"):
        kumo.settings.scratch_dir()
