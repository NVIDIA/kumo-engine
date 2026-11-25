# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`tfm cache-tokens` sets a finished build's time beside its `--profile` projection."""

import json
from pathlib import Path

import pytest

from tfm.cli.main import _projection

MANIFEST = {"build": {"seconds": 10}, "n_users": 5}
REPORT = {
    "workers": 2,
    "rows_per_block": 40,
    "identity": {"n_users": 5},
    "seconds": {"read_twice": {"projected": 9, "low": 8, "high": 11}},
}


def test_a_matching_report_gives_the_projection(tmp_path: Path) -> None:
    path = tmp_path / "cache.profile.json"
    path.write_text(json.dumps(REPORT))
    want = "build took 10 s; --profile projected 9 s (range 8-11 s)"
    assert _projection(path, MANIFEST, 2, 40) == want
    assert _projection(path, MANIFEST, 3, 40) is None


@pytest.mark.parametrize(
    "text",
    [json.dumps(REPORT)[:30], "null", json.dumps({**REPORT, "identity": None})],
    ids=["truncated", "null", "null identity"],
)
def test_an_unreadable_report_does_not_fail_the_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str
) -> None:
    path = tmp_path / "cache.profile.json"
    path.write_text(text)
    assert _projection(path, MANIFEST, 2, 40) is None
    assert "ignoring unreadable profile report" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["unicode", "io"])
def test_optional_profile_report_read_errors_do_not_fail_a_build(
    tmp_path, capsys, monkeypatch, failure
):
    path = tmp_path / "profile.json"
    path.write_bytes(b"\xff")
    if failure == "io":

        def inaccessible(*args, **kwargs):
            raise OSError("unreadable")

        monkeypatch.setattr(Path, "read_text", inaccessible)
    assert _projection(path, {}, 2, 40) is None
    assert "ignoring unreadable profile report" in capsys.readouterr().err
