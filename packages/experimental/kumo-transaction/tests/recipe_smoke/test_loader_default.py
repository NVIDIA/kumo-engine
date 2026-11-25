# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default native recipes and explicit dependency-light Python retrieval."""

from pathlib import Path

import pytest
import yaml

from tfm.cli.recipe import run_command


def test_explicit_python_sequence_recipe_without_native(monkeypatch, tmp_path):
    from tfm.data import native_history
    from tfm.extras import kumo_pytorch

    def forbidden():
        raise AssertionError("Explicit Python selection must not initialize native retrieval")

    monkeypatch.setattr(native_history, "native_api", forbidden)
    monkeypatch.setattr(kumo_pytorch, "native_api", forbidden)
    config = yaml.safe_load(Path("recipes/blueprint_v1/config.smoke.yaml").read_text())
    config["data"]["loader"] = "python"
    path = tmp_path / "python.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "run", device="cpu")
    assert (out / "model.pt").exists()


@pytest.mark.parametrize("name", ["blueprint_v1", "blueprint_v2", "pragma_like"])
def test_sequence_recipe_uses_native_by_default(name, monkeypatch, tmp_path):
    pytest.importorskip("kumo.pytorch.api", reason="requires the native extra")
    from tfm.data import native_history

    actual = native_history.temporary_engine
    calls = []

    def observed(*args, **kwargs):
        calls.append(True)
        return actual(*args, **kwargs)

    monkeypatch.setattr(native_history, "temporary_engine", observed)
    out = run_command(
        "smoke",
        config_path=Path("recipes") / name / "config.smoke.yaml",
        run_dir=tmp_path / "run",
        device="cpu",
    )
    assert calls
    assert (out / "model.pt").exists()
    resolved = yaml.safe_load((out / "resolved_config.yaml").read_text())
    assert resolved["data"]["loader"] == "kumo_pytorch"
