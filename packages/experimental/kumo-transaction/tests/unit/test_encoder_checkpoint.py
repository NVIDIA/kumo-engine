# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from tfm.training.checkpointing import load_encoder_state, save_encoder_state


@pytest.mark.parametrize("refresh_manifest", [False, True])
def test_full_expected_manifest_cannot_override_computed_encoder_checksum(
    tmp_path, monkeypatch, refresh_manifest
):
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "encoder.pt"
    save_encoder_state(model, path, metadata={"schema": "fixture"})
    expected = json.loads(path.with_suffix(".manifest.json").read_text())
    state = torch.load(path, weights_only=True)
    state["weight"].add_(1)
    torch.save(state, path)
    if refresh_manifest:
        import hashlib

        refreshed = {**expected, "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        path.with_suffix(".manifest.json").write_text(json.dumps(refreshed))

    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before installing state")

    monkeypatch.setattr(model, "load_state_dict", forbidden)
    with pytest.raises(ValueError, match="weights_sha256"):
        load_encoder_state(model, path, expected=expected)
