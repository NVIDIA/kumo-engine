# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import kumo_core.train.checkpoint as checkpoint


def test_load_trusted_checkpoint_with_full_metadata(monkeypatch) -> None:
    calls = []

    def load(path, *, map_location, weights_only):
        calls.append((path, map_location, weights_only))
        return {"state_dict": {}}

    monkeypatch.setattr(checkpoint, "pl_load", load)

    result = checkpoint.load_kumo_checkpoint(
        Path("model.ckpt"),
        map_location="cpu",
    )

    assert result == {"state_dict": {}}
    assert calls == [(Path("model.ckpt"), "cpu", False)]
