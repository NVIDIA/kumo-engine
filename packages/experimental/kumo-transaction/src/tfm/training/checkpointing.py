# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint and tokenizer metadata manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    manifest = path.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def load_checkpoint(model: nn.Module, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return payload.get("metadata", {})
