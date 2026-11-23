# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint and tokenizer metadata manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

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


class CheckpointPayload(TypedDict):
    """Weights and their authoritative embedded metadata, loaded together once."""

    state_dict: dict[str, Any]
    metadata: dict[str, Any]


def read_checkpoint(path: Path) -> CheckpointPayload:
    """Read a trusted local artifact without installing its state into a model."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Checkpoint requires a state_dict mapping")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a mapping")
    return {"state_dict": payload["state_dict"], "metadata": metadata}


def load_checkpoint(model: nn.Module, path: Path) -> dict[str, Any]:
    payload = read_checkpoint(path)
    model.load_state_dict(payload["state_dict"])
    return payload["metadata"]


def save_encoder_state(model: nn.Module, path: str | Path, *, metadata: dict[str, Any]) -> None:
    """Write a plain torch state_dict plus a manifest bound to its exact bytes.

    A user's model can load the weight file directly with torch.load(weights_only=True).
    Managed downstream runs additionally verify the feature/schema manifest.
    """
    import hashlib
    import json

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    manifest = {
        **metadata,
        "format": "tfm.graph-encoder.v1",
        "weights_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_encoder_state(
    model: nn.Module, path: str | Path, *, expected: dict[str, Any]
) -> dict[str, Any]:
    """Validate a plain encoder artifact's bound schema before installing weights."""
    import hashlib
    import io
    import json

    source = Path(path)
    raw = source.read_bytes()
    metadata = json.loads(source.with_suffix(".manifest.json").read_text())
    integrity = {
        "format": "tfm.graph-encoder.v1",
        "weights_sha256": hashlib.sha256(raw).hexdigest(),
    }
    for required in (integrity, expected):
        for key, value in required.items():
            if json.dumps(metadata.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
                raise ValueError(
                    f"Encoder checkpoint {key} differs from requested schema or weights"
                )
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return metadata
