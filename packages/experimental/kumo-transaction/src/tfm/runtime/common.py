# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle pieces shared by every recipe runtime.

Recipe runtimes differ in their data contract and architecture, not in their
run lifecycle: resolve config, write manifests, train, extract embeddings,
evaluate, report. That shared shape lives here so each stack only implements
what is genuinely different.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from tfm.config import RecipeConfig
from tfm.data.splits.temporal import parse_duration as parse_duration
from tfm.data.splits.temporal import parse_timestamp as parse_timestamp
from tfm.evaluation.reporting.report import write_json
from tfm.inference.embedding import EmbeddingTable
from tfm.training.backends import backend_capabilities, check_backend_supports
from tfm.training.checkpointing import save_checkpoint
from tfm.training.device import device_environment


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_provenance() -> dict[str, str | bool | None]:
    """Source-checkout identity independent of cwd; unknown in installed/no-Git builds."""
    unknown: dict[str, str | bool | None] = {"source_revision": None, "source_dirty": None}
    root = repo_root()
    if not (root / "pyproject.toml").is_file() or not (root / "src/tfm").is_dir():
        return unknown
    # These commands only inspect local source. Inherited repository/config
    # overrides must not redirect any command into another checkout or index.
    git_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        checkout = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            env=git_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if checkout.returncode or Path(checkout.stdout.strip()).resolve() != root:
            return unknown
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            env=git_env,
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            env=git_env,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return unknown
    if revision.returncode or dirty.returncode:
        return unknown
    return {"source_revision": revision.stdout.strip(), "source_dirty": bool(dirty.stdout.strip())}


def recipe_dir(recipe: str) -> Path:
    return repo_root() / "recipes" / recipe


def default_run_dir(root: Path | None = None, *, recipe: str) -> Path:
    base = root if root is not None else Path("outputs") / recipe
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return base / stamp


def apply_device_override(config: RecipeConfig, device: str | None) -> RecipeConfig:
    if device is None:
        return config
    return config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"device": device})}
    )


def write_resolved_config(run_dir: Path, resolved: dict[str, Any]) -> None:
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )


def write_environment_manifest(run_dir: Path, config: RecipeConfig, device: torch.device) -> None:
    write_json(
        run_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "requested_device": config.runtime.device,
            **device_environment(device),
        },
    )


def write_embeddings_manifest(
    run_dir: Path,
    tables: dict[str, EmbeddingTable],
    *,
    alignment_fields: list[str],
) -> None:
    train = tables["train"]
    write_json(
        run_dir / "embeddings_manifest.json",
        {
            "dim": int(train.vectors.shape[1]) if train.vectors.size else 0,
            "n_train": len(train.entity_ids),
            "n_validation": len(tables["validation"].entity_ids),
            "n_test": len(tables["test"].entity_ids),
            "alignment_fields": alignment_fields,
            "files": {
                "embeddings.npz": {
                    "vectors": sorted(tables),
                    "labels": [f"{side}_labels" for side in sorted(tables)],
                    "ids": ["entity_ids", "sequence_ids", "event_ids"],
                    "id_keys": "<side>_<id>, row for row with <side>",
                }
            },
        },
    )


def check_backend_for_run(
    config: RecipeConfig, device: torch.device, *, layout: str | None = None
) -> None:
    """Refuse a configuration the selected backend cannot execute (backend contracts)."""
    check_backend_supports(
        backend_capabilities(config.training.backend),
        backbone=config.model.backbone.strip().lower(),
        packing=config.training.packing,
        device_type=device.type,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        layout=layout,
        precision=config.training.precision,
    )


_STARTED = time.perf_counter()


def stage(message: str) -> None:
    """Print one stage of a recipe job, with the time since the job started.

    Only the primary process prints, so a multi-GPU job reads as one log.
    """
    if not is_primary_process():
        return
    minutes, seconds = divmod(round(time.perf_counter() - _STARTED), 60)
    print(f"[{minutes:3d}m{seconds:02d}s] {message}", flush=True)


def is_primary_process() -> bool:
    """Whether this process owns the run directory: rank 0 under a launcher, else always.

    Every rank of a `torchrun` job executes the same recipe, so without this gate
    each one writes the same files concurrently and interleaved writes of
    per-rank content produce invalid JSON.
    """
    return int(os.environ.get("RANK", "0")) == 0


def save_run_checkpoint(
    model: torch.nn.Module,
    run_dir: Path,
    metadata: dict[str, Any],
    stats: dict[str, Any],
) -> Path:
    """Write the eager checkpoint and training stats from the primary process only.

    A sharded run's parameters are per-rank shards, so `model.pt` would hold one
    rank's slice. Its consolidated weights are the backend's HuggingFace export,
    which the manifest points to instead. A replicated (DDP) run's rank 0 holds
    the full model and writes `model.pt` as usual.
    """
    ckpt = run_dir / "model.pt"
    if not is_primary_process():
        return ckpt
    backend = str(stats.get("backend", "eager"))
    sharded = (
        float(stats.get("world_size", 1.0)) > 1.0
        and backend_capabilities(backend).shards_parameters
    )
    if sharded:
        manifest = {
            **metadata,
            "path": str(stats.get("hf_checkpoint", "")),
            "format": "huggingface",
        }
    else:
        save_checkpoint(model, ckpt, metadata=metadata)
        manifest = {**metadata, "path": str(ckpt)}
    write_json(run_dir / "checkpoint_manifest.json", manifest)
    write_json(run_dir / "train_stats.json", stats)
    return ckpt


def resolve_report_metrics(
    run_dir: Path, metrics: dict[str, dict[str, float]] | None
) -> dict[str, dict[str, float]]:
    """Use provided metrics, else re-read the metrics written by evaluate."""
    if metrics is not None:
        return metrics
    loaded = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("metrics.json must contain an object")
    return loaded
