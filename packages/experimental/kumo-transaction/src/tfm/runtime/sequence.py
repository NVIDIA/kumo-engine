# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequence recipe lifecycle, using the same parsed configuration as the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tfm.config import RecipeConfig


def run_command(
    command: str,
    *,
    config: RecipeConfig,
    config_path: Path | None,
    run_dir: Path | None,
    device: str | None,
    reuse_checkpoint: bool = False,
) -> Path:
    if config.data.relational is not None:
        from tfm.runtime.relational import run_command as run_relational

        return run_relational(
            command,
            config=config,
            config_path=config_path,
            run_dir=run_dir,
            device=device,
            reuse_checkpoint=reuse_checkpoint,
        )
    if config.data.loader == "kumo_pytorch" and config.data.token_cache is None:
        from tfm.runtime.native_sequence import run_command as run_native

        return run_native(
            command,
            config=config,
            config_path=config_path,
            run_dir=run_dir,
            device=device,
            reuse_checkpoint=reuse_checkpoint,
        )
    from tfm.runtime import pipeline
    from tfm.runtime.common import is_primary_process, stage

    if config.runtime.stage != "full":
        raise ValueError("Explicit pretrain/downstream stages require a runner supporting them")
    # `train` needs only the pretraining corpus: with a flat-field token cache it
    # is read from disk and no rank loads the book's events.
    artifacts = pipeline.setup_run(
        config=config,
        recipe=config_path.resolve().parent.name if config_path is not None else None,
        run_dir=run_dir,
        device=device,
        pretraining_only=command == "train",
    )
    reuse = reuse_checkpoint and command != "train" and (artifacts.run_dir / "model.pt").exists()
    if reuse:
        stage(f"train: reusing {artifacts.run_dir / 'model.pt'}")
        stats: dict[str, Any] = {"final_loss": float("nan")}
    else:
        stats = pipeline.run_train(artifacts)
    if not is_primary_process():
        # Data-parallel ranks exist to train. Rank 0 alone infers, evaluates,
        # and reports, so artifacts are written once.
        return artifacts.run_dir
    if command == "train":
        stage(f"done: wrote {artifacts.run_dir} final_loss={float(stats['final_loss']):.4f}")
        return artifacts.run_dir

    tables = pipeline.run_infer(artifacts)
    if command == "infer":
        print(f"wrote embeddings under {artifacts.run_dir}")
        print({split: table.vectors.shape for split, table in tables.items()})
        return artifacts.run_dir

    metrics = pipeline.run_evaluate(artifacts, tables)
    if command == "evaluate":
        print(metrics)
        return artifacts.run_dir

    pipeline.run_report(artifacts, metrics)
    stage(f"done: wrote {artifacts.run_dir / 'model_card.md'}")
    return artifacts.run_dir
