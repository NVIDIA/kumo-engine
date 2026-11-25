# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe resolution and lifecycle dispatch behind `tfm <stage> --recipe <slug>`.

A recipe is a directory of configs under `recipes/<slug>/`, listed in
`recipes/catalog.yaml`. `--config` names a variant (`smoke` reads
`config.smoke.yaml`) or a path; without it the recipe's `config.yaml` runs,
and `smoke` runs `config.smoke.yaml`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "resolved_config.yaml",
    "data_validation_report.json",
    "split_manifest.json",
    "tokenizer_manifest.json",
    "checkpoint_manifest.json",
    "embeddings_manifest.json",
    "metrics.json",
    "metrics.md",
    "model_card.md",
    "environment.json",
)

COMMANDS: tuple[str, ...] = ("train", "infer", "evaluate", "report", "smoke")


def catalog_recipes() -> tuple[str, ...]:
    """Recipe slugs in `recipes/catalog.yaml`, in catalog order."""
    from tfm.runtime.common import repo_root

    catalog = yaml.safe_load((repo_root() / "recipes" / "catalog.yaml").read_text("utf-8"))
    return tuple(row["id"] for row in catalog["recipes"])


def resolve_config(recipe: str | None, config: str | Path | None, *, command: str) -> Path | None:
    """The config file a command runs, or `None` for the default recipe's smoke config."""
    from tfm.runtime.pipeline import default_config_path

    if recipe is None:
        if config is None:
            return None
        return Path(config)
    known = catalog_recipes()
    if recipe not in known:
        raise SystemExit(f"Unknown recipe {recipe!r}; recipes/catalog.yaml lists {list(known)}.")
    if config is not None and Path(config).suffix in {".yaml", ".yml"}:
        return Path(config)
    if config is None:
        variant = "smoke" if command == "smoke" else None
    else:
        variant = None if str(config) == "default" else str(config)
    try:
        return default_config_path(recipe, variant)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None


def run_command(
    command: str,
    *,
    config_path: Path | None,
    run_dir: Path | None,
    device: str | None,
    reuse_checkpoint: bool = False,
) -> Path:
    """Run one lifecycle command, including the stages it depends on.

    Each command is self-contained, so any stage can run without a prior run
    directory. `reuse_checkpoint` skips training when `run_dir/model.pt`
    exists, so a trained run can be re-evaluated.
    """
    from tfm.config import load_recipe_config
    from tfm.runtime.pipeline import default_config_path
    from tfm.runtime.registry import recipe_runner

    if command not in COMMANDS:
        raise ValueError(f"Unknown recipe command {command!r}; use one of {list(COMMANDS)}.")
    source = config_path if config_path is not None else default_config_path()
    config = load_recipe_config(source)
    result = recipe_runner(config.recipe_family)(
        command,
        config=config,
        config_path=source,
        run_dir=run_dir,
        device=device,
        reuse_checkpoint=reuse_checkpoint,
    )
    if command == "smoke":
        missing = [name for name in REQUIRED_ARTIFACTS if not (result / name).exists()]
        if missing:
            raise SystemExit(f"missing artifacts: {missing}")
        print(f"smoke ok: {result}")
    return result


__all__ = ["COMMANDS", "REQUIRED_ARTIFACTS", "catalog_recipes", "resolve_config", "run_command"]
