# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assets for the Blueprint v2 lab notebooks: the book, the pretrained checkpoint, the config.

The notebooks read one pragmatiq book and one pretrained lab-size checkpoint
(`config.yaml`, trained on one GB10). `fetch_lab_assets` resolves
each in this order:

1. a local path: `TFM_PRAGMATIQ_ROOT` for the book, `TFM_V2_CHECKPOINT` for a
   checkpoint directory holding `model.pt` and `tokenizer.json`
   (`scripts/package_lab_assets.py` builds one);
2. an earlier download under the lab cache (`~/.cache/tfm/lab`);
3. a download from `LAB_BOOK` / `LAB_CHECKPOINT`, verified by SHA-256.

With `TFM_NOTEBOOK_SMOKE=1` (set by CI) nothing is downloaded: the book is a
small generated one (`tfm.data.synthetic.lab_book`), the model is the same
architecture, untrained, and everything runs on CPU. Every notebook cell then
executes, so API breakage is caught, but the numbers and plots are those of
an untrained model on a toy book. `LabAssets.describe()` says which mode ran.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tfm.config import RecipeConfig, load_recipe_config
from tfm.config_migration import migrate_config
from tfm.contracts.errors import (
    IncompatibleCompositionError,
    InsufficientLabelSupportError,
    SchemaError,
)
from tfm.features.tokenizers import KeyValueTimeTokenizer
from tfm.runtime.common import repo_root
from tfm.runtime.data import PreparedData, prepare_data
from tfm.runtime.evaluate import (
    ProbeTask,
    event_entity_split,
    forecast_task,
    history_window_tasks,
)
from tfm.runtime.pipeline import RunArtifacts, build_model
from tfm.training.checkpointing import load_checkpoint
from tfm.training.device import resolve_device
from tfm.training.reproducibility import seed_everything

SMOKE_ENV = "TFM_NOTEBOOK_SMOKE"
BOOK_ENV = "TFM_PRAGMATIQ_ROOT"
CHECKPOINT_ENV = "TFM_V2_CHECKPOINT"
CACHE_ENV = "TFM_LAB_CACHE"


@dataclass(frozen=True)
class LabDownload:
    """One hosted archive: a `.tar.gz` whose single top-level directory is `folder`."""

    url: str
    folder: str
    sha256: str | None = None

    @property
    def hosted(self) -> bool:
        return "TODO" not in self.url


# TODO(lab-hosting): replace both URLs and fill in `sha256` once the archives are
# hosted. `scripts/package_lab_assets.py` builds the archives and prints their hashes.
LAB_BOOK = LabDownload(
    url="https://TODO.example.com/tfm-lab/synth_100k_gen2_m31_seed0.tar.gz",
    folder="synth_100k_gen2_m31_seed0",
)
LAB_CHECKPOINT = LabDownload(
    url="https://TODO.example.com/tfm-lab/blueprint_v2_lab_tuned.tar.gz",
    folder="blueprint_v2_lab_tuned",
)

LAB_CONFIG = Path("benchmarks/ablations/blueprint_v2_final_d.yaml")
"""The lab checkpoint's config, relative to the repository: final-sweep candidate D
(3,000 pretraining steps, LoRA rank 16), whose seed 42 is the checkpoint. Smoke mode
uses it; a checkpoint directory's own `config.yaml` takes its place in lab mode."""

LAB_USERS = 4000
"""Users the lab notebooks sample from the book by default. The recorded runs sample
10,000; 4,000 prepares in about two minutes. Pass `n_users` to `fetch_lab_assets` for more."""
SMOKE_USERS = 120

LAB_TASKS = ("fraud_txn", "churn_6m", "ltv_positive", "default_12m")
"""The downstream tasks the lab scores. `fraud_txn` is the recipe's `history_window`
probe: one row per labelled card transaction of the spliced identity-theft task."""

LAB_XGBOOST_PARAMS: dict[str, Any] = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_weight": 8.0,
    "subsample": 0.75,
    "colsample_bytree": 0.6,
    "reg_lambda": 0.04,
    "reg_alpha": 0.05,
    "scale_pos_weight": 1.0,
}
"""One XGBoost setting for every lab arm and task, so the notebooks run no search.
The median of the 18 trials the recorded Optuna searches selected (raw and history
arms, three tasks, three probe splits); early stopping picks the tree count. Equal
to `evaluation.tabular_baselines.xgboost_params` in `config.yaml`, which
the recipe's own evaluation uses; a unit test keeps the two equal."""


@dataclass(frozen=True)
class LabAssets:
    """What a lab notebook runs on, and whether it is the real thing."""

    smoke: bool
    book: Path
    checkpoint_dir: Path | None
    config: RecipeConfig
    device: torch.device
    run_dir: Path
    train_steps: int
    """Steps for the notebooks' from-scratch training demonstration."""

    def describe(self) -> str:
        if self.smoke:
            return (
                f"SMOKE MODE ({SMOKE_ENV}=1): a generated {SMOKE_USERS}-user book and an "
                f"untrained model on {self.device}. Cells run; numbers and plots mean nothing."
            )
        return (
            f"Lab mode: book {self.book}, {self.config.data.max_entities} users sampled; "
            f"checkpoint {self.checkpoint_dir}; device {self.device}."
        )


def lab_cache_dir() -> Path:
    return Path(os.environ.get(CACHE_ENV, Path.home() / ".cache" / "tfm" / "lab"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(download: LabDownload, cache_dir: Path) -> Path:
    """The extracted `download.folder` under `cache_dir`, downloading it if absent."""
    target = cache_dir / download.folder
    if target.is_dir():
        return target
    if not download.hosted:
        raise FileNotFoundError(
            f"{download.folder} is not hosted yet ({download.url}). Point "
            f"{BOOK_ENV} / {CHECKPOINT_ENV} at local copies, or run with {SMOKE_ENV}=1."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
        archive = Path(tmp) / f"{download.folder}.tar.gz"
        urllib.request.urlretrieve(download.url, archive)  # noqa: S310 - fixed https URL
        if download.sha256 is not None and sha256_file(archive) != download.sha256:
            raise SchemaError(
                f"{download.url} does not match its recorded SHA-256; delete it and retry."
            )
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        extracted = Path(tmp) / download.folder
        if not extracted.is_dir():
            raise SchemaError(f"{download.url} has no top-level {download.folder}/ directory.")
        shutil.move(str(extracted), target)
    return target


def _local(env: str) -> Path | None:
    value = os.environ.get(env)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{env}={value} is not a directory.")
    return path


def checkpoint_config(path: Path) -> RecipeConfig:
    """A checkpoint's `config.yaml`, written by this release or by 2.0.0a1.

    The GTC Berlin lab ships its checkpoint with a 2.0.0a1 config;
    `migrate_config` rewrites its keys and leaves a current config unchanged.
    """
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return migrate_config(raw).validated()


def _with(config: RecipeConfig, **sections: dict[str, Any]) -> RecipeConfig:
    raw = config.model_dump(mode="json")
    for section, values in sections.items():
        raw[section].update(values)
    return RecipeConfig.model_validate(raw)


def fetch_lab_assets(
    notebook: str,
    *,
    n_users: int | None = None,
    cache_dir: Path | None = None,
    run_root: Path | None = None,
) -> LabAssets:
    """Resolve the book, checkpoint, and config for lab notebook `notebook` (e.g. `"04"`).

    Outputs go to `run_root / notebook`, by default under
    `outputs/notebooks/pragmatiq/`.
    """
    root = run_root or repo_root() / "outputs" / "notebooks" / "blueprint_v2"
    run_dir = root / notebook
    cache = cache_dir or lab_cache_dir()
    base = load_recipe_config(repo_root() / LAB_CONFIG)
    smoke = os.environ.get(SMOKE_ENV, "") not in ("", "0")
    if smoke:
        from tfm.data.synthetic.lab_book import generate_lab_book

        users = min(n_users or SMOKE_USERS, SMOKE_USERS)
        book = generate_lab_book(run_dir / "lab_book", users, seed=base.seed)
        checkpoint = None
        device = "cpu"
        batch_size, steps, precision = 8, 3, "fp32"
        finetune: dict[str, int] = {"max_steps": 6, "eval_every": 2, "batch_size": 8}
    else:
        users = n_users or LAB_USERS
        book = _local(BOOK_ENV) or fetch(LAB_BOOK, cache)
        checkpoint = _local(CHECKPOINT_ENV) or fetch(LAB_CHECKPOINT, cache)
        device = "auto"
        batch_size, steps = base.training.batch_size, 150
        precision = base.training.precision
        finetune = {}
        saved = checkpoint / "config.yaml"
        if saved.exists():
            base = checkpoint_config(saved)
    config = _with(
        base,
        data={
            "path": str(book),
            "token_cache": None,
            "pretrain_windows": None,
            "n_entities": users,
            "max_entities": users,
        },
        training={"batch_size": batch_size, "precision": precision},
        runtime={"device": device},
    )
    if finetune:
        raw = config.model_dump(mode="json")
        raw["fraud_task"]["finetune"].update(finetune)
        config = RecipeConfig.model_validate(raw)
    return LabAssets(
        smoke=smoke,
        book=book,
        checkpoint_dir=checkpoint,
        config=config,
        device=resolve_device(device),
        run_dir=run_dir,
        train_steps=steps,
    )


def prepare_lab_data(assets: LabAssets) -> PreparedData:
    """The sampled book, tokenized with the checkpoint's own tokenizer (fitted in smoke mode)."""
    tokenizer = None
    if assets.checkpoint_dir is not None:
        state = json.loads((assets.checkpoint_dir / "tokenizer.json").read_text(encoding="utf-8"))
        tokenizer = KeyValueTimeTokenizer.from_state(state)
    return prepare_data(assets.config, fixture_dir=assets.run_dir / "fixture", tokenizer=tokenizer)


def load_lab_model(assets: LabAssets, data: PreparedData, *, pretrained: bool = True) -> nn.Module:
    """The lab model: the checkpoint's weights, or (with `pretrained=False`, and always in
    smoke mode) the same architecture at its seeded random initialization."""
    seed_everything(assets.config.seed)
    model = build_model(assets.config, data.tokenizer.vocab_size)
    if pretrained and assets.checkpoint_dir is not None:
        manifest = json.loads(
            (assets.checkpoint_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
        )
        if int(manifest["vocab_size"]) != data.tokenizer.vocab_size:
            raise IncompatibleCompositionError(
                f"The checkpoint's vocabulary has {manifest['vocab_size']} tokens but the "
                f"tokenizer has {data.tokenizer.vocab_size}; they were not saved together."
            )
        load_checkpoint(model, assets.checkpoint_dir / "model.pt")
    return model.to(assets.device).eval()


def lab_tasks(
    assets: LabAssets, data: PreparedData, names: Sequence[str] = LAB_TASKS
) -> dict[str, ProbeTask]:
    """Each named task's entity-disjoint probe rows, in `names` order.

    `fraud_txn` uses the spliced task's entity split; each forecast label has
    its own seeded split. Raises `InsufficientLabelSupportError` for a task the
    sample cannot support rather than dropping it.
    """
    config = assets.config
    tasks: dict[str, ProbeTask] = {}
    for name in names:
        if name == "fraud_txn":
            probes = history_window_tasks(data, config, event_entity_split(data, config.seed))
            task = probes.get("history_window")
            reason = "the sample has no labelled card transactions on one side"
        elif name in data.label_rows:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                task = forecast_task(data, config, name, data.label_rows[name], seed=config.seed)
            reason = "; ".join(str(w.message) for w in caught) or "no rows"
        else:
            raise InsufficientLabelSupportError(
                f"The book has no {name!r} labels; it has {sorted(data.label_rows)}."
            )
        if task is None:
            raise InsufficientLabelSupportError(f"Cannot build the {name!r} probe: {reason}.")
        tasks[name] = task
    return tasks


def lab_run(assets: LabAssets, data: PreparedData, model: nn.Module) -> RunArtifacts:
    """A `RunArtifacts` over the lab data and a model, for `run_train` / `run_evaluate`."""
    assets.run_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_dir=assets.run_dir,
        config=assets.config,
        data=data,
        model=model,
        device=assets.device,
        recipe_name="blueprint_v2",
    )


__all__ = [
    "BOOK_ENV",
    "CACHE_ENV",
    "CHECKPOINT_ENV",
    "LAB_BOOK",
    "LAB_CHECKPOINT",
    "LAB_CONFIG",
    "LAB_TASKS",
    "LAB_USERS",
    "LAB_XGBOOST_PARAMS",
    "SMOKE_ENV",
    "SMOKE_USERS",
    "LabAssets",
    "LabDownload",
    "checkpoint_config",
    "fetch",
    "sha256_file",
    "fetch_lab_assets",
    "lab_cache_dir",
    "lab_tasks",
    "load_lab_model",
    "prepare_lab_data",
]
