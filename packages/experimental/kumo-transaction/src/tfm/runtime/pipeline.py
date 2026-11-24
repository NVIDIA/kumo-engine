# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The recipe lifecycle, one path for every stack (ADR 0014).

setup -> train -> infer -> evaluate -> report. The config's slots decide what
is built; this module only sequences the stages and writes run artifacts:

- data and tokenizer: `tfm.runtime.data.prepare_data`
- model: `tfm.api.builders.model_from_config`, plus an optional flat backbone
- adaptation and probes: `tfm.runtime.evaluate.evaluate_run`
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from tfm.api.backbones import build_backbone
from tfm.api.builders import model_from_config
from tfm.api.model import TrxFM
from tfm.api.validation import validate_tokenizer_layout
from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.errors import IncompatibleCompositionError, UnsupportedCombinationError
from tfm.contracts.kvt import EventArrayReader
from tfm.contracts.tokenizers import (
    FieldVocabulary,
    FieldVocabularyTokenizer,
    SequenceTokenizer,
)
from tfm.data.collate import collate_batch, minibatches, shuffled_rows
from tfm.data.packing import pack_sequences
from tfm.data.token_cache.formats import cache_format_of
from tfm.evaluation.reporting.report import metrics_markdown, model_card_markdown, write_json
from tfm.features.tokenizers.registry import tokenizer_class
from tfm.inference.embedding import EmbeddingTable, extract_embeddings
from tfm.objectives.supervised import SupervisedHead
from tfm.runtime.common import (
    apply_device_override,
    check_backend_for_run,
    default_run_dir,
    is_primary_process,
    recipe_dir,
    resolve_report_metrics,
    save_run_checkpoint,
    stage,
    write_embeddings_manifest,
    write_environment_manifest,
    write_resolved_config,
)
from tfm.runtime.data import (
    PreparedData,
    PretrainingData,
    prepare_data,
    prepare_pretraining,
    tokenizer_for,
)
from tfm.runtime.evaluate import evaluate_run
from tfm.training.checkpointing import load_checkpoint
from tfm.training.device import resolve_device
from tfm.training.progress import TrainingProgress
from tfm.training.reproducibility import seed_everything
from tfm.training.trainer import build_trainer

DEFAULT_RECIPE = "blueprint_v1"


@dataclass
class RunArtifacts:
    """One run: its config, prepared data, model, and output directory.

    `data` is `PretrainingData` when the run was set up to train only from a
    flat-field token cache; such a run has no probe batches.
    """

    run_dir: Path
    config: RecipeConfig
    data: PreparedData | PretrainingData
    model: nn.Module
    device: torch.device
    recipe_name: str = DEFAULT_RECIPE

    @property
    def tokenizer(self) -> SequenceTokenizer:
        return self.data.tokenizer

    @property
    def prepared(self) -> PreparedData:
        """The full data stage; refused for a run set up to pretrain only."""
        if not isinstance(self.data, PreparedData):
            raise UnsupportedCombinationError(
                "This run was set up to pretrain only (setup_run(pretraining_only=True)); "
                "it holds no events or probe batches. Set it up again to infer or evaluate."
            )
        return self.data

    @property
    def train_batch(self) -> Any:
        return self.prepared.batches[0]

    @property
    def val_batch(self) -> Any:
        return self.prepared.batches[1]

    @property
    def test_batch(self) -> Any:
        return self.prepared.batches[2]


def default_config_path(recipe: str = DEFAULT_RECIPE, variant: str | None = "smoke") -> Path:
    """`recipes/<recipe>/config.<variant>.yaml`, or `config.yaml` when `variant` is None.

    The default is the CPU smoke config, so a bare `setup_run()` never starts a
    GPU-sized run.
    """
    name = "config.yaml" if variant is None else f"config.{variant}.yaml"
    path = recipe_dir(recipe) / name
    if not path.is_file():
        found = sorted(p.name for p in recipe_dir(recipe).glob("config*.yaml"))
        raise FileNotFoundError(f"Recipe {recipe!r} has no {name}; it has {found}.")
    return path


def build_model(
    config: RecipeConfig, vocab_size: int, *, tokenizer: SequenceTokenizer | None = None
) -> nn.Module:
    """The model every `model.composition` slot names. `flat_encoder` masks by the
    field layout of `tokenizer` (the fitted one), else of a tokenizer built from `config`."""
    head_objectives = [
        SupervisedHead(
            head=name,
            weight=config.fraud_task.supervised_head.loss_weight,
            detach_backbone=config.fraud_task.supervised_head.probe_only,
        )
        for name in config.model.composition.heads
        if config.fraud_task.supervised_head.enabled
    ]
    return model_from_config(
        config,
        vocab_size=vocab_size,
        backbone=build_backbone(config, vocab_size),
        head_objectives=head_objectives,
        vocabulary=_field_vocabulary(config, tokenizer),
    )


def _field_vocabulary(
    config: RecipeConfig, tokenizer: SequenceTokenizer | None
) -> FieldVocabulary | None:
    """The field layout `flat_encoder` masks by."""
    if config.model.composition.granularity.type != "flat_encoder":
        return None
    if tokenizer is None:
        tokenizer = tokenizer_for(config)
    if not isinstance(tokenizer, FieldVocabularyTokenizer):
        raise IncompatibleCompositionError(
            f"flat_encoder masks by field, which {tokenizer.name!r} does not declare: it "
            "needs a tokenizer with field_vocabulary() "
            "(tfm.contracts.tokenizers.FieldVocabularyTokenizer)."
        )
    return tokenizer.field_vocabulary()


def _granularity(model: nn.Module) -> Any:
    return model.granularity if isinstance(model, TrxFM) else cast(Any, model).spec


def setup_run(
    config_path: Path | None = None,
    run_dir: Path | None = None,
    *,
    config: RecipeConfig | None = None,
    device: str | None = None,
    recipe: str | None = None,
    pretraining_only: bool = False,
) -> RunArtifacts:
    """Prepare data and build the model. Pass `config` to run an edited config
    (for example one slot swapped) without writing it to disk first.

    `pretraining_only` with a flat-field `data.token_cache` reads the pretraining
    windows from the cache and loads no events, so every rank of a multi-GPU
    run skips the object path. Without such a cache it changes nothing: the
    pretraining corpus is then built from events either way.
    """
    if config is not None and config_path is not None:
        raise ValueError("Pass config_path or config, not both.")
    if config is not None:
        recipe_name = recipe or "custom"
    else:
        config_file = (
            Path(config_path) if config_path else default_config_path(recipe or DEFAULT_RECIPE)
        )
        config = load_recipe_config(config_file)
        recipe_name = recipe or config_file.resolve().parent.name
    config = apply_device_override(config, device)
    seed_everything(config.seed)
    resolved_device = resolve_device(config.runtime.device)
    # Before data preparation, which is the slow part at book scale.
    check_backend_for_run(config, resolved_device)
    out_dir = run_dir or default_run_dir(recipe=recipe_name)
    data: PreparedData | PretrainingData
    if pretraining_only and _cache_trains_without_book(config):
        stage(f"setup: pretraining windows from the token cache {config.data.token_cache}")
        data = prepare_pretraining(config)
    else:
        stage(f"setup: loading and tokenizing the {config.data.adapter} data")
        data = prepare_data(config, fixture_dir=out_dir / "pragmatiq_book")
    stage("setup: building the model")
    model = build_model(config, data.tokenizer.vocab_size, tokenizer=data.tokenizer)
    validate_tokenizer_layout(data.tokenizer, _granularity(model))
    if isinstance(model, EventArrayReader):
        if data.cache_format is not None:
            data.cache_format.check_event_arrays(data.token_cache, model.event_arrays)
        elif isinstance(data, PreparedData):
            data.batches[0].check_event_arrays(model.event_arrays)
    check_backend_for_run(config, resolved_device, layout=data.tokenizer.layout)
    artifacts = RunArtifacts(
        run_dir=out_dir,
        config=config,
        data=data,
        model=model,
        device=resolved_device,
        recipe_name=recipe_name,
    )
    if is_primary_process():
        _write_setup_manifests(artifacts)
    return artifacts


def _cache_trains_without_book(config: RecipeConfig) -> bool:
    return (
        config.data.token_cache is not None
        and cache_format_of(tokenizer_class(config.features.tokenizer)).trains_without_book
    )


def _write_pretraining_manifests(artifacts: RunArtifacts, data: PretrainingData) -> None:
    run_dir, config = artifacts.run_dir, artifacts.config
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(run_dir, config.model_dump_resolved())
    cache = data.token_cache
    write_json(
        run_dir / "data_validation_report.json",
        {
            "schema": "TransactionEvent",
            "adapter": config.data.adapter,
            "split_protocol": config.data.split_protocol,
            "data_path": config.data.path,
            "tokenizer_layout": data.tokenizer.layout,
            "token_cache": str(cache.path),
            "n_cached_users": int(cache.manifest["n_users"]),
            "n_cached_train_events": int(cache.manifest["n_events"]),
            "cache_population": cache.manifest["population"],
            "pretraining_only": True,
        },
    )
    write_json(
        run_dir / "split_manifest.json",
        {
            "observation_window": data.observation_window,
            "train_cutoff": data.train_cutoff.isoformat(),
            "n_pretrain_windows": len(data.pretrain_rows),
            "pretraining_only": True,
        },
    )
    write_json(run_dir / "tokenizer_manifest.json", data.tokenizer.manifest())
    write_environment_manifest(run_dir, config, artifacts.device)


def _write_setup_manifests(artifacts: RunArtifacts) -> None:
    if isinstance(artifacts.data, PretrainingData):
        _write_pretraining_manifests(artifacts, artifacts.data)
        return
    run_dir, config, data = artifacts.run_dir, artifacts.config, artifacts.data
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(run_dir, config.model_dump_resolved())
    write_json(
        run_dir / "data_validation_report.json",
        {
            "schema": "TransactionEvent",
            "adapter": config.data.adapter,
            "split_protocol": config.data.split_protocol,
            "data_path": config.data.path,
            "tokenizer_layout": data.tokenizer.layout,
            "n_events": len(data.events),
            "n_profiles": len(data.profiles),
            "n_train_sequences": artifacts.train_batch.batch_size,
            "n_validation_sequences": artifacts.val_batch.batch_size,
            "n_test_sequences": artifacts.test_batch.batch_size,
            "label_tasks": sorted(data.label_rows),
            "adapter_metadata": data.adapter_metadata,
        },
    )
    spec = data.spec
    write_json(
        run_dir / "split_manifest.json",
        {
            "split_protocol": config.data.split_protocol,
            "label_delay": str(spec.label_delay),
            "observation_window": config.data.observation_window,
            "n_train_sequences": artifacts.train_batch.batch_size,
            "n_validation_sequences": artifacts.val_batch.batch_size,
            "n_test_sequences": artifacts.test_batch.batch_size,
            "n_train_events": len(data.splits.train),
            "n_validation_events": len(data.splits.validation),
            "n_test_events": len(data.splits.test),
            **data.cutoffs.as_dict(),
        },
    )
    if data.fraud is not None:
        write_json(run_dir / "identity_theft_manifest.json", dict(data.fraud.manifest))
    write_json(run_dir / "tokenizer_manifest.json", data.tokenizer.manifest())
    write_environment_manifest(run_dir, config, artifacts.device)


def training_batches(artifacts: RunArtifacts) -> tuple[Sequence[Any], dict[str, Any]]:
    """Padded minibatches, or packed full-context rows when `training.packing`.

    With a token cache, minibatches are collated from disk as the trainer reads them.
    """
    config, data = artifacts.config, artifacts.data
    windows = config.data.pretrain_windows
    row_tokens = windows.max_tokens or config.data.max_tokens if windows else config.data.max_tokens
    if data.cache_format is not None and data.pretrain_rows is not None:
        model = artifacts.model
        event_arrays = model.event_arrays if isinstance(model, EventArrayReader) else ()
        batches, recorded = data.cache_format.batches(
            data.token_cache,
            data.pretrain_rows,
            data.tokenizer,
            config,
            max_length=row_tokens,
            device=artifacts.device,
            event_arrays=event_arrays,
        )
        return batches, {
            "packing": False,
            "shuffle": config.training.shuffle,
            "pretrain_windows": len(data.pretrain_rows),
            "pretrain_row_tokens": row_tokens,
            "pretrain_entities": len(set(data.pretrain_rows.user.tolist())),
            "token_cache": str(data.token_cache.path),
            "event_arrays": list(event_arrays),
            **recorded,
        }
    data = artifacts.prepared
    if not config.training.packing:
        if windows is not None:
            batch = collate_batch(
                data.pretrain_sequences,
                data.tokenizer,
                max_tokens=row_tokens,
                amount_key=config.data.amount_field,
            )
        else:
            batch = artifacts.train_batch
        if config.training.shuffle:
            batch = shuffled_rows(batch, config.seed)
        return (
            minibatches(batch, config.training.batch_size),
            {
                "packing": False,
                "shuffle": config.training.shuffle,
                "pretrain_windows": len(data.pretrain_sequences),
                "pretrain_row_tokens": row_tokens,
            },
        )
    model = artifacts.model
    if isinstance(model, TrxFM) and model.head_objective_specs:
        raise IncompatibleCompositionError(
            "training.packing cannot be combined with fraud_task.supervised_head: a packed "
            "row holds several entities, so its sequence-level label is undefined."
        )
    seq_length = config.model.composition.granularity.decoder.max_seq_len
    packed, stats = pack_sequences(
        data.pretrain_sequences or data.train_sequences, data.tokenizer, max_tokens=seq_length
    )
    if config.training.shuffle:
        packed = shuffled_rows(packed, config.seed)
    return minibatches(packed, config.training.batch_size), {
        "packing": True,
        "shuffle": config.training.shuffle,
        "packing_seq_length": seq_length,
        "pretrain_windows": len(data.pretrain_sequences),
        **stats.as_dict(),
    }


def _composition_summary(artifacts: RunArtifacts) -> dict[str, Any]:
    composition = artifacts.config.model.composition
    return {
        "granularity": composition.granularity.type,
        "tokenizer": artifacts.tokenizer.name,
        "layout": artifacts.tokenizer.layout,
        "position": composition.position,
        "attention": composition.attention,
        "backbone": artifacts.config.model.backbone,
        "objective": composition.objective,
        "auxiliary": list(artifacts.config.objective.auxiliary),
        "readout": composition.readout,
        "vocab_size": artifacts.tokenizer.vocab_size,
    }


def run_train(artifacts: RunArtifacts) -> dict[str, float | str]:
    batches, batch_stats = training_batches(artifacts)
    if is_primary_process():
        write_json(artifacts.run_dir / "packing_manifest.json", batch_stats)
    config = artifacts.config
    trainer = build_trainer(
        config.training.backend, checkpoint_dir=artifacts.run_dir / "hf_checkpoint"
    )
    progress = TrainingProgress(log_every=config.training.log_every)
    stage(
        f"train: {batch_stats.get('pretrain_windows', len(batches)):,} rows, "
        f"batch {config.training.batch_size} per process, backend {config.training.backend}"
    )
    stats = trainer.fit(
        artifacts.model,
        batches,
        max_steps=config.training.max_steps,
        learning_rate=config.training.learning_rate,
        seed=config.seed,
        device=artifacts.device,
        optimizer=config.training.optimizer,
        shuffle=config.training.shuffle,
        precision=config.training.precision,
        progress=progress,
    )
    if is_primary_process():
        write_json(
            artifacts.run_dir / "train_log.json",
            {
                "log_every": config.training.log_every,
                "loss": "the first process's minibatch loss at each step",
                "steps": progress.curve,
            },
        )
    save_run_checkpoint(
        artifacts.model,
        artifacts.run_dir,
        {
            "architecture": config.model.composition.granularity.type,
            "composition": _composition_summary(artifacts),
            "vocab_size": artifacts.tokenizer.vocab_size,
            "layout": artifacts.tokenizer.layout,
            "seed": config.seed,
        },
        stats,
    )
    return stats


def load_trained(artifacts: RunArtifacts) -> None:
    """Load `model.pt` from the run directory when it exists."""
    ckpt = artifacts.run_dir / "model.pt"
    if ckpt.exists():
        load_checkpoint(artifacts.model, ckpt)


def run_infer(artifacts: RunArtifacts) -> dict[str, EmbeddingTable]:
    load_trained(artifacts)
    stage(
        "infer: embedding "
        + ", ".join(
            f"{name} {batch.batch_size:,}"
            for name, batch in (
                ("train", artifacts.train_batch),
                ("validation", artifacts.val_batch),
                ("test", artifacts.test_batch),
            )
        )
        + " histories"
    )
    cfg = artifacts.config.evaluation
    extract = {
        "device": artifacts.device,
        "minibatch_size": cfg.inference_batch_size,
        "precision": cfg.inference_precision,
    }
    tables = {
        "train": extract_embeddings(artifacts.model, artifacts.train_batch, **extract),
        "validation": extract_embeddings(artifacts.model, artifacts.val_batch, **extract),
        "test": extract_embeddings(artifacts.model, artifacts.test_batch, **extract),
    }
    if is_primary_process():
        write_embeddings_manifest(
            artifacts.run_dir,
            tables,
            alignment_fields=["entity_ids", "sequence_ids", "event_ids", "fraud_labels"],
        )
        np.savez(
            artifacts.run_dir / "embeddings.npz", **cast(dict[str, Any], embedding_arrays(tables))
        )
    return tables


EMBEDDING_ID_FIELDS: tuple[str, ...] = ("entity_ids", "sequence_ids", "event_ids")


def embedding_arrays(tables: dict[str, EmbeddingTable]) -> dict[str, np.ndarray]:
    """`embeddings.npz`'s arrays: per side, the vectors with their labels and ids.

    `<side>` holds the vectors, and `<side>_labels`, `<side>_entity_ids`,
    `<side>_sequence_ids`, and `<side>_event_ids` hold the same rows in the same
    order, so a reader joins by id instead of trusting row order. Ids are plain
    strings, readable without `allow_pickle`.
    """
    arrays: dict[str, np.ndarray] = {}
    for side, table in tables.items():
        arrays[side] = table.vectors
        arrays[f"{side}_labels"] = table.fraud_labels
        for name in EMBEDDING_ID_FIELDS:
            arrays[f"{side}_{name}"] = np.asarray(getattr(table, name), dtype=np.str_)
    return arrays


def run_evaluate(
    artifacts: RunArtifacts, tables: dict[str, EmbeddingTable] | None = None
) -> dict[str, dict[str, float]]:
    """Every probe and fine-tune the config enables. `tables` is unused: probes
    re-embed their own entity-disjoint rows."""
    del tables
    load_trained(artifacts)
    stage("evaluate: probes and baselines")
    metrics, probe_manifest, finetune_manifest = evaluate_run(
        artifacts.model, artifacts.prepared, artifacts.config, artifacts.device
    )
    write_json(artifacts.run_dir / "probe_manifest.json", probe_manifest)
    write_json(artifacts.run_dir / "finetune_manifest.json", finetune_manifest)
    write_json(artifacts.run_dir / "metrics.json", metrics)
    (artifacts.run_dir / "metrics.md").write_text(metrics_markdown(metrics), encoding="utf-8")
    return metrics


def _data_description(config: RecipeConfig, data: PreparedData) -> str:
    adapter = config.data.adapter
    if adapter == "synthetic":
        return "synthetic TransactionEvent fixture (no real customer data)"
    if adapter == "tabformer":
        return (
            "IBM TabFormer public synthetic card transactions "
            "(local, entity-subsampled; not fetched by CI)"
        )
    if data.adapter_metadata.get("source") == "fixture":
        return "generated synthetic pragmatiq book (transaction and app sources, profiles)"
    return f"{adapter} book (local, entity-subsampled)"


def run_report(artifacts: RunArtifacts, metrics: dict[str, dict[str, float]] | None = None) -> None:
    report_metrics = resolve_report_metrics(artifacts.run_dir, metrics)
    summary = _composition_summary(artifacts)
    objectives = " + ".join([summary["objective"], *summary["auxiliary"]])
    architecture = (
        f"{summary['granularity']} over {summary['tokenizer']} tokens "
        f"({summary['layout']}), {summary['backbone']} backbone, {summary['position']} "
        f"positions, {objectives}, {summary['readout']} readout"
    )
    experimental = summary["granularity"] != "flat" or summary["backbone"] == "gru"
    limitations = (
        "Reference architecture only; not a production fraud, credit, or risk model. "
        "Synthetic or subsampled labels and short training. "
        + (
            "Experimental stack: import from tfm.experimental, not the core tfm.__all__. "
            "Byte-level text, not BPE. "
            if experimental
            else ""
        )
        + "Probes are entity-disjoint with per-period labels (ADR 0013); "
        "results are not numerically comparable to Blueprint v1's probe."
    )
    card = model_card_markdown(
        recipe=artifacts.recipe_name,
        seed=artifacts.config.seed,
        architecture=architecture,
        data=_data_description(artifacts.config, artifacts.prepared),
        metrics=report_metrics,
        limitations=limitations,
    )
    (artifacts.run_dir / "model_card.md").write_text(card, encoding="utf-8")


def run_smoke(
    run_dir: Path | None = None,
    config_path: Path | None = None,
    *,
    device: str | None = None,
    recipe: str | None = None,
) -> Path:
    artifacts = setup_run(config_path=config_path, run_dir=run_dir, device=device, recipe=recipe)
    run_train(artifacts)
    tables = run_infer(artifacts)
    metrics = run_evaluate(artifacts, tables)
    run_report(artifacts, metrics)
    return artifacts.run_dir
