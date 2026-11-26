# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph recipe orchestration: existing trainers, shared probe metrics, explicit stages."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from tfm.api.builders import build_position, model_from_config
from tfm.api.graph import (
    GraphBuildContext,
    build_graph_architecture,
    context_history,
    graph_architecture,
    graph_specs,
    task_head,
)
from tfm.attention.registry import build_attention
from tfm.config import RecipeConfig
from tfm.contracts.graph import GraphHistoryBatch, SequenceSupervisedBatch
from tfm.contracts.tokenizers import StatefulSequenceTokenizer
from tfm.data.graph import FutureTargetTransform
from tfm.data.graph_recipe import prepare_graph_data
from tfm.evaluation.fraud.baselines import evaluate_binary_probes, fit_predict_probe
from tfm.evaluation.fraud.metrics import classification_metrics
from tfm.evaluation.reporting.report import metrics_markdown, write_json
from tfm.extras.kumo import engine_version
from tfm.features.tokenizers.registry import tokenizer_class
from tfm.inference.embedding import EmbeddingTable
from tfm.layers.gnn import GNNEncoderModel, GNNSequenceModel, SequenceBinaryModel
from tfm.layers.profile_event_decoder import ProfileEventDecoderModel
from tfm.pooling.registry import build_readout
from tfm.position.none import NoPE
from tfm.position.time_rope import TimeRoPE
from tfm.runtime.common import (
    default_run_dir,
    source_provenance,
    write_environment_manifest,
    write_resolved_config,
)
from tfm.training.backends import backend_capabilities, check_backend_supports
from tfm.training.checkpointing import (
    load_encoder_state,
    read_checkpoint,
    save_checkpoint,
    save_encoder_state,
)
from tfm.training.device import resolve_device
from tfm.training.reproducibility import seed_everything
from tfm.training.selection import fit_selected


@torch.no_grad()
def _table(
    model: nn.Module, batches: list[GraphHistoryBatch], device: torch.device, *, view: str = "main"
) -> EmbeddingTable:
    model.eval()
    vectors, numeric, labels, entities, samples = [], [], [], [], []
    for original in batches:
        if original.graph.sharing != "disjoint":
            raise ValueError("Embedding evaluation requires disjoint per-sample graphs")
        original.assert_validated()
        if view == "sequence":
            history_view = SequenceSupervisedBatch.from_graph_history(original).for_execution(
                device
            )
            history_inputs = (
                history_view if isinstance(model, SequenceBinaryModel) else history_view.history
            )
            embedding = model(history_inputs, compute_loss=False).embeddings
        elif view == "graph":
            graph = original.graph.for_execution(device)
            if isinstance(model, GNNSequenceModel):
                embedding = model.graph_encoder(graph)
            elif isinstance(model, GNNEncoderModel):
                embedding = model.encoder(graph)
            else:
                raise ValueError("Graph view requires a graph encoder model")
        else:
            inputs = (
                original.graph.for_execution(device)
                if view == "graph_model"
                else original.for_execution(device)
            )
            embedding = model(inputs, compute_loss=False).embeddings
        batch = original  # Evaluation IDs, labels and raw features remain on CPU.
        if embedding is None or batch.labels is None or batch.history.numeric_features is None:
            raise ValueError("Probe embedding, labels, or raw features missing")
        vectors.append(embedding.cpu().float().numpy())
        numeric.append(batch.history.numeric_features.cpu().float().numpy())
        labels.append(batch.labels.cpu().float().numpy())
        entities.extend(a.entity_id for a in batch.history_anchors)
        samples.extend(a.sample_id for a in batch.history_anchors)
    return EmbeddingTable(
        np.concatenate(vectors),
        tuple(entities),
        tuple(samples),
        tuple("" for _ in entities),
        np.concatenate(labels),
        np.concatenate(numeric),
    )


@torch.no_grad()
def _probabilities(
    model: nn.Module,
    batches: list[GraphHistoryBatch],
    device: torch.device,
    *,
    graph_only: bool = False,
) -> np.ndarray:
    model.eval()
    values = []
    for original in batches:
        if original.graph.sharing != "disjoint":
            raise ValueError("Task evaluation requires disjoint per-sample graphs")
        if isinstance(model, SequenceBinaryModel):
            inputs = SequenceSupervisedBatch.from_graph_history(original).for_execution(device)
        else:
            original.assert_validated()
            inputs = (
                original.graph.for_execution(device)
                if graph_only
                else original.for_execution(device)
            )
        output = model(inputs, compute_loss=False)
        if output.predictions is None or "binary" not in output.predictions:
            raise ValueError("No supervised task head to evaluate")
        values.append(output.predictions["binary"].cpu().numpy())
    return np.concatenate(values)


def _graph_tabular(batches: list[GraphHistoryBatch]) -> np.ndarray:
    rows = []
    for batch in batches:
        for i in range(batch.graph.batch_size):
            row = []
            for name in batch.graph.node_features:
                x = batch.graph.node_features[name][batch.graph.node_sample[name] == i]
                row.extend([float(len(x)), *x.sum(0).tolist()])
            rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _module_types(model: nn.Module) -> dict[str, str]:
    return {
        name: f"{type(module).__module__}:{type(module).__qualname__}"
        for name, module in model.named_modules()
    }


def _transform_state(transform: FutureTargetTransform) -> dict[str, Any]:
    return {
        "input_columns": transform.input_columns,
        "selected": transform.selected,
        "mean": transform.mean.tolist(),
        "scale": transform.scale.tolist(),
    }


def _config_identity(config: RecipeConfig) -> str:
    return hashlib.sha256(
        json.dumps(
            config.model_dump(mode="json", exclude={"runtime": {"device"}}), sort_keys=True
        ).encode()
    ).hexdigest()


def _check_metadata(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        # Sidecar JSON normalizes tuples to lists; compare serialized values.
        if json.dumps(metadata.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
            label = {"config_sha256": "config identity", "data_sha256": "dataset identity"}.get(
                key, key
            )
            raise ValueError(f"Checkpoint {label} differs from the requested experiment")


def run_command(
    command: str,
    *,
    config_path: Path | None,
    config: RecipeConfig | None = None,
    run_dir: Path | None,
    device: str | None,
    reuse_checkpoint: bool = False,
) -> Path:
    """Run train/infer/evaluate/report/smoke for the experimental graph recipe family.

    Reuse checks config, dataset, target schema, and fitted preprocessing before
    loading weights. Checkpoints are trusted local artifacts, not remote downloads.
    """
    if config is None:
        if config_path is None:
            raise ValueError("Pass a parsed recipe config or its path")
        config = RecipeConfig.model_validate(yaml.safe_load(config_path.read_text()))
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
    if config.data.loader != "python":
        from tfm.runtime.native import run_command as run_native

        return run_native(
            command,
            config=config,
            config_path=config_path,
            run_dir=run_dir,
            device=device,
            reuse_checkpoint=reuse_checkpoint,
        )
    if device is not None:
        config = config.model_copy(
            update={"runtime": config.runtime.model_copy(update={"device": device})}
        )
    if config.runtime.stage == "pretrain" and command != "train":
        raise ValueError(
            "pretrain-only runs use train; scoring requires a downstream/full stage with labels"
        )
    name = config.model.composition.granularity.type
    layout = graph_architecture(name).layout
    out = run_dir or default_run_dir(recipe=name)
    out.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device or config.runtime.device)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Graph recipes currently support single-process eager training only")
    for settings, backend_layout in (
        (config.training, "graph"),
        (config.sequence_training, "graph_history"),
    ):
        check_backend_supports(
            backend_capabilities(settings.backend),
            backbone="eager",
            packing=settings.packing,
            device_type=resolved_device.type,
            world_size=1,
            layout=backend_layout,
            precision=settings.precision,
        )
    seed_everything(config.seed)
    identity = _config_identity(config)
    codec_identity = None
    if config.runtime.stage != "pretrain":
        codec_class = tokenizer_class(config.features.tokenizer)
        codec_identity = f"{codec_class.__module__}:{codec_class.__qualname__}"
    reused = reuse_checkpoint and command != "train" and (out / "model.pt").is_file()
    tokenizer, transform = None, None
    payload = read_checkpoint(out / "model.pt") if reused else None
    if payload is not None:
        metadata = payload["metadata"]
        if metadata.get("format") != "tfm.graph-experiment.v1":
            raise ValueError("Unsupported checkpoint format; expected tfm.graph-experiment.v1")
        _check_metadata(
            metadata,
            {
                "architecture": name,
                "amount_field": config.data.amount_field,
                "sequence_position": config.model.composition.position,
                "config_sha256": identity,
                "tokenizer_factory": codec_identity,
                "kumo_engine_version": engine_version(),
            },
        )
        required = {
            "tokenizer",
            "target_transform",
            "stages",
            "split",
            "source_revision",
            "source_dirty",
        }
        if not required.issubset(metadata):
            raise ValueError("Checkpoint metadata lacks required experiment provenance")
        tokenizer = tokenizer_class(config.features.tokenizer).from_state(metadata["tokenizer"])
        if not isinstance(tokenizer, StatefulSequenceTokenizer):
            raise ValueError("Checkpoint tokenizer must implement persistent sequence encoding")
        state = metadata["target_transform"]
        if (
            config.training.objective == "future_subgraph"
            and config.runtime.stage != "downstream"
            and state is None
        ):
            raise ValueError("Future checkpoints require fitted target statistics")
        transform = (
            None
            if state is None
            else FutureTargetTransform(
                tuple(state["input_columns"]),
                tuple(state["selected"]),
                torch.tensor(state["mean"]),
                torch.tensor(state["scale"]),
            )
        )
    data = prepare_graph_data(config, tokenizer=tokenizer, transform=transform)
    if reused:
        _check_metadata(metadata, {"data_sha256": data.fingerprint, "split": data.split_manifest})
    graph_spec, sequence_spec = graph_specs(config, data)
    future = config.training.objective == "future_subgraph" and config.runtime.stage != "downstream"
    columns = (
        ()
        if data.transform is None
        else tuple(data.transform.input_columns[i] for i in data.transform.selected)
    )
    context = GraphBuildContext(graph_spec, sequence_spec, columns)
    encoder_kind = name if layout == "graph" else config.model.composition.granularity.encoder_type
    graph_model = build_graph_architecture(encoder_kind, config, context)
    graph_encoder = graph_model.encoder
    if not isinstance(graph_encoder, nn.Module):
        raise ValueError("Graph architecture must expose its head-free encoder module")
    encoder_schema = {
        "graph_schema": asdict(graph_spec),
        "feature_columns": {
            table.name: list(table.features) for table in data.sampler.graph.tables
        },
        "feature_transform": "transaction-signed-log-v1",
        "amount_field": config.data.amount_field,
        "encoder_module_types": _module_types(graph_encoder),
        "kumo_engine_version": engine_version(),
    }
    encoder_source = None
    if config.runtime.stage == "downstream" and not reused:
        assert config.runtime.encoder_checkpoint is not None
        encoder_source = load_encoder_state(
            graph_encoder, config.runtime.encoder_checkpoint, expected=encoder_schema
        )
    attention = build_attention(config.model.composition.attention)
    model: nn.Module = graph_model
    stats: dict[str, Any] = {}
    if not reused and (config.runtime.stage != "downstream" or layout == "graph"):
        stats["graph"] = fit_selected(
            graph_model,
            data.pretrain,
            data.validation,
            config.training,
            seed=config.seed,
            device=resolved_device,
        )
    if config.runtime.stage == "pretrain":
        metadata = {
            "architecture": name,
            "format": "tfm.graph-experiment.v1",
            "config_sha256": identity,
            "data_sha256": data.fingerprint,
            "execution_stage": "pretrain",
            "graph_objective": config.training.objective,
            "stages": stats,
            "split": data.split_manifest,
            "target_transform": None
            if data.transform is None
            else _transform_state(data.transform),
            **encoder_schema,
            **source_provenance(),
        }
        save_checkpoint(graph_model, out / "model.pt", metadata=metadata)
        save_encoder_state(graph_encoder, out / "encoder.pt", metadata=metadata)
        write_resolved_config(out, config.model_dump(mode="json"))
        write_json(out / "model.manifest.json", metadata)
        write_json(out / "split_manifest.json", data.split_manifest)
        write_json(out / "train_log.json", stats)
        write_environment_manifest(out, config, resolved_device)
        return out
    assert sequence_spec is not None and data.tokenizer is not None
    position = build_position(config.model.composition.position)
    if not isinstance(position, (NoPE, TimeRoPE)):
        raise ValueError("Graph history supports NoPE or TimeRoPE")
    seed_everything(config.seed)
    if layout == "graph_history":
        model = model_from_config(
            config,
            vocab_size=sequence_spec.vocab_size,
            graph_context=GraphBuildContext(graph_spec, sequence_spec, columns, graph_encoder),
        )
        if not reused:
            stats["sequence"] = fit_selected(
                model,
                data.hybrid_train,
                data.probes["validation"],
                config.sequence_training,
                seed=config.seed,
                device=resolved_device,
            )
    # Matched shared-stack initialization and supervised rows, regardless of GNN objective.
    seed_everything(config.seed)
    gran = config.model.composition.granularity
    supervised_baseline = SequenceBinaryModel(
        sequence_spec,
        attention,
        readout=build_readout(config.model.composition.readout),
        history_model=context_history(
            gran.history_model, sequence_spec, attention, position=position
        ),
        position=position,
        head=task_head(gran.task_head, sequence_spec.hidden_size),
    )
    seed_everything(config.seed)
    baseline = (
        ProfileEventDecoderModel(
            sequence_spec,
            attention=attention,
            readout=build_readout(config.model.composition.readout),
            position=position,
        )
        if future
        else supervised_baseline
    )
    modules: dict[str, nn.Module] = {"model": model, "sequence_baseline": baseline}
    if future:
        modules["sequence_supervised"] = supervised_baseline
    bundle = nn.ModuleDict(modules)
    sequence_schema = {
        **asdict(sequence_spec),
        "accepted_layouts": sorted(sequence_spec.accepted_layouts),
    }
    if payload is not None:
        _check_metadata(
            metadata,
            {
                "graph_schema": asdict(graph_spec),
                "sequence_schema": sequence_schema,
                "graph_objective": "future_subgraph" if future else "binary",
                "sequence_attention": config.model.composition.attention,
                "hop_trimming": graph_spec.trim,
                "module_types": _module_types(bundle),
            },
        )
        bundle.load_state_dict(payload["state_dict"])
    else:
        stats["sequence_baseline"] = fit_selected(
            baseline,
            data.sequence_pretrain,
            data.sequence_validation,
            config.sequence_training,
            seed=config.seed,
            device=resolved_device,
        )
        if future:
            stats["sequence_supervised"] = fit_selected(
                supervised_baseline,
                data.sequence_supervised_train,
                data.sequence_supervised_validation,
                config.sequence_training,
                seed=config.seed,
                device=resolved_device,
            )
        metadata = {
            "architecture": name,
            "module_types": _module_types(bundle),
            "support_tier": "experimental",
            "format": "tfm.graph-experiment.v1",
            "kumo_engine_version": engine_version(),
            "config_sha256": identity,
            "data_sha256": data.fingerprint,
            **source_provenance(),
            "graph_schema": asdict(graph_spec),
            "sequence_schema": sequence_schema,
            "target_transform": None
            if data.transform is None
            else _transform_state(data.transform),
            "graph_objective": "future_subgraph" if future else "binary",
            "execution_stage": config.runtime.stage,
            "encoder_source": encoder_source,
            "sequence_attention": config.model.composition.attention,
            "sequence_position": config.model.composition.position,
            "amount_field": config.data.amount_field,
            "hop_trimming": graph_spec.trim,
            "tokenizer": data.tokenizer.state_dict(),
            "tokenizer_factory": codec_identity,
            "split": data.split_manifest,
            "stages": stats,
        }
        save_checkpoint(bundle, out / "model.pt", metadata=metadata)
        save_encoder_state(
            graph_encoder, out / "encoder.pt", metadata={**metadata, **encoder_schema}
        )
    bundle.to(resolved_device)
    write_resolved_config(out, config.model_dump(mode="json"))
    write_json(
        out / "data_validation_report.json",
        {
            "valid": True,
            "data_sha256": data.fingerprint,
            "source_observed_through": config.data.observed_through,
        },
    )
    write_json(out / "split_manifest.json", data.split_manifest)
    write_json(out / "tokenizer_manifest.json", data.tokenizer.manifest())
    write_json(out / "model.manifest.json", metadata)
    write_json(out / "checkpoint_manifest.json", metadata)
    write_environment_manifest(out, config, resolved_device)
    write_json(out / "train_log.json", metadata["stages"])
    if command == "train":
        return out
    tables = {
        side: _table(
            model, batches, resolved_device, view="graph_model" if layout == "graph" else "main"
        )
        for side, batches in data.probes.items()
    }
    arrays: dict[str, Any] = {}
    for side, table in tables.items():
        arrays[side] = table.vectors
        arrays[f"{side}_sample_ids"] = np.asarray(table.sequence_ids)
        arrays[f"{side}_entity_ids"] = np.asarray(table.entity_ids)
        arrays[f"{side}_labels"] = table.fraud_labels
    np.savez(out / "embeddings.npz", **arrays)
    write_json(
        out / "embeddings_manifest.json",
        {
            side: {
                "shape": list(t.vectors.shape),
                "sample_ids": t.sequence_ids,
                "entity_ids": t.entity_ids,
            }
            for side, t in tables.items()
        },
    )
    if command == "infer":
        return out
    train, test = tables["train"], tables["test"]
    metrics = evaluate_binary_probes(train, test, seed=config.seed)
    if config.model.composition.objective == "binary":
        metrics["task_head"] = classification_metrics(
            test.fraud_labels,
            _probabilities(
                model, data.probes["test"], resolved_device, graph_only=layout == "graph"
            ),
        )
    baseline_tables = {
        side: _table(baseline, batches, resolved_device, view="sequence")
        for side, batches in data.probes.items()
    }
    metrics.update(
        evaluate_binary_probes(
            baseline_tables["train"],
            baseline_tables["test"],
            seed=config.seed,
            prefix="sequence_only",
            arms=("embedding", "combined"),
        )
    )
    supervised_tables = (
        {
            side: _table(supervised_baseline, batches, resolved_device, view="sequence")
            for side, batches in data.probes.items()
        }
        if future
        else baseline_tables
    )
    metrics.update(
        evaluate_binary_probes(
            supervised_tables["train"],
            supervised_tables["test"],
            seed=config.seed,
            prefix="sequence_supervised",
            arms=("embedding", "combined"),
        )
    )
    metrics["sequence_supervised_task_head"] = classification_metrics(
        test.fraud_labels, _probabilities(supervised_baseline, data.probes["test"], resolved_device)
    )
    if isinstance(model, GNNSequenceModel):
        graph_tables = {
            side: _table(model, batches, resolved_device, view="graph")
            for side, batches in data.probes.items()
        }
        metrics.update(
            evaluate_binary_probes(
                graph_tables["train"],
                graph_tables["test"],
                seed=config.seed,
                prefix="graph_only",
                arms=("embedding", "combined"),
            )
        )
    scores = fit_predict_probe(
        _graph_tabular(data.probes["train"]),
        train.fraud_labels,
        _graph_tabular(data.probes["test"]),
        seed=config.seed,
    )
    metrics["graph_tabular"] = classification_metrics(test.fraud_labels, scores)
    write_json(
        out / "probe_manifest.json",
        {
            "task": data.split_manifest["probe_task"],
            "sample_ids": {s: t.sequence_ids for s, t in tables.items()},
            "head": "logistic_regression",
            "graph_policy": data.split_manifest["graph_policy"],
            "baseline": "historical next-event sequence arm in FutureSubgraph mode; matched binary sequence arm in all modes; graph-tabular counts and summed input features",
        },
    )
    write_json(out / "metrics.json", metrics)
    (out / "metrics.md").write_text(metrics_markdown(metrics))
    if command in {"report", "smoke"}:
        (out / "model_card.md").write_text(
            f"# {name}\n\nExperimental reference architecture.\n\n"
            + metrics_markdown(metrics)
            + "\nGenerated or sampled data; CPU smoke validates wiring only. No production or quality claim. Root-owner splits allow cutoff-bounded cross-owner graph context. The task is any available fraud in the truncated history; this is not a future-fraud forecast.\n"
        )
    return out
