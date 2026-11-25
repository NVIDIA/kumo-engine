# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe lifecycle for one bounded relational declaration in all three modes."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import nn

from tfm.api.builders import build_position, profile_event_spec_from_config
from tfm.attention.registry import build_attention
from tfm.compositions.gnn import GNNEncoder, GNNSequence
from tfm.config import RecipeConfig
from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.graph import (
    GraphBatch,
    GraphHistoryBatch,
    GraphSupervisedBatch,
    GraphTrainingBatch,
    SequencePredictionBatch,
    SequenceSupervisedBatch,
    utc_us,
)
from tfm.contracts.tokenizers import StatefulSequenceTokenizer, UsesProfiles
from tfm.data.adapters.registry import load_adapter_events
from tfm.data.graph import FutureTargetTransform, future_subgraph_targets
from tfm.data.relational import RelationalSource, timestamp
from tfm.data.relational_batches import RelationalBatch, RelationalBatches, anchor_chunks
from tfm.data.splits.temporal import parse_duration
from tfm.evaluation.fraud.baselines import fit_predict_probe
from tfm.evaluation.fraud.metrics import classification_metrics
from tfm.evaluation.reporting.report import metrics_markdown, write_json
from tfm.extras.kumo import engine_version
from tfm.extras.kumo_pytorch import SOURCE_REVISION
from tfm.features.tokenizers.registry import tokenizer_for
from tfm.heads.binary import Binary
from tfm.layers.gnn import GNNEncoderModel, GNNSequenceModel, SequenceBinaryModel
from tfm.pooling.registry import build_readout
from tfm.position.none import NoPE
from tfm.position.time_rope import TimeRoPE
from tfm.runtime.common import (
    default_run_dir,
    source_provenance,
    write_environment_manifest,
    write_resolved_config,
)
from tfm.training.checkpointing import (
    load_encoder_state,
    read_checkpoint,
    save_checkpoint,
    save_encoder_state,
)
from tfm.training.device import resolve_device
from tfm.training.reproducibility import seed_everything
from tfm.training.selection import fit_selected
from tfm.training.trainer import TrainableBatch, move_batch_for_execution

Mode = Literal["sequence", "graph", "hybrid"]


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, frozenset):
        return sorted(
            (_plain(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True)
        )
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, date):
        return value.isoformat()
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_plain(value), sort_keys=True, allow_nan=True).encode()
    ).hexdigest()


def _mode(config: RecipeConfig) -> Mode:
    kind = config.model.composition.granularity.type
    if kind == "gnn_encoder":
        return "graph"
    if kind == "gnn_sequence":
        return "hybrid"
    if kind == "profile_event_decoder":
        return "sequence"
    raise UnsupportedCombinationError(f"Relational runner does not support architecture {kind!r}")


def _supervised(values: Sequence[RelationalBatch]) -> list[TrainableBatch]:
    if not all(
        isinstance(value, (GraphSupervisedBatch, GraphHistoryBatch, SequenceSupervisedBatch))
        for value in values
    ):
        raise SchemaError("Training/scoring requires available entity-level targets")
    return cast(list[TrainableBatch], list(values))


def _history_controls(values: Sequence[RelationalBatch]) -> list[SequenceSupervisedBatch]:
    result = []
    for value in values:
        if not isinstance(value, GraphHistoryBatch):
            raise ValueError("Matched sequence controls require aligned hybrid batches")
        result.append(SequenceSupervisedBatch.from_graph_history(value))
    return result


@torch.no_grad()
def _predictions(
    model: nn.Module, batches: Sequence[RelationalBatch], device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    vectors, probabilities, targets, raw = [], [], [], []
    for original in batches:
        original.assert_validated()
        output = model(move_batch_for_execution(original, device), compute_loss=False)
        vectors.append(output.embeddings.detach().float().cpu().numpy())
        probabilities.append(output.predictions["binary"].detach().float().cpu().numpy())
        labels = getattr(original, "labels", None)
        targets.extend([math.nan] * len(output.embeddings) if labels is None else labels.tolist())
        if isinstance(
            original, (GraphHistoryBatch, SequenceSupervisedBatch, SequencePredictionBatch)
        ):
            assert original.history.numeric_features is not None
            raw.extend(original.history.numeric_features.tolist())
        else:
            graph = original.graph if isinstance(original, GraphSupervisedBatch) else original
            assert isinstance(graph, GraphBatch)
            for i in range(graph.batch_size):
                row = []
                for name, values in graph.node_features.items():
                    values = values[graph.node_sample[name] == i]
                    row.extend([float(len(values)), *values.sum(0).tolist()])
                raw.append(row)
    return (
        np.concatenate(vectors),
        np.concatenate(probabilities),
        np.asarray(targets),
        np.asarray(raw),
    )


def run_command(
    command: str,
    *,
    config: RecipeConfig,
    config_path: Path | None,
    run_dir: Path | None,
    device: str | None,
    reuse_checkpoint: bool = False,
) -> Path:
    """Train, evaluate or infer with generated/replayed aligned relational anchors."""
    del config_path
    if device is not None:
        config = config.model_copy(
            update={"runtime": config.runtime.model_copy(update={"device": device})}
        )
    task = config.data.relational
    if task is None:
        raise ValueError("Relational runner requires data.relational")
    mode = _mode(config)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise UnsupportedCombinationError(
            "Relational reference recipes require single-process eager training"
        )
    if config.runtime.stage == "pretrain" and command != "train":
        raise ValueError("Pretrain-only runs use train; scoring requires downstream supervision")
    resolved_device = resolve_device(device or config.runtime.device)
    seed_everything(config.seed)
    loaded = load_adapter_events(config, flat_fields=False)
    if loaded.relational is None:
        raise SchemaError(
            f"Adapter {config.data.adapter!r} did not supply a relational declaration"
        )
    source = RelationalSource(loaded.relational, task)
    gran = config.model.composition.granularity
    out = run_dir or default_run_dir(recipe=f"relational_{mode}")
    out.mkdir(parents=True, exist_ok=True)
    with RelationalBatches(
        source,
        loader=config.data.loader,
        depth=gran.depth,
        fanout=gran.fanout,
        max_events=config.data.sequence_length,
        observation_window=parse_duration(config.data.observation_window),
    ) as inputs:
        graph_spec = None
        if mode != "sequence":
            sampler = inputs.graph_sampler()  # capability errors precede fitting/training
            graph_spec = GNNEncoder(
                tuple((table.name, len(table.features)) for table in sampler.graph.tables),
                sampler.graph.directed_relations,
                task.root,
                foreign_key_relations=tuple(
                    (target, f"rev_{role}", parent)
                    for parent, role, target in sampler.graph.relations
                ),
                hidden_size=gran.hidden_size,
                depth=gran.depth,
                trim=gran.trim,
            )
        needs_history = mode != "graph" and config.runtime.stage != "pretrain"
        tokenizer = None
        if needs_history:
            candidate = tokenizer_for(config)
            if (
                not isinstance(candidate, StatefulSequenceTokenizer)
                or candidate.layout != "key_value_time"
            ):
                raise UnsupportedCombinationError(
                    "Relational histories require a stateful key_value_time tokenizer"
                )
            tokenizer = candidate
            train_histories = inputs.histories("train")
            if not train_histories or not any(history.events for history in train_histories):
                raise SchemaError("Tokenizer fitting requires observable training history")
            field = config.data.amount_field
            if not any(
                math.isfinite(event.numeric.get(field, math.nan))
                for history in train_histories
                for event in history.events
            ):
                raise SchemaError(
                    f"No eligible training-history event has finite amount_field {field!r}"
                )
            profiles = source.profiles(
                [item.anchor for item in inputs.anchors if item.split == "train"]
            )
            if isinstance(tokenizer, UsesProfiles):
                tokenizer.set_profiles(profiles)
            elif any(profile.attributes for profile in profiles.values()):
                raise UnsupportedCombinationError(
                    f"Tokenizer {tokenizer.name!r} cannot consume root profile allowlists"
                )
            tokenizer.fit(train_histories)
        schema = {
            "tables": _plain(source.dataset.tables),
            "relationships": _plain(source.dataset.relationships),
            "root": task.root,
            "paths": task.history_paths,
            "ownership": task.ownership,
            "target_level": "entity",
            "graph_features": "numeric_value_and_presence_v1",
            "feature_order": {
                table.name: {
                    kind: list(getattr(table, kind)) for kind in ("numeric", "categorical", "text")
                }
                for table in source.dataset.tables
            },
        }
        identity_config = config.model_dump_resolved()
        identity_config["runtime"].pop("device", None)
        expected = {
            "format": "tfm.relational-experiment.v1",
            "config_sha256": _digest(identity_config),
            "data_sha256": _digest(source.dataset),
            "anchors_sha256": _digest(inputs.anchors),
            "input_loader": (
                inputs.sampler.loader_manifest
                if inputs.sampler is not None
                else {
                    "name": config.data.loader,
                    "revision": SOURCE_REVISION if config.data.loader == "kumo_pytorch" else None,
                }
            ),
            "kumo_engine_version": None if mode == "sequence" else engine_version(),
            "schema": _plain(schema),
            "tokenizer": None if tokenizer is None else tokenizer.state_dict(),
        }
        reused = reuse_checkpoint and command != "train" and (out / "model.pt").is_file()
        payload = read_checkpoint(out / "model.pt") if reused else None
        if payload is not None:
            for key, value in expected.items():
                if _digest(payload["metadata"].get(key)) != _digest(value):
                    raise ValueError(
                        f"Relational checkpoint {key} differs from the requested experiment"
                    )
        attention = build_attention(config.model.composition.attention)
        position = build_position(config.model.composition.position)
        if not isinstance(position, (NoPE, TimeRoPE)):
            raise UnsupportedCombinationError(
                "Relational history position must be nope or time_rope"
            )
        sequence_spec = (
            None
            if tokenizer is None
            else profile_event_spec_from_config(config, vocab_size=tokenizer.vocab_size)
        )
        models = nn.ModuleDict()
        stages: dict[str, Any] = {}
        future = (
            mode != "sequence"
            and config.training.objective == "future_subgraph"
            and config.runtime.stage != "downstream"
        )
        stage_graph = None
        sequence_control = None
        target_transform = None
        encoder_source = None
        if graph_spec is not None:
            if future:
                assert inputs.sampler is not None
                target_batches = {}
                boundary = utc_us(timestamp(config.data.pretrain_end, "data.pretrain_end"))
                transform = None
                for split in ("train", "validation"):
                    selected = tuple(item for item in inputs.anchors if item.split == split)
                    anchors = tuple(item.anchor for item in selected)
                    raw = future_subgraph_targets(
                        inputs.sampler,
                        anchors,
                        task.root,
                        horizon_us=config.data.horizon_days * 86_400_000_000,
                        depth=gran.target_depth,
                        partition_end_us=boundary,
                        label_delay_us=config.data.label_delay_days * 86_400_000_000,
                    )
                    if transform is None:
                        transform = FutureTargetTransform.fit(raw)
                    normalized = transform.apply(raw)
                    batches = []
                    start = 0
                    for chunk in anchor_chunks(
                        selected,
                        config.training.batch_size,
                        split == "train" and config.data.training_graphs == "shared",
                    ):
                        stop = start + len(chunk)
                        target = replace(
                            normalized,
                            anchors=normalized.anchors[start:stop],
                            values=normalized.values[start:stop],
                            valid=normalized.valid[start:stop],
                            available_us=normalized.available_us[start:stop],
                        )
                        graph = inputs.sampler.sample(
                            target.anchors,
                            task.root,
                            sharing=config.data.training_graphs if split == "train" else "disjoint",
                            hop_trim=gran.trim,
                        )
                        batches.append(GraphTrainingBatch(graph, target))
                        start = stop
                    target_batches[split] = batches
                assert transform is not None
                target_transform = _plain(transform)
                stage_graph = GNNEncoderModel(graph_spec, normalized.columns).to(resolved_device)
                if payload is None:
                    stages["future_subgraph"] = fit_selected(
                        stage_graph,
                        target_batches["train"],
                        target_batches["validation"],
                        config.training,
                        seed=config.seed,
                        device=resolved_device,
                    )
            else:
                stage_graph = GNNEncoderModel(
                    graph_spec, objective=Binary(target_level="entity")
                ).to(resolved_device)
                if config.runtime.stage == "downstream":
                    assert config.runtime.encoder_checkpoint is not None
                    encoder_source = load_encoder_state(
                        stage_graph.encoder,
                        config.runtime.encoder_checkpoint,
                        expected={"schema": schema},
                    )
                if payload is None and (config.runtime.stage != "downstream" or mode == "graph"):
                    graph_train = inputs.batches(
                        "train",
                        mode="graph",
                        batch_size=config.training.batch_size,
                        sharing=config.data.training_graphs,
                        hop_trim=gran.trim,
                    )
                    graph_val = inputs.batches(
                        "validation",
                        mode="graph",
                        batch_size=config.training.batch_size,
                        hop_trim=gran.trim,
                    )
                    stages["graph_binary"] = fit_selected(
                        stage_graph,
                        _supervised(graph_train),
                        _supervised(graph_val),
                        config.training,
                        seed=config.seed,
                        device=resolved_device,
                    )
            models["graph_stage"] = stage_graph
        if config.runtime.stage == "pretrain":
            assert stage_graph is not None
            model: nn.Module = stage_graph
        elif mode == "graph":
            assert stage_graph is not None and graph_spec is not None
            model = stage_graph
            if future:
                supervised = GNNEncoderModel(
                    graph_spec, objective=Binary(target_level="entity")
                ).to(resolved_device)
                supervised.encoder.load_state_dict(stage_graph.encoder.state_dict())
                if payload is None:
                    stages["graph_downstream"] = fit_selected(
                        supervised,
                        _supervised(
                            inputs.batches(
                                "train",
                                mode="graph",
                                batch_size=config.training.batch_size,
                                sharing=config.data.training_graphs,
                                hop_trim=gran.trim,
                            )
                        ),
                        _supervised(
                            inputs.batches(
                                "validation",
                                mode="graph",
                                batch_size=config.training.batch_size,
                                hop_trim=gran.trim,
                            )
                        ),
                        config.training,
                        seed=config.seed,
                        device=resolved_device,
                    )
                model = supervised
        else:
            assert sequence_spec is not None
            kwargs = {
                "attention": attention,
                "position": position,
                "readout": build_readout(config.model.composition.readout),
                "target_level": "entity",
            }
            if mode == "sequence":
                model = SequenceBinaryModel(sequence_spec, **kwargs).to(resolved_device)
            else:
                assert stage_graph is not None and graph_spec is not None
                model = GNNSequenceModel(
                    GNNSequence(graph_spec, sequence_spec), stage_graph.encoder, **kwargs
                ).to(resolved_device)
            if mode == "hybrid":
                sequence_control = SequenceBinaryModel(sequence_spec, **kwargs).to(resolved_device)
                initial = model.state_dict()
                sequence_control.load_state_dict(
                    {name: initial[name] for name in sequence_control.state_dict()}
                )
                models["sequence_control"] = sequence_control
            if payload is None:
                settings = config.training if mode == "sequence" else config.sequence_training
                common = {
                    "mode": mode,
                    "batch_size": settings.batch_size,
                    "tokenizer": tokenizer,
                    "max_tokens": config.data.max_tokens,
                    "amount_key": config.data.amount_field,
                    "hop_trim": gran.trim,
                }
                train_batches = inputs.batches(
                    "train", sharing=config.data.training_graphs, **common
                )
                validation_batches = inputs.batches("validation", **common)
                stages[mode] = fit_selected(
                    model,
                    _supervised(train_batches),
                    _supervised(validation_batches),
                    settings,
                    seed=config.seed,
                    device=resolved_device,
                )
                if sequence_control is not None:
                    stages["sequence_control"] = fit_selected(
                        sequence_control,
                        _history_controls(train_batches),
                        _history_controls(validation_batches),
                        settings,
                        seed=config.seed,
                        device=resolved_device,
                    )
        models["model"] = model
        expected["target_transform"] = target_transform
        expected["encoder_source"] = encoder_source
        if payload is not None and _digest(payload["metadata"].get("encoder_source")) != _digest(
            encoder_source
        ):
            raise ValueError(
                "Relational checkpoint encoder_source differs from the requested encoder"
            )
        if payload is not None and _digest(payload["metadata"].get("target_transform")) != _digest(
            target_transform
        ):
            raise ValueError("Relational checkpoint target_transform differs from training targets")
        if payload is not None:
            models.load_state_dict(payload["state_dict"])
            metadata = payload["metadata"]
            write_json(out / "model.manifest.json", metadata)
        else:
            metadata = {**expected, **source_provenance(), "stages": stages}
            save_checkpoint(models, out / "model.pt", metadata=metadata)
        if stage_graph is not None:
            save_encoder_state(stage_graph.encoder, out / "encoder.pt", metadata={"schema": schema})
        write_resolved_config(out, config.model_dump_resolved())
        write_environment_manifest(out, config, resolved_device)
        write_json(out / "checkpoint_manifest.json", metadata)
        write_json(out / "tokenizer_manifest.json", {"state": expected["tokenizer"]})
        write_json(
            out / "data_validation_report.json",
            {"status": "passed", "data_sha256": expected["data_sha256"]},
        )
        write_json(
            out / "split_manifest.json",
            {
                "target_level": "entity",
                "anchors": _plain(inputs.anchors),
                "schema": schema,
                "table_row_counts": {name: len(rows) for name, rows in source.rows.items()},
                "loader": {"name": config.data.loader}
                if inputs.sampler is None
                else inputs.sampler.loader_manifest,
            },
        )
        if command == "train":
            return out
        evaluation = {}
        control_evaluation = {}
        arrays = {}
        for split in dict.fromkeys(item.split for item in inputs.anchors):
            batches = inputs.batches(
                split,
                mode=mode,
                batch_size=config.evaluation.inference_batch_size,
                tokenizer=tokenizer,
                max_tokens=config.data.max_tokens,
                amount_key=config.data.amount_field,
                hop_trim=gran.trim,
                require_labels=command != "infer",
            )
            values = _predictions(model, batches, resolved_device)
            evaluation[split] = values
            if sequence_control is not None and command != "infer":
                control_evaluation[split] = _predictions(
                    sequence_control, _history_controls(batches), resolved_device
                )
            arrays[split] = values[0]
            arrays[f"{split}_probabilities"] = values[1]
            arrays[f"{split}_entity_ids"] = np.asarray(
                [item.anchor.entity_id for item in inputs.anchors if item.split == split]
            )
            arrays[f"{split}_labels"] = values[2]
            arrays[f"{split}_labels_valid"] = np.isfinite(values[2])
            arrays[f"{split}_sample_ids"] = np.asarray(
                [item.anchor.sample_id for item in inputs.anchors if item.split == split]
            )
        np.savez(out / "embeddings.npz", **arrays)
        write_json(
            out / "embeddings_manifest.json",
            {
                split: {"shape": list(value[0].shape), "target_level": "entity"}
                for split, value in evaluation.items()
            },
        )
        if command == "infer":
            return out
        if "train" not in evaluation or "test" not in evaluation:
            raise SchemaError("Evaluation needs train and test anchors")
        train, test = evaluation["train"], evaluation["test"]
        metrics = {"task_head": classification_metrics(test[2], test[1])}
        for name, train_features, test_features in (
            ("raw", train[3], test[3]),
            ("embedding", train[0], test[0]),
            (
                "combined",
                np.concatenate((train[3], train[0]), axis=1),
                np.concatenate((test[3], test[0]), axis=1),
            ),
        ):
            metrics[name] = classification_metrics(
                test[2],
                fit_predict_probe(train_features, train[2], test_features, seed=config.seed),
            )
        if control_evaluation:
            control_train, control_test = control_evaluation["train"], control_evaluation["test"]
            metrics["sequence_control_task_head"] = classification_metrics(
                control_test[2], control_test[1]
            )
            metrics["sequence_control_embedding_logistic"] = classification_metrics(
                control_test[2],
                fit_predict_probe(
                    control_train[0], control_train[2], control_test[0], seed=config.seed
                ),
            )
        write_json(out / "metrics.json", metrics)
        (out / "metrics.md").write_text(metrics_markdown(metrics))
        (out / "model_card.md").write_text(
            f"# Relational {mode} reference example\n\nExperimental bounded loading; entity-level binary task.\n\nNo production or quality claim.\n"
        )
        return out
