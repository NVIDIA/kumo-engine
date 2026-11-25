# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe orchestration over source-table native tensor streams."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from tfm.api.builders import build_position, model_from_config, profile_event_spec_from_config
from tfm.api.graph import (
    GraphBuildContext,
    build_graph_architecture,
    context_history,
    graph_architecture,
    task_head,
)
from tfm.attention.registry import build_attention
from tfm.compositions.gnn import GNNEncoder
from tfm.config import RecipeConfig, RelationalTaskConfig, TrainingConfig
from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.graph import (
    GraphBatch,
    GraphHistoryBatch,
    GraphSupervisedBatch,
    GraphTrainingBatch,
    utc_us,
)
from tfm.contracts.resources import Closable
from tfm.data.adapters.registry import table_source
from tfm.data.graph import FutureTargetTransform
from tfm.data.native_inputs import Mode, NativeInputs
from tfm.data.native_targets import NativeFutureTargets
from tfm.data.native_validation import require_training_amount, validate_graph_policy
from tfm.data.relational_batches import RelationalBatch
from tfm.data.splits.temporal import parse_duration, parse_timestamp
from tfm.evaluation.fraud.baselines import fit_predict_probe
from tfm.evaluation.fraud.metrics import classification_metrics
from tfm.evaluation.reporting.report import metrics_markdown, write_json
from tfm.extras.kumo import engine_version
from tfm.heads.binary import Binary
from tfm.layers.gnn import GNNEncoderModel, SequenceBinaryModel
from tfm.pooling.registry import build_readout
from tfm.position.none import NoPE
from tfm.position.time_rope import TimeRoPE
from tfm.runtime.common import (
    default_run_dir,
    source_provenance,
    write_environment_manifest,
    write_resolved_config,
)
from tfm.runtime.relational import _digest, _mode, _plain, _predictions
from tfm.training.checkpointing import (
    load_encoder_state,
    read_checkpoint,
    save_checkpoint,
    save_encoder_state,
)
from tfm.training.device import resolve_device
from tfm.training.reproducibility import seed_everything
from tfm.training.selection import fit_selected
from tfm.training.trainer import BatchSource


def _encoder(model: nn.Module) -> nn.Module:
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("Graph architecture must expose its head-free encoder module")
    return encoder


def _recorded_predictions(
    model: nn.Module,
    batches: BatchSource[Any],
    device: torch.device,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
]:
    entities, samples, graph_rows = [], [], []

    def delivered() -> Iterator[RelationalBatch]:
        source = iter(batches)
        try:
            for batch in source:
                anchors = (
                    batch.graph.anchors
                    if isinstance(batch, (GraphHistoryBatch, GraphSupervisedBatch))
                    else batch.anchors
                )
                graph = (
                    batch.graph
                    if isinstance(batch, (GraphHistoryBatch, GraphSupervisedBatch))
                    else batch
                    if isinstance(batch, GraphBatch)
                    else None
                )
                if graph is not None:
                    graph.assert_validated()
                    for i in range(graph.batch_size):
                        row = []
                        for name, features in graph.node_features.items():
                            selected = features[graph.node_sample[name] == i]
                            row.extend([float(len(selected)), *selected.sum(0).tolist()])
                        graph_rows.append(row)
                entities.extend(anchor.entity_id for anchor in anchors)
                samples.extend(anchor.sample_id for anchor in anchors)
                yield batch
        finally:
            if isinstance(source, Closable):
                source.close()

    values = _predictions(model, delivered(), device, binary_required=False)
    return values, tuple(entities), tuple(samples), np.asarray(graph_rows, dtype=np.float32)


def run_command(
    command: str,
    *,
    config: RecipeConfig,
    config_path: Path | None,
    run_dir: Path | None,
    device: str | None,
    reuse_checkpoint: bool = False,
) -> Path:
    """Execute shared training stages with file-backed sources and native encodings."""
    del config_path
    if device is not None:
        config = config.model_copy(
            update={"runtime": config.runtime.model_copy(update={"device": device})}
        )
    task = config.data.relational
    legacy = task is None
    target_level: Literal["entity", "sequence"] = "sequence" if legacy else "entity"
    future = config.training.objective == "future_subgraph" and config.runtime.stage != "downstream"
    ssl_train: str | tuple[str, ...] = "train"
    ssl_validation = "validation"
    anchor_factory = None
    fit_split: str | tuple[str, ...] = "train"
    if legacy:
        from tfm.data.native_transaction_anchors import SSL_TRAIN, SSL_VALIDATION, graph_anchors

        task = RelationalTaskConfig(
            root="users",
            history_paths={"events": ("events",)},
            cutoffs={"train": config.data.probe_cutoff},
        )

        def build_anchors(source, path):
            return graph_anchors(source, config, path)

        anchor_factory = build_anchors
        if future:
            ssl_train, ssl_validation, fit_split = SSL_TRAIN, SSL_VALIDATION, SSL_TRAIN[0]
    assert task is not None
    observation_seconds = (
        None if legacy else int(parse_duration(config.data.observation_window).total_seconds())
    )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise UnsupportedCombinationError(
            "Native recipe streams currently require single-process eager training"
        )
    if config.runtime.stage == "pretrain" and command != "train":
        raise ValueError("Pretrain-only runs use train; scoring requires downstream supervision")
    mode = (
        (
            "graph"
            if graph_architecture(config.model.composition.granularity.type).layout == "graph"
            else "hybrid"
        )
        if legacy
        else _mode(config)
    )
    needs_history = (mode != "graph" or legacy) and config.runtime.stage != "pretrain"
    if needs_history and (
        config.features.tokenizer != "key_value_time"
        or config.features.tokenizer_options
        or config.features.fields is not None
    ):
        raise UnsupportedCombinationError(
            "Source-native encoding uses typed TableSource allowlists and fitted native encoders; Python tokenizer options require data.loader: python"
        )
    out = run_dir or default_run_dir(recipe=f"native_{mode}")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    resolved_device = resolve_device(device or config.runtime.device)
    identity_config = config.model_dump_resolved()
    identity_config["runtime"].pop("device", None)
    payload = (
        read_checkpoint(out / "model.pt")
        if reuse_checkpoint and command != "train" and (out / "model.pt").is_file()
        else None
    )
    source_data = (
        config.data.model_copy(update={"load_labels": False})
        if config.runtime.stage == "pretrain" and future
        else config.data
    )
    source = table_source(source_data, seed=config.seed, fixture_dir=out / "source-tables")
    input_keys = {
        alias
        for table in source.tables
        for mapping in (table.numeric, table.categorical, table.text)
        for alias in mapping
    }
    if needs_history and input_keys.intersection(config.features.hashed_keys):
        raise UnsupportedCombinationError(
            "Native history hash projection is required for configured hashed_keys"
        )
    gran = config.model.composition.granularity
    with NativeInputs(
        source,
        root=task.root,
        history_paths=task.history_paths,
        task=task,
        anchor_factory=anchor_factory,
        fit_split=fit_split,
        target_level=target_level,
        work_dir=out / "native-work",
        max_events=config.data.sequence_length,
        batch_size=config.training.batch_size,
        depth=gran.depth,
        fanout=gran.fanout,
        observation_seconds=observation_seconds,
        amount_key=config.data.amount_field,
        features=config.features,
    ) as inputs:
        if payload is not None:
            if payload["metadata"].get("format") != "tfm.source-native-experiment.v1" or payload[
                "metadata"
            ].get("config_sha256") != _digest(identity_config):
                raise ValueError(
                    "Native checkpoint config identity differs from requested experiment"
                )
            requested = inputs.manifest()
            saved = payload["metadata"].get("input_loader", {})
            if _digest({k: v for k, v in requested.items() if k != "tasks"}) != _digest(
                {k: v for k, v in saved.items() if k != "tasks"}
            ):
                raise ValueError(
                    "Native checkpoint source/anchor identity differs from requested experiment"
                )
            inputs.restore_fitted(out / "native-fitted", saved.get("tasks", {}))
        require_training_amount(
            inputs.normalized,
            root=task.root,
            paths=task.history_paths,
            amount_key=config.data.amount_field,
            observation_seconds=observation_seconds,
            split=SSL_TRAIN[-1] if legacy and future else "train",
        )
        if mode != "sequence":
            validate_graph_policy(
                inputs.normalized,
                root=task.root,
                paths=task.history_paths,
                fanout=gran.fanout,
                observation_seconds=observation_seconds,
                canonical_ties=not legacy,
            )
        for name in (
            ("graph",)
            if not needs_history
            else ("history",)
            if mode == "sequence"
            else ("graph", "history")
            if mode == "graph"
            else ("graph", "hybrid", "history")
        ):
            inputs.task(name)
        graph_spec = None
        graph_tables = [
            table
            for table in source.tables
            if table.role == "input"
            and (source.graph_numeric is None or table.name in source.graph_numeric)
        ]
        names = {table.name for table in graph_tables}
        relations = [
            (role.parent, role.name, role.child)
            for role in source.relationships
            if role.parent in names and role.child in names
        ]
        if mode != "sequence":
            graph_spec = GNNEncoder(
                tuple(
                    (
                        table.name,
                        (
                            len(source.graph_numeric[table.name])
                            if source.graph_numeric is not None
                            else max(1, 2 * len(table.numeric))
                        ),
                    )
                    for table in graph_tables
                ),
                tuple(
                    edge
                    for parent, role, child in relations
                    for edge in ((parent, role, child), (child, f"rev_{role}", parent))
                ),
                task.root,
                foreign_key_relations=tuple(
                    (child, f"rev_{role}", parent) for parent, role, child in relations
                ),
                hidden_size=gran.hidden_size,
                depth=gran.depth,
                trim=gran.trim,
            )
        schema = {
            "tables": _plain(source.tables),
            "relationships": _plain(source.relationships),
            "graph_numeric": _plain(source.graph_numeric),
            "root": task.root,
            "history_paths": task.history_paths,
            "profile_paths": _plain(source.profile_paths),
            "event_labels": _plain(source.event_labels),
            "target_level": target_level,
        }
        identity_config = config.model_dump_resolved()
        identity_config["runtime"].pop("device", None)
        future = (
            mode != "sequence"
            and config.training.objective == "future_subgraph"
            and config.runtime.stage != "downstream"
        )
        cache, target_transform = None, None
        if future:
            cache = inputs.resources.enter_context(
                NativeFutureTargets(
                    inputs.normalized,
                    root=task.root,
                    depth=gran.target_depth,
                    fanout=gran.fanout,
                    horizon_us=config.data.horizon_days * 86_400_000_000,
                    observed_through_us=utc_us(
                        source.observed_through or parse_timestamp(config.data.observed_through)
                    ),
                    partition_end_us=utc_us(parse_timestamp(config.data.pretrain_end)),
                    label_delay_us=config.data.label_delay_days * 86_400_000_000,
                )
            )
            cache.prepare(
                (*((ssl_train,) if isinstance(ssl_train, str) else ssl_train), ssl_validation)
            )
            target_transform = cache.fit(ssl_train)
        encoder_schema = {
            "amount_field": config.data.amount_field,
            "feature_transform": "transaction-signed-log-v1"
            if legacy
            else "numeric-value-presence-v1"
            if source.graph_numeric is None
            else "declared-source-scalars-v1",
            "graph": _plain(graph_spec),
            "tables": _plain(tuple(table for table in source.tables if table.name in names)),
            "relationships": relations,
            "graph_numeric": _plain(source.graph_numeric),
        }
        expected = {
            "format": "tfm.source-native-experiment.v1",
            "architecture": config.model.composition.granularity.type,
            "graph_objective": "future_subgraph" if future else "binary",
            "graph_schema": _plain(graph_spec),
            "hop_trimming": None if graph_spec is None else graph_spec.trim,
            "sequence_attention": config.model.composition.attention if needs_history else None,
            "kumo_engine_version": None if graph_spec is None else engine_version(),
            "config_sha256": _digest(identity_config),
            "input_loader": inputs.manifest(),
            "schema": schema,
            "target_transform": None if target_transform is None else _plain(target_transform),
            "tokenizer": None
            if not needs_history
            else {
                "backend": "kumo_pytorch",
                "fitted_state": inputs.task("history").fitted_state.fingerprint,
            },
        }
        if payload is not None:
            for key, value in expected.items():
                if _digest(payload["metadata"].get(key)) != _digest(value):
                    raise ValueError(f"Native checkpoint {key} differs from requested experiment")
        stages, models = {}, nn.ModuleDict()

        def batches(
            split: str,
            *,
            selected_mode: Mode = mode,
            settings: TrainingConfig | None = None,
            sharing: Literal["shared", "disjoint"] = "disjoint",
            require_labels: bool = True,
        ) -> BatchSource[Any]:
            settings = settings or config.training
            return inputs.batches(
                split,
                mode=selected_mode,
                batch_size=settings.batch_size,
                max_tokens=config.data.max_tokens,
                seed=config.seed,
                shuffle=split == "train" and settings.shuffle,
                require_labels=require_labels,
                sharing=sharing,
                hop_trim=gran.trim,
            )

        graph_stage = None
        encoder_source = None
        if graph_spec is not None:
            if future:
                assert cache is not None and target_transform is not None
                transform: FutureTargetTransform = target_transform
                target_cache: NativeFutureTargets = cache

                def targets(value: Any) -> GraphTrainingBatch:
                    graph = value.graph if isinstance(value, GraphSupervisedBatch) else value
                    if not isinstance(graph, GraphBatch):
                        raise TypeError("FutureSubgraph inputs require graph batches")
                    return GraphTrainingBatch(
                        graph, transform.apply(target_cache.raw(graph.anchors))
                    )

                columns = tuple(transform.input_columns[i] for i in transform.selected)
                graph_stage = build_graph_architecture(
                    gran.encoder_type
                    if mode == "hybrid"
                    else config.model.composition.granularity.type,
                    config,
                    GraphBuildContext(
                        graph_spec, target_columns=columns, target_level=target_level
                    ),
                ).to(resolved_device)
                if payload is None:
                    stages["future_subgraph"] = fit_selected(
                        graph_stage,
                        inputs.batches(
                            ssl_train,
                            mode="graph",
                            batch_size=config.training.batch_size,
                            seed=config.seed,
                            shuffle=config.training.shuffle,
                            sharing=config.data.training_graphs,
                            hop_trim=gran.trim,
                            decorate=targets,
                        ),
                        inputs.batches(
                            ssl_validation,
                            mode="graph",
                            batch_size=config.training.batch_size,
                            hop_trim=gran.trim,
                            decorate=targets,
                        ),
                        config.training,
                        seed=config.seed,
                        device=resolved_device,
                    )
            else:
                graph_stage = build_graph_architecture(
                    gran.encoder_type
                    if mode == "hybrid"
                    else config.model.composition.granularity.type,
                    config,
                    GraphBuildContext(graph_spec, target_level=target_level),
                ).to(resolved_device)
                if config.runtime.stage == "downstream":
                    assert config.runtime.encoder_checkpoint is not None
                    encoder_expected = {"schema": encoder_schema}
                    if payload is not None:
                        encoder_expected.update(payload["metadata"].get("encoder_source") or {})
                    encoder_source = load_encoder_state(
                        _encoder(graph_stage),
                        config.runtime.encoder_checkpoint,
                        expected=encoder_expected,
                    )
                if payload is None and (config.runtime.stage != "downstream" or mode == "graph"):
                    stages["graph_binary"] = fit_selected(
                        graph_stage,
                        batches(
                            "train", selected_mode="graph", sharing=config.data.training_graphs
                        ),
                        batches("validation", selected_mode="graph"),
                        config.training,
                        seed=config.seed,
                        device=resolved_device,
                    )
            models["graph_stage"] = graph_stage
        sequence_control = None
        if mode == "graph" or config.runtime.stage == "pretrain":
            assert graph_stage is not None
            model: nn.Module = graph_stage
            if future and config.runtime.stage != "pretrain" and not legacy:
                assert graph_spec is not None
                downstream = GNNEncoderModel(
                    graph_spec, objective=Binary(target_level=target_level)
                ).to(resolved_device)
                downstream.encoder.load_state_dict(_encoder(graph_stage).state_dict())
                if payload is None:
                    stages["graph_downstream"] = fit_selected(
                        downstream,
                        batches(
                            "train", selected_mode="graph", sharing=config.data.training_graphs
                        ),
                        batches("validation", selected_mode="graph"),
                        config.training,
                        seed=config.seed,
                        device=resolved_device,
                    )
                model = downstream
        else:
            if legacy:
                seed_everything(config.seed)
            native_task = inputs.task("history")
            sequence_spec = profile_event_spec_from_config(
                config, vocab_size=native_task.spaces["model"].size
            )
            position = build_position(config.model.composition.position)
            if not isinstance(position, (NoPE, TimeRoPE)):
                raise UnsupportedCombinationError(
                    "Native history position must be nope or time_rope"
                )
            kwargs = {
                "attention": build_attention(config.model.composition.attention),
                "position": position,
                "readout": build_readout(config.model.composition.readout),
                "target_level": target_level,
                "history_model": context_history(
                    gran.history_model,
                    sequence_spec,
                    build_attention(config.model.composition.attention),
                    position=position,
                ),
                "head": task_head(gran.task_head, sequence_spec.hidden_size),
            }
            if mode == "sequence":
                model = SequenceBinaryModel(sequence_spec, **kwargs).to(resolved_device)
            else:
                assert graph_spec is not None and graph_stage is not None
                model = model_from_config(
                    config,
                    vocab_size=sequence_spec.vocab_size,
                    graph_context=GraphBuildContext(
                        graph_spec,
                        sequence_spec,
                        encoder=_encoder(graph_stage),
                        target_level=target_level,
                    ),
                ).to(resolved_device)
                sequence_control = SequenceBinaryModel(sequence_spec, **kwargs).to(resolved_device)
                initial = model.state_dict()
                sequence_control.load_state_dict(
                    {name: initial[name] for name in sequence_control.state_dict()}
                )
                models["sequence_control"] = sequence_control
            if payload is None:
                settings = config.training if mode == "sequence" else config.sequence_training
                stages[mode] = fit_selected(
                    model,
                    batches("train", settings=settings, sharing=config.data.training_graphs),
                    batches("validation", settings=settings),
                    settings,
                    seed=config.seed,
                    device=resolved_device,
                )
                if sequence_control is not None:
                    stages["sequence_control"] = fit_selected(
                        sequence_control,
                        batches("train", selected_mode="sequence", settings=settings),
                        batches("validation", selected_mode="sequence", settings=settings),
                        settings,
                        seed=config.seed,
                        device=resolved_device,
                    )
        historical_control = None
        if legacy and needs_history:
            history_task = inputs.task("history")
            spec = profile_event_spec_from_config(
                config, vocab_size=history_task.spaces["model"].size
            )
            if mode == "graph":
                seed_everything(config.seed)
                control_position = build_position(config.model.composition.position)
                if not isinstance(control_position, (NoPE, TimeRoPE)):
                    raise UnsupportedCombinationError(
                        "Native history position must be nope or time_rope"
                    )
                sequence_control = SequenceBinaryModel(
                    spec,
                    attention=build_attention(config.model.composition.attention),
                    position=control_position,
                    readout=build_readout(config.model.composition.readout),
                    history_model=context_history(
                        gran.history_model,
                        spec,
                        build_attention(config.model.composition.attention),
                        position=control_position,
                    ),
                    head=task_head(gran.task_head, spec.hidden_size),
                    target_level=target_level,
                ).to(resolved_device)
                models["sequence_control"] = sequence_control
                if payload is None:
                    stages["sequence_control"] = fit_selected(
                        sequence_control,
                        batches(
                            "train", selected_mode="sequence", settings=config.sequence_training
                        ),
                        batches(
                            "validation",
                            selected_mode="sequence",
                            settings=config.sequence_training,
                        ),
                        config.sequence_training,
                        seed=config.seed,
                        device=resolved_device,
                    )
            if future:
                from tfm.api.builders import profile_event_decoder_from_config

                historical_config = config.model_copy(
                    update={
                        "model": config.model.model_copy(
                            update={
                                "composition": config.model.composition.model_copy(
                                    update={"objective": "next_event_fields"}
                                )
                            }
                        )
                    }
                )
                seed_everything(config.seed)
                historical_control = profile_event_decoder_from_config(
                    historical_config, vocab_size=history_task.spaces["model"].size
                ).to(resolved_device)
                models["historical_control"] = historical_control
                if payload is None:
                    stages["historical_control"] = fit_selected(
                        historical_control,
                        inputs.batches(
                            ssl_train,
                            mode="history",
                            batch_size=config.sequence_training.batch_size,
                            max_tokens=config.data.max_tokens,
                            seed=config.seed,
                            shuffle=config.sequence_training.shuffle,
                        ),
                        inputs.batches(
                            ssl_validation,
                            mode="history",
                            batch_size=config.sequence_training.batch_size,
                            max_tokens=config.data.max_tokens,
                        ),
                        config.sequence_training,
                        seed=config.seed,
                        device=resolved_device,
                    )
        models["model"] = model
        expected["encoder_source"] = encoder_source
        if payload is not None:
            if _digest(payload["metadata"].get("encoder_source")) != _digest(encoder_source):
                raise ValueError("Native checkpoint encoder_source differs from requested encoder")
            models.load_state_dict(payload["state_dict"])
            metadata = payload["metadata"]
            write_json(out / "model.manifest.json", metadata)
        else:
            metadata = {**expected, **source_provenance(), "stages": stages}
            save_checkpoint(models, out / "model.pt", metadata=metadata)
        if graph_stage is not None:
            save_encoder_state(
                _encoder(graph_stage), out / "encoder.pt", metadata={"schema": encoder_schema}
            )
        inputs.save_fitted(out / "native-fitted")
        inputs.save_anchors(out / "anchors.parquet")
        write_resolved_config(out, config.model_dump_resolved())
        write_environment_manifest(out, config, resolved_device)
        write_json(out / "checkpoint_manifest.json", metadata)
        write_json(
            out / "tokenizer_manifest.json",
            {"backend": "kumo_pytorch", "input_loader": inputs.manifest()},
        )
        write_json(
            out / "data_validation_report.json",
            {"status": "passed", "input_loader": inputs.manifest()},
        )
        splits = [
            row[0]
            for row in inputs.normalized.connection.execute(
                "SELECT DISTINCT split FROM anchors ORDER BY rowid"
            )
            if not legacy or row[0] in {"train", "validation", "test"}
        ]
        write_json(
            out / "split_manifest.json",
            {
                "target_level": target_level,
                "counts": dict(
                    inputs.normalized.connection.execute(
                        "SELECT split,COUNT(*) FROM anchors GROUP BY split"
                    )
                ),
                "input_loader": inputs.manifest(),
            },
        )
        if command == "train":
            return out
        evaluation, controls, historical, graph_features, arrays = {}, {}, {}, {}, {}
        for split in splits:
            stream = inputs.batches(
                split,
                mode=mode,
                batch_size=config.evaluation.inference_batch_size,
                max_tokens=config.data.max_tokens,
                require_labels=command != "infer",
                hop_trim=gran.trim,
            )
            values, entity_ids, sample_ids, graph_features[split] = _recorded_predictions(
                model, stream, resolved_device
            )
            if command == "infer":
                labels = np.asarray(
                    [
                        inputs.normalized.connection.execute(
                            "SELECT target FROM anchor_labels WHERE sample_id=?", (sid,)
                        ).fetchone()[0]
                        for sid in sample_ids
                    ],
                    dtype=np.float32,
                )
                values = values[0], values[1], labels, values[3]
            evaluation[split] = values
            arrays[split], arrays[f"{split}_probabilities"], arrays[f"{split}_labels"] = values[:3]
            arrays[f"{split}_labels_valid"] = np.isfinite(values[2])
            arrays[f"{split}_sample_ids"] = np.asarray(sample_ids)
            arrays[f"{split}_entity_ids"] = np.asarray(entity_ids)
            if sequence_control is not None and command != "infer":
                controls[split] = _predictions(
                    sequence_control,
                    inputs.batches(
                        split,
                        mode="sequence",
                        batch_size=config.evaluation.inference_batch_size,
                        max_tokens=config.data.max_tokens,
                        require_labels=True,
                    ),
                    resolved_device,
                )
            if historical_control is not None and command != "infer":
                from tfm.runtime.native_sequence import predictions

                historical[split] = predictions(
                    historical_control,
                    inputs.batches(
                        split,
                        mode="history",
                        batch_size=config.evaluation.inference_batch_size,
                        max_tokens=config.data.max_tokens,
                        require_labels=True,
                    ),
                    resolved_device,
                )
        np.savez(out / "embeddings.npz", **arrays)
        write_json(
            out / "embeddings_manifest.json",
            {
                split: {"shape": list(value[0].shape), "target_level": target_level}
                for split, value in evaluation.items()
            },
        )
        if command == "infer":
            return out
        if not {"train", "test"} <= evaluation.keys():
            raise SchemaError("Evaluation requires train and test anchors")
        train, test = evaluation["train"], evaluation["test"]
        metrics = (
            {"task_head": classification_metrics(test[2], test[1])}
            if np.isfinite(test[1]).all()
            else {}
        )
        train_raw, test_raw = (
            (controls["train"][3], controls["test"][3])
            if legacy and controls
            else (train[3], test[3])
        )
        for name, x, y in [
            ("raw", train_raw, test_raw),
            ("embedding", train[0], test[0]),
            (
                "combined",
                np.concatenate((train_raw, train[0]), 1),
                np.concatenate((test_raw, test[0]), 1),
            ),
        ]:
            metrics[name] = classification_metrics(
                test[2], fit_predict_probe(x, train[2], y, seed=config.seed)
            )
        if controls:
            metrics["sequence_control_task_head"] = classification_metrics(
                controls["test"][2], controls["test"][1]
            )
            metrics["sequence_control_embedding_logistic"] = classification_metrics(
                controls["test"][2],
                fit_predict_probe(
                    controls["train"][0],
                    controls["train"][2],
                    controls["test"][0],
                    seed=config.seed,
                ),
            )
        if legacy:
            metrics["graph_tabular"] = classification_metrics(
                test[2],
                fit_predict_probe(
                    graph_features["train"], train[2], graph_features["test"], seed=config.seed
                ),
            )
            if controls:
                metrics["sequence_supervised_embedding"] = metrics[
                    "sequence_control_embedding_logistic"
                ]
                metrics["sequence_supervised_task_head"] = metrics["sequence_control_task_head"]
                if not historical:
                    metrics["sequence_only_embedding"] = metrics[
                        "sequence_control_embedding_logistic"
                    ]
        if historical:
            metrics["historical_sequence_embedding_logistic"] = classification_metrics(
                historical["test"]["labels"],
                fit_predict_probe(
                    historical["train"]["vectors"],
                    historical["train"]["labels"],
                    historical["test"]["vectors"],
                    seed=config.seed,
                ),
            )
        if legacy and historical:
            metrics["sequence_only_embedding"] = metrics["historical_sequence_embedding_logistic"]
        write_json(out / "metrics.json", metrics)
        (out / "metrics.md").write_text(metrics_markdown(metrics))
        (out / "model_card.md").write_text(
            f"# Source-native {mode} reference example\n\nExperimental native retrieval and fitted tensor encoding.\n\nNo production or quality claim.\n"
        )
        return out
