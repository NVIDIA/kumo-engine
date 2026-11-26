# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequence recipes over continuous source-table native tensor deliveries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from tfm.config import RecipeConfig
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.graph import utc_us
from tfm.contracts.tokenizers import FieldVocabulary
from tfm.data.adapters.registry import table_source
from tfm.data.native_inputs import Mode, NativeInputs
from tfm.data.native_transaction_anchors import sequence_anchors
from tfm.data.native_validation import require_training_amount
from tfm.data.native_views import SourceRows
from tfm.data.splits.entity import split_entities
from tfm.data.splits.temporal import parse_duration, parse_timestamp
from tfm.evaluation.fraud.baselines import fit_predict_probe
from tfm.evaluation.fraud.metrics import classification_metrics
from tfm.evaluation.reporting.report import metrics_markdown, write_json
from tfm.runtime.common import (
    default_run_dir,
    source_provenance,
    write_environment_manifest,
    write_resolved_config,
)
from tfm.runtime.pipeline import build_model
from tfm.runtime.relational import _digest
from tfm.training.checkpointing import read_checkpoint, save_checkpoint
from tfm.training.device import resolve_device
from tfm.training.reproducibility import seed_everything
from tfm.training.trainer import build_trainer, move_batch_for_execution


def _field_vocabulary(task: Any) -> FieldVocabulary:
    """Use native namespace allocations instead of rebuilding Python token IDs."""
    import json

    state = task.fitted_state.to_dict()
    ranges = []
    for key, members in state["spaces"]["model"]["members"].items():
        identity = json.loads(key)
        if identity[:3] != ["inputs", "history", "fields"]:
            continue
        ids = sorted(global_id for local_id, global_id in members if local_id >= 2)
        if not ids or ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError("Native financial fields require contiguous ordinary ID ranges")
        ranges.append((ids[0], ids[-1] + 1, identity[-1].rsplit("/", 1)[-1].removeprefix("flat_")))
    ranges.sort()
    return FieldVocabulary(
        task.spaces["model"].size,
        tuple(row[2] for row in ranges),
        tuple((row[0], row[1]) for row in ranges),
        0,
        1,
        3,
        2,
    )


@torch.no_grad()
def predictions(
    model: nn.Module,
    batches: Any,
    device: torch.device,
    precision: Literal["fp32", "bf16"] = "fp32",
) -> dict[str, np.ndarray]:
    """Retain derived evaluation arrays, never a split of input model tensors."""
    model.eval()
    arrays: dict[str, list[Any]] = {
        key: [] for key in ("vectors", "raw", "labels", "entities", "samples")
    }
    iterator = iter(batches)
    try:
        for original in iterator:
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
            ):
                output = model(move_batch_for_execution(original, device), compute_loss=False)
            arrays["vectors"].extend(output.embeddings.detach().float().cpu().numpy())
            if original.numeric_features is None:
                raise ValueError("Native sequence delivery requires aligned raw features")
            arrays["raw"].extend(original.numeric_features.numpy())
            labels = original.fraud_labels
            arrays["labels"].extend(
                [float("nan")] * original.batch_size if labels is None else labels.tolist()
            )
            arrays["entities"].extend(original.entity_ids)
            arrays["samples"].extend(original.sequence_ids)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    return {key: np.asarray(value) for key, value in arrays.items()}


def run_command(
    command: str,
    *,
    config: RecipeConfig,
    config_path: Path | None,
    run_dir: Path | None,
    device: str | None,
    reuse_checkpoint: bool = False,
) -> Path:
    """Train and score without an event corpus, tokenizer fit or whole-split inputs."""
    del config_path
    if device is not None:
        config = config.model_copy(
            update={"runtime": config.runtime.model_copy(update={"device": device})}
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise UnsupportedCombinationError(
            "Source-native recipe streams currently require single-process training; select Python or an explicit prepared token cache for distributed runs"
        )
    if config.runtime.stage != "full":
        raise UnsupportedCombinationError(
            "Explicit encoder stages require the graph/relational recipe runner"
        )
    if config.fraud_task.finetune.enabled or config.evaluation.surprise:
        raise UnsupportedCombinationError(
            "Source-native fine-tuning/surprise are not wired yet; use data.loader: python"
        )
    if command != "train" and not config.data.load_labels:
        raise ValueError("Scoring a binary sequence task requires an available label source")
    if config.data.relational is not None:
        raise ValueError("Relational sequence tasks use the shared relational runner")
    if config.features.tokenizer_options:
        raise UnsupportedCombinationError(
            "Native source encoding does not accept Python tokenizer constructors; use data.loader: python"
        )
    kind = config.features.tokenizer
    mode: Mode = "flat" if kind == "transaction_field_tokenizer" else "history"
    if kind not in {"transaction_field_tokenizer", "key_value_time"}:
        raise UnsupportedCombinationError(
            "Source-native sequence delivery currently requires financial fields or key_value_time"
        )
    if config.training.packing:
        raise UnsupportedCombinationError(
            "Native stream packing is not implemented; use an explicit token cache or data.loader: python"
        )
    windows = config.data.pretrain_windows
    if windows is not None and (mode != "flat" or config.fraud_task.supervised_head.enabled):
        raise UnsupportedCombinationError(
            "Native window groups currently support flat self-supervised training; other window recipes require an explicit token cache or data.loader: python"
        )
    if config.data.fraud_labels != "native":
        raise UnsupportedCombinationError(
            "Source-native spliced task projections are not implemented; use data.loader: python"
        )
    out = run_dir or default_run_dir(recipe="native_sequence")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    resolved_device = resolve_device(device or config.runtime.device)
    identity = config.model_dump_resolved()
    identity["runtime"].pop("device", None)
    payload = (
        read_checkpoint(out / "model.pt")
        if reuse_checkpoint and command != "train" and (out / "model.pt").is_file()
        else None
    )
    source = table_source(config.data, seed=config.seed, fixture_dir=out / "source-tables")
    fixed = None
    if mode == "flat":
        from tfm.data.native_financial import financial_source
        from tfm.features.tokenizers.financial import FIELD_ORDER

        source, fixed = financial_source(
            source, tuple(config.features.fields or FIELD_ORDER), out / "financial-source"
        )
    elif any(
        key in mapping
        for table in source.tables
        for mapping in (table.numeric, table.categorical, table.text)
        for key in config.features.hashed_keys
    ):
        raise UnsupportedCombinationError(
            "Native hash projection is required for active hashed_keys; use data.loader: python"
        )
    if windows is not None:
        from tfm.data.native_windows import flat_window_source

        source = flat_window_source(source, config, out / "window-source")
    if config.data.split_protocol == "precomputed":
        views = {
            "train": {"events": SourceRows(partition_column="partition", partitions=("train",))},
            "validation": {
                "events": SourceRows(
                    partition_column="partition", partitions=("train", "validation")
                )
            },
        }
    else:
        views = {
            side: {"events": SourceRows(before_us=utc_us(parse_timestamp(cutoff)))}
            for side, cutoff in (
                ("train", config.data.temporal_split.train_end),
                ("validation", config.data.temporal_split.validation_end),
            )
        }
    forecast_splits = {
        f"forecast_{table.name.removeprefix('labels_')}_{side}": table.name
        for table in source.tables
        if table.role == "labels" and table.name.startswith("labels_")
        for side in ("train", "test")
    }
    with NativeInputs(
        source,
        root="users",
        history_paths={
            "events": ("events",),
            **({"window_events": ("window_events",)} if windows is not None else {}),
        },
        anchor_factory=lambda source, path: sequence_anchors(source, config, path),
        work_dir=out / "native-work",
        max_events=max(config.data.sequence_length, windows.events if windows is not None else 0),
        batch_size=config.training.batch_size,
        fixed_vocabularies=fixed,
        split_families={"history_train": "train", "history_test": "test", "pretrain": "train"},
        split_namespaces={
            "history_train": "history_window",
            "history_test": "history_window",
            "pretrain": "pretraining",
            **forecast_splits,
        },
        fit_split="pretrain" if windows is not None else "train",
        exclusive_terminal_paths=windows is not None,
        fit_view="train",
        views=views,
        split_views={
            "train": "train",
            "validation": "validation",
            "test": "full",
            "pretrain": "train",
            "history_train": "full",
            "history_test": "full",
            **dict.fromkeys(forecast_splits, "full"),
        },
        target_level="sequence",
        observation_seconds=int(parse_duration(config.data.observation_window).total_seconds()),
        amount_key=config.data.amount_field,
        features=config.features,
    ) as inputs:
        if payload is not None:
            metadata = payload["metadata"]
            if metadata.get("format") != "tfm.source-native-sequence.v1" or metadata.get(
                "config_sha256"
            ) != _digest(identity):
                raise ValueError(
                    "Native sequence checkpoint configuration differs from requested experiment"
                )
            saved, requested = metadata.get("input_loader", {}), inputs.manifest()
            if _digest({k: v for k, v in saved.items() if k != "tasks"}) != _digest(
                {k: v for k, v in requested.items() if k != "tasks"}
            ):
                raise ValueError("Native sequence checkpoint source/anchor identity differs")
            inputs.restore_fitted(out / "native-fitted", saved.get("tasks", {}))
        require_training_amount(
            inputs.normalized,
            root="users",
            paths={"events": ("events",)},
            amount_key=config.data.amount_field,
            exclusive_cutoff=config.data.split_protocol != "precomputed",
            observation_seconds=int(parse_duration(config.data.observation_window).total_seconds()),
        )
        task = inputs.task(mode)
        inputs.task(mode, view="validation")
        inputs.task(mode, view="full")
        model = build_model(
            config,
            task.spaces["model"].size,
            vocabulary=_field_vocabulary(task) if mode == "flat" else None,
        ).to(resolved_device)
        expected = {
            "format": "tfm.source-native-sequence.v1",
            "config_sha256": _digest(identity),
            "input_loader": inputs.manifest(),
            "layout": "flat_fields" if mode == "flat" else "key_value_time",
        }
        if payload is not None:
            if any(
                _digest(payload["metadata"].get(k)) != _digest(value)
                for k, value in expected.items()
            ):
                raise ValueError("Native sequence checkpoint encoding metadata differs")
            model.load_state_dict(payload["state_dict"])
            write_json(out / "model.manifest.json", payload["metadata"])
        else:
            settings = config.training
            stream = inputs.batches(
                "pretrain" if windows is not None else "train",
                mode=mode,
                max_tokens=(windows.max_tokens or config.data.max_tokens)
                if windows is not None
                else config.data.max_tokens,
                history_limit=windows.events
                if windows is not None
                else config.data.sequence_length,
                seed=config.seed,
                shuffle=settings.shuffle,
            )
            stats = build_trainer(settings.backend, checkpoint_dir=out / "hf_checkpoint").fit(
                model,
                stream,
                max_steps=settings.max_steps,
                learning_rate=settings.learning_rate,
                seed=config.seed,
                device=resolved_device,
                optimizer=settings.optimizer,
                shuffle=False,
                precision=settings.precision,
            )
            save_checkpoint(
                model,
                out / "model.pt",
                metadata={**expected, **source_provenance(), "stats": stats},
            )
        inputs.save_fitted(out / "native-fitted")
        inputs.save_anchors(out / "anchors.parquet")
        write_resolved_config(out, config.model_dump_resolved())
        write_environment_manifest(out, config, resolved_device)
        write_json(out / "tokenizer_manifest.json", {"backend": "kumo_pytorch", **expected})
        write_json(out / "checkpoint_manifest.json", expected)
        write_json(
            out / "data_validation_report.json",
            {"status": "passed", "input_loader": inputs.manifest()},
        )
        write_json(
            out / "split_manifest.json",
            {
                "target_level": "sequence",
                "input_loader": inputs.manifest(),
                "counts": dict(
                    inputs.normalized.connection.execute(
                        "SELECT split,COUNT(*) FROM anchors GROUP BY split"
                    )
                ),
            },
        )
        if command == "train":
            return out
        arrays = {
            side: predictions(
                model,
                inputs.batches(
                    side,
                    mode=mode,
                    batch_size=config.evaluation.inference_batch_size,
                    max_tokens=config.data.max_tokens,
                    history_limit=config.data.sequence_length,
                    view=side if side in views else "full",
                ),
                resolved_device,
                config.evaluation.inference_precision,
            )
            for side in ("train", "validation", "test")
        }
        saved_arrays: dict[str, Any] = {
            f"{side}_{key}": value
            for side, values in arrays.items()
            for key, value in values.items()
        }
        np.savez(out / "embeddings.npz", **saved_arrays)
        write_json(
            out / "embeddings_manifest.json",
            {
                side: {"shape": list(values["vectors"].shape), "target_level": "sequence"}
                for side, values in arrays.items()
            },
        )
        if command == "infer":
            return out
        train, test = arrays["train"], arrays["test"]
        from tfm.contracts.errors import InsufficientLabelSupportError
        from tfm.data.native_controls import last_event_control

        tasks: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        manifest: dict[str, Any] = {"tasks": {}, "skipped": {}}
        try:
            split = split_entities(
                list(
                    zip(test["entities"].tolist(), test["labels"].astype(int).tolist(), strict=True)
                ),
                seed=config.seed,
            )
        except InsufficientLabelSupportError as error:
            manifest["skipped"]["fraud"] = str(error)
        else:
            heldout = set(split.evaluation)
            tr = np.asarray([owner not in heldout for owner in train["entities"]], dtype=np.bool_)
            te = np.asarray([owner in heldout for owner in test["entities"]], dtype=np.bool_)
            tasks["fraud"] = (
                {key: value[tr] for key, value in train.items()},
                {key: value[te] for key, value in test.items()},
            )
            manifest["fraud_split"] = split.as_dict()
        if config.evaluation.history_window.enabled:
            counts = dict(
                inputs.normalized.connection.execute(
                    "SELECT split,COUNT(*) FROM anchors GROUP BY split"
                )
            )
            if all(counts.get(f"history_{side}", 0) for side in ("train", "test")):
                history = tuple(
                    predictions(
                        model,
                        inputs.batches(
                            f"history_{side}",
                            mode=mode,
                            batch_size=config.evaluation.inference_batch_size,
                            max_tokens=config.data.max_tokens,
                            history_limit=config.data.sequence_length,
                            view="full",
                        ),
                        resolved_device,
                    )
                    for side in ("train", "test")
                )
                single = tuple(
                    predictions(
                        model,
                        inputs.batches(
                            f"history_{side}",
                            mode=mode,
                            batch_size=config.evaluation.inference_batch_size,
                            max_tokens=config.data.max_tokens,
                            history_limit=config.data.sequence_length,
                            view="full",
                            decorate=last_event_control,
                        ),
                        resolved_device,
                    )
                    for side in ("train", "test")
                )
                if all(len(set(row["labels"].tolist())) == 2 for row in history):
                    tasks["history_window"] = history[0], history[1]
                    tasks["single_event"] = single[0], single[1]
                else:
                    manifest["skipped"]["history_window"] = "one class on a side"
            else:
                manifest["skipped"]["history_window"] = "no labelled events on one side"
        for (task_name,) in inputs.normalized.connection.execute(
            "SELECT DISTINCT task FROM forecast_labels ORDER BY task"
        ):
            sides = []
            for side in ("train", "test"):
                values = predictions(
                    model,
                    inputs.batches(
                        f"forecast_{task_name}_{side}",
                        mode=mode,
                        batch_size=config.evaluation.inference_batch_size,
                        max_tokens=config.data.max_tokens,
                        history_limit=config.data.sequence_length,
                        view="full",
                    ),
                    resolved_device,
                    config.evaluation.inference_precision,
                )
                values["labels"] = np.asarray(
                    [
                        inputs.normalized.connection.execute(
                            "SELECT value FROM forecast_labels WHERE sample_id=?", (sid,)
                        ).fetchone()[0]
                        for sid in values["samples"]
                    ],
                    dtype=np.float32,
                )
                sides.append(values)
            tasks[task_name] = sides[0], sides[1]
        manifest["skipped"].update(
            dict(inputs.normalized.connection.execute("SELECT task,reason FROM task_skips"))
        )
        metrics: dict[str, dict[str, float]] = {}
        baselines = config.evaluation.tabular_baselines
        for name, (left, right) in tasks.items():
            prefix = "" if name == "fraud" else f"{name}_"
            manifest["tasks"][name] = {
                "n_train": len(left["labels"]),
                "n_eval": len(right["labels"]),
                "target_level": "sequence"
                if name == "fraud"
                else "event"
                if name in {"history_window", "single_event"}
                else "entity",
            }
            if baselines.enabled and baselines.xgboost_enabled:
                continue
            for arm, x, y in (
                ("raw", left["raw"], right["raw"]),
                ("embedding", left["vectors"], right["vectors"]),
                (
                    "combined",
                    np.concatenate((left["raw"], left["vectors"]), 1),
                    np.concatenate((right["raw"], right["vectors"]), 1),
                ),
            ):
                if (
                    (
                        arm == "raw"
                        and (name == "single_event" or not config.fraud_task.use_raw_features)
                    )
                    or (arm == "embedding" and not config.fraud_task.use_tfm_embeddings)
                    or (
                        arm == "combined"
                        and not (
                            config.fraud_task.use_raw_features
                            and config.fraud_task.use_tfm_embeddings
                        )
                    )
                ):
                    continue
                metrics[prefix + arm] = classification_metrics(
                    right["labels"],
                    fit_predict_probe(
                        x, left["labels"], y, seed=config.seed, head=config.fraud_task.head
                    ),
                    k=config.evaluation.precision_at_k,
                )
        if baselines.enabled:
            from tfm.data.native_reference import ReferenceProfiles, ReferenceRows
            from tfm.evaluation.tabular import evaluate_tabular_baselines

            references = ReferenceRows(inputs)
            profiles = ReferenceProfiles(references, arrays["test"]["entities"].tolist())
            manifest["tabular_baselines"] = {}
            for name, (left, right) in tasks.items():
                if name == "single_event":
                    continue
                kwargs: dict[str, Any] = dict(
                    profiles=profiles,
                    seed=config.seed,
                    top_k=baselines.top_k,
                    coded_keys=config.features.categorical_keys,
                    exclude_keys=baselines.exclude_keys,
                    n_jobs=baselines.n_jobs,
                    device=baselines.device,
                    k=config.evaluation.precision_at_k,
                    prefix="" if name == "fraud" else name,
                    train_embeddings=left["vectors"],
                    eval_embeddings=right["vectors"],
                )
                if name == "history_window" and "single_event" in tasks:
                    kwargs["embedding_views"] = {
                        "single_event": (
                            tasks["single_event"][0]["vectors"],
                            tasks["single_event"][1]["vectors"],
                        )
                    }
                args = (
                    references.rows(left["samples"].tolist()),
                    left["labels"],
                    references.rows(right["samples"].tolist()),
                    right["labels"],
                )
                try:
                    scores, detail = evaluate_tabular_baselines(
                        *args,
                        xgboost_trials=baselines.xgboost_trials,
                        xgboost_params=baselines.xgboost_params,
                        **kwargs,
                    )
                except InsufficientLabelSupportError as error:
                    scores, detail = evaluate_tabular_baselines(*args, xgboost_trials=0, **kwargs)
                    detail["xgboost_skipped"] = str(error)
                metrics.update(scores)
                manifest["tabular_baselines"][name] = detail
        write_json(out / "probe_manifest.json", manifest)
        write_json(out / "finetune_manifest.json", {"enabled": False})
        write_json(out / "metrics.json", metrics)
        (out / "metrics.md").write_text(metrics_markdown(metrics))
        (out / "model_card.md").write_text(
            "# Source-native sequence reference example\n\nExperimental native retrieval and fitted tensor encoding.\n"
        )
        return out
