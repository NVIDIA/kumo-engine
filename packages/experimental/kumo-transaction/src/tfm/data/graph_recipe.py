# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared relational recipe data: explicit owners, clocks, codecs, and targets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta

import numpy as np
import torch

from tfm.config import RecipeConfig
from tfm.contracts.errors import SchemaError
from tfm.contracts.graph import (
    FutureSubgraphTargets,
    GraphAnchor,
    GraphBatch,
    GraphHistoryBatch,
    GraphSupervisedBatch,
    GraphTrainingBatch,
    SequenceSupervisedBatch,
    utc_us,
)
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.schema import TransactionEvent, TransactionSequence, TransferEdge
from tfm.contracts.tokenizers import StatefulSequenceTokenizer
from tfm.data.adapters.registry import load_adapter_events
from tfm.data.collate import collate_batch, sequence_label
from tfm.data.graph import FutureTargetTransform, GraphSampler, future_subgraph_targets
from tfm.data.graph_transactions import transaction_graph
from tfm.data.labels.horizon import available_label_value
from tfm.data.native_graph import NativeGraphSampler
from tfm.data.native_history import build_entity_sequences
from tfm.data.validation.schema import validate_events
from tfm.features.tokenizers.registry import tokenizer_for

DAY_US = 86_400_000_000


@dataclass
class PreparedGraphData:
    """Fitted training-only transforms with auditable sample mappings."""

    sampler: GraphSampler
    tokenizer: StatefulSequenceTokenizer | None
    transform: FutureTargetTransform | None
    pretrain: list[GraphTrainingBatch | GraphSupervisedBatch]
    validation: list[GraphTrainingBatch | GraphSupervisedBatch]
    hybrid_train: list[GraphHistoryBatch]
    probes: dict[str, list[GraphHistoryBatch]]
    sequence_pretrain: list[KeyValueTimeBatch] | list[SequenceSupervisedBatch]
    sequence_validation: list[KeyValueTimeBatch] | list[SequenceSupervisedBatch]
    sequence_supervised_train: list[SequenceSupervisedBatch]
    sequence_supervised_validation: list[SequenceSupervisedBatch]
    fingerprint: str
    split_manifest: dict[str, object]


def _timestamp(text: str) -> datetime:
    time = datetime.fromisoformat(text.replace("Z", "+00:00"))
    utc_us(time)
    return time


def _events(config: RecipeConfig) -> tuple[list[TransactionEvent], tuple[TransferEdge, ...]]:
    requires_labels = config.runtime.stage != "pretrain" or config.training.objective == "binary"
    loaded = load_adapter_events(config, flat_fields=False)
    # Task labels are not read by any pretraining-only feature/target builder,
    # including when a user adapter emits them regardless of load_labels.
    events = (
        list(loaded.events)
        if requires_labels
        else [replace(event, labels={}) for event in loaded.events]
    )
    return validate_events(events), loaded.transfers


def _require_training_amount(
    events: list[TransactionEvent], owners: list[str], cutoff: datetime, field: str
) -> None:
    training_owners = set(owners)
    if not any(
        event.entity_id in training_owners
        and event.event_timestamp <= cutoff
        and field in event.numeric
        and math.isfinite(event.numeric[field])
        for event in events
    ):
        raise SchemaError(
            f"No train-side event by {cutoff.isoformat()} has finite numeric amount_field {field!r}; "
            "set data.amount_field to the source's observed money field."
        )


@torch.inference_mode(False)
@torch.no_grad()
def prepare_graph_data(
    config: RecipeConfig,
    *,
    tokenizer: StatefulSequenceTokenizer | None = None,
    transform: FutureTargetTransform | None = None,
) -> PreparedGraphData:
    """Prepare explicit task-supervised or future-pretraining graph stages.

    Supervision uses existing available sequence labels at the probe cutoff.
    That mode never constructs future labels or requires historical SSL windows.
    FutureSubgraph fits targets/codecs only on the historical training corpus.
    """
    events, transfers = _events(config)
    data, g = config.data, config.model.composition.granularity
    objective = config.training.objective if config.runtime.stage != "downstream" else "binary"
    needs_supervision = config.runtime.stage != "pretrain" or objective == "binary"
    needs_sequences = config.runtime.stage != "pretrain"
    probe_cutoff, coverage = _timestamp(data.probe_cutoff), _timestamp(data.observed_through)
    if not events or (needs_supervision and probe_cutoff > coverage):
        raise ValueError("Probe cutoff/source coverage is outside the supplied observations")
    sampler_type = NativeGraphSampler if data.loader == "kumo_pytorch" else GraphSampler
    sampler = sampler_type(
        transaction_graph(
            events, transfers, observed_through=coverage, amount_field=data.amount_field
        ),
        depth=g.depth,
        fanout=g.fanout,
    )
    owners = sorted({e.entity_id for e in events})
    if len(owners) < 12:
        raise ValueError("Graph recipe needs at least 12 observed owners from the selected source")
    owners = [
        owners[i] for i in np.random.default_rng(config.seed).permutation(len(owners)).tolist()
    ]
    n_train, n_validation = int(0.6 * len(owners)), int(0.2 * len(owners))
    split = {
        "train": owners[:n_train],
        "validation": owners[n_train : n_train + n_validation],
        "test": owners[n_train + n_validation :],
    }
    if min(map(len, split.values())) < 2:
        raise ValueError("Graph recipe needs at least two owners per probe split")

    def anchors_at(ids: list[str], cutoff: datetime) -> tuple[GraphAnchor, ...]:
        return tuple(GraphAnchor(f"{owner}:{cutoff.isoformat()}", owner, cutoff) for owner in ids)

    def sample(anchors: tuple[GraphAnchor, ...], *, training: bool = False) -> GraphBatch:
        return sampler.sample(
            anchors,
            "users",
            sharing=data.training_graphs if training else "disjoint",
            hop_trim=g.trim,
        )

    raw: list[FutureSubgraphTargets] = []
    raw_val: FutureSubgraphTargets | None = None
    fitted: FutureTargetTransform | None = None
    ssl_train: list[str] = []
    ssl_val: list[str] = []
    boundary: datetime | None = None
    codec_cutoff, codec_owners = probe_cutoff, split["train"]
    if objective == "future_subgraph":
        boundary = _timestamp(data.pretrain_end)
        horizon, delay = timedelta(days=data.horizon_days), timedelta(days=data.label_delay_days)
        if needs_supervision and probe_cutoff <= boundary:
            raise ValueError("Future pretraining requires a later probe cutoff")
        ssl_n = max(2, len(split["train"]) // 4)
        ssl_val, ssl_train = split["train"][:ssl_n], split["train"][ssl_n:]
        codec_cutoff, codec_owners = boundary - delay - 4 * horizon, ssl_train
        _require_training_amount(
            events, ssl_train, boundary - delay - 2 * horizon, data.amount_field
        )
        for k in (4, 3, 2):
            raw.append(
                future_subgraph_targets(
                    sampler,
                    anchors_at(ssl_train, boundary - delay - k * horizon),
                    "users",
                    horizon_us=data.horizon_days * DAY_US,
                    depth=g.target_depth,
                    partition_end_us=utc_us(boundary),
                    label_delay_us=data.label_delay_days * DAY_US,
                )
            )
        all_train = replace(
            raw[0],
            anchors=tuple(a for b in raw for a in b.anchors),
            values=torch.cat([b.values for b in raw]),
            valid=torch.cat([b.valid for b in raw]),
            available_us=torch.cat([b.available_us for b in raw]),
        )
        fitted = FutureTargetTransform.fit(all_train) if transform is None else transform
        raw_val = future_subgraph_targets(
            sampler,
            anchors_at(ssl_val, boundary - delay - horizon),
            "users",
            horizon_us=data.horizon_days * DAY_US,
            depth=g.target_depth,
            partition_end_us=utc_us(boundary),
            label_delay_us=data.label_delay_days * DAY_US,
        )
    else:
        if transform is not None:
            raise ValueError("Supervised graph training must not receive future-target statistics")
        _require_training_amount(events, split["train"], probe_cutoff, data.amount_field)

    if not needs_sequences:
        tokenizer = None
    else:
        codec_owner_set = set(codec_owners)
        fit_sequences = [
            s
            for s in build_entity_sequences(
                events, codec_cutoff, max_events=data.sequence_length, loader=data.loader
            )
            if s.entity_id in codec_owner_set
        ]
        if len(fit_sequences) != len(codec_owners):
            raise ValueError("Training owners need observed history at the codec-fit cutoff")
        if tokenizer is None:
            candidate = tokenizer_for(config)
            if not isinstance(candidate, StatefulSequenceTokenizer):
                raise ValueError("Graph histories require a registered stateful sequence tokenizer")
            if candidate.layout != "key_value_time":
                raise ValueError("Graph histories require key_value_time tokenization")
            tokenizer = candidate
            tokenizer.fit(fit_sequences)
    sequences = (
        {}
        if not needs_supervision
        else {
            s.entity_id: s
            for s in build_entity_sequences(
                events, probe_cutoff, max_events=data.sequence_length, loader=data.loader
            )
        }
    )
    # Certify a label source for each observed owner, independently of history
    # truncation. An app-only retained history still has the existing zero label.
    covered_owners = (
        set()
        if not needs_supervision
        else {
            e.entity_id
            for e in events
            if e.event_timestamp <= probe_cutoff
            and available_label_value(e, "fraud", probe_cutoff) is not None
        }
    )

    def labelled_history(anchor: GraphAnchor) -> TransactionSequence:
        if anchor.entity_id not in sequences:
            raise ValueError("Probe root has no history at the probe cutoff")
        if anchor.entity_id not in covered_owners:
            raise ValueError("Missing available fraud label coverage for supervised owner")
        return replace(sequences[anchor.entity_id], sequence_id=anchor.sample_id)

    probes: dict[str, list[GraphHistoryBatch]] = {}
    supervised_splits = split if needs_sequences else {}
    for side, ids in supervised_splits.items():
        batches = []
        for start in range(0, len(ids), config.sequence_training.batch_size):
            anchors = anchors_at(
                ids[start : start + config.sequence_training.batch_size], probe_cutoff
            )
            histories = [labelled_history(anchor) for anchor in anchors]
            assert tokenizer is not None
            batch = collate_batch(
                histories, tokenizer, max_tokens=data.max_tokens, amount_key=data.amount_field
            )
            if not isinstance(batch, KeyValueTimeBatch):
                raise ValueError("Graph history tokenizer must produce KeyValueTimeBatch")
            labels = batch.fraud_labels
            batch = replace(batch, fraud_labels=None)
            batches.append(
                GraphHistoryBatch(
                    sample(anchors),
                    batch,
                    anchors,
                    labels,
                    torch.full((len(anchors),), utc_us(probe_cutoff), dtype=torch.long),
                    utc_us(probe_cutoff),
                )
            )
        probes[side] = batches
    hybrid_train = [
        replace(b, graph=sample(b.graph.anchors, training=True)) for b in probes.get("train", [])
    ]

    def supervised_batches(side: str, *, training: bool) -> list[GraphSupervisedBatch]:
        anchors = anchors_at(split[side], probe_cutoff)
        labels = torch.tensor(
            [sequence_label(labelled_history(a)) for a in anchors], dtype=torch.float32
        )
        available = torch.full((len(anchors),), utc_us(probe_cutoff), dtype=torch.long)
        result = []
        for start in range(0, len(anchors), config.training.batch_size):
            stop = start + config.training.batch_size
            result.append(
                GraphSupervisedBatch(
                    sample(anchors[start:stop], training=training),
                    labels[start:stop],
                    available[start:stop],
                    utc_us(probe_cutoff),
                )
            )
        return result

    def ssl_batches(targets: FutureSubgraphTargets, *, training: bool) -> list[GraphTrainingBatch]:
        assert fitted is not None
        normalized = fitted.apply(targets)
        result = []
        for start in range(0, len(targets.anchors), config.training.batch_size):
            stop = start + config.training.batch_size
            labels = replace(
                normalized,
                anchors=normalized.anchors[start:stop],
                values=normalized.values[start:stop],
                valid=normalized.valid[start:stop],
                available_us=normalized.available_us[start:stop],
            )
            result.append(GraphTrainingBatch(sample(labels.anchors, training=training), labels))
        return result

    def sequence_batches(targets: list[FutureSubgraphTargets]) -> list[KeyValueTimeBatch]:
        assert tokenizer is not None
        result = []
        for target in targets:
            histories = {
                s.entity_id: s
                for s in build_entity_sequences(
                    events,
                    target.anchors[0].cutoff,
                    max_events=data.sequence_length,
                    loader=data.loader,
                )
            }
            for start in range(0, len(target.anchors), config.sequence_training.batch_size):
                selected = [
                    replace(histories[a.entity_id], sequence_id=a.sample_id)
                    for a in target.anchors[start : start + config.sequence_training.batch_size]
                ]
                batch = tokenizer.collate(selected, max_length=data.max_tokens)
                if not isinstance(batch, KeyValueTimeBatch):
                    raise ValueError("Graph histories require key_value_time batches")
                result.append(batch)
        return result

    graph_train: list[GraphTrainingBatch | GraphSupervisedBatch]
    graph_validation: list[GraphTrainingBatch | GraphSupervisedBatch]
    baseline_train: list[KeyValueTimeBatch] | list[SequenceSupervisedBatch]
    baseline_validation: list[KeyValueTimeBatch] | list[SequenceSupervisedBatch]
    matched_train = [SequenceSupervisedBatch.from_graph_history(b) for b in hybrid_train]
    matched_validation = [
        SequenceSupervisedBatch.from_graph_history(b) for b in probes.get("validation", [])
    ]
    if objective == "future_subgraph":
        assert raw_val is not None
        graph_train = list(b for targets in raw for b in ssl_batches(targets, training=True))
        graph_validation = list(ssl_batches(raw_val, training=False))
        baseline_train, baseline_validation = (
            (sequence_batches(raw), sequence_batches([raw_val]))
            if config.runtime.stage == "full"
            else ([], [])
        )
    else:
        graph_train = list(supervised_batches("train", training=True))
        graph_validation = list(supervised_batches("validation", training=False))
        baseline_train, baseline_validation = matched_train, matched_validation

    fingerprint = hashlib.sha256(
        json.dumps(
            {"events": [asdict(e) for e in events], "transfers": [asdict(e) for e in transfers]},
            default=str,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest: dict[str, object] = {
        "dataset": data.adapter,
        "input_loader": sampler.loader_manifest
        if isinstance(sampler, NativeGraphSampler)
        else {"name": "python"},
        "data_sha256": fingerprint,
        "probe_owners": split,
        "graph_objective": objective,
        "execution_stage": config.runtime.stage,
        "requires_task_labels": needs_supervision,
        "probe_task": "any_available_fraud_in_truncated_history",
        "probe_cutoff": probe_cutoff.isoformat(),
        "ssl_train_owners": ssl_train,
        "ssl_validation_owners": ssl_val,
        "ssl_train_samples": [a.as_dict() for b in raw for a in b.anchors],
        "ssl_validation_samples": [] if raw_val is None else [a.as_dict() for a in raw_val.anchors],
        "pretrain_end": None if boundary is None else boundary.isoformat(),
        "codec_fit_cutoff": codec_cutoff.isoformat() if needs_sequences else None,
        "source_observed_through": coverage.isoformat(),
        "horizon_us": data.horizon_days * DAY_US if objective == "future_subgraph" else None,
        "label_delay_us": data.label_delay_days * DAY_US
        if objective == "future_subgraph"
        else None,
        "graph_policy": "cutoff-bounded induced graph over sampled owners; cross-owner context allowed",
        "training_graphs": data.training_graphs,
        "validation_graphs": "disjoint",
        "prediction_graphs": "disjoint",
        "sampling": {
            "depth": g.depth,
            "fanout": g.fanout,
            "target_depth": g.target_depth if objective == "future_subgraph" else None,
            "hop_trimming": g.trim,
            "ties": "native index-local recent ties"
            if isinstance(sampler, NativeGraphSampler)
            else "canonical PK ascending",
            "event_key_encoding": "json-array-v1",
        },
    }
    manifest = json.loads(json.dumps(manifest, default=str))
    return PreparedGraphData(
        sampler,
        tokenizer,
        fitted,
        graph_train,
        graph_validation,
        hybrid_train,
        probes,
        baseline_train,
        baseline_validation,
        matched_train,
        matched_validation,
        fingerprint,
        manifest,
    )
