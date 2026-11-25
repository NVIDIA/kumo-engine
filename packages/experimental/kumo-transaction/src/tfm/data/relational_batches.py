# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aligned sequence/graph views of one adapter declaration and anchor stream."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from dataclasses import replace
from datetime import timedelta
from typing import Literal

import torch

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.graph import (
    GraphBatch,
    GraphHistoryBatch,
    GraphSupervisedBatch,
    SequencePredictionBatch,
    SequenceSupervisedBatch,
    utc_us,
)
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.schema import TransactionEvent, TransactionSequence
from tfm.contracts.tokenizers import StatefulSequenceTokenizer, UsesProfiles
from tfm.data.collate import collate_batch
from tfm.data.graph import GraphSampler
from tfm.data.native_history import Loader, NativeHistories
from tfm.data.relational import RelationalAnchor, RelationalSource

RelationalBatch = (
    GraphBatch
    | GraphHistoryBatch
    | GraphSupervisedBatch
    | SequenceSupervisedBatch
    | SequencePredictionBatch
)


def retrieve_histories(
    source: RelationalSource,
    anchors: Sequence[RelationalAnchor],
    *,
    loader: Loader,
    max_events: int,
) -> tuple[TransactionSequence, ...]:
    """Project complete visible paths, then retrieve independent anchor histories.

    Native indexing is bounded and duplicates projected occurrences per anchor;
    it never flattens ownership or changes terminal timestamps. Path availability
    is certified by the resolver before indexing. No fitted encoding is involved.
    """
    if max_events < 1 or loader not in {"python", "kumo_pytorch"}:
        raise ValueError("Choose a known loader and a positive history limit")
    if loader == "python":
        return tuple(source.history(item.anchor, max_events=max_events) for item in anchors)
    windows = tuple(
        TransactionSequence(
            item.anchor.entity_id,
            tuple(record.event for record in source.resolve_history(item.anchor)),
            item.anchor.cutoff,
            observation_window=None,
            sequence_id=item.anchor.sample_id,
        )
        for item in anchors
    )
    selected: dict[int, tuple[TransactionEvent, ...]] = {i: () for i in range(len(windows))}
    nonempty = [i for i, window in enumerate(windows) if window.events]
    if nonempty:
        with NativeHistories((), windows=windows) as native:
            task = native.bind(
                [(str(i), windows[i].cutoff_time) for i in nonempty], ["inputs"] * len(nonempty)
            )
            seen = set()
            for batch in task.batches(
                split="inputs", batch_size=128, shuffle=False, output_format="model"
            ):
                for position, events in native.decode(batch).items():
                    if position in seen:
                        raise SchemaError("Native relational histories repeated an anchor")
                    seen.add(position)
                    selected[nonempty[position]] = events[-max_events:]
            if seen != set(range(len(nonempty))):
                raise SchemaError("Native relational histories dropped an anchor")
    return tuple(replace(window, events=selected[i]) for i, window in enumerate(windows))


class RelationalBatches:
    """Reusable inputs with graph allocation only for graph-consuming modes.

    Fit the supplied tokenizer on training histories before requesting history
    batches. Closing releases native graph snapshots; no GPU tensors are cached.
    """

    def __init__(
        self,
        source: RelationalSource,
        *,
        loader: Loader = "python",
        depth: int = 6,
        fanout: int = 10,
        max_events: int = 256,
        observation_window: timedelta | None = None,
    ) -> None:
        if loader not in {"python", "kumo_pytorch"}:
            raise ValueError("Unknown relational loader")
        self.source = source
        self.loader: Loader = loader
        self.depth, self.fanout, self.max_events = depth, fanout, max_events
        self.anchors = source.anchors()
        if observation_window is not None:
            if observation_window.total_seconds() <= 0:
                raise ValueError("observation_window must be positive")
            self.anchors = tuple(
                replace(
                    item,
                    anchor=replace(
                        item.anchor,
                        observation_start=max(
                            item.anchor.cutoff - observation_window,
                            item.anchor.observation_start
                            or item.anchor.cutoff - observation_window,
                        ),
                    ),
                )
                for item in self.anchors
            )
        self.sampler: GraphSampler | None = None
        self.resources = ExitStack()
        self.closed = False

    def __enter__(self) -> RelationalBatches:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.resources.close()
        self.closed = True

    def graph_sampler(self) -> GraphSampler:
        """Validate graph projection before creating any native resources."""
        if self.closed:
            raise ValueError("Relational batches are closed")
        if self.sampler is None:
            graph = self.source.graph()
            if self.loader == "python":
                self.sampler = GraphSampler(graph, depth=self.depth, fanout=self.fanout)
            else:
                from tfm.data.native_relational_graph import NativeRelationalGraphSampler

                native = NativeRelationalGraphSampler(graph, depth=self.depth, fanout=self.fanout)
                self.resources.callback(native.close)
                self.sampler = native
        return self.sampler

    def histories(self, split: str) -> tuple[TransactionSequence, ...]:
        """One ordered history per selected anchor; empty histories remain explicit."""
        if self.closed:
            raise ValueError("Relational batches are closed")
        return retrieve_histories(
            self.source,
            [item for item in self.anchors if item.split == split],
            loader=self.loader,
            max_events=self.max_events,
        )

    @staticmethod
    def _chunks(
        items: Sequence[RelationalAnchor], size: int, shared: bool
    ) -> Iterator[tuple[RelationalAnchor, ...]]:
        if size < 1:
            raise ValueError("batch_size must be positive")
        chunk: list[RelationalAnchor] = []
        for item in items:
            if chunk and (
                len(chunk) == size
                or item.label_cutoff != chunk[0].label_cutoff
                or shared
                and (item.anchor.cutoff, item.anchor.observation_start)
                != (chunk[0].anchor.cutoff, chunk[0].anchor.observation_start)
            ):
                yield tuple(chunk)
                chunk = []
            chunk.append(item)
        if chunk:
            yield tuple(chunk)

    @torch.inference_mode(False)
    @torch.no_grad()
    def batches(
        self,
        split: str,
        *,
        mode: Literal["sequence", "graph", "hybrid"],
        batch_size: int,
        tokenizer: StatefulSequenceTokenizer | None = None,
        max_tokens: int = 256,
        amount_key: str = "amount",
        sharing: Literal["disjoint", "shared"] = "disjoint",
        hop_trim: bool = True,
        require_labels: bool = True,
    ) -> list[RelationalBatch]:
        """Build guarded CPU batches; labels retain their explicit entity level."""
        if self.closed or mode not in {"sequence", "graph", "hybrid"}:
            raise ValueError("Choose an open relational source and a known mode")
        items = tuple(item for item in self.anchors if item.split == split)
        if not items:
            raise SchemaError(f"Relational task has no {split!r} anchors")
        # Validate the entire requested split before constructing any inputs.
        if require_labels and any(item.target is None for item in items):
            raise SchemaError(f"Relational split {split}: missing available entity labels")
        if mode != "graph" and tokenizer is not None:
            profiles = self.source.profiles([item.anchor for item in items])
            if isinstance(tokenizer, UsesProfiles):
                tokenizer.set_profiles(profiles)
            elif any(profile.attributes for profile in profiles.values()):
                raise UnsupportedCombinationError(
                    f"History path {self.source.task.root}: tokenizer cannot consume root profile allowlists"
                )
        sampler = None if mode == "sequence" else self.graph_sampler()
        histories = (
            {}
            if mode == "graph"
            else {history.sequence_id: history for history in self.histories(split)}
        )
        result: list[RelationalBatch] = []
        for chunk in self._chunks(items, batch_size, sharing == "shared" and mode != "sequence"):
            anchors = tuple(item.anchor for item in chunk)
            labelled = all(item.target is not None for item in chunk)
            labels = (
                torch.tensor([item.target for item in chunk], dtype=torch.float32)
                if labelled
                else None
            )
            available = (
                torch.tensor(
                    [
                        utc_us(item.label_available_at)
                        for item in chunk
                        if item.label_available_at is not None
                    ],
                    dtype=torch.long,
                )
                if labelled
                else None
            )
            cutoff = (
                utc_us(chunk[0].label_cutoff)
                if labelled and chunk[0].label_cutoff is not None
                else None
            )
            graph = (
                None
                if sampler is None
                else sampler.sample(
                    anchors, self.source.task.root, sharing=sharing, hop_trim=hop_trim
                )
            )
            if mode == "graph":
                assert graph is not None
                if labels is None:
                    result.append(graph)
                else:
                    assert available is not None and cutoff is not None
                    result.append(
                        GraphSupervisedBatch(
                            graph, labels, available, cutoff, target_level="entity"
                        )
                    )
                continue
            if tokenizer is None:
                raise ValueError("Sequence/hybrid batches require a fitted stateful tokenizer")
            history = collate_batch(
                [histories[anchor.sample_id] for anchor in anchors],
                tokenizer,
                max_tokens=max_tokens,
                amount_key=amount_key,
            )
            if not isinstance(history, KeyValueTimeBatch):
                raise ValueError("Relational histories currently require key_value_time layout")
            history = replace(history, fraud_labels=None)
            if mode == "hybrid":
                assert graph is not None
                result.append(
                    GraphHistoryBatch(
                        graph, history, anchors, labels, available, cutoff, target_level="entity"
                    )
                )
            elif labels is None:
                result.append(SequencePredictionBatch(history, anchors, target_level="entity"))
            else:
                assert available is not None and cutoff is not None
                result.append(
                    SequenceSupervisedBatch(
                        history, anchors, labels, available, cutoff, target_level="entity"
                    )
                )
        return result
