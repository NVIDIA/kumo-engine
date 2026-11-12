# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A materialized task: a prepared plan bound to its anchors and fitted state.

`prepare` produces a `Plan`: the contract, bound to a dataset through an engine,
with nothing that depends on which examples will be drawn. `Plan.materialize`
binds anchors and produces a `Task` — the unit that is iterated, served from,
and written as an artifact. It owns everything that depends on the anchors:
their canonical order, the state fitted on them, and, in an engine that
materializes eagerly, labels and anchor-to-graph edges.

A task carries its identity. Its fingerprints — contract, inputs, dataset,
anchors, fitted state, output schema — are what a later scheduler composing
several tasks, or a resume, checks against.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import replace as dataclass_replace
from typing import Any, Literal

import torch

from kumo.pytorch.anchors import (
    ALL_ANCHORS,
    PADDING_KEY,
    AllAnchors,
    RequestAnchors,
    SplitSelection,
    describe_selection,
    padded_positions,
    split_selection,
)
from kumo.pytorch.artifact import Artifact, FitRecord, FittedState
from kumo.pytorch.assembly import PreparedBatchPlan, PreviewResult, _fingerprint, preview_plan
from kumo.pytorch.contract.batch import Batch
from kumo.pytorch.contract.sample import SampleRequest
from kumo.pytorch.engine import AnchorTask
from kumo.pytorch.spec import DataLoaderSpec

__all__ = ["Task", "TaskBatches"]


class TaskBatches(Iterator[Batch]):
    """Batches from one pass over a task, resumable from a saved position.

    ``state_dict`` records how many batches have been consumed under this exact
    configuration; ``load_state_dict`` on a fresh iterator with the same
    configuration continues after them. Resuming replays the skipped batches
    rather than seeking, which is correct but slow for a long pass.
    """

    def __init__(
        self,
        produce: Callable[[], Iterator[Batch]],
        *,
        task_fingerprint: str,
        config: Mapping[str, Any],
    ) -> None:
        self._produce = produce
        self._task_fingerprint = task_fingerprint
        self._config = dict(config)
        self._iterator: Iterator[Batch] | None = None
        self._consumed = 0
        self._skip = 0

    def __iter__(self) -> TaskBatches:
        return self

    def __next__(self) -> Batch:
        if self._iterator is None:
            self._iterator = self._produce()
            for _ in range(self._skip):
                next(self._iterator)
        batch = next(self._iterator)
        self._consumed += 1
        return batch

    def state_dict(self) -> dict[str, Any]:
        """The resume position: the task fingerprint, batch configuration and batches consumed."""
        return {
            "version": 1,
            "task": self._task_fingerprint,
            "config": dict(self._config),
            "batches": self._consumed,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Continue after the batches ``state`` records, before iteration starts.

        :raises ValueError: when iteration has started, or the state belongs to
            another task, split, seed, epoch, rank or batch configuration.
        """
        if self._iterator is not None:
            raise ValueError("load_state_dict must be called before iteration starts")
        if state.get("version") != 1:
            raise ValueError(f"unsupported batch state version {state.get('version')!r}")
        if state.get("task") != self._task_fingerprint:
            raise ValueError("this state was saved for a different task")
        if dict(state.get("config", {})) != self._config:
            raise ValueError(
                "this state was saved under a different batch configuration "
                "(split, seed, epoch, shuffle, rank, world size or batch size)"
            )
        self._skip = self._consumed = int(state["batches"])


class Task:
    """Anchors bound to a plan: what is iterated, served from, and written out.

    Produced by `Plan.materialize`. Reads through the plan's open binding, so it
    is usable while that plan is open.

    Attributes: ``fitted_state`` (a `FittedState`), ``output_schema``, and
    ``fingerprints``, covering the contract, inputs, dataset, anchors, input
    state, target state, output schema and the task as a whole.
    """

    def __init__(
        self,
        *,
        contract: DataLoaderSpec,
        assembler: PreparedBatchPlan,
        anchors: AnchorTask,
        dataset_fingerprint: str,
        anchors_fingerprint: str,
        fit: Mapping[str, FitRecord | None],
        has_targets: bool = True,
    ) -> None:
        self.contract = contract
        self._assembler = assembler
        self._anchors = anchors
        serialized = assembler.fitted_state
        manifest = assembler.manifest
        self.fitted_state = FittedState(
            input_fingerprint=contract.input_fingerprint,
            contract_fingerprint=contract.fingerprint,
            _input_state=assembler._fitted_input_state(),
            _target_encoders=dict(assembler._target_encoders) if has_targets else None,
            _serialized=serialized,
            _fit=dict(fit),
        )
        self.output_schema: Mapping[str, Any] = {
            "inputs": {name: value["summary"] for name, value in manifest["inputs"].items()},
            "targets": {} if assembler.inference else dict(manifest["targets"]),
        }
        fingerprints = {
            "contract": contract.fingerprint,
            "inputs": contract.input_fingerprint,
            "dataset": dataset_fingerprint,
            "anchors": anchors_fingerprint,
            "input_state": _fingerprint(
                {key: value for key, value in serialized.items() if key != "targets"}
            ),
            "target_state": _fingerprint(serialized.get("targets", {})),
            "output_schema": _fingerprint(self.output_schema),
        }
        fingerprints["task"] = _fingerprint(fingerprints)
        self.fingerprints: Mapping[str, str] = fingerprints

    @property
    def manifest(self) -> Mapping[str, Any]:
        """Summaries and fingerprints of the fitted state, as ``manifest.json`` records them."""
        return self._assembler.manifest

    @property
    def encoders(self) -> Mapping[str, Any]:
        """Named fitted transformations, before any final ID-space translation."""
        return self._assembler.encoders

    @property
    def spaces(self) -> Mapping[str, Any]:
        """Sealed final ID dictionaries, including their declared special roles."""
        return self._assembler.spaces

    def output(self, name: str) -> Any:
        """Inspect an input's field encoders and complete output conversions."""
        return self._assembler.output(name=name)

    def target(self, name: str) -> Any:
        """Inspect the fitted encodings of an independently encoded target."""
        return self._assembler.target(name=name)

    @property
    def inference(self) -> bool:
        """Whether batches are inference batches: no targets, inputs unmasked."""

        return self._assembler.inference

    def count(self, split: str | Collection[str | None] | AllAnchors | None = None) -> int:
        """Anchors in ``split``: a name, a set of names, or every anchor when omitted."""

        return self._anchors.count(_selection(split))

    def batches(
        self,
        requests: Sequence[SampleRequest] | None = None,
        *,
        split: str | Collection[str | None] | AllAnchors | None = None,
        batch_size: int | None = None,
        epoch: int = 0,
        seed: int = 0,
        shuffle: bool = False,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool | None = None,
        chunk_batches: int = 8,
        output_format: Literal["inspectable", "model"] = "inspectable",
    ) -> TaskBatches:
        """Emit batches over this task's anchors, or over requests built in code.

        Without ``requests``, iterates the materialized anchors of ``split`` — a
        name, a set of names (``None`` selects anchors with no split), or every
        anchor when omitted — reading them a chunk at a time
        so memory follows batch size rather than anchor count. With
        ``requests``, serves exactly those — the path a request handler or a
        preview has — through the same fitted state.

        Every rank gets the same number of anchors: a short rank is topped up by
        repeating one of its own anchors, and ``Batch.metadata["is_padding"]``
        marks those rows so a loss can exclude them. ``drop_last`` applies to
        the final batch of each rank's pass, not to every chunk.

        Order is a pure function of ``seed``, ``epoch``, ``shuffle``, ``rank``
        and ``world_size``; ranks are disjoint before padding. A token budget
        (`Batching.max_tokens`) with ``world_size > 1`` is refused.
        ``output_format="model"`` refuses raw types without a tensor
        representation before fetching a batch; ``"inspectable"`` keeps them
        as Arrow arrays. Both emit the same `Batch` structure.
        """

        if requests is not None and split is not None:
            raise ValueError("split selects materialized anchors; requests carry their own")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must identify one member of a positive world size")
        if world_size > 1 and self.contract.batch.batching.max_tokens is not None:
            raise ValueError(
                "Batching.max_tokens with world_size > 1 is not supported yet: padding "
                "equalizes anchors per rank, but a token budget makes the number of "
                "batches depend on lengths, so ranks can take different numbers of steps "
                "and hang synchronous data-parallel training. Use max_samples only for "
                "distributed runs until batches are planned globally and assigned whole "
                "to ranks."
            )
        selection = _selection(split)
        if drop_last is None and self.inference:
            drop_last = False  # inference scores every anchor, whatever the training policy
        if chunk_batches <= 0:
            raise ValueError("chunk_batches must be positive")
        config = {
            "split": describe_selection(selection) if requests is None else None,
            "requests": None if requests is None else _fingerprint(tuple(requests)),
            "batch_size": batch_size,
            "epoch": epoch,
            "seed": seed,
            "shuffle": shuffle,
            "rank": rank,
            "world_size": world_size,
            "drop_last": drop_last,
            "chunk_batches": chunk_batches,
            "output_format": output_format,
        }
        if output_format not in {"inspectable", "model"}:
            raise ValueError("output_format must be 'inspectable' or 'model'")
        if requests is None:
            self._anchors.count(selection)  # an unknown split is refused now, not mid-iteration
            max_samples = (
                batch_size if batch_size is not None else self.contract.batch.batching.max_samples
            )
            if max_samples is None:
                raise ValueError(
                    "reading materialized anchors requires a batch size: pass batch_size "
                    "or set Batching.max_samples"
                )

            def produce() -> Iterator[Batch]:
                return self._anchor_batches(
                    selection,
                    chunk=max_samples * chunk_batches,
                    batch_size=batch_size,
                    epoch=epoch,
                    seed=seed,
                    shuffle=shuffle,
                    rank=rank,
                    world_size=world_size,
                    drop_last=drop_last,
                    output_format=output_format,
                )

        else:
            request_list = tuple(requests)

            def produce() -> Iterator[Batch]:
                return request_batches(
                    self._assembler,
                    request_list,
                    batch_size=batch_size,
                    epoch=epoch,
                    seed=seed,
                    shuffle=shuffle,
                    rank=rank,
                    world_size=world_size,
                    drop_last=drop_last,
                    output_format=output_format,
                )

        return TaskBatches(produce, task_fingerprint=self.fingerprints["task"], config=config)

    def _anchor_batches(
        self,
        selection: SplitSelection,
        *,
        chunk: int,
        batch_size: int | None,
        epoch: int,
        seed: int,
        shuffle: bool,
        rank: int,
        world_size: int,
        drop_last: bool | None,
        output_format: Literal["inspectable", "model"] = "inspectable",
    ) -> Iterator[Batch]:
        positions = padded_positions(
            self._anchors.order(
                selection,
                seed=seed,
                epoch=epoch,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size,
            ),
            count=self._anchors.count(selection),
            world_size=world_size,
        )
        pending = _chunks(positions, chunk)
        current = next(pending, None)
        while current is not None:
            following = next(pending, None)
            requests = _mark_padding(
                self._anchors.fetch(selection, [position for position, _ in current]),
                [padding for _, padding in current],
            )
            for batch in self._assembler.batches(
                requests,
                batch_size=batch_size,
                epoch=epoch,
                seed=seed,
                drop_last=drop_last if following is None else False,
                output_format=output_format,
            ):
                yield _surface_padding(batch)
            current = following

    def preview(self, requests: Sequence[SampleRequest], *, seed: int = 0) -> PreviewResult:
        """Raw samples and one encoded batch for up to eight requests."""

        if not requests:
            raise ValueError("preview requires at least one explicit request")
        if len(requests) > 8:
            raise ValueError("preview is limited to 8 requests; provide a smaller selection")
        return preview_plan(self._assembler, requests, seed=seed)

    def bound_contract(self) -> DataLoaderSpec:
        """This contract with every fitted value pinned into the declarations."""

        return self._assembler.bound_contract()

    def to_artifact(self) -> Artifact:
        """This task's contract, fitted state and identity, reading and writing nothing.

        Equal to what `Artifact.load` reads back after `Artifact.save` writes it.
        Refused for an inference task given only ``state.inputs``: an artifact holds
        complete fitted state.
        """

        if not self.fitted_state.has_targets:
            raise ValueError(
                "this inference task holds only the input half of its fitted state, and an "
                "artifact holds the complete state; save the training task's artifact, or "
                "reuse Artifact.load(directory) of the one it was trained with"
            )
        return Artifact(
            contract=self.contract,
            bound_contract=self.bound_contract(),
            fitted_state=self.fitted_state,
            fingerprints=dict(self.fingerprints),
            output_schema=json.loads(json.dumps(self.output_schema)),
            manifest=json.loads(json.dumps(self.manifest)),
        )

    def close(self) -> None:
        """Release the materialized anchors. The plan's binding stays open."""

        self._anchors.close()

    def __enter__(self) -> Task:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()


def request_batches(
    assembler: PreparedBatchPlan,
    requests: Sequence[SampleRequest],
    *,
    batch_size: int | None,
    epoch: int,
    seed: int,
    shuffle: bool,
    rank: int,
    world_size: int,
    drop_last: bool | None,
    output_format: Literal["inspectable", "model"] = "inspectable",
) -> Iterator[Batch]:
    """Batches over requests built in code, through ``assembler``'s fitted state."""

    anchors = RequestAnchors(requests)
    chosen = list(
        padded_positions(
            anchors.order(
                ALL_ANCHORS,
                seed=seed,
                epoch=epoch,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size,
            ),
            count=anchors.count(ALL_ANCHORS),
            world_size=world_size,
        )
    )
    ordered = _mark_padding(
        anchors.fetch(ALL_ANCHORS, [position for position, _ in chosen]),
        [padding for _, padding in chosen],
    )
    for batch in assembler.batches(
        ordered,
        batch_size=batch_size,
        epoch=epoch,
        seed=seed,
        drop_last=drop_last,
        output_format=output_format,
    ):
        yield _surface_padding(batch)


def _selection(split: str | Collection[str | None] | AllAnchors | None) -> SplitSelection:
    selection = split_selection(split)
    return ALL_ANCHORS if selection is None else selection


def _chunks(positions: Iterator[tuple[int, bool]], size: int) -> Iterator[list[tuple[int, bool]]]:
    pending: list[tuple[int, bool]] = []
    for item in positions:
        pending.append(item)
        if len(pending) == size:
            yield pending
            pending = []
    if pending:
        yield pending


def _mark_padding(
    requests: Sequence[SampleRequest], padding: Sequence[bool]
) -> list[SampleRequest]:
    """Mark padding rows, giving each its own sample ID.

    A padding row repeats an anchor its rank already has, and sample IDs must be
    unique within an assembler call. Start with ``<id>#padding<k>`` and advance
    the suffix past caller IDs and earlier padding IDs in this request chunk.
    """

    used_ids = {request.sample_id for request in requests}
    marked = []
    for index, (request, is_padding) in enumerate(zip(requests, padding, strict=True)):
        if is_padding:
            suffix = index
            sample_id = f"{request.sample_id}#padding{suffix}"
            while sample_id in used_ids:
                suffix += 1
                sample_id = f"{request.sample_id}#padding{suffix}"
            used_ids.add(sample_id)
            request = dataclass_replace(
                request,
                sample_id=sample_id,
                metadata={**request.metadata, PADDING_KEY: True},
            )
        marked.append(request)
    return marked


def _surface_padding(batch: Batch) -> Batch:
    """Move the internal marker to ``metadata["is_padding"]``, one flag per row."""

    request_metadata = batch.metadata.get("request_metadata", ())
    flags = [bool(entry.get(PADDING_KEY, False)) for entry in request_metadata]
    metadata = dict(batch.metadata)
    metadata["request_metadata"] = tuple(
        {key: value for key, value in entry.items() if key != PADDING_KEY}
        for entry in request_metadata
    )
    metadata["is_padding"] = torch.tensor(flags, dtype=torch.bool)
    return dataclass_replace(batch, metadata=metadata)
