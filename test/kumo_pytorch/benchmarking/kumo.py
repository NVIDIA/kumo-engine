# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Kumo API adapter, independent of source schema and model architecture."""

import hashlib
import json
from collections.abc import Callable, Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .interface import Workload, _nothing


def anchor_identity(table):
    """Compare the same Parquet population across independently located runs."""
    from kumo.pytorch.engines.parquet_anchors import _files

    binding = table.to_dict()
    binding.pop("source")
    files = []
    for path in _files(table.source.path):
        with open(path, "rb") as stream:
            files.append(hashlib.file_digest(stream, "sha256").hexdigest())
    return hashlib.sha256(
        json.dumps([binding, files], sort_keys=True).encode()
    ).hexdigest()


@dataclass
class KumoInputs:
    contract: Any
    dataset: Any
    engine: Any
    fitting_anchors: Any
    materialize_options: Mapping[str, Any] = field(default_factory=dict)
    batch_options: Mapping[str, Any] = field(default_factory=dict)
    reference: bool = False
    prepare_plan: Callable[[Any, Any], Any] | None = None
    observe: Callable[[Any], None] | None = None


def kumo_workload(
    *,
    identity,
    inputs,
    count,
    consumer_factory=None,
    synchronize=_nothing,
    inapplicable=None,
    device="cpu",
    load_fitted_state=None,
    save_fitted_state=None,
    probe_reopen=False,
):
    """Build a Workload using any Kumo contract/dataset/backend and model.

    ``inputs(recorder) -> KumoInputs`` prepares sources/anchors and records actual
    input preparation. ``fitting_anchors`` is an AnchorTable owned by the task;
    ``batch_options`` selects its split. ``consumer_factory(task, recorder)``
    optionally returns ``step(batch, recorder) -> bool`` with its own optimizer.
    Each factory call must create fresh mutable engine/model/optimizer state.

    Enclosing prepare/materialize spans and scoped internal observers retain
    separate ownership. Unsupported internal boundaries remain incomplete.
    """
    stack = ExitStack()
    state = {}

    def prepare(recorder):
        from kumo.pytorch.api import AnchorTable, Artifact
        from kumo.pytorch.api import prepare as prepare_task

        from .kumo_observers import KumoObservers

        observers = stack.enter_context(KumoObservers(recorder))
        with recorder.span("ingestion"):
            selected = inputs(recorder)
            if not isinstance(selected, KumoInputs):
                raise TypeError("inputs must return KumoInputs")
            if not isinstance(selected.fitting_anchors, AnchorTable):
                raise TypeError(
                    "Benchmarks require an AnchorTable, not explicit requests"
                )
            if selected.batch_options.get("output_format", "model") != "model":
                raise ValueError("Benchmarks require output_format='model'")
            if "epoch" in selected.batch_options:
                raise ValueError("The benchmark runner owns epoch selection")
            if "requests" in selected.batch_options:
                raise ValueError(
                    "Select the task's AnchorTable split; explicit requests are not supported"
                )
            with recorder.span("anchors.prepare"):
                anchors_identity = anchor_identity(selected.fitting_anchors)
            state["inputs"] = selected
            if selected.observe is not None:
                selected.observe(observers)
            plan_factory = selected.prepare_plan or (
                lambda contract, dataset: prepare_task(
                    contract, dataset, selected.engine
                )
            )
            if selected.reference and selected.prepare_plan is None:
                observers.patch(
                    selected.engine, "open", observers.timed("reference.load")
                )
            with recorder.span("task.setup"):
                plan = stack.enter_context(
                    plan_factory(selected.contract, selected.dataset)
                )
        observers.attach_source(getattr(plan, "_source", None))
        if probe_reopen:
            plan.close()
            plan = stack.enter_context(
                plan_factory(selected.contract, selected.dataset)
            )
            observers.attach_source(getattr(plan, "_source", None))
        with (
            recorder.span("materialization"),
            recorder.span("reference.fit") if selected.reference else nullcontext(),
        ):
            options = dict(selected.materialize_options)
            loaded = options.get("fitted")
            if load_fitted_state is not None:
                if "fitted" in options:
                    raise ValueError("Choose one fitted-state source")
                with recorder.span("state.load"):
                    loaded = Artifact.load(load_fitted_state).fitted_state
                options.pop("fit_split", None)
                options["fitted"] = loaded
            task = stack.enter_context(
                plan.materialize(selected.fitting_anchors, **options)
            )
            if selected.prepare_plan is not None:
                synchronize()
            if (
                loaded is not None
                and loaded.has_targets
                and task.fitted_state.fingerprint != loaded.fingerprint
            ):
                raise ValueError("Reused fitted state changed during materialization")
        if save_fitted_state is not None:
            Path(save_fitted_state).mkdir(parents=True, exist_ok=False)
            task.to_artifact().save(save_fitted_state)
        recorder.write(
            "fitting_state",
            mode="fresh"
            if loaded is None
            else "reused"
            if loaded.has_targets
            else "partial_reuse",
            fingerprint=task.fitted_state.fingerprint,
            contract_fingerprint=selected.contract.fingerprint,
            anchors_fingerprint=anchors_identity,
            task_anchors_fingerprint=task.fingerprints["anchors"],
            loaded_from=str(load_fitted_state)
            if load_fitted_state is not None
            else None,
            saved_to=str(save_fitted_state) if save_fitted_state is not None else None,
        )
        state["task"] = task
        state["recorder"] = recorder
        state["observers"] = observers
        observers.attach_source(
            getattr(getattr(task, "_assembler", None), "_source", None)
        )
        if consumer_factory is not None:
            state["step"] = consumer_factory(task, recorder)
            if not callable(state["step"]):
                raise TypeError("consumer_factory must return a callable training step")

    def batches(epoch):
        from kumo.pytorch.api import EventSource

        selected = state["inputs"]
        recorder = state["recorder"]
        options = {"output_format": "model", **selected.batch_options}
        policy = selected.contract.batch.batching
        observation = (
            nullcontext()
            if selected.reference
            else state["observers"].native_epoch(
                epoch,
                length_planning=policy.max_tokens is not None
                or policy.bucket_by_length,
                history=any(
                    isinstance(value.source, EventSource)
                    for value in selected.contract.batch.inputs.values()
                ),
            )
        )
        with observation:
            yield from iterate(epoch, selected, recorder, options)

    def iterate(epoch, selected, recorder, options):
        with recorder.span("reference.plan") if selected.reference else nullcontext():
            iterator = iter(state["task"].batches(epoch=epoch, **options))
            if selected.prepare_plan is not None:
                synchronize()
        while True:
            with (
                recorder.span("reference.batch")
                if selected.reference
                else nullcontext()
            ):
                try:
                    batch = next(iterator)
                    if selected.prepare_plan is not None:
                        synchronize()
                except StopIteration:
                    return
            yield batch

    def step(batch, recorder):
        return state["step"](batch, recorder)

    return Workload(
        identity=identity,
        prepare=prepare,
        batches=batches,
        count=count,
        step=step if consumer_factory is not None else None,
        synchronize=synchronize,
        close=stack.close,
        inapplicable=inapplicable or {},
        device=device,
    )
