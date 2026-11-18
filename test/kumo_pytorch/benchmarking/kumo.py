# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Kumo API adapter, independent of source schema and model architecture."""

import hashlib
import json
from collections.abc import Callable, Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .interface import BatchCounts, Workload, _nothing


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


def counts(batch, *, event_tables=()):
    """Count emitted input positions, graph occurrences and logical tensor bytes.

    Tokens exclude targets/parallel field IDs. Events are history occurrences
    (unique source-event indices per emitted history); graph-only uses emitted
    nodes in the declared event tables. Output size counts tensor occurrences in inputs/targets,
    not metadata or unique storage allocations.
    """
    import torch

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from tensors(item)

    payload = list(tensors(batch.inputs)) + list(tensors(batch.targets))
    history, graph = batch.inputs.get("history"), batch.inputs.get("graph")
    tokens = events = nodes = edges = None
    if history is not None:
        tokens = int(history["valid"].sum())
        if "sequence" in history:
            events = int(history["sequence"]["valid"].sum())
        else:
            indices = history["source_event_indices"][history["valid"]]
            events = int(indices[indices >= 0].unique().numel())
    if graph is not None:
        nodes = sum(len(node["ids"]) for node in graph["nodes"].values())
        edges = sum(edge["index"].shape[1] for edge in graph["edges"].values())
        if events is None:
            events = (
                sum(len(graph["nodes"][name]["ids"]) for name in event_tables)
                if event_tables
                else None
            )
    return BatchCounts(
        len(batch.metadata["sample_ids"]),
        int(batch.metadata["is_padding"].sum()),
        tokens=tokens,
        nodes=nodes,
        edges=edges,
        events=events,
        output_elements=sum(t.numel() for t in payload),
        output_bytes=sum(t.numel() * t.element_size() for t in payload),
    )


def consumer(selection, family, *, capacity, device, seed):
    """Create the selected task consumer on the caller-selected runner device."""
    import torch

    from ..bindings import binding_for, sequence_capacity
    from ..models import build_model

    def create(task, recorder):
        model = build_model(
            task,
            selection,
            family,
            max_seq_len=sequence_capacity(binding_for(selection, family), capacity()),
            seed=seed,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)

        def step(batch, recorder):
            with recorder.span("model.adapt"):
                model.validate_batch(batch)
            if device != "cpu":
                with recorder.span("transfer.host"):
                    batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            result = model.training_step(batch, validate_inputs=False)
            if not result.weight:
                return False
            with recorder.span("model.backward"):
                result.loss.backward()
            with recorder.span("model.optimizer"):
                optimizer.step()
            return True

        return step

    return create


def factory(config, workdir):
    import pyarrow.parquet as pq
    import torch

    from kumo.pytorch import api as r

    from ..bindings import (
        contract_for,
        dataset_definition,
        load_bindings,
        profile,
    )
    from ..bindings import selection as selection_for
    from ..bindings import task_definition
    from ..data import prepare_binding, schema_for, verify_dataset
    from ..population_anchors import projected_rows

    allowed = {
        "data_dir",
        "binding",
        "family",
        "workload",
        "anchors_per_split",
        "batch_size",
        "threads",
        "seed",
        "device",
        "training",
        "prepared_dir",
        "index_dir",
        "index_action",
        "load_fitted_state",
        "save_fitted_state",
        "cache_mode",
        "backend",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown benchmark options: {sorted(set(config) - allowed)}")
    bindings = load_bindings()
    if config.get("binding"):
        binding = bindings[config["binding"]]
        if (
            config.get("family", binding["representation"]) != binding["representation"]
            or config.get("workload", binding["selector"]["workload_id"])
            != binding["selector"]["workload_id"]
        ):
            raise ValueError("Benchmark options contradict the selected binding")
    else:
        candidates = [
            b
            for b in bindings.values()
            if b["campaigns"]
            and b["representation"] == config["family"]
            and b["selector"]["workload_id"] == config["workload"]
        ]
        if len(candidates) != 1:
            raise ValueError("Select one registered binding for the benchmark")
        binding = candidates[0]
    family, workload = binding["representation"], binding["selector"]["workload_id"]
    selection = selection_for(binding)
    for key in ("anchors_per_split", "batch_size", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    device = config.get("device", "cpu")
    cache_mode = config.get("cache_mode", "uncontrolled")
    if cache_mode not in {"uncontrolled", "cold-warm"}:
        raise ValueError("cache_mode must be uncontrolled or cold-warm")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("Training supports CPU or runner-provided CUDA")
    seed = config.get("seed", 0)
    training = config.get("training", True)
    backend = config.get("backend", "diskgraph")
    if backend not in {"diskgraph", "polars", "gpu"}:
        raise ValueError("backend must be diskgraph, polars or gpu")
    reference = backend != "diskgraph"
    if type(training) is not bool:
        raise ValueError("training must be boolean")
    if reference and (
        training or cache_mode != "uncontrolled" or "index_dir" in config
    ):
        raise ValueError(
            "Reference backends support loader timing only with uncontrolled caches and no DiskGraph index"
        )
    if backend == "gpu" and not device.startswith("cuda"):
        raise ValueError("The GPU reference requires a runner-provided CUDA device")
    if not training and backend != "gpu" and device != "cpu":
        raise ValueError(
            "DiskGraph and Polars loader-only execution requires device=cpu"
        )
    data_dir = Path(config["data_dir"]).resolve()
    previous_threads = torch.get_num_threads()
    state = {}
    # These declarations follow the selected contract, never probe availability.
    inapplicable = {
        "bpe.fit.elapsed_ns": "Selected contract uses categorical/bucket encoders without BPE",
    }
    if reference:
        inapplicable.update(
            {
                name: "Reference preparation has no DiskGraph index or Rust sampling/retrieval operations"
                for name in (
                    "index.build.elapsed_ns",
                    "index.reopen.elapsed_ns",
                    "index.bytes",
                    "native.sample.elapsed_ns",
                    "native.retrieve.elapsed_ns",
                    "native.aggregate.elapsed_ns",
                )
            }
        )
    else:
        inapplicable.update(
            {
                f"reference.{stage}.elapsed_ns": "DiskGraph execution; reference comparison is a separate experiment"
                for stage in ("load", "fit", "plan", "batch")
            }
        )
    if family == "graphsage":
        inapplicable.update(
            {
                name: "Graph-only inputs have no token sequence"
                for name in (
                    "tokens",
                    "tokens_per_second",
                    "python.tokenize.elapsed_ns",
                )
            }
        )
        inapplicable["native.aggregate.elapsed_ns"] = (
            "Graph-only contract has no temporal aggregates"
        )
    else:
        if family != "combined":
            inapplicable.update(
                {
                    name: "Sequence-only inputs emit no graph"
                    for name in (
                        "nodes",
                        "edges",
                        "nodes_per_second",
                        "edges_per_second",
                    )
                }
            )
    if task_definition(binding)["kind"] != "masked_values":
        inapplicable["python.masks.elapsed_ns"] = (
            "Task does not apply masked-value corruption"
        )
    if config.get("load_fitted_state"):
        inapplicable.update(
            {
                f"{name}.elapsed_ns": "Complete fitted state is loaded without refitting"
                for name in (
                    "input.fit",
                    "target.fit",
                    "vocabulary.fit",
                    "statistics.fit",
                )
            }
        )
    else:
        inapplicable["state.load.elapsed_ns"] = (
            "Fresh fitting run; paired campaign run measures state loading"
        )
    if not dataset_definition(binding).get("prepare") or (
        config.get("prepared_dir") and Path(config["prepared_dir"]).exists()
    ):
        inapplicable["source.convert.elapsed_ns"] = (
            "Source tables already use the declared schema; verified on read"
        )
    index_action = config.get("index_action")
    if index_action is not None:
        if (
            reference
            or index_action not in {"build", "reopen"}
            or not config.get("index_dir")
        ):
            raise ValueError(
                "Index actions require DiskGraph and an explicit index_dir"
            )
        exists = Path(config["index_dir"]).exists()
        if exists != (index_action == "reopen"):
            raise ValueError(
                "Build requires a new index; reopen requires the existing index"
            )
        if index_action == "build":
            inapplicable["index.reopen.elapsed_ns"] = (
                "Separate lifecycle runs measure index reopening"
            )
    if config.get("index_dir") and Path(config["index_dir"]).exists():
        inapplicable["index.build.elapsed_ns"] = (
            "Caller supplied an existing index; reopen is measured"
        )
    if device == "cpu":
        inapplicable.update(
            {
                name: "CPU execution has no device allocation or transfer"
                for name in (
                    "gpu.peak_allocated_bytes",
                    "gpu.peak_reserved_bytes",
                    "gpu.utilization_percent",
                    "transfer.host.elapsed_ns",
                    "transfer.device_ns",
                )
            }
        )
    if not training:
        inapplicable.update(
            {
                name: "Loader-only run has no model or optimizer"
                for name in (
                    "training.step.elapsed_ns",
                    "steps",
                    "optimizer_updates",
                    "steps_per_second",
                )
            }
        )

    def inputs(recorder):
        torch.set_num_threads(config["threads"])
        with recorder.span("source.read") as details:
            details["scope"] = (
                "checksum verification of pinned sources, bindings and anchors"
            )
            files = verify_dataset(data_dir, binding)
        prepared = Path(config.get("prepared_dir", workdir / "prepared"))
        dataset, table, projection, manifest, capacity = prepare_binding(
            data_dir,
            workdir,
            binding,
            files,
            limit=config["anchors_per_split"],
            prepared_dir=prepared,
            recorder=recorder,
        )
        state["capacity"] = capacity
        source_counts = {
            role: pq.ParquetFile(item["path"]).metadata.num_rows
            for role, item in manifest["sources"].items()
        }
        recorder.measure(
            "source.rows", sum(source_counts.values()), source_rows=source_counts
        )
        recorder.measure(
            "source.bytes",
            sum(item["bytes"] for item in manifest["sources"].values()),
            scope="declared source tables; anchor files recorded separately",
        )
        recorder.measure(
            "converted.rows", sum(t["rows"] for t in manifest["tables"].values())
        )
        recorder.measure(
            "converted.bytes",
            sum(Path(path).stat().st_size for path in manifest["table_paths"].values()),
        )
        graph = family in {"graphsage", "combined"}
        recorder.write(
            "dataset_inputs",
            sources=manifest["sources"],
            tables=manifest["tables"],
            anchors=list(projected_rows(workdir / "anchors.parquet", "train")),
            files=list(files.values()),
            history_capacity=capacity,
            data_scope="complete train pass over bounded physical-row selection; full selected-root histories",
        )
        prepare_plan = observe = None
        if backend == "gpu":
            from .gpu_reference import bind

            engine = None
            prepare_plan, observe = bind(
                recorder, split_dir=workdir / "gpu-split", device=device
            )
        elif backend == "polars":
            from ..reference_engine.polars import PolarsEngine

            engine = PolarsEngine(num_threads=config["threads"])
        else:
            engine = r.DiskGraphEngine(
                index=Path(config.get("index_dir", workdir / "index")),
                ingest="never" if index_action == "reopen" else "if_missing",
                num_threads=config["threads"],
            )
        contract = contract_for(
            schema_for(dataset, binding), binding, max_events=capacity
        )
        if not graph:
            contract = replace(
                contract,
                batch=replace(
                    contract.batch,
                    batching=replace(
                        contract.batch.batching,
                        bucket_by_length=True,
                        length_input="history",
                    ),
                ),
            )
        return KumoInputs(
            contract,
            dataset,
            engine,
            table,
            materialize_options={
                "fit_split": "train",
                "encoder_cutoff": projection["encoder_cutoff_seconds"],
            },
            batch_options={
                "split": "train",
                "batch_size": config["batch_size"],
                "seed": seed,
                "output_format": "model",
            },
            reference=reference,
            prepare_plan=prepare_plan,
            observe=observe,
        )

    result = kumo_workload(
        identity={
            "dataset": binding["selector"]["dataset_id"],
            "version": binding["selector"]["dataset_version"],
            "workload": workload,
            "model": family if training else None,
            "backend": backend,
            "output_device": device if backend == "gpu" else "cpu",
            "input_family": family,
            "anchor_version": selection["selector"]["anchor_version"],
            "anchors_per_split": config["anchors_per_split"],
            "population": "bounded train selection",
        },
        inputs=inputs,
        count=lambda batch: counts(
            batch,
            event_tables=(profile(binding, "input")["graph"]["node_table"],)
            if family in {"graphsage", "combined"}
            else (),
        ),
        consumer_factory=consumer(
            selection,
            family,
            capacity=lambda: state["capacity"],
            device=device,
            seed=seed,
        )
        if training
        else None,
        synchronize=(lambda: torch.cuda.synchronize(device))
        if device != "cpu"
        else (lambda: None),
        device=device,
        inapplicable=inapplicable,
        probe_reopen=not reference and index_action is None,
        load_fitted_state=config.get("load_fitted_state"),
        save_fitted_state=config.get("save_fitted_state"),
    )
    close = result.close
    if cache_mode == "cold-warm":
        from .file_cache import FilePageCache

        index = Path(config.get("index_dir", workdir / "index"))
        result.before_epoch = FilePageCache(
            lambda: (p for p in index.rglob("*") if not p.is_dir()),
            release_mappings=True,
        )

    def cleanup():
        try:
            close()
        finally:
            torch.set_num_threads(previous_threads)

    result.close = cleanup
    return result
