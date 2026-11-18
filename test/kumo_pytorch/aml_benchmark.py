# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AML binding for the general benchmark API; no Arena runtime dependency.

One complete pass means the explicitly selected train anchors, not all AML
anchors. Histories are retained in full for selected roots. Correctness audits
remain in the existing suite and are not repeated inside timed loader calls.
"""

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from .benchmarking import BatchCounts
from .benchmarking.kumo import KumoInputs, kumo_workload
from .data import ROOT


def counts(batch):
    """Count emitted input positions, graph occurrences and logical tensor bytes.

    Tokens exclude targets/parallel field IDs. Events are history occurrences
    (unique source-event indices per emitted history); graph-only uses emitted
    transaction nodes. Output size counts tensor occurrences in inputs/targets,
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
            events = len(graph["nodes"].get("transactions", {}).get("ids", ()))
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
    """Create the existing AML consumer on the caller-selected runner device."""
    import torch

    from .models.aml import build_model

    def create(task, recorder):
        model = build_model(
            task, selection, family, max_seq_len=11 * capacity(), seed=seed
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

    from .aml_inputs import load_selections, verify_inputs
    from .aml_source import aml_data_schema
    from .contracts import build_contract
    from .population_anchors import project_population
    from .run_aml import prepare

    allowed = {
        "data_dir",
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
        raise ValueError(
            f"Unknown AML benchmark options: {sorted(set(config) - allowed)}"
        )
    family, workload = config["family"], config["workload"]
    selections = json.loads((ROOT / "workloads/aml.json").read_text())
    selection = next(
        (
            s
            for s in selections
            if s["selector"]["workload_id"] == workload and family in s["models"]
        ),
        None,
    )
    if selection is None:
        raise ValueError("Unregistered AML workload/model combination")
    for key in ("anchors_per_split", "batch_size", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    device = config.get("device", "cpu")
    cache_mode = config.get("cache_mode", "uncontrolled")
    if cache_mode not in {"uncontrolled", "cold-warm"}:
        raise ValueError("cache_mode must be uncontrolled or cold-warm")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("AML training supports CPU or runner-provided CUDA")
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
        "bpe.fit.elapsed_ns": "AML contracts use categorical encoders, not BPE",
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
    if family != "pragma":
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
    if config.get("prepared_dir") and Path(config["prepared_dir"]).exists():
        inapplicable["source.convert.elapsed_ns"] = (
            "Caller supplied already converted source tables; verified on read"
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
            files = verify_inputs(data_dir)
        with recorder.span("anchors.prepare"):
            anchors = load_selections(data_dir, limit=config["anchors_per_split"])
        prepared = Path(config.get("prepared_dir", workdir / "prepared"))
        with recorder.span(
            "source.convert" if not prepared.exists() else "source.read"
        ) as details:
            details["scope"] = (
                "complete selected-root history mapping, including source reads and output verification"
            )
            dataset, identities, manifest, capacity = prepare(
                data_dir, prepared, anchors, files
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
            scope="transaction source and account binding; anchor files recorded separately",
        )
        recorder.measure(
            "converted.rows", sum(t["rows"] for t in manifest["tables"].values())
        )
        recorder.measure(
            "converted.bytes",
            sum(
                (prepared / t["filename"]).stat().st_size
                for t in manifest["tables"].values()
            ),
        )
        graph = family in {"graphsage", "combined"}
        native = workload != "account-involvement-next24h"
        with recorder.span("anchors.prepare"):
            groups = anchors[selection["selector"]["anchor_version"]][
                selection["anchor_binding"]["split_column"]
            ]
            table, projection = project_population(
                data_dir
                / "anchors/aml-data/v8/hi-small/account-daily"
                / selection["selector"]["anchor_version"]
                / "anchors.parquet",
                workdir / "anchors.parquet",
                identities,
                graph=graph,
                native=native,
                limit=config["anchors_per_split"],
                splits=("train",),
            )
        recorder.write(
            "dataset_inputs",
            sources=manifest["sources"],
            tables=manifest["tables"],
            anchors=groups["train"],
            files=list(files.values()),
            history_capacity=capacity,
            data_scope="complete train pass over bounded physical-row selection; full selected-root histories",
        )
        prepare_plan = observe = None
        if backend == "gpu":
            from .benchmarking.gpu_reference import bind

            engine = None
            prepare_plan, observe = bind(
                recorder, split_dir=workdir / "gpu-split", device=device
            )
        elif backend == "polars":
            from .reference_engine.polars import PolarsEngine

            engine = PolarsEngine(num_threads=config["threads"])
        else:
            engine = r.DiskGraphEngine(
                index=Path(config.get("index_dir", workdir / "index")),
                ingest="never" if index_action == "reopen" else "if_missing",
                num_threads=config["threads"],
            )
        contract = build_contract(
            aml_data_schema(), family, native=native, max_events=capacity
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
            "dataset": "aml-data/hi-small",
            "version": "v8",
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
        count=counts,
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
        from .benchmarking.file_cache import FilePageCache

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
