# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and train registered bindings on a pinned extract or their full population."""

import hashlib
import json
import math
import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC
from functools import partial
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

from e2e.datasets.binding_sources import PINS, full_source
from kumo.pytorch import api
from tests.e2e.binding_plan import BINDINGS, NIGHTLY_HISTORY_EVENTS, TRAINING_BATCH_SIZE
from tests.e2e.cuda_metrics import training_gpu_window

_SPLITS = ("train", "valid", "test")


def training_contract(spec):
    """Select the latest events before fitting and after merging event families."""
    limit = NIGHTLY_HISTORY_EVENTS
    events = {
        name: replace(event, order="descending", limit=limit)
        for name, event in spec.sample.events.items()
    }
    inputs = dict(spec.batch.inputs)
    if history := inputs.get("history"):
        source = replace(history.source, max_events=limit, keep="last", order="ascending")
        inputs["history"] = replace(history, source=source)

    def limit_layer(layer):
        return replace(
            layer,
            samples=tuple(replace(hop, fanout=min(hop.fanout, limit)) for hop in layer.samples),
        )

    neighborhoods = {
        name: replace(graph, layers=tuple(limit_layer(layer) for layer in graph.layers))
        for name, graph in spec.sample.neighborhoods.items()
    }
    sample = replace(spec.sample, events=events, neighborhoods=neighborhoods)
    return replace(spec, sample=sample, batch=replace(spec.batch, inputs=inputs))


def full_tables(binding, files, directory):
    if binding.startswith(("E", "L")):
        from e2e.datasets.aml import prepare_full_aml

        return prepare_full_aml(files, directory, binding)
    from e2e.datasets.population_tables import prepare_full

    return prepare_full(binding, files, directory)


def _source_heads(binding):
    if binding in ("S01", "E03"):
        return {"masked_values": None}
    if binding in ("S02", "E04"):
        return {"next_token": None}
    if binding == "M03":
        return {k: ds.field("next_" + k + "_valid") for k in ("event_type", "event_subtype")}
    if binding == "M04":
        return dict.fromkeys(("count", "generated_count"), ds.field("future_count").is_valid())
    if binding == "L01":
        return {k: ds.field("next_" + k).is_valid() for k in ("amount", "currency")}
    return {"label": ds.field("label" if binding == "S04" else "involvement").is_valid()}


def source_population(binding, files):
    raw = ds.dataset(files["anchors"])
    manifest = json.loads(files["anchor_manifest"].read_text())
    if manifest["statistics"]["rows"] != raw.count_rows():
        raise ValueError(f"{binding}: anchor manifest row count differs from source")
    column = "forecast_split" if binding in ("E01", "E02", "E05", "L01") else "split"
    splits = {s: ds.field(column) == s for s in _SPLITS}
    counts = {s: raw.count_rows(filter=pred) for s, pred in splits.items()}
    if min(counts.values()) <= 0 or sum(counts.values()) != raw.count_rows(
        filter=ds.field(column).is_valid()
    ):
        raise ValueError(f"{binding}: invalid source split counts {counts}")
    last_train = max(
        pc.max(batch["as_of_time"]).as_py()
        for batch in raw.to_batches(columns=["as_of_time"], filter=splits["train"])
        if len(batch)
    )
    cutoff = int(last_train.replace(tzinfo=UTC).timestamp()) - 1
    targets = {
        name: None
        if eligible is None
        else {s: raw.count_rows(filter=pred & eligible) for s, pred in splits.items()}
        for name, eligible in _source_heads(binding).items()
    }
    identities = pq.read_table(
        files["anchors"], columns=["sample_id", column], read_dictionary=[column]
    ).rename_columns(["sample_id", "split"])
    return column, counts, targets, cutoff, identities


def full_batches(task, identities, batch_size, split, epoch=0, *, ordered=False):
    """Verify each original split row exactly once, including the final batch."""
    remaining = pc.fill_null(pc.equal(identities["split"], split), False).to_numpy().copy()
    sample_ids = identities["sample_id"]
    previous = -1
    for batch in task.batches(
        split=split,
        batch_size=batch_size,
        epoch=epoch,
        seed=17,
        shuffle=False,
        drop_last=False,
        output_format="model",
    ):
        meta = batch.metadata
        ids = zip(
            (r["anchor_row_position"] for r in meta["request_metadata"]),
            meta["sample_ids"],
            strict=True,
        )
        if len(meta["sample_ids"]) != batch.batch_size:
            raise ValueError(f"{split}: native batch size differs from its sample IDs")
        for position, sample_id in ids:
            if not 0 <= position < len(remaining) or not remaining[position]:
                raise ValueError(f"{split}: unexpected or repeated source row {position}")
            expected = sample_ids[position].as_py()
            if sample_id != expected:
                raise ValueError(
                    f"{split} row {position}: expected sample ID {expected!r}, got {sample_id!r}"
                )
            if ordered and position <= previous:
                raise ValueError(f"{split}: source row {position} arrived after {previous}")
            previous = position
            remaining[position] = False
        if set(meta["splits"]) != {split} or meta["is_padding"].any():
            raise ValueError(f"{split}: unexpected split or padding in native batch")
        yield batch
    if remaining.any():
        raise ValueError(f"{split}: native loader omitted source row {remaining.argmax()}")


def _allocated_bytes(files, work):
    # Count only this binding's staged files and derived work while the native task is live.
    seen, size = set(), 0
    for path in chain(files.values(), work.rglob("*")):
        info = path.stat()
        key = info.st_dev, info.st_ino
        if key not in seen:
            seen.add(key)
            size += info.st_blocks * 512
    return size


@contextmanager
def _native_training(task):
    """An extract must reach DiskGraph's model batch path for every timed batch."""
    from kumo.pytorch.engines.native_anchors import NativeAnchors

    if not isinstance(task._anchors, NativeAnchors):
        raise ValueError("Throughput requires native DiskGraph Parquet anchors")
    fallback = RuntimeError("Throughput used the Python request batcher")
    with (
        patch.object(task._anchors, "_requests", side_effect=fallback),
        patch.object(task._assembler, "batches", side_effect=fallback),
    ):
        yield


def run_full_binding(binding, e2e_dir, output, *, bounded=False):
    from tests.e2e.runs import _provenance

    output.parent.mkdir(parents=True, exist_ok=True)
    _, _, selector, representation = BINDINGS[binding]
    report = dict(
        _provenance("binding-throughput" if bounded else "binding-train"),
        binding=binding,
        selector=selector,
        representation=representation,
        scope="extract" if bounded else "full_source",
        complete=False,
        run_id=int(os.environ["GITHUB_RUN_ID"]) if "GITHUB_RUN_ID" in os.environ else None,
        run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"])
        if "GITHUB_RUN_ATTEMPT" in os.environ
        else None,
        metrics={},
    )
    metrics = report["metrics"]

    def publish(stage):
        report["stage"] = stage
        pending = output.with_suffix(".tmp")
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        pending.replace(output)

    try:
        publish("setup")
        if sys.platform != "linux" or not torch.cuda.is_available():
            raise RuntimeError("Binding training requires a Linux CUDA worker")
        import resource

        from tests.e2e.binding_training import train_full_binding

        staging = float(os.environ.get("KUMO_PYTORCH_STAGE_SECONDS", 0))
        if not math.isfinite(staging) or staging < 0:
            raise ValueError("Source staging duration must be finite and nonnegative")
        report["staging_seconds"] = staging
        device = torch.device("cuda", torch.cuda.current_device())
        batch_size = TRAINING_BATCH_SIZE
        e2e_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=e2e_dir, prefix=f"{binding}-") as directory:
            work = Path(directory)
            publish("extract" if bounded else "ingestion")
            start = perf_counter()
            root = Path(os.environ["KUMO_PYTORCH_DATA_ROOT"])
            if bounded:
                from e2e.datasets.throughput_sources import load_extract

                spec, data, anchors, files, cutoff, identities, counts, targets = load_extract(
                    binding, root
                )
                pin = (PINS / "throughput" / f"{binding.lower()}.json").read_bytes()
                report["extract_manifest_sha256"] = hashlib.sha256(pin).hexdigest()
                upstream = json.loads(pin)["upstream_sha256"]
                for key in ("source_manifest", "anchor_manifest"):
                    report[key + "_sha256"] = upstream[key]
                report["extract_seconds"] = perf_counter() - start
                column = "split"
            else:
                files = full_source(binding, root)
                column, counts, targets, cutoff, identities = source_population(binding, files)
                for key in ("source_manifest", "anchor_manifest"):
                    report[key + "_sha256"] = hashlib.sha256(files[key].read_bytes()).hexdigest()
            report.update(
                split_counts=counts,
                split_column=column,
                expected_eligible=targets,
                encoder_cutoff=cutoff,
                fit_split="train",
                batch_size=batch_size,
                device=device.type,
                history_limit=NIGHTLY_HISTORY_EVENTS,
            )
            if bounded:
                publish("ingestion")
                start = perf_counter()
            else:
                spec, data, anchors = full_tables(binding, files, work / "prepared")
            engine = api.DiskGraphEngine(index=work / "index", num_threads=4)
            with api.prepare(training_contract(spec), data, engine) as plan:
                metrics["ingestion_seconds"] = (perf_counter() - start) + (
                    0 if bounded else staging
                )
                publish("materialization")
                start = perf_counter()
                with plan.materialize(anchors, fit_split="train", encoder_cutoff=cutoff) as task:
                    metrics["materialization_seconds"] = perf_counter() - start
                    observed = {split: task.count(split) for split in counts}
                    if observed != counts:
                        raise ValueError(
                            f"Source split counts {counts} differ from task {observed}"
                        )
                    report["fingerprints"] = dict(task.fingerprints)
                    report["validation"] = []
                    if not bounded:
                        report["epochs"] = []
                    publish("warmup" if bounded else "training")

                    def save_split(row):
                        split = row["split"]
                        for head, loss in row["heads"].items():
                            expected = targets[head]
                            if (
                                expected is not None
                                and not (bounded and split == "train")
                                and loss["eligible"] != expected[split]
                            ):
                                raise ValueError(
                                    f"{split}: eligible {head} count differs from source"
                                )
                        if "phase" in row:
                            report[row["phase"]] = row
                        elif split == "test":
                            report["test"] = row
                        else:
                            report["epochs" if split == "train" else "validation"].append(row)
                        if bounded:
                            following = {
                                "warmup": "measurement",
                                "measurement": "validation",
                                "valid": "test",
                                "test": "finalize",
                            }
                            publish(following[row.get("phase", split)])
                        else:
                            publish("training")

                    with _native_training(task) if bounded else nullcontext():
                        train_full_binding(
                            task,
                            binding,
                            batch_size=batch_size,
                            device=device,
                            split_batches=partial(full_batches, task, identities, batch_size),
                            train_window=partial(training_gpu_window, device),
                            on_split=save_split,
                            bounded=bounded,
                        )
                    metrics["ssd_bytes"] = _allocated_bytes(files, work)
        epochs = [report["measurement"]] if bounded else report["epochs"]
        seconds = sum(e["elapsed_ns"] for e in epochs) / 1e9
        readings = sum(e["gpu"]["samples"] for e in epochs)
        metrics.update(
            training_seconds=seconds,
            samples_per_second=sum(e["samples"] for e in epochs) / seconds,
            tokens_per_second=sum(e["tokens"] for e in epochs) / seconds,
            average_gpu_utilization_percent=sum(
                e["gpu"]["samples"] * e["gpu"]["mean_percent"] for e in epochs
            )
            / readings,
            peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        report["complete"] = True
        publish("complete")
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        publish(report["stage"])
        raise
