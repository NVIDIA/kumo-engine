# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time a complete CPU model-batch epoch on an existing package task."""

import os
from collections import Counter
from contextlib import ExitStack, contextmanager
from functools import wraps
from math import ceil
from statistics import mean, median, pstdev
from threading import Lock, get_ident
from time import perf_counter_ns
from unittest.mock import patch

import torch
from torch.utils._pytree import tree_leaves

from kumo.pytorch import api

NATIVE_COUNTERS = ("sampling_ns", "retrieval_ns", "rows_examined", "rows_decoded")
_OBSERVER_LOCK = Lock()


class _StageObserver:
    def __init__(self, stack):
        self._stack = stack
        self._thread = get_ident()
        self.stages = {}

    @contextmanager
    def measure(self, stage, /, *args, **kwargs):
        name = None
        if get_ident() == self._thread:
            name = stage(*args, **kwargs) if callable(stage) else stage
        if name is None:
            yield
            return
        started = perf_counter_ns()
        try:
            yield
        finally:
            elapsed = perf_counter_ns() - started
            totals = self.stages.setdefault(name, {"calls": 0, "elapsed_ns": 0})
            totals["calls"] += 1
            totals["elapsed_ns"] += elapsed

    def wrap(self, owner, name, stage):
        original = getattr(owner, name)

        @wraps(original)
        def timed(*args, **kwargs):
            with self.measure(stage, *args, **kwargs):
                return original(*args, **kwargs)

        self._stack.enter_context(patch.object(owner, name, timed))


@contextmanager
def stage_observer(install=None, *, optional=False):
    """Scope inclusive stage timers to one loader thread and restore every wrapped call."""
    if optional and os.getenv("KUMO_PYTORCH_PROFILE_LOADER") != "1":
        yield None
        return
    with ExitStack() as stack:
        if not _OBSERVER_LOCK.acquire(blocking=False):
            raise RuntimeError("a loader stage observer is already active")
        stack.callback(_OBSERVER_LOCK.release)
        observer = _StageObserver(stack)
        if install is not None:
            install(observer)
        yield observer


def install_stages(observer):
    import diskgraph

    from kumo.pytorch import assembly, encoding, preparation
    from kumo.pytorch.engines.diskgraph import DiskGraphSource
    from kumo.pytorch.engines.native_anchors import NativeAnchors

    def encoder_fit(config, dtype):
        if encoding.learns_from_values(config, dtype):
            if isinstance(config, api.Vocabulary):
                return "vocabulary.fit"
            if isinstance(config, (api.Buckets, api.ZScore)):
                return "statistics.fit"
        return None

    observer.wrap(assembly, "_fit_inputs", "input.fit")
    observer.wrap(assembly, "_fit_targets", "target.fit")
    observer.wrap(
        assembly, "fit_encoder", lambda identity, config, dtype, values: encoder_fit(config, dtype)
    )
    observer.wrap(
        preparation,
        "fit_from_stats",
        lambda identity, encoder, stats: encoder_fit(encoder.config, encoder.dtype),
    )
    # Cached and supplied tokenizers skip both training calls.
    observer.wrap(encoding, "train_text", "bpe.fit")
    observer.wrap(diskgraph, "train_tokenizer", "bpe.fit")
    # Native iterator setup includes attaching the plan. Length evaluation happens in next().
    observer.wrap(NativeAnchors, "_iter", "python.plan")
    observer.wrap(assembly.PreparedBatchPlan, "_compose_batches", "python.plan")
    observer.wrap(assembly.PreparedBatchPlan, "_collate", "python.postprocess")
    observer.wrap(
        assembly,
        "_rows_for_inputs",
        lambda spec, declarations, *args, **kwargs: "python.merge" if declarations else None,
    )
    observer.wrap(encoding.FittedEncoder, "encode_arrow", "python.encode")
    observer.wrap(assembly, "_token_rows", "python.tokenize")
    observer.wrap(assembly, "_mask_tokens", "python.masks")
    observer.wrap(assembly, "_target_outputs", "python.labels")
    observer.wrap(assembly, "_column_output", "python.assemble")
    observer.wrap(assembly, "_token_output", "python.assemble")
    observer.wrap(DiskGraphSource, "sample_graph_batch", "python.assemble")


def _latencies(waits):
    if not waits:
        return None
    return {"p50_ns": median(waits), "p95_ns": sorted(waits)[ceil(0.95 * len(waits)) - 1]}


def _dimensions(task, batch, event_tables):
    tensors = [t for t in tree_leaves((batch.inputs, batch.targets)) if isinstance(t, torch.Tensor)]
    counts = Counter(
        padded=int(batch.metadata["is_padding"].sum()),
        output_elements=sum(t.numel() for t in tensors),
        output_bytes=sum(t.numel() * t.element_size() for t in tensors),
    )
    counts["samples"] = batch.batch_size - counts["padded"]
    for name, spec in task.contract.batch.inputs.items():
        output = batch.inputs[name]
        if isinstance(spec.representation, api.Graph):
            nodes = {table: len(node["ids"]) for table, node in output["nodes"].items()}
            counts["nodes"] += sum(nodes.values())
            counts["edges"] += sum(edge["index"].shape[1] for edge in output["edges"].values())
            if event_tables:
                counts["events"] += sum(nodes[table] for table in nodes.keys() & event_tables)
        elif not isinstance(spec.representation.row, api.Columns):
            counts["tokens"] += int(output["valid"].sum())
    return counts


def measure_epoch(task, make_batches, *, split, epoch, batch_size, event_tables=()):
    """Count returned batches once. Terminal exhaustion contributes only to elapsed and wait."""
    source = task._assembler._source
    requests, sizes, profiles = [], [], []
    totals = Counter(samples=0, padded=0)
    start = perf_counter_ns()
    batches = make_batches(
        task, split=split, epoch=epoch, batch_size=batch_size, shuffle=False, drop_last=False
    )
    startup = perf_counter_ns() - start
    while True:
        previous = source.last_sampling_profile
        request = perf_counter_ns()
        batch = next(batches, None)
        end = perf_counter_ns()
        if batch is None:
            break
        if not requests:
            first_batch = end - start
        elif len(requests) == 2:
            warmup_end = request
            warmup_samples = totals["samples"]
        requests.append(end - request)
        sizes.append(batch.batch_size)
        profile = source.last_sampling_profile
        fresh = profile is not previous and profile is not None
        profiles.append(
            profile if fresh and all(key in profile for key in NATIVE_COUNTERS) else None
        )
        totals.update(_dimensions(task, batch, event_tables))

    elapsed = end - start
    terminal = end - request
    profiled = sum(p is not None for p in profiles)
    native = None
    if profiled and profiled == len(profiles):
        native = {key: sum(p[key] for p in profiles) for key in NATIVE_COUNTERS}
    steady = None
    later = requests[2:]
    if len(later) >= 4:
        later_samples = totals["samples"] - warmup_samples
        window = end - warmup_end
        steady = {
            "batches": len(later),
            "samples": later_samples,
            "elapsed_ns": window,
            "samples_per_s": later_samples * 1e9 / window,
            **_latencies(later),
            "cv": pstdev(later) / mean(later) if any(later) else None,
        }
    rates = {
        key + "_per_s": totals[key] * 1e9 / elapsed
        for key in ("tokens", "nodes", "edges")
        if key in totals
    }
    return {
        "epoch": epoch,
        "batches": len(requests),
        "batch_sizes": sizes,
        **totals,
        "samples_per_s": totals["samples"] * 1e9 / elapsed,
        **rates,
        "native_profile": native,
        "native_profiles": profiles,
        "profiled_batches": profiled,
        "startup_ns": startup,
        "first_request_ns": requests[0] if requests else None,
        "first_batch_ns": first_batch if requests else None,
        "request_ns": requests,
        "terminal_ns": terminal,
        "elapsed_ns": elapsed,
        "wait_ns": sum(requests) + terminal,
        "latency": _latencies(requests),
        "later_latency": _latencies(requests[1:]),
        "post_warmup": steady,
    }


def parquet_metadata(path):
    """Count the rows and encoded bytes of one Parquet file and identify it by content."""
    import hashlib

    import pyarrow.parquet as pq

    with open(path, "rb") as stream:
        rows = pq.read_metadata(stream).num_rows
        stream.seek(0)
        return {
            "rows": rows,
            "bytes": os.fstat(stream.fileno()).st_size,
            "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
        }
