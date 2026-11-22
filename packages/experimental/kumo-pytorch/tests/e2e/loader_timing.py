# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time a complete CPU model-batch epoch on an existing package task."""

from collections import Counter
from math import ceil
from statistics import mean, median, pstdev
from time import perf_counter_ns

import torch
from torch.utils._pytree import tree_leaves

from kumo.pytorch import api

NATIVE_COUNTERS = ("sampling_ns", "retrieval_ns", "rows_examined", "rows_decoded")


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
            counts["nodes"] += sum(len(node["ids"]) for node in output["nodes"].values())
            counts["edges"] += sum(edge["index"].shape[1] for edge in output["edges"].values())
            if event_tables:
                counts["events"] += sum(
                    len(node["ids"])
                    for name, node in output["nodes"].items()
                    if name in event_tables
                )
        elif not isinstance(spec.representation.row, api.Columns):
            counts["tokens"] += int(output["valid"].sum())
    return counts


def measure_epoch(task, *, split, epoch, batch_size, event_tables=()):
    """Count returned batches once. Terminal exhaustion contributes only to elapsed and wait."""
    source = task._assembler._source
    requests, sizes, profiles = [], [], []
    totals = Counter(samples=0, padded=0)
    start = perf_counter_ns()
    batches = task.batches(
        split=split,
        epoch=epoch,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        output_format="model",
    )
    startup = perf_counter_ns() - start
    while True:
        previous = source.last_sampling_profile
        if len(requests) == 2:
            warmup_end = perf_counter_ns()
            warmup_samples = totals["samples"]
        request = perf_counter_ns()
        batch = next(batches, None)
        end = perf_counter_ns()
        if batch is None:
            break
        if not requests:
            first_batch = end - start
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
