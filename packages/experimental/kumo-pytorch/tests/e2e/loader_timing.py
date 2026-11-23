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
        self.native = []

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

    def wrap(self, owner, name, stage=None, *, after=None):
        original = getattr(owner, name)

        @wraps(original)
        def timed(*args, **kwargs):
            with self.measure(stage, *args, **kwargs):
                result = original(*args, **kwargs)
            if after is not None and get_ident() == self._thread:
                after(result, *args, **kwargs)
            return result

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


def install_native(observer):
    from kumo.pytorch.engines.diskgraph import DiskGraphSource
    from kumo.pytorch.engines.native_anchors import NativeAnchors

    def record(operation, profile):
        if profile is not None:
            observer.native.append({"operation": operation, **profile})

    def temporal(result, *_args, **_kwargs):
        record("temporal", {k: result[k] for k in ("timings_ns", "stats")})

    def runtime(_result, _anchors, _stage, runtime):
        record("runtime", runtime.get("native_profile"))
        if "history" in runtime:
            temporal(runtime["history"])

    def sample(_result, source, *_args):
        profile = source.last_sampling_profile
        # A reused runtime neighborhood already owns its combined native profile.
        if profile is not (getattr(source, "_runtime_batch", None) or {}).get("native_profile"):
            record("sample", profile)

    for method in ("_batch_values", "_requests"):
        observer.wrap(NativeAnchors, method, after=runtime)
    observer.wrap(DiskGraphSource, "_query_values", after=temporal)
    observer.wrap(DiskGraphSource, "_sample_graph_events", after=sample)
    observer.wrap(
        DiskGraphSource,
        "_feature_arrays",
        after=lambda _, source, *_args: record("fetch", source.last_retrieval_profile),
    )


def _native_totals(batches):
    if not batches or not all(batches):
        return None
    totals = Counter()
    missing = set()
    for operations in batches:
        for profile in operations:
            operation = profile["operation"]
            if operation == "temporal":
                raw = profile["timings_ns"] | profile["stats"]
                fields = {
                    "sampling_ns": "traversal",
                    "retrieval_ns": "decode_and_select",
                    "aggregation_ns": "aggregation",
                    "rows_decoded": "rows_decoded",
                    "rows_examined": "rows_considered",
                }
            else:
                raw = profile
                fields = {key: key for key in NATIVE_COUNTERS}
                if profile.get("fused"):
                    fields["sample_and_retrieve_ns"] = fields.pop("sampling_ns")
                    del fields["retrieval_ns"]
                elif operation == "sample":
                    del fields["retrieval_ns"]
                elif operation == "fetch":
                    del fields["sampling_ns"]
            for key, field in fields.items():
                if field in raw:
                    totals[key] += raw[field]
                else:
                    missing.add(key)
    return {key: total for key, total in totals.items() if key not in missing}


def _history_events(output):
    if "sequence" in output:
        return int(output["sequence"]["valid"].sum())
    if "source_event_indices" not in output:  # Column histories have one row per event.
        return int(output["valid"].sum())
    indices = output["source_event_indices"]
    valid = output["valid"] & (indices >= 0)
    if indices.ndim == 2:
        owners = torch.arange(len(indices))[:, None].expand_as(indices)
    else:
        lengths = output["offsets"].diff()
        owners = torch.arange(len(lengths)).repeat_interleave(lengths)
    return torch.stack((owners[valid], indices[valid]), dim=1).unique(dim=0).size(0)


def _dimensions(task, batch, event_tables):
    tensors = [t for t in tree_leaves((batch.inputs, batch.targets)) if isinstance(t, torch.Tensor)]
    counts = Counter(
        padded=int(batch.metadata["is_padding"].sum()),
        output_elements=sum(t.numel() for t in tensors),
        output_bytes=sum(t.numel() * t.element_size() for t in tensors),
    )
    counts["samples"] = batch.batch_size - counts["padded"]
    event_nodes = 0
    for name, spec in task.contract.batch.inputs.items():
        output = batch.inputs[name]
        if isinstance(spec.representation, api.Graph):
            nodes = {table: len(node["ids"]) for table, node in output["nodes"].items()}
            counts["nodes"] += sum(nodes.values())
            counts["edges"] += sum(edge["index"].shape[1] for edge in output["edges"].values())
            event_nodes += sum(nodes[table] for table in nodes.keys() & event_tables)
        elif not isinstance(spec.representation.row, api.Columns):
            counts["tokens"] += int(output["valid"].sum())
        if isinstance(spec.source, api.EventSource) and spec.source.collection == "history":
            counts["events"] += _history_events(output)
    if event_tables and "nodes" in counts:
        counts.setdefault("events", event_nodes)
    return counts


def _rates(counts, elapsed):
    return {
        key + "_per_s": counts[key] * 1e9 / elapsed
        for key in ("samples", "tokens", "nodes", "edges")
        if key in counts
    }


def measure_epoch(task, make_batches, record, observer, *, split, batch_size, event_tables=()):
    """Count returned batches once. Terminal exhaustion contributes only to elapsed and wait."""
    install_native(observer)
    source = task._assembler._source
    requests, sizes, profiles, operations = [], [], [], []
    record.update(
        batch_sizes=sizes,
        request_ns=requests,
        native_profiles=profiles,
        native_operations=operations,
    )
    totals = Counter(samples=0, padded=0)
    epoch = record["epoch"]
    start = perf_counter_ns()
    batches = make_batches(
        task, split=split, epoch=epoch, batch_size=batch_size, shuffle=False, drop_last=False
    )
    record["startup_ns"] = perf_counter_ns() - start
    while True:
        previous = source.last_sampling_profile
        observer.native = []
        request = perf_counter_ns()
        try:
            batch = next(batches, None)
        except Exception as error:
            end = perf_counter_ns()
            failed_wait = end - request
            record.update(
                totals,
                error=f"{type(error).__name__}: {error}",
                batches=len(sizes),
                failed_request_ns=failed_wait,
                wait_ns=sum(requests) + failed_wait,
                elapsed_ns=end - start,
            )
            raise
        end = perf_counter_ns()
        if batch is None:
            break
        if not requests:
            first_batch = end - start
        elif len(requests) == 2:
            warmup_end = request
            warmup_counts = totals.copy()
        requests.append(end - request)
        sizes.append(batch.batch_size)
        profile = source.last_sampling_profile
        fresh = profile is not previous and profile is not None
        profiles.append(profile if fresh and all(k in profile for k in NATIVE_COUNTERS) else None)
        operations.append(observer.native)
        totals.update(_dimensions(task, batch, event_tables))

    elapsed = end - start
    terminal = end - request
    steady = None
    later = requests[2:]
    if len(later) >= 4:
        later_counts = {key: value - warmup_counts[key] for key, value in totals.items()}
        window = end - warmup_end
        steady = {
            "batches": len(later),
            **later_counts,
            "elapsed_ns": window,
            **_rates(later_counts, window),
            **_latencies(later),
            "cv": pstdev(later) / mean(later) if any(later) else None,
        }
    record |= {
        "complete": True,
        "batches": len(requests),
        **totals,
        **_rates(totals, elapsed),
        "native_profile": {k: sum(p[k] for p in profiles) for k in NATIVE_COUNTERS}
        if profiles and all(profiles)
        else None,
        "native_totals": _native_totals(operations),
        "profiled_batches": sum(map(bool, profiles)),
        "first_request_ns": requests[0] if requests else None,
        "first_batch_ns": first_batch if requests else None,
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
