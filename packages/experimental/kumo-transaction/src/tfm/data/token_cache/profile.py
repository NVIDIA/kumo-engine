# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project a token-cache build from a sample of its blocks, on the node that will run it.

The whole book is planned; one block per worker is built at once, in a fresh process
each: the largest block, the first, the last, and evenly spaced ones between. Each
phase's time per block is fitted as a fixed cost plus a cost per row, and every
planned block, at its own size, is handed in order to the first free worker, as the
build's pool does. A spill of each sampled block is written to scratch and read back, to compare reading the book
once (spill, then read the spill) with reading it twice (decode again). The spill holds
11 bytes per field, 16 per event, and each block's distinct values once.
"""

from __future__ import annotations

import copy
import functools
import heapq
import os
import resource
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from tfm.contracts.errors import SchemaError
from tfm.contracts.sources import BlockPlan, ColumnarSource, PlannedBlock
from tfm.contracts.tokenizers import ColumnarTokenizer
from tfm.data.token_cache.build import (
    MAX_PARTS,
    CacheRequest,
    FitSpec,
    encode_part,
    fit_statistics_block,
)
from tfm.data.token_cache.store import open_file_budget

_CHUNK = 1 << 24


def _peak_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _spill(path: Path, n: int) -> tuple[float, float]:
    """Seconds to write `n` bytes to `path` with `fsync`, and to read them back."""
    chunk = np.random.default_rng(0).integers(0, 256, min(n, _CHUNK), np.uint8).tobytes()
    started = time.perf_counter()
    with open(path, "wb") as file:
        for at in range(0, n, _CHUNK):
            file.write(chunk[: min(_CHUNK, n - at)])
        file.flush()
        os.fsync(file.fileno())
    written = time.perf_counter() - started
    if sys.platform == "linux":
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    started = time.perf_counter()
    with open(path, "rb") as file:
        while file.read(_CHUNK):
            pass
    read = time.perf_counter() - started
    path.unlink()
    return written, read


def _fit_sample(
    source: ColumnarSource[Any],
    block: PlannedBlock,
    fit: FitSpec,
    users: Collection[str] | None,
    tokenizer: ColumnarTokenizer[Any],
    profiles: Path | None,
    scratch: Path,
    amount_field: str,
) -> dict[str, Any]:
    """Decode, fit and spill one block, timed."""
    entered = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        data = source.read_block(block, users=users)
        decode = time.perf_counter() - started
    events, fields = data.n_events, len(data.pair_event)
    # Each distinct value once: its UTF-8 bytes, a 4-byte offset and its 8-byte number.
    dictionary = len("".join(data.values).encode()) + 12 * len(data.values)
    started = time.perf_counter()
    statistics = fit_statistics_block(data, tokenizer, fit, profiles, amount_field).statistics
    fit_s = time.perf_counter() - started
    del data
    spill = 11 * fields + 16 * events + dictionary
    written, read = _spill(scratch / f"spill-{block.index:05d}", spill)
    return {
        "block": block.index,
        "entered": entered,
        "events": events,
        "fields": fields,
        "decode_s": decode,
        "fit_s": fit_s,
        "spill_bytes": spill,
        "dictionary_bytes": dictionary,
        "spill_write_s": written,
        "spill_read_s": read,
        "unsorted": bool(caught),
        "peak_bytes": _peak_bytes(),
        "statistics": statistics,
    }


def _encode_sample(
    source: ColumnarSource[Any],
    block: PlannedBlock,
    tokenizer_type: type[ColumnarTokenizer[Any]],
    tokenizer_state: Mapping[str, Any],
    users: Collection[str] | None,
    scratch: Path,
    amount_field: str,
) -> dict[str, Any]:
    """Encode one block into its own part file, timed."""
    started = time.perf_counter()
    encode_part(
        source,
        [block],
        tokenizer_type,
        tokenizer_state,
        users,
        str(scratch),
        block.index,
        "0" * 32,
        amount_field,
    )
    return {
        "encode_s": time.perf_counter() - started,
        "cache_bytes": (scratch / "parts" / f"{block.index:05d}.bin").stat().st_size,
        "peak_bytes": _peak_bytes(),
    }


def fit_block_seconds(rows: ArrayLike, seconds: ArrayLike) -> tuple[float, float]:
    """`(fixed, per_row)` with `seconds ~ fixed + per_row * rows`, by least squares.

    Through the origin when the rows are all equal or either fitted term is negative.
    """
    x, y = np.asarray(rows, np.float64), np.asarray(seconds, np.float64)
    if np.ptp(x) > 0:
        per_row, fixed = np.polyfit(x, y, 1)
        if fixed >= 0 and per_row >= 0:
            return float(fixed), float(per_row)
    return 0.0, float(y.sum() / x.sum())


def makespan(tasks: Sequence[float], workers: int) -> float:
    """Seconds until `tasks` finish when each, in order, goes to the first free worker."""
    free = [0.0] * max(1, min(workers, len(tasks)))
    for t in tasks:
        heapq.heapreplace(free, free[0] + t)
    return max(free)


def _sample_blocks(plan: BlockPlan[Any], k: int) -> list[int]:
    """`k` block indices: the largest, the first, the last, then evenly spaced ones."""
    n = len(plan.blocks)
    if n <= k:
        return list(range(n))
    largest = int(np.argmax([b.n_rows for b in plan.blocks]))
    spaced = np.linspace(0, n - 1, k + 2).round().astype(int).tolist()
    picks = dict.fromkeys([largest, 0, n - 1, *spaced, *range(n)])
    return sorted(list(picks)[:k])


def _available_bytes() -> int:
    if sys.platform == "linux":
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def profile_token_cache(
    request: CacheRequest,
    *,
    workers: int = 8,
    rows_per_block: int = 1_000_000,
    scratch: Path | None = None,
) -> dict[str, Any]:
    """Project `build_token_cache(request, ..., workers, rows_per_block)` on this node.

    Spill files and sampled parts go to a temporary directory under `scratch` (the
    system's temporary directory by default), removed afterwards.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}.")
    users = None if request.users is None else frozenset(request.users)
    started = time.perf_counter()
    plan = request.source.plan(rows_per_block=rows_per_block, users=users)
    plan_s = time.perf_counter() - started
    if not plan.blocks:
        raise SchemaError(f"{request.source} has no events to cache.")
    picks = [plan.blocks[i] for i in _sample_blocks(plan, workers)]
    work = Path(tempfile.mkdtemp(prefix="tfm-profile-", dir=scratch))
    try:
        (work / "parts").mkdir()
        (work / "users").mkdir()
        profiles: Path | None = work / "profiles.parquet"
        started = time.perf_counter()
        if not request.source.write_profiles(users, work / "profiles.parquet"):
            profiles = None
        profiles_s = time.perf_counter() - started

        def pool() -> ProcessPoolExecutor:
            return ProcessPoolExecutor(
                max_workers=len(picks), mp_context=get_context("spawn"), max_tasks_per_child=1
            )

        unfitted = copy.deepcopy(request.tokenizer)
        pool_started = time.time()
        with pool() as run:
            fits = list(
                run.map(
                    _fit_sample,
                    repeat(request.source),
                    picks,
                    repeat(request.fit),
                    repeat(users),
                    repeat(unfitted),
                    repeat(profiles),
                    repeat(work),
                    repeat(request.amount_field),
                )
            )
        tokenizer = copy.deepcopy(request.tokenizer).fit_stats(
            functools.reduce(unfitted.merge, (f.pop("statistics") for f in fits))
        )
        state = tokenizer.state_dict()
        with pool() as run:
            encodes = list(
                run.map(
                    _encode_sample,
                    repeat(request.source),
                    picks,
                    repeat(type(tokenizer)),
                    repeat(state),
                    repeat(users),
                    repeat(work),
                    repeat(request.amount_field),
                )
            )
        scratch_free = shutil.disk_usage(work).free
    finally:
        shutil.rmtree(work)
    return _project(
        request,
        plan,
        fits,
        encodes,
        workers,
        rows_per_block,
        plan_s,
        profiles_s,
        min(f["entered"] for f in fits) - pool_started,
        scratch_free,
        tokenizer.vocab_size,
    )


def _project(
    request: CacheRequest,
    plan: BlockPlan[Any],
    fits: list[dict[str, Any]],
    encodes: list[dict[str, Any]],
    workers: int,
    rows_per_block: int,
    plan_s: float,
    profiles_s: float,
    start_s: float,
    scratch_free: int,
    vocab_size: int,
) -> dict[str, Any]:
    events = np.array([f["events"] for f in fits], np.float64)
    scale = plan.n_rows / events.sum()
    rows = [plan.blocks[f["block"]].n_rows for f in fits]
    block_rows = np.array([b.n_rows for b in plan.blocks], np.float64)
    runs = np.array_split(np.arange(len(plan.blocks)), min(len(plan.blocks), MAX_PARTS))
    decode = np.array([f["decode_s"] for f in fits])
    fit = np.array([f["fit_s"] for f in fits]) + decode
    encode = np.array([e["encode_s"] for e in encodes])
    write = np.array([f["spill_write_s"] for f in fits])
    read = np.array([f["spill_read_s"] for f in fits])
    spill = write + read

    def wall(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
        """Both passes scheduled over the plan; the range scales by the sample's worst fits."""
        lines = [fit_block_seconds(rows, s) for s in (first, second)]
        fitted = [a + b * np.asarray(rows) for a, b in lines]
        spread = (first + second) / (fitted[0] + fitted[1])
        (a1, b1), (a2, b2) = lines
        per_run = [float((a2 + b2 * block_rows[r]).sum()) for r in runs]
        passes = makespan(list(a1 + b1 * block_rows), workers) + makespan(per_run, workers)
        serial = plan_s + profiles_s + start_s
        return {
            "projected": round(float(serial + passes), 1),
            "low": round(float(serial + passes * min(1.0, spread.min())), 1),
            "high": round(float(serial + passes * max(1.0, spread.max())), 1),
        }

    spill_needed = int(sum(f["spill_bytes"] for f in fits) * scale)
    ratio = float(spill.sum() / decode.sum())
    peak = max(r["peak_bytes"] for r in fits + encodes)
    parts = min(len(plan.blocks), MAX_PARTS)
    budget = open_file_budget()
    once_pays = ratio < 1 and spill_needed <= scratch_free
    return {
        "identity": request.identity(),
        "workers": workers,
        "rows_per_block": rows_per_block,
        "book": {"events": plan.n_rows, "users": plan.n_users, "blocks": len(plan.blocks)},
        "sample": {"blocks": [f["block"] for f in fits], "events": int(events.sum())},
        "seconds": {
            "plan": round(plan_s, 1),
            "profiles": round(profiles_s, 1),
            "worker_start": round(start_s, 1),
            "read_twice": wall(fit, encode),
            "read_once": wall(fit + write, encode - decode + read),
        },
        "spill": {
            "bytes_needed": spill_needed,
            "dictionary_bytes": int(sum(f["dictionary_bytes"] for f in fits) * scale),
            "scratch_free": scratch_free,
            "spill_over_redecode": round(ratio, 2),
            "per_block": [round(float(r), 2) for r in spill / decode],
        },
        "cache": {
            "bytes": int(sum(e["cache_bytes"] for e in encodes) * scale),
            "parts": parts,
            "sample_vocab_size": vocab_size,
            "open_file_budget": budget,
        },
        "memory": {
            "worker_peak_bytes": peak,
            "all_workers_bytes": peak * workers,
            "available_bytes": _available_bytes(),
        },
        "user_ids": {
            "longest_bytes": plan.longest_user_id,
            "array_bytes": plan.n_users * max(plan.longest_user_id, 1),
        },
        "checks": {"blocks_with_unsorted_time": sum(f["unsorted"] for f in fits)},
        "read_once_suggested": once_pays,
    }


def format_profile(report: Mapping[str, Any]) -> str:
    """`report` as lines for a terminal."""
    gb = 1e9
    book, sec, spill = report["book"], report["seconds"], report["spill"]
    cache, memory, ids = report["cache"], report["memory"], report["user_ids"]
    twice, once = sec["read_twice"], sec["read_once"]
    lines = [
        f"book: {book['events']:,} events, {book['users']:,} users, {book['blocks']:,} blocks "
        f"of up to {report['rows_per_block']:,} rows; sampled {len(report['sample']['blocks'])} "
        f"blocks, {report['sample']['events']:,} events, {report['workers']} workers",
        f"build, reading the book twice (the build's mode): {twice['projected']:,.0f} s "
        f"(range {twice['low']:,.0f}-{twice['high']:,.0f} s), of which planning "
        f"{sec['plan']:,.1f} s, copying profiles {sec['profiles']:,.1f} s and starting workers "
        f"{sec['worker_start']:,.1f} s",
        f"reading it once instead: {once['projected']:,.0f} s; spill time / decode time "
        f"{spill['spill_over_redecode']:.2f} (below 1 favours reading once); scratch needed "
        f"{spill['bytes_needed'] / gb:,.1f} GB ({spill['dictionary_bytes'] / spill['bytes_needed']:.0%} "
        f"of it distinct values), free {spill['scratch_free'] / gb:,.1f} GB",
        f"cache: {cache['bytes'] / gb:,.2f} GB in {cache['parts']} part files; vocabulary "
        f"fitted on the sample: {cache['sample_vocab_size']:,} (the whole book's is at least that)",
        f"memory: {memory['worker_peak_bytes'] / gb:.2f} GB per worker at peak, "
        f"{memory['all_workers_bytes'] / gb:,.1f} GB for all workers, "
        f"{memory['available_bytes'] / gb:,.1f} GB available",
        f"user ids: longest {ids['longest_bytes']} bytes; every id is stored at that length, "
        f"{ids['array_bytes'] / gb:,.2f} GB for all users",
    ]
    budget = cache["open_file_budget"]
    if cache["parts"] > budget:
        lines.append(
            f"open files: {cache['parts']} part files exceed this shell's budget of {budget}; "
            f"raise the limit to {4 * cache['parts']} (`ulimit -n {4 * cache['parts']}`) so "
            "readers memory-map them"
        )
    if memory["all_workers_bytes"] > memory["available_bytes"]:
        lines.append(
            "memory: the workers' peaks exceed available memory; lower --workers or --rows-per-block"
        )
    unsorted = report["checks"]["blocks_with_unsorted_time"]
    if unsorted:
        lines.append(
            f"order: {unsorted} sampled blocks hold events not sorted by (user_id, ts); the build "
            "sorts them in memory, which costs time. Sort the book by (user_id, ts) to skip this"
        )
    lines.append(
        "suggestion: "
        + (
            "reading once would be faster here, but the build only reads twice; this is a "
            "measurement to report, not a mode to pick"
            if report["read_once_suggested"]
            else "read twice, as the build does"
        )
    )
    lines += [
        "cannot know: the vocabulary is fitted on the sample only, so cache size and encode "
        "time are estimates (the whole book has more distinct values)",
        "cannot know: a cloud volume with burst credits reads faster in a short run than in "
        "a long build",
        "cannot know: the range spans the sampled blocks' differences, not the variation "
        "between repeated builds",
        "not projected: merging the users' table at the end of the build, which grows with users",
    ]
    if sys.platform != "linux":
        lines.append("cannot know: off Linux the spill's read-back may come from memory")
    return "\n".join(lines)


__all__ = ["fit_block_seconds", "format_profile", "makespan", "profile_token_cache"]
