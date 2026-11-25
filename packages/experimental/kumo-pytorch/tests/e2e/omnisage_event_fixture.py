# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt timestamped actor–item interactions to the single-device OmniSage benchmark.

Each input row supplies integer actor and item IDs and an integer UTC Unix second. IDs
are numbered independently by first appearance in stable chronological order. All events
contribute to first-observed actor–item edges. For sequences, keep the first source row
per actor and second: this avoids fabricating an order inside a simultaneous basket.
Consecutive distinct items form undirected edges as of their first observed transition.
The graph freezes at the 35th percentile of source event seconds, after degree-based
pruning of the edges observed by that cutoff. All training sequences start strictly after
the snapshot, and every logged label comes from an actual later interaction. Simulated
visual/text arrays let the benchmark keep the model recipe on corpora without media.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa

from kumo.pytorch import api
from tests.e2e.omnisage_paper_data import (
    PaperFixture,
    _at,
    _paper_schema,
    _selector,
    _source,
    _vectors,
)


def _read_events(path, limit):
    with np.load(path, allow_pickle=False) as data:
        columns = [data[k] for k in ("actor", "item", "time")]
    if any(col.ndim != 1 or col.dtype.kind not in "iu" for col in columns):
        raise ValueError("actor, item and time must be one-dimensional integer arrays")
    actor, item, time = columns
    if not len(time) or len(actor) != len(time) or len(item) != len(time):
        raise ValueError("expected nonempty actor, item and time arrays of equal length")
    if any(col.dtype.kind == "u" and int(col.max()) > np.iinfo(np.int64).max for col in columns):
        raise ValueError("source integer columns must fit in int64")
    time = time.astype(np.int64, copy=False)
    order = np.argsort(time, kind="stable")
    clock = time[order]
    if limit is not None:
        if limit < 1:
            raise ValueError("max_events must be positive or None")
        if len(order) > limit:
            end = np.searchsorted(clock, clock[limit], side="left")
            if not end:
                raise ValueError("max_events splits the first timestamp group; increase the cap")
            order = order[:end]
    stats = {
        "raw_events": len(time),
        "raw_actors": len(np.unique(actor)),
        "raw_items": len(np.unique(item)),
        "used_events": len(order),
        "event_cap": limit,
        "source_start_second": int(clock[0]),
        "source_end_second": int(clock[-1]),
        "used_start_second": int(clock[0]),
        "used_end_second": int(time[order[-1]]),
    }
    return actor[order], item[order], time[order], stats


def _catalog(values, time):
    _, first, inverse = np.unique(values, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(len(first), dtype=np.int64)
    remap[order] = np.arange(len(first))
    return remap[inverse], time[first[order]]


def _first_edges(left, right, times):
    order = np.argsort(times, kind="stable")
    _, first = np.unique(np.column_stack((left[order], right[order])), axis=0, return_index=True)
    i = order[np.sort(first)]
    return left[i], right[i], times[i]


def _degrees(values):
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def _snapshot(pb, pp, at, npins, nboards, seed):
    """Algorithm 2 on the graph visible at this snapshot, with independent endpoint draws."""
    pb = tuple(col[pb[2] <= at] for col in pb)
    pp = tuple(col[pp[2] <= at] for col in pp)
    board, pin, _ = pb
    left, right, _ = pp

    def summarize(board, pin, left, right):
        board_degree = np.bincount(board, minlength=nboards)
        pin_degree = np.bincount(np.r_[pin, left, right], minlength=npins)
        return {
            "pb_edges": len(board),
            "pp_edges": len(left),
            "supported_pins": int(np.count_nonzero(pin_degree)),
            "supported_boards": int(np.count_nonzero(board_degree)),
            "snapshot_catalog_pins": npins,
            "snapshot_catalog_boards": nboards,
            "pb_density": len(board) / (npins * nboards),
            "pp_density": 2 * len(left) / max(1, npins * (npins - 1)),
            "board_degree": _degrees(board_degree),
            "pin_degree": _degrees(pin_degree),
        }

    before = summarize(board, pin, left, right)
    # An undirected edge is dropped if either endpoint prunes its incident copy.
    # Pin degree includes both the actor–item and consecutive item–item relations.
    board_degree = np.bincount(board, minlength=nboards)
    pin_degree = np.bincount(np.r_[pin, left, right], minlength=npins)
    rng = np.random.default_rng(seed)

    def retain(degree):
        target = np.clip(np.power(degree, 0.86), 10, 10_000)
        return rng.random(len(degree)) < np.minimum(target / degree, 1)

    pb_keep = retain(board_degree[board]) & retain(pin_degree[pin])
    pp_keep = retain(pin_degree[left]) & retain(pin_degree[right])
    pb = tuple(col[pb_keep] for col in pb)
    pp = tuple(col[pp_keep] for col in pp)
    after = summarize(pb[0], pb[1], pp[0], pp[1])
    return pb, pp, before, after


def _anchors(
    root,
    steps,
    batch_size,
    seed,
    client,
    clock,
    user,
    pin,
    t,
    src,
    next_idx,
    item_birth,
    graph_roots,
    future_window,
):
    # src and next_idx index consecutive timestamp groups for the same actor. A training
    # cutoff falls strictly before the positive; the item must already exist at this cutoff.
    pair_start = np.maximum.reduce(
        (t[src], t[next_idx] - future_window + 1, item_birth[pin[next_idx]])
    )
    pair_end = t[next_idx] - 1
    eligible = (pair_start <= pair_end) & (pin[src] != pin[next_idx])
    # Ensure at least two historical actions for each selected sequence.
    enough = (src > 0) & (user[src] == user[np.maximum(0, src - 1)])
    if not np.any(eligible & enough):
        raise ValueError(
            "no observed sequence has two past actions and a known future item in this window"
        )
    # Prefer cutoffs spread over the middle/late portion of the observed event clock.
    targets = np.quantile(clock, np.linspace(0.45, 0.85, steps), method="nearest")
    viable = np.unique(pair_end[eligible & enough])
    fallback = viable[np.linspace(0, len(viable) - 1, min(len(viable), 256), dtype=int)]
    sampled = {name: ([], [], []) for name in ("pins", "boards", "users")}
    random = np.random.default_rng(seed)
    cutoffs, eligible_counts = [], []

    def add(table, ids, at, positives=None):
        entities, clocks, labels = sampled[table]
        entities.extend(ids.tolist())
        clocks.extend([int(at)] * len(ids))
        if positives is None:
            labels.extend([[] for _ in ids])
        else:
            labels.extend([[{"pin": int(pk)}] for pk in positives])

    for target in targets:
        choices = np.r_[target, fallback[np.argsort(np.abs(fallback - target), kind="stable")]]
        for at in choices:
            mask = eligible & (pair_start <= at) & (at <= pair_end)
            pairs = np.flatnonzero(mask)
            if not len(pairs):
                continue
            # Future DrawRows uses all retained activity records in its open interval. Limit
            # user anchors to people whose possible positive items are all known at the cutoff.
            cold = user[(t > at) & (t < at + future_window) & (item_birth[pin] > at)]
            seq_rows = np.flatnonzero(mask & enough & ~np.isin(user[src], cold))
            if len(seq_rows):
                break
        else:
            raise ValueError("no common cutoff supports graph, engagement and sequence examples")
        cutoffs.append(int(at))
        eligible_counts.append(
            {"engagement_transitions": len(pairs), "sequence_users": len(seq_rows)}
        )
        for table in ("pins", "boards"):
            selected = random.choice(pairs, batch_size)
            ids = pin[src[selected]] if table == "pins" else user[src[selected]]
            add(table, ids, at, pin[next_idx[selected]])
            add(table, random.choice(graph_roots[table], batch_size), at)
        add("users", user[src[random.choice(seq_rows, batch_size)]], at)
    selectors = {
        name: _selector(root, name, entities, times, labels, client)
        for name, (entities, times, labels) in sampled.items()
    }
    return selectors, tuple(cutoffs), eligible_counts


def make_event_fixture(
    root,
    events_path,
    *,
    steps,
    batch_size,
    history_length,
    random_negatives,
    seed=42,
    client=None,
    max_events=100_000,
    future_window_seconds=None,
):
    """Write a bounded chronological interaction prefix as Parquet and return (fixture, stats)."""
    if min(steps, batch_size, random_negatives) < 1 or history_length < 2:
        raise ValueError("positive step and batch sizes and history length >=2 are required")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    raw_actor, raw_item, clock, stats = _read_events(events_path, max_events)
    actor, actor_birth = _catalog(raw_actor, clock)
    item, item_birth = _catalog(raw_item, clock)
    na, ni = len(actor_birth), len(item_birth)
    b, p, pb_time = _first_edges(actor, item, clock)

    # input was already stably ordered by time; preserve source order for ties.
    by_actor = np.argsort(actor, kind="stable")
    keep = np.r_[True, (np.diff(actor[by_actor]) != 0) | (np.diff(clock[by_actor]) != 0)]
    keep = by_actor[keep]
    user, pin, t = actor[keep], item[keep], clock[keep]
    src = np.flatnonzero(user[:-1] == user[1:])
    nxt = src + 1
    distinct = pin[src] != pin[nxt]
    first, second = np.minimum(pin[src], pin[nxt]), np.maximum(pin[src], pin[nxt])
    left, right, pp_time = _first_edges(first[distinct], second[distinct], t[nxt][distinct])
    snapshot = int(np.quantile(clock, 0.35, method="nearest"))
    pb, pp, before, after = _snapshot(
        (b, p, pb_time),
        (left, right, pp_time),
        snapshot,
        int(np.searchsorted(item_birth, snapshot, side="right")),
        int(np.searchsorted(actor_birth, snapshot, side="right")),
        seed,
    )
    b, p, pb_time = pb
    left, right, pp_time = pp
    graph_roots = {"pins": np.intersect1d(p, np.r_[left, right]), "boards": np.unique(b)}
    if any(not len(roots) for roots in graph_roots.values()):
        raise ValueError("the pruned historical snapshot needs both actor–item and item–item banks")
    post = t > snapshot
    user, pin, t = user[post], pin[post], t[post]
    src = np.flatnonzero(user[:-1] == user[1:])
    nxt = src + 1
    if future_window_seconds is None:
        future_window_seconds = 7 * 24 * 60 * 60
    if future_window_seconds < 2:
        raise ValueError(
            "future_window_seconds must exceed one second for an open integer interval"
        )
    selectors, cutoffs, eligible = _anchors(
        root,
        steps,
        batch_size,
        seed,
        client,
        clock,
        user,
        pin,
        t,
        src,
        nxt,
        item_birth,
        graph_roots,
        future_window_seconds,
    )
    boards, pins = np.arange(na), np.arange(ni)
    tables = {
        "pins": pa.table({"id": pins, "image": _vectors(pins, 1152, seed + 1)}),
        "boards": pa.table({"id": boards, "annotation": _vectors(boards, 128, seed + 2)}),
        "users": pa.table({"id": boards}),
        "membership": pa.table({"id": np.arange(len(b)), "board": b, "pin": p, "at": _at(pb_time)}),
        "related": pa.table(
            {"id": np.arange(len(left)), "src": left, "dst": right, "at": _at(pp_time)}
        ),
        "activity": pa.table(
            {
                "id": np.arange(len(t)),
                "user": user,
                "pin": pin,
                "second": t,
                "at": _at(t),
                "unit_weight": np.ones(len(t), dtype=np.int64),
            }
        ),
    }
    sources = {name: _source(root, name, table, client) for name, table in tables.items()}
    sizes = {"pins": ni, "boards": na, "users": na}
    stats.update(
        {
            "catalog": sizes,
            "actor_item_first_edges": len(b),
            "item_item_first_edges": len(left),
            "sequence_events": len(t),
            "unique_actor_seconds": len(keep),
            "snapshot_second": snapshot,
            "snapshot_before": before,
            "snapshot_after": after,
            "same_actor_second_rows_omitted_from_sequence": len(clock) - len(keep),
            "observed_transitions": len(src),
            "future_window_seconds": int(future_window_seconds),
            "cutoff_seconds": list(cutoffs),
            "eligible_per_step": eligible,
            "board_degree": _degrees(np.bincount(b, minlength=na)),
            "pin_degree": _degrees(np.bincount(np.r_[p, left, right], minlength=ni)),
            "pruning": {
                "alpha": 0.86,
                "min_degree": 10,
                "max_degree": 10000,
                "seed": seed,
                "sampling": "Bernoulli per endpoint; an undirected edge survives both draws.",
                "snapshot_quantile": 0.35,
            },
            "sequence_ties": "First row per actor/second for sequences. Membership uses all rows.",
            "features": "Generated 1024-dimensional visual and 128-dimensional text arrays.",
        }
    )
    fixture = PaperFixture(
        root,
        _paper_schema(),
        api.Dataset(sources),
        selectors,
        sizes,
        steps,
        batch_size,
        history_length,
        random_negatives,
        cutoffs,
        int(future_window_seconds),
        {"pins": item_birth, "boards": actor_birth},
    )
    return fixture, stats
