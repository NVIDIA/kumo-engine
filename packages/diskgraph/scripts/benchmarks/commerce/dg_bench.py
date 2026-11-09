# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DiskGraph neighbourhood context: TemporalMostRecent sampling plus raw feature fetch.

    map   --index I --seeds S --n N --out seedmap.parquet         customer_id -> global id for the seeds
    dump  --index I --seedmap M --hops H --n N --out rows/         per-seed context rows (for correctness)
    rate  --index I --seedmap M --hops H --n N --procs P --threads T --batch B [--cold]   throughput
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def plan_json(hops):
    return json.dumps({"layers": [{"fanouts": {}, "default_fanout": {
        "fanout": f, "mode": "temporal_most_recent", "absent_key_policy": "uniform_fallback"}}
        for f in C.plan(hops)]})


def engine(index, threads):
    from diskgraph import SamplerEngine
    return SamplerEngine(index, random_seed=0, num_threads=threads)


def sample(eng, names, gids, anchors, pj):
    """Returns {type_name: (seed_index, global_id, hop)} and the stats dict."""
    from diskgraph import TimeUnit
    parts, stats = {}, None
    for it in eng.sample({"customers": gids.tolist()}, pj, False,
                         seed_timestamps={"customers": anchors.tolist()}, seed_timestamp_unit=TimeUnit.Seconds):
        if it["type"] == "node":
            parts.setdefault(it["type_id"], []).append(it["batch"])
        elif it["type"] == "stats":
            stats = json.loads(it["stats"])
    out = {}
    for tid, bs in parts.items():
        t = pa.Table.from_batches(bs)
        si = t["id_a"].to_numpy().astype(np.int64)
        gid = t["id_b"].to_numpy().astype(np.int64)
        counts = stats["nodes_per_type"][tid]
        hop = np.repeat(np.arange(len(counts)), counts)
        assert len(hop) == len(gid), (names[tid], len(hop), len(gid))
        out[names[tid]] = (si, gid, hop)
    return out, stats


def fetch(eng, name, gids, cols=None):
    bs = [it["batch"] for it in eng.fetch_features([{"node_type": name, "node_ids": gids.tolist(),
                                                     "feature_names": cols or []}])]
    return pa.Table.from_batches(bs) if bs else None


def type_names(eng):
    return json.loads(eng.get_type_mapping())["node_types"]


def cmd_map(a):
    eng = engine(a.index, a.threads)
    meta = json.loads((Path(a.index) / "index_metadata.json").read_text())
    n_cust = next(t["max_id"] for t in meta["node_types"] if t["name"] == "customers") + 1
    cid, anchor = C.load_seeds(a.seeds, a.n)
    want = pa.array(cid)
    found_g, found_c = [], []
    step = 20_000_000
    t0 = time.time()
    for lo in range(0, n_cust, step):
        g = np.arange(lo, min(lo + step, n_cust), dtype=np.int64)
        t = fetch(eng, "customers", g, ["customer_id"])
        m = pc.is_in(t["customer_id"], value_set=want).to_numpy(zero_copy_only=False)
        found_g.append(g[m])
        found_c.append(t["customer_id"].to_numpy()[m])
        print(f"map {lo + len(g)}/{n_cust} {time.time() - t0:.0f}s", flush=True)
    fg, fc = np.concatenate(found_g), np.concatenate(found_c)
    order = dict(zip(fc.tolist(), fg.tolist()))
    gid = np.array([order[c] for c in cid.tolist()], dtype=np.int64)
    pq.write_table(pa.table({"customer_id": cid, "anchor_s": anchor, "gid": gid}), a.out)
    print(f"mapped {len(gid)} seeds in {time.time() - t0:.0f}s")


def cmd_dump(a):
    eng = engine(a.index, a.threads)
    names = type_names(eng)
    sm = pq.read_table(a.seedmap).slice(0, a.n)
    cid, anchor, gid = (sm[c].to_numpy() for c in ["customer_id", "anchor_s", "gid"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pj = plan_json(a.hops)
    acc = {}
    for lo in range(0, len(cid), a.batch):
        sl = slice(lo, lo + a.batch)
        nodes, _ = sample(eng, names, gid[sl], anchor[sl], pj)
        for name, (si, g, hop) in nodes.items():
            t = fetch(eng, name, g, C.KEY[name])
            t = t.append_column("seed", pa.array(cid[sl][si])).append_column("hop", pa.array(hop.astype(np.int8)))
            acc.setdefault(name, []).append(t)
    for name, ts in acc.items():
        pq.write_table(pa.concat_tables(ts), out / f"{name}.parquet")
    print(json.dumps({name: sum(t.num_rows for t in ts) for name, ts in acc.items()}))


def worker(a, slot, gid, anchor, q, go):
    eng = engine(a.index, a.threads)
    names = type_names(eng)
    pj = plan_json(a.hops)
    q.put(("ready", slot))
    go.wait()
    r0, ru0 = C.proc_read_bytes(), C.rusage()
    t0 = time.perf_counter()
    times, nodes, rows = [], 0, 0
    for lo in range(0, len(gid), a.batch):
        sl = slice(lo, lo + a.batch)
        out, stats = sample(eng, names, gid[sl], anchor[sl], pj)
        nodes += stats["total_nodes"]
        if not a.no_features:
            for name, (_, g, _) in out.items():
                t = fetch(eng, name, g)
                rows += t.num_rows if t is not None else 0
        times.append(time.perf_counter() - t0)
        if times[-1] > a.cap_s:
            break
    ru1 = C.rusage()
    q.put(("done", slot, {"seeds": min(len(gid), len(times) * a.batch), "times": times, "nodes": nodes, "rows": rows,
                          "read_bytes": C.proc_read_bytes() - r0, "cpu_s": ru1["utime"] + ru1["stime"] - ru0["utime"] - ru0["stime"],
                          "maxrss_kb": ru1["maxrss_kb"]}))


def qget(q, ps):
    """Queue get that fails instead of hanging when a worker died."""
    import queue
    while True:
        try:
            return q.get(timeout=10)
        except queue.Empty:
            dead = [p.exitcode for p in ps if p.exitcode not in (None, 0)]
            if dead:
                for p in ps:
                    p.kill()
                raise SystemExit(f"worker died with exit code {dead[0]}")


def cmd_rate(a):
    sm = C.tile(pq.read_table(a.seedmap), a.offset + a.n).slice(a.offset, a.n)
    gid, anchor = sm["gid"].to_numpy(), sm["anchor_s"].to_numpy()
    for rep, kind in enumerate(["cold", "warm"] if a.cold else ["warm"]):
        if kind == "cold":
            if a.drop_all:
                C.drop_all_caches()
            else:
                C.evict([a.index])
        ctx = mp.get_context("spawn")
        q, go = ctx.Queue(), ctx.Event()
        chunks = np.array_split(np.arange(len(gid)), a.procs)
        ps = [ctx.Process(target=worker, args=(a, i, gid[c], anchor[c], q, go)) for i, c in enumerate(chunks)]
        for p in ps:
            p.start()
        for _ in ps:
            assert qget(q, ps)[0] == "ready"
        t0 = time.perf_counter()
        go.set()
        res = [qget(q, ps)[2] for _ in ps]
        wall = time.perf_counter() - t0
        for p in ps:
            p.join()
        seeds = sum(r["seeds"] for r in res)
        first = min(r["times"][0] for r in res if r["times"])
        # Steady state: seeds after each worker's first batch over the time after it.
        steady_seeds = sum(r["seeds"] - min(a.batch, r["seeds"]) for r in res)
        last = max(r["times"][-1] for r in res if r["times"])
        steady_t = np.mean([r["times"][-1] - r["times"][0] for r in res if len(r["times"]) > 1] or [np.nan])
        out = {"system": "diskgraph", "label": a.label, "cache": kind, "hops": a.hops, "plan": C.plan(a.hops),
               "seeds": seeds, "procs": a.procs, "threads": a.threads, "batch": a.batch,
               "features": not a.no_features, "wall_s": round(wall, 3), "seeds_per_s": round(seeds / last, 2),
               "status": "ok" if seeds == len(gid) else "capped",
               "steady_seeds_per_s": round(steady_seeds / steady_t, 2) if steady_t == steady_t and steady_t > 0 else None,
               "first_batch_s": round(first, 3), "nodes_per_seed": round(sum(r["nodes"] for r in res) / seeds, 1),
               "rows": sum(r["rows"] for r in res), "read_mb_s": round(sum(r["read_bytes"] for r in res) / wall / 2**20, 1),
               "read_gb": round(sum(r["read_bytes"] for r in res) / 2**30, 2),
               "cpu_util": round(sum(r["cpu_s"] for r in res) / wall / os.cpu_count(), 3),
               "cores_busy": round(sum(r["cpu_s"] for r in res) / wall, 1),
               "peak_rss_gb_sum": round(sum(r["maxrss_kb"] for r in res) / 2**20, 2)}
        print("RESULT " + json.dumps(out), flush=True)
        if a.results:
            with open(a.results, "a") as f:
                f.write(json.dumps(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["map", "dump", "rate"])
    ap.add_argument("--index", required=True)
    ap.add_argument("--seeds")
    ap.add_argument("--seedmap")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cold", action="store_true", help="run a cold pass (cache dropped) and then a warm pass")
    ap.add_argument("--drop-all", action="store_true", help="drop the whole page cache (EC2 box)")
    ap.add_argument("--no-features", action="store_true")
    ap.add_argument("--cap-s", type=float, default=1800)
    ap.add_argument("--label", default="")
    ap.add_argument("--results")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not 1 <= a.hops <= len(C.PLAN6) or min(a.n, a.batch, a.procs, a.threads) < 1 or a.offset < 0 or a.cap_s <= 0:
        ap.error("hops must be 1 to 6; n, batch, procs, threads and cap-s must be positive; offset must be nonnegative")
    if not Path(a.index, "index_metadata.json").is_file():
        ap.error("--index must contain index_metadata.json from a completed snapshot")
    if a.cmd == "map" and (not a.seeds or not a.out):
        ap.error("map requires --seeds and --out")
    if a.cmd in ("rate", "dump") and not a.seedmap:
        ap.error("rate and dump require --seedmap")
    if a.cmd == "dump" and not a.out:
        ap.error("dump requires --out")
    {"map": cmd_map, "dump": cmd_dump, "rate": cmd_rate}[a.cmd](a)


if __name__ == "__main__":
    main()
