"""Shared schema, plans and seed handling for the neighbourhood-context benchmark."""
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
TABLES = ["customers", "merchants", "purchases", "merchant_views", "searches", "favorites", "reviews"]
EVENTS = ["purchases", "merchant_views", "searches", "favorites", "reviews"]
TIME = {"customers": "signup_time", "merchants": "created_time", "purchases": "purchase_time",
        "merchant_views": "view_time", "searches": "search_time", "favorites": "created_time",
        "reviews": "review_time"}
PK = {"customers": "customer_id", "merchants": "merchant_id"}
FKS = {e: (["customer_id", "merchant_id"] if e != "searches" else ["customer_id"]) for e in EVENTS}
FK_DST = {"customer_id": "customers", "merchant_id": "merchants"}
# Columns that identify a raw row.
KEY = {"customers": ["customer_id"], "merchants": ["merchant_id"], "purchases": ["purchase_id"],
       "merchant_views": ["customer_id", "merchant_id", "view_time", "source"],
       "searches": ["customer_id", "search_time", "query_term", "category", "result_page"],
       "favorites": ["customer_id", "merchant_id", "created_time"],
       "reviews": ["customer_id", "merchant_id", "review_time", "stars"]}
PLAN6 = [24, 8, 6, 8, 6, 8]


def plan(hops):
    return PLAN6[:hops]


def relations():
    """(src_type, dst_type, kind, column): every FK in both directions, as DiskGraph builds them."""
    out = []
    for e in EVENTS:
        for col in FKS[e]:
            out.append((FK_DST[col], e, "rev", col))
            out.append((e, FK_DST[col], "fk", col))
    return out


def load_seeds(path, n, offset=0):
    t = pq.read_table(path, columns=["customer_id", "anchor"])
    t = tile(t, offset + n).slice(offset, n)
    cid = t["customer_id"].to_numpy().astype(np.int64)
    anchor = t["anchor"].cast(pa.timestamp("s")).cast(pa.int64()).to_numpy().astype(np.int64)
    return cid, anchor


def tile(t, n):
    """Repeat a seed table when a run asks for more seeds than the rung has (small rungs only)."""
    if t.num_rows == 0:
        raise ValueError("seed table is empty; supply a nonempty evaluation seed file")
    return pa.concat_tables([t] * (-(-n // t.num_rows))) if n > t.num_rows else t


def rusage():
    r = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"utime": r.ru_utime + c.ru_utime, "stime": r.ru_stime + c.ru_stime,
            "maxrss_kb": max(r.ru_maxrss, c.ru_maxrss)}


def proc_read_bytes():
    try:
        return int(open("/proc/self/io").read().split("read_bytes: ")[1].split()[0])
    except Exception:
        return 0


def evict(paths):
    """Drop the page cache for every file under paths (per-file, safe on a shared box)."""
    n = 0
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else [Path(r) / f for r, _, fs in os.walk(p) for f in fs]
        for f in files:
            try:
                fd = os.open(f, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    return n


def drop_all_caches():
    """Whole-box drop (EC2 box only)."""
    os.sync()
    Path("/proc/sys/vm/drop_caches").write_text("3\n")


def rss_gb(pids):
    n = 0
    for pid in pids:
        try:
            n += int(open(f"/proc/{pid}/statm").read().split()[1])
        except OSError:
            pass
    return n * os.sysconf("SC_PAGE_SIZE") / 2**30


def qget(q, ps, rss_cap_gb=0, deadline=None):
    """Queue get that fails instead of hanging when a worker died, or kills the workers past an RSS cap.
    Past the deadline it kills the workers and returns None."""
    import queue
    while True:
        try:
            return q.get(timeout=2 if rss_cap_gb else 10)
        except queue.Empty:
            if deadline and time.perf_counter() > deadline:
                for p in ps:
                    p.kill()
                return None
            dead = [p.exitcode for p in ps if p.exitcode not in (None, 0)]
            rss = rss_gb([p.pid for p in ps]) if rss_cap_gb else 0
            if dead or rss > rss_cap_gb > 0:
                for p in ps:
                    p.kill()
                raise SystemExit(f"worker died with exit code {dead[0]}" if dead else f"workers' RSS {rss:.0f} GiB > cap")


def rate(a, worker, head, evict_paths):
    """Spawns a.procs workers over the seeds, then runs the cold (with --cold) and warm passes; all workers start
    each pass together. worker(a, slot, cid, anchor, passes, ready, done, gos) puts one dict per pass on done.
    Writes one RESULT line per pass."""
    import multiprocessing as mp
    cid, anchor = load_seeds(a.seeds, a.n, a.offset)
    passes = ["cold", "warm"] if a.cold else ["warm"]
    if a.cold:
        if a.drop_all:
            drop_all_caches()
        else:
            evict(evict_paths)
    ctx = mp.get_context("spawn")
    ready, done_q, gos = ctx.Queue(), ctx.Queue(), [ctx.Event() for _ in passes]
    chunks = np.array_split(np.arange(len(cid)), a.procs)
    ps = [ctx.Process(target=worker, args=(a, i, cid[c], anchor[c], passes, ready, done_q, gos)) for i, c in enumerate(chunks)]
    for p in ps:
        p.start()
    for i, kind in enumerate(passes):
        cap = getattr(a, "rss_cap_gb", 0)
        t0 = time.perf_counter()
        note = "killed at 2x cap-s; a batch had not finished"
        try:
            for _ in ps:
                qget(ready, ps, cap)
            t0 = time.perf_counter()
            gos[i].set()
            kill = t0 + 2 * a.cap_s if getattr(a, "hard_cap", False) else None
            res = [qget(done_q, ps, cap, kill) for _ in ps]
        except SystemExit as e:
            if not getattr(a, "catch", False):
                raise
            res, note = [None], str(e)
        wall = time.perf_counter() - t0
        if None in res:
            out = head | {"cache": kind, "hops": a.hops, "plan": plan(a.hops), "status": "did_not_finish", "seeds": 0,
                          "seeds_requested": len(cid), "procs": a.procs, "threads": a.threads, "batch": a.batch,
                          "wall_s": round(wall, 3), "seeds_per_s": 0.0, "first_batch_s": None,
                          "note": note}
            print("RESULT " + json.dumps(out), flush=True)
            if a.results:
                with open(a.results, "a") as f:
                    f.write(json.dumps(out) + "\n")
            return
        done = sum(r["seeds"] for r in res)
        bad = [r["status"] for r in res if r["status"] not in ("ok", "capped")]
        status = "ok" if all(r["status"] == "ok" for r in res) else (bad[0] if bad else "capped")
        last = max((r["times"][-1] for r in res if r["times"]), default=None)
        steady = [(r["seeds"] - min(a.batch, r["seeds"])) / (r["times"][-1] - r["times"][0])
                  for r in res if len(r["times"]) > 1]
        out = head | {"cache": kind, "hops": a.hops, "plan": plan(a.hops), "status": status, "seeds": done,
                      "seeds_requested": len(cid), "procs": a.procs, "threads": a.threads, "batch": a.batch,
                      "wall_s": round(wall, 3), "seeds_per_s": round(done / last, 3) if done else 0.0,
                      "steady_seeds_per_s": round(sum(steady), 3) if len(steady) == len(res) else None,
                      "first_batch_s": round(min(r["times"][0] for r in res if r["times"]), 3) if done else None,
                      "batches_done": sum(len(r["times"]) for r in res), "rows": sum(r["rows"] for r in res),
                      "read_gb": round(sum(r["read_bytes"] for r in res) / 2**30, 2),
                      "read_mb_s": round(sum(r["read_bytes"] for r in res) / wall / 2**20, 1),
                      "cpu_util": round(sum(r["cpu_s"] for r in res) / wall / os.cpu_count(), 3),
                      "cores_busy": round(sum(r["cpu_s"] for r in res) / wall, 1),
                      "peak_rss_gb_sum": round(sum(r["maxrss_kb"] for r in res) / 2**20, 2)}
        if "prepare_s" in res[0]:
            out["prepare_s"] = round(max(r["prepare_s"] for r in res), 2)
        print("RESULT " + json.dumps(out), flush=True)
        if a.results:
            with open(a.results, "a") as f:
                f.write(json.dumps(out) + "\n")
    for p in ps:
        p.join()


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


class Clock:
    def __init__(self):
        self.t0 = time.perf_counter()

    def __call__(self):
        return time.perf_counter() - self.t0
