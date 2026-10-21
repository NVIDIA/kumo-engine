"""End-to-end Kumo Engine benchmark for one scale-bench rung.

    <kumo venv>/bin/python kumo/run_e2e.py --graph fraud --data <dir or s3://.../fraud/1B> \
        --scratch <local NVMe dir> --out runs/fraud-1B [--monitor] \
        [--stages snapshot,tt,fit,pt,predict] [--epochs 1 --max-steps 1000] \
        [--hops 2 --fanout 16,8] [--predict-limit 2000000]

Stages: snapshot (graph.snapshot), tt (generate_training_table), fit (suggest_model_plan + fit),
pt (generate_prediction_table), predict. Prerequisites of a requested stage run implicitly
(a snapshot with unchanged data is reused). Every stage prints `MARK <stage> <seconds> ...`
and <out>/results.json is rewritten after each stage.
"""
import argparse
import json
import os
import re
import resource
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = None
STAGES = ["snapshot", "tt", "fit", "pt", "predict"]
NEEDS = {"tt": ["snapshot"], "fit": ["snapshot", "tt"], "pt": ["snapshot"],
         "predict": ["snapshot", "tt", "fit", "pt"]}


# ---------------------------------------------------------------- monitor (separate process)

def _meminfo():
    out = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0]) * 1024
    return out


def _proc_tree(root):
    children = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = open(f"/proc/{d}/stat").read()
        except OSError:
            continue
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        children.setdefault(ppid, []).append(int(d))
    tree, todo = [], [root]
    while todo:
        p = todo.pop()
        tree.append(p)
        todo.extend(children.get(p, []))
    return tree


def _rss(pid):
    try:
        return int(open(f"/proc/{pid}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        return 0


def _pss(pid):
    try:
        for line in open(f"/proc/{pid}/smaps_rollup"):
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _block_device(path):
    st = os.stat(path)
    try:
        name = Path(os.path.realpath(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}")).name
    except OSError:
        return None
    return name


def _diskstats(dev):
    for line in open("/proc/diskstats"):
        f = line.split()
        if f[2] == dev:
            return {"rd_sect": int(f[5]), "wr_sect": int(f[9]), "io_ms": int(f[12]), "t": time.time()}
    return None


def _gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10).stdout
        rows = [list(map(float, r.split(","))) for r in out.strip().splitlines()]
        return {"gpu_util": max(r[0] for r in rows), "gpu_mem_mib": sum(r[1] for r in rows)}
    except Exception:
        return {}


def _du(path):
    try:
        out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, timeout=300).stdout
        return int(out.split()[0])
    except Exception:
        return None


def monitor_main(pid, out, scratch, interval, heavy_interval):
    out, scratch = Path(out), Path(scratch)
    dev = _block_device(scratch)
    prev = _diskstats(dev) if dev else None
    last_heavy = 0.0
    with open(out / "monitor.jsonl", "a") as f:
        while os.path.exists(f"/proc/{pid}") and not (out / ".monitor_stop").exists():
            now = time.time()
            m = _meminfo()
            stage = (out / ".stage").read_text().strip() if (out / ".stage").exists() else ""
            tree = _proc_tree(pid)
            usage = shutil.disk_usage(scratch)
            s = {"t": now, "stage": stage,
                 "mem_available": m["MemAvailable"], "anon": m["AnonPages"], "cached": m["Cached"],
                 "mapped": m["Mapped"], "dirty": m["Dirty"], "shmem": m["Shmem"],
                 "main_rss": _rss(pid), "tree_rss": sum(_rss(p) for p in tree), "nprocs": len(tree),
                 "scratch_fs_used": usage.used, **_gpu()}
            cur = _diskstats(dev) if dev else None
            if prev and cur and cur["t"] > prev["t"]:
                dt = cur["t"] - prev["t"]
                s["read_mbps"] = (cur["rd_sect"] - prev["rd_sect"]) * 512 / dt / 1e6
                s["write_mbps"] = (cur["wr_sect"] - prev["wr_sect"]) * 512 / dt / 1e6
                s["disk_util"] = min(100.0, (cur["io_ms"] - prev["io_ms"]) / dt / 10)
            prev = cur
            if now - last_heavy >= heavy_interval:
                last_heavy = now
                s["main_pss"] = _pss(pid)
                s["tree_pss"] = sum(_pss(p) for p in tree)
                s["scratch_du"] = _du(scratch)
            f.write(json.dumps(s) + "\n")
            f.flush()
            time.sleep(max(0.0, interval - (time.time() - now)))


def summarize_monitor(path, mem_total):
    rows = [json.loads(l) for l in open(path)] if path.exists() else []
    if not rows:
        return {}

    def peaks(rs):
        def mx(k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return max(v) if v else None

        def mean(k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return round(sum(v) / len(v), 1) if v else None
        avail = [r["mem_available"] for r in rs]
        return {"samples": len(rs), "seconds": round(rs[-1]["t"] - rs[0]["t"], 1),
                "box_used_gib_peak": round((mem_total - min(avail)) / 2**30, 2),
                "mem_available_gib_min": round(min(avail) / 2**30, 2),
                **{f"{k}_gib_peak": (round(mx(k) / 2**30, 2) if mx(k) is not None else None)
                   for k in ("anon", "cached", "mapped", "main_rss", "tree_rss", "main_pss",
                             "tree_pss", "scratch_du")},
                "scratch_fs_growth_gib_peak": round((mx("scratch_fs_used") - rows[0]["scratch_fs_used"]) / 2**30, 2),
                "read_mbps_peak": mx("read_mbps"), "write_mbps_peak": mx("write_mbps"),
                "read_mbps_mean": mean("read_mbps"), "disk_util_mean": mean("disk_util"),
                "gpu_util_mean": mean("gpu_util"), "gpu_util_peak": mx("gpu_util"),
                "gpu_mem_gib_peak": round(mx("gpu_mem_mib") / 1024, 2) if mx("gpu_mem_mib") else None,
                "nprocs_peak": mx("nprocs")}
    out = {"overall": peaks(rows), "by_stage": {}}
    for st in dict.fromkeys(r["stage"] for r in rows):
        out["by_stage"][st] = peaks([r for r in rows if r["stage"] == st])
    return out


# ---------------------------------------------------------------- log tee + progress-bar parser

_BAR = re.compile(r"^(Epoch \d+|Validation|Sanity Checking|Testing|Predicting)[^|]*\|.*?\|\s*(\d+)/(\d+|\?)\s*"
                  r"\[(\d+:)?(\d+):(\d+)<[^,]*,\s*([\d.]+)(it/s|s/it)")


def _secs(h, m, s):
    return (int(h[:-1]) if h else 0) * 3600 + int(m) * 60 + int(s)


class Tee:
    """Writes through to the real stream and the run log; parses Lightning progress bars."""

    def __init__(self, stream, log, state):
        self.stream, self.log, self.state, self.buf = stream, log, state, ""

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)
        self.buf += data
        *parts, self.buf = re.split(r"[\r\n]", self.buf)
        for p in parts:
            self._parse(p.strip())
        return len(data)

    def _parse(self, line):
        if not line:
            return
        if "Traceback" in line or "Error" in line or "rror:" in line:
            errs = self.state.setdefault("log_errors", [])
            if len(errs) < 60 and line not in errs:
                errs.append(line[:400])
        if re.search(r"SIGTERM|SIGKILL|worker.*(shut ?down|did not exit|terminat)", line, re.I):
            ev = self.state.setdefault("worker_shutdown", [])
            if len(ev) < 200:
                ev.append({"t": round(time.time(), 1), "line": line[:300]})
        if "reducing the batch size by half" in line:
            self.state["oom_halvings"] = self.state.get("oom_halvings", 0) + 1
            self.state.pop("epochs", None)  # the failed attempt's progress bars don't count
        m = _BAR.match(line)
        if not m:
            return
        kind = m.group(1)
        if kind.startswith("Epoch"):
            ep = int(kind.split()[1])
            cur = self.state.setdefault("epochs", {}).setdefault(ep, {"in_val": False})
            n, el = int(m.group(2)), _secs(m.group(4), m.group(5), m.group(6))
            rate = float(m.group(7)) if m.group(8) == "it/s" else 1 / max(float(m.group(7)), 1e-9)
            if not cur["in_val"]:
                cur.update(train_steps=n, train_elapsed_s=el, train_bar_rate=rate)
            cur.update(total_steps=n, total_elapsed_s=el, bar_rate=rate)
            self.state["current_epoch"] = ep
        elif kind == "Validation" and "current_epoch" in self.state:
            self.state["epochs"][self.state["current_epoch"]]["in_val"] = True
        elif kind == "Predicting":
            self.state["predict_bar"] = {"steps": int(m.group(2)), "elapsed_s": _secs(m.group(4), m.group(5), m.group(6)),
                                         "rate": float(m.group(7)) if m.group(8) == "it/s" else 1 / float(m.group(7))}

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.stream.fileno()

    def __getattr__(self, name):
        return getattr(self.stream, name)


# ---------------------------------------------------------------- sampled-subgraph statistics

def _pcts(a):
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return None
    p50, p90, p99 = np.percentile(a, [50, 90, 99])
    return {"mean": round(float(a.mean()), 2), "p50": float(p50), "p90": float(p90), "p99": float(p99),
            "max": float(a.max())}


def _bitmap(maps, key, size):
    import numpy as np
    bm = maps.get(key)
    if bm is None or bm.size < size:
        maps[key] = np.zeros(size, dtype=bool)
        if bm is not None:
            maps[key][:bm.size] = bm
    return maps[key]


class SubgraphStats:
    """Per-seed sizes, within-seed duplicates and distinct node coverage of sampled batches.

    Reads the HeteroData each step receives: per node type `n_id` (global node ids), `batch`
    (which seed tree a node belongs to) and `num_sampled_nodes` (per-hop counts, nodes ordered by
    hop); per edge type `edge_index` and `num_sampled_edges`. Per-seed sizes come from `batch`.
    """

    def __init__(self, limit, node_counts, seed_type):
        self.limit, self.node_counts, self.seed_type = limit, dict(node_counts or {}), seed_type
        self.phases = {}

    def _phase(self, name):
        return self.phases.setdefault(name, {"batches": 0, "seeds": [], "cols": {}, "nodes_hop": {},
                                             "edges_hop": {}, "totals": {}, "back_to_seed": 0,
                                             "seen": {}, "seen_hop": {}, "curve": [], "t_in": [], "t_out": [],
                                             "gpu_peak": 0, "structure": None})

    def collect(self, phase, batch, per_seed=True):
        import numpy as np
        ph = self._phase(phase)
        t_in = time.perf_counter()
        ph["t_in"].append(t_in)
        if ph["batches"] >= self.limit and per_seed:
            ph["t_out"].append(time.perf_counter())
            return
        h = batch
        ntypes = [t for t in h.node_types if "n_id" in h[t]]
        S = 0
        for t in ntypes:
            if "batch" in h[t] and h[t].batch.numel():
                S = max(S, int(h[t].batch.max()) + 1)
        if ph["structure"] is None:
            ph["structure"] = {"node_types": ntypes, "edge_types": ["__".join(e) for e in h.edge_types]}
        rows = {}
        seed_gid = None
        if self.seed_type in ntypes and "batch_size" in h[self.seed_type]:
            seed_gid = h[self.seed_type].n_id[:int(h[self.seed_type].batch_size)].cpu().numpy()
        for t in ntypes:
            nid = h[t].n_id.cpu().numpy().astype(np.int64)
            if nid.size == 0:
                continue
            nsn = [int(x) for x in getattr(h[t], "num_sampled_nodes", [nid.size])]
            hop = np.repeat(np.arange(len(nsn)), nsn) if sum(nsn) == nid.size else np.zeros(nid.size, np.int64)
            for k, c in enumerate(np.bincount(hop)):
                ph["nodes_hop"].setdefault(t, {}).setdefault(k, 0)
                ph["nodes_hop"][t][k] += int(c)
            need = max(int(nid.max()) + 1, self.node_counts.get(t) or 0)
            _bitmap(ph["seen"], t, need)[nid] = True
            for k in np.unique(hop):
                _bitmap(ph["seen_hop"].setdefault(t, {}), int(k), need)[nid[hop == k]] = True
            tot = ph["totals"].setdefault(t, {"nodes": 0, "dup": 0})
            tot["nodes"] += int(nid.size)
            if not per_seed or "batch" not in h[t] or S == 0:
                continue
            b = h[t].batch.cpu().numpy().astype(np.int64)
            tot["dup"] += int(nid.size - np.unique(b * need + nid).size)
            if t == self.seed_type and seed_gid is not None:
                m = hop > 0
                if b.max(initial=-1) < seed_gid.size:
                    ph["back_to_seed"] += int((nid[m] == seed_gid[b[m]]).sum())
            H = len(nsn)
            mat = np.bincount(b * H + hop, minlength=S * H).reshape(S, H)
            for k in range(H):
                rows[f"nodes/{t}/hop{k}"] = mat[:, k]
            rows[f"nodes/{t}"] = mat.sum(1)
        for et in h.edge_types:
            if "edge_index" not in h[et]:
                continue
            orig = getattr(h[et], "_orig_edge_index", None)
            ei = orig if orig is not None else h[et].edge_index
            name = "__".join(et)
            nse = [int(x) for x in getattr(h[et], "num_sampled_edges", [ei.shape[1]])]
            for k, c in enumerate(nse):
                ph["edges_hop"].setdefault(name, {}).setdefault(k, 0)
                ph["edges_hop"][name][k] += c
            if per_seed and ei.numel() and "batch" in h[et[2]] and S:
                rows[f"edges/{name}"] = np.bincount(h[et[2]].batch[ei[1]].cpu().numpy(), minlength=S)
        if per_seed and S:
            rows["nodes/_all"] = sum((v for k, v in rows.items() if k.startswith("nodes/") and k.count("/") == 1),
                                      np.zeros(S, np.int64))
            rows["edges/_all"] = sum((v for k, v in rows.items() if k.startswith("edges/")), np.zeros(S, np.int64))
            for k, v in rows.items():
                ph["cols"].setdefault(k, []).append((len(ph["seeds"]), v))
            ph["seeds"].append(S)
        ph["batches"] += 1
        if ph["batches"] in (1, 10, 25, 50, 100, 200, 500, 1000, 2000) or (ph["batches"] % 1000 == 0):
            ph["curve"].append({"batches": ph["batches"], **{t: int(m.sum()) for t, m in ph["seen"].items()}})
        ph["t_out"].append(time.perf_counter())

    def summary(self):
        import numpy as np
        out = {}
        for name, ph in self.phases.items():
            nb = ph["batches"]
            if not nb:
                continue
            seeds = ph["seeds"]
            per_seed = {}
            for k, lst in sorted(ph["cols"].items()):
                arrs = [np.zeros(s, np.int64) for s in seeds]
                for i, v in lst:
                    arrs[i] = v
                per_seed[k] = _pcts(np.concatenate(arrs))
            curve = ph["curve"]
            if not curve or curve[-1]["batches"] != nb:
                curve.append({"batches": nb, **{t: int(m.sum()) for t, m in ph["seen"].items()}})
            cov = {}
            for t, m in ph["seen"].items():
                total = self.node_counts.get(t) or m.size
                d = int(m.sum())
                c = {"distinct": d, "node_count": total, "fraction": round(d / total, 6)}
                prev = [p for p in curve if p["batches"] <= nb // 2]
                if prev and nb > prev[-1]["batches"]:
                    rate = (d - prev[-1].get(t, 0)) / (nb - prev[-1]["batches"])
                    c["proj_1000_batches_linear_upper"] = round(min(1.0, (d + rate * max(0, 1000 - nb)) / total), 6)
                cov[t] = c
            gaps = [ph["t_in"][i + 1] - ph["t_out"][i] for i in range(min(len(ph["t_in"]) - 1, len(ph["t_out"])))]
            gaps = gaps[5:] if len(gaps) > 20 else gaps
            spent = [b - a for a, b in zip(ph["t_in"], ph["t_out"])]
            totals = {t: dict(v, dup_share=round(v["dup"] / v["nodes"], 6) if v["nodes"] else None)
                      for t, v in ph["totals"].items()}
            out[name] = {
                "batches": nb, "seeds": int(sum(seeds)), "seeds_per_batch": _pcts(seeds),
                "per_seed": per_seed,
                "batch_nodes_by_type_hop": {t: {f"hop{k}": round(v / nb, 1) for k, v in sorted(d.items())}
                                            for t, d in ph["nodes_hop"].items()},
                "batch_edges_by_type_hop": {t: {f"hop{k}": round(v / nb, 1) for k, v in sorted(d.items())}
                                            for t, d in ph["edges_hop"].items()},
                "duplicates": totals,
                "dup_share_all": round(sum(v["dup"] for v in totals.values())
                                       / max(1, sum(v["nodes"] for v in totals.values())), 6),
                "back_to_seed_nodes_per_seed": round(ph["back_to_seed"] / max(1, sum(seeds)), 3),
                "coverage": cov, "coverage_curve": curve,
                "distinct_by_hop": {t: {f"hop{k}": int(m.sum()) for k, m in sorted(d.items())}
                                    for t, d in ph["seen_hop"].items()},
                "steps_timed": len(gaps),
                "its_excl_stats": round(1 / float(np.median(gaps)), 3) if gaps else None,
                "its_excl_stats_mean": round(len(gaps) / sum(gaps), 3) if gaps and sum(gaps) > 0 else None,
                "stats_overhead_s": round(sum(spent), 1),
                "gpu_peak_alloc_gib": round(ph["gpu_peak"] / 2**30, 2) if ph["gpu_peak"] else None,
                "structure": ph["structure"]}
        return out

    def callback(self):
        import pytorch_lightning as pl
        import torch
        stats = self

        class _CB(pl.Callback):
            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                stats.collect("fit", batch)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if torch.cuda.is_available():
                    ph = stats._phase("fit")
                    ph["gpu_peak"] = max(ph["gpu_peak"], torch.cuda.max_memory_allocated())

            def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
                stats.collect("predict", batch, per_seed=stats._phase("predict")["batches"] < stats.limit)
        return _CB()

    def install(self):
        import kumo_core.train.diskgraph_nested_trainer as dnt
        cls, cb = dnt.DiskgraphNestedTrainer, self.callback()
        orig = cls.create_pl_trainer

        def create_pl_trainer(trainer_self, *a, **k):
            r = orig(trainer_self, *a, **k)
            trainer_self.pl_trainer.callbacks.append(cb)
            return r
        cls.create_pl_trainer = create_pl_trainer


# ---------------------------------------------------------------- main

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True, choices=["commerce", "fraud"])
    ap.add_argument("--rung", default=None, help="rung name, e.g. 100M, 1B, 50B (reads <data-root>/<graph>/<rung>)")
    ap.add_argument("--data-root", default=DATA_ROOT, help="dataset root, required with --rung")
    ap.add_argument("--data", default=None, help="explicit rung directory (local path or s3://...); overrides --rung")
    ap.add_argument("--scratch", required=True, help="local NVMe scratch root (KUMO_HOME)")
    ap.add_argument("--out", required=True, help="run directory for results.json, run.log, monitor.jsonl")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--s3-region", default=None)
    ap.add_argument("--run-mode", default="fast", choices=["debug", "fast", "normal", "best"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1000, help="max train steps per epoch")
    ap.add_argument("--max-val-steps", type=int, default=None, help="default: max(50, max_steps // 10)")
    ap.add_argument("--max-test-steps", type=int, default=None, help="default: same as val")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8, help="loader workers; 0 keeps DiskGraph auto_config")
    ap.add_argument("--hops", default="3",
                    help="sampling hops (default 3); 0 = no graph (entity features only); "
                         "'suggested' keeps suggest_model_plan's neighbours")
    ap.add_argument("--fanout", default="64,12,6,4,4,4", help="neighbours per hop, comma list (first --hops used)")
    ap.add_argument("--eval-seeds", default=None,
                    help="Parquet (local or s3://) of test seeds: customer_id, anchor, target; after each predict, "
                         "score them at their own anchor and run --eval-script on the top-k")
    ap.add_argument("--eval-script", default=None, help="evaluate.py: --seeds S --pred P --k K --json OUT")
    ap.add_argument("--also-fanout", action="append", default=[],
                    help="after the main fit and predict, fit and predict again on the same snapshot and training "
                         "table with this fanout (all hops); repeatable. Results go under stages['fit@hopsN:<fanout>']")
    ap.add_argument("--no-explain", action="store_true", help="skip post-fit explanations")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--predict-limit", type=int, default=None, help="score only the first N prediction rows")
    ap.add_argument("--predict-ids", default=None,
                    help="Parquet file or dir (local or s3://) whose first column lists the entity ids to score")
    ap.add_argument("--predict-sample-seed", type=int, default=None,
                    help="with --predict-limit: a seeded uniform sample of N rows instead of the first N")
    ap.add_argument("--predict-workers", type=int, default=1)
    ap.add_argument("--predict-loader-workers", type=int, default=None, help="loader workers per prediction scorer; defaults to --num-workers")
    ap.add_argument("--top-k", type=int, default=10, help="predictions per entity for link prediction")
    ap.add_argument("--threshold", type=float, default=0.5, help="binary classification threshold")
    ap.add_argument("--subgraph-stats", type=int, default=0, metavar="N",
                    help="record sampled-subgraph sizes per seed for N train batches (all predict batches for "
                         "coverage) into stages.fit/predict.subgraph_stats")
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--monitor-interval", type=float, default=10.0)
    ap.add_argument("--monitor-heavy-interval", type=float, default=60.0, help="PSS (smaps) and du cadence")
    ap.add_argument("--_monitor", nargs=5, help=argparse.SUPPRESS)
    return ap.parse_args()


def git_sha():
    try:
        sha = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", str(HERE)],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def read_manifest(data):
    try:
        if data.startswith("s3://"):
            from pyarrow import fs
            f, path = fs.FileSystem.from_uri(data.rstrip("/") + "/manifest.json")
            with f.open_input_stream(path) as h:
                return json.loads(h.read())
        return json.loads((Path(data) / "manifest.json").read_text())
    except Exception as e:
        print(f"no readable manifest.json under {data}: {e!r}", flush=True)
        return None


def limits():
    names = ["NOFILE", "NPROC", "MEMLOCK", "AS", "STACK", "CORE"]
    out = {n: list(resource.getrlimit(getattr(resource, f"RLIMIT_{n}"))) for n in names}
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        if os.path.exists(p):
            out["cgroup_memory_max"] = open(p).read().strip()
            break
    return out


def main():
    args = parse_args()
    if args._monitor:
        pid, out, scratch, iv, hv = args._monitor
        monitor_main(int(pid), out, scratch, float(iv), float(hv))
        return

    if args.data is None:
        if args.rung is None or args.data_root is None:
            sys.exit("pass --data, or both --rung and --data-root")
        args.data = f"{args.data_root.rstrip('/')}/{args.graph}/{args.rung}"
    out, scratch = Path(args.out).resolve(), Path(args.scratch).resolve()
    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not requested or set(requested) - set(STAGES):
        sys.exit(f"--stages must contain stages from {STAGES}")
    if out.exists() and any(out.iterdir()):
        sys.exit("--out must be a new or empty directory; keep prior results for comparison")
    if not args.data.startswith("s3://") and not Path(args.data).is_dir():
        sys.exit("--data must be an existing dataset directory or s3:// prefix")
    if args.epochs < 1 or args.max_steps < 1 or args.num_workers < 0 or args.predict_workers < 1:
        sys.exit("epochs, max-steps and predict-workers must be positive; num-workers must be nonnegative")
    if args.predict_loader_workers is not None and args.predict_loader_workers < 1:
        sys.exit("--predict-loader-workers must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        sys.exit("--batch-size must be positive")
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "tmp").mkdir(exist_ok=True)
    os.environ["KUMO_HOME"] = str(scratch)
    os.environ["TMPDIR"] = str(scratch / "tmp")
    is_s3 = args.data.startswith("s3://")

    state = {}
    log = open(out / "run.log", "a", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log, state), Tee(sys.stderr, log, state)

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = set(requested) - set(STAGES)
    if bad:
        sys.exit(f"unknown stages {sorted(bad)}; choose from {STAGES}")
    run = {s for r in requested for s in [r, *NEEDS.get(r, [])]}
    stages = [s for s in STAGES if s in run]

    for f in ("monitor.jsonl", ".monitor_stop", ".stage"):
        (out / f).unlink(missing_ok=True)
    mon = None
    if args.monitor:
        mon = subprocess.Popen([sys.executable, __file__, "--graph", args.graph, "--data", "-", "--scratch", "-",
                                "--out", "-", "--_monitor", str(os.getpid()), str(out), str(scratch),
                                str(args.monitor_interval), str(args.monitor_heavy_interval)])

    t_import = time.time()
    import kumo.engine as kumo  # raises the soft NOFILE limit
    from importlib.metadata import PackageNotFoundError, version

    def _version(dist):
        try:
            return version(dist)
        except PackageNotFoundError:
            return None
    t_import = time.time() - t_import

    results = {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")} | {"stages_run": stages},
        "env": {"host": socket.gethostname(), "python": sys.version.split()[0],
                "kumo_fat_bundle": _version("kumo-fat-bundle"), "diskgraph": _version("diskgraph"),
                "kumo_engine": kumo.__version__, "kumo_engine_sha": os.environ.get("KUMO_ENGINE_SHA"),
                "scale_bench_git_sha": git_sha(), "limits_after_import": limits(),
                "cpu_count": os.cpu_count(), "mem_total_gib": round(_meminfo()["MemTotal"] / 2**30, 1),
                "kumo_import_s": round(t_import, 1)},
        "stages": {}, "errors": [],
    }
    man = read_manifest(args.data)
    if man:
        results["data_manifest"] = {k: man.get(k) for k in ("generator_version", "git_sha", "graph", "rung",
                                                             "target_rows", "shards", "seed", "profile")}
    try:
        import torch
        results["env"]["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    print("ENV", json.dumps(results["env"]), flush=True)

    def save():
        results["log_errors"] = state.get("log_errors", [])
        results["worker_shutdown_log"] = state.get("worker_shutdown", [])
        tmp = out / "results.json.tmp"
        tmp.write_text(json.dumps(results, indent=1, default=str))
        tmp.replace(out / "results.json")

    def mark(stage, secs, **kv):
        print(f"MARK {stage} {secs:.1f} " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)

    kumo.configure(scratch_dir=str(scratch))
    sys.path.insert(0, str(HERE))
    from build_graph import commerce_graph, fraud_graph

    conn = (kumo.S3Connector(root_dir=args.data, region=args.s3_region) if is_s3
            else kumo.FileConnector(root_dir=str(Path(args.data).resolve())))
    staging = {}
    backend_cls = type(conn.source_backend)
    orig_prepare = backend_cls.prepare_ingest

    def timed_prepare(self, connector, table_name, *a, **k):
        t0 = time.time()
        r = orig_prepare(self, connector, table_name, *a, **k)
        staging[table_name] = round(time.time() - t0, 1)
        print(f"STAGING {table_name} {staging[table_name]}s", flush=True)
        return r
    backend_cls.prepare_ingest = timed_prepare

    if args.num_workers:
        import kumo_core.train.diskgraph_nested_trainer as dnt
        orig_cfg = dnt._ttng_diskgraph_runtime_config

        def pinned_cfg(runtime, meta, plan, batch_size):
            auto = orig_cfg(runtime, meta, plan, batch_size)
            if auto is None:
                return None
            predicting = (out / ".stage").read_text().startswith("predict")
            workers = args.predict_loader_workers if predicting and args.predict_loader_workers is not None else args.num_workers
            used = dict(auto, num_workers=workers)
            print(f"LOADER auto_config={auto} -> used={used}", flush=True)
            state.setdefault("loader", []).append({"auto": auto, "used": used})
            return used
        dnt._ttng_diskgraph_runtime_config = pinned_cfg

    import logging

    class _PlanCapture(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "sampling plan" in msg:
                state.setdefault("sampling_plans", []).append(msg[:20000])
                print("SAMPLING_PLAN", msg[:4000], flush=True)
    sampler_log = logging.getLogger("kumo_core.train.diskgraph_nested_trainer")
    sampler_log.setLevel(min(sampler_log.getEffectiveLevel(), logging.INFO))
    sampler_log.addHandler(_PlanCapture(level=logging.INFO))

    graph, query = (commerce_graph if args.graph == "commerce" else fraud_graph)(conn)
    pquery = kumo.PredictiveQuery(graph, query)
    pquery.validate()
    task_type = str(pquery.get_task_type())
    results["config"]["query"] = query
    results["config"]["task_type"] = task_type

    ctx = {}
    if args.subgraph_stats:
        ctx["subgraph_stats"] = SubgraphStats(args.subgraph_stats, {}, re.search(r"FOR EACH (\w+)\.", query).group(1))
        ctx["subgraph_stats"].install()
    tail = ["eval"] if args.eval_seeds and "fit" in stages else []
    runs = [(s, s, None) for s in stages + tail]
    if "fit" in stages:
        for fan in args.also_fanout:
            tag = f"hops{len(fan.split(','))}:{fan}"
            runs += [(f"{s}@{tag}", s, fan) for s in ("fit", "predict", *tail) if s in stages + tail]
    for stage, fn, fan in runs:
        if fan:
            args.hops, args.fanout = str(len(fan.split(","))), fan
        (out / ".stage").write_text(stage)
        t0 = time.time()
        try:
            info = STAGE_FNS[fn](args, kumo, graph, pquery, ctx, scratch, out, state, staging)
        except BaseException as e:
            secs = time.time() - t0
            traceback.print_exc()
            results["stages"][stage] = {"seconds": round(secs, 1), "ok": False}
            results["errors"].append({"stage": stage, "error": repr(e)[:2000],
                                      "traceback": traceback.format_exc()[-6000:]})
            mark(stage, secs, ok=False, error=type(e).__name__)
            save()
            break
        secs = time.time() - t0
        results["stages"][stage] = {"seconds": round(secs, 1), "ok": True, **info}
        mark(stage, secs, **{k: v for k, v in info.items() if not isinstance(v, (dict, list)) and "/" not in str(v)})
        save()

    (out / ".stage").write_text("done")
    if mon:
        (out / ".monitor_stop").touch()
        mon.wait(timeout=max(30, args.monitor_heavy_interval + 30))
        results["monitor"] = summarize_monitor(out / "monitor.jsonl", _meminfo()["MemTotal"])
    results["total_seconds"] = round(sum(s["seconds"] for s in results["stages"].values()), 1)
    save()
    print("RESULTS", out / "results.json", flush=True)
    sys.exit(1 if results["errors"] else 0)


# ---------------------------------------------------------------- stages

def _dir_bytes(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def stage_snapshot(args, kumo, graph, pquery, ctx, scratch, out, state, staging):
    before = {p.parent for p in (scratch / "snapshots").glob("*/*/_SUCCESS")}
    snap = graph.snapshot()
    ctx["snapshot"] = snap
    idx = Path(snap.index_path)
    stats = json.loads((idx / "ingest_stats.json").read_text()).get("stats", {})
    by_dir = {}
    for f in idx.rglob("*"):
        if f.is_file():
            top = f.relative_to(idx).parts
            key = "/".join(top[:2]) if top[0] == "nodes" else top[0]
            by_dir[key] = by_dir.get(key, 0) + f.stat().st_size
    by_dir.pop("train_tables", None)
    index_bytes = sum(by_dir.values())
    src = stats.get("source_bytes_by_table", {})
    ctx["node_counts"] = stats.get("node_counts") or {}
    return {"reused": idx in before, "index_path": str(idx), "index_bytes": index_bytes,
            "index_gib": round(index_bytes / 2**30, 2), "nodes": stats.get("total_nodes"),
            "edges": stats.get("total_edges"), "source_rows": sum(stats.get("source_rows_by_table", {}).values()),
            "source_bytes_decoded": sum(src.values()),
            "bytes_per_edge": round(index_bytes / stats["total_edges"], 1) if stats.get("total_edges") else None,
            "node_counts": stats.get("node_counts"), "edge_counts": stats.get("edge_counts"),
            "staging_s": dict(staging), "perf": stats.get("perf"),
            "index_bytes_by_dir": dict(sorted(by_dir.items(), key=lambda kv: -kv[1])),
            "key_failures": {k: stats.get(k) for k in ("key_parse_failures", "key_lookup_failures",
                                                       "key_membership_failures", "pk_duplicate_total")}}


def stage_tt(args, kumo, graph, pquery, ctx, scratch, out, state, staging):
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    tt = pquery.generate_training_table(ctx["snapshot"])
    ctx["tt"] = tt
    info = {"path": str(tt.train_table_path), "required_stages": list(tt.required_stages)}
    d = ds.dataset(str(tt.train_table_path), format="parquet")
    info["rows"] = d.count_rows()
    info["columns"] = d.schema.names
    split_col = next((c for c in d.schema.names if c.upper() == "SPLIT"), None)
    if split_col:
        vc = pc.value_counts(d.to_table(columns=[split_col])[split_col]).to_pylist()
        info["rows_by_split"] = {str(v["values"]): v["counts"] for v in vc}
    tgt = next((c for c in d.schema.names if c.upper().startswith("TARGET")), None)
    if tgt and pquery.get_task_type().name.endswith("CLASSIFICATION"):
        col = d.to_table(columns=[tgt])[tgt]
        try:
            info["positive_rate"] = round(pc.mean(pc.cast(col, "float64")).as_py(), 5)
        except Exception:
            pass
    elif tgt:
        col = d.to_table(columns=[tgt])[tgt]
        try:
            info["mean_targets_per_row"] = round(pc.mean(pc.list_value_length(col)).as_py(), 3)
        except Exception:
            pass
    return info


def stage_fit(args, kumo, graph, pquery, ctx, scratch, out, state, staging):
    from kumoapi.model_plan import RunMode
    snap = ctx["snapshot"]
    plan = pquery.suggest_model_plan(snap, run_mode=RunMode(args.run_mode))
    suggested_nn = plan.neighbor_sampling.num_neighbors
    plan.training_job.num_experiments = 1
    plan.training_job.refit_trainval = False
    plan.training_job.refit_full = False
    plan.training_job.disable_explain = args.no_explain
    if args.seed is not None:
        plan.training_job.manual_seed = args.seed
    plan.optimization.max_epochs = args.epochs
    plan.optimization.max_steps_per_epoch = args.max_steps
    val = args.max_val_steps or max(50, args.max_steps // 10)
    plan.optimization.max_val_steps = val
    plan.optimization.max_test_steps = args.max_test_steps or val
    if args.batch_size:
        plan.optimization.batch_size = [args.batch_size]
    if args.hops != "suggested":
        hops, fan = int(args.hops), [int(x) for x in args.fanout.split(",")]
        if hops > len(fan):
            raise ValueError(f"--hops {hops} needs at least {hops} --fanout values")
        plan.neighbor_sampling.num_neighbors = [[0]] if hops == 0 else [fan[:hops]]
    print("MODEL_PLAN", plan, flush=True)
    (out / "model_plan.txt").write_text(str(plan))
    trainer = kumo.Trainer(model_plan=plan)
    ctx["trainer"] = trainer
    state.pop("epochs", None)
    state.pop("oom_halvings", None)
    state.pop("sampling_plans", None)
    t0 = time.time()
    sg = ctx.get("subgraph_stats")
    if sg:
        sg.node_counts = dict(ctx.get("node_counts") or {})
    res = trainer.fit(ctx["tt"], snapshot=snap)
    fit_s = time.time() - t0
    metrics = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.metrics.items()}
    epochs = {int(k): {kk: vv for kk, vv in v.items() if kk != "in_val"} for k, v in state.get("epochs", {}).items()}
    planned_bs = plan.optimization.batch_size
    halvings = state.get("oom_halvings", 0)
    bs = _trained_batch_size(scratch, t0)
    train_its = [e["train_steps"] / e["train_elapsed_s"] for e in epochs.values()
                 if e.get("train_elapsed_s") and e.get("train_steps")]
    # the final bar rate in each epoch's pre-validation phase is tqdm's smoothed train-only it/s
    info = {"metrics": metrics, "fit_s": round(fit_s, 1), "epochs": epochs,
            "train_its": round(sum(train_its) / len(train_its), 2) if train_its else None,
            "train_its_bar": [e.get("train_bar_rate") for e in epochs.values()],
            "batch_size": bs, "planned_batch_size": _compact(planned_bs), "cuda_oom_halvings": halvings,
            "num_neighbors": _compact(plan.neighbor_sampling.num_neighbors),
            "suggested_num_neighbors": _compact(suggested_nn),
            "effective_sampling_plan": (state.get("sampling_plans") or [None])[0], "loader": state.get("loader", [])[-1:]}
    if sg:
        info["subgraph_stats"] = sg.summary().get("fit")
    if info["train_its"] and bs:
        info["train_samples_per_s"] = round(info["train_its"] * bs, 1)
    for k in ("val_auroc", "test_auroc", "val_auprc", "test_auprc", "val_map@10", "test_map@10"):
        if k in metrics:
            info[k] = metrics[k]
    return info


def _trained_batch_size(scratch, since):
    """loader_kwargs.batch_size of the last trial config written by this fit (AutoML picks from the plan's list)."""
    cfgs = [p for p in (scratch / "trainers").glob("**/automl/version_*/config.json") if p.stat().st_mtime >= since]
    if not cfgs:
        return None
    cfg = json.loads(max(cfgs, key=lambda p: p.stat().st_mtime).read_text())
    return (cfg.get("loader_kwargs") or {}).get("batch_size")


def _compact(x):
    return re.sub(r"\s+", "", str(x))


def stage_pt(args, kumo, graph, pquery, ctx, scratch, out, state, staging):
    pt = pquery.generate_prediction_table(ctx["snapshot"])
    ctx["pt"] = pt
    return {"rows": pt.row_count, "timestamps": [str(t) for t in pt.timestamps][:4], "path": str(pt.table_path)}


def stage_predict(args, kumo, graph, pquery, ctx, scratch, out, state, staging):
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    pt, full = ctx["pt"], ctx["pt"].row_count
    info = {"population_rows": full}
    if args.predict_ids or (args.predict_limit and full > args.predict_limit):
        import numpy as np
        import pyarrow.compute as pc
        t0 = time.time()
        src = ds.dataset(str(pt.table_path), format="parquet")
        sub = scratch / "bp_subset" / f"run-{os.getpid()}"
        shutil.rmtree(sub, ignore_errors=True)
        (sub / "rows").mkdir(parents=True)
        cols = [c for c in src.schema.names if c in ("ENTITY1", "TIMESTAMP")]
        if args.predict_ids:
            ids = ds.dataset(args.predict_ids, format="parquet").to_table().column(0)
            ids = pc.cast(ids, src.schema.field("ENTITY1").type)
            rows = src.to_table(columns=cols, filter=pc.field("ENTITY1").isin(ids))
            info["predict_ids"] = {"path": args.predict_ids, "ids": len(ids), "matched_rows": rows.num_rows}
        elif args.predict_sample_seed is not None:
            rows = src.to_table(columns=cols)
            idx = np.sort(np.random.default_rng(args.predict_sample_seed).choice(rows.num_rows, args.predict_limit,
                                                                                 replace=False))
            rows = rows.take(idx)
            info["predict_sample_seed"] = args.predict_sample_seed
        else:
            rows = src.head(args.predict_limit, columns=cols)
        pq.write_table(rows, sub / "rows" / "part-0.parquet")
        pt = kumo.PredictionTable.from_source_table(kumo.FileConnector(root_dir=str(sub))["rows"],
                                                    pq=pquery, snapshot=ctx["snapshot"])
        info["subset_build_s"] = round(time.time() - t0, 1)
    name = pquery.get_task_type().name
    t0 = time.time()
    res = ctx["trainer"].predict(pt, snapshot=ctx["snapshot"], num_workers=args.predict_workers,
                                 binary_classification_threshold=args.threshold if name == "BINARY_CLASSIFICATION" else None,
                                 num_classes_to_return=args.top_k if "LINK" in name else None)
    secs = time.time() - t0
    rows = res.num_rows
    info.update(scored_rows=pt.row_count, predict_s=round(secs, 1), rows_per_s=round(pt.row_count / secs, 1),
                output_rows=rows, files=len(res.files), predict_bar=state.get("predict_bar"),
                full_population_est_s=round(full / (pt.row_count / secs), 1))
    if ctx.get("subgraph_stats"):
        info["subgraph_stats"] = ctx["subgraph_stats"].summary().get("predict")
    return info


def stage_eval(args, kumo, graph, pquery, ctx, scratch, out, state, staging):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    d = scratch / "eval" / f"run-{os.getpid()}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "rows").mkdir(parents=True)
    seeds = ds.dataset(args.eval_seeds, format="parquet").to_table()
    pq.write_table(seeds, d / "seeds.parquet")
    pt_type = ds.dataset(str(ctx["pt"].table_path), format="parquet").schema
    ts_type = pt_type.field("TIMESTAMP").type
    if pa.types.is_integer(ts_type):  # match the prediction table's integer epoch unit
        v = ds.dataset(str(ctx["pt"].table_path), format="parquet").head(1, columns=["TIMESTAMP"])[0][0].as_py()
        unit = next(u for u, lo in (("ns", 1e17), ("us", 1e14), ("ms", 1e11), ("s", 0)) if abs(v) >= lo)
        ts = pc.cast(pc.cast(seeds["anchor"], pa.timestamp(unit)), ts_type)
    else:
        ts = pc.cast(seeds["anchor"], ts_type)
    rows = pa.table({"ENTITY1": pc.cast(seeds["customer_id"], pt_type.field("ENTITY1").type), "TIMESTAMP": ts})
    pq.write_table(rows, d / "rows" / "part-0.parquet")
    anchors = sorted({str(t) for t in pc.unique(rows["TIMESTAMP"]).to_pylist()})
    print(f"EVAL seeds={rows.num_rows} anchors={anchors[:4]}", flush=True)
    pt = kumo.PredictionTable.from_source_table(kumo.FileConnector(root_dir=str(d))["rows"],
                                                pq=pquery, snapshot=ctx["snapshot"])
    t0 = time.time()
    res = ctx["trainer"].predict(pt, snapshot=ctx["snapshot"], num_workers=args.predict_workers,
                                 num_classes_to_return=args.top_k)
    predict_s = time.time() - t0
    raw = ds.dataset(list(res.files), format="parquet").to_table()
    print(f"EVAL predict output schema: {raw.schema}", flush=True)
    names = {c.upper(): c for c in raw.column_names}
    ent = names.get("ENTITY") or names.get("ENTITY1") or raw.column_names[0]
    cls = next(c for c in raw.column_names if "CLASS" in c.upper())
    score = next((c for c in raw.column_names if "SCORE" in c.upper()), None)
    if pa.types.is_list(raw[cls].type):
        lens = pc.list_value_length(raw[cls])
        idx = pc.list_parent_indices(raw[cls])
        flat = {"customer_id": pc.take(raw[ent], idx), "merchant_id": pc.list_flatten(raw[cls])}
        if score and pa.types.is_list(raw[score].type):
            flat["score"] = pc.list_flatten(raw[score])
        else:
            starts = pc.take(pc.subtract(pc.cumulative_sum(lens), lens), idx)
            flat["rank"] = pc.add(pc.subtract(pa.array(range(len(idx)), pa.int64()), starts), 1)
        preds = pa.table(flat)
    else:
        preds = pa.table({"customer_id": raw[ent], "merchant_id": raw[cls], "score": raw[score]})
    preds = preds.set_column(0, "customer_id", pc.cast(preds["customer_id"], pa.int64()))
    preds = preds.set_column(1, "merchant_id", pc.cast(preds["merchant_id"], pa.int64()))
    pq.write_table(preds, d / "preds.parquet")
    r = subprocess.run([sys.executable, args.eval_script, "--seeds", str(d / "seeds.parquet"), "--pred",
                        str(d / "preds.parquet"), "--k", str(args.top_k), "--json", str(d / "eval.json")],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr, flush=True)
    r.check_returncode()
    return {"seeds": rows.num_rows, "anchors": anchors[:4], "predict_s": round(predict_s, 1),
            "pred_rows": preds.num_rows, "pred_schema": str(raw.schema)[:1000],
            **json.loads((d / "eval.json").read_text())}


STAGE_FNS = {"snapshot": stage_snapshot, "tt": stage_tt, "fit": stage_fit, "pt": stage_pt, "predict": stage_predict, "eval": stage_eval}

if __name__ == "__main__":
    main()
