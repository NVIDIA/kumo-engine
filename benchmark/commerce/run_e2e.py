# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Kumo Engine benchmark for one scale-bench rung.

    <kumo venv>/bin/python kumo/run_e2e.py --graph commerce --data <dir or s3://.../commerce/1B> \
        --scratch <local NVMe dir> --out runs/commerce-1B \
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
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = None
STAGES = ["snapshot", "tt", "fit", "pt", "predict"]
NEEDS = {"tt": ["snapshot"], "fit": ["snapshot", "tt"], "pt": ["snapshot"],
         "predict": ["snapshot", "tt", "fit", "pt"]}


def score_with_loader_workers(score_worker, loader_workers, **kwargs):
    """Apply the benchmark loader count inside a spawned prediction scorer."""
    import kumo_core.train.diskgraph_nested_trainer as dnt
    original = dnt._ttng_diskgraph_runtime_config

    def configured(*args, **options):
        auto = original(*args, **options)
        return None if auto is None else dict(auto, num_workers=loader_workers)

    dnt._ttng_diskgraph_runtime_config = configured
    try:
        return score_worker(**kwargs)
    finally:
        dnt._ttng_diskgraph_runtime_config = original


# ---------------------------------------------------------------- environment

def _meminfo():
    out = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0]) * 1024
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


# ---------------------------------------------------------------- main

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True, choices=["commerce"])
    ap.add_argument("--rung", default=None, help="rung name, e.g. 100M, 1B, 50B (reads <data-root>/<graph>/<rung>)")
    ap.add_argument("--data-root", default=DATA_ROOT, help="dataset root, required with --rung")
    ap.add_argument("--data", default=None, help="explicit rung directory (local path or s3://...); overrides --rung")
    ap.add_argument("--scratch", required=True, help="local NVMe scratch root (KUMO_HOME)")
    ap.add_argument("--out", required=True, help="run directory for results.json and run.log")
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

    run = {s for r in requested for s in [r, *NEEDS.get(r, [])]}
    stages = [s for s in STAGES if s in run]

    t_import = time.time()
    from importlib.metadata import PackageNotFoundError, version

    import kumo.engine as kumo  # raises the soft NOFILE limit

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
    from build_graph import commerce_graph

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

    if args.num_workers or args.predict_loader_workers is not None:
        import kumo_core.train.diskgraph_nested_trainer as dnt
        orig_cfg = dnt._ttng_diskgraph_runtime_config

        def pinned_cfg(runtime, meta, plan, batch_size):
            auto = orig_cfg(runtime, meta, plan, batch_size)
            if auto is None:
                return None
            predicting = (out / ".stage").read_text().startswith("predict")
            workers = args.predict_loader_workers if predicting and args.predict_loader_workers is not None else args.num_workers
            used = dict(auto, num_workers=workers) if workers else auto
            print(f"LOADER auto_config={auto} -> used={used}", flush=True)
            state.setdefault("loader", []).append({"auto": auto, "used": used})
            return used
        dnt._ttng_diskgraph_runtime_config = pinned_cfg

        from functools import partial

        import kumo_core.prediction as prediction
        original_run_workers = prediction._run_prediction_workers
        prediction_loaders = args.predict_loader_workers if args.predict_loader_workers is not None else args.num_workers

        def run_prediction_workers(worker_count, score_worker, worker_kwargs):
            scorer = partial(score_with_loader_workers, score_worker, prediction_loaders) if prediction_loaders else score_worker
            return original_run_workers(worker_count, scorer, worker_kwargs)

        prediction._run_prediction_workers = run_prediction_workers

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

    graph, query = commerce_graph(conn)
    pquery = kumo.PredictiveQuery(graph, query)
    pquery.validate()
    task_type = str(pquery.get_task_type())
    results["config"]["query"] = query
    results["config"]["task_type"] = task_type

    ctx = {}
    for stage in stages:
        (out / ".stage").write_text(stage)
        t0 = time.time()
        try:
            info = STAGE_FNS[stage](args, kumo, graph, pquery, ctx, scratch, out, state, staging)
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

    results["total_seconds"] = round(sum(s["seconds"] for s in results["stages"].values()), 1)
    save()
    print("RESULTS", out / "results.json", flush=True)
    sys.exit(1 if results["errors"] else 0)


# ---------------------------------------------------------------- stages

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
    return info


STAGE_FNS = {"snapshot": stage_snapshot, "tt": stage_tt, "fit": stage_fit, "pt": stage_pt, "predict": stage_predict}

if __name__ == "__main__":
    main()
