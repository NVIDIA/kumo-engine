"""Generate one rung of one graph.

    python -m scalebench.generate run --graph commerce --rung 1B --root s3://bucket/prefix
    python -m scalebench.generate run ... --shard-range 0:400  # this machine's shards
    python -m scalebench.generate run ... --split 3/8          # or: every 8th unit
    python -m scalebench.generate merge --graph commerce --rung 1B --root ...

A rung is shards [0, k). Work units are (family, file group); a file group is a
fixed run of consecutive shards, so file contents do not depend on the worker
count or on how units are split across machines.
"""
import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from scalebench import profile as P
from scalebench import schema, storage
from scalebench.commerce import Commerce

GENERATORS = {"commerce": Commerce}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH = "_truth_"
STAMP = "_VALIDATED.json"


def parse_rows(s):
    s = s.strip().upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    return cfg


def effective_profile(cfg, graph, profile_override=None):
    path = profile_override or cfg["graphs"][graph]["profile"]
    path = path if os.path.isabs(path) else os.path.join(REPO, path)
    pert = cfg["perturbation"]
    prof = P.load(path)
    base = cfg["graphs"][graph].get("profile_base")
    if base:  # synthetic knobs and fields the real profile lacks come from the base
        prof = P.overlay(P.load(base if os.path.isabs(base) else os.path.join(REPO, base)), prof)
    prof = P.perturb(prof, pert["seed"], pert["min"], pert["max"])
    for key, cap in cfg.get("vocab_caps", {}).get(graph, {}).items():
        table, col = key.split(".")
        spec = prof["tables"][table]["columns"][col]
        multi = schema.kinds(graph, table)[col] == "multicategorical"
        field = ("multicat", "token_cardinality") if multi else ("categorical", "cardinality")
        spec[field[0]][field[1]] = min(int(spec[field[0]][field[1]]), int(cap))
        spec[field[0]]["capped"] = True  # fixed vocabulary at every rung (never per-shard)
    return prof, path


def make_generator(cfg, graph, prof):
    return GENERATORS[graph](prof, cfg["graphs"][graph], cfg["seed"])


def git_sha():
    try:
        sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                                        text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def versions():
    import numpy
    import scipy
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "pyarrow": pa.__version__,
            "scipy": scipy.__version__}


def table_path(base, graph, rung, table, part):
    if table.startswith(TRUTH):
        return f"{base}/_ground_truth/{graph}/{rung}/{table[len(TRUTH):]}/{part}"
    return f"{base}/{graph}/{rung}/{table}/{part}"


def parquet_bytes(table, files_cfg):
    buf = io.BytesIO()
    pq.write_table(table, buf, compression=files_cfg["compression"],
                   compression_level=files_cfg["compression_level"])
    return buf.tell()


def calibrate(gen, files_cfg):
    """Measures shard 0 to pick shards-per-file per family and rows-per-row-group per
    table. Shard 0 is identical in every rung and on every machine, so the layout is too."""
    layout = {}
    for fam, tables in gen.families.items():
        out = getattr(gen, "gen_" + fam)(0)
        conformed = {t: conform(gen.graph, t, out[t]) for t in tables}
        sizes = {t: max(parquet_bytes(conformed[t], files_cfg), 1) for t in tables}
        rows = {t: max(out[t].num_rows, 1) for t in tables}
        spf = max(1, int(files_cfg["target_bytes"] // max(sizes.values())))
        # row groups are sized by in-memory (decoded Arrow) bytes, which is what readers hold
        rg = {t: max(10_000, int(files_cfg["row_group_bytes"] * rows[t] / max(conformed[t].nbytes, 1)))
              for t in tables}
        layout[fam] = {"shards_per_file": spf, "row_group_rows": rg, "bytes_per_shard": sizes}
    return layout


def conform(graph, table, tbl):
    if table.startswith(TRUTH):
        return tbl
    sch = schema.arrow_schema(graph, table)
    return tbl.select(sch.names).cast(sch)


class Roller:
    """Writes one table of one file group: ~row_group_rows per row group, a new file
    (part-GGGGG-k) whenever the current one exceeds max_bytes."""

    def __init__(self, fs, base, graph, rung, table, group, rg_rows, files_cfg):
        self.fs, self.base, self.graph, self.rung, self.table = fs, base, graph, rung, table
        self.group, self.rg_rows, self.cfg = group, rg_rows, files_cfg
        self.pending, self.pending_rows, self.k = [], 0, 0
        self.files, self.writer = [], None

    def _open(self):
        self.buf = pa.BufferOutputStream()
        self.writer = pq.ParquetWriter(self.buf, self.schema, compression=self.cfg["compression"],
                                       compression_level=self.cfg["compression_level"])
        self.rows = 0

    def _close_file(self):
        self.writer.close()
        data = self.buf.getvalue()
        name = f"part-{self.group:05d}.parquet" if self.k == 0 else f"part-{self.group:05d}-{self.k}.parquet"
        path = table_path(self.base, self.graph, self.rung, self.table, name)
        storage.write_bytes(self.fs, path, data)
        self.files.append({"path": path[len(self.base) + 1:], "rows": self.rows, "bytes": data.size,
                           "sha256": hashlib.sha256(data).hexdigest()})
        self.writer, self.k = None, self.k + 1

    def _write(self, tbl):
        if self.writer is None:
            self._open()
        self.writer.write_table(tbl, row_group_size=self.rg_rows)
        self.rows += tbl.num_rows
        if self.buf.tell() >= self.cfg["max_bytes"]:
            self._close_file()

    def add(self, tbl):
        self.schema = tbl.schema
        self.pending.append(tbl)
        self.pending_rows += tbl.num_rows
        if self.pending_rows >= self.rg_rows:
            merged = pa.concat_tables(self.pending).combine_chunks()
            full = merged.num_rows // self.rg_rows * self.rg_rows
            self._write(merged.slice(0, full))
            rest = merged.slice(full)
            self.pending, self.pending_rows = [rest], rest.num_rows

    def close(self):
        rest = [t for t in self.pending if t.num_rows]
        if rest:
            self._write(pa.concat_tables(rest).combine_chunks())
        if self.writer is not None:
            self._close_file()
        return self.files


_STATE = {}


def _init(cfg, graph, prof, rung, root, layout, run_key):
    pa.set_cpu_count(2)
    _STATE.update(gen=make_generator(cfg, graph, prof), cfg=cfg, graph=graph, rung=rung,
                  root=root, layout=layout, run_key=run_key)


def unit_record_path(base, graph, rung, unit):
    return f"{manifest_dir(base, graph, rung)}/_units/{unit[0]}-{unit[1]:05d}.json"


def run_unit(unit):
    fam, group, s0, s1 = unit
    st = _STATE
    gen, lay = st["gen"], st["layout"][fam]
    fs, base = storage.filesystem(st["root"])
    t0 = time.time()
    rollers = {t: Roller(fs, base, st["graph"], st["rung"], t, group, lay["row_group_rows"][t],
                         st["cfg"]["files"]) for t in gen.families[fam]}
    for shard in range(s0, s1):
        out = getattr(gen, "gen_" + fam)(shard)
        for t, tbl in out.items():
            rollers[t].add(conform(st["graph"], t, tbl))
    files = {t: r.close() for t, r in rollers.items()}
    res = {"unit": [fam, group, s0, s1], "files": files, "seconds": time.time() - t0, "run_key": st["run_key"]}
    # recorded by the worker, so a unit that finished survives a failure elsewhere (--resume)
    storage.write_bytes(fs, unit_record_path(base, st["graph"], st["rung"], unit), json.dumps(res).encode())
    return res


def plan_units(gen, layout, shards):
    units = []
    for fam in gen.families:
        spf = layout[fam]["shards_per_file"]
        for g in range(math.ceil(shards / spf)):
            units.append((fam, g, g * spf, min((g + 1) * spf, shards)))
    cost = {u: layout[u[0]]["bytes_per_shard"][gen.families[u[0]][0]] * (u[3] - u[2]) for u in units}
    return sorted(units, key=lambda u: -cost[u])


def manifest_dir(base, graph, rung):
    return f"{base}/{graph}/{rung}"


def cmd_run(args):
    if args.workers < 1 or (args.shards is not None and args.shards < 1):
        sys.exit("workers and shards must be positive")
    if not args.shards and parse_rows(args.rung) < 1:
        sys.exit("rung must request a positive row count")
    cfg = load_config(args.config)
    prof, prof_path = effective_profile(cfg, args.graph, args.profile)
    gen = make_generator(cfg, args.graph, prof)
    per_shard = gen.expected_rows_per_shard()
    target = None if args.shards else parse_rows(args.rung)
    block = gen.block_shards  # a rung is a whole number of market blocks
    shards = args.shards or block * max(1, round(target / sum(per_shard.values()) / block))
    if shards % block:
        sys.exit(f"--shards must be a multiple of block_shards={block}")
    fs, base = storage.filesystem(args.root)
    stamp = f"{manifest_dir(base, args.graph, args.rung)}/{STAMP}"
    if storage.exists(fs, stamp):
        fs.delete_file(stamp)  # data is about to change; it must be validated again
    config_hash = P.digest({"config": cfg, "profile": prof, "graph": args.graph})
    sha = git_sha()
    # resume only into the same plan, code and libraries
    run_key = P.digest({"config_hash": config_hash, "shards": shards, "git_sha": sha, "versions": versions()})
    t0 = time.time()
    layout = calibrate(gen, cfg["files"])
    calib_s = time.time() - t0
    units = plan_units(gen, layout, shards)
    if args.shard_range:
        a, b = (int(x) for x in args.shard_range.split(":"))
        mine = [u for u in units if a <= u[2] < b]  # a unit belongs to the range holding its first shard
        tag = f"range-{a}-{b}"
    else:
        idx, n_split = (int(x) for x in args.split.split("/"))
        mine = units[idx::n_split]
        tag = f"split-{idx}-of-{n_split}"
    whole = len(mine) == len(units)
    print(f"{args.graph} {args.rung}: {shards} shards, {len(units)} units ({len(mine)} here), "
          f"~{sum(per_shard.values()) * shards / 1e9:.2f}B rows expected", flush=True)
    t1 = time.time()
    results = []
    results, todo = [], []
    for u in mine:
        done = unit_record_path(base, args.graph, args.rung, u)
        if args.resume and storage.exists(fs, done):
            res = json.loads(storage.read_bytes(fs, done))
            if res.get("run_key") == run_key:
                results.append(res)
                continue
        todo.append(u)
    if results:
        print(f"  resuming: {len(results)} units already done", flush=True)
    ctx = mp.get_context("spawn")
    # ProcessPoolExecutor fails loudly if a worker dies (e.g. OOM) instead of hanging;
    # on any failure, queued units are cancelled
    pool = ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init,
                               initargs=(cfg, args.graph, prof, args.rung, args.root, layout, run_key))
    try:
        futures = [pool.submit(run_unit, u) for u in todo]
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            results.append(res)
            if args.verbose or i % max(1, len(todo) // 20) == 0:
                print(f"  [{i + 1}/{len(todo)}] {res['unit']} {res['seconds']:.1f}s "
                      f"elapsed {time.time() - t1:.0f}s", flush=True)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    wall = time.time() - t1
    man = {
        "generator_version": cfg["generator_version"], "git_sha": sha, "versions": versions(),
        "graph": args.graph, "rung": args.rung, "target_rows": target, "shards": shards,
        "seed": cfg["seed"], "perturbation": cfg["perturbation"],
        "profile": os.path.basename(prof_path), "profile_is_placeholder": bool(prof.get("placeholder")),
        "config_hash": config_hash,
        "shard_entities": cfg["graphs"][args.graph]["shard_entities"],
        "expected_rows_per_shard": per_shard, "layout": layout,
        "split": {"tag": tag, "units": [r["unit"] for r in results]},
        "timing": {"calibrate_s": calib_s, "generate_s": wall, "workers": args.workers},
    }
    man["tables"] = collect_tables(results)
    fs, base = storage.filesystem(args.root)
    mdir = manifest_dir(base, args.graph, args.rung)
    name = "manifest.json" if whole else f"manifest.part-{tag}.json"
    if whole:
        man = finalize(man, len(units))
    storage.write_bytes(fs, f"{mdir}/{name}", json.dumps(man, indent=1).encode())
    storage.write_bytes(fs, f"{base}/_ground_truth/{args.graph}/{args.rung}/effective_profile.json",
                        json.dumps(prof, indent=1).encode())
    rows = sum(t["rows"] for t in man["tables"].values())
    nbytes = sum(t["bytes"] for t in man["tables"].values())
    print(f"done: {rows:,} rows, {nbytes / 1e9:.2f} GB in {wall:.0f}s -> "
          f"{rows / wall / 1e6:.1f}M rows/s, {nbytes / wall / 1e9:.3f} GB/s", flush=True)


def collect_tables(results):
    tables = {}
    for res in results:
        for t, files in res["files"].items():
            e = tables.setdefault(t, {"rows": 0, "bytes": 0, "files": []})
            for f in files:
                e["rows"] += f["rows"]
                e["bytes"] += f["bytes"]
                e["files"].append(f)
    for e in tables.values():
        e["files"].sort(key=lambda f: f["path"])
    return tables


def finalize(man, n_units):
    if len(man["split"]["units"]) != n_units:
        sys.exit(f"manifest covers {len(man['split']['units'])} of {n_units} units")
    man.pop("split")
    truth = {t: e for t, e in man["tables"].items() if t.startswith(TRUTH)}
    man["tables"] = {t: e for t, e in man["tables"].items() if not t.startswith(TRUTH)}
    man["ground_truth"] = {t[len(TRUTH):]: e for t, e in truth.items()}
    man["schema"] = schema.describe(man["graph"])
    man["notes"] = ("File paths are relative to the root that holds <graph>/<rung>/. "
                    "Run verify.py for file hashes, schemas and row counts. This is not distribution validation. "
                    "_ground_truth/ holds synthetic hidden state; never join it into training data.")
    man["total_rows"] = sum(e["rows"] for e in man["tables"].values())
    man["total_bytes"] = sum(e["bytes"] for e in man["tables"].values())
    return man


def cmd_merge(args):
    fs, base = storage.filesystem(args.root)
    mdir = manifest_dir(base, args.graph, args.rung)
    parts = [p for p, _ in storage.list_files(fs, mdir) if p.rsplit("/", 1)[1].startswith("manifest.part-")]
    mans = [json.loads(storage.read_bytes(fs, p)) for p in parts]
    if not mans:
        sys.exit("no partial manifests (manifest.part-*.json) found")
    for key in ["config_hash", "shards", "git_sha", "versions", "layout"]:
        if len({json.dumps(m.get(key), sort_keys=True) for m in mans}) != 1:
            sys.exit(f"partial manifests disagree on {key}")
    man = dict(mans[0])
    units = [u for m in mans for u in m["split"]["units"]]
    tables = {}
    for m in mans:
        for t, e in m["tables"].items():
            d = tables.setdefault(t, {"rows": 0, "bytes": 0, "files": []})
            d["rows"] += e["rows"]
            d["bytes"] += e["bytes"]
            d["files"] += e["files"]
    for e in tables.values():
        e["files"].sort(key=lambda f: f["path"])
    man["tables"] = tables
    man["split"] = {"units": units}
    man["timing"] = {"generate_s_per_part": {m["split"]["tag"]: m["timing"]["generate_s"] for m in mans}}
    cfg = load_config(args.config)
    prof, _ = effective_profile(cfg, args.graph, args.profile)
    gen = make_generator(cfg, args.graph, prof)
    expected = plan_units(gen, man["layout"], man["shards"])
    if sorted(map(tuple, units)) != sorted(expected):
        sys.exit("partial manifests do not cover the planned units exactly once")
    man = finalize(man, len(expected))
    storage.write_bytes(fs, f"{mdir}/manifest.json", json.dumps(man, indent=1).encode())
    print(f"merged {len(mans)} parts: {man['total_rows']:,} rows")


def default_workers():
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return max(1, min(os.cpu_count() - 4, int(ram_gb // 6)))  # commerce activity workers peak near 5.3 GB


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["run", "merge"]:
        p = sub.add_parser(name)
        p.add_argument("--graph", required=True, choices=list(GENERATORS))
        p.add_argument("--rung", required=True, help="target total rows, e.g. 10M, 1B, 50B")
        p.add_argument("--root", required=True, help="local dir or s3://bucket/prefix")
        p.add_argument("--config", default=os.path.join(REPO, "config.json"))
        p.add_argument("--profile", help="override the profile path in config.json")
        if name == "run":
            p.add_argument("--workers", type=int, default=default_workers(),
                           help="default: min(cpus - 4, RAM / 6 GB)")
            part = p.add_mutually_exclusive_group()
            part.add_argument("--split", default="0/1", help="i/N: run every N-th unit starting at i")
            part.add_argument("--shard-range", help="a:b: run the units whose first shard is in [a, b)")
            p.add_argument("--shards", type=int, help="override the shard count derived from --rung")
            p.add_argument("--verbose", action="store_true")
            p.add_argument("--resume", action="store_true", help="skip units already recorded under _units/")
    args = ap.parse_args(argv)
    {"run": cmd_run, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
