"""Runs the DiskGraph sampling matrix for one rung: hops 2..6, cold then warm.

    python run_matrix.py --rung 50B --index I --seedmap M --vcpus 128 --dg-threads 8 \
        [--hops 2,3,4,5,6] [--scale 1.0] [--cap-s 1800] [--drop-all] [--out results-50B.jsonl]

Each cell is a separate process tree (dg_bench.py rate); results append to --out, its log next to it.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Per hop: (batch per call, seeds per cell at scale 1.0)
HOP = {2: (512, 1_000_000), 3: (256, 300_000), 4: (128, 200_000), 5: (64, 60_000), 6: (64, 60_000)}


def run(cmd, log):
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"  rc={rc} {time.time() - t0:.0f}s", flush=True)
    if rc:
        raise SystemExit(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--seedmap", required=True)
    ap.add_argument("--vcpus", type=int, required=True)
    ap.add_argument("--dg-threads", type=int, default=1, help="sampler threads per process; processes = vcpus / this")
    ap.add_argument("--hops", default="2,3,4,5,6")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--cap-s", type=float, default=1800)
    ap.add_argument("--drop-all", action="store_true")
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.vcpus < 1 or a.dg_threads < 1 or a.vcpus % a.dg_threads or a.scale <= 0 or a.cap_s <= 0:
        ap.error("vcpus must be a positive multiple of dg-threads; scale and cap-s must be positive")
    try:
        hops = [int(x) for x in a.hops.split(",")]
    except ValueError:
        ap.error("--hops must be comma-separated integers from 2 to 6")
    if not hops or any(h not in HOP for h in hops):
        ap.error("--hops must contain values from 2 to 6")
    out = Path(a.out or HERE / f"results-{a.rung}.jsonl")
    if out.exists():
        ap.error("--out already exists; choose a new results file")
    out.parent.mkdir(parents=True, exist_ok=True)
    log = str(out.with_suffix(".log"))
    for h in hops:
        batch, n = HOP[h]
        n = max(batch * a.vcpus, int(n * a.scale))
        run([a.py, str(HERE / "dg_bench.py"), "rate", "--index", a.index, "--seedmap", a.seedmap, "--hops", str(h),
             "--n", str(n), "--batch", str(batch), "--procs", str(a.vcpus // a.dg_threads),
             "--threads", str(a.dg_threads), "--cold", "--cap-s", str(a.cap_s), "--label", a.rung, "--results", str(out)]
            + (["--drop-all"] if a.drop_all else []), log)


if __name__ == "__main__":
    main()
