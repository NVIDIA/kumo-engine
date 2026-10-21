"""Builds results-<rung>.json from the matrix lines, the ingest record and the run's environment.

    python aggregate.py --rung 50B --jsonl $OUT_DIR/results-50B.jsonl --parquet-bytes N \
        (--dg-snapshot $OUT_DIR/snap-50B/results.json | --dg-snapshot-s S --dg-index-bytes B) \
        [--env "$(cat $OUT_DIR/env-50B.json)"] [--footnote "..."] [--out results-50B.json]
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True)
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--dg-snapshot", help="run_e2e results.json (stages.snapshot)")
    ap.add_argument("--dg-snapshot-s", type=float)
    ap.add_argument("--dg-index-bytes", type=int)
    ap.add_argument("--parquet-bytes", type=int, required=True)
    ap.add_argument("--env", default="{}")
    ap.add_argument("--footnote", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    cells = {}
    for path in a.jsonl:
        for line in open(path):
            r = json.loads(line)
            if r.get("system") == "diskgraph":
                cells[(r["hops"], r["cache"])] = r
    if a.dg_snapshot:
        s = json.load(open(a.dg_snapshot))["stages"]["snapshot"]
        ingest = {"seconds": s.get("seconds") or s.get("elapsed_s"), "disk_gib": s["index_bytes"] / 2**30, "detail": s}
    else:
        ingest = {"seconds": a.dg_snapshot_s, "disk_gib": (a.dg_index_bytes or 0) / 2**30}
    out = {"rung": a.rung, "env": json.loads(a.env), "ingest": ingest, "parquet_gib": a.parquet_bytes / 2**30,
           "throughput": sorted(cells.values(), key=lambda r: (r["hops"], r["cache"])), "footnote": a.footnote}
    p = Path(a.out or f"results-{a.rung}.json")
    p.write_text(json.dumps(out, indent=1, default=str))
    print(p)
    for r in out["throughput"]:
        print(f'h{r["hops"]} {r["cache"]:4} {r["seeds_per_s"]:>12} seeds/s  steady {r.get("steady_seeds_per_s")}'
              f'  first {r["first_batch_s"]}s  n={r["seeds"]} {r.get("status", "ok")}  cores {r["cores_busy"]}'
              f'  read {r["read_mb_s"]} MB/s  rss {r.get("peak_rss_gb_sum")} GB')


if __name__ == "__main__":
    main()
