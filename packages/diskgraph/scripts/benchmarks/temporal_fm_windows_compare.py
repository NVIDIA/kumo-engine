# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Short baseline/candidate pairs pinned to the same CPU; no overlapping queries."""
import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("data_root", type=Path, help="Directory containing fm-s, fm-minute, fm-hour, fm-day")
parser.add_argument("--baseline", type=Path, required=True, help="Parent of baseline diskgraph package")
parser.add_argument("--candidate", type=Path, required=True, help="Parent of candidate diskgraph package")
parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
parser.add_argument("--units", nargs="+", default=["s", "minute", "hour", "day"],
                    choices=["s", "minute", "hour", "day"])
parser.add_argument("--cycles", type=int, default=8)
parser.add_argument("--repeats", type=int, default=11)
parser.add_argument("--cutoff-offset-seconds", type=int, default=0)
parser.add_argument("--case", action="append", dest="case_names")
parser.add_argument("--baseline-revision", default="unspecified")
parser.add_argument("--candidate-revision", default="unspecified")
parser.add_argument("--worker", type=Path, default=Path(__file__).with_name("temporal_fm_windows.py"),
                    help="Worker implementing the same JSON request/response protocol")
args = parser.parse_args()
if args.cycles < 1 or args.repeats < 1:
    parser.error("cycles and repeats must be positive")
root = args.data_root.resolve()
packages = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
native_hashes = {}
for variant, package in packages.items():
    extensions = list((package / "diskgraph").glob("_c*.so")) + list((package / "diskgraph").glob("_c*.pyd"))
    if len(extensions) != 1:
        raise ValueError(f"Expected one native extension in {package / 'diskgraph'}: {extensions}")
    native_hashes[variant] = hashlib.sha256(extensions[0].read_bytes()).hexdigest()
if len(set(native_hashes.values())) != 2:
    raise ValueError("Baseline and candidate native libraries are identical; check build isolation")
worker_script = args.worker.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
raw_output = output / "raw"
raw_output.mkdir()
cpu = None
if hasattr(os, "sched_getaffinity"):
    cpu = sorted(os.sched_getaffinity(0))[0]
    os.sched_setaffinity(0, {cpu})
fixture_hashes = {hashlib.sha256((root / f"fm-{unit}" / "events.parquet").read_bytes()).hexdigest()
                  for unit in args.units}
if len(fixture_hashes) != 1:
    raise ValueError("Clock indexes must use identical source events")
cases = [
    'age1_upper_only_narrow_k24', 'age1_upper_only_wide_k24',
    'age1_w7_narrow_k24', 'age1_w30_narrow_k24', 'age1_w30_wide_k24',
    'age1_w90_narrow_k24', 'age1_w365_narrow_k24',
    'age365_upper_only_narrow_k24', 'age1_w30_lengths', 'age1_w30_shared_sum',
    'unbounded_control', 'mismatched_control',
    'age1_w30_all_rows', 'degree16_upper_only', 'degree16_w30',
    'degree128_upper_only', 'degree128_w30',
]
cycles = args.cycles
cutoff_offset = args.cutoff_offset_seconds
if args.case_names:
    cases = args.case_names
if len(cases) != len(set(cases)):
    parser.error("case names must be unique")
summary = {
    "protocol": "Adjacent pairs with balanced AB/BA order per case; fresh workers each cycle; two warmups per batch; one native thread",
    "cycles": cycles, "measured_calls_per_batch": args.repeats,
    "cutoff_offset_seconds": cutoff_offset, "cpu_affinity": [cpu] if cpu is not None else None,
    "source_events_sha256": fixture_hashes.pop(),
    "worker_sha256": hashlib.sha256(worker_script.read_bytes()).hexdigest(),
    "native_sha256": native_hashes,
    "baseline_revision": args.baseline_revision, "candidate_revision": args.candidate_revision,
    "packages": {variant: str(path) for variant, path in packages.items()}, "units": {},
}
for unit in args.units:
    records = {case: [] for case in cases}
    for cycle in range(cycles):
        workers = {}
        try:
            for variant in ('baseline', 'candidate'):
                env = dict(os.environ, PYTHONPATH=str(packages[variant]),
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                workers[variant] = subprocess.Popen(
                    [sys.executable, str(worker_script), str(root / f"fm-{unit}"),
                     "--label", variant, "--repeats", str(args.repeats), "--serve",
                     "--cutoff-offset-seconds", str(cutoff_offset)],
                    env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            order = list(cases)
            random.Random(82 + cycle).shuffle(order)
            for case in order:
                pair = {}
                variants = ('baseline', 'candidate') if (cases.index(case) + cycle) % 2 == 0 else ('candidate', 'baseline')
                for variant in variants:
                    worker = workers[variant]
                    worker.stdin.write(json.dumps([case]) + '\n')
                    worker.stdin.flush()
                    response = worker.stdout.readline()
                    assert response, (unit, cycle, case, variant, worker.poll())
                    run = json.loads(response)
                    (raw_output / f'{unit}-cycle{cycle}-{case}-{variant}.json').write_text(response)
                    pair[variant] = run['cases'][case]
                assert pair['baseline']['sha256'] == pair['candidate']['sha256'], (unit, case)
                records[case].append(pair)
        finally:
            for worker in workers.values():
                worker.stdin.close()
            for worker in workers.values():
                assert worker.wait() == 0
        print(f'Finished paired {unit} cycle {cycle}', flush=True)
    summary['units'][unit] = {}
    for case, pairs in records.items():
        assert len({pair[v]['sha256'] for pair in pairs for v in pair}) == 1
        ratios = [pair['baseline']['median_call_ns'] / pair['candidate']['median_call_ns'] for pair in pairs]
        values = {}
        for variant in ('baseline', 'candidate'):
            values[variant] = {
                'median_call_ns': statistics.median(pair[variant]['median_call_ns'] for pair in pairs),
                'batch_medians_ns': [pair[variant]['median_call_ns'] for pair in pairs],
                'stats': pairs[0][variant]['stats'],
            }
        summary['units'][unit][case] = {
            **values, 'median_paired_speedup': statistics.median(ratios),
            'paired_speedups': ratios, 'paired_speedup_min': min(ratios), 'paired_speedup_max': max(ratios),
            'selectivity': pairs[0]['baseline']['selectivity'], 'spec': pairs[0]['baseline']['spec'],
            'matching_sha256': pairs[0]['baseline']['sha256'],
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
