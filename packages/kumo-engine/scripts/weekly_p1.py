#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run five complete CPU sampling operations on the frozen commerce/100M graph."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    'weekly',
    Path(__file__).with_name('weekly_check.py'))
weekly = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(weekly)


def workload_digest(manifest):
    value = {
        k: v
        for k, v in manifest.items()
        if k not in ('time_tolerance', 'calibration_only')
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True,
                   separators=(',', ':')).encode()).hexdigest()


def reference_operations(path, checksum, manifest, manifest_sha256):
    if weekly.digest(path) != checksum:
        raise ValueError('Approved P1 reference checksum differs')
    value = weekly.read(path)
    sources = value.get('measurement_results_sha256', [])
    if (value.get('format') != 'kumo-engine-weekly-p1-reference-v1'
            or value.get('status') != 'CALIBRATED'
            or value.get('manifest_sha256') != manifest_sha256
            or value.get('workload_sha256') != workload_digest(manifest)
            or value.get('physical_fits') != 0 or not sources or any(
                not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v)
                for v in sources) or check_operations(
                    value.get('operations', []), value.get('operations', []),
                    atol_seconds=0., rtol=0.)['status'] != 'PASS'):
        raise ValueError(
            'P1 reference is incomplete or uses a different workload')
    return value['operations']


def check_operations(current, reference, *, atol_seconds, rtol):
    if not all(weekly.finite(v) and v >= 0 for v in (atol_seconds, rtol)):
        raise ValueError('Approve finite nonnegative P1 timing tolerances')
    valid = len(current) == len(reference) == 5
    differences = []
    for index, (left, right) in enumerate(zip(current, reference)):
        passed = (left.get('status') == right.get('status') == 'PASS'
                  and left.get('seeds') == right.get('seeds') == 300000
                  and type(left.get('batches')) is int and left['batches'] > 0
                  and left['batches'] == right.get('batches') and all(
                      type(left.get(k)) is int and left[k] > 0
                      and left[k] == right.get(k)
                      for k in ('nodes', 'edges', 'fetched_rows'))
                  and isinstance(left.get('membership_sha256'), str)
                  and len(left['membership_sha256']) == 64 and
                  left['membership_sha256'] == right.get('membership_sha256'))
        valid = valid and passed
        differences.append(
            dict(operation=index, status='PASS' if passed else 'FAIL'))
    durations = [[row.get('seconds') for row in rows]
                 for rows in (reference, current)]
    timing = dict(status='FAIL', atol_seconds=atol_seconds, rtol=rtol)
    if all(
            len(values) == 5 and all(
                weekly.finite(v) and v > 0 for v in values)
            for values in durations):
        before, after = map(statistics.median, durations)
        maximum = before * (1 + rtol) + atol_seconds
        timing.update(status='PASS' if after <= maximum else 'FAIL',
                      reference_median_seconds=before,
                      candidate_median_seconds=after,
                      maximum_median_seconds=maximum)
    valid = valid and timing['status'] == 'PASS'
    return dict(status='PASS' if valid else 'FAIL', operations=differences,
                timing=timing)


def join_workers(processes, timeout):
    deadline = time.monotonic() + timeout
    try:
        for process in processes:
            if process.wait(timeout=max(.01, deadline - time.monotonic())) != 0:
                raise ValueError('A native CPU sampling worker failed')
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()


def inputs(root, checksum):
    if weekly.digest(root / 'p1.json') != checksum:
        raise ValueError('P1 input manifest checksum differs')
    value = weekly.read(root / 'p1.json')
    work = value['work']
    tolerance = value.get('time_tolerance', {})
    if (not isinstance(tolerance, dict)
            or set(tolerance) != {'atol_seconds', 'rtol'} or not all(
                weekly.finite(tolerance[k]) and tolerance[k] >= 0
                for k in ('atol_seconds', 'rtol'))):
        raise ValueError('Approve P1 timing tolerances before execution')
    if (value['format'] != 'kumo-engine-weekly-p1-v1'
            or value['dataset'] != 'commerce/100M' or work['n'] != 300000
            or type(work['offset']) is not int or work['offset'] < 0 or any(
                type(work[k]) is not int or work[k] <= 0
                for k in ('batch', 'procs', 'threads', 'hops'))
            or work['procs'] > work['n'] or not 1 <= work['hops'] <= 6
            or sum(value['row_counts'].values()) < 100000000):
        raise ValueError(
            'Use the complete frozen commerce/100M sampling workload')
    for name in value['files']:
        weekly.data_file(root, name, value['files'])
    metadata = weekly.read(
        weekly.data_file(root,
                         value['index'].rstrip('/') + '/ingest_stats.json',
                         value['files']))
    if metadata['stats']['node_counts'] != value['row_counts']:
        raise ValueError(
            'Native graph population differs from the source manifest')
    index = root / value['index']
    actual = {str(p.relative_to(root)) for p in index.rglob('*') if p.is_file()}
    if actual != {
            n
            for n in value['files']
            if n.startswith(value['index'].rstrip('/') + '/')
    }:
        raise ValueError('P1 native index has missing or unlisted files')
    return value


def sample_worker(root, manifest, slot, output):
    import numpy as np
    import pyarrow.parquet as pq

    source = Path(__file__).resolve(
    ).parents[2] / 'diskgraph/scripts/benchmarks/commerce/dg_bench.py'
    spec = importlib.util.spec_from_file_location('weekly_sampler', source)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    work = manifest['work']
    table = pq.read_table(
        weekly.data_file(root, manifest['seedmap'], manifest['files']))
    if (not {'customer_id', 'anchor_s', 'gid'} <= set(table.schema.names)
            or any(table[c].null_count
                   for c in ('customer_id', 'anchor_s', 'gid'))
            or len(set(table['customer_id'].to_pylist())) != len(table)):
        raise ValueError('P1 seed map is incomplete or repeated')
    population = pq.read_table(
        weekly.data_file(root, manifest['population'], manifest['files']))
    original = population.slice(0, len(table))
    if (original['customer_id'].to_pylist() != table['customer_id'].to_pylist()
            or
            original['anchor'].cast('timestamp[s]').cast('int64').to_pylist()
            != table['anchor_s'].to_pylist()):
        raise ValueError(
            'P1 seed map differs from the stored prediction population')
    index = root / manifest['index']
    engine = benchmark.engine(str(index), work['threads'])
    mapped = benchmark.fetch(engine, 'customers', table['gid'].to_numpy(),
                             ['customer_id'])
    if mapped is None or mapped['customer_id'].to_pylist(
    ) != table['customer_id'].to_pylist():
        raise ValueError(
            'P1 global IDs do not resolve to the stored business IDs')
    selected = benchmark.C.tile(table, work['offset'] + work['n']).slice(
        work['offset'], work['n'])
    positions = np.array_split(np.arange(work['n']), work['procs'])[slot]
    selected = selected.take(positions)
    names, plan = benchmark.type_names(engine), benchmark.plan_json(
        work['hops'])
    checksum = hashlib.sha256()
    report = dict(status='RUNNING', slot=slot, pid=os.getpid(), seeds=0,
                  batches=0, nodes=0, edges=0, fetched_rows=0, seconds=0.)
    started = time.monotonic()
    for offset in range(0, len(selected), work['batch']):
        batch = selected.slice(offset, work['batch'])
        nodes, stats = benchmark.sample(engine, names, batch['gid'].to_numpy(),
                                        batch['anchor_s'].to_numpy(), plan)
        identities = []
        for name, (seed, gids, hops) in sorted(nodes.items()):
            frame = benchmark.fetch(engine, name, gids)
            if frame is None or len(frame) != len(gids):
                raise ValueError('P1 feature fetch omitted sampled nodes')
            report['fetched_rows'] += len(frame)
            identities.append([
                name,
                sorted(zip(seed.tolist(), gids.tolist(), hops.tolist()))
            ])
        checksum.update(json.dumps(identities, separators=(',', ':')).encode())
        if stats['total_nodes'] <= 0 or not nodes:
            raise ValueError('P1 sampling returned no native context')
        report['nodes'] += stats['total_nodes']
        report['edges'] += stats['total_edges']
        report['batches'] += 1
        report['seeds'] += len(batch)
    report.update(status='PASS', seconds=time.monotonic() - started,
                  membership_sha256=checksum.hexdigest())
    weekly.save(output, report)
    return report


def run(args):
    with args.host_lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args)


def run_locked(args):
    manifest = inputs(args.input_root, args.manifest_sha256)
    baseline = reference_operations(
        args.reference, args.reference_sha256, manifest,
        args.manifest_sha256) if args.reference else None
    if args.output.exists():
        raise ValueError('Use a fresh P1 evidence directory')
    if args.output.resolve().is_relative_to(args.input_root.resolve()):
        raise ValueError('Keep evidence outside immutable inputs')
    args.output.mkdir(parents=True)
    report = dict(format='kumo-engine-weekly-p1-result-v1', status='RUNNING',
                  manifest_sha256=args.manifest_sha256, physical_fits=0,
                  workload_sha256=workload_digest(manifest), operations=[])
    try:
        if not sys.flags.isolated or sys.prefix == sys.base_prefix:
            raise ValueError('Use an isolated wheel environment')
        distribution = importlib.metadata.distribution('kumo-engine')
        if weekly.digest(args.wheel) != args.wheel_sha256:
            raise ValueError('Selected P1 wheel checksum differs')
        helper = Path(__file__).with_name('nightly_check.py')
        helper_spec = importlib.util.spec_from_file_location(
            'p1_install', helper)
        module = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(module)
        module.verify_installed_files(args.wheel, distribution)
        report['runtime'] = dict(
            version=distribution.version, wheel_sha256=args.wheel_sha256,
            dependencies={
                d.metadata['Name']: d.version
                for d in importlib.metadata.distributions()
            })
        for operation in range(5):
            operation_started = time.monotonic()
            directory = args.output / f'operation-{operation}'
            directory.mkdir()
            processes = []
            try:
                for slot in range(manifest['work']['procs']):
                    log = (directory / f'worker-{slot}.log').open('w')
                    command = [
                        sys.executable, '-I',
                        str(Path(__file__).resolve()), '--input-root',
                        str(args.input_root), '--manifest-sha256',
                        args.manifest_sha256, '--output',
                        str(directory / f'worker-{slot}.json'), '--worker-slot',
                        str(slot)
                    ]
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
                                 KUMO_ACCELERATOR='cpu'))
                    processes.append((process, log))
                join_workers([process for process, _ in processes],
                             args.timeout_seconds)
            finally:
                for process, log in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    log.close()
            workers = [
                weekly.read(directory / f'worker-{slot}.json')
                for slot in range(len(processes))
            ]
            row = dict(
                status='PASS', seeds=sum(r['seeds'] for r in workers),
                seconds=time.monotonic() - operation_started,
                batches=sum(r['batches'] for r in workers),
                nodes=sum(r['nodes'] for r in workers),
                edges=sum(r['edges']
                          for r in workers), fetched_rows=sum(r['fetched_rows']
                                                              for r in workers),
                membership_sha256=hashlib.sha256(''.join(
                    r['membership_sha256']
                    for r in workers).encode()).hexdigest(), workers=workers)
            report['operations'].append(row)
            weekly.save(args.output / 'p1-result.json', report)
        report['comparison'] = check_operations(
            report['operations'], baseline or report['operations'],
            **manifest['time_tolerance'])
        report['status'] = report['comparison']['status']
        inputs(args.input_root, args.manifest_sha256)
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
    finally:
        weekly.save(args.output / 'p1-result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-root', 'output', 'host-lock', 'reference', 'wheel'):
        parser.add_argument('--' + name, type=Path, required=name
                            in ('input-root', 'output'))
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--reference-sha256')
    parser.add_argument('--wheel-sha256')
    parser.add_argument('--mode', choices=('reference', 'candidate'),
                        default='candidate')
    parser.add_argument('--worker-slot', type=int)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    args = parser.parse_args()
    if args.worker_slot is not None:
        # The parent already verifies the full index before and after the block.
        manifest = weekly.read(
            weekly.data_file(args.input_root, 'p1.json',
                             {'p1.json': args.manifest_sha256}))
        sample_worker(args.input_root, manifest, args.worker_slot, args.output)
        return 0
    if not args.host_lock or (args.reference and not args.reference_sha256):
        parser.error(
            'Bind the shared host lock and the approved reference checksum')
    if not args.wheel or not args.wheel_sha256 or (args.mode == 'candidate'
                                                   and not args.reference):
        parser.error(
            'Bind the selected wheel and an approved reference for candidate work'
        )
    return 0 if run(args)['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
