# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run checked baseline/candidate commands in paired AB/BA order.

Only complete-command wall time is measured. Commands must synchronize their
own CUDA work. This tool does not infer kernel time or validate model quality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import sys
from pathlib import Path

if __package__:
    from .author_tools import (
        git_state,
        handle_termination,
        machine,
        new_output,
        run_process,
        sha256,
        write_json,
    )
else:
    from author_tools import (
        git_state,
        handle_termination,
        machine,
        new_output,
        run_process,
        sha256,
        write_json,
    )

SIDES = ('baseline', 'candidate')


def positive(value, name, *, integer=False, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (not zero and value == 0)
            or (integer and not isinstance(value, int))):
        raise ValueError(
            f'{name} must be a finite {"nonnegative" if zero else "positive"} number'
        )
    return value


def strings(value, name):
    if not isinstance(value, list) or not value or not all(
            isinstance(x, str) and x for x in value):
        raise ValueError(f'{name} must be a nonempty argv list')
    return value


def invalid_constant(value):
    raise ValueError(f'Non-finite JSON number: {value}')


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(), parse_constant=invalid_constant)
    if not isinstance(plan, dict):
        raise ValueError('Plan must be a JSON object')
    if type(plan.get(
            'schema_version')) is not int or plan['schema_version'] != 1:
        raise ValueError('Expected schema_version: 1')
    for name in ['workload_id', 'scope']:
        if not isinstance(plan.get(name), str) or not plan[name].strip():
            raise ValueError(f'{name} is required')
    positive(plan.get('timeout_seconds'), 'timeout_seconds')
    positive(plan.get('pairs', 4), 'pairs', integer=True)
    positive(plan.get('warmups', 1), 'warmups', integer=True, zero=True)
    context = plan.get('context', {})
    if not isinstance(context, dict):
        raise ValueError('context must be an object')
    for name in ['seed', 'workers', 'cache_state', 'device']:
        if name not in context or context[name] is None:
            raise ValueError(
                f'context.{name} must be declared (use an explicit unknown/not-applicable description)'
            )
    expected = plan.get('correctness_expect')
    if not isinstance(expected, dict) or not expected:
        raise ValueError(
            'correctness_expect must declare a nonempty expected JSON object')
    if not isinstance(plan.get('inputs'), list) or not plan['inputs']:
        raise ValueError(
            'inputs must identify at least the dataset/configuration manifest')
    for item in plan['inputs']:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            raise ValueError('Each input must declare a file path')
        source = (path.parent / item['path']).resolve()
        if not source.is_file():
            raise ValueError(f'Missing input manifest: {source}')
        expected_hash = item.get('sha256')
        actual = sha256(source)
        if expected_hash is not None and expected_hash != actual:
            raise ValueError(f'Input checksum differs: {source}')
        item.update(path=str(source), sha256=actual)
    for side in SIDES:
        spec = plan[side]
        if not isinstance(spec, dict) or not isinstance(
                spec.get('cwd'), str) or not isinstance(spec.get('python'),
                                                        str):
            raise ValueError(f'{side} must declare cwd and python strings')
        cwd = (path.parent / spec['cwd']).resolve()
        if not cwd.is_dir():
            raise ValueError(f'{side}: cwd must exist')
        spec['cwd'] = str(cwd)
        if spec['python'] == '{runner_python}':
            spec['python'] = sys.executable
        python = os.path.abspath(
            os.path.expanduser(str(
                path.parent /
                spec['python']))) if '/' in spec['python'] else shutil.which(
                    spec['python'])
        if not python or not os.access(python, os.X_OK):
            raise ValueError(f'{side}: python must identify an executable')
        spec['python'] = python
        strings(spec['argv'], f'{side}.argv')
        checker = strings(spec['correctness_argv'], f'{side}.correctness_argv')
        if not any('{result}' in arg for arg in checker):
            raise ValueError(
                f'{side}: correctness_argv must write its JSON to {{result}}')
        env = spec.get('env', {})
        if not isinstance(env, dict) or not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in env.items()):
            raise ValueError(f'{side}.env must map strings to strings')
        if any(
                re.search(
                    r'(?:^|_)(?:TOKEN|PASSWORD|SECRET|CREDENTIALS?|ACCESS_KEY|PRIVATE_KEY)(?:_|$)',
                    k, re.I) for k in env):
            raise ValueError('Do not put credentials in a benchmark plan')
        artifacts = spec.get('artifacts', [])
        if not isinstance(artifacts, list) or not all(
                isinstance(p, str) for p in artifacts):
            raise ValueError(f'{side}.artifacts must list file paths')
        spec['artifacts'] = [
            dict(path=str((path.parent / p).resolve()), sha256=sha256(
                (path.parent / p).resolve())) for p in artifacts
        ]
    return plan


def expand(argv, spec, result):
    return [
        a.replace('{python}', spec['python']).replace('{result}', str(result))
        for a in argv
    ]


def render(report):
    lines = [
        '# Paired command benchmark', '', f"Status: **{report['status']}**", '',
        f"Workload: `{report['plan']['workload_id']}`", '',
        report['plan']['scope'], '',
        '**Timing boundary: complete command wall time**, including process startup and shutdown.',
        'No kernel/GPU timing, memory measurement, learning-quality check, or deployment claim is inferred.',
        '',
        'Correctness receipts are produced by the author-supplied checker; review that checker and its scope.',
        ''
    ]
    if 'summary' in report:
        summary = report['summary']
        lines += [
            '| Measure | Seconds / ratio |', '| --- | ---: |',
            f"| Baseline median command time | {summary['baseline_median_seconds']:.6f} |",
            f"| Candidate median command time | {summary['candidate_median_seconds']:.6f} |",
            f"| Median paired baseline/candidate ratio | {summary['median_paired_ratio']:.4f} |",
            f"| Paired ratio range | {summary['ratio_min']:.4f}–{summary['ratio_max']:.4f} |",
            '',
            'Ratios above one mean the candidate command finished sooner. No automatic performance gate applies.',
            ''
        ]
    if report.get('error'):
        lines += [f"Error: {report['error']}", '']
    lines += [
        f"Pairs: {report['plan'].get('pairs', 4)}; fresh-process warmups per side: {report['plan'].get('warmups', 1)}.",
        '', '## Comparability warnings', ''
    ]
    lines += [f'- {w}' for w in report['warnings']]
    lines += [
        '',
        'Raw commands, logs, all samples, Git state, input hashes and interpreter probes are in `results.json`.',
        'Only explicitly listed files are hashed; manifest contents are not recursively verified.',
        ''
    ]
    return '\n'.join(lines)


def benchmark(plan: dict, output: Path) -> int:
    initial_states = {s: git_state(Path(plan[s]['cwd'])) for s in SIDES}
    output = new_output(
        output,
        tuple(
            Path(initial_states[s].get('root', plan[s]['cwd'])) for s in SIDES))
    report = {
        'schema_version':
        1,
        'status':
        'running',
        'plan':
        plan,
        'machine':
        machine(),
        'variants': {},
        'runs': [],
        'warnings': [
            'Same host does not establish comparable load, clocks, thermal state, storage or page cache.',
            'Timed commands must complete and synchronize device work; preparation and output conversion are included if the command includes them.',
            'Inherited environment and installed dependency versions are not captured. Only explicit per-side environment overrides are recorded.',
            'Untracked and ignored file contents are not fingerprinted by Git state. List relevant native libraries/wheels/scripts as artifacts.',
            'The checker runs before timing. Timed harnesses must validate their own outputs for claims about each measured iteration.',
        ]
    }
    result_file = output / 'results.json'

    def execute(side, phase, index, argv):
        spec = plan[side]
        receipt = output / f'{phase}-{index}-{side}.json'
        row = run_process(expand(argv, spec, receipt), Path(spec['cwd']),
                          output / f'{phase}-{index}-{side}.log',
                          plan['timeout_seconds'],
                          dict(os.environ, **spec.get('env', {})))
        row.update(side=side, phase=phase, index=index)
        report['runs'].append(row)
        write_json(result_file, report)
        if row['status'] != 'ok':
            raise ValueError(
                f'{phase} {side} failed ({row["status"]}); see {row["log"]}')
        return receipt, row

    try:
        for side in SIDES:
            spec = plan[side]
            state = initial_states[side]
            report['variants'][side] = state
            if state['dirty'] is not False:
                report['warnings'].append(
                    f'{side}: dirty or unavailable Git state; revision alone does not identify the tested code.'
                )
            probe = [
                spec['python'], '-c',
                'import json,platform,sys; print(json.dumps({"executable":sys.executable,"version":sys.version,"prefix":sys.prefix,"base_prefix":sys.base_prefix,"platform":platform.platform()}))'
            ]
            execute(side, 'interpreter', 0, probe)
            receipt, _ = execute(side, 'correctness', 0,
                                 spec['correctness_argv'])
            if receipt.is_symlink() or not receipt.is_file():
                raise ValueError(
                    f'{side}: checker did not create a regular JSON receipt')
            actual = json.loads(receipt.read_text(),
                                parse_constant=invalid_constant)
            if json.dumps(actual, sort_keys=True,
                          allow_nan=False) != json.dumps(
                              plan['correctness_expect'], sort_keys=True,
                              allow_nan=False):
                raise ValueError(
                    f'{side}: checker receipt differs from correctness_expect')
            report['variants'][side]['correctness_receipt'] = actual
            report['variants'][side]['correctness_sha256'] = sha256(receipt)
        if plan['baseline'].get('env', {}) != plan['candidate'].get('env', {}):
            report['warnings'].append(
                'Per-side environment overrides differ; attribute any comparison accordingly.'
            )
        if plan['baseline']['python'] != plan['candidate']['python']:
            report['warnings'].append(
                'Separate interpreters are used; compare their logged versions and installed dependencies.'
            )
        for index in range(plan.get('warmups', 1)):
            for side in SIDES:
                execute(side, 'warmup', index, plan[side]['argv'])
        ratios = []
        for index in range(plan.get('pairs', 4)):
            pair = {}
            for side in (SIDES if index % 2 == 0 else SIDES[::-1]):
                _, row = execute(side, 'measure', index, plan[side]['argv'])
                pair[side] = row['wall_seconds']
            ratios.append(pair['baseline'] / pair['candidate'])
        if len(ratios) < 3:
            report['warnings'].append(
                'Fewer than three pairs: this is a runner smoke check, not stable performance evidence.'
            )
        if any(r['wall_seconds'] < 1 for r in report['runs']
               if r['phase'] == 'measure'):
            report['warnings'].append(
                'Subsecond complete-command timings can be dominated by startup and process-wait granularity. Use a longer repeated workload or a specialized timer.'
            )
        # An input changing during the comparison invalidates the whole result.
        for item in plan['inputs']:
            if sha256(Path(item['path'])) != item['sha256']:
                raise ValueError(
                    f'Input manifest changed during the run: {item["path"]}')
        for side in SIDES:
            after = git_state(Path(plan[side]['cwd']))
            report['variants'][side]['git_after'] = after
            before = {
                k: v
                for k, v in report['variants'][side].items() if k in after
            }
            if before != after:
                raise ValueError(f'{side}: Git state changed during the run')
            for artifact in plan[side]['artifacts']:
                if sha256(Path(artifact['path'])) != artifact['sha256']:
                    raise ValueError(
                        f'{side}: listed artifact changed during the run')
        report['summary'] = {
            f'{side}_median_seconds':
            statistics.median(r['wall_seconds'] for r in report['runs']
                              if r['side'] == side and r['phase'] == 'measure')
            for side in SIDES
        }
        report['summary'].update(median_paired_ratio=statistics.median(ratios),
                                 ratio_min=min(ratios), ratio_max=max(ratios),
                                 paired_ratios=ratios)
        report['status'] = 'ok'
    except (OSError, ValueError, KeyboardInterrupt) as error:
        report['status'] = 'failed'
        report['error'] = str(error) or 'Interrupted'
    finally:
        write_json(result_file, report)
        (output / 'summary.md').write_text(render(report))
    return 0 if report['status'] == 'ok' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='new directory outside both worktrees')
    args = parser.parse_args(argv)
    try:
        if sys.version_info < (3, 11):
            raise ValueError('Use Python 3.11 or newer for author tools')
        plan = read_plan(args.plan.expanduser().resolve())
        return benchmark(plan, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    with handle_termination():
        sys.exit(main())
