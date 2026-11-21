#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the same installed-wheel customer-model test across platforms and devices."""
import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLATFORMS = ('databricks', 'sagemaker', 'snowflake')
ACCELERATORS = ('cpu', 'gpu')
PHASES = ('platform-prerequisite', 'train', 'reload')


def workload(platform):
    suffix = '_snowflake' if platform == 'snowflake' else ''
    return HERE / ('wheel_mlops_pytorch' + suffix + '.py')


def clean_report(report):
    if report.get('status') not in ('passed', 'blocked') or any(
            report.get(key) for key in ('error', 'collection_error', 'close_error')):
        raise ValueError('The platform attempt did not complete cleanly')


def phase_receipts(platform, execution):
    report = json.loads((execution / 'summary.json').read_text())
    clean_report(report)
    if platform != 'snowflake':
        cleanup = json.loads((execution / 'cleanup.json').read_text())
        if cleanup.get('status') != 'passed':
            raise ValueError('Platform cleanup did not pass')
        verdict = report.get('verdict', {})
        key = 'engine_checks' if platform == 'databricks' else 'engine'
        if verdict.get(key) != 'passed':
            raise ValueError('The platform evidence validator did not pass')
        return report['phases']
    if report.get('stages', {}).get('preflight', {}).get('status') != 'passed':
        raise ValueError('Snowflake preflight did not pass')
    stage = json.loads((execution / 'lifecycle/report.json').read_text())
    clean_report(stage)
    if stage.get('cleanup', {}).get('status') != 'passed':
        raise ValueError('Snowflake cleanup did not pass')
    receipt = stage['notebook_receipt']
    names = receipt['logical_files']

    def read(relative):
        matches = [filename for filename, logical in names.items() if logical == relative]
        if len(matches) != 1 or Path(matches[0]).name != matches[0]:
            raise ValueError('Missing or ambiguous selected phase evidence')
        path = execution / 'lifecycle/download' / matches[0]
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['artifacts'].get(matches[0]):
            raise ValueError('Phase evidence checksum differs from notebook receipt')
        return json.loads(path.read_text())

    summary = read('lifecycle/summary.json')
    clean_report(summary)
    result = {}
    for name in PHASES:
        phase = read('lifecycle/' + name + '.json')
        recorded = summary['phases'][name]
        if recorded.get('status') != phase.get('status') or recorded.get('returncode') != 0:
            raise ValueError('Selected worker status or exit code did not pass')
        result[name] = dict(phase, returncode=recorded['returncode'])
    return result


def verify(platform, accelerator, execution):
    phases = phase_receipts(platform, execution)
    if set(phases) != set(PHASES) or any(
            phase.get('status') != 'passed' or phase.get('returncode') != 0
            for phase in phases.values()):
        raise ValueError('All customer-model phases must pass')
    expected_hash = hashlib.sha256(workload(platform).read_bytes()).hexdigest()
    selected = ('cuda',) if accelerator == 'gpu' else ('cpu',)
    identities = []
    for phase in ('train', 'reload'):
        receipt = phases[phase]
        if receipt.get('workload', {}).get('sha256') != expected_hash:
            raise ValueError('Phase evidence describes another customer workload')
        versions = receipt['result']['versions']
        counts = {device: sum(v.get('device') == device for v in versions) for device in selected}
        if len(versions) != 2 * len(selected) or counts != dict.fromkeys(selected, 2):
            raise ValueError('Both model versions must complete on every selected device')
        identity = {(v['version'], v['run'], v['device']) for v in versions}
        if (len(identity) != len(versions)
                or len({v['version'] for v in versions}) != len(versions)
                or len({v['run'] for v in versions}) != len(versions)):
            raise ValueError('Model version evidence contains duplicates')
        identities.append(identity)
    if identities[0] != identities[1]:
        raise ValueError('Reload recovered different model versions or devices')
    return dict(status='passed', accelerator=accelerator,
                phases=dict.fromkeys(PHASES, 'passed'),
                devices={('gpu' if d == 'cuda' else d): 'passed' for d in selected})


def commands(args, platform, accelerator, root, source):
    python, wheel = sys.executable, str(args.wheel.resolve())
    extra = [value for path in args.workload_wheel for value in ('--workload-wheel', str(path.resolve()))]
    execution = root / 'execution'
    common = ['--wheel', wheel, '--accelerator', accelerator]
    if platform == 'snowflake':
        base = HERE / 'snowflake_notebook'
        candidate, payload = root / 'candidate.json', root / 'payload'
        runtime = 'V2.9-' + accelerator.upper() + '-PY3.12'
        return [
            [python, str(base / 'staging.py'), *common, *extra,
             '--source-ref', source, '--output', str(candidate)],
            [python, str(base / 'prepare.py'), '--accelerator', accelerator,
             '--candidate', str(candidate), '--output-dir', str(payload),
             '--workload', str(workload(platform)),
             '--workload-helper', str(workload('databricks'))],
            [python, str(base / 'launch.py'), '--execute', '--runtime', runtime,
             '--payload-manifest', str(payload / 'manifest.json'),
             '--preflight-payload', str(payload / 'preflight'),
             '--lifecycle-payload', str(payload / 'lifecycle'),
             '--output-dir', str(execution)],
            [python, str(base / 'verify.py'), '--summary', str(execution / 'summary.json')],
        ]
    candidate = args.candidate or root.parent / 'candidate.json'
    command = [python, str(HERE / (platform + '_notebook') /
                           ('lifecycle_notebook.py' if platform == 'databricks' else 'run.py')),
               '--execute', '--engine-checks-only', *common, *extra,
               '--candidate', str(candidate), '--workload', str(workload(platform)),
               '--output-dir', str(execution)]
    if platform == 'databricks':
        cluster = args.gpu_cluster_id if accelerator == 'gpu' else args.cpu_cluster_id
        command += ['--cluster-id', cluster or '<' + accelerator.upper() + '_TEST_CLUSTER_ID>']
    else:
        command += ['--scenario', 'engine']
    return [command]


def run(args):
    platforms = PLATFORMS if args.platform == 'all' else (args.platform,)
    accelerators = ACCELERATORS if args.accelerator == 'all' else (args.accelerator,)
    if args.wheel is None or len(args.workload_wheel) != 2:
        raise ValueError('Supply the Engine wheel and matching DiskGraph/kumo-pytorch wheels')
    if args.execute and 'databricks' in platforms:
        for accelerator in accelerators:
            if not getattr(args, accelerator + '_cluster_id'):
                raise ValueError('Supply the dedicated ' + accelerator + ' Databricks cluster ID')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    root = args.output_dir.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = dict(status='running' if args.execute else 'planned',
                  cases={p + '-' + a: {'status': 'planned'} for p in platforms for a in accelerators})
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.execute and args.candidate is None and any(p != 'snowflake' for p in platforms):
        subprocess.run([sys.executable, str(HERE / 'platform_candidate.py'), '--wheel',
                        str(args.wheel.resolve()), '--source-ref', source,
                        '--output', str(root / 'candidate.json')], check=True)
    for platform in platforms:
        for accelerator in accelerators:
            case = root / (platform + '-' + accelerator)
            case.mkdir(mode=0o700)
            plan = commands(args, platform, accelerator, case, source)
            (case / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
            result = dict(status='planned')
            if args.execute:
                try:
                    with (case / 'private.log').open('x') as log:
                        # Snowflake's launcher preserves explicit UI gaps in its exit status.
                        for index, command in enumerate(plan):
                            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
                            if code and not (platform == 'snowflake' and index == 2):
                                raise RuntimeError('Platform command failed; inspect private.log')
                    result = verify(platform, accelerator, case / 'execution')
                except Exception:
                    with (case / 'private.log').open('a') as log:
                        traceback.print_exc(file=log)
                    result = dict(status='failed')
                finally:
                    if platform == 'snowflake':
                        recovery = subprocess.run([sys.executable, str(HERE / 'snowflake_notebook/recover.py'),
                                                   '--execute', '--output-dir', str(case / 'execution')],
                                                  capture_output=True)
                        with (case / 'private.log').open('ab') as log:
                            log.write(recovery.stdout + recovery.stderr)
                        if recovery.returncode:
                            result = dict(status='failed')
            report['cases'][case.name] = result
            report['status'] = ('passed' if all(v['status'] == 'passed' for v in report['cases'].values())
                                else 'failed' if any(v['status'] == 'failed' for v in report['cases'].values())
                                else 'running') if args.execute else 'planned'
            (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--platform', choices=(*PLATFORMS, 'all'), default='all')
    parser.add_argument('--accelerator', choices=(*ACCELERATORS, 'all'), default='all')
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--workload-wheel', type=Path, action='append', default=[])
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--cpu-cluster-id')
    parser.add_argument('--gpu-cluster-id')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return result['status'] not in ('passed', 'planned')


if __name__ == '__main__':
    raise SystemExit(main())
