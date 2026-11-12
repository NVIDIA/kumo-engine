# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed-wheel desktop qualification, without a registry or experiment tracker."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import resource
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
SUITES = {
    'quickstart': ['test_readme_quickstart.py'],
    'retrieval':
    ['test_temporal_link_candidates.py', 'test_static_link_split.py'],
    'exports': [
        'test_training_table_export.py::test_node_export_roundtrip',
        'test_training_table_export.py::test_link_export_roundtrip',
        'test_training_table_export.py::test_custom_training_table_declares_integer_timestamp_unit',
        'test_prediction_output_columns.py::test_link_prediction_respects_explicit_count',
        'test_prediction_output_columns.py::test_link_embeddings_do_not_require_count',
    ],
    'warm_start':
    ['test_warm_start.py::test_bundle_warm_start_fit_reload_and_predict'],
    'sampling': ['test_static_link_pseudo_time.py'],
}


@cache
def helpers():
    spec = importlib.util.spec_from_file_location(
        'desktop_lifecycle_helpers', DIRECTORY / 'wheel_mlops_lifecycle.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(destination):
    repo = DIRECTORY.parents[2]
    names = {name.split('::')[0] for suite in SUITES.values() for name in suite}
    names.update({
        'desktop_validation.py', 'desktop_worker.py', 'desktop_evidence.py',
        'conftest.py', 'wheel_mlops_lifecycle.py', 'wheel_gpu_smoke.py'
    })
    hashes = {}
    for name in sorted(names):
        source = DIRECTORY / name
        target = destination / source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(source.relative_to(repo))] = helpers().digest(source)
    for name in ('README.md', 'packages/kumo-engine/README.md'):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, target)
        hashes[name] = helpers().digest(target)
    helpers().write(destination / 'recipe.json', hashes)


def junit_verdict(path):
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    counts = {
        key: sum(int(s.get(key, '0')) for s in suites)
        for key in ('tests', 'errors', 'failures', 'skipped')
    }
    status = ('failed'
              if not counts['tests'] or counts['errors'] or counts['failures']
              else 'blocked' if counts['skipped'] else 'passed')
    return dict(status=status, **counts)


def qualification_status(cases, profiles):
    values = list(cases.values()) + [
        profiles.get(name, {'status': 'blocked'}) for name in ('cpu', 'mps')
    ]
    if not cases or any(item['status'] == 'failed' for item in values):
        return 'failed'
    return 'passed' if all(item['status'] == 'passed'
                           for item in values) else 'blocked'


def run(args):
    h = helpers()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='failed', cases={}, profiles={}, command=sys.argv)
    report['scope'] = 'Desktop recovery and retrieval; no MLOps execution'
    try:
        h.require(
            sys.platform == 'darwin' and platform.machine().lower() == 'arm64',
            'Desktop qualification requires native macOS ARM64')
        report['environment'] = h.check_install(args.wheel,
                                                args.expected_version,
                                                platform_name='macos-arm64')
        import torch
        report['hardware'] = dict(
            system=platform.platform(),
            mps_available=torch.backends.mps.is_available())
        report['recipe'] = json.loads(
            DIRECTORY.parents[2].joinpath('recipe.json').read_text())

        def execute(name, command, cwd=DIRECTORY, timeout=1200):
            before = time.monotonic()
            result = dict(status='failed', command=command)
            try:
                # The shared runner kills timed-out workers and their descendants.
                with (root / (name + '.log')).open('w') as log:
                    code = h.execute(command, cwd, log, timeout)
                result.update(returncode=code,
                              status='passed' if code == 0 else 'failed')
            except Exception:
                result['error'] = traceback.format_exc()
            result['seconds'] = time.monotonic() - before
            report['cases'][name] = result
            h.write(root / 'summary.json', report)
            return result

        for device in ('cpu', 'mps'):
            if device == 'mps' and not report['hardware']['mps_available']:
                report['profiles'][device] = dict(
                    status='blocked', reason='Runner exposes no MPS device')
                continue
            profile = root / device
            profile.mkdir()
            command = [
                sys.executable, '-I',
                str(DIRECTORY / 'desktop_worker.py'), '--root',
                str(profile), '--accelerator', device
            ]
            produce = execute(device + '-produce', command + ['produce'])
            if produce['status'] == 'passed':
                shutil.rmtree(profile / 'producer')
                h.require(not (profile / 'producer').exists(),
                          'Producer state survived restart')
                consume = execute(device + '-consume', command + ['consume'])
                report['profiles'][device] = dict(status=consume['status'])
            else:
                report['profiles'][device] = dict(status='failed')

        for name, selectors in SUITES.items():
            junit = root / (name + '-junit.xml')
            receipt = root / (name + '-devices.json')
            # Configuration selects CPU through the public Trainer argument;
            # the evidence plugin calls the real constructor and runtime.
            os.environ['DESKTOP_DEVICE_RECEIPT'] = str(receipt)
            command = [
                sys.executable, '-m', 'pytest', '-p', 'desktop_evidence',
                '--confcutdir=.', '-q', '-s', '--junitxml=' + str(junit),
                *selectors
            ]
            result = execute(name, command)
            if junit.is_file():
                result.update(junit_verdict(junit))
            else:
                result.update(status='failed', reason='JUnit evidence missing')
            if result.get('returncode') != 0:
                result['status'] = 'failed'
            result['device_receipt'] = str(receipt)
            h.require(receipt.is_file(), 'Native device evidence is missing')
            devices = json.loads(receipt.read_text())['devices']
            h.require(devices and all(d['device'] == 'cpu' for d in devices),
                      'Native CPU execution was not observed')
        command = [
            sys.executable, '-I',
            str(DIRECTORY / 'desktop_worker.py'), '--root',
            str(root / 'cpu'), '--accelerator', 'cpu'
        ]
        execute('incomplete-artifacts', command + ['incomplete'])
        version = importlib.metadata.version('faiss-cpu')
        try:
            execute('remove-faiss', [
                sys.executable, '-I', '-m', 'pip', '--isolated', 'uninstall',
                '-y', 'faiss-cpu'
            ])
            execute('missing-retrieval-dependency', command + ['no-faiss'])
        finally:
            execute('restore-faiss', [
                sys.executable, '-I', '-m', 'pip', '--isolated', 'install',
                'faiss-cpu==' + version
            ])
        execute('final-dependencies',
                [sys.executable, '-I', '-m', 'pip', '--isolated', 'check'])
        report['status'] = qualification_status(report['cases'],
                                                report['profiles'])
        report['mps_qualification'] = report['profiles'].get(
            'mps', {'status': 'blocked'})
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        report['seconds'] = time.monotonic() - started
        scale = 1024 if sys.platform == 'darwin' else 1
        report['resources'] = dict(
            maxrss_parent_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / scale, maxrss_children_kib=resource.getrusage(
                resource.RUSAGE_CHILDREN).ru_maxrss / scale,
            scope='Process maxima; not simultaneous process-tree memory')
        h.write(root / 'summary.json', report)
        print(
            json.dumps(
                dict(status=report['status'],
                     summary=str(root / 'summary.json'))))
    return 0 if report['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-to', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--expected-version')
    args = parser.parse_args()
    if args.prepare_to:
        prepare(args.prepare_to)
        return 0
    if not all((args.output_dir, args.wheel, args.expected_version)):
        parser.error(
            'Execution requires --output-dir, --wheel, --expected-version')
    args.wheel = args.wheel.resolve(strict=True)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
