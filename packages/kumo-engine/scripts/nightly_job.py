#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare immutable nightly data and remove owned temporary work."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

SCALE_TIERS = {
    'kumo-engine-nightly-job-v2': ('100M', '1B'),
    'kumo-engine-weekly-job-v2': ('10B', '25B'),
}

WORKLOADS = ('adtech', 'delivery', 'commerce', 'streaming', 'community',
             'payments', 'fraud')


def input_path(name):
    if (not isinstance(name, str) or not name or '\\' in name
            or any(c in name for c in ('\0', '\n', '\r'))
            or PurePosixPath(name).is_absolute()
            or PurePosixPath(name).as_posix() != name
            or any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('Use normalized relative frozen input paths')
    return PurePosixPath(name)


def workload_for_case(manifest, case):
    selected = [
        row for row in manifest['cases']
        if row.get('variant', row['task']) + '-' + row['arm'] == case
    ]
    if len(selected) != 1:
        raise ValueError('Select one declared case for workload inputs')
    family = manifest['graphs'][selected[0]['graph']]['profile']
    if family not in WORKLOADS:
        raise ValueError('Unknown workload family')
    return family


def workload_files(manifest, workload):
    """Select transport files while preserving the complete parent manifest."""
    if workload not in WORKLOADS:
        raise ValueError('Unknown workload family')
    files = manifest['files']
    for name, checksum in files.items():
        path = input_path(name)
        if (path.suffix not in ('.json', '.parquet', '.bin', '')
                or not isinstance(checksum, str)
                or not re.fullmatch('[0-9a-f]{64}', checksum)):
            raise ValueError('Frozen input paths and checksums must be valid')
    graphs = {
        name: graph
        for name, graph in manifest['graphs'].items()
        if graph['profile'] == workload
    }
    cases = [case for case in manifest['cases'] if case['graph'] in graphs]
    if not graphs or not cases:
        raise ValueError('Workload has no declared graph or cases')
    needed = {manifest['reference_dependencies']}
    inventory = manifest.get('source_objects', {}).get(workload)
    if inventory:
        needed.add(inventory)
    for graph in graphs.values():
        needed.add(graph['schema'])
        for source in graph['sources'].values():
            input_path(source)
            members = {
                name
                for name in files
                if name == source or name.startswith(source + '/')
            }
            if not members:
                raise ValueError('Workload graph source has no frozen files')
            needed.update(members)
    source_manifest = f'source-manifests/{workload}.json'
    if source_manifest in files:
        needed.add(source_manifest)
    for case in cases:
        for key in ('descriptor', 'prediction_input', 'preparation_probe'):
            if key in case:
                needed.add(case[key])
        if 'catalog' in case:
            needed.add(case['catalog']['path'])
        for key in ('normal_rows', 'control_rows'):
            values = case.get(key, [])
            needed.update([values] if isinstance(values, str) else values)
        directory = input_path(case['descriptor']).parent.as_posix()
        if directory != '.':
            needed.update(name for name in files
                          if name.startswith(directory + '/'))
    if not needed <= files.keys():
        raise ValueError('Workload references an undeclared frozen input')
    return needed


def selected_workload(value, case):
    api = load_script(Path(__file__).with_name('nightly_check.py'))
    selected = [
        row for row in api.expected_cases(value) if api.case_name(row) == case
    ]
    if not selected:
        raise ValueError('Select one configured case for workload inputs')
    return api.task_profile(value, selected[0])['graph']


def configuration(raw):
    value = json.loads(raw)
    required = {
        'format', 'min_free_bytes', 'reference_identity', 'scales', 'reference',
        'policy', 'requirements'
    }
    if (not isinstance(value, dict) or not required <= set(value)
            or set(value) - required - {'archive', 'variants'}
            or not isinstance(value['format'], str)
            or value['format'] not in SCALE_TIERS
            or type(value['min_free_bytes']) is not int
            or value['min_free_bytes'] <= 0):
        raise ValueError(
            'Bind reviewed nightly objects and required free space')
    identity = value['reference_identity']
    if (not isinstance(identity, dict)
            or set(identity) != {'version', 'wheel_sha256'}
            or not isinstance(identity['version'], str)
            or not identity['version'] or any(c.isspace()
                                              for c in identity['version'])
            or not isinstance(identity['wheel_sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', identity['wheel_sha256'])):
        raise ValueError(
            'Bind the approved reference version and wheel checksum')

    def binding(item, local=False):
        if local:
            if (not isinstance(item, dict) or set(item) != {'path', 'sha256'}):
                raise ValueError(
                    'Local reviewed data need a relative path and checksum')
            path, checksum = item['path'], item['sha256']
            if (not isinstance(path, str) or not path or '\\' in path
                    or '\n' in path or '\r' in path or path.startswith('/')
                    or any(p in ('', '.', '..') for p in path.split('/'))
                    or not isinstance(checksum, str)
                    or not re.fullmatch('[0-9a-f]{64}', checksum)):
                raise ValueError(
                    'Local reviewed data need a relative path and checksum')
            return
        if (not isinstance(item, dict)
                or set(item) != {'bucket', 'region', 'key', 'sha256'}
                or any(not isinstance(v, str) or not v or '\n' in v or '\r' in v
                       for v in item.values())
                or not re.fullmatch('[a-z0-9][a-z0-9.-]*', item['bucket'])
                or not re.fullmatch('[a-z0-9-]+', item['region'])
                or not re.fullmatch('[0-9a-f]{64}', item['sha256'])
                or '\\' in item['key'] or item['key'].startswith('/')
                or any(p in ('', '.', '..') for p in item['key'].split('/'))):
            raise ValueError(
                'Nightly objects need regular S3 keys and checksums')

    for name in ('reference', 'policy', 'requirements'):
        binding(value[name], local=True)
    if value.get('archive') is not None:
        item = value['archive']
        binding(item, local=isinstance(item, dict) and 'path' in item)
    if (not isinstance(value['scales'], dict)
            or set(value['scales']) != set(SCALE_TIERS[value['format']])):
        raise ValueError('Declare both input scales for the configured tier')
    for scale, item in value['scales'].items():
        allowed = {'inputs', 'input_manifest_sha256', 'pending', 'workloads'}
        if (not isinstance(item, dict)
                or not {'inputs', 'input_manifest_sha256'} <= set(item)
                or set(item) - allowed):
            raise ValueError('Each scale needs its frozen input identity')
        if item['inputs'] is None and item['input_manifest_sha256'] is None:
            if (not isinstance(item.get('pending'), str)
                    or not item['pending'].strip()):
                raise ValueError(f'{scale} pending inputs need an explanation')
            continue
        if (not isinstance(item['input_manifest_sha256'], str)
                or not re.fullmatch('[0-9a-f]{64}',
                                    item['input_manifest_sha256'])):
            raise ValueError('Each scale needs its frozen manifest checksum')
        inputs = item['inputs']
        if isinstance(inputs, dict) and 'by_graph' in inputs:
            if (set(inputs) != {'by_graph', 'manifest'}
                    or not isinstance(inputs['by_graph'], dict)
                    or set(inputs['by_graph']) != set(WORKLOADS)):
                raise ValueError('Inputs need all seven graph archive bindings')
            binding(inputs['manifest'])
            if inputs['manifest']['sha256'] != item['input_manifest_sha256']:
                raise ValueError('The shared input manifest checksum differs')
            for archive in inputs['by_graph'].values():
                binding(archive)
        else:
            binding(inputs)
        if 'workloads' in item:
            workloads = item['workloads']
            if not isinstance(workloads,
                              dict) or set(workloads) != set(WORKLOADS):
                raise ValueError('Declare all seven workload archive bindings')
            for archive in workloads.values():
                binding(archive)
    if 'variants' in value:
        expected = dict(
            streaming=dict(task='temporal_link', graph='streaming',
                           metric='recall@50', top_k=50, raw_limit=51),
            community=dict(task='regression', graph='community', metric='mae'),
            payments=dict(task='binary', graph='payments', metric='auroc',
                          min_hops=4, max_hops=6),
            fraud=dict(task='binary', graph='fraud', metric='auroc', min_hops=4,
                       max_hops=6))
        variants = value['variants']
        if not isinstance(variants, dict) or set(variants) != set(expected):
            raise ValueError('Declare all four workload variants')
        for name, fields in expected.items():
            item = variants[name]
            if (not isinstance(item, dict)
                    or set(item) != set(fields) | {'query'}
                    or not isinstance(item['query'], str)
                    or not item['query'].strip()
                    or any(item[key] != wanted
                           or type(item[key]) is not type(wanted)
                           for key, wanted in fields.items())):
                raise ValueError(f'Invalid {name} workload variant')
    return value


def scale_configuration(value, scale):
    item = value['scales'][scale]
    if item['inputs'] is None or item['input_manifest_sha256'] is None:
        raise ValueError(
            f'{scale} inputs are pending: {item.get("pending", "not frozen")}')
    return item


def create_work(root, minimum_free):
    root = Path(root).resolve()
    free = shutil.disk_usage(root).free
    if free < minimum_free:
        raise ValueError(
            f'Insufficient free disk: {free} < {minimum_free} bytes')
    work = Path(tempfile.mkdtemp(prefix='engine-nightly-', dir=root))
    (work / '.nightly-job').write_text('kumo-engine-nightly-job-v2\n')
    return work


def owned_work(work, runner_temp):
    work = Path(work)
    marker = work / '.nightly-job'
    if (work.is_symlink()
            or work.resolve().parent != Path(runner_temp).resolve()
            or not work.name.startswith('engine-nightly-')
            or not marker.is_file() or marker.is_symlink()
            or marker.read_text() != 'kumo-engine-nightly-job-v2\n'):
        raise ValueError('Cleanup requires this runner\'s marked job directory')
    return work


def cleanup_work(work, runner_temp):
    work = owned_work(work, runner_temp)
    shutil.rmtree(work)


def stage_object(item, target, local_root):
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temporary = Path(output.name)
    try:
        if 'path' in item:
            source = local_root / item['path']
            if (source.is_symlink() or not source.is_file() or
                    not source.resolve().is_relative_to(local_root.resolve())):
                raise ValueError(
                    'Local reviewed data must stay inside the repository')
            shutil.copyfile(source, temporary)
        else:
            import boto3
            boto3.client('s3', region_name=item['region']).download_file(
                item['bucket'], item['key'], str(temporary))
        with temporary.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != item['sha256']:
            raise ValueError('Staged object checksum differs')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(value, work, scale='100M', local_root=None, case=None):
    selected = scale_configuration(value, scale)
    destination = work / 'inputs'
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Prepare requires a fresh input destination')
    separate = (isinstance(selected['inputs'], dict)
                and 'by_graph' in selected['inputs'])
    family = None
    if separate or 'workloads' in selected or case is not None:
        if not case:
            raise ValueError('Select a case for workload-specific inputs')
        family = selected_workload(value, case)
        archives = (selected['inputs']['by_graph']
                    if separate else selected.get('workloads', {}))
        if family not in archives:
            raise ValueError('Workload-specific archive is not bound')
    inputs = archives[family] if family else selected['inputs']
    local_root = (Path(local_root) if local_root is not None else
                  Path(__file__).parents[1] / 'test/nightly')
    for name in ('inputs', 'reference', 'policy', 'requirements'):
        item = inputs if name == 'inputs' else value[name]
        target = work / ('inputs.tar' if name == 'inputs' else name + '.json')
        stage_object(item, target, local_root)
    manifest = None
    if separate:
        stage_object(selected['inputs']['manifest'], work / 'manifest.json',
                     local_root)
        manifest = json.loads((work / 'manifest.json').read_text())
    with tarfile.open(work / 'inputs.tar') as archive:
        members = archive.getmembers()
        if any(not (m.isfile() or m.isdir()) for m in members):
            raise ValueError('Input archive must contain regular data files')
        names = [m.name for m in members]
        for name in names:
            input_path(name.rstrip('/'))
        if len(names) != len(set(names)):
            raise ValueError('Input archive contains duplicate paths')
        if family:
            if not separate:
                header = archive.extractfile('manifest.json').read()
                if hashlib.sha256(header).hexdigest(
                ) != selected['input_manifest_sha256']:
                    raise ValueError('Input manifest checksum differs')
                manifest = json.loads(header)
            if (manifest.get('scale') != scale
                    or workload_for_case(manifest, case) != family):
                raise ValueError(
                    'Workload archive belongs to another case or scale')
            expected = workload_files(manifest, family)
            if separate:
                if family in manifest.get('source_objects', {}):
                    expected = {
                        name
                        for name in expected if not name.startswith('raw/')
                    }
            else:
                expected |= {'manifest.json'}
            if {m.name for m in members if m.isfile()} != expected:
                raise ValueError(
                    'Workload archive contains missing or extra files')
        if sum(m.size for m in members) > shutil.disk_usage(work).free:
            raise ValueError('Insufficient free disk to extract nightly inputs')
        archive.extractall(destination, filter='data')
    (work / 'inputs.tar').unlink()
    if separate:
        (work / 'manifest.json').replace(destination / 'manifest.json')
    if manifest and family in manifest.get('source_objects', {}):
        sources = load_script(Path(__file__).with_name('nightly_sources.py'))
        started = time.monotonic()
        staged = sources.stage(destination, manifest, family,
                               input_path=input_path)
        staged.update(family=family, case=case,
                      seconds=time.monotonic() - started)
        (work / 'source-staging.json').write_text(json.dumps(staged) + '\n')


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'build':
        parser = argparse.ArgumentParser(
            description='Build one identified wheel')
        parser.add_argument('--source', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(arguments[1:])
        builder = Path(__file__).with_name('nightly_build.py')
        print(json.dumps(load_script(builder).build(args.source, args.output)))
        return 0
    if arguments and arguments[0] in ('run', 'combine'):
        if not any(
                arg == '--configuration' or arg.startswith('--configuration=')
                for arg in arguments[1:]):
            arguments += [
                '--configuration',
                str(Path(__file__).parents[1] / 'test/nightly/inputs.json')
            ]
        return load_script(
            Path(__file__).with_name('nightly_check.py')).main(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('create', 'prepare', 'cleanup'))
    parser.add_argument('--scale', choices=('100M', '1B', '10B', '25B'),
                        default=os.environ.get('NIGHTLY_SCALE', '100M'))
    parser.add_argument('--configuration', type=Path)
    parser.add_argument('--case', default=os.environ.get('NIGHTLY_CASE'))
    args = parser.parse_args(arguments)
    root = Path(os.environ['RUNNER_TEMP'])
    if args.command == 'cleanup':
        work = os.environ.get('NIGHTLY_JOB_ROOT')
        if work:
            owned_work(Path(work), root)
            evidence = Path(work + '-evidence')
            if (os.environ.get('NIGHTLY_EVIDENCE_UPLOADED') == 'true'
                    and evidence.is_dir() and not evidence.is_symlink()
                    and evidence.parent.resolve() == root.resolve()
                    and (Path(work) / '.nightly-evidence').is_file()
                    and not (Path(work) / '.nightly-evidence').is_symlink()
                    and (Path(work) / '.nightly-evidence').read_text()
                    == str(evidence)):
                shutil.rmtree(evidence)
            cleanup_work(Path(work), root)
        return
    value = configuration(args.configuration.read_text(
    ) if args.configuration else os.environ['NIGHTLY_CONFIGURATION'])
    selected = scale_configuration(value, args.scale)
    if 'workloads' in selected:
        if not args.case:
            raise ValueError('Select a case for workload-specific inputs')
        selected_workload(value, args.case)
    if args.command == 'create':
        work = create_work(root, value['min_free_bytes'])
        evidence = Path(str(work) + '-evidence')
        evidence.mkdir()
        (work / '.nightly-evidence').write_text(str(evidence))
        with open(os.environ['GITHUB_ENV'], 'a') as stream:
            stream.write(
                f'NIGHTLY_JOB_ROOT={work}\nNIGHTLY_EVIDENCE_ROOT={evidence}\n')
            stream.write('NIGHTLY_MANIFEST_SHA256=' +
                         selected['input_manifest_sha256'] + '\n')
            stream.write('NIGHTLY_SCALE=' + args.scale + '\n')
            if args.case:
                stream.write('NIGHTLY_CASE=' + args.case + '\n')
            stream.write('NIGHTLY_REFERENCE_SHA256=' +
                         value['reference']['sha256'] + '\n')
            stream.write('NIGHTLY_POLICY_SHA256=' + value['policy']['sha256'] +
                         '\n')
            stream.write('NIGHTLY_REQUIREMENTS_SHA256=' +
                         value['requirements']['sha256'] + '\n')
        print(
            json.dumps(
                dict(
                    free_bytes=shutil.disk_usage(root).free,
                    required_free_bytes=value['min_free_bytes'],
                    cpus=os.cpu_count(),
                    system_memory_bytes=os.sysconf('SC_PAGE_SIZE') *
                    os.sysconf('SC_PHYS_PAGES'))))
        return
    work = Path(os.environ['NIGHTLY_JOB_ROOT'])
    owned_work(work, root)
    prepare(
        value, work, scale=args.scale,
        local_root=(args.configuration.parent if args.configuration else None),
        case=args.case)


if __name__ == '__main__':
    raise SystemExit(main())
