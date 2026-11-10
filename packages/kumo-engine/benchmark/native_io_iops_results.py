# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add portable request-trace replay to the unchanged canonical trial archive."""

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from types import ModuleType

from io_utils import fsutil

sys.dont_write_bytecode = True


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def modules(repo: str) -> tuple[ModuleType, ModuleType, ModuleType]:
    benchmark = str(Path(repo).resolve() / 'packages/kumo-engine/benchmark')
    sys.path.insert(0, benchmark)
    import native_io_iops
    import native_io_results
    import native_io_trace
    for module in (native_io_results, native_io_iops, native_io_trace):
        if str(Path(module.__file__).resolve().parent) != benchmark:
            raise ValueError('Wrong replay source directory')
    return native_io_results, native_io_iops, native_io_trace


def seal(path: str, records: list[dict]) -> dict:
    payload = ''.join(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                 allow_nan=False) + '\n' for row in records).encode()
    data = gzip.compress(payload, compresslevel=6, mtime=0)
    fsutil.write_binary(path, data)
    return dict(sha256=hashlib.sha256(data).hexdigest(), records=len(records))


def unseal(path: str, oracle: dict) -> list[dict]:
    data = fsutil.read_binary(path)
    if hashlib.sha256(data).hexdigest() != oracle['sha256']:
        raise ValueError('Archive hash mismatch')
    records = [json.loads(line) for line in gzip.decompress(data).decode().splitlines()]
    if len(records) != oracle['records']:
        raise ValueError('Archive cardinality mismatch')
    return records


def trial_records(directory: str, canonical: ModuleType) -> list[dict]:
    manifest = canonical.load(directory + '/manifest.json')
    return unseal(directory + '/raw-trials.jsonl.gz', dict(
        sha256=manifest['archive_sha256'], records=manifest['archive_records']))


def source_files(repo: str, canonical: ModuleType) -> dict:
    paths = ['packages/kumo-engine/' + path for path in canonical.code_oracle(repo)]
    paths += ['packages/kumo-engine/benchmark/' + name for name in (
        'native_io_results.py', 'native_io_tasks.py', 'native_io_iops.py',
        'native_io_phases.py', 'native_io_trace.py', 'native_io_trace.c')]
    return {path: dict(text=fsutil.read_text(repo + '/' + path)) for path in paths}


def validate_sources(repo: str, sources: dict) -> None:
    for path, record in sources.items():
        relative = Path(path)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid source path')
        if digest(record['text']) != record['sha256']:
            raise ValueError('Source snapshot hash mismatch')
        if digest(fsutil.read_text(repo + '/' + path)) != record['sha256']:
            raise ValueError('Replay requires the exact measured source bytes')


def source_path(path: str) -> str:
    prefix = 'packages/kumo-engine/'
    parts = path.split(prefix)
    if len(parts) != 2 or '..' in Path(parts[1]).parts:
        raise ValueError('Invalid measured source path')
    return prefix + parts[1]


def measurement(row: dict, trace: dict) -> dict:
    source = dict(row['matrix_spec']['source_provenance'])
    source['file_sha256'] = {source_path(path): value for path, value in source['file_sha256'].items()}
    return dict(source=source,
                native_sha256=row['provenance']['native_sha256'],
                build=row['provenance']['build'], config=row['iops']['config'],
                kernel=trace['host_kernel'], btf_sha256=trace['btf_sha256'],
                btf_offsets=trace['btf_offsets'],
                device={key: trace['device'][key] for key in (
                    'path', 'major', 'minor', 'io_stat_key', 'sysfs', 'filesystem_device')})


def validate_envelope(record: dict, envelope: dict) -> tuple[dict, dict, dict]:
    if digest(record['source_text']) != record['source_sha256']:
        raise ValueError('Original row hash mismatch')
    row = json.loads(record['source_text'])
    spec, saved = row['matrix_spec'], row['iops']
    expected = dict(trial_source_file=record['source_file'],
                    trial_source_sha256=record['source_sha256'], variant=record['variant'],
                    trace_source_file=spec['iops_paths']['trace'],
                    stop_source_file=spec['iops_paths']['stop'])
    if any(envelope[key] != value for key, value in expected.items()):
        raise ValueError('Trace envelope belongs to a different original row/path')
    for kind in ('trace', 'stop'):
        if digest(envelope[kind + '_source_text']) != envelope[kind + '_source_sha256']:
            raise ValueError('Original sidecar text hash mismatch')
    if (saved['trace'] != envelope['trace_source_file'] or
            saved['trace_sha256'] != envelope['trace_source_sha256']):
        raise ValueError('Trace differs from the original accepted row')
    trace = json.loads(envelope['trace_source_text'])
    stop = json.loads(envelope['stop_source_text'])
    if type(stop['exit_code']) is not int or stop['exit_code'] != 0 or stop['result_path'] != spec['result']:
        raise ValueError('STOP must prove exit0 for the exact original result')
    return row, trace, stop


def derive_rates(trace: dict, rates: dict) -> dict:
    output = {}
    for phase, rate in rates.items():
        count, seconds = trace['counts'][phase], trace['phases'][phase]['seconds']
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid physical phase denominator')
        if any(not count[kind] and count[byte_field] for kind, byte_field in (
                ('reads', 'read_bytes'), ('writes', 'write_bytes'))):
            raise ValueError('Completed payload without a completed request')
        native, native_rate = (rate['application_completed_read_sqes'],
                               rate['application_read_requests_per_second'])
        workload_seconds = rate['workload_seconds']
        if (type(workload_seconds) not in (int, float) or
                not math.isfinite(workload_seconds) or workload_seconds <= 0):
            raise ValueError('Invalid native workload denominator')
        if (native is None) != (native_rate is None):
            raise ValueError('Inconsistent native counter availability')
        if native is not None and (
                type(native) is not int or native < 0 or
                type(native_rate) not in (int, float) or not math.isfinite(native_rate) or
                native_rate < 0 or native_rate != native / workload_seconds):
            raise ValueError('Invalid native completed-SQE counter/rate')
        output[phase] = dict(
            physical_read_iops=count['reads'] / seconds,
            physical_write_iops=count['writes'] / seconds,
            physical_total_iops=(count['reads'] + count['writes']) / seconds,
            physical_interval_seconds=seconds, physical_completed_reads=count['reads'],
            physical_completed_writes=count['writes'], physical_read_bytes=count['read_bytes'],
            physical_write_bytes=count['write_bytes'],
            mean_completed_read_payload_kib=(count['read_bytes'] / count['reads'] / 1024
                                             if count['reads'] else None),
            native_completed_read_sqes=rate['application_completed_read_sqes'],
            native_read_requests_per_second=rate['application_read_requests_per_second'],
            workload_interval_seconds=rate['workload_seconds'])
    return output


def validate_record(record: dict, envelope: dict, sources: dict,
                    oracle: dict, iops: ModuleType, tracer: ModuleType) -> dict:
    row, trace, _ = validate_envelope(record, envelope)
    spec, saved = row['matrix_spec'], row['iops']
    config, provenance = spec['iops_config'], spec['source_provenance']
    if any(row[key] != spec[key] for key in ('task', 'regime', 'backend', 'repeat')):
        raise ValueError('Row identity differs from the executed control')
    if row['provenance']['build']['native']['sha256'] != row['provenance']['native_sha256']:
        raise ValueError('Embedded native build provenance differs from loaded library')
    if (saved['status'] != 'PASS' or saved['config'] != config or
            config['trace_mode'] != 'on' or config['phase_markers'] != '1'):
        raise ValueError('Only identically controlled traced observations qualify')
    if (provenance['git_status'] or provenance['tracked_diff_sha256'] != digest('') or
            measurement(row, trace) != oracle):
        raise ValueError('Mixed measured source/native/build/BTF/device/instrumentation')
    for path, expected in provenance['file_sha256'].items():
        if sources[source_path(path)]['sha256'] != expected:
            raise ValueError('Measured source pin differs from replay source oracle')
    for name, field in [('native_io_trace.py', 'collector_sha256'),
                        ('native_io_trace.c', 'bpf_source_sha256')]:
        if trace[field] != sources['packages/kumo-engine/benchmark/' + name]['sha256']:
            raise ValueError('Collector source differs from the measured source pin')
    if (trace['request_capacity'] != config['request_capacity'] or
            (config['device'] is not None and trace['device']['path'] != config['device'])):
        raise ValueError('Collector capacity/device differs from requested controls')
    tracer.validate_collection(trace, row, saved['gate'])
    iops.validate_result(row, spec)
    iops.validate_attribution(trace, row)
    rates = iops.phase_rates(row, trace)
    if rates != saved['phase_rates']:
        raise ValueError('Saved phase rates differ from raw counters/clocks')
    return dict(variant=record['variant'], source_file=record['source_file'],
                source_sha256=record['source_sha256'], task=row['task'],
                regime=row['regime'], backend=row['backend'], repeat=row['repeat'],
                phases=derive_rates(trace, rates))


def observations(records: list[dict], envelopes: list[dict], sources: dict,
                 oracle: dict, iops: ModuleType, tracer: ModuleType) -> dict:
    by_row = {item['trial_source_file']: item for item in envelopes}
    if len(by_row) != len(envelopes) or len(records) != len(envelopes):
        raise ValueError('Expected exactly one unique trace/STOP per original row')
    output = []
    process_ids, cgroups, service_units = set(), set(), set()
    for record in records:
        envelope = by_row.pop(record['source_file'])
        row, _, _ = validate_envelope(record, envelope)
        for value, seen in (
                (row['native_io']['fit']['after']['pid'], process_ids),
                (row['iops']['gate']['cgroup_id'], cgroups),
                (row['iops']['service_unit'], service_units)):
            if value in seen:
                raise ValueError('Reused model process, cgroup, or service identity')
            seen.add(value)
        item = validate_record(record, envelope, sources, oracle, iops, tracer)
        output.append(item)
    if by_row:
        raise ValueError('Unbound trace sidecars')
    validate_grid(output)
    return dict(status='INSTRUMENTED_OBSERVATIONS', records=output, medians=median_rates(output),
                definition='Median of five per-trial rates; physical uses BPF seconds, native uses workload seconds',
                payload_definition='Completed-request payload, not bytes transferred exclusively in the window; zero-read trials undefined',
                latency_scope='Original untraced latency experiment remains separate; stress overhead qualification remains HOLD')


def validate_grid(rows: list[dict]) -> None:
    keys = [(row['task'], row['regime'], row['backend'], row['repeat']) for row in rows]
    expected = {(task, regime, backend, repeat)
                for task in ('binary', 'regression', 'temporal_link')
                for regime in ('resident', 'constrained')
                for backend in ('mmap', 'buffered', 'direct') for repeat in range(5)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Expected exactly six task/regime groups times three backends times five repetitions')


def median_rates(output: list[dict]) -> dict:
    groups = {}
    for item in output:
        key = '/'.join((item['variant'], item['task'], item['regime'], item['backend']))
        groups.setdefault(key, []).append(item)
    medians = {}
    metrics = ('physical_read_iops', 'physical_write_iops', 'physical_total_iops',
               'native_read_requests_per_second', 'mean_completed_read_payload_kib')
    for key, rows in groups.items():
        if len(rows) != 5 or {row['repeat'] for row in rows} != set(range(5)):
            raise ValueError('Each IOPS group requires five distinct repetitions')
        phases = {}
        for phase in ('training', 'e2e'):
            values = {}
            for metric in metrics:
                data = [row['phases'][phase][metric] for row in rows]
                if metric == 'mean_completed_read_payload_kib':
                    defined = [value for value in data if value is not None]
                    values[metric] = statistics.median(defined) if defined else None
                    values['payload_defined_trials'] = len(defined)
                    values['payload_total_trials'] = len(rows)
                    continue
                if any(value is None for value in data) and not all(value is None for value in data):
                    raise ValueError('Mixed availability of native mmap counters')
                values[metric] = None if data[0] is None else statistics.median(data)
            phases[phase] = values
        medians[key] = phases
    return medians


def evidence_files(path: str, prefix: str = '') -> list[dict]:
    output = []
    for entry in fsutil.listdir(path) if fsutil.isdir(path) else [path]:
        if fsutil.isdir(path) and not Path(entry).resolve().is_relative_to(Path(path).resolve()):
            raise ValueError('Evidence symlink escapes its metadata directory')
        name = prefix + Path(entry).name
        if fsutil.isdir(entry):
            if name.endswith(('-prepared', '-work')):
                raise ValueError('Refuse to archive graph/model work directories')
            output.extend(evidence_files(entry, name + '/'))
        else:
            if fsutil.filesize(entry) > 32 * 1024**2:
                raise ValueError('Evidence exceeds metadata-only file limit')
            text = fsutil.read_text(entry)
            output.append(dict(path=name, source_file=entry, text=text, sha256=digest(text)))
    return output


def validate_evidence(name: str, files: list[dict]) -> None:
    objects = [json.loads(item['text']) for item in files if item['path'].endswith('.json')]
    if name == 'stress20':
        trials = [item for item in objects if isinstance(item, dict) and item.get('control_fixture')]
        if len(trials) != 20:
            raise ValueError('Preserve all twenty stress control records')
    if name == 'public8':
        trials = [item for item in objects if isinstance(item, dict) and 'matrix_spec' in item]
        keys = {(row['backend'], row['repeat'], row['iops']['config']['trace_mode']) for row in trials}
        expected = {(backend, repeat, mode) for backend in ('mmap', 'direct')
                    for repeat in range(2) for mode in ('on', 'off')}
        if len(trials) != 8 or keys != expected:
            raise ValueError('Preserve all eight public overhead control records')


def package(args, canonical: ModuleType, iops: ModuleType, tracer: ModuleType) -> dict:
    if fsutil.exists(args.output_dir):
        raise FileExistsError(args.output_dir)
    if not {'stress20', 'public8'} <= {entry.partition('=')[0] for entry in args.evidence}:
        raise ValueError('Preserve the separate stress20 and public8 overhead evidence')
    fsutil.makedirs(args.output_dir)
    native_args = argparse.Namespace(**vars(args))
    native_args.output_dir, native_args.diagnostic = args.output_dir + '/rows', False
    canonical.package(native_args, canonical.driver(args.repo_root))
    records = trial_records(native_args.output_dir, canonical)
    if len(records) != 90:
        raise ValueError('Expected the complete 90-observation experiment')
    envelopes = []
    for record in records:
        row = json.loads(record['source_text'])
        item = dict(variant=record['variant'], trial_source_file=record['source_file'],
                    trial_source_sha256=record['source_sha256'])
        for kind in ('trace', 'stop'):
            path = row['matrix_spec']['iops_paths'][kind]
            text = fsutil.read_text(path)
            item.update({kind + '_source_file': path, kind + '_source_text': text,
                         kind + '_source_sha256': digest(text)})
        envelopes.append(item)
    sources = source_files(args.repo_root, canonical)
    for item in sources.values():
        item['sha256'] = digest(item['text'])
    validate_sources(args.repo_root, sources)
    oracle = measurement(json.loads(records[0]['source_text']),
                         json.loads(envelopes[0]['trace_source_text']))
    derived = observations(records, envelopes, sources, oracle, iops, tracer)
    manifest = dict(schema_version=1, helper_sha256=digest(fsutil.read_text(__file__)),
                    sources=sources, measurement_oracle=oracle,
                    traces=seal(args.output_dir + '/raw-traces.jsonl.gz', envelopes), evidence={})
    for entry in args.evidence:
        name, separator, path = entry.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in manifest['evidence']:
            raise ValueError('Use unique simple NAME=PATH evidence names')
        files = evidence_files(path)
        validate_evidence(name, files)
        manifest['evidence'][name] = seal(args.output_dir + '/evidence-' + name + '.jsonl.gz', files)
    canonical.dump(args.output_dir + '/observations.json', derived)
    manifest['observations_sha256'] = digest(fsutil.read_text(args.output_dir + '/observations.json'))
    canonical.dump(args.output_dir + '/iops-manifest.json', manifest)
    return dict(status='PACKAGED', records=len(records), groups=len(derived['medians']))


def replay(args, canonical: ModuleType, iops: ModuleType, tracer: ModuleType) -> dict:
    root = args.replay
    manifest = canonical.load(root + '/iops-manifest.json')
    if not {'stress20', 'public8'} <= manifest['evidence'].keys():
        raise ValueError('Missing required overhead qualification evidence')
    if digest(fsutil.read_text(__file__)) != manifest['helper_sha256']:
        raise ValueError('IOPS helper source hash mismatch')
    validate_sources(args.repo_root, manifest['sources'])
    native_args = argparse.Namespace(repo_root=args.repo_root, replay=root + '/rows')
    canonical_result = canonical.replay(native_args, canonical.driver(args.repo_root))
    records = trial_records(root + '/rows', canonical)
    if len(records) != 90:
        raise ValueError('Expected the complete 90-observation experiment')
    envelopes = unseal(root + '/raw-traces.jsonl.gz', manifest['traces'])
    derived = observations(records, envelopes, manifest['sources'], manifest['measurement_oracle'], iops, tracer)
    text = fsutil.read_text(root + '/observations.json')
    if digest(text) != manifest['observations_sha256'] or json.loads(text) != derived:
        raise ValueError('IOPS observations do not replay from raw counters')
    for name, oracle in manifest['evidence'].items():
        files = unseal(root + '/evidence-' + name + '.jsonl.gz', oracle)
        if len({item['path'] for item in files}) != len(files):
            raise ValueError('Duplicate evidence path')
        if any(digest(item['text']) != item['sha256'] for item in files):
            raise ValueError('Original overhead evidence text hash mismatch')
        validate_evidence(name, files)
    return dict(status='IOPS_REPLAY_PASS', canonical=canonical_result,
                records=len(records), groups=len(derived['medians']))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--variant', action='append', default=[])
    parser.add_argument('--output-dir')
    parser.add_argument('--replay')
    parser.add_argument('--evidence', action='append', default=[])
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--order-seed', type=int, default=20261109)
    parser.add_argument('--prediction-atol', type=float, default=1e-6)
    args = parser.parse_args()
    if bool(args.output_dir) == bool(args.replay) or args.repetitions != 5:
        parser.error('Select packaging or replay; IOPS observations require five repetitions')
    canonical, iops, tracer = modules(args.repo_root)
    result = replay(args, canonical, iops, tracer) if args.replay else package(args, canonical, iops, tracer)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
