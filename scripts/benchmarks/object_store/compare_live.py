# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare existing, identical S3/ADLS tables through production bulk readers.

This script only reads cloud objects; provisioning and fixture copying happen
separately. Supply trusted, complete-table manifests containing SHA256 and
footer row counts. Use --providers adls or --providers s3 for native-host
single-provider phases; the default runs alternating provider pairs. Each fresh
child prepares local Parquet through the actual provider fetch callback and
shared core raw-copy path, without the engine.

Setup/authentication and manifest identity preflight are outside the timer.
The preparation timer includes production listing/version guards and transfers.
The nested transfer timer includes production fetch's own metadata, retries,
and checks; the remainder is NOT a pure metadata-latency measurement. SHA256,
footer verification, final manifest checks, and cleanup are outside both timers.
There are no request hooks/proxies: request/retry counts and token refresh
counts are explicitly uninstrumented. Cloud placement, TLS, auth, retries, and
provider request geometry differ; this measures end-to-end production paths.

Per-table manifest schema (all objects, not a sampled prefix):
{schema_version:1, table, bucket, s3_prefix, adls_account, filesystem,
 adls_prefix, endpoint_suffix?:"dfs.core.windows.net",
 files:[{relative_path,s3_key,bytes,rows,sha256,source_etag,adls_etag}],
 totals:{files,bytes,rows}}
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(',', ':')) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


OUTPUT_HASH_CHUNK_BYTES = 1 << 20
MAX_VERIFICATION_WORKERS = 8


def output_sha256(path):
    """Hash only a downloaded output; advise away bytes after hashing them.

    Provenance/code/manifest hashes continue to use sha256 without cache advice.
    One reusable buffer bounds each active hash worker's Python payload.
    Advice is opportunistic and never changes file contents or correctness.
    """
    digest = hashlib.sha256()
    buffer = bytearray(OUTPUT_HASH_CHUNK_BYTES)
    view = memoryview(buffer)
    supported = hasattr(os, 'posix_fadvise') and hasattr(
        os, 'POSIX_FADV_DONTNEED')
    advice_calls = 0
    advice_errors = set()
    offset = 0
    with path.open('rb', buffering=0) as source:
        while True:
            amount = source.readinto(buffer)
            if not amount:
                break
            digest.update(view[:amount])
            if supported:
                try:
                    os.posix_fadvise(source.fileno(), offset, amount,
                                     os.POSIX_FADV_DONTNEED)
                    advice_calls += 1
                except OSError as error:
                    advice_errors.add(error.errno)
            offset += amount
    return digest.hexdigest(), dict(supported=supported, calls=advice_calls,
                                    error_numbers=sorted(advice_errors))


def verify_outputs(output, entries, requested_workers):
    import pyarrow.parquet as pq
    from kumo_core.sources.cpu import cpu_budget

    expected = {entry['relative_path'] for entry in entries}
    if {
            path.relative_to(output).as_posix()
            for path in output.rglob('*') if path.is_file()
    } != expected:
        raise ValueError('Local staged membership differs from manifest')
    # Prefer recently completed/warm files; retain deterministic tie breaking.
    ordered = sorted(
        entries, key=lambda entry: ((output / entry['relative_path']).stat(
        ).st_mtime_ns, entry['relative_path']), reverse=True)
    workers = min(requested_workers, MAX_VERIFICATION_WORKERS, cpu_budget(),
                  len(entries))
    if workers < 1:
        raise ValueError(
            'Verification needs a positive worker count and nonempty table')

    def verify(entry):
        path = output / entry['relative_path']
        if path.is_symlink():
            raise ValueError('Staged output must not be a symbolic link')
        digest, advice = output_sha256(path)
        actual = dict(relative_path=entry['relative_path'],
                      bytes=path.stat().st_size,
                      rows=pq.ParquetFile(path).metadata.num_rows,
                      sha256=digest)
        if any(actual[key] != entry[key]
               for key in ('bytes', 'rows', 'sha256')):
            raise ValueError(
                f'Local SHA256/size/footer mismatch: {entry["relative_path"]}')
        return actual, advice

    verified = []
    advice_records = []
    entries_to_submit = iter(ordered)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        try:
            for entry in entries_to_submit:
                pending.add(pool.submit(verify, entry))
                if len(pending) == workers:
                    break
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    actual, advice = future.result()
                    verified.append(actual)
                    advice_records.append(advice)
                for _ in done:
                    entry = next(entries_to_submit, None)
                    if entry is None:
                        break
                    pending.add(pool.submit(verify, entry))
        except BaseException:
            for future in pending:
                future.cancel()
            # The executor waits for active hash readers before caller cleanup.
            raise
    verified.sort(key=lambda entry: entry['relative_path'])
    config = dict(
        requested_workers=requested_workers, effective_workers=workers,
        maximum_workers=MAX_VERIFICATION_WORKERS,
        hash_chunk_bytes=OUTPUT_HASH_CHUNK_BYTES,
        order='newest output mtime_ns first, reverse relative_path for ties',
        output_record_order='relative_path ascending',
        cache_advice='POSIX_FADV_DONTNEED on output ranges after SHA256 update',
        cache_advice_supported=all(record['supported']
                                   for record in advice_records),
        cache_advice_calls=sum(record['calls'] for record in advice_records),
        cache_advice_error_numbers=sorted({
            number
            for record in advice_records
            for number in record['error_numbers']
        }))
    return verified, config


def relative_path(value):
    path = PurePosixPath(value)
    if (not value or value != path.as_posix() or path.is_absolute()
            or any(part in ('.', '..') for part in path.parts)
            or '\\' in value):
        raise ValueError(
            'Manifest paths must be canonical relative POSIX paths')
    return value


def read_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported live-benchmark manifest version')
    for field in ('s3_prefix', 'adls_prefix'):
        manifest[field] = relative_path(manifest[field].rstrip('/'))
    if not manifest['files']:
        raise ValueError('A full table must contain at least one file')
    names = set()
    for entry in manifest['files']:
        name = relative_path(entry['relative_path'])
        if name in names or not name.endswith('.parquet'):
            raise ValueError('Manifest needs distinct Parquet paths')
        names.add(name)
        if entry['s3_key'] != manifest['s3_prefix'] + '/' + name:
            raise ValueError('S3 key is outside the complete table prefix')
        if (type(entry['bytes']) is not int or entry['bytes'] <= 0
                or type(entry['rows']) is not int or entry['rows'] < 0
                or len(entry['sha256']) != 64
                or any(char not in '0123456789abcdef'
                       for char in entry['sha256']) or not entry['source_etag']
                or not entry['adls_etag']):
            raise ValueError(
                'Manifest needs sizes, rows, SHA256, and both pinned ETags')
    expected = dict(files=len(names),
                    bytes=sum(item['bytes'] for item in manifest['files']),
                    rows=sum(item['rows'] for item in manifest['files']))
    if manifest['totals'] != expected:
        raise ValueError('Manifest totals do not match its objects')
    return manifest


def memory():
    result = {}
    for filename, fields in [('status', {'VmHWM', 'VmRSS'}),
                             ('smaps_rollup', {'Pss'})]:
        for line in Path('/proc/self', filename).read_text().splitlines():
            key, separator, value = line.partition(':')
            if separator and key in fields:
                result[key + '_kib'] = int(value.split()[0])
    return result


def metadata():
    result: dict[str, object] = dict(python=sys.version,
                                     hostname=socket.gethostname(),
                                     platform=platform.platform(),
                                     cpu_count=os.cpu_count())
    result['dependencies'] = {
        name: importlib.metadata.version(name)
        for name in ('pyarrow', 'fsspec', 'boto3', 'botocore', 'azure-core',
                     'azure-identity', 'azure-storage-file-datalake',
                     'azure-storage-blob', 'requests', 'urllib3')
    }
    # Hash the implementation actually imported by the child, including dirty
    # checkout changes that a commit identifier alone would not identify.
    import kumo_adls.sdk_filesystem
    import kumo_adls.transfer
    import kumo_core.sources.object_store
    import kumo_s3.access
    result['provider_code'] = {
        module.__name__:
        dict(path=module.__file__, sha256=sha256(Path(module.__file__)))
        for module in (kumo_adls.sdk_filesystem, kumo_adls.transfer,
                       kumo_core.sources.object_store, kumo_s3.access)
    }
    return result


def make_reader(provider, manifest, args):
    import pyarrow as pa
    from kumo_core.sources.cpu import copy_workers
    from kumo_core.sources.object_store import ObjectStoreSourceReader
    from kumo_core.types.enums import ConnectorType

    pa.set_io_thread_count(args.worker_count)
    if copy_workers(args.worker_count) != args.worker_count:
        raise ValueError('Host CPU/copy ceiling is below the requested workers')
    if provider == 's3':
        from kumo_s3.access import source_access
        from kumo_s3.session import S3_MAX_RETRIES
        if args.s3_anonymous:
            from types import SimpleNamespace
            session = SimpleNamespace(get_credentials=lambda: None,
                                      region_name=args.s3_region)
        else:
            import boto3
            session = boto3.Session(region_name=args.s3_region)
        access = source_access(manifest['bucket'], session=session,
                               default_region=args.s3_region,
                               allow_anonymous=args.s3_anonymous)
        root = manifest['bucket'] + '/' + manifest['s3_prefix']
        reader = ObjectStoreSourceReader(access.filesystem, root=root,
                                         file_format='parquet',
                                         file_suffix='parquet',
                                         fetch=access.fetch,
                                         object_identity=access.object_identity,
                                         source_type=ConnectorType.S3)
        config = dict(
            authentication='anonymous'
            if args.s3_anonymous else 'ambient boto3 credentials',
            retry_strategy='AwsDefaultS3RetryStrategy',
            retry_max_attempts=S3_MAX_RETRIES,
            arrow_io_threads=pa.io_thread_count(),
            fetch='kumo_s3.access.source_access.fetch',
            http_pool='native Arrow public API does not expose pool size')
        return reader, config, lambda: None

    from kumo_adls import session, transfer
    from kumo_adls.backend import ADLSSourceReader
    from kumo_adls.source_access import source_access
    if transfer.CONNECTIONS != args.worker_count:
        raise ValueError(
            'ADLS transfer ceiling differs from the file-worker ceiling')
    suffix = manifest.get('endpoint_suffix', 'dfs.core.windows.net')
    access = source_access(None, manifest['adls_account'], suffix)
    authority = f'{manifest["filesystem"]}@{manifest["adls_account"]}.{suffix}'
    reader = ADLSSourceReader(access.filesystem,
                              root=authority + '/' + manifest['adls_prefix'],
                              file_format='parquet', file_suffix='parquet',
                              fetch=access.fetch,
                              object_identity=access.object_identity,
                              source_type=ConnectorType.ADLS)
    config = dict(
        authentication='ambient DefaultAzureCredential',
        credential_selection=os.environ.get('AZURE_TOKEN_CREDENTIALS',
                                            'default chain'),
        transfer_ceiling=transfer.CONNECTIONS, http_pool=transfer.CONNECTIONS,
        arrow_io_threads=pa.io_thread_count(),
        bulk_chunk_bytes=transfer.BULK_DOWNLOAD_CHUNK_BYTES,
        bulk_max_workers=transfer.MAX_BULK_DOWNLOAD_CONCURRENCY,
        fetch='kumo_adls.source_access.source_access.fetch',
        retries='Azure SDK production defaults; actual retries uninstrumented')
    return reader, config, session.close


def identity_token(value):
    # SDK List Paths and HEAD can spell one opaque token with/without quotes.
    return value[1:-1] if value.startswith('"') and value.endswith(
        '"') else value


def inventory(reader, manifest, provider, workers):
    found = reader._objects('', reader.root,
                            max_entries=4 * len(manifest['files']) + 1024,
                            include_hidden=True)
    expected = {entry['relative_path']: entry for entry in manifest['files']}
    actual = {entry.path[len(reader.root) + 1:]: entry for entry in found}
    if set(actual) != set(expected):
        raise ValueError(
            'Remote table membership differs from the complete manifest')
    ordered = sorted(actual)
    identities = reader._identities([actual[name] for name in ordered], workers)
    key = 'source_etag' if provider == 's3' else 'adls_etag'
    recorded = {}
    for name, identity in zip(ordered, identities):
        entry = expected[name]
        if (actual[name].size != entry['bytes']
                or int(identity[-1]) != entry['bytes']
                or identity_token(identity[0]) != identity_token(entry[key])):
            raise ValueError(
                f'Remote object differs from its pinned manifest: {name}')
        recorded[name] = identity
    return recorded


def run_child(args):
    from kumo_core.sources.types import ReadMode

    manifest = read_manifest(args.manifest[0])
    args.run_dir.mkdir(parents=True, exist_ok=False)
    close = lambda: None
    try:
        reader, config, close = make_reader(args.provider, manifest, args)
        preflight_start = time.perf_counter()
        before_identity = inventory(reader, manifest, args.provider,
                                    args.worker_count)
        preflight_seconds = time.perf_counter() - preflight_start
        provenance = metadata()
        total_bytes = manifest['totals']['bytes']
        allowance = total_bytes + max(1 << 30, args.worker_count * (64 << 20))
        if shutil.disk_usage(args.run_dir).free < allowance:
            raise ValueError(
                'Insufficient local free space for the complete table and headroom'
            )
        transfer = {}
        original = reader._download_objects

        def download(*positional, **keywords):
            started = time.perf_counter()
            cpu = time.process_time()
            result = original(*positional, **keywords)
            transfer['elapsed_seconds'] = time.perf_counter() - started
            transfer['process_cpu_seconds'] = time.process_time() - cpu
            return result

        reader._download_objects = download
        before_memory = memory()
        cpu = time.process_time()
        started = time.perf_counter()
        prepare_started_utc_ns = time.time_ns()
        staged = reader.prepare('', str(args.run_dir / 'parquet'), columns=None,
                                mode=ReadMode.STREAMED, transform=False,
                                worker_ceiling=args.worker_count,
                                max_entries=4 * len(manifest['files']) + 1024,
                                max_bytes=total_bytes)
        prepare_finished_utc_ns = time.time_ns()
        elapsed = time.perf_counter() - started
        cpu = time.process_time() - cpu
        after_memory = memory()
        if staged.source_size != total_bytes:
            raise ValueError('Staged total size differs from manifest')
        after_identity = inventory(reader, manifest, args.provider,
                                   args.worker_count)
        if before_identity != after_identity:
            raise ValueError('Remote identities changed across the trial')
        verification_started = time.perf_counter()
        verified, verification_config = verify_outputs(
            args.run_dir / 'parquet', manifest['files'],
            args.verification_workers)
        result = dict(
            accepted=True, provider=args.provider, table=manifest['table'],
            selected_providers=args.providers, trial_scope='paired'
            if len(args.providers) == 2 else 'single-provider',
            workers=args.worker_count, worker_limit_kind='ceiling',
            effective_file_workers='uninstrumented', stored_bytes=total_bytes,
            rows=manifest['totals']['rows'], files=len(verified),
            prepare_seconds=elapsed, process_cpu_seconds=cpu,
            prepare_started_utc_ns=prepare_started_utc_ns,
            prepare_finished_utc_ns=prepare_finished_utc_ns,
            transfer_phase=transfer, non_transfer_prepare_seconds=elapsed -
            transfer['elapsed_seconds'], mib_per_second=total_bytes / elapsed /
            (1 << 20), rows_per_second=manifest['totals']['rows'] / elapsed,
            preflight_seconds=preflight_seconds,
            verification_seconds=time.perf_counter() - verification_started,
            verification_config=verification_config,
            mode_used=staged.mode_used.value, data_version=staged.version,
            memory_before=before_memory, memory_after=after_memory,
            client_config=config, metadata=provenance,
            request_counts='uninstrumented', retry_counts='uninstrumented',
            token_refresh_counts='uninstrumented',
            remote_identities_before=before_identity,
            remote_identities_after=after_identity, verified_files=verified)
        write_json(args.child_result, result)
    finally:
        close()
        if args.run_dir.exists():
            shutil.rmtree(args.run_dir)


def trial_providers(providers, index):
    """Alternate two-provider order; singleton trials never synthesize a pair."""
    if (not providers or len(set(providers)) != len(providers)
            or any(provider not in ('adls', 's3') for provider in providers)):
        raise ValueError('Select one or both distinct supported providers')
    return tuple(providers) if index % 2 == 0 else tuple(reversed(providers))


def run_parent(args):
    manifests = [(path.resolve(strict=True), read_manifest(path))
                 for path in args.manifest]
    if len({manifest['table'] for _, manifest in manifests}) != len(manifests):
        raise ValueError('Table names must be unique across manifests')
    args.output.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=False)
    placement = json.loads(args.placement.read_text()) if args.placement else {
        'status': 'not supplied'
    }
    write_json(
        args.results / 'arguments.json',
        dict(
            arguments={
                name: [str(item) for item in value] if isinstance(value, list)
                else str(value) if isinstance(value, Path) else value
                for name, value in vars(args).items()
            }, created_utc=datetime.now(timezone.utc).isoformat(),
            benchmark_sha256=sha256(Path(__file__)), placement=placement,
            planned_download_bytes=len(args.providers) * args.trials *
            len(args.workers) * sum(manifest['totals']['bytes']
                                    for _, manifest in manifests)))
    pair = 0
    for index, (_, manifest) in enumerate(manifests):
        pinned_manifest = args.results.resolve() / f'manifest-{index}.json'
        write_json(pinned_manifest, manifest)
        manifest_digest = sha256(pinned_manifest)
        for workers in args.workers:
            for trial in range(args.trials):
                providers = trial_providers(args.providers, pair)
                pair += 1
                for order, provider in enumerate(providers):
                    label = f'table{index}-w{workers}-t{trial}-{order}-{provider}'
                    run_dir = args.output.resolve() / (label + '-' +
                                                       uuid4().hex)
                    child_result = args.results / (label + '.json')
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()), '--providers',
                        *args.providers, '--child', '--manifest',
                        str(pinned_manifest), '--provider', provider,
                        '--worker-count',
                        str(workers), '--run-dir',
                        str(run_dir), '--child-result',
                        str(child_result), '--verification-workers',
                        str(args.verification_workers)
                    ]
                    if args.s3_region:
                        command.extend(['--s3-region', args.s3_region])
                    if args.s3_anonymous:
                        command.append('--s3-anonymous')
                    environment = os.environ.copy()
                    environment.update(KUMO_SOURCE_COPY_WORKERS=str(workers),
                                       KUMO_ADLS_TRANSFER_WORKERS=str(workers))
                    try:
                        completed = subprocess.run(command, check=False,
                                                   capture_output=True,
                                                   text=True, env=environment,
                                                   timeout=args.trial_timeout)
                        if completed.returncode:
                            raise RuntimeError(completed.stderr
                                               or completed.stdout)
                        if sha256(pinned_manifest) != manifest_digest:
                            raise ValueError(
                                'Pinned manifest changed during trial')
                        result = json.loads(child_result.read_text())
                        result.update(trial=trial, provider_order=order,
                                      placement=placement,
                                      manifest_sha256=manifest_digest)
                        write_json(child_result, result)
                        print(
                            json.dumps(
                                dict(
                                    trial=label, accepted=True,
                                    prepare_seconds=result['prepare_seconds'])),
                            flush=True)
                    except Exception as error:
                        write_json(
                            args.results / (label + '.rejected.json'),
                            dict(accepted=False, table=manifest['table'],
                                 provider=provider, reason=str(error)))
                        raise
                    finally:
                        if run_dir.exists():
                            shutil.rmtree(run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--placement', type=Path,
                        help='Measured client/AWS/Azure placement JSON')
    parser.add_argument(
        '--workers', type=int, nargs='+', default=[32],
        help='File/transfer worker ceiling; shared memory budget may lower it')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument(
        '--providers', choices=['adls',
                                's3'], nargs='+', default=['adls', 's3'],
        help='Both alternating providers, or one native-host provider')
    parser.add_argument(
        '--verification-workers', type=int, default=8,
        help='Untimed output hash workers, capped by CPU and eight')
    parser.add_argument('--trial-timeout', type=float, default=14400)
    parser.add_argument('--s3-region')
    parser.add_argument('--s3-anonymous', action='store_true')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--provider', choices=['s3', 'adls'],
                        help=argparse.SUPPRESS)
    parser.add_argument('--worker-count', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--child-result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if args.provider not in args.providers:
            parser.error('Child provider is outside the selected phase')
        run_child(args)
    else:
        if len(set(args.providers)) != len(args.providers):
            parser.error('--providers must not contain duplicates')
        if args.output is None or args.results is None:
            parser.error('--output and --results are required')
        if (any(workers < 1 for workers in args.workers) or args.trials < 1
                or args.verification_workers < 1):
            parser.error(
                'Workers, verification workers and trials must be positive')
        run_parent(args)


if __name__ == '__main__':
    main()
