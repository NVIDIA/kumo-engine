# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen-geometry ADLS/S3 stream diagnostic, not a production bulk benchmark.

The fixed 8 MiB windows, 8 MiB SDK initial response, 4 MiB SDK chunk setting,
and 1 MiB transport blocks are diagnostic controls, not production defaults.
They intentionally do not track provider tuning. Both clients run the same
file worker loop: open (HEAD), sequential 8 MiB
reads, buffered atomic local output, then a fresh open (HEAD). ADLS explicitly
conditions reads on the opened ETag; native Arrow S3 has no public If-Match
option. The server records this difference rather than emulating it away.
Native S3 exposes no HTTP pool-size option; its Arrow I/O pool is configured
before construction. Every trial must prove exact request geometry and output
parity. Source hashing warms the OS cache before all timed trials.

Run --check-protocol first: one real fixture file, one worker, both clients,
with no timing fields published. Full defaults are 24 fresh-process trials.
The caller must supply the source, local output, and results directories.
"""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen
from uuid import uuid4

# Keep this diagnostic contract stable instead of following production defaults.
WINDOW = 8 << 20
NETWORK_BLOCK = 1 << 20
DIAGNOSTIC_GEOMETRY = dict(
    scope='matched-stream diagnostic, not production bulk transfer',
    derived_from_production=False, range_bytes=WINDOW,
    adls_initial_get_bytes=WINDOW, adls_chunk_bytes=4 << 20,
    network_block_bytes=NETWORK_BLOCK)
BUCKET = 'curated'
ACCOUNT = 'benchaccount'
AUTHORITY = f'{BUCKET}@{ACCOUNT}.dfs.core.windows.net'


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(',', ':')) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def etag(stat):
    return f'"{stat.st_size:x}-{stat.st_mtime_ns:x}"'


def make_manifest(source, maximum, expected):
    import pyarrow.parquet as pq

    paths = sorted(source.rglob('*.parquet'))
    if not paths or (maximum is None and len(paths) != expected):
        raise ValueError(
            f'Expected {expected} Parquet files; found {len(paths)}')
    if maximum is not None:
        # A one-file protocol check must exercise multiple range requests.
        paths = ([max(paths, key=lambda path: path.stat().st_size)]
                 if maximum == 1 else paths[:maximum])
    entries = []
    for path in paths:
        if path.is_symlink():
            raise ValueError(f'Source symlinks are unsupported: {path}')
        before = path.stat()
        digest = sha256(path)
        rows = pq.ParquetFile(path).metadata.num_rows
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size,
                                                    after.st_mtime_ns):
            raise ValueError(f'Source changed while preparing manifest: {path}')
        entries.append(
            dict(path=path.relative_to(source).as_posix(), size=after.st_size,
                 etag=etag(after), sha256=digest, rows=rows))
    return dict(source=str(source), files=entries, subset_diagnostic=maximum
                is not None,
                total_bytes=sum(entry['size'] for entry in entries))


class RequestState:
    def __init__(self, manifest):
        self.source = Path(manifest['source'])
        self.files = {entry['path']: entry for entry in manifest['files']}
        self.condition = threading.Condition()
        self.active = {'HEAD': 0, 'GET': 0}
        self.reset(0)

    def idle(self):
        if not self.condition.wait_for(lambda: not any(self.active.values()),
                                       timeout=60):
            raise TimeoutError('Object handlers did not become idle')

    def reset(self, latency):
        with self.condition:
            self.idle()
            self.latency = latency
            self.maximum = {'HEAD': 0, 'GET': 0}
            self.successes = 0
            self.failures = 0
            self.body_bytes = 0
            self.unknown = []
            self.trace = {name: dict(heads=[], gets=[]) for name in self.files}

    def begin(self, method):
        with self.condition:
            self.active[method] += 1
            self.maximum[method] = max(self.maximum[method],
                                       self.active[method])

    def finish(self, method, name, detail, success, body_bytes):
        with self.condition:
            if name in self.trace and detail is not None:
                self.trace[name]['heads' if method ==
                                 'HEAD' else 'gets'].append(detail)
            else:
                self.unknown.append(dict(method=method, path=name))
            self.successes += int(success)
            self.failures += int(not success)
            self.body_bytes += body_bytes
            self.active[method] -= 1
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            self.idle()
            return dict(max_active=dict(self.maximum), successes=self.successes,
                        failures=self.failures,
                        response_body_bytes=self.body_bytes,
                        unknown=list(self.unknown), files=self.trace)


class ObjectHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    disable_nagle_algorithm = True

    def setup(self):
        super().setup()
        self.connection.settimeout(60)

    def log_message(self, *args):
        pass

    def control(self):
        parsed = urlsplit(self.path)
        if not parsed.path.startswith('/__control/'):
            return False
        try:
            if parsed.path == '/__control/reset':
                # Control requests come only from the parent loopback process.
                from urllib.parse import parse_qs
                self.server.state.reset(
                    float(parse_qs(parsed.query)['latency_ms'][0]))
                value = {'reset': True}
            elif parsed.path == '/__control/snapshot':
                value = self.server.state.snapshot()
            else:
                raise ValueError('Unknown control endpoint')
            body = json.dumps(value, separators=(',', ':')).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Content-Type', 'application/json')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True
        except (TimeoutError, ValueError, KeyError):
            self.send_error(500, 'Control request failed')
        return True

    def do_HEAD(self):
        self.object_request('HEAD')

    def do_GET(self):
        if not self.control():
            self.object_request('GET')

    def do_POST(self):
        # Keep unexpected methods visible in the trial rejection evidence.
        with self.server.state.condition:
            self.server.state.unknown.append(dict(method='POST',
                                                  path=self.path))
            self.server.state.failures += 1
        self.send_error(405)

    def object_request(self, method):
        state = self.server.state
        state.begin(method)
        name = self.path
        detail = None
        body_bytes = 0
        success = False
        try:
            path = unquote(urlsplit(self.path).path).lstrip('/')
            if path.startswith(ACCOUNT + '/'):
                path = path[len(ACCOUNT) + 1:]
            prefix = BUCKET + '/'
            if not path.startswith(prefix):
                self.send_error(404)
                return
            name = path[len(prefix):]
            if name not in state.files:
                self.send_error(404)
                return
            source = state.source / name
            stat = source.stat()
            identity = etag(stat)
            conditional = self.headers.get('If-Match')
            detail = dict(
                if_match=conditional, etag=identity,
                auth_scheme=self.headers.get('Authorization', '').split(' ',
                                                                        1)[0])
            if conditional is not None and conditional not in ('*', identity):
                self.send_error(412)
                return
            time.sleep(state.latency / 1000)
            start, end = 0, stat.st_size - 1
            if method == 'GET':
                standard = self.headers.get('Range')
                azure = self.headers.get('x-ms-range')
                if standard and azure and standard != azure:
                    raise ValueError('Conflicting range headers')
                selected = standard or azure or ''
                match = re.fullmatch(r'bytes=(\d+)-(\d+)', selected)
                if match is None:
                    raise ValueError('Expected one explicit byte range')
                start, end = map(int, match.groups())
                detail.update(
                    window=[start, end],
                    range_header='Range' if standard else 'x-ms-range')
                if start > end or end >= stat.st_size:
                    self.send_error(416)
                    return
            length = end - start + 1
            self.send_response(206 if method == 'GET' else 200)
            self.send_header('Content-Length', str(length))
            self.send_header('ETag', identity)
            self.send_header('Last-Modified',
                             self.date_time_string(stat.st_mtime))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('x-ms-blob-type', 'BlockBlob')
            self.send_header('x-ms-request-id', 'loopback')
            self.send_header('x-amz-request-id', 'loopback')
            if method == 'GET':
                self.send_header('Content-Range',
                                 f'bytes {start}-{end}/{stat.st_size}')
            self.end_headers()
            if method == 'GET':
                with source.open('rb') as stream:
                    stream.seek(start)
                    remaining = length
                    while remaining:
                        data = stream.read(min(NETWORK_BLOCK, remaining))
                        if not data:
                            raise OSError(
                                'Source became shorter during response')
                        self.wfile.write(data)
                        body_bytes += len(data)
                        remaining -= len(data)
                self.wfile.flush()
            success = True
        except (OSError, ValueError):
            # A partial/failed response invalidates the entire trial.
            self.close_connection = True
        finally:
            state.finish(method, name, detail, success, body_bytes)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 128


def make_client(provider, endpoint, workers):
    import pyarrow as pa
    import pyarrow.fs as fs

    pa.set_io_thread_count(workers)
    if provider == 's3':
        native = fs.S3FileSystem(
            access_key=ACCOUNT, secret_key='benchmark', region='us-east-1',
            scheme='http', endpoint_override=endpoint,
            force_virtual_addressing=False, allow_delayed_open=False,
            connect_timeout=30, request_timeout=60,
            retry_strategy=fs.AwsStandardS3RetryStrategy(max_attempts=1))

        def identity(stream):
            metadata = stream.metadata()
            metadata = {
                (key.decode() if isinstance(key, bytes) else key).lower():
                value.decode() if isinstance(value, bytes) else value
                for key, value in metadata.items()
            }
            return dict(size=stream.size(), etag=metadata['etag'])

        return SimpleNamespace(
            open=lambda name: native.open_input_file(BUCKET + '/' + name),
            identity=identity,
            read=lambda stream, length, offset: stream.read_at(length, offset),
            close=lambda: None,
            config=dict(authentication='synthetic_sigv4', retry_max_attempts=1,
                        io_threads=workers,
                        http_pool='not configurable through native public API'))

    import requests
    from azure.core.credentials import AzureNamedKeyCredential
    from azure.core.pipeline.transport import RequestsTransport
    from azure.storage.filedatalake import DataLakeServiceClient
    from kumo_adls import transfer
    from kumo_adls.sdk_filesystem import ADLSFileSystem
    from requests.adapters import HTTPAdapter

    if transfer.CONNECTIONS != workers:
        raise ValueError(
            f'ADLS transfer ceiling {transfer.CONNECTIONS} differs from {workers} workers'
        )
    session = requests.Session()
    session.trust_env = False
    adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers,
                          pool_block=True, max_retries=0)
    session.mount('http://', adapter)
    transport = RequestsTransport(session=session, session_owner=True,
                                  connection_data_block_size=NETWORK_BLOCK)
    if transport.connection_config.data_block_size != NETWORK_BLOCK:
        raise ValueError('Azure transport did not apply the network block size')
    client = DataLakeServiceClient(
        f'http://{endpoint}/{ACCOUNT}', credential=AzureNamedKeyCredential(
            ACCOUNT,
            base64.b64encode(b'benchmark').decode()),
        max_single_get_size=WINDOW, max_chunk_get_size=4 << 20,
        connection_data_block_size=NETWORK_BLOCK, retry_total=0,
        connection_timeout=30, read_timeout=60, transport=transport)
    filesystem = ADLSFileSystem(client, ACCOUNT, 'dfs.core.windows.net')

    def read(stream, length, offset):
        stream.seek(offset)
        return stream.read(length)

    return SimpleNamespace(
        open=lambda name: filesystem.open(AUTHORITY + '/' + name, 'rb',
                                          cache_type='none'),
        identity=lambda stream: dict(size=stream.size, etag=stream.etag),
        read=read, close=client.close, config=dict(
            authentication='synthetic_shared_key', retry_total=0,
            http_pool=workers, initial_get_bytes=WINDOW, chunk_bytes=4 << 20,
            network_block_bytes=transport.connection_config.data_block_size,
            transfer_ceiling=transfer.CONNECTIONS))


def memory():
    # Capture before output hashing/footer verification, which can raise HWM.
    result = {}
    for filename, fields in [('status', {'VmHWM', 'VmRSS'}),
                             ('smaps_rollup', {'Pss'})]:
        for line in Path('/proc/self', filename).read_text().splitlines():
            key, separator, value = line.partition(':')
            if separator and key in fields:
                result[key + '_kib'] = int(value.split()[0])
    return result


def versions():
    from kumo_adls import sdk_filesystem, transfer

    result = dict(
        python=sys.version, platform=sys.platform, cpu_count=os.cpu_count(),
        source_sha256={
            module.__name__: sha256(Path(module.__file__))
            for module in (sdk_filesystem, transfer)
        })
    for name in ('pyarrow', 'fsspec', 'azure-core',
                 'azure-storage-file-datalake', 'azure-storage-blob',
                 'requests', 'urllib3'):
        result[name] = importlib.metadata.version(name)
    return result


def run_child(args):
    import pyarrow.parquet as pq

    manifest = json.loads(args.manifest.read_text())
    files = manifest['files']
    client = make_client(args.provider, args.endpoint, args.worker_count)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    before_memory = memory()

    def copy(entry):
        name, size = entry['path'], entry['size']
        destination = args.run_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with client.open(name) as source:
                before = client.identity(source)
                if before != dict(size=size, etag=entry['etag']):
                    raise ValueError(
                        f'Opened source differs from manifest: {name}')
                with tempfile.NamedTemporaryFile(dir=destination.parent,
                                                 buffering=NETWORK_BLOCK,
                                                 delete=False) as target:
                    temporary = Path(target.name)
                    for offset in range(0, size, WINDOW):
                        length = min(WINDOW, size - offset)
                        data = client.read(source, length, offset)
                        if len(data) != length or target.write(data) != length:
                            raise OSError(f'Short read or local write: {name}')
                        del data
            with client.open(name) as source:
                after = client.identity(source)
            if before != after:
                raise ValueError(f'Source identity changed: {name}')
            os.replace(temporary, destination)
            temporary = None
            return dict(path=name, before=before, after=after)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    try:
        cpu = time.process_time()
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.worker_count) as pool:
            identities = list(pool.map(copy, files))
        elapsed = time.perf_counter() - started
        cpu = time.process_time() - cpu
        after_memory = memory()
        verified = []
        for entry in files:
            path = args.run_dir / entry['path']
            actual = dict(path=entry['path'], size=path.stat().st_size,
                          sha256=sha256(path),
                          rows=pq.ParquetFile(path).metadata.num_rows)
            if any(actual[key] != entry[key]
                   for key in ('size', 'sha256', 'rows')):
                raise ValueError(f'Output verification failed: {entry["path"]}')
            verified.append(actual)
        result = dict(provider=args.provider, workers=args.worker_count,
                      protocol_only=args.check_protocol,
                      geometry=DIAGNOSTIC_GEOMETRY, versions=versions(),
                      client_config=client.config, memory_before=before_memory,
                      memory_after=after_memory, identities=identities,
                      verified_files=verified,
                      stored_bytes=manifest['total_bytes'])
        if not args.check_protocol:
            result.update(
                elapsed_seconds=elapsed, process_cpu_seconds=cpu,
                mib_per_second=manifest['total_bytes'] / elapsed / (1 << 20))
        write_json(args.child_result, result)
    finally:
        client.close()
        if args.run_dir.exists():
            shutil.rmtree(args.run_dir)


def rpc(endpoint, operation):
    with urlopen(f'http://{endpoint}/__control/{operation}',
                 timeout=65) as response:
        return json.load(response)


def validate_trace(trace, manifest, workers, provider):
    if trace['unknown'] or trace['failures']:
        raise ValueError('Unexpected or failed requests: ' + json.dumps(trace))
    expected_requests = 0
    if trace['max_active']['GET'] > workers or trace['max_active'][
            'HEAD'] > workers:
        raise ValueError(
            'Observed request concurrency exceeded file-worker limit')
    for entry in manifest['files']:
        observed = trace['files'][entry['path']]
        expected = [[offset, min(offset + WINDOW, entry['size']) - 1]
                    for offset in range(0, entry['size'], WINDOW)]
        actual = sorted(get['window'] for get in observed['gets'])
        if len(observed['heads']) != 2 or actual != expected:
            raise ValueError(
                f'Unmatched request geometry: {entry["path"]}: {observed}')
        if any(item['etag'] != entry['etag']
               for item in observed['heads'] + observed['gets']):
            raise ValueError('Source ETag changed during request trace')
        if provider == 'adls' and any(get['if_match'] != entry['etag']
                                      for get in observed['gets']):
            raise ValueError('An ADLS GET lacked its raw ETag condition')
        expected_auth = 'SharedKey' if provider == 'adls' else 'AWS4-HMAC-SHA256'
        if any(item['auth_scheme'] != expected_auth
               for item in observed['heads'] + observed['gets']):
            raise ValueError('A storage request lacked its provider signature')
        expected_requests += 2 + len(expected)
    if (trace['successes'] != expected_requests
            or trace['response_body_bytes'] != manifest['total_bytes']):
        raise ValueError(
            'Request totals or transferred bytes differ from manifest')


def run_parent(args):
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    results = args.results.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError(
            'Source and temporary output directories must be separate')
    if results.is_relative_to(source):
        raise ValueError('Results must be outside the read-only source fixture')
    output.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=False)
    maximum = 1 if args.check_protocol else args.max_files
    manifest = make_manifest(source, maximum, args.expected_files)
    manifest_path = results / 'manifest.json'
    write_json(manifest_path, manifest)
    write_json(
        results / 'arguments.json',
        dict(arguments=vars_for_json(args), geometry=DIAGNOSTIC_GEOMETRY,
             benchmark_sha256=sha256(Path(__file__))))
    state = RequestState(manifest)
    server = Server(('127.0.0.1', 0), ObjectHandler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f'127.0.0.1:{server.server_port}'
    workers = [1] if args.check_protocol else args.workers
    latencies = [0] if args.check_protocol else args.latencies_ms
    trials = 1 if args.check_protocol else args.trials
    pair = 0
    try:
        for worker_count in workers:
            for latency in latencies:
                for trial in range(trials):
                    providers = ('adls', 's3') if pair % 2 == 0 else ('s3',
                                                                      'adls')
                    pair += 1
                    for order, provider in enumerate(providers):
                        label = f'w{worker_count}-l{latency}-t{trial}-{order}-{provider}'
                        run_dir = output / (label + '-' + uuid4().hex)
                        child_result = results / (label + '.child.json')
                        if run_dir.exists():
                            raise FileExistsError(
                                f'Refusing an existing output: {run_dir}')
                        rpc(endpoint, f'reset?latency_ms={latency}')
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()), '--child',
                            '--provider', provider, '--endpoint', endpoint,
                            '--worker-count',
                            str(worker_count), '--manifest',
                            str(manifest_path), '--run-dir',
                            str(run_dir), '--child-result',
                            str(child_result)
                        ]
                        if args.check_protocol:
                            command.append('--check-protocol')
                        try:
                            environment = os.environ.copy()
                            environment.update(
                                KUMO_ADLS_TRANSFER_WORKERS=str(worker_count),
                                KUMO_SOURCE_COPY_WORKERS=str(worker_count),
                                NO_PROXY='127.0.0.1,localhost',
                                no_proxy='127.0.0.1,localhost')
                            completed = subprocess.run(
                                command, check=False, capture_output=True,
                                text=True, timeout=args.trial_timeout,
                                env=environment)
                            trace = rpc(endpoint, 'snapshot')
                            write_json(results / (label + '.trace.json'), trace)
                            if completed.returncode:
                                raise RuntimeError(completed.stderr
                                                   or completed.stdout)
                            validate_trace(trace, manifest, worker_count,
                                           provider)
                            result = json.loads(child_result.read_text())
                            result.update(
                                latency_ms=latency, trial=trial,
                                provider_order=order,
                                subset_diagnostic=manifest['subset_diagnostic'],
                                trace_file=label + '.trace.json', accepted=True)
                            write_json(results / (label + '.json'), result)
                            child_result.unlink()
                            print(
                                json.dumps(dict(trial=label, accepted=True),
                                           separators=(',', ':')), flush=True)
                        except Exception as error:
                            write_json(results / (label + '.rejected.json'),
                                       dict(accepted=False, reason=str(error)))
                            raise
                        finally:
                            if run_dir.exists():
                                shutil.rmtree(run_dir)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def vars_for_json(args):
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        help='Complete immutable local Parquet table root')
    parser.add_argument('--output', type=Path,
                        help='Temporary local output root')
    parser.add_argument('--results', type=Path,
                        help='New results directory (must not exist)')
    parser.add_argument('--workers', type=int, nargs='+', default=[4, 32])
    parser.add_argument('--latencies-ms', type=float, nargs='+',
                        default=[0, 10])
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--expected-files', type=int, default=580)
    parser.add_argument(
        '--max-files', type=int,
        help='Subset diagnostic only; not a full-fixture result')
    parser.add_argument('--check-protocol', action='store_true',
                        help='One-file pilot; publish no timings')
    parser.add_argument('--trial-timeout', type=float, default=900)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    for option in ('provider', 'endpoint'):
        parser.add_argument('--' + option, help=argparse.SUPPRESS)
    for option in ('manifest', 'run-dir', 'child-result'):
        parser.add_argument('--' + option, type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-count', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        run_child(args)
    else:
        if args.source is None or args.output is None or args.results is None:
            parser.error('--source, --output, and --results are required')
        if (any(workers < 1 for workers in args.workers) or args.trials < 1
                or any(latency < 0 for latency in args.latencies_ms)
                or (args.max_files is not None and args.max_files < 1)):
            parser.error(
                'Worker/trial/file counts must be positive and latencies nonnegative'
            )
        run_parent(args)


if __name__ == '__main__':
    main()
