# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run installed-wheel Snowflake notebook preflight and lifecycle from one local process."""
import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from resources import TARGET

USER = TARGET['user']
ROLE = TARGET['role']
DATABASE = TARGET['database']
SCHEMA = TARGET['schema']
STAGE = TARGET['stage']
WAREHOUSE = TARGET['warehouse']
POOL = TARGET['pool']


def configured_account():
    if not os.environ.get('SNOWFLAKE_TEST_CONFIG'):
        return 'configured-at-execution'
    from resources import require
    return require()['account']


def save(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                         delete=False) as output:
            temporary = Path(output.name)
            output.write(json.dumps(value, indent=2, default=str) + '\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def literal(value):
    return "'" + value.replace("'", "''") + "'"


_DIGESTS = None


def local_digest_cache():
    global _DIGESTS
    folder = Path('/tmp') / f'kumo-snowflake-local-cache-{os.getuid()}'
    folder.mkdir(mode=0o700, exist_ok=True)
    info = folder.lstat()
    if folder.is_symlink() or not folder.is_dir() or info.st_uid != os.getuid(
    ) or info.st_mode & 0o077:
        raise ValueError(
            'Local checksum cache must be private and owner-controlled')
    cache = folder / 'digests.json'
    if _DIGESTS is None:
        _DIGESTS = {}
        if cache.exists() or cache.is_symlink():
            info = cache.lstat()
            if cache.is_symlink() or not cache.is_file(
            ) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError(
                    'Local checksum manifest must be private and owner-controlled'
                )
            try:
                contents = json.loads(cache.read_text())
                if isinstance(contents, dict):
                    _DIGESTS = contents
            except (ValueError, OSError):
                pass
    return cache, _DIGESTS


def file_identity(info):
    return ':'.join(
        str(value) for value in (info.st_dev, info.st_ino, info.st_size,
                                 info.st_mtime_ns, info.st_ctime_ns))


def remember_digest(cache, entries, key, digest):
    import tempfile

    entries[key] = digest
    limited = dict(list(entries.items())[-128:])
    with tempfile.NamedTemporaryFile(mode='w', dir=cache.parent,
                                     delete=False) as output:
        temporary = Path(output.name)
        json.dump(limited, output)
    os.replace(temporary, cache)


def sha(path):
    before = path.stat()
    size = before.st_size
    large = size >= 128 * 1024**2
    key = file_identity(before)
    if large:
        cache, entries = local_digest_cache()
        known = entries.get(key)
        if isinstance(known, str) and re.fullmatch(r'[0-9a-f]{64}', known):
            if file_identity(path.stat()) != key:
                raise ValueError('File changed during checksum lookup: ' +
                                 path.name)
            print(
                f'Reusing local checksum: {path.name} ({size / 1024**3:.2f} GiB)',
                flush=True)
            return known
        print(
            f'Verifying changed or new file: {path.name} ({size / 1024**3:.2f} GiB)...',
            flush=True)
    digest = hashlib.sha256()
    total = 0
    next_update = 512 * 1024**2
    with path.open('rb') as stream:
        if file_identity(os.fstat(stream.fileno())) != key:
            raise ValueError('File changed before checksum read: ' + path.name)
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
            total += len(chunk)
            if large and total >= next_update:
                print(
                    f'  Verified {total / 1024**3:.2f}/{size / 1024**3:.2f} GiB',
                    flush=True)
                next_update += 512 * 1024**2
        if file_identity(os.fstat(stream.fileno())) != key:
            raise ValueError('File changed during checksum read: ' + path.name)
    if file_identity(path.stat()) != key:
        raise ValueError('File changed during checksum read: ' + path.name)
    result = digest.hexdigest()
    if large:
        remember_digest(cache, entries, key, result)
        print(f'Verified {path.name}', flush=True)
    return result


def prepare(root, runtime):
    if not re.fullmatch(r'V[0-9]+(?:\.[0-9]+)*-CPU-PY3\.[0-9]+', runtime):
        raise ValueError(
            'Select an explicit x86 CPU Container Runtime version, e.g. V2.9-CPU-PY3.12'
        )
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    attempt = uuid4().hex[:12]
    payload = root / 'payload'
    payload.mkdir()
    prefix = f'{STAGE}/notebook-output-probe/{attempt}'
    bundle = f'{DATABASE}.{SCHEMA}.MLOPS_PROBE_{attempt.upper()}'
    spec = (
        'bundle:\n  type: custom\n  compute_type: compute_pool\n  language: python\n'
        '  compute_options:\n    compute_pool: ' + POOL +
        '\n    query_warehouse: ' + WAREHOUSE + '\n    runtime_version: ' +
        json.dumps(runtime) + '\n')
    (payload / 'code_bundle.yml').write_text(spec)
    manifest = dict(
        attempt=attempt, account=configured_account(), user=USER, role=ROLE,
        bundle=bundle, stage_prefix=prefix, runtime=runtime,
        payload_sha256={path.name: sha(path)
                        for path in payload.iterdir()},
        create_sql=(
            f'CREATE CODE BUNDLE {bundle} FROM {literal(prefix + "/payload")}'),
        mutations=[
            'Upload only this attempt payload',
            'Create a new uniquely named test Code Bundle',
            'Run the notebook on KUMO_TEST_POOL; retain resources and receipts'
        ])
    save(root / 'plan.json', manifest)
    return manifest


def prepare_payload(root, runtime, source, approved=None):
    import shutil

    source = Path(source).resolve()
    allowed = {
        'lifecycle.ipynb', 'lifecycle-config.json', 'snowflake.json',
        'wheel_mlops_lifecycle.py', 'wheel_mlops_platform.py',
        'wheel_gpu_smoke.py', 'wheelhouse.tar', 'wheelhouse-manifest.json',
        'offline_support.py'
    }
    paths = list(source.iterdir())
    if approved is None:
        raise ValueError('An explicit payload manifest is required')
    remote_assets = approved['remote_assets']
    if set(remote_assets) != {
            'wheelhouse.tar', approved['candidate']['wheel_file']
    }:
        raise ValueError(
            'Remote assets must identify the candidate and offline dependencies'
        )
    for name, asset in remote_assets.items():
        validate_remote_asset(name, asset)
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError('Payloads must contain flat regular files')
        if path.name not in allowed and not (
                path.name.startswith('kumo_engine-') and path.suffix == '.whl'):
            raise ValueError('Unexpected lifecycle payload file: ' + path.name)
    config = json.loads((source / 'lifecycle-config.json').read_text())
    if set(config) != {
            'wheel_file', 'wheel_sha256', 'expected_version', 'mode'
    }:
        raise ValueError('Unknown lifecycle configuration fields')
    if config['mode'] not in ('preflight', 'lifecycle', 'negative'):
        raise ValueError('Select preflight, lifecycle, or negative')
    if {
            key: value
            for key, value in config.items() if key != 'mode'
    } != approved['candidate']:
        raise ValueError('Candidate differs from the reviewed manifest')
    expected_files = approved['modes'][config['mode']]
    if {path.name
            for path in paths} != set(expected_files) - set(remote_assets):
        raise ValueError('Payload file set differs from the reviewed manifest')
    if any(sha(path) != expected_files[path.name] for path in paths):
        raise ValueError('Payload content differs from the reviewed manifest')
    if config['wheel_file'] not in remote_assets or remote_assets[
            config['wheel_file']]['sha256'] != config['wheel_sha256']:
        raise ValueError('Candidate wheel differs from the staged asset')
    if not allowed.issubset({path.name for path in paths} | set(remote_assets)):
        raise ValueError('Incomplete lifecycle payload')
    profile = json.loads((source / 'snowflake.json').read_text())
    if any(
            profile.get(key) != value
            for key, value in approved['profile_target'].items()):
        raise ValueError(
            'Snowflake profile differs from the reviewed test targets')
    manifest = prepare(root, runtime)
    print(f'Preparing {config["mode"]} payload: {root}', flush=True)
    for path in paths:
        if path.stat().st_size >= 128 * 1024**2:
            print(f'  Referencing unchanged local file: {path.name}',
                  flush=True)
            (root / 'payload' / path.name).symlink_to(path)
        else:
            print(
                f'  Copying {path.name} ({path.stat().st_size / 1024**2:.1f} MiB)',
                flush=True)
            shutil.copyfile(path, root / 'payload' / path.name)
    bundle = manifest['bundle']
    stage = manifest['stage_prefix']
    create_sql = f'CREATE CODE BUNDLE {bundle} FROM {literal(stage + "/notebook")}'
    execute_sql = f"EXECUTE CODE BUNDLE {bundle} ENTRYPOINT = 'lifecycle.ipynb' ARGUMENTS = ({literal(stage)})"
    manifest.update(
        kind='lifecycle', mode=config['mode'], source_directory=str(source),
        create_sql=create_sql, execute_sql=execute_sql,
        wheel_sha256=config['wheel_sha256'], payload_sha256={
            path.name: sha(path)
            for path in (root / 'payload').iterdir()
        })
    manifest['remote_assets'] = remote_assets
    manifest['payload_sha256'].update({
        name: asset['sha256']
        for name, asset in remote_assets.items()
    })
    save(root / 'plan.json', manifest)
    return manifest


def connection_parameters(key_file):
    from resources import require

    target = require()
    logging.getLogger('snowflake.connector').setLevel(logging.ERROR)
    pem = os.environ.get('SNOWFLAKE_PRIVATE_KEY')
    if pem:
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
            load_pem_private_key,
        )
        passphrase = os.environ.get('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE')
        key = load_pem_private_key(
            pem.encode(), password=passphrase.encode() if passphrase else None)
        der = key.private_bytes(Encoding.DER, PrivateFormat.PKCS8,
                                NoEncryption())
    else:
        if os.environ.get('GITHUB_ACTIONS') == 'true':
            raise ValueError('GitHub runs require the environment RSA secrets')
        if key_file is None:
            raise ValueError(
                'Set SNOWFLAKE_PRIVATE_KEY or supply --key-file for the local memory cache'
            )
        from key_cache import load_der
        der = load_der(key_file)
    parameters = dict(
        account=target['account'], user=target['user'], role=target['role'],
        database=target['database'], schema=target['schema'],
        warehouse=target['warehouse'], authenticator='SNOWFLAKE_JWT',
        private_key=der, login_timeout=30, network_timeout=300,
        socket_timeout=120, client_session_keep_alive=False,
        session_parameters={'QUERY_TAG': 'kumo_notebook_output_probe'})
    return parameters


def connect(key_file):
    import snowflake.connector
    from resources import require

    target = require()
    connection = snowflake.connector.connect(**connection_parameters(key_file))
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT CURRENT_ACCOUNT()')
            row = cursor.fetchone()
        if row != (target['account_locator'], ):
            raise ValueError(
                'Connected account does not match the authorized test account')
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        raise ValueError(
            'Cannot verify the authorized Snowflake test account') from None
    return connection


def cache_location_allowed(location, name):
    if not isinstance(location, str) or not isinstance(
            name, str) or not re.fullmatch(r'[A-Za-z0-9_.+-]+', name):
        return False
    base = re.escape(STAGE)
    old = base + r'/notebook-output-probe/[0-9a-f]{12}/payload/' + re.escape(
        name)
    cached = base + r'/mlops-validation/payload-cache/[0-9a-f]{64}/' + re.escape(
        name) + r'/objects/[0-9a-f]{12}/' + re.escape(name)
    return bool(re.fullmatch(old, location) or re.fullmatch(cached, location))


def validate_remote_asset(name, asset):
    if (not isinstance(asset, dict) or asset.get('name') != name
            or not re.fullmatch(r'[0-9a-f]{64}', str(asset.get('sha256', '')))
            or type(asset.get('size')) is not int or asset['size'] <= 0
            or not cache_location_allowed(asset.get('location'), name)
            or not isinstance(asset.get('remote'), dict)):
        raise ValueError('Invalid staged asset: ' + name)


def existing_remote_asset(cursor, name, asset):
    validate_remote_asset(name, asset)
    observed = list_object(cursor, asset['location'])
    if observed != asset['remote'] or observed['size'] != asset['size']:
        raise ValueError('Staged asset changed or disappeared: ' + name)
    print('Reusing pinned staged asset:', name, flush=True)
    return dict(asset, action='reused')


def list_object(cursor, location):
    cursor.execute(f'LIST {location}')
    rows = cursor.fetchall()
    if not rows:
        return None
    columns = [
        str(column.name if hasattr(column, 'name') else column[0]).lower()
        for column in (cursor.description or ())
    ]
    required = {'name', 'size', 'last_modified'}
    if not required.issubset(columns):
        raise ValueError('Unexpected LIST columns: ' + ', '.join(columns))
    if any(len(row) != len(columns) for row in rows):
        raise ValueError('LIST rows differ from the declared result columns')
    expected_stage, expected_path = location.removeprefix('@').split('/', 1)
    allowed_stages = {
        expected_stage.upper(),
        expected_stage.rsplit('.', 1)[-1].upper()
    }

    def exact(row):
        remote = str(row['name']).removeprefix('@')
        if '/' not in remote:
            return False
        remote_stage, remote_path = remote.split('/', 1)
        return remote_stage.upper(
        ) in allowed_stages and remote_path == expected_path

    matches = [
        row for values in rows if exact(row := dict(zip(columns, values)))
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError('Ambiguous staged object: ' + location)
    row = matches[0]
    return dict(name=str(row['name']), size=int(row['size']),
                md5=row.get('md5'), last_modified=str(row['last_modified']))


def download_file(cursor, location, directory):
    cursor.execute(
        f'GET {location} {literal("file://" + str(directory.resolve()) + "/")}')
    cursor.fetchall()
    files = list(directory.iterdir())
    if len(files) != 1 or not files[0].is_file() or files[0].is_symlink():
        raise ValueError('Expected one staged file: ' + location)
    if files[0].name != location.rsplit('/', 1)[-1]:
        raise ValueError('Unexpected downloaded filename')
    return files[0]


def ensure_cached_file(cursor, path, expected_hash, work, candidates=None):
    import tempfile

    if sha(path) != expected_hash:
        raise ValueError('Payload changed after planning: ' + path.name)
    size = path.stat().st_size
    bucket = f'{STAGE}/mlops-validation/payload-cache/{expected_hash}/{path.name}'
    receipt_location = bucket + '/cache-receipt.json'
    receipt = None
    if list_object(cursor, receipt_location):
        with tempfile.TemporaryDirectory(prefix='cache-receipt-',
                                         dir=work) as temp:
            downloaded = download_file(cursor, receipt_location, Path(temp))
            if downloaded.stat().st_size > 65536:
                raise ValueError('Oversized cache receipt')
            try:
                receipt = json.loads(downloaded.read_text())
            except ValueError:
                receipt = None
    if isinstance(receipt, dict) and all(
            receipt.get(key) == value for key, value in dict(
                account=configured_account(), stage=STAGE, name=path.name,
                sha256=expected_hash, size=size).items()):
        location = receipt.get('location', '')
        if cache_location_allowed(location, path.name):
            observed = list_object(cursor, location)
            if observed and observed['size'] == size and observed.get(
                    'md5') and observed == receipt.get('remote'):
                print('Reusing staged file:', path.name, flush=True)
                return dict(name=path.name, sha256=expected_hash, size=size,
                            location=location, action='reused', remote=observed)
    locations = list(candidates or ())
    if isinstance(receipt, dict) and cache_location_allowed(
            receipt.get('location', ''), path.name):
        locations.insert(0, receipt['location'])
    location = None
    for candidate in dict.fromkeys(locations):
        if not cache_location_allowed(candidate, path.name):
            continue
        before = list_object(cursor, candidate)
        if before is None or before['size'] != size:
            continue
        print('Verifying existing staged file:', path.name, flush=True)
        with tempfile.TemporaryDirectory(prefix='cache-adopt-',
                                         dir=work) as temp:
            downloaded = download_file(cursor, candidate, Path(temp))
            if sha(downloaded) != expected_hash:
                continue
        observed = list_object(cursor, candidate)
        if before != observed:
            continue
        location, action = candidate, 'adopted'
        break
    if location is None:
        folder = bucket + '/objects/' + uuid4().hex[:12]
        location = folder + '/' + path.name
        print(
            f'Uploading uncached file: {path.name} ({size / 1024**2:.1f} MiB)',
            flush=True)
        cursor.execute(
            f'PUT {literal("file://" + str(path.resolve()))} {folder} AUTO_COMPRESS=FALSE OVERWRITE=FALSE'
        )
        rows = cursor.fetchall()
        if not rows or any(str(row[6]).upper() != 'UPLOADED' for row in rows):
            raise ValueError('Cache upload incomplete: ' + path.name)
        observed = list_object(cursor, location)
        if observed is None or observed['size'] != size:
            raise ValueError('Cache upload metadata mismatch')
        action = 'uploaded'
    receipt = dict(account=configured_account(), stage=STAGE, name=path.name,
                   sha256=expected_hash, size=size, location=location,
                   remote=observed)
    with tempfile.TemporaryDirectory(prefix='cache-publish-', dir=work) as temp:
        local = Path(temp) / 'cache-receipt.json'
        save(local, receipt)
        cursor.execute(
            f'PUT {literal("file://" + str(local))} {bucket} AUTO_COMPRESS=FALSE OVERWRITE=TRUE'
        )
        rows = cursor.fetchall()
        if not rows or any(str(row[6]).upper() != 'UPLOADED' for row in rows):
            raise ValueError('Cache receipt upload incomplete')
    print(('Adopted' if action == 'adopted' else 'Cached') + ' staged file:',
          path.name, flush=True)
    return dict(name=path.name, sha256=expected_hash, size=size,
                location=location, action=action, remote=observed)


def lifecycle_verdict(by_path, manifest):
    phases = ('fit', 'score', 'evaluate', 'negative')
    needed = {'lifecycle/summary.json'
              } | {f'lifecycle/{name}.json'
                   for name in phases}
    if not needed.issubset(by_path):
        return None
    summary = json.loads(by_path['lifecycle/summary.json'].read_text())
    states = {}
    for name in phases:
        phase = json.loads(by_path[f'lifecycle/{name}.json'].read_text())
        state = phase['status']
        recorded = summary['phases'][name]
        if not (state == recorded['status']):
            raise AssertionError('Conflicting lifecycle phase results')
        if state == 'passed':
            if not (recorded['returncode'] == 0):
                raise AssertionError(
                    'Lifecycle worker did not exit successfully')
            if not (phase['environment']['sha256'] == manifest['wheel_sha256']):
                raise AssertionError('Lifecycle phase used another wheel')
        states[name] = state
    state = 'passed' if all(s == 'passed' for s in states.values()) else (
        'failed' if 'failed' in states.values() else 'blocked')
    return dict(
        status=state, phases=states, remaining_requirements={
            name: gap
            for name, gap in summary.get('required_gaps', {}).items()
            if gap.get('status') != 'passed'
        })


def validate_lifecycle_exit(receipt, by_path, manifest):
    result = receipt['lifecycle']
    if manifest['mode'] == 'negative':
        if not (result['returncode'] == 0):
            raise AssertionError('Permission worker did not exit successfully')
        return
    summary = json.loads(by_path['lifecycle/summary.json'].read_text())
    if not (result['summary'] == summary):
        raise AssertionError('Worker summary differs from exported summary')
    checks = lifecycle_verdict(by_path, manifest)
    if not (checks is not None and checks['status'] == 'passed'):
        raise AssertionError('Engine phases did not pass')
    if result['returncode'] != 0:
        if not (result['returncode'] == 1
                and summary['status'] == receipt['status'] == 'blocked'):
            raise AssertionError('Unexpected worker exit')
        gaps = checks['remaining_requirements']
        if not (set(gaps) == {'exported_epoch_history', 'ui_review'}):
            raise AssertionError('Unexpected qualification gaps')
        if not (gaps['exported_epoch_history']['status'] == 'not implemented'):
            raise AssertionError('Unexpected export result')
        if not (gaps['ui_review']['status'] == 'blocked'):
            raise AssertionError('Unexpected UI result')
        if not (not summary.get('error')):
            raise AssertionError('Worker orchestration error remains')


def print_result(report, summary_path):
    checks = report.get('engine_checks')
    permission_only = 'permission-denial check only' in report.get('scope', '')
    if checks:
        print('Engine tests:', checks['status'].upper(), flush=True)
        plan_status = 'INCOMPLETE' if report['status'] == 'blocked' else report[
            'status'].upper()
        print('Full testing plan:', plan_status, flush=True)
        labels = {
            'exported_epoch_history': 'Epoch-history export: not implemented.',
            'ui_review': 'Snowsight UI verification: not run.'
        }
        if checks.get('remaining_requirements'):
            print('Still required:', flush=True)
            for name, gap in checks['remaining_requirements'].items():
                print('  -', labels.get(name,
                                        'Qualification check incomplete.'),
                      flush=True)
    elif permission_only:
        print('Permission-denial check:', report['status'].upper(),
              '(no training run)', flush=True)
    else:
        print('Snowflake check:', report['status'].upper(), flush=True)
    for stage in report.get('stages', {}).values():
        if stage.get('reason'):
            print('Stage did not complete; inspect private evidence.',
                  flush=True)
        error = stage.get('collection_error') or stage.get('error')
        if error:
            message = error.get('message', str(error))
            intentional_block = 'Lifecycle prerequisite/test is blocked' in message
            if not intentional_block:
                print('Runner error; inspect private evidence.', flush=True)
    if report.get('error'):
        print('Runner error; inspect private evidence.', flush=True)
    if report.get('close_error'):
        print('Runner cleanup error; inspect private evidence.', flush=True)
    print('Summary:', summary_path, flush=True)


def show_saved_result(root):
    root = Path(root).resolve(strict=True)
    report = json.loads((root / 'summary.json').read_text())
    for name, stage in report.get('stages', {}).items():
        saved = root / name / 'report.json'
        if not saved.is_file():
            continue
        result = json.loads(saved.read_text())
        for key in ('reason', 'error', 'collection_error'):
            if key in result:
                stage[key] = result[key]
        if name != 'lifecycle' or result.get('collection_error'):
            continue
        receipt = result.get('notebook_receipt', {})
        paths = {}
        for filename, relative in receipt.get('logical_files', {}).items():
            if relative in {
                    'lifecycle/summary.json', 'lifecycle/fit.json',
                    'lifecycle/score.json', 'lifecycle/evaluate.json',
                    'lifecycle/negative.json'
            }:
                if not (re.fullmatch('[A-Za-z0-9_.-]+', filename)):
                    raise AssertionError('Unsafe evidence filename')
                path = root / name / 'download' / filename
                if not (sha(path) == receipt['artifacts'][filename]):
                    raise AssertionError('Saved evidence checksum mismatch')
                paths[relative] = path
        checks = lifecycle_verdict(paths, result['plan'])
        if checks is not None:
            report['engine_checks'] = checks
    print_result(report, root / 'summary.json')
    return report


def cleanup_submission(connection, manifest, *, query_unknown, bundle_created,
                       bundle_pending):
    attempt = manifest['attempt']
    if (not re.fullmatch('[a-f0-9]{12}', attempt) or manifest['stage_prefix']
            != STAGE + '/notebook-output-probe/' + attempt or manifest['bundle']
            != f'{DATABASE}.{SCHEMA}.MLOPS_PROBE_' + attempt.upper()):
        raise ValueError('Cleanup requires exact generated attempt resources')
    failures = []
    if query_unknown:
        failures.append('query_state')
    else:
        if bundle_pending:
            failures.append('bundle_create_unknown')
        if bundle_created:
            try:
                with connection.cursor() as cursor:
                    cursor.execute('DROP CODE BUNDLE ' + manifest['bundle'])
            except Exception:
                failures.append('bundle')
        try:
            with connection.cursor() as cursor:
                cursor.execute('REMOVE ' + manifest['stage_prefix'] + '/')
        except Exception:
            failures.append('stage')
    return dict(status='failed' if failures else 'passed',
                failed_resources=failures)


def execute(root, manifest, connection, timeout):
    if manifest.get('kind') != 'lifecycle':
        raise ValueError('Only prepared lifecycle notebook payloads are supported')
    report = dict(
        status='running',
        scope=('Notebook execution and exported outputs; UI excluded'),
        plan=manifest, transfers={}, queries=[])
    submitted = None
    cleanup_owned = False
    bundle_created = False
    bundle_pending = False
    submission_pending = False
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT CURRENT_ACCOUNT(),CURRENT_USER(),CURRENT_ROLE(),CURRENT_DATABASE(),CURRENT_SCHEMA()'
            )
            report['caller'] = dict(
                zip(['account', 'user', 'role', 'database', 'schema'],
                    cursor.fetchone()))
            if not (all(
                (report['caller'][key] == value
                 for (key,
                      value) in dict(user=USER, role=ROLE, database=DATABASE,
                                     schema=SCHEMA).items()))):
                raise AssertionError('Unexpected submitting identity/context')
            report['execution_timeout_seconds'] = timeout
            report['statement_timeout_seconds'] = timeout + 60
            cursor.execute(
                f'ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout + 60}'
            )
            profile_path = root / 'payload' / 'snowflake.json'
            if not (sha(profile_path)
                    == manifest['payload_sha256']['snowflake.json']):
                raise AssertionError('Profile changed after planning')
            denied = json.loads(
                profile_path.read_text()).get('denied_artifact_root')
            if not denied:
                report['status'] = 'blocked'
                report[
                    'reason'] = 'Configure denied_artifact_root before dispatch; no notebook or fits started'
                return report
            if not re.fullmatch(
                    r'@[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*/[A-Za-z0-9_/-]+',
                    denied):
                raise ValueError('Invalid denied-stage fixture location')
            report['operation'] = 'check_denied_stage_fixture'
            print('Checking denied-stage fixture exists and is readable...',
                  flush=True)
            try:
                cursor.execute(f'LIST {denied.split("/", 1)[0]}')
                cursor.fetchall()
            except Exception as error:
                report['status'] = 'blocked'
                report[
                    'reason'] = 'Denied-stage fixture is not ready; run setup-denied-stage.sql as ACCOUNTADMIN before dispatch'
                report['fixture_error'] = dict(type=type(error).__name__,
                                               message=str(error))
                return report
            cursor.execute('LIST ' + manifest['stage_prefix'] + '/')
            if cursor.fetchall():
                raise ValueError(
                    'Attempt stage prefix already exists; refusing cleanup ownership'
                )
            expected_prefix = STAGE + '/notebook-output-probe/' + manifest[
                'attempt']
            if (manifest['stage_prefix'] != expected_prefix
                    or not re.fullmatch('[a-f0-9]{12}', manifest['attempt']) or
                    manifest['bundle'] != f'{DATABASE}.{SCHEMA}.MLOPS_PROBE_' +
                    manifest['attempt'].upper()):
                raise ValueError(
                    'Cleanup requires exact generated attempt resources')
            cleanup_owned = True
            report['cleanup_owned'] = True
            report['operation'] = 'prepare_payload'
            save(root / 'report.json', report)
            cached_files = {}
            for path in sorted((root / 'payload').iterdir()):
                if path.name not in {
                        'code_bundle.yml', 'lifecycle.ipynb'
                }:
                    report['operation'] = 'cache_lookup'
                    report['current_file'] = path.name
                    save(root / 'report.json', report)
                    cached_files[path.name] = ensure_cached_file(
                        cursor, path, manifest['payload_sha256'][path.name],
                        root)
                    report['cached_files'] = cached_files
                    save(root / 'report.json', report)
                    continue
                print(
                    f'Uploading {path.name} ({path.stat().st_size / 1024**2:.1f} MiB)...',
                    flush=True)
                if not (sha(path) == manifest['payload_sha256'][path.name]):
                    raise AssertionError('Payload changed after planning')
                target = 'notebook'
                cursor.execute(
                    f'PUT {literal("file://"+str(path.resolve()))} {manifest["stage_prefix"]}/{target} AUTO_COMPRESS=FALSE OVERWRITE=FALSE'
                )
                report['transfers'][path.name] = cursor.fetchall()
                save(root / 'report.json', report)
                if not (report['transfers'][path.name] and all(
                    (str(row[6]).upper() == 'UPLOADED'
                     for row in report['transfers'][path.name]))):
                    raise AssertionError('Payload upload incomplete')
            for name, asset in manifest.get('remote_assets', {}).items():
                cached_files[name] = existing_remote_asset(cursor, name, asset)
                report['cached_files'] = cached_files
                save(root / 'report.json', report)
            if cached_files:
                locations_file = root / 'payload-locations.json'
                save(locations_file, cached_files)
                cursor.execute(
                    f'PUT {literal("file://" + str(locations_file.resolve()))} {manifest["stage_prefix"]}/payload AUTO_COMPRESS=FALSE OVERWRITE=FALSE'
                )
                rows = cursor.fetchall()
                if not (rows and all(
                    (str(row[6]).upper() == 'UPLOADED' for row in rows))):
                    raise AssertionError('Cache references upload incomplete')
                report['transfers']['payload-locations.json'] = rows
                manifest['cache_references_sha256'] = sha(locations_file)
                manifest[
                    'execute_sql'] = f'EXECUTE CODE BUNDLE {manifest["bundle"]} ENTRYPOINT = \'lifecycle.ipynb\' ARGUMENTS = ({literal(manifest["stage_prefix"])}, {literal(manifest["cache_references_sha256"])})'
                save(root / 'plan.json', manifest)
                save(root / 'report.json', report)
            import retention
            if manifest.get('mode') == 'lifecycle':
                retained_profile = json.loads(
                    (root / 'payload' / 'snowflake.json').read_text())
                if any(
                        retained_profile.get(name) != 'KUMO_TEST_MLOPS'
                        for name in ('experiment', 'model_prefix')):
                    raise ValueError(
                        'Lifecycle retention requires dedicated test prefixes')
            retained = retention.write(cursor, root, manifest)
            report['retention'] = dict(status='recorded',
                                       query_tag=retained['query_tag'])
            report['operation'] = 'prepare_bundle'
            save(root / 'report.json', report)
            report['operation'] = 'create_bundle'
            save(root / 'report.json', report)
            print(
                'Creating notebook Code Bundle (wheel archive stays on stage)...',
                flush=True)
            bundle_pending = True
            cursor.execute(manifest['create_sql'])
            bundle_created = True
            bundle_pending = False
            report['queries'].append(dict(kind='create', query_id=cursor.sfqid))
            report['operation'] = 'bundle_created'
            save(root / 'report.json', report)
            report['operation'] = 'execute_notebook'
            save(root / 'report.json', report)
            print('Submitting notebook execution...', flush=True)
            cursor.execute('ALTER SESSION SET QUERY_TAG = ' +
                           literal(retained['query_tag']))
            submission_pending = True
            cursor.execute_async(manifest['execute_sql'])
            submitted = cursor.sfqid
            submission_pending = False
            report['execution_query_id'] = submitted
            save(root / 'report.json', report)
            deadline = time.monotonic() + timeout
            while True:
                status = connection.get_query_status_throw_if_error(submitted)
                if report.get('execution_status') != status.name:
                    print('Notebook execution:', status.name, flush=True)
                report['execution_status'] = status.name
                save(root / 'report.json', report)
                if not connection.is_still_running(status):
                    report['terminal_status'] = status.name
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('Notebook execution deadline exceeded')
                time.sleep(2)
            cursor.get_results_from_sfqid(submitted)
            report['execution_result'] = cursor.fetchall()
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = dict(type=type(error).__name__, message=str(error))
        if submitted:
            try:
                with connection.cursor() as cursor:
                    cursor.execute('SELECT SYSTEM$CANCEL_QUERY(%s)',
                                   (submitted, ))
                    report['cancel_result'] = cursor.fetchall()
            except Exception as cancel_error:
                report['cancel_error'] = str(cancel_error)
            try:
                cancel_deadline = time.monotonic() + 30
                while True:
                    state = connection.get_query_status(submitted)
                    report['post_cancel_status'] = state.name
                    if not connection.is_still_running(state):
                        report['terminal_status'] = state.name
                        report['remote_query_still_running'] = False
                        break
                    if time.monotonic() >= cancel_deadline:
                        report['remote_query_still_running'] = True
                        break
                    time.sleep(1)
            except Exception as status_error:
                report['post_cancel_status_error'] = str(status_error)
                report['remote_query_still_running'] = 'unknown'
    finally:
        download = root / 'download'
        download.mkdir(exist_ok=True)
        if submitted:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f'GET {manifest["stage_prefix"]}/receipts/ {literal("file://"+str(download.resolve())+"/")}'
                    )
                    report['download_result'] = cursor.fetchall()
                receipt_name = 'lifecycle-result.json'
                receipt = json.loads((download / receipt_name).read_text())
                report['notebook_receipt'] = receipt
                if not (receipt['attempt'] == manifest['attempt']):
                    raise AssertionError('Wrong notebook attempt')
                if not (all((receipt['identity'][name] == report['caller'][name]
                             for name in ('account', 'user', 'role', 'database',
                                          'schema')))):
                    raise AssertionError(
                        'Notebook identity differs from submitter')
                if receipt.get('status') == 'passed' or 'mode' in receipt:
                    if not (receipt.get('mode') == manifest['mode']):
                        raise AssertionError('Unexpected notebook mode')
                if receipt.get(
                        'status') == 'passed' or 'wheel_sha256' in receipt:
                    if not (receipt.get('wheel_sha256')
                            == manifest['wheel_sha256']):
                        raise AssertionError('Unexpected candidate wheel')
                if not (receipt['status']
                        in ('passed', 'failed', 'blocked')):
                    raise AssertionError('Invalid notebook result status')
                artifacts = receipt.get('artifacts', {})
                logical = receipt.get('logical_files', {})
                if not (isinstance(artifacts, dict)
                        and isinstance(logical, dict)):
                    raise AssertionError('Invalid evidence manifest')
                if not (set(artifacts) == set(logical)):
                    raise AssertionError(
                        'Evidence names do not match logical paths')
                if not (len(set(logical.values())) == len(logical)):
                    raise AssertionError('Duplicate logical evidence paths')
                for name, expected_hash in artifacts.items():
                    if not (re.fullmatch('[A-Za-z0-9_.-]+', name)):
                        raise AssertionError('Unsafe output filename')
                    if not (sha(download / name) == expected_hash):
                        raise AssertionError('Output checksum mismatch')
                by_path = {
                    relative: download / name
                    for name, relative in logical.items()
                }
                if receipt['status'] == 'passed' or (
                        manifest['mode'] == 'lifecycle'
                        and 'lifecycle/summary.json' in by_path):
                    required = {
                        'venv.log', 'install.log', 'pip-check.log',
                        'child-auth.stdout.json', 'child-auth.stderr.log',
                        'dependency-manifest.json', 'cached-files.json'
                    }
                    if manifest['mode'] == 'preflight':
                        required |= {
                            'candidate-admission.json',
                            'candidate-admission.stderr.log'
                        }
                    elif manifest['mode'] == 'negative':
                        required |= {
                            'lifecycle/platform.json',
                            'lifecycle/negative.json'
                        }
                    else:
                        required |= {
                            'lifecycle/summary.json',
                            'lifecycle/platform.json'
                        }
                        required |= {
                            f'lifecycle/{phase}.json'
                            for phase in ('fit', 'score', 'evaluate',
                                          'negative')
                        }
                    if not (required.issubset(by_path)):
                        raise AssertionError('Required evidence is missing')
                    if not (receipt['pip_check_returncode'] ==
                            receipt['child_auth_returncode'] == 0):
                        raise AssertionError(
                            'Prerequisite checks did not pass')
                    if not (receipt['payload_references_sha256']
                            == manifest['cache_references_sha256']):
                        raise AssertionError(
                            'Notebook cache references hash differs from submission'
                        )
                    observed_cache = json.loads(
                        by_path['cached-files.json'].read_text())
                    expected_cache = {
                        name: digest
                        for name, digest in
                        manifest['payload_sha256'].items() if name not in
                        {'code_bundle.yml', 'lifecycle.ipynb'}
                    }
                    if not ({
                            name: entry['sha256']
                            for (name, entry) in observed_cache.items()
                    } == expected_cache):
                        raise AssertionError(
                            'Cached payload differs from planned files')
                    if not (observed_cache == report['cached_files']):
                        raise AssertionError(
                            'Downloaded cache references differ from submitted references'
                        )
                    if not (json.loads(
                            by_path['dependency-manifest.json'].read_text())
                            == json.loads(
                                (root / 'payload' /
                                 'wheelhouse-manifest.json').read_text())):
                        raise AssertionError(
                            'Dependency wheel manifest differs from payload'
                        )
                    child_identity = json.loads(
                        by_path['child-auth.stdout.json'].read_text())
                    if not (all(
                        (child_identity[key] == report['caller'][key]
                         for key in ('account', 'user', 'role')))):
                        raise AssertionError(
                            'Fresh child identity mismatch')
                    if manifest['mode'] == 'preflight':
                        if not (receipt['candidate_admission_returncode']
                                == 0):
                            raise AssertionError(
                                'Candidate admission did not pass')
                        admitted = json.loads(
                            by_path['candidate-admission.json'].read_text())
                        if not (admitted['sha256']
                                == manifest['wheel_sha256']):
                            raise AssertionError('Admission wheel differs')
                    else:
                        validate_lifecycle_exit(receipt, by_path, manifest)
                        phases = ('negative',
                                  ) if manifest['mode'] == 'negative' else (
                                      'fit', 'score', 'evaluate',
                                      'negative')
                        for phase in phases:
                            if not (json.loads(
                                    by_path[f'lifecycle/{phase}.json'].
                                    read_text())['status'] == 'passed'):
                                raise AssertionError(
                                    'A lifecycle phase did not pass')
                        if manifest['mode'] == 'negative':
                            proof = json.loads(
                                by_path['lifecycle/negative.json'].
                                read_text())
                            if proof.get('result', {}).get(
                                    'denied_artifact_access') != 'passed':
                                raise AssertionError(
                                    'Missing actual permission-denial evidence'
                                )
                            if not (proof.get('environment',
                                              {}).get('sha256')
                                    == manifest['wheel_sha256']):
                                raise AssertionError(
                                    'Permission check did not admit the selected wheel'
                                )
                if 'lifecycle/summary.json' in by_path:
                    if not (json.loads(
                            by_path['lifecycle/summary.json'].read_text())
                            ['status'] == receipt['status']):
                        raise AssertionError(
                            'Aggregate differs from lifecycle summary')
                if manifest['mode'] == 'lifecycle':
                    checks = lifecycle_verdict(by_path, manifest)
                    if checks is not None:
                        report['engine_checks'] = checks
                if receipt['status'] in ('blocked', 'failed'):
                    report['status'] = receipt['status']
                elif receipt['status'] == 'passed' and 'error' not in report:
                    report['status'] = 'passed'
            except Exception as error:
                report['status'] = 'failed'
                report['collection_error'] = dict(type=type(error).__name__,
                                                  message=str(error))
        if cleanup_owned:
            report['cleanup'] = cleanup_submission(
                connection, manifest, query_unknown=submission_pending
                or bool(submitted and not report.get('terminal_status')),
                bundle_created=bundle_created, bundle_pending=bundle_pending)
            if report['cleanup']['status'] != 'passed':
                report['status'] = 'failed'
        save(root / 'report.json', report)
    return report


def run_pipeline(root, plans, key_file, timeout):
    """Unlock once and reuse one connection; never dispatch training after failed preflight."""
    report = dict(status='running', started_unix=time.time(), stages={},
                  scope=('Installed-wheel notebook tests; UI excluded'))
    started = time.monotonic()
    connection = None
    try:
        connection = connect(key_file)
        if list(plans) == ['negative']:
            report[
                'scope'] = 'Snowflake notebook permission-denial check only; no model fits or prediction jobs'
        for name in plans:
            stage_root = root / name
            print('Starting notebook:', name, flush=True)
            stage_started = time.monotonic()
            result = execute(stage_root, plans[name], connection, timeout)
            result['elapsed_seconds'] = time.monotonic() - stage_started
            save(stage_root / 'report.json', result)
            report['stages'][name] = dict(
                status=result['status'], report=str(stage_root / 'report.json'),
                elapsed_seconds=result['elapsed_seconds'],
                execution_query_id=result.get('execution_query_id'))
            for key in ('reason', 'error', 'collection_error'):
                if key in result:
                    report['stages'][name][key] = result[key]
            if result.get('engine_checks') is not None:
                report['engine_checks'] = result['engine_checks']
            save(root / 'summary.json', report)
            if result['status'] != 'passed':
                report['status'] = result['status']
                if name == 'preflight':
                    report['stages']['lifecycle'] = dict(
                        status='blocked',
                        reason='Preflight did not pass; no model fits dispatched'
                    )
                break
        else:
            report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = dict(type=type(error).__name__, message=str(error))
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception as error:
                report['close_error'] = dict(type=type(error).__name__,
                                             message=str(error))
                report['status'] = 'failed'
        report['elapsed_seconds'] = time.monotonic() - started
        save(root / 'summary.json', report)
    return report


def boolean(value):
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise argparse.ArgumentTypeError('Expected true or false')


def wait_for_compute(key_file, timeout):
    """Observe only the dedicated pool; never resume or alter cloud resources."""
    deadline = time.monotonic() + timeout
    with connect(key_file) as connection, connection.cursor() as cursor:
        while True:
            cursor.execute('DESCRIBE COMPUTE POOL KUMO_TEST_POOL')
            rows = cursor.fetchall()
            columns = [item[0].lower() for item in cursor.description]
            if len(rows) != 1:
                raise ValueError(
                    'Expected exactly the dedicated test compute pool')
            pool = dict(zip(columns, rows[0]))
            if (pool['name'] != POOL
                    or pool['instance_family'] != 'GEN_X64_G2_8'):
                raise ValueError('Dedicated test pool identity differs')
            if pool.get('error_code'):
                raise RuntimeError('Dedicated test pool error: ' +
                                   str(pool.get('status_message')))
            if pool['state'] in ('ACTIVE', 'IDLE'):
                return dict(pool=POOL, state=pool['state'])
            if pool['state'] not in ('STARTING', 'RESIZING'):
                raise RuntimeError('Dedicated test pool is not starting: ' +
                                   pool['state'])
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    'Dedicated test pool startup deadline exceeded')
            time.sleep(min(10, max(0, deadline - time.monotonic())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', default='V2.9-CPU-PY3.12')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--key-file', type=Path,
                        help=('Encrypted key for the local memory cache'))
    parser.add_argument('--preflight-payload', type=Path,
                        help='Prepared preflight payload directory')
    parser.add_argument('--lifecycle-payload', type=Path,
                        help='Prepared lifecycle payload directory')
    parser.add_argument(
        '--negative-only', action='store_true', help=(
            'Run only the Snowflake notebook permission-denial test; no training'
        ))
    parser.add_argument('--negative-payload', type=Path, default=None)
    parser.add_argument('--timeout', type=int, default=1800,
                        help=('Notebook deadline in seconds; default 1800'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--wait-compute', action='store_true',
                        help='Read-only wait for the dedicated pool startup')
    parser.add_argument(
        '--show-result', type=Path,
        help='Explain a saved run without authentication or another execution')
    parser.add_argument('--payload-manifest', type=Path,
                        help='Manifest produced by prepare.py')
    parser.add_argument(
        '--ui', nargs='?', const=True, default=False, type=boolean,
        help='Only verify an existing run in Snowsight; never starts training')
    parser.add_argument(
        '--ui-run', type=Path,
        help='Retained lifecycle directory; default latest valid receipt')
    parser.add_argument('--ui-credentials', type=Path,
                        help='Owner-only local Snowsight credential JSON')
    parser.add_argument('--ui-timeout', type=int, default=60)
    parser.add_argument(
        '--ui-notebook', type=Path,
        help='Prepared graph notebook configuration; requires --ui')
    args = parser.parse_args(argv)
    if args.wait_compute:
        if (args.execute or args.ui or args.show_result
                or args.negative_only or args.payload_manifest
                or args.ui_notebook or args.ui_run or args.ui_credentials):
            parser.error(
                '--wait-compute cannot be combined with execution modes')
        if args.timeout <= 0:
            parser.error('--timeout must be positive')
        print(json.dumps(wait_for_compute(args.key_file, args.timeout)),
              flush=True)
        return 0
    if args.ui:
        if args.show_result or args.negative_only:
            parser.error(
                '--ui cannot be combined with another selected test mode')
        if args.ui_timeout <= 0:
            parser.error('--ui-timeout must be positive')
        import ui as module
        options = dict(source=args.ui_run, output=args.output_dir,
                       execute=args.execute,
                       credential_file=args.ui_credentials,
                       timeout=args.ui_timeout)
        if args.ui_notebook:
            if args.ui_run:
                parser.error('--ui-notebook cannot be combined with --ui-run')
            options['notebook'] = args.ui_notebook
            options['key_file'] = args.key_file
        result = module.run(**options)
        return result['status'] not in ('planned', 'passed')
    if args.ui_run or args.ui_credentials or args.ui_notebook:
        parser.error(
            '--ui-run, --ui-notebook and --ui-credentials require --ui')
    if args.show_result is not None:
        if args.execute or args.negative_only:
            parser.error(
                '--show-result only reads saved evidence; omit execution options'
            )
        show_saved_result(args.show_result)
        return 0
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if not args.payload_manifest or (
            args.negative_only and
            not args.negative_payload) or (not args.negative_only and not all(
                (args.preflight_payload, args.lifecycle_payload))):
        parser.error(
            'Supply --payload-manifest and the selected prepared payload directories'
        )
    approved = json.loads(args.payload_manifest.read_text())
    root = args.output_dir or Path('/tmp') / ('kumo-snowflake-e2e-' +
                                              uuid4().hex[:12])
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    plans = {}
    selected = [
        ('negative', args.negative_payload)
    ] if args.negative_only else [('preflight', args.preflight_payload),
                                  ('lifecycle', args.lifecycle_payload)]
    for mode, payload in selected:
        print(
            f'Preparing {mode}: verifying reviewed files and copying payload. Large files may take a minute.',
            flush=True)
        plans[mode] = prepare_payload(root / mode, args.runtime, payload,
                                      approved=approved)
        if plans[mode]['mode'] != mode:
            raise ValueError('Payload mode does not match its stage')
    print('Prepared reviewed notebook payloads:', root, flush=True)
    if not args.execute:
        print('Plan only: no connection, credential prompt or cloud execution.')
        return 0
    import signal

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Notebook runner interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    report = run_pipeline(root, plans, args.key_file, args.timeout)
    print_result(report, root / 'summary.json')
    return report['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
