# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expire ledger-bound, terminal Snowflake lifecycle attempts after one day."""
import argparse
import json
import re
import tempfile
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

import launch

CONTROL = launch.STAGE + '/mlops-validation/retention/'
TTL = 24 * 60 * 60
HISTORY_WINDOW = 6 * TTL
TERMINAL = {'SUCCESS', 'FAILED_WITH_ERROR', 'FAILED_WITH_INCIDENT'}


def resources(attempt):
    if not isinstance(attempt, str) or not re.fullmatch('[a-f0-9]{12}',
                                                        attempt):
        raise ValueError('Retention requires a generated attempt identifier')
    name = 'KUMO_TEST_MLOPS_' + attempt.upper()
    bundle = f'{launch.DATABASE}.{launch.SCHEMA}.MLOPS_PROBE_{attempt.upper()}'
    return dict(
        bundle=bundle,
        submission=launch.STAGE + '/notebook-output-probe/' + attempt + '/',
        model=f'{launch.DATABASE}.{launch.SCHEMA}.{name}',
        experiment=f'{launch.DATABASE}.{launch.SCHEMA}.{name}',
        artifacts=launch.STAGE + '/mlops-validation/' + attempt + '/',
        ledger=CONTROL + attempt + '/retention-ledger.json')


def ledger(manifest, created=None):
    targets = resources(manifest['attempt'])
    if manifest['bundle'] != targets['bundle'] or manifest[
            'stage_prefix'] + '/' != targets['submission']:
        raise ValueError('Retention cannot adopt non-attempt resources')
    return dict(version=1, attempt=manifest['attempt'],
                created_unix=int(time.time() if created is None else created),
                lifecycle=manifest.get('mode') == 'lifecycle',
                query_tag='kumo-notebook-test:' + manifest['attempt'],
                execute_sql=manifest['execute_sql'])


def validate(value):
    if not isinstance(value, dict) or set(value) != {
            'version', 'attempt', 'created_unix', 'lifecycle', 'query_tag',
            'execute_sql'
    }:
        raise ValueError('Invalid retention ledger fields')
    targets = resources(value['attempt'])
    if (value['version'] != 1 or type(value['created_unix']) is not int
            or value['created_unix'] <= 0
            or type(value['lifecycle']) is not bool
            or value['query_tag'] != 'kumo-notebook-test:' + value['attempt']
            or not isinstance(value['execute_sql'], str)
            or not value['execute_sql'].startswith('EXECUTE CODE BUNDLE ' +
                                                   targets['bundle'] +
                                                   ' ENTRYPOINT = ')
            or len(value['execute_sql']) > 4096):
        raise ValueError('Invalid retention ledger identity')
    return targets


def write(cursor, root, manifest):
    value = ledger(manifest)
    targets = validate(value)
    cursor.execute('LIST ' + targets['ledger'])
    if cursor.fetchall():
        raise ValueError('Retention ledger already exists; refusing ownership')
    filename = Path(root) / 'retention-ledger.json'
    launch.save(filename, value)
    cursor.execute('PUT ' +
                   launch.literal('file://' + str(filename.resolve())) + ' ' +
                   CONTROL + value['attempt'] +
                   '/ AUTO_COMPRESS=FALSE OVERWRITE=FALSE')
    rows = cursor.fetchall()
    if not rows or not all(
            len(row) > 6 and str(row[6]).upper() == 'UPLOADED' for row in rows):
        raise RuntimeError(
            'Retention ledger upload failed; no notebook submitted')
    # GET and compare before submitting: PUT success alone is not durable proof.
    with tempfile.TemporaryDirectory(prefix='kumo-retention-check-',
                                     dir=root) as work:
        actual = launch.download_file(
            cursor, CONTROL + value['attempt'] + '/retention-ledger.json',
            Path(work))
        if json.loads(actual.read_text()) != value:
            raise RuntimeError('Retention ledger differs after upload')
    return value


def rows(cursor):
    names = [
        str(
            getattr(
                column, 'name',
                column[0] if isinstance(column, tuple) else column)).lower()
        for column in cursor.description
    ]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def timestamp(value):
    if hasattr(value, 'timestamp'):
        return value.timestamp()
    raise ValueError('Provider timestamp is required')


def query_proof(cursor, value):
    validate(value)
    cursor.execute(
        f'SELECT query_id,query_tag,user_name,role_name,query_text,execution_status,end_time '
        f'FROM TABLE({launch.DATABASE}.INFORMATION_SCHEMA.QUERY_HISTORY_BY_USER('
        f'USER_NAME => {launch.literal(launch.USER)}, RESULT_LIMIT => 10000))')
    history = rows(cursor)
    if len(history) >= 10000:
        raise ValueError('Query history is truncated; preserving attempt')
    bundle = resources(value['attempt'])['bundle']
    matches = [
        row for row in history
        if str(row.get('query_text', '')).startswith('EXECUTE CODE BUNDLE ' +
                                                     bundle + ' ')
    ]
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous notebook query history')
    query = matches[0]
    if (query['query_tag'] != value['query_tag']
            or query['user_name'] != launch.USER
            or query['role_name'] != launch.ROLE
            or query['query_text'] != value['execute_sql']):
        raise ValueError('Notebook terminal query identity is unconfirmed')
    return query


def terminal_proof(cursor, value, now, ttl):
    validate(value)
    if now - value['created_unix'] < ttl or now - value[
            'created_unix'] > HISTORY_WINDOW:
        raise ValueError(
            'Attempt is recent or outside the provable query-history window')
    query = query_proof(cursor, value)
    if (not isinstance(query.get('execution_status'), str)
            or query['execution_status'].upper() not in TERMINAL):
        raise ValueError('Notebook terminal query identity is unconfirmed')
    ended = timestamp(query['end_time'])
    if not value['created_unix'] <= ended <= now - ttl:
        raise ValueError('Notebook terminal state has not passed retention')
    return ended


def inspect(cursor, kind, qualified, value, ended):
    cursor.execute(
        'SHOW ' + {
            'MODEL': 'MODELS',
            'EXPERIMENT': 'EXPERIMENTS',
            'CODE BUNDLE': 'CODE BUNDLES'
        }[kind] + f' IN SCHEMA {launch.DATABASE}.{launch.SCHEMA}')
    found = [
        row for row in rows(cursor)
        if row['name'] == qualified.rsplit('.', 1)[-1]
    ]
    if not found:
        return None
    if len(found) != 1:
        raise ValueError('Ambiguous retained object')
    observed = found[0]
    owner = 'owner_role' if kind == 'CODE BUNDLE' else 'owner'
    required = {'database_name', 'schema_name', 'created_on', owner}
    if not required <= observed.keys():
        raise ValueError('Provider ownership metadata is incomplete')
    if (observed['database_name'] != launch.DATABASE
            or observed['schema_name'] != launch.SCHEMA
            or observed[owner] != launch.ROLE or
        (kind == 'CODE BUNDLE' and observed.get('owner_role_type') != 'ROLE')
            or not value['created_unix'] <= timestamp(
                observed['created_on']) <= ended):
        raise ValueError('Retained object ownership or creation time changed')
    return observed


def protected_locations():
    root = Path(__file__).parent
    assets = json.loads((root / 'graph-assets.json').read_text())
    # A retired dependency cache lived under an attempt prefix.
    legacy = launch.STAGE + '/notebook-output-probe/b85262f060c8/payload/wheelhouse.tar'
    return (*[asset['location'] for asset in assets.values()], legacy)


def expire(cursor, value, *, execute=False, now=None, ttl=TTL):
    if type(ttl) is not int or ttl < TTL:
        raise ValueError('Retention must be at least 24 hours')
    now = time.time() if now is None else now
    targets = validate(value)
    if any(
            location.startswith(targets[key])
            for location in protected_locations()
            for key in ('submission', 'artifacts')):
        raise ValueError('Attempt contains a protected shared cache asset')
    ended = terminal_proof(cursor, value, now, ttl)
    objects = [('CODE BUNDLE', targets['bundle'])]
    if value['lifecycle']:
        objects += [('MODEL', targets['model']),
                    ('EXPERIMENT', targets['experiment'])]
    snapshots = [(kind, name, inspect(cursor, kind, name, value, ended))
                 for kind, name in objects]
    if execute:
        terminal_proof(cursor, value, now, ttl)
        for kind, name, before in snapshots:
            if inspect(cursor, kind, name, value, ended) != before:
                raise ValueError('Retained object changed during cleanup')
            if before is not None:
                cursor.execute('DROP ' + kind + ' ' + name)
        cursor.execute('REMOVE ' + targets['submission'])
        if value['lifecycle']:
            cursor.execute('REMOVE ' + targets['artifacts'])
        cursor.execute('REMOVE ' + CONTROL + value['attempt'] + '/')
    return dict(status='passed' if execute else 'planned',
                attempt=value['attempt'])


def sweep(connection, *, execute=False, limit=100, ttl=TTL):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('Retention sweep limit must be between 1 and 100')
    if type(ttl) is not int or ttl < TTL:
        raise ValueError('Retention must be at least 24 hours')
    now = time.time()
    result = dict(status='passed', expired=0, planned=0, preserved=0, recent=0,
                  outside_history=0, cleanup_errors=0)
    with connection.cursor() as cursor:
        cursor.execute('LIST ' + CONTROL)
        inventory = rows(cursor)
        # LIST returns a stage-relative name, never an SQL location.
        prefix = launch.STAGE.rsplit(
            '.', 1)[-1].lstrip('@').lower() + '/mlops-validation/retention/'
        eligible = []
        for row in inventory:
            if not re.fullmatch(
                    re.escape(prefix) + r'[a-f0-9]{12}/retention-ledger\.json',
                    row['name']):
                continue
            try:
                modified = row['last_modified']
                if isinstance(modified, str):
                    modified = parsedate_to_datetime(modified)
                age = now - timestamp(modified)
                if age < ttl:
                    result['recent'] += 1
                elif age > HISTORY_WINDOW:
                    result['outside_history'] += 1
                else:
                    eligible.append((age, row['name']))
            except (ValueError, KeyError, TypeError):
                result['preserved'] += 1
        # Unprovable old ledgers must not consume the next sweep's budget.
        selected = [name for _, name in sorted(eligible, reverse=True)[:limit]]
        for name in selected:
            attempt = name[len(prefix):].split('/')[0]
            try:
                with tempfile.TemporaryDirectory(
                        prefix='kumo-retention-read-') as work:
                    file = launch.download_file(
                        cursor, CONTROL + attempt + '/retention-ledger.json',
                        Path(work))
                    if file.stat().st_size > 65536:
                        raise ValueError('Oversized retention ledger')
                    value = json.loads(file.read_text())
                if value['attempt'] != attempt:
                    raise ValueError('Retention ledger path identity differs')
                expire(cursor, value, execute=execute, ttl=ttl, now=now)
                result['expired' if execute else 'planned'] += 1
            except (ValueError, KeyError, TypeError):
                result['preserved'] += 1
            except Exception:
                result['preserved'] += 1
                result['cleanup_errors'] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--key-file', type=Path)
    args = parser.parse_args()
    from resources import require
    require()
    with launch.connect(args.key_file) as connection:
        print(
            json.dumps(sweep(connection, execute=args.execute,
                             limit=args.limit)))


if __name__ == '__main__':
    main()
