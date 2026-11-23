# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover owned notebook submissions after an interrupted Actions step."""
import argparse
import json
import tempfile
import time
from pathlib import Path

import launch
import retention

UNSUBMITTED = ('prepare_payload', 'check_denied_stage_fixture', 'cache_lookup')
RECORDED_UNSUBMITTED = ('prepare_bundle', 'bundle_created')


def unsubmitted_ledger(cursor, root, manifest, *, required=False):
    path = root / 'retention-ledger.json'
    if not path.is_file():
        if required:
            raise ValueError('Missing owned retention ledger')
        return
    value = json.loads(path.read_text())
    targets = retention.validate(value)
    if value != retention.ledger(manifest, created=value['created_unix']):
        raise ValueError('Unsubmitted ledger differs from the owned plan')
    cursor.execute('LIST ' + targets['ledger'])
    if not cursor.fetchall():
        if required:
            raise ValueError('Missing remote owned retention ledger')
        return
    with tempfile.TemporaryDirectory(dir=root) as work:
        actual = launch.download_file(cursor, targets['ledger'], Path(work))
        if json.loads(actual.read_text()) != value:
            raise ValueError('Remote unsubmitted ledger differs')
    return value, targets


def remove_unsubmitted_ledger(connection, root, manifest):
    with connection.cursor() as cursor:
        owned = unsubmitted_ledger(cursor, root, manifest)
        if owned is None:
            return
        _, targets = owned
        cursor.execute('REMOVE ' + targets['ledger'])


def recover(connection, root, timeout=30):
    root = Path(root)
    report_file = root / 'report.json'
    if not report_file.is_file():
        return dict(status='skipped')
    report = json.loads(report_file.read_text())
    recorded_unsubmitted = (report.get('cleanup_owned') is True
                            and report.get('operation') in RECORDED_UNSUBMITTED)
    if report.get('cleanup', {}).get('status') == 'passed':
        if (recorded_unsubmitted
                or (report.get('retention', {}).get('status') != 'recorded'
                    and report.get('cleanup_owned') is True
                    and report.get('operation') in UNSUBMITTED)):
            remove_unsubmitted_ledger(connection, root, report['plan'])
        return dict(status='passed')
    if report.get('retention', {}).get('status') != 'recorded':
        if report.get('cleanup_owned') is True and report.get(
                'operation') in UNSUBMITTED:
            result = launch.cleanup_submission(connection, report['plan'],
                                               query_unknown=False,
                                               bundle_created=False,
                                               bundle_pending=False)
            if result['status'] == 'passed':
                remove_unsubmitted_ledger(connection, root, report['plan'])
            return result
        return dict(status='preserved', reason='ownership_unconfirmed')
    if recorded_unsubmitted:
        manifest = report['plan']
        before = None
        with connection.cursor() as cursor:
            value, targets = unsubmitted_ledger(cursor, root, manifest,
                                                required=True)
            if report['operation'] == 'bundle_created':
                ended = time.time()
                before = retention.inspect(cursor, 'CODE BUNDLE',
                                           targets['bundle'], value, ended)
                if retention.inspect(cursor, 'CODE BUNDLE', targets['bundle'],
                                     value, ended) != before:
                    raise ValueError('Owned bundle changed during recovery')
        result = launch.cleanup_submission(connection, manifest,
                                           query_unknown=False,
                                           bundle_created=before is not None,
                                           bundle_pending=False)
        if result['status'] == 'passed':
            remove_unsubmitted_ledger(connection, root, manifest)
        return result
    value = json.loads((root / 'retention-ledger.json').read_text())
    manifest = report['plan']
    targets = retention.validate(value)
    if manifest['attempt'] != value['attempt'] or manifest['bundle'] != targets[
            'bundle']:
        raise ValueError('Recovery report and ledger name different attempts')
    with connection.cursor() as cursor:
        query = retention.query_proof(cursor, value)
        query_id = query.get('query_id')
        if not isinstance(query_id, str) or not query_id:
            raise ValueError('Recovery needs the owned query identity')
        if str(query.get('execution_status',
                         '')).upper() not in retention.TERMINAL:
            cursor.execute('SELECT SYSTEM$CANCEL_QUERY(%s)', (query_id, ))
            deadline = time.monotonic() + timeout
            while connection.is_still_running(
                    connection.get_query_status(query_id)):
                if time.monotonic() >= deadline:
                    return dict(status='preserved', reason='query_not_terminal')
                time.sleep(1)
        ended = retention.terminal_proof(cursor, value, time.time(), 0)
        before = retention.inspect(cursor, 'CODE BUNDLE', targets['bundle'],
                                   value, ended)
        if retention.inspect(cursor, 'CODE BUNDLE', targets['bundle'], value,
                             ended) != before:
            raise ValueError('Owned bundle changed during recovery')
    return launch.cleanup_submission(connection, manifest, query_unknown=False,
                                     bundle_created=before is not None,
                                     bundle_pending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('Plan only; no recovery connection or mutation')
        return
    folders = []
    results = {}
    for name in ('preflight', 'lifecycle'):
        folder = args.output_dir / name
        if not (folder / 'report.json').is_file():
            continue
        try:
            report = json.loads((folder / 'report.json').read_text())
            if not isinstance(report, dict) or any(
                    not isinstance(report.get(key, {}), dict)
                    for key in ('cleanup', 'retention')):
                raise ValueError('Invalid report')
        except (OSError, ValueError):
            results[name] = dict(status='preserved',
                                 reason='journal_unavailable')
            continue
        unsubmitted = ((folder / 'retention-ledger.json').is_file()
                       and report.get('cleanup_owned') is True and
                       (report.get('operation') in RECORDED_UNSUBMITTED or
                        (report.get('retention', {}).get('status') != 'recorded'
                         and report.get('operation') in UNSUBMITTED)))
        if report.get('cleanup', {}).get('status') != 'passed' or unsubmitted:
            folders.append(folder)
    if not folders:
        if results:
            launch.save(args.output_dir / 'recovery.json', results)
            print(json.dumps(results))
        else:
            print('No retained submissions to recover')
        return
    from resources import require
    require()
    try:
        with launch.connect(None) as connection:
            for folder in folders:
                try:
                    results[folder.name] = recover(connection, folder)
                except Exception:
                    results[folder.name] = dict(
                        status='preserved',
                        reason='proof_or_provider_unavailable')
    except Exception:
        results.update({
            folder.name:
            dict(status='preserved', reason='connection_unavailable')
            for folder in folders
        })
    launch.save(args.output_dir / 'recovery.json', results)
    print(json.dumps(results))


if __name__ == '__main__':
    main()
