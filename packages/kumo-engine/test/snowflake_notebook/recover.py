# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover owned notebook submissions after an interrupted Actions step."""
import argparse
import json
import time
from pathlib import Path

import launch
import retention


def recover(connection, root, timeout=30):
    root = Path(root)
    report_file = root / 'report.json'
    if not report_file.is_file():
        return dict(status='skipped')
    report = json.loads(report_file.read_text())
    if report.get('cleanup', {}).get('status') == 'passed':
        return dict(status='passed')
    if report.get('retention', {}).get('status') != 'recorded':
        if report.get('cleanup_owned') is True and report.get('operation') in (
                None, 'check_denied_stage_fixture', 'cache_lookup'):
            return launch.cleanup_submission(connection, report['plan'],
                                             query_unknown=False,
                                             bundle_created=False,
                                             bundle_pending=False)
        return dict(status='preserved', reason='ownership_unconfirmed')
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
    folders = [
        args.output_dir / name for name in ('preflight', 'lifecycle')
        if (args.output_dir / name / 'report.json').is_file() and json.loads((
            args.output_dir / name / 'report.json'
        ).read_text()).get('cleanup', {}).get('status') != 'passed'
    ]
    if not folders:
        print('No retained submissions to recover')
        return
    from resources import require
    require()
    results = {}
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
        results = {
            folder.name: dict(status='preserved',
                              reason='connection_unavailable')
            for folder in folders
        }
    (args.output_dir /
     'recovery.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
