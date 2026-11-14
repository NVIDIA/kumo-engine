# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover stale attempts from private control records in the test Volume."""
import argparse
import hashlib
import hmac
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import quote, urlencode

from client import VOLUME, WORKSPACE_DIR, Client, configuration

MAX_RECORDS = 200
MAX_ATTEMPTS = 20
MAX_BYTES = 131072
MAX_FILES = 10000


def scope(config):
    if (config.get('volume') != VOLUME
            or config.get('workspace_dir') != WORKSPACE_DIR
            or not config.get('expected_user')):
        raise ValueError(
            'Retention requires the fixed test resources and identity')
    return {
        key: config[key]
        for key in ('host', 'volume', 'workspace_dir', 'expected_user')
    }


def identity(client, config):
    scope(config)
    if client.api('/api/2.0/preview/scim/v2/Me').get(
            'userName') != config['expected_user']:
        raise ValueError(
            'Retention PAT identity differs from the test principal')


def file_path(config, attempt):
    if not isinstance(attempt, str) or not re.fullmatch('[a-f0-9]{12}',
                                                        attempt):
        raise ValueError('Invalid retention attempt')
    return config['volume'] + '/retention/' + attempt + '/resources.json'


def signature(client, payload):
    token = getattr(client, 'token', None)
    if not isinstance(token, str) or not token:
        raise ValueError('Retention requires the authenticated PAT')
    canonical = json.dumps(payload, sort_keys=True,
                           separators=(',', ':')).encode()
    return hmac.new(token.encode(),
                    b'kumo-engine-databricks-retention-v1\0' + canonical,
                    hashlib.sha256).hexdigest()


def read(client, path):
    data = client.api('/api/2.0/fs/files' + quote(path, safe='/'), binary=True,
                      max_bytes=MAX_BYTES)
    if not isinstance(data, bytes) or len(data) > MAX_BYTES:
        raise ValueError('Retention record exceeds its byte budget')
    envelope = json.loads(data)
    if (not isinstance(envelope, dict)
            or set(envelope) != {'payload', 'hmac_sha256'}
            or not isinstance(envelope['hmac_sha256'], str)
            or not hmac.compare_digest(envelope['hmac_sha256'],
                                       signature(client, envelope['payload']))):
        raise ValueError('Retention signature cannot be verified')
    return envelope['payload']


def remove_empty_directory(client, record_path):
    endpoint = ('/api/2.0/fs/directories' +
                quote(record_path.rsplit('/', 1)[0], safe='/'))
    try:
        listing = client.api(endpoint + '?page_size=1')
    except RuntimeError as error:
        if str(error) == 'Databricks API HTTP 404':
            return True
        raise
    if listing.get('contents') or listing.get('next_page_token'):
        return False
    client.api(endpoint, 'DELETE')
    return True


class ControlRecord:
    def __init__(self, client, config, spec, lifecycle):
        self.client, self.config = client, config
        self.spec = {
            key: spec[key]
            for key in ('attempt', 'notebook_path', 'output')
        }
        if lifecycle:
            self.spec['experiment'] = spec['experiment']
        self.kind = 'lifecycle' if lifecycle else 'graph'
        self.path = file_path(config, spec['attempt'])
        self.previous = None

    def document(self, ledger, root):
        submitted = Path(root) / 'submitted-run.json'
        return dict(
            version=1, config=scope(self.config), kind=self.kind,
            spec=self.spec, ledger=ledger, submitted=json.loads(
                submitted.read_text()) if submitted.is_file() else None)

    def put(self, value, overwrite):
        raw = (json.dumps(
            dict(payload=value, hmac_sha256=signature(self.client, value)),
            sort_keys=True) + '\n').encode()
        if len(raw) > MAX_BYTES:
            raise ValueError('Retention record exceeds its byte budget')
        self.client.api(
            '/api/2.0/fs/files' + quote(self.path, safe='/') + '?overwrite=' +
            str(overwrite).lower(), 'PUT', raw=raw)
        self.previous = json.loads(json.dumps(value))

    def create(self, ledger, root):
        identity(self.client, self.config)
        self.client.api(
            '/api/2.0/fs/directories' +
            quote(self.path.rsplit('/', 1)[0], safe='/'), 'PUT')
        self.put(self.document(ledger, root), False)

    def bind(self, ledger, root):
        identity(self.client, self.config)
        value = self.document(ledger, root)
        if read(self.client, self.path) != value:
            raise ValueError(
                'Remote retention state differs from recovery input')
        self.previous = json.loads(json.dumps(value))

    def save(self, ledger, root):
        if read(self.client, self.path) != self.previous:
            raise ValueError('Retention record changed during cleanup')
        self.put(self.document(ledger, root), True)

    def complete(self):
        if read(self.client, self.path) != self.previous:
            raise ValueError('Retention record changed during cleanup')
        self.client.api('/api/2.0/fs/files' + quote(self.path, safe='/'),
                        'DELETE')
        try:
            return remove_empty_directory(self.client, self.path)
        except RuntimeError:
            logging.getLogger(__name__).warning(
                'Attempt resources and control record removed; '
                'the empty control directory will be retried by retention')
            return False


def records(client, config):
    prefix = config['volume'] + '/retention'
    endpoint = '/api/2.0/fs/directories' + quote(prefix, safe='/')
    entries, tokens, token = [], set(), None
    for _ in range(10):
        try:
            page = client.api(endpoint + '?' + urlencode(
                dict(page_size=100, **({
                    'page_token': token
                } if token else {}))))
        except RuntimeError as error:
            if str(error) == 'Databricks API HTTP 404' and not token:
                return []
            raise
        for entry in page.get('contents', []):
            path = entry.get('path', '').removesuffix('/')
            if (entry.get('is_directory') is not True
                    or path.rsplit('/', 1)[0] != prefix
                    or not re.fullmatch('[a-f0-9]{12}',
                                        path.rsplit('/', 1)[-1])):
                raise ValueError('Retention listing escaped its control prefix')
            if path in entries:
                raise ValueError('Retention listing repeated an attempt')
            entries.append(path)
        if len(entries) > MAX_RECORDS:
            raise ValueError('Retention inventory exceeds the record budget')
        token = page.get('next_page_token')
        if not token:
            return [path + '/resources.json' for path in entries]
        if not isinstance(token, str) or token in tokens:
            raise ValueError('Retention pagination is ambiguous')
        tokens.add(token)
    raise ValueError('Retention pagination exceeds the page budget')


def sweep(client, config, root, *, execute=False, ttl_hours=24, now_ms=None):
    from cleanup import AttemptResources, recover

    if type(ttl_hours) is not int or ttl_hours < 24:
        raise ValueError('Retention TTL must be at least 24 hours')
    identity(client, config)
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    candidates, preserved = [], {}
    empty_controls_removed = 0

    def preserve(reason):
        preserved[reason] = preserved.get(reason, 0) + 1

    for path in records(client, config):
        try:
            value = read(client, path)
            ledger, spec = value['ledger'], value['spec']
            if (set(value) != {
                    'version', 'config', 'kind', 'spec', 'ledger', 'submitted'
            } or value['version'] != 1 or value['config'] != scope(config)
                    or value['kind'] not in ('graph', 'lifecycle')
                    or ledger.get('admitted') is not True
                    or type(ledger.get('started_ms')) is not int
                    or not 0 < ledger['started_ms'] <= now_ms
                    or path != file_path(config, spec['attempt'])
                    or ledger.get('attempt') != spec['attempt']):
                raise ValueError('Foreign retention record')
            if now_ms - ledger['started_ms'] < ttl_hours * 3600000:
                preserve('recent')
                continue
            flags = ('notebook_created', 'admitted', 'experiment_pending',
                     'submission_pending', 'import_pending',
                     'experiment_removed', 'model_removed', 'notebook_removed')
            if any(key in ledger and type(ledger[key]) is not bool
                   for key in flags):
                raise ValueError('Retention state flags are ambiguous')
            submitted = value['submitted']
            unknown_creation = any(ledger.get(key) for key in
                                   ('import_pending', 'experiment_pending'))
            if unknown_creation or (ledger.get('submission_pending')
                                    and submitted is None):
                preserve('unknown_submission_or_creation')
                continue
            if submitted is not None and (set(submitted) != {'run_id'} or type(
                    submitted['run_id']) is not int
                                          or submitted['run_id'] <= 0):
                raise ValueError('Invalid submitted run receipt')
            directory = root / spec['attempt']
            directory.mkdir(exist_ok=True)
            for name in ('submitted-run.json', 'cleanup.json'):
                (directory / name).unlink(missing_ok=True)
            for name, data in [('scenario.json', spec),
                               ('resources.json', ledger)]:
                (directory / name).write_text(json.dumps(data) + '\n')
            if submitted is not None:
                (directory /
                 'submitted-run.json').write_text(json.dumps(submitted) + '\n')
            AttemptResources(client, config, spec, directory,
                             lifecycle=value['kind'] == 'lifecycle',
                             resume=True)
            candidates.append((ledger['started_ms'], directory))
        except Exception as error:
            if execute and str(error) == 'Databricks API HTTP 404':
                try:
                    if remove_empty_directory(client, path):
                        empty_controls_removed += 1
                        continue
                except RuntimeError:
                    pass
            preserve('unproven_record')
    cleaned = 0
    budget = [MAX_FILES]
    for _, directory in sorted(candidates)[:MAX_ATTEMPTS]:
        if execute:
            try:
                recover(directory, client, config, file_budget=budget)
                cleaned += 1
            except Exception:
                preserve('unproven_or_failed_cleanup')
    result = dict(status='passed', execute=execute, eligible=len(candidates),
                  cleaned=cleaned, preserved=preserved,
                  empty_controls_removed=empty_controls_removed,
                  deferred=max(0,
                               len(candidates) - MAX_ATTEMPTS))
    (root /
     'retention-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--ttl-hours', type=int, default=24)
    args = parser.parse_args()
    config = configuration()
    result = sweep(Client(config), config, args.output_dir,
                   execute=args.execute, ttl_hours=args.ttl_hours)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
