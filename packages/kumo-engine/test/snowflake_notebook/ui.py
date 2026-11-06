# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read retained lifecycle receipts and verify their rendered Snowsight results."""
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


class UIBlocked(RuntimeError):
    """A UI prerequisite is unavailable."""


def expectations(source):
    source = Path(source).resolve(strict=True)
    download = source / 'lifecycle/download'
    receipt = json.loads((download / 'lifecycle-result.json').read_text())
    identity = receipt['identity']
    if (identity['account'], identity['database'],
            identity['schema']) != ('SVB14499', 'KUMO_TEST_DB', 'KUMO_TEST'):
        raise ValueError('Source receipt names another Snowflake target')
    candidate = receipt['wheel_sha256']
    if not re.fullmatch('[0-9a-f]{64}', candidate):
        raise ValueError('Invalid candidate hash')
    checksums = {}

    def read(logical):
        names = [
            name for name, path in receipt['logical_files'].items()
            if path == logical
        ]
        if len(names) != 1 or not re.fullmatch(r'[A-Za-z0-9_.-]+', names[0]):
            raise ValueError('Missing or invalid receipt: ' + logical)
        path = download / names[0]
        if path.is_symlink():
            raise ValueError('Receipt cannot be a symlink')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != receipt['artifacts'][names[0]]:
            raise ValueError('Receipt checksum mismatch: ' + logical)
        checksums[logical] = digest
        return data.decode()

    phases = {}
    urls = set()
    for name in ('fit', 'score', 'evaluate', 'negative'):
        phase = json.loads(read('lifecycle/' + name + '.json'))
        if phase['status'] != 'passed' or phase['environment'][
                'sha256'] != candidate:
            raise ValueError('Source phase did not pass on the candidate: ' +
                             name)
        phases[name] = phase
        if name != 'negative':
            urls.update(
                re.findall(r'https://app\.snowflake\.com/[^\s\x1b]+',
                           read('lifecycle/' + name + '.log')))
    model = phases['fit']['result']['model']
    if not re.fullmatch(r'EGU_MLOPS_[A-F0-9]{12}', model):
        raise ValueError('Unexpected test model identity')
    runs = []

    def visit(value, kind):
        if isinstance(value, dict):
            if all(key in value for key in ('run', 'experiment', 'metrics',
                                            'history', 'params')):
                run = value['run']
                if (not re.fullmatch('[A-Z0-9_]+', run) or value['experiment']
                        != f'"KUMO_TEST_DB"."KUMO_TEST"."{model}"'
                        or value['params'].get('kumo.model_name') != model):
                    raise ValueError(
                        'Tracking reference differs from selected model')
                canonical = f'https://app.snowflake.com/_deeplink/#/experiments/databases/KUMO_TEST_DB/schemas/KUMO_TEST/experiments/{model}/runs/{run}'
                matching = [url for url in urls if url == canonical]
                if len(matching) != 1:
                    raise ValueError('No unique observed Snowsight URL for ' +
                                     run)
                runs.append(
                    dict(name=run, kind=kind, url=matching[0],
                         metrics=value['metrics'], history=value['history'],
                         params=value['params']))
            else:
                for child in value.values():
                    visit(child, kind)
        elif isinstance(value, list):
            for child in value:
                visit(child, kind)

    for phase, count in (('fit', 2), ('score', 4), ('evaluate', 4)):
        before = len(runs)
        visit(phases[phase]['result'], phase)
        if len(runs) - before != count:
            raise ValueError('Missing tracking references: ' + phase)
    if len({run['name'] for run in runs}) != len(runs):
        raise ValueError('Duplicate tracking references')
    versions = phases['fit']['result']['versions']
    if [version['version'] for version in versions] != ['V1', 'V2']:
        raise ValueError('Both registered versions are required')
    return dict(
        source_run=str(source), candidate_sha256=candidate,
        source_checksums=checksums, account='lfwgwbp/kumo_us_west_test_5',
        database='KUMO_TEST_DB', schema='KUMO_TEST', model=model, versions=[
            dict(name=v['version'], bundle=v['bundle']) for v in versions
        ], runs=runs,
        scope='Rendered MLOps runs, metrics, epoch charts, and model versions')


def latest_source():
    candidates = sorted(
        Path('/tmp').glob('egu-snowflake-e2e-*'),
        key=lambda p: p.stat().st_mtime, reverse=True)
    for source in candidates:
        if not source.is_dir():
            continue
        try:
            expectations(source)
            return source
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise UIBlocked(
        'No complete retained lifecycle found; supply --ui-run RUN_DIRECTORY')


def credentials(path=None):
    if path is None and os.environ.get('SNOWFLAKE_UI_CREDENTIALS'):
        value = json.loads(os.environ['SNOWFLAKE_UI_CREDENTIALS'])
    else:
        if os.environ.get('GITHUB_ACTIONS') == 'true':
            raise UIBlocked(
                'GitHub runs require the environment browser secret')
        path = Path(
            path or Path.home() / '.config/kumo/snowflake-ui/credentials.json')
        with path.open() as stream:
            info = os.fstat(stream.fileno())
            if path.is_symlink() or not stat.S_ISREG(
                    info.st_mode) or info.st_uid != os.getuid(
                    ) or info.st_mode & 0o077:
                raise UIBlocked(
                    'Browser credentials must be an owner-only regular file')
            value = json.load(stream)
    if set(value) != {
            'account_url', 'user', 'password', 'totp_secret'
    } or not all(isinstance(v, str) and v for v in value.values()):
        raise UIBlocked('Incomplete browser credentials')
    if value[
            'account_url'] != 'https://app.snowflake.com/lfwgwbp/kumo_us_west_test_5/':
        raise UIBlocked('Browser credentials name another account')
    if value['user'] != 'KUMO_SNOWFLAKE_UI_TEST':
        raise UIBlocked('Use the dedicated browser test identity')
    if not re.fullmatch('[A-Z2-7]+={0,6}', value['totp_secret']):
        raise UIBlocked('Invalid TOTP setup secret')
    return value


def run(source=None, output=None, execute=False, credential_file=None,
        timeout=60, notebook=None, key_file=None):
    root = Path(output) if output else Path(
        tempfile.mkdtemp(prefix='egu-snowflake-ui-'))
    if output:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = dict(status='blocked', mode='ui_only', training_submitted=False,
                  assertions=[])
    try:
        directory = Path(__file__).parent
        report['test_revision'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=directory, text=True).strip()
        report['test_files'] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [*directory.glob('*.py'), *directory.glob('*.cjs')])
        }
        if notebook:
            from graph_notebook import load_config
            expected = load_config(notebook)
            report.update(
                mode='notebook_saved_ui'
                if expected['action'] == 'verify' else 'notebook_graph_ui',
                candidate_sha256=expected['candidate']['wheel_sha256'])
        else:
            expected = expectations(source or latest_source())
            report.update(source_run=expected['source_run'],
                          candidate_sha256=expected['candidate_sha256'])
        (root /
         'expectations.json').write_text(json.dumps(expected, indent=2) + '\n')
        if not execute:
            report.update(
                status='planned',
                reason='UI plan only; no credentials loaded or browser launched'
            )
        else:
            from resources import require

            try:
                target = require()
            except ValueError as error:
                raise UIBlocked(str(error)) from None
            auth = credentials(credential_file)
            if (auth['account_url'] != target['account_url']
                    or auth['user'] != target['browser_user']):
                raise UIBlocked(
                    'Browser identity differs from the protected test configuration'
                )
            node = shutil.which('node')
            if node is None and (Path.cwd() / '.nodeenv/bin/node').is_file():
                node = str(Path.cwd() / '.nodeenv/bin/node')
            if not node:
                raise UIBlocked('Node.js is unavailable')
            module = os.environ.get('KUMO_PLAYWRIGHT_MODULE')
            if not module:
                for folder in (Path(__file__).parent /
                               'browser/node_modules/playwright',
                               Path.cwd() / 'ui/node_modules/playwright',
                               Path(__file__).resolve().parents[4] /
                               'ui/node_modules/playwright'):
                    if folder.exists():
                        module = str(folder)
                        break
            if not module:
                raise UIBlocked(
                    'Playwright is unavailable; install the existing UI test dependencies'
                )
            config = dict(expected=expected, auth=auth, output=str(root),
                          timeout=timeout * 1000, playwright_module=module)
            if notebook:
                config['notebook'] = expected
                config['python'] = sys.executable
                config['key_file'] = str(key_file) if key_file else None
            environment = dict(os.environ)
            child = subprocess.run(
                [node, str(Path(__file__).with_name('ui.cjs'))],
                input=json.dumps(config), text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=3600 if notebook else timeout *
                (len(expected['runs']) + 8), check=False, env=environment)
            result_file = root / 'browser-result.json'
            if not result_file.is_file():
                raise UIBlocked(
                    'Browser worker did not produce a result; verify Chromium and its shared libraries'
                )
            result = json.loads(result_file.read_text())
            report.update(result)
            if notebook and (root / 'notebook-checks.json').exists():
                report['notebook_checks'] = json.loads(
                    (root / 'notebook-checks.json').read_text())
            if notebook and (root / 'notebook-receipt.json').exists():
                from graph_notebook import collect
                try:
                    receipt = json.loads(
                        (root / 'notebook-receipt.json').read_text())
                    report['exports'] = collect(receipt, expected,
                                                root / 'exports', key_file)
                except Exception as error:
                    report.update(
                        status='failed', collection_error=type(error).__name__ +
                        ': ' + str(error))
                    report.setdefault(
                        'reason',
                        'Graph export collection failed: ' + str(error))
            if child.returncode and report['status'] == 'passed':
                report.update(status='failed',
                              reason='Browser exited unsuccessfully')
    except (UIBlocked, FileNotFoundError, subprocess.TimeoutExpired) as error:
        report.update(status='blocked', reason=str(error))
    except Exception as error:
        report.update(status='failed',
                      reason=type(error).__name__ + ': ' + str(error))
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('UI result:', report['status'], 'Summary:', root / 'summary.json',
          flush=True)
    if report.get('reason'):
        print(report['reason'], flush=True)
    return report
