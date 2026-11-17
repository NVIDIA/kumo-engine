# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retry cleanup of exactly one prepared graph attempt in a fresh browser."""
import argparse
import json
from pathlib import Path

import launch
from graph_notebook import load_config

import ui


def cleanup(output, *, execute=False, key_file=None, credential_file=None,
            notebook=None, ui_output=None, remove_stage=True):
    output = Path(output)
    configuration = Path(notebook) if notebook else output / 'notebook.json'
    ui_output = Path(ui_output) if ui_output else output / 'ui'
    if not configuration.exists():
        return {'status': 'not_run', 'reason': 'No prepared notebook attempt'}
    spec = load_config(configuration)
    if spec['action'] != 'create':
        return {'status': 'not_run', 'reason': 'Read-only notebook inspection'}
    if not execute:
        return {'status': 'planned', 'attempt': spec['attempt']}
    marker = ui_output / 'notebook-creation-started.json'
    if not marker.exists():
        return {
            'status': 'not_run',
            'reason': 'Notebook creation never acquired ownership'
        }
    if json.loads(marker.read_text()) != {
            'attempt': spec['attempt'],
            'sha256': spec['sha256']
    }:
        raise ValueError('Notebook creation ownership marker differs')
    original = ui_output / 'summary.json'
    if not original.exists():
        original = ui_output / 'browser-result.json'
    evidence = original
    state = json.loads(original.read_text()) if original.exists() else {}
    completed = state.get('notebook_cleanup', {})
    if completed.get('status') != 'passed':
        spec['action'] = 'cleanup'
        retry = ui_output / 'cleanup-notebook.json'
        retry.write_text(json.dumps(spec) + '\n')
        # A killed browser cannot run its finally block. Authenticate again
        # and stop/delete only the checksum-bound attempt with durable ownership.
        from uuid import uuid4
        retry_root = ui_output.parent / ('cleanup-ui-' + uuid4().hex[:12])
        result = ui.run(output=retry_root, execute=True,
                        credential_file=credential_file, notebook=retry,
                        key_file=key_file)
        if (result.get('status') != 'passed'
                or result.get('notebook_cleanup', {}).get('status') != 'passed'
                or result.get('browser_cleanup', {}).get('status') != 'passed'):
            raise RuntimeError('Notebook cleanup unconfirmed; preserving stage')
        evidence = retry_root / 'summary.json'
    prefix = launch.STAGE + '/mlops-validation/graph-notebook/' + spec[
        'attempt'] + '/'
    result = {'status': 'passed', 'evidence': str(evidence)}
    if remove_stage:
        if (state.get('exports', {}).get('status') != 'passed'
                or 'collection_error' in state):
            result['stage_cleanup'] = {'status': 'blocked'}
        else:
            with launch.connect(
                    key_file) as connection, connection.cursor() as cursor:
                cursor.execute('REMOVE ' + prefix)
            result['stage_cleanup'] = {'status': 'passed'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--ui-credentials', type=Path)
    args = parser.parse_args()
    result = cleanup(args.output_dir, execute=args.execute,
                     key_file=args.key_file,
                     credential_file=args.ui_credentials)
    print('Graph cleanup:', result['status'])
    if result.get('stage_cleanup', {}).get('status') == 'blocked':
        print('Stage evidence preserved: export collection was not confirmed')


if __name__ == '__main__':
    main()
