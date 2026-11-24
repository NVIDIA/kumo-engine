# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep one-use MLflow authentication URLs in memory and outside retained evidence."""
import json
import subprocess
from pathlib import Path

from expected_ui import expectations


def verify(client, spec, directory, *, native=False):
    request = dict(directory=str(Path(directory).resolve()),
                   graph=spec['scenario'] in ('graph', 'all'))
    if native and spec['scenario'] in ('engine', 'all'):
        request['expected_mlflow'] = expectations(Path(directory) / 'outputs')
        response = client.sm.create_presigned_mlflow_tracking_server_url(
            TrackingServerName=spec['tracking_server'], ExpiresInSeconds=300,
            SessionExpirationDurationInSeconds=1800)
        request['mlflow_url'] = response['AuthorizedUrl']
    result = subprocess.run(
        ['node', str(Path(__file__).with_suffix('.cjs'))],
        input=json.dumps(request), text=True, capture_output=True, timeout=600)
    path = Path(directory) / 'browser/result.json'
    if not path.is_file():
        raise RuntimeError(
            'Notebook browser verification did not produce a receipt')
    report = json.loads(path.read_text())
    failed = (result.returncode or report.get('status') != 'passed'
              or (request['graph'] and
                  (not isinstance(report.get('notebook_export'), dict)
                   or report['notebook_export'].get('status') != 'passed'))
              or ('mlflow_url' in request and
                  (not isinstance(report.get('native_mlflow'), dict)
                   or report['native_mlflow'].get('status') != 'passed')))
    if native:
        from studio import verify_studio
        try:
            report['native_studio'] = verify_studio(client, spec, directory)
        except Exception:
            report['status'] = 'failed'
            report['native_studio'] = dict(status='failed')
            raise
        finally:
            path.write_text(json.dumps(report, indent=2) + '\n')
    if failed:
        raise RuntimeError(
            'Notebook browser verification failed; see browser/result.json')
    return report
