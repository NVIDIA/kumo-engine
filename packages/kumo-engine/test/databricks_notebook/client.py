# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PAT client restricted to the explicitly configured test workspace."""
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


VOLUME = '/Volumes/kumo_test_catalogue/kumo_test/test_stage/mlops-validation/notebook-ui'
WORKSPACE_DIR = '/Shared/kumo_test_catalogue/KUMO_TEST/notebook-ui'
DENIED_VOLUME = '/Volumes/kumo_test_denied/kumo_test/test_denied'


def configuration(cluster_id=None):
    if cluster_id is not None and not re.fullmatch(
            r'[0-9]{4}-[0-9]{6}-[a-z0-9]+', cluster_id):
        raise ValueError(
            'cluster_id must identify dedicated test compute or be omitted')
    return dict(host=os.environ.get('KUMO_DATABRICKS_TEST_HOST', ''),
                volume=VOLUME, workspace_dir=WORKSPACE_DIR,
                expected_user=os.environ.get('KUMO_DATABRICKS_TEST_USER', ''),
                cluster_id=cluster_id, denied_volume=DENIED_VOLUME)


class Client:
    def __init__(self, config):
        self.host = config['host']
        self.token = os.environ.get('KUMO_DATABRICKS_TEST_PAT', '')
        if not re.fullmatch(r'[A-Za-z0-9._-]+', self.token):
            raise ValueError('KUMO_DATABRICKS_TEST_PAT is required')
        if (not re.fullmatch(r'https://[a-z0-9-]+\.cloud\.databricks\.com',
                             self.host)
                or self.host != os.environ.get('KUMO_DATABRICKS_TEST_HOST')):
            raise ValueError(
                'KUMO_DATABRICKS_TEST_HOST must name the test workspace')
        self.opener = urllib.request.build_opener(NoRedirect())

    def api(self, path, method='GET', body=None, raw=None, binary=False):
        if not path.startswith('/api/') or '#' in path:
            raise ValueError('Expected a workspace API path')
        data = raw if raw is not None else (
            json.dumps(body).encode() if body is not None else None)
        request = urllib.request.Request(
            self.host + path, data=data, method=method, headers={
                'Authorization':
                'Bearer ' + self.token,
                'Content-Type': ('application/octet-stream'
                                 if raw is not None else 'application/json'),
            })
        retryable = method in ('GET', 'DELETE', 'PUT') or path in (
            '/api/2.2/jobs/runs/cancel', '/api/2.0/mlflow/experiments/delete')
        for attempt in range(4):
            try:
                with self.opener.open(request, timeout=90) as response:
                    value = response.read()
                    return value if binary else json.loads(
                        value) if value else {}
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                if method == 'DELETE' and status == 404:
                    return {}
                if not retryable or status not in (429, 500, 502, 503,
                                                   504) or attempt == 3:
                    raise APIError(status) from None
            except (OSError, http.client.HTTPException):
                if not retryable or attempt == 3:
                    raise RuntimeError(
                        'Databricks API connection failed') from None
            time.sleep(2**attempt)


class APIError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f'Databricks API HTTP {status}')


def admit_cluster(client, cluster_id):
    """Validate existing test compute; Jobs handles any required auto-start."""
    if not cluster_id:
        raise ValueError('A dedicated KUMO_TEST_ x86 cluster_id is required')
    configuration(cluster_id)
    cluster = client.api('/api/2.0/clusters/get?cluster_id=' + cluster_id)
    driver = cluster.get('driver_node_type_id')
    if driver is None:
        driver = cluster.get('node_type_id')
    if (cluster.get('cluster_id') != cluster_id
            or not cluster.get('cluster_name', '').startswith('KUMO_TEST_')
            or cluster.get('node_type_id') not in ('i3.xlarge', 'i3.2xlarge')
            or driver not in ('i3.xlarge', 'i3.2xlarge')
            or cluster.get('num_workers') != 0):
        raise ValueError('Use a dedicated single-node x86 KUMO_TEST_ cluster')
    if cluster.get('state') not in ('RUNNING', 'PENDING', 'RESTARTING', 'TERMINATED'):
        raise ValueError('Dedicated test cluster is not available for notebook Jobs')
    return cluster
