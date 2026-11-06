# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PAT client restricted to the explicitly configured test workspace."""
import json
import os
import re
import urllib.error
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


HOST = 'https://dbc-117b701d-939a.cloud.databricks.com'
VOLUME = '/Volumes/kumo_test_catalogue/kumo_test/test_stage/mlops-validation/notebook-ui'
WORKSPACE_DIR = '/Shared/kumo_test_catalogue/KUMO_TEST/notebook-ui'


def configuration(cluster_id=None):
    if cluster_id is not None and not re.fullmatch(
            r'[0-9]{4}-[0-9]{6}-[a-z0-9]+', cluster_id):
        raise ValueError(
            'cluster_id must identify dedicated test compute or be omitted')
    return dict(host=HOST, volume=VOLUME, workspace_dir=WORKSPACE_DIR,
                cluster_id=cluster_id)


class Client:
    def __init__(self, config):
        self.host = config['host']
        self.token = os.environ.get('KUMO_DATABRICKS_TEST_PAT', '')
        if not re.fullmatch(r'[A-Za-z0-9._-]+', self.token):
            raise ValueError('KUMO_DATABRICKS_TEST_PAT is required')
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
        try:
            with self.opener.open(request, timeout=90) as response:
                value = response.read()
                return value if binary else json.loads(value) if value else {}
        except urllib.error.HTTPError as error:
            # API responses can contain notebook output; retain only the status.
            raise RuntimeError(f'Databricks API HTTP {error.code}') from None
        except urllib.error.URLError:
            raise RuntimeError('Databricks API connection failed') from None
