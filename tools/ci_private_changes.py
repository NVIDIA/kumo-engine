# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Choose private jobs from the approved PR's complete changed-file list."""

import fnmatch
import json
import os


def matches(files, patterns):
    return any(
        fnmatch.fnmatchcase(path, pattern) for path in files
        for pattern in patterns)


def selection(files, force=False):
    code = force or any(
        not matches([path], ('ui/**', 'docs/**', 'helm/**', 'tools/**',
                             'benchmark/**', '*.md')) or path in
        ('tools/run_private_tests.py', 'tools/ci_private_changes.py')
        for path in files)
    shared = ('.github/actions/setup/**',
              '.github/workflows/testing_private.yml', 'test/ci_resources.py',
              'kumo/testing/**', 'setup.py', 'uv.lock',
              'tools/run_private_tests.py', 'tools/ci_private_changes.py')
    gcp = force or matches(
        files, (*shared, 'kumo/connector/**', 'io_utils/**', 'rest/**',
                'packages/kumo-core/**', 'test/rest/integ/test_gcs*',
                'test/rest/integ/test_bigquery*',
                'test/rest/integ/test_cases/test_bigquery_e2e_fixtures.json'))
    snowflake = force or matches(
        files, (*shared, 'kumo/connector/**',
                'rest/backends/data_backend/activities/**',
                'test/rest/integ/test_snowflake*',
                'test/rest/integ/test_cases/test_snowflake_e2e_fixtures.json',
                'rest/devserver/configs/diskgraph-test.json'))
    groups = [
        'aws', 'aws-public-api', 'gcp', 'azure', 'snowflake', 'databricks'
    ] if code else []
    if code and gcp:
        groups.append('gcp-e2e')
    # A sentinel avoids GitHub's invalid empty matrix; the test job skips it.
    matrix = {
        'include': [{
            'group': group
        } for group in groups] or [{
            'group': 'none'
        }]
    }
    return matrix, gcp, snowflake


if __name__ == '__main__':
    matrix, gcp, snowflake = selection(
        json.loads(os.environ['CHANGED_FILES']),
        os.environ.get('FORCE_PRIVATE_TESTS') == 'true')
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'matrix={json.dumps(matrix)}\n')
        output.write(
            f'run={str(matrix["include"][0]["group"] != "none").lower()}\n')
        output.write(f'gcp-e2e={str(gcp).lower()}\n')
        output.write(f'snowflake-e2e={str(snowflake).lower()}\n')
