# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check environment protection before GitHub can create a deployment job."""
import json
import os
import urllib.error
import urllib.request

REPOSITORY = 'NVIDIA/kumo-engine'
ENVIRONMENT = 'snowflake-mlops-test'
BRANCH = 'egu/mlops-platform-integration'


def validate(environment, policies):
    if environment.get('deployment_branch_policy') != dict(
            protected_branches=False, custom_branch_policies=True):
        raise ValueError('Environment must use selected deployment branches')
    branches = policies.get('branch_policies', [])
    if len(branches) != 1 or branches[0].get(
            'name') != BRANCH or branches[0].get('type') != 'branch':
        raise ValueError('Environment must allow only branch ' + BRANCH)


def main():
    if os.environ['GITHUB_REPOSITORY'] != REPOSITORY or os.environ[
            'GITHUB_REF'] != 'refs/heads/' + BRANCH:
        raise SystemExit('This trial only runs on its feature branch')
    base = f'https://api.github.com/repos/{REPOSITORY}/environments/{ENVIRONMENT}'

    def get(suffix=''):
        request = urllib.request.Request(
            base + suffix, headers={
                'Authorization': 'Bearer ' + os.environ['GITHUB_TOKEN'],
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28',
            })
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise SystemExit(
                f'Cannot inspect {ENVIRONMENT} (HTTP {error.code}). '
                'A repository admin must create it, restrict it to ' + BRANCH +
                ', and add the two RSA environment secrets.') from None

    validate(get(), get('/deployment-branch-policies?per_page=100'))
    print(
        'Confirmed: existing environment permits only the trial feature branch')


if __name__ == '__main__':
    main()
