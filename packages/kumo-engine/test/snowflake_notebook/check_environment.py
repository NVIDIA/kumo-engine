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
TEAM_ID = 19956283


def validate(environment, policies):
    if environment.get('can_admins_bypass') is not False:
        raise ValueError('Administrator bypass must be disabled')
    rules = [
        rule for rule in environment.get('protection_rules', [])
        if rule.get('type') == 'required_reviewers'
    ]
    if len(rules) != 1 or [
        (item.get('type'), item.get('reviewer', {}).get('id'))
            for item in rules[0].get('reviewers', [])
    ] != [('Team', TEAM_ID)]:
        raise ValueError(
            'Environment must require NVIDIA/kumo-engineers approval')
    if environment.get('deployment_branch_policy') != dict(
            protected_branches=False, custom_branch_policies=True):
        raise ValueError('Environment must use selected deployment branches')
    branches = policies.get('branch_policies', [])
    if len(branches) != 1 or branches[0].get(
            'name') != BRANCH or branches[0].get('type') != 'branch':
        raise ValueError('Environment must allow only branch ' + BRANCH)


def main():
    if os.environ['GITHUB_REPOSITORY'] != REPOSITORY or os.environ[
            'GITHUB_REF'] != 'refs/heads/' + BRANCH or os.environ.get(
                'GITHUB_EVENT_NAME') != 'workflow_dispatch':
        raise SystemExit(
            'This trial requires manual dispatch on its feature branch')
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
                ', require the maintainer team reviewer, disable administrator bypass, and add the environment secrets.'
            ) from None

    validate(get(), get('/deployment-branch-policies?per_page=100'))
    print(
        'Confirmed: maintainer approval, no administrator bypass, and trial branch restriction'
    )


if __name__ == '__main__':
    main()
