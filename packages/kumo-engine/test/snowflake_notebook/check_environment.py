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
MAIN_BRANCH = 'main'
SAGEMAKER_BYOM_BRANCH = 'egu/byom-mlops-lifecycle'
TEAM_ID = 19956283


def validate(environment, policies, *, required_branches=(MAIN_BRANCH, )):
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
    if rules[0].get('prevent_self_review') is not True:
        raise ValueError('Environment must prevent deployment self-review')
    if environment.get('deployment_branch_policy') != dict(
            protected_branches=False, custom_branch_policies=True):
        raise ValueError('Environment must use selected deployment branches')
    branches = policies.get('branch_policies', [])
    if (len(branches) != len(required_branches)
            or policies.get('total_count', len(branches)) != len(required_branches)
            or {rule.get('name') for rule in branches} != set(required_branches)
            or any(rule.get('type') != 'branch' for rule in branches)):
        raise ValueError('Environment must allow only ' + ', '.join(required_branches)
                         + '; additional branches, wildcards and tags are forbidden')


def main(environment=ENVIRONMENT, branch=None, *, sagemaker_byom_trial=False):
    """Require main unless dispatching the temporary SageMaker BYOM matrix."""
    ref = os.environ.get('GITHUB_REF', '')
    trial = (sagemaker_byom_trial and environment == 'sagemaker-mlops-test'
             and ref == 'refs/heads/' + SAGEMAKER_BYOM_BRANCH)
    if (os.environ.get('GITHUB_REPOSITORY') != REPOSITORY
            or not (ref == 'refs/heads/' + MAIN_BRANCH or trial)
            or os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'):
        raise SystemExit('Notebook checks require manual dispatch on main')
    base = f'https://api.github.com/repos/{REPOSITORY}/environments/{environment}'

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
                f'Cannot inspect {environment} (HTTP {error.code}). '
                'A repository admin must allow only main, require independent '
                'maintainer-team approval with self-review disabled, '
                'and add the environment secrets.') from None

    required_branches = (MAIN_BRANCH, SAGEMAKER_BYOM_BRANCH) if trial else (MAIN_BRANCH, )
    validate(get(), get('/deployment-branch-policies?per_page=100'),
             required_branches=required_branches)
    print(
        'Confirmed: maintainer approval, self-review disabled, and allowed deployment branches'
    )


if __name__ == '__main__':
    main()
