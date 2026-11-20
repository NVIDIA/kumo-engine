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
TEST_BRANCH = 'egu/byom-mlops-lifecycle'
BRANCH_TEST_ENVIRONMENTS = {'snowflake-mlops-test', 'sagemaker-mlops-test'}
TEAM_ID = 19956283


def validate(environment, policies, allowed_branches=(MAIN_BRANCH, ),
             required_branch=MAIN_BRANCH):
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
    if rules[0].get('prevent_self_review') is not True:
        raise ValueError('Environment must prevent deployment self-review')
    if environment.get('deployment_branch_policy') != dict(
            protected_branches=False, custom_branch_policies=True):
        raise ValueError('Environment must use selected deployment branches')
    branches = policies.get('branch_policies', [])
    names = [entry.get('name') for entry in branches]
    if (policies.get('total_count', len(branches)) != len(branches)
            or len(set(names)) != len(names)
            or any(entry.get('type') != 'branch' for entry in branches)
            or not set(names).issubset(allowed_branches)
            or MAIN_BRANCH not in names or required_branch not in names):
        raise ValueError(
            'Environment must allow main and only explicitly approved '
            'test branches; wildcards and tags are forbidden')


def main(environment=ENVIRONMENT, branch=None):
    """Require an explicitly approved ref and independent maintainer approval."""
    allowed = ((MAIN_BRANCH,
                TEST_BRANCH) if environment in BRANCH_TEST_ENVIRONMENTS else
               (MAIN_BRANCH, ))
    ref = os.environ.get('GITHUB_REF', '')
    if (os.environ.get('GITHUB_REPOSITORY') != REPOSITORY
            or ref not in {'refs/heads/' + name
                           for name in allowed}
            or os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'):
        raise SystemExit(
            'Notebook checks require manual dispatch on main or an approved test branch'
        )
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
                'A repository admin must select approved branches, require independent '
                'maintainer-team approval, disable administrator bypass, '
                'and add the environment secrets.') from None

    validate(get(), get('/deployment-branch-policies?per_page=100'),
             allowed_branches=allowed,
             required_branch=ref.removeprefix('refs/heads/'))
    print(
        'Confirmed: maintainer approval, no administrator bypass, and selected branch restriction'
    )


if __name__ == '__main__':
    main()
