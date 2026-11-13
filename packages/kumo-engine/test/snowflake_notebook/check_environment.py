# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check environment protection before GitHub can create a deployment job."""
import json
import os
import re
import urllib.error
import urllib.request

REPOSITORY = 'NVIDIA/kumo-engine'
ENVIRONMENT = 'snowflake-mlops-test'
MAIN_BRANCH = 'main'
TEAM_ID = 19956283


def validate(environment, policies, branch=None, selected_branch=MAIN_BRANCH):
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
    # TODO(before merge): enable prevent_self_review in all platform environments.
    # Allow maintainer self-approval only during explicitly allowlisted branch trials.
    if selected_branch == MAIN_BRANCH and rules[0].get(
            'prevent_self_review') is not True:
        raise ValueError('Environment must prevent deployment self-review')
    if environment.get('deployment_branch_policy') != dict(
            protected_branches=False, custom_branch_policies=True):
        raise ValueError('Environment must use selected deployment branches')
    branches = policies.get('branch_policies', [])
    names = [item.get('name') for item in branches]
    allowed = {MAIN_BRANCH, branch} if branch is not None else set(names)
    if (not 1 <= len(branches) <= 2
            or any(not isinstance(name, str)
                   or not re.fullmatch(r'[A-Za-z0-9_./-]+', name)
                   for name in names)
            or any(item.get('type') != 'branch'
                   for item in branches) or len(set(names)) != len(names)
            or policies.get('total_count', len(branches)) != len(branches)
            or not set(names) <= allowed or MAIN_BRANCH not in names
            or selected_branch not in names):
        raise ValueError(
            'Environment must allow main and optionally one explicitly selected trial '
            'branch, including the selected branch; wildcards and tags are forbidden'
        )


def main(environment=ENVIRONMENT, branch=None):
    ref = os.environ.get('GITHUB_REF', '')
    allowed_refs = ({'refs/heads/' + MAIN_BRANCH, 'refs/heads/' +
                     branch} if branch is not None else None)
    if (os.environ.get('GITHUB_REPOSITORY') != REPOSITORY
            or not ref.startswith('refs/heads/')
            or (allowed_refs is not None and ref not in allowed_refs)
            or os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'):
        raise SystemExit(
            'Notebook checks require manual dispatch on main or a selected trial branch'
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
                'A repository admin must create it, allow main and optionally '
                + (branch or 'one explicit trial branch') +
                ', require the maintainer team reviewer, disable administrator bypass, and add the environment secrets.'
            ) from None

    validate(get(), get('/deployment-branch-policies?per_page=100'), branch,
             selected_branch=ref.removeprefix('refs/heads/'))
    print(
        'Confirmed: maintainer approval, no administrator bypass, and selected branch restriction'
    )


if __name__ == '__main__':
    main()
