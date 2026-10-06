# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

import pytest

# Installed-wheel CI invokes the pytest executable with the repository omitted
# from sys.path. The release planner is intentionally repository tooling, not
# part of the public kumo-api wheel, so load it explicitly from the checkout.
sys.path.insert(0, str(Path(__file__).parents[1]))
from tools.release_plan import plan_release  # noqa: E402

RELEASE_BRANCHES = {'v0.91', 'v0.92'}
RELEASE_TAGS = {'v0.91.0', 'v0.92.0'}
RELEASE_WORKFLOW = (Path(__file__).parents[3] / '.github' / 'workflows' /
                    'kumo-api-release.yml')
HOTFIX_WORKFLOW = (Path(__file__).parents[3] / '.github' / 'workflows' /
                   'kumo-api-hotfix-release.yml')


def test_plans_prebumped_major_release_without_version_commit():
    plan = plan_release(current_version='1.0.0', requested_branch='v1.0',
                        existing_branches=RELEASE_BRANCHES,
                        existing_tags=RELEASE_TAGS, dry_run=False)

    assert plan.target_branch == 'v1.0'
    assert plan.target_tag == 'v1.0.0'
    assert plan.previous_tag == 'v0.92.0'
    assert not plan.needs_version_update
    assert plan.mutate_remote


def test_plans_approved_prebumped_v3_release_from_v0_92():
    plan = plan_release(current_version='3.0.0', requested_branch='v3.0',
                        existing_branches=RELEASE_BRANCHES,
                        existing_tags=RELEASE_TAGS | {'v3.0.0b1'},
                        dry_run=True)

    assert plan.target_branch == 'v3.0'
    assert plan.target_tag == 'v3.0.0'
    assert plan.previous_tag == 'v0.92.0'
    assert not plan.needs_version_update
    assert not plan.mutate_remote


@pytest.mark.parametrize('existing_tags', [
    {'v0.91.0'},
    RELEASE_TAGS | {'v0.93.0'},
    RELEASE_TAGS | {'v2.99.7'},
])
def test_rejects_v3_jump_without_exact_approved_v0_92_baseline(existing_tags):
    with pytest.raises(ValueError, match='not an approved predecessor'):
        plan_release(current_version='3.0.0', requested_branch='v3.0',
                     existing_branches=RELEASE_BRANCHES,
                     existing_tags=existing_tags, dry_run=True)


def test_plans_normal_major_release_with_version_commit():
    plan = plan_release(current_version='0.92.0', requested_branch='v1.0',
                        existing_branches=RELEASE_BRANCHES,
                        existing_tags=RELEASE_TAGS, dry_run=False)

    assert plan.previous_tag == 'v0.92.0'
    assert plan.needs_version_update


def test_plans_normal_minor_release_with_version_commit():
    plan = plan_release(current_version='0.92.0', requested_branch='v0.93',
                        existing_branches=RELEASE_BRANCHES,
                        existing_tags=RELEASE_TAGS, dry_run=False)

    assert plan.target_tag == 'v0.93.0'
    assert plan.previous_tag == 'v0.92.0'
    assert plan.needs_version_update


def test_plans_normal_minor_release_after_patch_release():
    plan = plan_release(current_version='0.92.3', requested_branch='v0.93',
                        existing_branches=RELEASE_BRANCHES,
                        existing_tags=RELEASE_TAGS | {'v0.92.3'},
                        dry_run=False)

    assert plan.target_tag == 'v0.93.0'
    assert plan.previous_tag == 'v0.92.0'
    assert plan.needs_version_update


def test_rejects_existing_target_branch():
    with pytest.raises(ValueError, match='Release branch v1.0 already exists'):
        plan_release(current_version='1.0.0', requested_branch='v1.0',
                     existing_branches=RELEASE_BRANCHES | {'v1.0'},
                     existing_tags=RELEASE_TAGS, dry_run=True)


def test_rejects_existing_target_tag():
    with pytest.raises(ValueError, match='Release tag v1.0.0 already exists'):
        plan_release(current_version='1.0.0', requested_branch='v1.0',
                     existing_branches=RELEASE_BRANCHES,
                     existing_tags=RELEASE_TAGS | {'v1.0.0'}, dry_run=True)


@pytest.mark.parametrize('current_version,requested_branch', [
    ('1.0', 'v1.0'),
    ('1.0.0', '1.0'),
    ('1.0.0', 'v1.0.0'),
    ('1.0.0', 'v01.0'),
    ('1.0.0', 'v2.0'),
    ('0.92.0', 'v0.94'),
])
def test_rejects_invalid_or_nonsequential_versions(current_version,
                                                   requested_branch):
    with pytest.raises(ValueError):
        plan_release(current_version=current_version,
                     requested_branch=requested_branch,
                     existing_branches=RELEASE_BRANCHES,
                     existing_tags=RELEASE_TAGS, dry_run=True)


def test_dry_run_preserves_plan_but_disables_remote_mutation():
    dry_run = plan_release(current_version='1.0.0', requested_branch='v1.0',
                           existing_branches=RELEASE_BRANCHES,
                           existing_tags=RELEASE_TAGS, dry_run=True)
    release = plan_release(current_version='1.0.0', requested_branch='v1.0',
                           existing_branches=RELEASE_BRANCHES,
                           existing_tags=RELEASE_TAGS, dry_run=False)

    assert dry_run.target_version == release.target_version
    assert dry_run.previous_tag == release.previous_tag
    assert not dry_run.mutate_remote
    assert release.mutate_remote


def test_release_workflow_retains_artifact_before_remote_mutation():
    workflow = RELEASE_WORKFLOW.read_text(encoding='utf-8')

    dry_run_input = workflow.split('      dry_run:', maxsplit=1)[1]
    dry_run_input = dry_run_input.split('\n\nenv:', maxsplit=1)[0]
    assert '        default: true\n' in dry_run_input

    build = workflow.index('python -m build --wheel')
    retain = workflow.index('actions/upload-artifact@')
    push = workflow.index('git push --atomic')
    publish = workflow.index('twine upload --skip-existing')
    github_release = workflow.index('gh release create')
    assert build < retain < push < publish < github_release
    assert 'retention-days: 14' in workflow
    assert 'Live releases require refs/heads/master' in workflow


def test_release_workflow_supports_explicit_beta_without_final_branch():
    workflow = RELEASE_WORKFLOW.read_text(encoding='utf-8')

    assert 'expected_release_version:' in workflow
    assert 'artifact_version=${RELEASE_VERSION_INPUT#v}' in workflow
    assert 'release_tag=$RELEASE_VERSION_INPUT' in workflow
    assert 'TAG: kumo-api-${{ steps.release-plan.outputs.release_tag }}' in workflow
    assert 'git push origin "$TAG"' in workflow
    assert 'Development Status :: 4 - Beta' in workflow
    assert '--prerelease' in workflow


def test_release_workflows_namespace_branches_and_tags():
    release_workflow = RELEASE_WORKFLOW.read_text(encoding='utf-8')
    hotfix_workflow = HOTFIX_WORKFLOW.read_text(encoding='utf-8')

    assert ('RELEASE_BRANCH: kumo-api-${{ '
            'steps.release-plan.outputs.target_branch }}') in release_workflow
    assert ('TAG: kumo-api-${{ steps.release-plan.outputs.release_tag }}'
            in release_workflow)
    assert "git tag --list 'kumo-api-v*'" in release_workflow
    assert "^kumo-api-v[0-9]+\\.[0-9]+$" in hotfix_workflow
    assert 'NEW_TAG=kumo-api-$NEW_PATCH' in hotfix_workflow
