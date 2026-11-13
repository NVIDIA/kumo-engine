# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the generated CloudShell migration with a stateful AWS CLI stub."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/test/sagemaker_notebook'
DOMAIN = 'd-test123'
ACCOUNT = '926922431314'
SUITE = 'kumo-engine-sagemaker-validation'


@pytest.fixture
def migration(tmp_path, monkeypatch):
    if not shutil.which('jq'):
        pytest.skip('Standalone migration requires jq')
    monkeypatch.syspath_prepend(str(HERE))
    # Other test modules may have loaded a config module from another suite.
    old = {name: sys.modules.pop(name, None) for name in ('config', 'policies')}
    spec = importlib.util.spec_from_file_location('migration_generator',
                                                  HERE / 'prepare_access.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    config = generator.plan()
    config['studio'] = generator.studio_plan()
    legacy = json.loads((HERE / 'legacy_permissions.json').read_text())

    def resolved(value):
        return json.loads(
            json.dumps(value).replace('__KUMO_TEST_ACCOUNT__', ACCOUNT).replace(
                '__KUMO_TEST_DOMAIN_ID__', DOMAIN))

    authored = json.loads(json.dumps(config))
    config, legacy = resolved(config), resolved(legacy)
    with monkeypatch.context() as previous:
        previous.setattr(sys.modules['policies'], 'SPACE_NAME',
                         config['studio']['profile']['UserProfileName'])
        previous_studio = resolved(generator.studio_plan())
    shared_trust = dict(
        Version='2012-10-17', Statement=[
            dict(Effect='Allow',
                 Principal=dict(AWS='arn:aws:iam::926922431314:root'),
                 Action='sts:AssumeRole'), legacy['runner_trust']
        ])
    state = dict(
        roles={
            'gha-kumo-workflow-tests':
            dict(
                trust=shared_trust, policies={
                    'kumo-engine-sagemaker-notebook': legacy['runner_policy'],
                    'kumo-engine-sagemaker-ui-read': legacy['ui_policy'],
                    'unrelated': dict(Version='2012-10-17', Statement=[])
                }),
            'kumo-mlops-validation-sagemaker':
            dict(
                trust=dict(Version='2012-10-17', Statement=[]), policies={
                    'kumo-engine-sagemaker-notebook':
                    legacy['execution_policy'],
                    'pipeline-steps':
                    legacy['steps_policy'],
                    'kumo-engine-sagemaker-studio':
                    legacy['execution_studio_policy']
                })
        }, attached=[
            'arn:aws:iam::926922431314:policy/kumo-engine-sagemaker-studio',
            'arn:aws:iam::aws:policy/ReadOnlyAccess'
        ], managed=legacy['studio_policy'], managed_owned=True, profiles={},
        spaces={}, log=[], mutations=[])
    state_path = tmp_path / 'state.json'
    state_path.write_text(json.dumps(state))
    script = generator.standalone(tmp_path, authored)
    (tmp_path / 'aws').write_text('#!' + sys.executable + '\n' + STUB)
    (tmp_path / 'aws').chmod(0o700)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
               AWS_STUB_STATE=str(state_path))

    def run(change=None, previous=False):
        if previous:
            current = json.loads(state_path.read_text())
            for key, desired in config['roles'].items():
                permission = json.loads(json.dumps(desired['permissions']))
                permission['Statement'] += previous_studio.get(
                    key.lower() + '_permissions', {}).get('Statement', [])
                current['roles'][desired['arn'].rsplit('/', 1)[1]] = dict(
                    trust=desired['trust'],
                    duration=10800 if key == 'RUNNER' else 3600,
                    tags=[dict(Key='KumoTestSuite', Value=SUITE)],
                    policies={'kumo-engine-sagemaker-validation': permission})
            profile = json.loads(json.dumps(config['studio']['profile']))
            name = profile['UserProfileName']
            profile.update(UserProfileArn=(
                f'arn:aws:sagemaker:us-west-2:{ACCOUNT}:user-profile/{DOMAIN}/{name}'),
                           Status='InService')
            current['profiles'][name] = profile
            state_path.write_text(json.dumps(current))
        if change:
            current = json.loads(state_path.read_text())
            change(current)
            state_path.write_text(json.dumps(current))
        result = subprocess.run(['sh', str(script)], env=env,
                                capture_output=True, text=True, timeout=40)
        return result, json.loads(state_path.read_text())

    yield run, config, script
    for name in ('config', 'policies'):
        sys.modules.pop(name, None)
        if old[name] is not None:
            sys.modules[name] = old[name]


STUB = r'''
import json, os, sys
from pathlib import Path
state_file=Path(os.environ['AWS_STUB_STATE']); state=json.loads(state_file.read_text())
a=sys.argv[1:]
if a[:1]==['--region']:
 assert a[1]=='us-west-2'; a=a[2:]
service,action=a[:2]
def arg(name,default=None):
 return a[a.index(name)+1] if name in a else default
def persist(): state_file.write_text(json.dumps(state))
def fail(code):
 persist(); print('An error occurred ('+code+') when calling stub',file=sys.stderr); sys.exit(1)
def out(value): persist(); print(json.dumps(value)); sys.exit(0)
def document(name): return json.loads(Path(arg(name).removeprefix('file://')).read_text())
state['log'].append(a)
mutating=action.startswith(('create-','put-','update-','delete-','detach-','attach-','tag-'))
if mutating: state['mutations'].append(a)
if state.get('deny')==action: fail('AccessDenied')
account='926922431314'; domain='d-test123'; base='arn:aws:sagemaker:us-west-2:'+account
role=arg('--role-name'); policy=arg('--policy-name')
if service=='sts' and action=='get-caller-identity':
 persist(); print(state.get('account',account)); sys.exit(0)
if service=='iam':
 if action=='get-open-id-connect-provider':
  if state.get('provider_missing'): fail('NoSuchEntity')
  out(dict(Url=state.get('provider_url','token.actions.githubusercontent.com'),ClientIDList=state.get('audiences',['sts.amazonaws.com'])))
 if action=='get-role':
  if role not in state['roles']: fail('NoSuchEntity')
  r=state['roles'][role]
  out(dict(Role=dict(Arn='arn:aws:iam::'+account+':role/'+role,Path='/',AssumeRolePolicyDocument=r['trust'],MaxSessionDuration=r.get('duration',3600),Tags=r.get('tags',[]),PermissionsBoundary=r.get('boundary'))))
 if action=='get-role-policy':
  if role not in state['roles'] or policy not in state['roles'][role]['policies']: fail('NoSuchEntity')
  value=json.loads(json.dumps(state['roles'][role]['policies'][policy]))
  if role=='kumo-engine-sagemaker-test-runner':
   reads=state.setdefault('runner_policy_reads',0)+1; state['runner_policy_reads']=reads
   if reads==state.get('drift_after_policy_read'):
    state['roles'][role]['policies'][policy]['Statement'].append(dict(Effect='Deny',Action='*',Resource='*'))
  out(value)
 if action=='list-attached-role-policies':
  out(dict(AttachedPolicies=[dict(PolicyArn=p) for p in (state['attached'] if role=='gha-kumo-workflow-tests' else state['roles'][role].get('attached',[]))]))
 if action=='list-role-policies': out(dict(PolicyNames=list(state['roles'][role]['policies'])))
 if action=='get-policy': out(dict(Policy=dict(Arn=arg('--policy-arn'),DefaultVersionId='v1')))
 if action=='get-policy-version': out(state['managed'])
 if action=='list-policy-tags': out(dict(Tags=[dict(Key='KumoTestSuite',Value='kumo-engine-sagemaker-studio')] if state['managed_owned'] else []))
 if action=='create-role':
  if role in state['roles']: fail('EntityAlreadyExists')
  state['roles'][role]=dict(trust=document('--assume-role-policy-document'),duration=int(arg('--max-session-duration')),tags=[dict(Key='KumoTestSuite',Value='kumo-engine-sagemaker-validation')],policies={}); out({})
 if action=='put-role-policy': state['roles'][role]['policies'][policy]=document('--policy-document'); out({})
 if action=='delete-role-policy': del state['roles'][role]['policies'][policy]; out({})
 if action=='detach-role-policy': state['attached'].remove(arg('--policy-arn')); out({})
 if action=='update-assume-role-policy': state['roles'][role]['trust']=document('--policy-document'); out({})
if service=='sagemaker':
 if action=='list-domains': out(dict(Domains=[dict(DomainId=domain,DomainName='kumo-test-sagemaker')]))
 if action=='describe-domain': out(dict(DomainId=domain,DomainArn=base+':domain/'+domain,DomainName='kumo-test-sagemaker',Status='InService',AuthMode='IAM',DefaultUserSettings=dict(ExecutionRole='legacy-default-untouched')))
 if action=='list-tags': out(dict(Tags=[dict(Key='KumoTestSuite',Value='kumo-engine-sagemaker-studio')] if state.get('studio_owned',True) else []))
 if action in ('describe-user-profile','describe-space'):
  kind='profiles' if action=='describe-user-profile' else 'spaces'
  name=arg('--user-profile-name') if kind=='profiles' else arg('--space-name')
  if name not in state[kind]: fail('ResourceNotFound')
  out(state[kind][name])
 if action in ('create-user-profile','create-space'):
  d=document('--cli-input-json'); kind='profiles' if action=='create-user-profile' else 'spaces'
  name=d['UserProfileName'] if kind=='profiles' else d['SpaceName']
  if name in state['spaces' if kind=='profiles' else 'profiles']: fail('ResourceInUse')
  arnkey='UserProfileArn' if kind=='profiles' else 'SpaceArn'; arntype='user-profile' if kind=='profiles' else 'space'
  d.update({arnkey:base+':'+arntype+'/'+domain+'/'+name,'Status':'InService'}); state[kind][name]=d
  if kind=='spaces' and state.get('late_drift'):
   state['roles']['gha-kumo-workflow-tests']['policies']['kumo-engine-sagemaker-notebook']['Statement']=[]
  out({})
fail('UnexpectedOperation:'+service+':'+action)
'''


def test_fresh_migration_and_identical_rerun_preserve_unrelated_state(
        migration):
    run, config, script = migration
    result, state = run()
    assert result.returncode == 0, result.stderr
    roles = state['roles']
    assert roles['gha-kumo-workflow-tests']['policies'] == {
        'unrelated': {
            'Version': '2012-10-17',
            'Statement': []
        }
    }
    assert len(roles['gha-kumo-workflow-tests']['trust']['Statement']) == 1
    assert roles['kumo-mlops-validation-sagemaker']['policies'] == {}
    assert state['attached'] == ['arn:aws:iam::aws:policy/ReadOnlyAccess']
    for key, desired in config['roles'].items():
        role = roles[desired['arn'].rsplit('/', 1)[1]]
        assert role['trust'] == desired['trust']
        expected = desired['permissions']['Statement'] + (config['studio'].get(
            key.lower() + '_permissions', {}).get('Statement', []))
        assert role['policies']['kumo-engine-sagemaker-validation'][
            'Statement'] == expected
    assert roles['kumo-engine-sagemaker-test-runner']['duration'] == 10800
    assert set(state['profiles']) == {'kumo-test-validation'}
    assert set(state['spaces']) == {'kumo-test-validation-space'}
    assert all(a[1] not in ('create-domain', 'update-domain', 'delete-policy',
                            'delete-role', 'create-app')
               for a in state['mutations'])
    assert any(a[:2] == ['sagemaker', 'describe-domain'] for a in state['log'])
    count = len(state['mutations'])
    result, again = run()
    assert result.returncode == 0, result.stderr
    assert len(again['mutations']) == count
    assert script.name == 'update-sagemaker-test-permissions.sh'
    assert (script.stat().st_mode & 0o777) == 0o700


@pytest.mark.parametrize('case', [
    'runner-policy', 'execution-policy', 'steps-policy', 'managed-policy', 'managed-tags',
    'trust', 'denied', 'account', 'new-role', 'provider-missing',
    'provider-url', 'provider-audience'
])
def test_all_state_preflight_rejects_drift_before_any_mutation(migration, case):
    run, config, _ = migration

    def change(state):
        if case == 'runner-policy':
            state['roles']['gha-kumo-workflow-tests']['policies'][
                'kumo-engine-sagemaker-notebook']['Statement'].append({
                    'Effect':
                    'Deny',
                    'Action':
                    '*',
                    'Resource':
                    '*'
                })
        if case == 'execution-policy':
            state['roles']['kumo-mlops-validation-sagemaker']['policies'][
                'kumo-engine-sagemaker-notebook']['Statement'] = []
        if case == 'steps-policy':
            state['roles']['kumo-mlops-validation-sagemaker']['policies'][
                'pipeline-steps']['Statement'] = []
        if case == 'managed-policy':
            state['managed']['Statement'] = []
        if case == 'managed-tags':
            state['managed_owned'] = False
        if case == 'trust':
            state['roles']['gha-kumo-workflow-tests']['trust']['Statement'][1][
                'Condition']['StringEquals'][
                    'token.actions.githubusercontent.com:aud'] = 'other'
        if case == 'denied':
            state['deny'] = 'get-role-policy'
        if case == 'provider-missing':
            state['provider_missing'] = True
        if case == 'provider-url':
            state['provider_url'] = 'other.example'
        if case == 'provider-audience':
            state['audiences'] = ['other']
        if case == 'account':
            state['account'] = '000000000000'
        if case == 'new-role':
            desired = config['roles']['RUNNER']
            state['roles'][desired['arn'].rsplit('/', 1)[1]] = dict(
                trust=desired['trust'], policies={}, duration=10800, tags=[])

    result, state = run(change)
    assert result.returncode != 0
    assert state['mutations'] == []


def test_invalid_runtime_arguments_stop_before_aws(migration):
    _, _, script = migration
    result = subprocess.run(['sh', str(script), '--studio'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Usage:' in result.stderr


def test_single_unrelated_legacy_trust_statement_is_preserved(migration):
    run, _, _ = migration

    def single(state):
        trust = state['roles']['gha-kumo-workflow-tests']['trust']
        trust['Statement'] = trust['Statement'][0]

    result, state = run(single)
    assert result.returncode == 0, result.stderr
    assert isinstance(
        state['roles']['gha-kumo-workflow-tests']['trust']['Statement'], dict)
    assert not any(a[1] == 'update-assume-role-policy'
                   for a in state['mutations'])


@pytest.mark.parametrize('case', [
    'trust', 'attachment', 'inline', 'policy', 'boundary', 'profile', 'space',
    'role-suffix', 'studio-owner'
])
def test_existing_owned_resources_reject_drift_without_new_mutations(
        migration, case):
    run, _, _ = migration
    first, state = run()
    assert first.returncode == 0, first.stderr
    count = len(state['mutations'])

    def change(current):
        role = current['roles']['kumo-engine-sagemaker-test-runner']
        if case == 'trust':
            role['trust']['Statement'][0]['Condition'] = {}
        if case == 'attachment':
            role['attached'] = ['arn:aws:iam::aws:policy/AdministratorAccess']
        if case == 'inline':
            role['policies']['extra'] = dict(Statement=[])
        if case == 'policy':
            role['policies']['kumo-engine-sagemaker-validation'][
                'Statement'].append(dict(Effect='Deny', Action='*', Resource='*'))
        if case == 'boundary':
            role['boundary'] = dict(PermissionsBoundaryArn='unexpected-boundary')
        if case == 'profile':
            current['profiles']['kumo-test-validation']['UserSettings'][
                'ExecutionRole'] = 'another-role'
        if case == 'studio-owner':
            current['studio_owned'] = False
        if case == 'role-suffix':
            current['profiles']['kumo-test-validation']['UserSettings'][
                'ExecutionRole'] += '-admin'
        if case == 'space':
            current['spaces']['kumo-test-validation-space']['OwnershipSettings'][
                'OwnerUserProfileName'] = 'another-profile'

    result, state = run(change)
    assert result.returncode != 0
    assert len(state['mutations']) == count


def test_concurrent_legacy_drift_is_rechecked_before_any_revocation(migration):
    run, _, _ = migration
    result, state = run(lambda current: current.update(late_drift=True))
    assert result.returncode != 0
    assert state['spaces']
    assert not any(a[1] in ('delete-role-policy', 'detach-role-policy',
                            'update-assume-role-policy')
                   for a in state['mutations'])


def test_failed_resource_creation_preserves_legacy_grants(migration):
    run, _, _ = migration
    result, state = run(
        lambda current: current.update(deny='create-user-profile'))
    assert result.returncode != 0
    assert not any(a[1] in ('delete-role-policy', 'detach-role-policy',
                            'update-assume-role-policy')
                   for a in state['mutations'])
    assert 'kumo-engine-sagemaker-notebook' in state['roles'][
        'gha-kumo-workflow-tests']['policies']


def test_resume_original_partial_deployment_and_rerun(migration):
    run, config, _ = migration
    result, state = run(previous=True)
    assert result.returncode == 0, result.stderr
    updates = [a for a in state['mutations'] if a[1] == 'put-role-policy']
    assert {a[a.index('--role-name') + 1] for a in updates} == {
        'kumo-engine-sagemaker-test-runner',
        'kumo-engine-sagemaker-test-execution'
    }
    assert len(updates) == 2
    assert not any(a[1] in ('create-role', 'create-user-profile')
                   for a in state['mutations'])
    assert set(state['spaces']) == {config['studio']['space']['SpaceName']}
    assert state['spaces']['kumo-test-validation-space'][
        'OwnershipSettings']['OwnerUserProfileName'] == 'kumo-test-validation'
    count = len(state['mutations'])
    result, again = run()
    assert result.returncode == 0, result.stderr
    assert len(again['mutations']) == count


@pytest.mark.parametrize('case', ['deny', 'concurrent-drift'])
def test_previous_policy_upgrade_rejects_unexpected_changes(migration, case):
    run, _, _ = migration

    def change(state):
        if case == 'deny':
            state['roles']['kumo-engine-sagemaker-test-runner']['policies'][
                'kumo-engine-sagemaker-validation']['Statement'].append(
                    dict(Effect='Deny', Action='*', Resource='*'))
        else:
            state['drift_after_policy_read'] = 3

    result, state = run(change, previous=True)
    assert result.returncode != 0
    assert state['mutations'] == []


def test_space_failure_after_policy_upgrade_can_resume(migration):
    run, _, _ = migration
    result, state = run(lambda s: s.update(deny='create-space'), previous=True)
    assert result.returncode != 0
    assert 'kumo-engine-sagemaker-notebook' in state['roles'][
        'gha-kumo-workflow-tests']['policies']
    updates = [a for a in state['mutations'] if a[1] == 'put-role-policy']
    assert len(updates) == 2
    result, state = run(lambda s: s.pop('deny'))
    assert result.returncode == 0, result.stderr
    assert len([a for a in state['mutations'] if a[1] == 'put-role-policy']) == 2


def test_exact_previous_dedicated_policies_upgrade_without_widening_drift(
        migration):
    run, config, _ = migration
    first, state = run()
    assert first.returncode == 0, first.stderr
    legacy = json.loads((HERE / 'legacy_permissions.json').read_text())
    previous = json.loads(
        json.dumps(legacy['dedicated_previous_policies']).replace(
            '__KUMO_TEST_DOMAIN_ID__', DOMAIN).replace('__KUMO_TEST_ACCOUNT__',
                                                       ACCOUNT))

    def seed(current):
        for key, role in config['roles'].items():
            current['roles'][role['arn'].rsplit('/', 1)[1]]['policies'][
                'kumo-engine-sagemaker-validation'] = previous[key]

    upgraded, state = run(seed)
    assert upgraded.returncode == 0, upgraded.stderr
    execution = state['roles']['kumo-engine-sagemaker-test-execution'][
        'policies']['kumo-engine-sagemaker-validation']
    assert 'sagemaker-mlflow:*' not in json.dumps(execution)
