// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const {generateKeyPairSync} = require('node:crypto');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const test = require('node:test');
const {runInNewContext} = require('node:vm');
const {plan, start, cleanup, capacity, report, registration} = require('../../.github/scripts/engine-e2e-pool.cjs');
const templates = Object.fromEntries(['g7e.8xlarge', 'g7e.24xlarge'].map(instance =>
  [instance, {template: 'lt-123', version: '2', subnets: ['subnet-1', 'subnet-2']}]));
const requirement = name => ({scratch_gib: 100, instance: name.replace(/-control$/, '') === 'fraud_binary' ? 'g7e.24xlarge' : 'g7e.8xlarge'});
function fixture(tier = 'weekly', runControls = false, scales = [tier === 'weekly' ? '25B' : '100M']) {
  const env = {GITHUB_REPOSITORY: 'NVIDIA/kumo-engine', GITHUB_RUN_ID: '41', GITHUB_RUN_ATTEMPT: '2',
    GITHUB_SHA: 'a'.repeat(40), GITHUB_REF: 'refs/heads/main', ENGINE_E2E_MAX_PARALLEL: '3',
    EXPIRY_ROLE: 'arn:aws:iam::123456789012:role/expiry',
    RUN_CONTROLS: String(runControls), AWS_REGION: 'us-west-2', RESULTS_S3: 's3://results/root', CAPACITY_ROLE: 'arn:aws:iam::123456789012:role/capacity', CAPACITY_FUNCTION: 'arn:aws:lambda:us-west-2:123456789012:function:capacity'};
  const selected = plan(tier, scales, templates, 3, requirement, env), calls = [], claims = new Map();
  const run = {run_attempt: 2, head_sha: env.GITHUB_SHA, head_branch: 'main', head_repository: {full_name: env.GITHUB_REPOSITORY},
    name: 'Kumo Engine ' + tier, path: '.github/workflows/kumo-engine-' + tier + '.yml', event: 'workflow_dispatch',
    run_started_at: '2026-11-04T12:00:00Z', status: 'in_progress'};
  const jobs = selected.matrix.include.map(r => ({name: 'weekly / engine-e2e:' + r.case + ':' + r.scale,
    labels: [r.label], status: 'queued'}));
  const owned = selected.pools.map(p => ({AutoScalingGroupName: p.label,
    LaunchTemplate: {LaunchTemplateId: p.template, Version: '3'},
    Tags: [{Key: 'EngineE2EPool', Value: 'engine-e2e-41-2'}, {Key: 'EngineE2ERepository', Value: env.GITHUB_REPOSITORY}]}));
  const activeGroups = [...owned, {AutoScalingGroupName: 'engine-e2e-41-1-old', Tags: []}];
  const io = {env, now: () => Date.parse('2026-11-05T12:00:00Z') / 1000,
    preconditionFailed: error => error.message === 'existing claim',
    api: async (path, method = 'GET', data) => {
      assert.ok(!path.startsWith('orgs/'), 'Runner APIs must use repository scope');
      calls.push({path, method, data});
      if (path.includes('/attempts/')) return {jobs};
      if (path.includes('/actions/runs/')) return run;
      if (path.endsWith('/registration-token')) {
        assert.equal(path, 'repos/NVIDIA/kumo-engine/actions/runners/registration-token');
        return {token: 'fresh-registration'};
      }
      return {runners: [{id: 1, name: owned[0].AutoScalingGroupName + '-host', labels: [{name: owned[0].AutoScalingGroupName}]},
        {id: 2, name: 'engine-e2e-41-1-old-host', labels: [{name: 'engine-e2e-41-1-old'}]},
        {id: 3, name: owned[0].AutoScalingGroupName + '-other', labels: [{name: 'foreign'}]},
        {id: 4, name: 'foreign', labels: [{name: owned[0].AutoScalingGroupName}]}]};
    }, aws: async (service, operation, input) => {
      calls.push({service, operation, input});
      if (operation === 'create-launch-template-version') return {LaunchTemplateVersion: {VersionNumber: 3}};
      if (operation === 'describe-auto-scaling-groups') return {AutoScalingGroups: activeGroups.filter(g =>
        !input.AutoScalingGroupNames || input.AutoScalingGroupNames.includes(g.AutoScalingGroupName))};
      if (operation === 'delete-auto-scaling-group') activeGroups.splice(activeGroups.findIndex(g => g.AutoScalingGroupName === input.AutoScalingGroupName), 1);
      if (operation === 'suspend-processes') {
        const group = activeGroups.find(g => g.AutoScalingGroupName === input.AutoScalingGroupName);
        group.SuspendedProcesses = [...(group.SuspendedProcesses || []), ...input.ScalingProcesses.map(ProcessName => ({ProcessName}))];
      }
      if (operation === 'get-object') return claims.get(input.Key);
      if (operation === 'put-object') {
        if (claims.has(input.Key)) throw Error('existing claim');
        claims.set(input.Key, JSON.parse(input.Body));
      }
      return {};
    }};
  return {io, selected, calls, owned, run, jobs, claims};
}
async function expired(...args) {
  const f = fixture(...args); await start(f.selected, f.io);
  f.event = JSON.parse(f.calls.find(c => c.operation === 'create-schedule').input.Target.Input);
  f.io.now = () => f.event.deadline;
  return f;
}
const claimKey = row => 'root/41/2/' + row.scale + '/' + row.case + '.admission.json';
function poolEvent(f, deadline) {
  const {instance, label} = f.selected.pools[0];
  return {repo: f.io.env.GITHUB_REPOSITORY, run: '41', attempt: '2', sha: f.run.head_sha, deadline, instance, label,
    cases: f.selected.matrix.include.filter(r => r.label === label)};
}
test('cases select fixed types, independent labels and one shared VM budget', () => {
  const {selected, io} = fixture(), rows = selected.matrix.include;
  assert.equal(rows.length, 12);
  assert.equal(new Set(rows.map(r => r.label)).size, 2);
  assert.equal(rows.at(-1).instance, 'g7e.24xlarge');
  assert.equal(selected.pools.reduce((n, p) => n + p.count, 0), 3);
  for (const mapping of [{}, {...templates, 'g7e.24xlarge': {...templates['g7e.24xlarge'], version: '$Latest'}}])
    assert.throws(() => plan('weekly', ['25B'], mapping, 3, requirement, io.env));
  assert.throws(() => plan('weekly', ['25B'], templates, 1, requirement, io.env), /one worker per instance type/);
});
test('controls add separate case jobs only when requested for either tier', () => {
  for (const [tier, scales] of [['nightly', ['100M', '1B']], ['weekly', ['10B', '25B']]]) {
    const baseline = fixture(tier, false, scales), opted = fixture(tier, true, scales);
    delete baseline.io.env.RUN_CONTROLS;
    assert.equal(plan(tier, scales, templates, 3, requirement, baseline.io.env).matrix.include.length, 24);
    assert.equal(baseline.selected.matrix.include.length, 24);
    const rows = opted.selected.matrix.include;
    assert.equal(rows.length, 48);
    assert.equal(new Set(rows.map(r => r.case + '/' + r.scale)).size, 48);
    for (const scale of scales) {
      assert.ok(rows.some(r => r.case === 'fraud_binary' && r.scale === scale && r.instance === 'g7e.24xlarge'));
      assert.ok(rows.some(r => r.case === 'fraud_binary-control' && r.scale === scale && r.instance === 'g7e.24xlarge'));
    }
    assert.equal(opted.selected.pools.reduce((n, p) => n + p.count, 0), 3);
  }
});
test('capacity reports preserve separate claims for all 48 normal and control jobs', async () => {
  const f = await expired('weekly', true, ['10B', '25B']);
  const normal = f.event.cases.find(r => r.case === 'adtech_binary' && r.scale === '10B');
  f.claims.set(claimKey(normal), {state: 'admitted'});
  await capacity(f.event, f.io);
  const dispatched = JSON.parse(f.calls.find(c => c.path?.endsWith('/dispatches')).data.inputs.payload);
  const rows = (await report(dispatched, f.io)).include;
  assert.equal(rows.length, 47);
  assert.ok(!rows.some(r => r.case === 'adtech_binary' && r.scale === '10B'));
  assert.ok(rows.some(r => r.case === 'adtech_binary-control' && r.scale === '10B'));
  f.claims.set(claimKey(normal), {state: 'capacity_failed', instance: normal.instance, deadline: f.event.deadline});
  assert.equal((await report(f.event, f.io)).include.length, 48);
  await assert.rejects(report({...f.event, cases: [f.event.cases[0], f.event.cases[0]]}, f.io), /Invalid report cases/);
  await assert.rejects(report({...f.event, cases: [{...normal, case: 'adtech_binary-control-control'}]}, f.io), /Invalid report case/);
});
test('capacity requests return without waiting and pin the same type across subnets', async () => {
  const f = fixture('weekly', true, ['10B', '25B']); await start(f.selected, f.io);
  const versions = f.calls.filter(c => c.operation === 'create-launch-template-version');
  const groups = f.calls.filter(c => c.operation === 'create-auto-scaling-group');
  assert.equal(groups.length, 2);
  assert.deepEqual(versions.map(c => c.input.LaunchTemplateData.InstanceType), f.selected.pools.map(p => p.instance));
  for (const c of versions) {
    const script = Buffer.from(c.input.LaunchTemplateData.UserData, 'base64');
    assert.ok(script.length <= 16384, 'The largest control-enabled UserData must fit the EC2 limit');
    const rows = JSON.parse(script.toString().match(/^POOL_CASES='(.+)'$/m)[1]);
    assert.ok(rows.length > 0 && rows.every(r => r.instance === c.input.LaunchTemplateData.InstanceType));
  }
  for (const c of groups) {
    assert.equal(c.input.VPCZoneIdentifier, 'subnet-1,subnet-2');
    assert.equal(c.input.AvailabilityZoneDistribution.CapacityDistributionStrategy, 'balanced-best-effort');
    assert.equal(c.input.DesiredCapacity, 0);
    assert.equal(f.calls.find(v => v.operation === 'set-desired-capacity' &&
      v.input.AutoScalingGroupName === c.input.AutoScalingGroupName).input.DesiredCapacity, c.input.MaxSize);
    assert.deepEqual(c.input.InstanceMaintenancePolicy, {MinHealthyPercentage: 0, MaxHealthyPercentage: 100});
  }
});
for (const [name, tier, event, newest, allowed] of [
  ['superseded scheduled nightly', 'nightly', 'schedule', 42, false],
  ['latest scheduled nightly', 'nightly', 'schedule', 41, true],
  ['nightly newer than the listing', 'nightly', 'schedule', 40, true],
  ['manual reference run', 'nightly', 'workflow_dispatch', 42, true],
  ['scheduled weekly', 'weekly', 'schedule', 42, true],
  ['untrusted scheduled nightly', 'nightly', 'schedule', 42, null]
]) test('controller admission: ' + name, async () => {
  const f = fixture(tier), outputs = [], requests = [], writes = [];
  f.run.event = event;
  if (allowed === null) f.run.head_branch = 'other';
  const env = {...f.io.env, TIER: tier, SCALES: JSON.stringify([tier === 'nightly' ? '100M' : '25B']),
    ENGINE_E2E_TEMPLATES_JSON: JSON.stringify(templates), ENGINE_E2E_ENABLED: 'true', ENGINE_E2E_SCHEDULER_ENABLED: 'true', GITHUB_OUTPUT: 'fixture-output'};
  const execution = runInNewContext(readFileSync(join(__dirname, '../../.github/scripts/engine-e2e-pool.cjs'), 'utf8') + '\nmain();', {
    module: {exports: {}}, process: {env, argv: ['node', 'controller', 'start']}, Buffer,
    __dirname: join(__dirname, '../../.github/scripts'),
    require: name => name === 'node:fs' ? {readFileSync, appendFileSync: (path, value) => outputs.push(value)} :
      name === 'node:child_process' ? {execFileSync: (file, args) => {
        if (file === 'python') return JSON.stringify(requirement(args[0].split('/').at(-1).slice(0, -3)));
        if (file === 'gh') {
          requests.push(args[1]);
          if (args[1] === 'repos/NVIDIA/kumo-engine/actions/runs/41') return JSON.stringify(f.run);
          assert.equal(args[1], 'repos/NVIDIA/kumo-engine/actions/workflows/kumo-engine-nightly.yml/runs?branch=main&event=schedule&per_page=1');
          return JSON.stringify({workflow_runs: [{id: newest}]});
        }
        assert.equal(file, 'aws'); writes.push(args);
        return JSON.stringify({LaunchTemplateVersion: {VersionNumber: 3}});
      }} : require(name)
  });
  if (allowed === null) await assert.rejects(execution, /Expected the current main-branch Engine run/);
  else await execution;
  assert.equal(writes.length > 0, allowed === true);
  assert.equal(outputs.length, allowed === true ? 1 : 0);
  if (allowed === true) assert.match(outputs[0], /^plan=.*\nattempt=2\n$/);
  assert.equal(requests.length, tier === 'nightly' && event === 'schedule' && allowed !== null ? 2 : 1);
});
test('expiry is armed before allocation, and failure prevents VM capacity', async () => {
  const f = fixture(), aws = f.io.aws;
  f.io.aws = async (...args) => { if (args[0] === 'scheduler') throw Error('expiry unavailable'); return aws(...args); };
  await assert.rejects(start(f.selected, f.io), /expiry unavailable/);
  assert.equal(f.calls.filter(c => c.operation === 'create-auto-scaling-group').length, 0);
  const g = fixture(); await start(g.selected, g.io);
  const operations = g.calls.filter(c => c.operation);
  assert.equal(operations[0].operation, 'create-schedule');
  assert.equal(operations[0].input.Target.Arn, g.io.env.CAPACITY_FUNCTION);
  const expiry = operations.find(c => c.input.Target?.Arn.endsWith(':deleteAutoScalingGroup'));
  assert.equal(JSON.parse(expiry.input.Target.Input).ForceDelete, true);
  assert.equal(operations[1].input.Target.Arn, 'arn:aws:scheduler:::aws-sdk:autoscaling:suspendProcesses');
});
test('cleanup deletes only owned groups and runner registrations', async () => {
  const f = fixture(); await cleanup(f.io);
  assert.deepEqual(f.calls.filter(c => c.operation === 'delete-auto-scaling-group').map(c => c.input.AutoScalingGroupName),
    f.owned.map(g => g.AutoScalingGroupName));
  assert.deepEqual(f.calls.filter(c => c.method === 'DELETE').map(c => c.path), ['repos/NVIDIA/kumo-engine/actions/runners/1']);
});
test('cleanup waits for every owned group before deleting dependent resources', async () => {
  const f = fixture(), aws = f.io.aws; let polls = 0, sleeps = 0;
  f.io.sleep = async () => { sleeps++; };
  f.io.aws = async (service, operation, input) => {
    assert.notEqual(operation, 'wait group-not-exists', 'Auto Scaling has no CLI waiter');
    if (operation === 'describe-auto-scaling-groups' && input.AutoScalingGroupNames) {
      assert.deepEqual(f.calls.filter(c => c.operation === 'delete-auto-scaling-group').map(c => c.input.AutoScalingGroupName), input.AutoScalingGroupNames);
      return {AutoScalingGroups: polls++ < 2 ? f.owned : []};
    }
    if (operation === 'describe-launch-template-versions') assert.equal(polls, 3);
    return aws(service, operation, input);
  };
  await cleanup(f.io);
  assert.equal(sleeps, 2);
  assert.equal(f.calls.filter(c => c.method === 'DELETE').length, 1);
});
test('cleanup stops within its job window if group deletion stalls', async () => {
  const f = fixture(), aws = f.io.aws; let now = 0;
  f.io.now = () => now;
  f.io.sleep = async ms => { now += ms / 1000; assert.ok(now < 900, 'Cleanup exceeded its job window'); };
  f.io.aws = async (service, operation, input) => {
    assert.notEqual(operation, 'wait group-not-exists', 'Auto Scaling has no CLI waiter');
    if (operation === 'describe-auto-scaling-groups' && input.AutoScalingGroupNames) return {AutoScalingGroups: f.owned};
    return aws(service, operation, input);
  };
  await assert.rejects(cleanup(f.io), /Group deletion timed out/);
  assert.ok(now > 0 && now < 900);
  assert.equal(f.calls.filter(c => c.operation === 'describe-launch-template-versions' || c.method === 'DELETE').length, 0);
});
test('capacity and provider deadlines keep the configured wait and execution windows', async () => {
  for (const [tier, wait, execution] of [['nightly', 4, 3], ['weekly', 24, 6]]) {
    const f = fixture(tier); await start(f.selected, f.io);
    const schedules = f.calls.filter(c => c.operation === 'create-schedule');
    assert.equal(Date.parse(schedules[0].input.StartDate) / 1000 - f.io.now(), wait * 3600);
    assert.equal(Date.parse(schedules[2].input.StartDate) / 1000 - f.io.now(), (wait + execution) * 3600 + 1800);
  }
});
test('capacity reporting does not restart work after an early terminal run', async () => {
  for (const conclusion of ['cancelled', 'failure']) {
    const f = await expired();
    Object.assign(f.run, {status: 'completed', conclusion, updated_at: new Date((f.event.deadline - 1) * 1000).toISOString()});
    await capacity(f.event, f.io);
    assert.equal(f.claims.size, 0);
    assert.equal(f.calls.filter(c => c.path?.includes('/attempts/') || c.path?.endsWith('/dispatches') || c.service === 's3').length, 0);
  }
});
for (const [name, offset, saved] of [
  ['capacity reporting records queues that expire at the capacity deadline', 0, false],
  ['a saved failed claim survives a lost report after the original queue expires', 60, true]
]) test(name, async () => {
  const f = await expired(), row = f.event.cases.at(-1);
  f.event.cases = [row];
  Object.assign(f.run, {status: 'completed', conclusion: 'cancelled', updated_at: new Date((f.event.deadline + offset) * 1000).toISOString()});
  Object.assign(f.jobs.at(-1), {status: 'completed', runner_id: 0});
  if (saved) f.claims.set(claimKey(row), {state: 'capacity_failed', instance: row.instance, deadline: f.event.deadline});
  await capacity(f.event, f.io);
  assert.equal(f.claims.get(claimKey(row)).state, 'capacity_failed');
  assert.equal(f.calls.filter(c => c.path?.endsWith('/dispatches')).length, 1);
  if (saved) assert.deepEqual(await report(f.event, f.io), {include: [row]});
});
test('replacement reports reject foreign roots and preserve admitted claims', async () => {
  const f = await expired(), row = f.event.cases.at(-1);
  f.event.cases = [row]; f.claims.set(claimKey(row), {state: 'admitted'});
  await assert.rejects(report(f.event, f.io), /running case/);
  await assert.rejects(report({...f.event, results: 's3://another-bucket'}, f.io), /configured result/);
  f.run.head_branch = 'other';
  await assert.rejects(report(f.event, f.io), /Untrusted/);
});
test('registration tokens require the live source, exact pool label and an open deadline', async () => {
  const f = fixture(), original = {...f.run}, event = poolEvent(f, f.io.now() + 1);
  for (const status of ['queued', 'in_progress']) {
    f.run.status = status; assert.equal(await registration(event, f.io), 'fresh-registration');
  }
  for (const bad of [{repo: 'someone/else'}, {attempt: '3'}, {sha: 'b'.repeat(40)}, {deadline: f.io.now()},
    {instance: 'invalid'}, {label: 'another-pool'}]) await assert.rejects(registration({...event, ...bad}, f.io));
  for (const bad of [{head_branch: 'feature'}, {event: 'pull_request'}, {status: 'completed'},
    {status: 'completed', conclusion: 'cancelled'}, {path: 'other.yml'}]) {
    Object.assign(f.run, bad); await assert.rejects(registration(event, f.io)); Object.assign(f.run, original);
  }
  assert.equal(f.calls.filter(c => c.path?.endsWith('/registration-token')).length, 2);
});
test('drained pools stop launches without stopping active cases or unmaterialized jobs', async () => {
  const f = fixture(), pool = f.selected.pools[0], event = poolEvent(f, f.io.now() + 60);
  const own = f.jobs.filter(j => j.labels.includes(pool.label));
  own.forEach(j => { j.status = 'completed'; });
  own[0].status = 'in_progress';
  const missing = f.jobs.splice(f.jobs.indexOf(own[1]), 1)[0];
  assert.equal(await registration(event, f.io), 'fresh-registration');
  f.jobs.push(missing);
  assert.equal(await registration(event, f.io), null);
  assert.deepEqual(f.calls.filter(c => c.service === 'autoscaling' && c.operation !== 'describe-auto-scaling-groups')
    .map(c => [c.operation, c.input]), [['suspend-processes', {AutoScalingGroupName: pool.label, ScalingProcesses: ['Launch']}] ]);
  own[0].status = 'queued';
  assert.equal(await registration(event, f.io), null, 'A failed/suspended pool cannot register another runner');
});
test('bootstrap failures report the check before the deadline and preserve admitted or running cases', async () => {
  const f = await expired('weekly', true, ['10B', '25B']), pool = f.selected.pools[0];
  f.io.now = () => f.event.deadline - 60;
  const event = {...f.event, kind: 'bootstrap', instance: pool.instance, label: pool.label,
    cases: f.event.cases.filter(r => r.label === pool.label), error: 'Bootstrap failed: local SSD (exit 1)'};
  f.io.env.RESULTS_S3 = event.results;
  f.claims.set(claimKey(event.cases[0]), {state: 'admitted'});
  f.jobs.find(j => j.name.endsWith('engine-e2e:' + event.cases[1].case + ':' + event.cases[1].scale)).status = 'in_progress';
  f.jobs.splice(f.jobs.findIndex(j => j.name.endsWith('engine-e2e:' + event.cases[2].case + ':' + event.cases[2].scale)), 1);
  await registration(event, f.io);
  const dispatch = f.calls.find(c => c.path?.endsWith('/dispatches'));
  assert.ok(dispatch, 'A failed boot must dispatch its failure before the capacity deadline');
  const dispatched = JSON.parse(dispatch.data.inputs.payload);
  const rows = (await report(dispatched, f.io)).include;
  assert.equal(rows.length, event.cases.length - 2);
  assert.ok(rows.every(r => r.phase === 'bootstrap' && r.error === event.error && r.label === pool.label));
  assert.equal(f.claims.get(claimKey(event.cases[0])).state, 'admitted');
  assert.equal(f.claims.has(claimKey(event.cases[1])), false);
  assert.ok(f.calls.some(c => c.operation === 'suspend-processes' && c.input.ScalingProcesses.includes('Launch')));
  assert.equal(f.calls.filter(c => c.path?.endsWith('/registration-token')).length, 0);
  await registration({...event, error: 'Bootstrap failed: Python 3.12 (exit 2)'}, f.io);
  f.io.now = () => f.event.deadline;
  await capacity({...f.event, cases: event.cases}, f.io);
  const replay = JSON.parse(f.calls.filter(c => c.path?.endsWith('/dispatches')).at(-1).data.inputs.payload);
  assert.ok((await report(replay, f.io)).include.every(r => r.phase === 'bootstrap' && r.error === event.error));
  f.owned[0].Tags = [];
  await assert.rejects(registration(event, f.io));
});
test('repository runner administration uses its credential while workflow reads keep the job token', () => {
  const calls = [], env = {GH_TOKEN: 'runner-admin', GITHUB_TOKEN: 'workflow-token'};
  const requests = [
    ['repos/NVIDIA/kumo-engine/actions/runners?per_page=100', 'GET', 'runner-admin'],
    ['repos/NVIDIA/kumo-engine/actions/runners/1', 'DELETE', 'runner-admin'],
    ['repos/NVIDIA/kumo-engine/actions/runners/registration-token', 'POST', 'runner-admin'],
    ['repos/NVIDIA/kumo-engine/actions/runs/41', 'GET', 'workflow-token']
  ];
  runInNewContext(readFileSync(join(__dirname, '../../.github/scripts/engine-e2e-pool.cjs'), 'utf8') + `
    for (const [path, method] of requests) io.api(path, method);`, {
    requests, module: {exports: {}}, process: {env}, require: name => name === 'node:child_process' ? {
      execFileSync: (file, args, options) => { calls.push([args[1], options.env.GH_TOKEN]); return '{}'; }
    } : require(name)
  });
  assert.deepEqual(calls, requests.map(([path, , token]) => [path, token]));
});
test('registration broker restricts its fresh installation token to this repository and its required permissions', async () => {
  const f = fixture(), handler = {}, calls = [], secret = {app_id: 1, installation_id: 7,
    private_key: generateKeyPairSync('rsa', {modulusLength: 2048}).privateKey.export({type: 'pkcs8', format: 'pem'})};
  const responses = {
    '/app/installations/7/access_tokens': {token: 'installation-token'},
    '/repos/NVIDIA/kumo-engine/actions/runs/41': f.run,
    '/repos/NVIDIA/kumo-engine/actions/runners/registration-token': {token: 'fresh-registration'},
    '/repos/NVIDIA/kumo-engine/actions/runs/41/attempts/2/jobs': {jobs: f.jobs}
  };
  runInNewContext(readFileSync(join(__dirname, '../../.github/scripts/engine-e2e-capacity.cjs'), 'utf8'), {
    exports: handler, Buffer, process: {env: {RUNNER_SECRET: 'registration-secret'}},
    require: name => name.endsWith('engine-e2e-pool.cjs') ? {registration} :
      name === '@aws-sdk/client-secrets-manager' ? {
        SecretsManagerClient: class {async send() { return {SecretString: JSON.stringify(secret)}; }}, GetSecretValueCommand: class {}
      } : name === '@aws-sdk/client-s3' ? {S3Client: class {}, PutObjectCommand: class {}, GetObjectCommand: class {}} :
      name === '@aws-sdk/client-auto-scaling' ? {
        AutoScalingClient: class {async send(command) { return {AutoScalingGroups: f.owned.filter(g =>
          command.input.AutoScalingGroupNames.includes(g.AutoScalingGroupName))}; }},
        DescribeAutoScalingGroupsCommand: class {constructor(input) { this.input = input; }}, SuspendProcessesCommand: class {}
      } : require(name),
    fetch: async (url, options) => {
      calls.push(options);
      const value = responses[new URL(url).pathname]; assert.ok(value, 'Unexpected GitHub endpoint');
      return {ok: true, status: options.method === 'POST' ? 201 : 200, json: async () => value};
    }
  });
  assert.equal(await handler.handler({kind: 'registration', ...poolEvent(f, Date.now() / 1000 + 60)}), 'fresh-registration');
  assert.deepEqual(JSON.parse(calls[0].body), {repositories: ['kumo-engine'], permissions: {administration: 'write', actions: 'write'}});
  assert.ok(calls.slice(1).every(c => c.headers.Authorization === 'Bearer installation-token'));
});
