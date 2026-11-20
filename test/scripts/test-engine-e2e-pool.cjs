// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const {createHash, generateKeyPairSync} = require('node:crypto');
const {mkdtempSync, readFileSync} = require('node:fs');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const test = require('node:test');
const {runInNewContext} = require('node:vm');
const {plan, start, cleanup, capacity, report, registration, sweep} = require('../../.github/scripts/engine-e2e-pool.cjs');
const REGIONS = ['us-west-2', 'us-east-2', 'us-east-1', 'eu-west-2', 'eu-central-1', 'eu-north-1', 'ap-northeast-1', 'ap-northeast-2', 'ap-south-1'];
const STATUS = {AccessDenied: 403, ValidationError: 400, Throttling: 400, IdempotentParameterMismatchError: 400, InternalFailure: 500,
  ResourceContentionFault: 500, IdempotentCallInProgress: 500, IdempotentCallInProgressFault: 500};
const sha256 = value => createHash('sha256').update(value).digest('hex');
const runnerName = (label, id) => label + '-' + sha256(id).slice(0, 12);
const sdkError = (name, status, attempts) => Object.assign(Error(name + ' from the fixture'), {$fault: status < 500 ? 'client' : 'server',
  $retryable: undefined, $metadata: {httpStatusCode: status, requestId: '11111111-2222-3333-4444-555555555555', attempts,
    totalRetryDelay: attempts > 1 ? 847 : 0}, name, Type: 'Sender', Code: name, Error: {Type: 'Sender', Code: name, Message: name + ' from the fixture'}});
const configure = (regions = 2, subnets = 2, instances = ['g7e.8xlarge', 'g7e.24xlarge'], subnet = k => 'subnet-' + k) =>
  REGIONS.slice(0, regions).map((region, n) => ({region, templates: Object.fromEntries(instances.map(instance =>
    [instance, {template: 'lt-' + (n + 1) + '23', version: '2', subnets: Array.from({length: subnets}, (_, k) => subnet(n * subnets + k + 1))}]))}));
const templates = configure();
const requirement = name => ({scratch_gib: 100, instance: name.replace(/-control$/, '') === 'fraud_binary' ? 'g7e.24xlarge' : 'g7e.8xlarge'});
function fixture(tier = 'weekly', runControls = false, scales = [tier === 'weekly' ? '25B' : '100M'], config = templates, need = requirement) {
  const env = {GITHUB_REPOSITORY: 'NVIDIA/kumo-engine', GITHUB_RUN_ID: '41', GITHUB_RUN_ATTEMPT: '2',
    GITHUB_SHA: 'a'.repeat(40), GITHUB_REF: 'refs/heads/main', ENGINE_E2E_MAX_PARALLEL: '3',
    EXPIRY_ROLE: 'arn:aws:iam::123456789012:role/expiry', ENGINE_E2E_TEMPLATES_JSON: JSON.stringify(config),
    RUN_CONTROLS: String(runControls), AWS_REGION: 'us-west-2', RESULTS_S3: 's3://results/root', CAPACITY_ROLE: 'arn:aws:iam::123456789012:role/capacity', CAPACITY_FUNCTION: 'arn:aws:lambda:us-west-2:123456789012:function:capacity'};
  const selected = plan(tier, scales, config, 3, need, env), calls = [], claims = new Map(), etags = new Map(), zones = {};
  const now = Date.parse('2026-11-05T12:00:00Z') / 1000, deadline = now + (tier === 'nightly' ? 4 : 24) * 3600;
  const run = {run_attempt: 2, head_sha: env.GITHUB_SHA, head_branch: 'main', head_repository: {full_name: env.GITHUB_REPOSITORY},
    name: 'Kumo Engine ' + tier, path: '.github/workflows/kumo-engine-' + tier + '.yml', event: 'workflow_dispatch',
    run_started_at: '2026-11-04T12:00:00Z', status: 'in_progress'};
  const jobs = selected.matrix.include.map(r => ({name: 'weekly / engine-e2e:' + r.case + ':' + r.scale,
    labels: [r.label], status: 'queued'}));
  const owned = selected.pools.flatMap(p => p.regions.map(r => ({region: r.region, AutoScalingGroupName: p.label,
    LaunchTemplate: {LaunchTemplateId: r.template, Version: '3'}, MaxSize: Math.min(p.count, r.limit || p.count), DesiredCapacity: 0, Instances: [],
    VPCZoneIdentifier: r.subnets.join(','), Tags: Object.entries({EngineE2EPool: 'engine-e2e-41-2', EngineE2ERepository: env.GITHUB_REPOSITORY,
      EngineE2ESha: env.GITHUB_SHA, EngineE2EDeadline: String(deadline), EngineE2ECount: String(p.count)}).map(([Key, Value]) => ({Key, Value}))})));
  const activeGroups = [...owned, {region: 'us-west-2', AutoScalingGroupName: 'engine-e2e-41-1-old', Tags: []}];
  const runners = [{id: 1, name: owned[0].AutoScalingGroupName + '-host', labels: [{name: owned[0].AutoScalingGroupName}]},
    {id: 2, name: 'engine-e2e-41-1-old-host', labels: [{name: 'engine-e2e-41-1-old'}]},
    {id: 3, name: owned[0].AutoScalingGroupName + '-other', labels: [{name: 'foreign'}]},
    {id: 4, name: 'foreign', labels: [{name: owned[0].AutoScalingGroupName}]}];
  let version = 0, instances = 0, owners = 0;
  const io = {env, now: () => now, uuid: () => 'owner-' + ++owners,
    preconditionFailed: error => error.message === 'existing claim' || [409, 412].includes(error.$metadata?.httpStatusCode),
    api: async (path, method = 'GET', data) => {
      assert.ok(!path.startsWith('orgs/'), 'Runner APIs must use repository scope');
      calls.push({path, method, data});
      if (path.includes('/attempts/')) return {jobs};
      if (path.includes('/actions/runs/')) return run;
      if (path.endsWith('/registration-token')) {
        assert.equal(path, 'repos/NVIDIA/kumo-engine/actions/runners/registration-token');
        return {token: 'fresh-registration'};
      }
      return {runners};
    }, aws: async (service, operation, input, region) => {
      calls.push({service, operation, input, region});
      const group = activeGroups.find(g => g.region === region && g.AutoScalingGroupName === input.AutoScalingGroupName);
      if (operation === 'create-launch-template-version') return {LaunchTemplateVersion: {VersionNumber: 3}};
      if (operation === 'create-auto-scaling-group') group.Tags.push(...input.Tags.filter(t => t.Key === 'EngineE2EKey').map(({Key, Value}) => ({Key, Value})));
      if (operation === 'describe-launch-template-versions') return {LaunchTemplateVersions: [{VersionNumber: 1},
        {VersionNumber: 3, VersionDescription: 'engine-e2e-41-2-g7e.8xlarge'}, {VersionNumber: 4, VersionDescription: 'engine-e2e-41-20-g7e.8xlarge'}]};
      if (operation === 'describe-auto-scaling-groups') return {AutoScalingGroups: activeGroups.filter(g => g.region === region &&
        (!input.AutoScalingGroupNames || input.AutoScalingGroupNames.includes(g.AutoScalingGroupName)))};
      if (operation === 'delete-auto-scaling-group') activeGroups.splice(activeGroups.indexOf(group), 1);
      if (operation === 'suspend-processes')
        group.SuspendedProcesses = [...(group.SuspendedProcesses || []), ...input.ScalingProcesses.map(ProcessName => ({ProcessName}))];
      if (operation === 'set-desired-capacity') group.DesiredCapacity = input.DesiredCapacity;
      if (operation === 'terminate-instance-in-auto-scaling-group') {
        const g = activeGroups.find(g => g.region === region && g.Instances?.some(i => i.InstanceId === input.InstanceId));
        g.Instances = g.Instances.filter(i => i.InstanceId !== input.InstanceId);
        g.DesiredCapacity -= input.ShouldDecrementDesiredCapacity ? 1 : 0;
      }
      if (operation === 'launch-instances') {
        const zone = region + '/' + input.SubnetIds[0], answer = zones[zone] ?? 0;
        const AvailabilityZone = region + 'abcdefghijklmnopqrstuvwxyz'[group.VPCZoneIdentifier.split(',').indexOf(input.SubnetIds[0])];
        if (answer instanceof Error) throw answer;
        if (typeof answer === 'string') throw Object.assign(Error(answer + ' from the fixture in ' + zone), {name: answer,
          $metadata: {httpStatusCode: STATUS[answer]}});
        if (group.DesiredCapacity + input.RequestedCapacity > group.MaxSize)
          throw Object.assign(Error('Your requested capacity exceeds the Maximum Size of the Auto Scaling Group'), {name: 'ValidationError',
            $metadata: {httpStatusCode: 400}});
        if (answer?.token) return {AutoScalingGroupName: input.AutoScalingGroupName, ClientToken: input.ClientToken};
        if (typeof answer === 'object') return {Instances: [], Errors: [{ErrorCode: answer.code, ErrorMessage: answer.code + ' from the fixture in ' + zone,
          ...(answer.az === null ? {} : {AvailabilityZone: answer.az || AvailabilityZone})}]};
        const ids = Array.from({length: Math.min(answer, input.RequestedCapacity)}, () => 'i-' + String(++instances).padStart(8, '0'));
        zones[zone] = answer - ids.length; group.DesiredCapacity += ids.length;
        group.Instances.push(...ids.map(InstanceId => ({InstanceId, LifecycleState: 'Pending'})));
        return {Instances: ids.length ? [{SubnetId: input.SubnetIds[0], AvailabilityZone, InstanceIds: ids}] : [],
          Errors: ids.length < input.RequestedCapacity ? [{ErrorCode: 'InsufficientInstanceCapacity',
            ErrorMessage: 'We currently do not have sufficient capacity in ' + zone, AvailabilityZone, InstanceType: null}] : []};
      }
      if (operation === 'get-object') return claims.get(input.Key);
      if (operation === 'read-object') return claims.has(input.Key) ? {Body: claims.get(input.Key), ETag: etags.get(input.Key)} : null;
      if (operation === 'put-object') {
        if (input.IfNoneMatch === '*' ? claims.has(input.Key) : input.IfMatch !== undefined && etags.get(input.Key) !== input.IfMatch)
          throw Error('existing claim');
        claims.set(input.Key, JSON.parse(input.Body)); etags.set(input.Key, '"' + ++version + '"');
        return {ETag: etags.get(input.Key)};
      }
      return {};
    }};
  return {io, selected, calls, owned, run, jobs, claims, etags, zones, runners, deadline};
}
async function started(...args) {
  const f = fixture(...args); await start(f.selected, f.io);
  f.event = JSON.parse(f.calls.find(c => c.operation === 'create-schedule').input.Target.Input);
  f.sweeps = f.calls.filter(c => c.input?.Name?.endsWith('-sweep')).map(c => JSON.parse(c.input.Target.Input));
  return f;
}
async function expired(...args) {
  const f = await started(...args);
  f.io.now = () => f.event.deadline;
  return f;
}
const claimKey = row => 'root/41/2/' + row.scale + '/' + row.case + '.admission.json';
const stateKey = instance => 'root/41/2/pools/' + instance + '.sweep.json';
const closedKey = (instance, region) => 'root/41/2/pools/' + instance + '/closed/' + region + '.json';
const failureKey = instance => 'root/41/2/pools/' + instance + '.admission.json';
const launches = f => f.calls.filter(c => c.operation === 'launch-instances');
const dispatches = f => f.calls.filter(c => c.path?.endsWith('/dispatches')).map(c => JSON.parse(c.data.inputs.payload));
const seed = (f, instance, launched, extra = {}) => {
  f.claims.set(stateKey(instance), {position: 0, last_errors: {}, lease_until: 0, launched, ...extra});
  f.etags.set(stateKey(instance), '"seed"');
};
function poolEvent(f, deadline, region = 'us-west-2', instance_id = 'i-00000099') {
  const {instance, label} = f.selected.pools[0];
  const group = f.owned.find(g => g.region === region && g.AutoScalingGroupName === label);
  if (!group.Instances.some(i => i.InstanceId === instance_id)) group.Instances.push({InstanceId: instance_id, LifecycleState: 'InService'});
  return {repo: f.io.env.GITHUB_REPOSITORY, run: '41', attempt: '2', sha: f.run.head_sha, deadline, instance, label, region, instance_id,
    cases: f.selected.matrix.include.filter(r => r.label === label)};
}
function quiet(action) {
  const log = console.error, messages = [];
  console.error = message => messages.push(message);
  return action().finally(() => { console.error = log; }).then(() => messages);
}
test('cases select fixed types, independent labels and one shared VM budget', () => {
  const {selected, io} = fixture(), rows = selected.matrix.include;
  assert.equal(rows.length, 12);
  assert.equal(new Set(rows.map(r => r.label)).size, 2);
  assert.equal(rows.at(-1).instance, 'g7e.24xlarge');
  assert.equal(selected.pools.reduce((n, p) => n + p.count, 0), 3);
  assert.deepEqual(selected.pools.map(p => p.regions.map(r => r.region)), [['us-west-2', 'us-east-2'], ['us-west-2', 'us-east-2']]);
  assert.throws(() => plan('weekly', ['25B'], templates, 1, requirement, io.env), /one worker per instance type/);
});
test('templates are a strictly validated list of regions in sweep order', () => {
  const {io} = fixture(), pinned = {template: 'lt-123', version: '2', subnets: ['subnet-1']};
  const typed = t => [{region: 'us-west-2', templates: {'g7e.8xlarge': t, 'g7e.24xlarge': pinned}}];
  for (const config of [{}, [], {'g7e.8xlarge': pinned}, [{region: 'us-west-2'}], [{region: 'mars-1', templates: {}}],
    [{region: 'us-west-2', templates: []}], [...templates, templates[0]], [{...templates[0], extra: true}],
    typed({...pinned, version: '$Latest'}), typed({...pinned, template: 'template'}), typed({...pinned, subnets: []}),
    typed({...pinned, subnets: ['subnet-1', 'subnet-1']}), typed({...pinned, subnets: ['sg-1']}), typed({...pinned, subnets: 'subnet-1'}),
    typed({...pinned, limit: 0}), typed({...pinned, limit: '2'}), typed({...pinned, limits: 2}), typed(null)])
    assert.throws(() => plan('weekly', ['25B'], config, 3, requirement, io.env), /ENGINE_E2E_TEMPLATES_JSON|region entry|pinned template/);
  assert.throws(() => plan('weekly', ['25B'], configure(2, 2, ['g7e.8xlarge']), 3, requirement, io.env), /g7e\.24xlarge in at least one region/);
  const partial = [{region: 'us-west-2', templates: {'g7e.8xlarge': {...pinned, limit: 1}}}, ...configure(2).slice(1)];
  const {pools} = plan('weekly', ['25B'], partial, 3, requirement, io.env);
  assert.deepEqual(pools.map(p => p.regions), [[{region: 'us-west-2', ...pinned, limit: 1}, {region: 'us-east-2', ...partial[1].templates['g7e.8xlarge']}],
    [{region: 'us-east-2', ...partial[1].templates['g7e.24xlarge']}]]);
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
test('start arms expiry, then creates an empty suspended group in every region of each pool', async () => {
  const config = [...configure(2), {region: 'eu-west-2', templates: {'g7e.8xlarge': {template: 'lt-923', version: '4', subnets: ['subnet-9'], limit: 1}}}];
  const f = fixture('weekly', true, ['10B', '25B'], config); await start(f.selected, f.io);
  const regional = f.calls.filter(c => c.service && c.input.Target?.Arn !== f.io.env.CAPACITY_FUNCTION);
  const keys = Object.fromEntries(f.calls.filter(c => c.input?.Name?.endsWith('-sweep')).map(c => JSON.parse(c.input.Target.Input)).map(e => [e.label, e.key]));
  assert.ok(Object.values(keys).every(key => /^[0-9a-f]{64}$/.test(key)) && new Set(Object.values(keys)).size === 2, 'One random key per pool');
  assert.deepEqual(regional.map(c => [c.operation, c.region, c.input.Name || c.input.LaunchTemplateId || c.input.AutoScalingGroupName]),
    f.selected.pools.flatMap(p => p.regions.flatMap(r => [['create-schedule', r.region, p.label], ['create-launch-template-version', r.region, r.template],
      ['create-auto-scaling-group', r.region, p.label], ['suspend-processes', r.region, p.label]])));
  assert.equal(regional.filter(c => c.operation === 'create-auto-scaling-group').length, 5);
  for (const p of f.selected.pools) for (const r of p.regions) {
    const at = operation => regional.find(c => c.operation === operation && c.region === r.region &&
      [c.input.Name, c.input.VersionDescription, c.input.AutoScalingGroupName].includes(p.label)).input;
    assert.deepEqual(JSON.parse(at('create-schedule').Target.Input), {AutoScalingGroupName: p.label, ForceDelete: true});
    assert.equal(at('create-schedule').Target.Arn, 'arn:aws:scheduler:::aws-sdk:autoscaling:deleteAutoScalingGroup');
    assert.deepEqual([at('create-launch-template-version').SourceVersion, at('create-launch-template-version').LaunchTemplateData.InstanceType], [r.version, p.instance]);
    const group = at('create-auto-scaling-group');
    assert.deepEqual([group.MinSize, group.DesiredCapacity, group.MaxSize, group.VPCZoneIdentifier, group.LaunchTemplate],
      [0, 0, Math.min(p.count, r.limit || p.count), r.subnets.join(','), {LaunchTemplateId: r.template, Version: '3'}]);
    assert.equal(group.NewInstancesProtectedFromScaleIn, true, 'A lower desired capacity never removes a VM');
    assert.deepEqual(group.Tags.map(t => [t.Key, t.Value, t.PropagateAtLaunch]), [['EngineE2EPool', 'engine-e2e-41-2', true],
      ['EngineE2ERepository', 'NVIDIA/kumo-engine', true], ['EngineE2ESha', f.io.env.GITHUB_SHA, true],
      ['EngineE2EDeadline', String(Math.floor(f.io.now()) + 24 * 3600), true], ['EngineE2ECount', String(p.count), false],
      ['EngineE2EKey', sha256(keys[p.label]), false]]);
    assert.ok(!at('create-launch-template-version').LaunchTemplateData.TagSpecifications[0].Tags.some(t => ['EngineE2ECount', 'EngineE2EKey'].includes(t.Key)));
    assert.ok(!JSON.stringify(f.calls.filter(c => !c.input?.Name?.endsWith('-sweep'))).includes(keys[p.label]), 'Only the sweep schedule carries the key');
    assert.deepEqual(group.InstanceMaintenancePolicy, {MinHealthyPercentage: 0, MaxHealthyPercentage: 100});
    assert.deepEqual(at('suspend-processes').ScalingProcesses, ['Launch', 'AZRebalance']);
  }
  assert.equal(f.calls.find(c => c.operation === 'create-auto-scaling-group' && c.region === 'eu-west-2').input.MaxSize, 1);
  assert.ok(!f.calls.some(c => c.operation === 'set-desired-capacity' || c.input?.Target?.Arn.endsWith(':suspendProcesses')));
  for (const c of f.calls.filter(c => c.operation === 'create-launch-template-version')) {
    const script = Buffer.from(c.input.LaunchTemplateData.UserData, 'base64');
    assert.ok(script.length <= 16384, 'The largest control-enabled UserData must fit the EC2 limit');
    const rows = JSON.parse(script.toString().match(/^POOL_CASES='(.+)'$/m)[1]);
    assert.ok(rows.length > 0 && rows.every(r => r.instance === c.input.LaunchTemplateData.InstanceType));
    assert.match(script.toString(), /^AWS_REGION='us-west-2'$/m, 'VMs in every region call the broker in the home region');
  }
});
test('each pool gets a one-minute sweep schedule in the home region with a compact event', async () => {
  const config = configure(9, 4, ['g7e.8xlarge'], k => 'subnet-' + String(k).padStart(17, '0'));
  const f = fixture('weekly', true, ['10B', '25B'], config, () => ({scratch_gib: 100, instance: 'g7e.8xlarge'}));
  f.io.env.RESULTS_S3 = 's3://results-bucket-name/engine-e2e/results'; await start(f.selected, f.io);
  const [schedule, ...others] = f.calls.filter(c => c.input?.Name?.endsWith('-sweep')), pool = f.selected.pools[0];
  assert.equal(others.length, 0);
  assert.ok(f.calls.indexOf(schedule) > f.calls.findLastIndex(c => c.operation === 'suspend-processes'), 'Sweeps start after regional setup');
  assert.deepEqual([schedule.region, schedule.input.Name, schedule.input.ScheduleExpression, schedule.input.ActionAfterCompletion,
    schedule.input.Target.Arn, schedule.input.Target.RoleArn, schedule.input.Target.RetryPolicy, schedule.input.FlexibleTimeWindow],
  ['us-west-2', pool.label + '-sweep', 'rate(1 minute)', 'DELETE', f.io.env.CAPACITY_FUNCTION, f.io.env.CAPACITY_ROLE, {MaximumRetryAttempts: 0}, {Mode: 'OFF'}]);
  assert.ok(!Object.hasOwn(schedule.input, 'StartDate'), 'A rate schedule without StartDate starts at creation');
  assert.equal(Date.parse(schedule.input.EndDate) / 1000, f.deadline);
  const event = JSON.parse(schedule.input.Target.Input);
  assert.ok(schedule.input.Target.Input.length <= 8192, 'EventBridge Scheduler limits Input to 8192 characters');
  assert.match(event.key, /^[0-9a-f]{64}$/);
  assert.deepEqual(event, {kind: 'sweep', repo: 'NVIDIA/kumo-engine', run: '41', attempt: '2', sha: f.run.head_sha, deadline: f.deadline,
    results: f.io.env.RESULTS_S3, label: pool.label, instance: 'g7e.8xlarge', count: 3, key: event.key,
    cases: f.selected.matrix.include.map(r => ({case: r.case, scale: r.scale})), regions: config.map(r => ({region: r.region, subnets: r.templates['g7e.8xlarge'].subnets}))});
  assert.deepEqual([event.cases.length, event.regions.length, event.regions.flatMap(r => r.subnets).length], [48, 9, 36]);
});
test('a region that fails setup is reported with only its AWS error code and left out of the sweep', async () => {
  const f = fixture('weekly', false, ['25B'], configure(3)), aws = f.io.aws;
  const summary = join(mkdtempSync(join(tmpdir(), 'engine-e2e-pool-')), 'summary.md');
  f.io.env.GITHUB_STEP_SUMMARY = summary;
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'create-auto-scaling-group' && region === 'us-east-2' && input.AutoScalingGroupName.endsWith('g7e.8xlarge'))
      throw Error('\nAn error occurred (AccessDenied) when calling the CreateAutoScalingGroup operation: User: arn:aws:sts::123456789012:' +
        'assumed-role/pool/run is not authorized to perform: autoscaling:CreateAutoScalingGroup on subnet-3\n');
    if (operation === 'create-launch-template-version' && region === 'us-east-1' && input.LaunchTemplateData.InstanceType === 'g7e.24xlarge')
      throw Error('connect ECONNREFUSED for lt-323 in arn:aws:ec2:us-east-1:123456789012:launch-template/lt-323');
    return aws(service, operation, input, region);
  };
  const messages = await quiet(() => start(f.selected, f.io));
  const sweeps = f.calls.filter(c => c.input?.Name?.endsWith('-sweep')).map(c => JSON.parse(c.input.Target.Input));
  assert.deepEqual(sweeps.map(e => [e.instance, e.regions.map(r => r.region)]),
    [['g7e.8xlarge', ['us-west-2', 'us-east-1']], ['g7e.24xlarge', ['us-west-2', 'us-east-2']]]);
  assert.ok(!f.calls.some(c => c.operation === 'suspend-processes' && c.region === 'us-east-2' && c.input.AutoScalingGroupName.endsWith('g7e.8xlarge')));
  assert.deepEqual(messages, ['Skipped us-east-2 for engine-e2e-41-2-g7e.8xlarge: AccessDenied',
    'Skipped us-east-1 for engine-e2e-41-2-g7e.24xlarge: Error']);
  const text = readFileSync(summary, 'utf8');
  assert.equal(text, messages.map(m => '- ' + m + '\n').join(''));
  assert.ok(!/arn:|123456789012|subnet-|lt-|ECONNREFUSED|An error occurred/.test(text), 'The public summary carries no AWS text');
});
test('a pool without any region that passed setup fails the run before any sweep', async () => {
  const f = fixture(), aws = f.io.aws;
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'create-launch-template-version' && input.LaunchTemplateData.InstanceType === 'g7e.24xlarge') throw Error('template unavailable');
    return aws(service, operation, input, region);
  };
  let failure;
  const messages = await quiet(() => start(f.selected, f.io).catch(error => { failure = error; }));
  assert.match(failure.message, /^No region passed setup for engine-e2e-41-2-g7e\.24xlarge$/);
  assert.equal(messages.length, 2);
  assert.ok(!f.calls.some(c => c.input?.Name?.endsWith('-sweep')));
  assert.deepEqual(f.calls.filter(c => c.operation === 'create-auto-scaling-group').map(c => c.input.AutoScalingGroupName),
    ['engine-e2e-41-2-g7e.8xlarge', 'engine-e2e-41-2-g7e.8xlarge']);
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
    module: {exports: {}}, process: {env, argv: ['node', 'controller', 'start']}, Buffer, URL,
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
  if (allowed === true) {
    assert.match(outputs[0], /^plan=.*\nattempt=2\n$/);
    assert.ok(writes.every(args => args.at(-2) === '--region' && REGIONS.slice(0, 2).includes(args.at(-1))), 'Every AWS call names its region');
    assert.deepEqual([...new Set(writes.map(args => args.at(-1)))], ['us-west-2', 'us-east-2']);
  }
  assert.equal(requests.length, tier === 'nightly' && event === 'schedule' && allowed !== null ? 2 : 1);
});
test('the sweep key never reaches a log, the step summary, the job output or the plan', async () => {
  const f = fixture(), texts = [], inputs = [];
  const env = {...f.io.env, TIER: 'weekly', SCALES: JSON.stringify(['25B']), ENGINE_E2E_ENABLED: 'true', ENGINE_E2E_SCHEDULER_ENABLED: 'true',
    GITHUB_OUTPUT: 'fixture-output', GITHUB_STEP_SUMMARY: 'fixture-summary'};
  await runInNewContext(readFileSync(join(__dirname, '../../.github/scripts/engine-e2e-pool.cjs'), 'utf8') + '\nmain();', {
    module: {exports: {}}, process: {env, argv: ['node', 'controller', 'start']}, Buffer, URL, __dirname: join(__dirname, '../../.github/scripts'),
    console: {error: message => texts.push('log:' + message), log: message => texts.push('log:' + message)},
    require: name => name === 'node:fs' ? {readFileSync, appendFileSync: (path, value) => texts.push(path + ':' + value)} :
      name === 'node:child_process' ? {execFileSync: (file, args) => {
        if (file === 'python') return JSON.stringify(requirement(args[0].split('/').at(-1).slice(0, -3)));
        if (file === 'gh') return JSON.stringify(f.run);
        inputs.push(JSON.parse(args[args.indexOf('--cli-input-json') + 1]));
        if (args[1] === 'create-auto-scaling-group' && args.at(-1) === 'us-east-2' && inputs.at(-1).AutoScalingGroupName.endsWith('g7e.8xlarge'))
          throw Object.assign(Error('Command failed'), {stderr: '\nAn error occurred (AccessDenied) when calling the CreateAutoScalingGroup operation\n'});
        return JSON.stringify({LaunchTemplateVersion: {VersionNumber: 3}});
      }} : require(name)
  });
  const keys = inputs.filter(i => i.Name?.endsWith('-sweep')).map(i => JSON.parse(i.Target.Input).key);
  assert.ok(keys.length === 2 && keys.every(key => /^[0-9a-f]{64}$/.test(key)));
  assert.deepEqual(inputs.filter(i => i.Tags).map(i => [i.AutoScalingGroupName, i.Tags.find(t => t.Key === 'EngineE2EKey').Value]),
    [['engine-e2e-41-2-g7e.8xlarge', sha256(keys[0])], ['engine-e2e-41-2-g7e.8xlarge', sha256(keys[0])],
      ['engine-e2e-41-2-g7e.24xlarge', sha256(keys[1])], ['engine-e2e-41-2-g7e.24xlarge', sha256(keys[1])]]);
  assert.deepEqual(texts.filter(text => !text.startsWith('fixture-output:plan=')), ['log:Skipped us-east-2 for engine-e2e-41-2-g7e.8xlarge: AccessDenied',
    'fixture-summary:- Skipped us-east-2 for engine-e2e-41-2-g7e.8xlarge: AccessDenied\n']);
  assert.equal(texts.filter(text => text.startsWith('fixture-output:plan=')).length, 1);
  assert.ok(keys.every(key => texts.every(text => !text.includes(key) && !text.includes(sha256(key)))));
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
  assert.deepEqual([operations[1].operation, operations[1].region, operations[1].input.Target.Arn],
    ['create-schedule', 'us-west-2', 'arn:aws:scheduler:::aws-sdk:autoscaling:deleteAutoScalingGroup']);
  assert.equal(JSON.parse(operations[1].input.Target.Input).ForceDelete, true);
  assert.ok(!g.calls.some(c => c.input?.Name?.endsWith('-wait')));
  const h = fixture(), regional = h.io.aws;
  h.io.aws = async (service, operation, input, region) => {
    if (service === 'scheduler' && region === 'us-east-2') throw Error('expiry unavailable');
    return regional(service, operation, input, region);
  };
  await quiet(() => start(h.selected, h.io));
  assert.ok(!h.calls.some(c => c.region === 'us-east-2'), 'A region without expiry gets no group');
});
test('cleanup deletes only owned groups, template versions and runner registrations in every region', async () => {
  const f = fixture(); await cleanup(f.io);
  assert.deepEqual(f.calls.filter(c => c.operation === 'delete-auto-scaling-group').map(c => c.region + '/' + c.input.AutoScalingGroupName).sort(),
    f.owned.map(g => g.region + '/' + g.AutoScalingGroupName).sort());
  assert.ok(f.calls.filter(c => c.operation === 'describe-auto-scaling-groups').every(c => REGIONS.slice(0, 2).includes(c.region)));
  assert.deepEqual(f.calls.filter(c => c.operation === 'delete-launch-template-versions').map(c => [c.region, c.input]),
    [['us-west-2', {LaunchTemplateId: 'lt-123', Versions: ['3']}], ['us-east-2', {LaunchTemplateId: 'lt-223', Versions: ['3']}]]);
  assert.deepEqual(f.calls.filter(c => c.method === 'DELETE').map(c => c.path), ['repos/NVIDIA/kumo-engine/actions/runners/1']);
});
test('cleanup waits for every owned group in every region before deleting dependent resources', async () => {
  const f = fixture(), aws = f.io.aws; let polls = 0, sleeps = 0;
  f.io.sleep = async () => { sleeps++; };
  f.io.aws = async (service, operation, input, region) => {
    assert.notEqual(operation, 'wait group-not-exists', 'Auto Scaling has no CLI waiter');
    if (operation === 'describe-auto-scaling-groups' && input.AutoScalingGroupNames) {
      assert.equal(f.calls.filter(c => c.operation === 'delete-auto-scaling-group').length, f.owned.length, 'Every region deletes before any wait');
      assert.deepEqual(f.calls.filter(c => c.operation === 'delete-auto-scaling-group' && c.region === region).map(c => c.input.AutoScalingGroupName),
        input.AutoScalingGroupNames);
      return {AutoScalingGroups: polls++ < 2 ? f.owned.filter(g => g.region === region) : []};
    }
    if (operation === 'describe-launch-template-versions') assert.equal(polls, 4);
    return aws(service, operation, input, region);
  };
  await cleanup(f.io);
  assert.equal(sleeps, 2);
  assert.equal(f.calls.filter(c => c.method === 'DELETE').length, 1);
});
test('cleanup stops waiting within its job window if group deletion stalls and still removes the rest', async () => {
  const f = fixture(), aws = f.io.aws; let now = 0, failure;
  f.io.now = () => now;
  f.io.sleep = async ms => { now += ms / 1000; assert.ok(now < 900, 'Cleanup exceeded its job window'); };
  f.io.aws = async (service, operation, input, region) => {
    assert.notEqual(operation, 'wait group-not-exists', 'Auto Scaling has no CLI waiter');
    if (operation === 'describe-auto-scaling-groups' && input.AutoScalingGroupNames) return {AutoScalingGroups: f.owned};
    return aws(service, operation, input, region);
  };
  const messages = await quiet(() => cleanup(f.io).catch(error => { failure = error; }));
  assert.equal(failure.message, 'Cleanup failed in us-west-2 (Group deletion timed out), us-east-2 (Group deletion timed out)');
  assert.deepEqual(messages, ['us-west-2', 'us-east-2'].map(region => 'Cleanup failed in ' + region + ': Group deletion timed out'));
  assert.ok(now > 0 && now < 900);
  assert.deepEqual(f.calls.filter(c => c.operation === 'delete-launch-template-versions').map(c => c.region), ['us-west-2', 'us-east-2']);
  assert.deepEqual(f.calls.filter(c => c.method === 'DELETE').map(c => c.path), ['repos/NVIDIA/kumo-engine/actions/runners/1']);
});
test('cleanup goes on past a failing middle region in every phase and names it at the end', async () => {
  for (const phase of ['delete', 'wait', 'versions']) {
    const f = fixture('weekly', false, ['25B'], configure(3)), aws = f.io.aws; let failure;
    f.io.sleep = async () => {};
    f.io.aws = async (service, operation, input, region) => {
      const failing = region === 'us-east-2' && (phase === 'versions' ? operation === 'describe-launch-template-versions' :
        operation === 'describe-auto-scaling-groups' && !input.AutoScalingGroupNames === (phase === 'delete'));
      if (failing) throw Error('\nAn error occurred (Throttling) when calling the ' + operation + ' operation (reached max retries: 2): ' +
        'Rate exceeded for arn:aws:sts::123456789012:assumed-role/pool/run\n');
      return aws(service, operation, input, region);
    };
    const messages = await quiet(() => cleanup(f.io).catch(error => { failure = error; }));
    assert.equal(failure.message, 'Cleanup failed in us-east-2 (Throttling)', phase);
    assert.deepEqual(messages, ['Cleanup failed in us-east-2: Throttling']);
    assert.deepEqual([...new Set(f.calls.filter(c => c.operation === 'delete-auto-scaling-group').map(c => c.region))],
      phase === 'delete' ? ['us-west-2', 'us-east-1'] : ['us-west-2', 'us-east-2', 'us-east-1']);
    assert.deepEqual(f.calls.filter(c => c.operation === 'delete-launch-template-versions').map(c => c.region),
      phase === 'versions' ? ['us-west-2', 'us-east-1'] : ['us-west-2', 'us-east-2', 'us-east-1']);
    assert.deepEqual(f.calls.filter(c => c.method === 'DELETE').map(c => c.path), ['repos/NVIDIA/kumo-engine/actions/runners/1']);
  }
});
test('cleanup deletes the other groups of a region after one group deletion fails, and waits only for accepted deletions', async () => {
  const f = fixture('weekly', false, ['25B'], configure(1)), aws = f.io.aws, waited = [];
  let failure, now = 0;
  f.io.now = () => now;
  f.io.sleep = async ms => { now += ms / 1000; };
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'delete-auto-scaling-group' && input.AutoScalingGroupName.endsWith('g7e.8xlarge'))
      throw Error('\nAn error occurred (ScalingActivityInProgress) when calling the DeleteAutoScalingGroup operation: Scaling activity is in progress\n');
    if (operation === 'describe-auto-scaling-groups' && input.AutoScalingGroupNames) waited.push(...input.AutoScalingGroupNames);
    return aws(service, operation, input, region);
  };
  const messages = await quiet(() => cleanup(f.io).catch(error => { failure = error.message; }));
  assert.deepEqual(f.calls.filter(c => c.operation === 'delete-auto-scaling-group').map(c => c.input.AutoScalingGroupName), ['engine-e2e-41-2-g7e.24xlarge']);
  assert.deepEqual([waited, failure, messages], [['engine-e2e-41-2-g7e.24xlarge'], 'Cleanup failed in us-west-2 (ScalingActivityInProgress)',
    ['Cleanup failed in us-west-2: ScalingActivityInProgress']]);
  assert.deepEqual(f.calls.filter(c => c.method === 'DELETE').map(c => c.path), ['repos/NVIDIA/kumo-engine/actions/runners/1']);
});
test('cleanup fails with a clear message and deletes nothing without a region list', async () => {
  for (const value of ['', '[]', undefined]) {
    const f = fixture();
    f.io.env.ENGINE_E2E_TEMPLATES_JSON = value;
    await assert.rejects(cleanup(f.io), /^AssertionError.*: Set ENGINE_E2E_TEMPLATES_JSON to a list of regions in sweep order$/);
    assert.equal(f.calls.length, 0);
  }
});
test('capacity, sweep and provider deadlines keep the configured wait and execution windows', async () => {
  for (const [tier, wait, execution] of [['nightly', 4, 3], ['weekly', 24, 6]]) {
    const f = fixture(tier); await start(f.selected, f.io);
    const schedules = f.calls.filter(c => c.operation === 'create-schedule'), at = (c, key) => Date.parse(c.input[key]) / 1000 - f.io.now();
    assert.equal(at(schedules[0], 'StartDate'), wait * 3600);
    const expiry = schedules.filter(c => c.input.Target.Arn.endsWith(':deleteAutoScalingGroup'));
    assert.equal(expiry.length, 4);
    assert.ok(expiry.every(c => at(c, 'StartDate') === (wait + execution) * 3600 + 1800));
    const sweeps = schedules.filter(c => c.input.Name.endsWith('-sweep'));
    assert.ok(sweeps.length === 2 && sweeps.every(c => !Object.hasOwn(c.input, 'StartDate') && at(c, 'EndDate') === wait * 3600));
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
test('the public lifecycle payload carries only the run, its result root and the failed cases on every path', async () => {
  const keys = ['repo', 'run', 'attempt', 'sha', 'deadline', 'results', 'cases'], payloads = [];
  const collect = f => payloads.push(...f.calls.filter(c => c.path?.endsWith('/dispatches')).map(c => JSON.parse(c.data.inputs.payload)));
  const deadline = await expired();
  await capacity(deadline.event, deadline.io); collect(deadline);
  const booted = await started('weekly', false, ['25B'], configure(1));
  await sweep(booted.sweeps[0], booted.io);
  await registration({...poolEvent(booted, booted.deadline, 'us-west-2', 'i-0123456789abcdef0'), kind: 'bootstrap',
    error: 'Bootstrap failed: GPU (exit 1)'}, booted.io);
  booted.owned[0].Instances = [];
  await sweep(booted.sweeps[0], booted.io); collect(booted);
  const swept = await started(), event = swept.sweeps[0];
  for (const region of ['us-west-2', 'us-east-2'])
    swept.claims.set(closedKey(event.instance, region), {region, reason: 'Bootstrap failed: local SSD setup (exit 7)', at: 0});
  await sweep(event, swept.io); collect(swept);
  assert.equal(payloads.length, 3);
  for (const payload of payloads) {
    assert.deepEqual(Object.keys(payload), keys);
    assert.ok(payload.cases.every(r => ['case', 'scale', 'instance', 'label'].every(k => typeof r[k] === 'string')));
    assert.ok(!/i-0123456789abcdef0|us-west-2|Bootstrap failed/.test(JSON.stringify(payload)), 'No VM, region or error text in the public payload');
    assert.ok((await report(payload, payload === payloads[0] ? deadline.io : payload === payloads[1] ? booted.io : swept.io)).include.length);
  }
});
test('capacity counts only case jobs that carry the pool label', async () => {
  const f = await expired(), row = f.event.cases[0], other = f.event.cases[1];
  f.jobs[0].labels = ['engine-e2e-41-1-' + row.instance];
  Object.assign(f.jobs[1], {labels: ['engine-e2e-41-1-' + other.instance], status: 'in_progress'});
  f.jobs.push({name: f.jobs[1].name, labels: [other.label], status: 'queued'});
  await capacity(f.event, f.io);
  assert.equal(f.claims.has(claimKey(row)), false, 'A same-name job of another pool is not this case');
  assert.equal(f.claims.get(claimKey(other)).state, 'capacity_failed', 'The queued job with the pool label decides the case');
});
test('a claim write that meets an S3 409 without a stored claim writes the claim once more, and throws if that fails too', async () => {
  for (const conflicts of [1, 2]) {
    const f = await expired(), row = f.event.cases[3], aws = f.io.aws;
    let puts = 0;
    f.io.aws = async (service, operation, input, region) => {
      if (operation === 'put-object' && input.Key === claimKey(row) && ++puts <= conflicts)
        throw Object.assign(Error('A conflicting conditional operation is currently in progress against this resource.'),
          {name: 'ConditionalRequestConflict', $metadata: {httpStatusCode: 409}});
      return aws(service, operation, input, region);
    };
    if (conflicts === 2) await assert.rejects(capacity(f.event, f.io), {name: 'ConditionalRequestConflict'});
    else await capacity(f.event, f.io);
    assert.deepEqual([puts, f.claims.get(claimKey(row))?.state], [2, conflicts === 1 ? 'capacity_failed' : undefined], String(conflicts));
    assert.deepEqual(dispatches(f).map(p => p.cases.some(r => r.case === row.case && r.scale === row.scale)), conflicts === 1 ? [true] : []);
  }
});
test('replacement reports reject foreign roots and preserve admitted claims', async () => {
  const f = await expired(), row = f.event.cases.at(-1);
  f.event.cases = [row]; f.claims.set(claimKey(row), {state: 'admitted'});
  await assert.rejects(report(f.event, f.io), /running case/);
  await assert.rejects(report({...f.event, results: 's3://another-bucket'}, f.io), /configured result/);
  f.run.head_branch = 'other';
  await assert.rejects(report(f.event, f.io), /Untrusted/);
});
test('capacity failures at the deadline list the last error of every zone within 1000 characters', async () => {
  const f = await started(), event = f.sweeps[0];
  f.zones['us-east-2/subnet-3'] = 'AccessDenied';
  await sweep(event, f.io);
  f.io.now = () => f.deadline;
  await capacity(f.event, f.io);
  const row = f.event.cases.find(r => r.label === event.label), claim = f.claims.get(claimKey(row));
  assert.equal(claim.error, 'No runner capacity before tier deadline. Last zone errors: InsufficientInstanceCapacity in ' +
    'us-west-2a, us-west-2b; AccessDenied in us-east-2.');
  assert.equal(claim.state, 'capacity_failed');
  assert.match(f.claims.get(stateKey(event.instance)).last_errors['us-east-2'], /^AccessDenied: AccessDenied from the fixture in us-east-2\/subnet-3$/,
    'The AWS message stays in the private sweep state');
  const rows = (await report(JSON.parse(f.calls.find(c => c.path?.endsWith('/dispatches')).data.inputs.payload), f.io)).include;
  assert.deepEqual(rows.find(r => r.case === row.case && r.scale === row.scale), {...row, error: claim.error});
  assert.ok(rows.filter(r => r.label !== event.label).every(r => !r.error && !r.phase), 'A pool without zone errors keeps the default text');
  const g = await expired(), many = Object.fromEntries(Array.from({length: 60}, (_, k) =>
    ['ap-south-1#' + (k + 1), 'VcpuLimitExceeded' + k + ': ' + 'x'.repeat(200)]));
  g.claims.set(stateKey('g7e.8xlarge'), {last_errors: many});
  await capacity(g.event, g.io);
  const long = g.claims.get(claimKey(g.event.cases[0])).error;
  assert.ok(long.length === 1000 && long.endsWith('...') && long.startsWith('No runner capacity before tier deadline. Last zone errors: VcpuLimitExceeded0 in ap-south-1#1; '));
  assert.equal((await report({...g.event, cases: [g.event.cases[0]]}, g.io)).include[0].error, long);
  for (const error of ['x', 'No runner capacity before tier deadline', 'No runner capacity before tier deadline. ' + 'x'.repeat(1000), 7]) {
    g.claims.set(claimKey(g.event.cases[0]), {state: 'capacity_failed', instance: g.event.cases[0].instance, deadline: g.event.deadline, error});
    await assert.rejects(report({...g.event, cases: [g.event.cases[0]]}, g.io), /Invalid capacity failure detail/);
  }
  const h = await expired();
  h.claims.set(stateKey('g7e.8xlarge'), {last_errors: {'us-west-2/subnet-1': 'Unsupported: in subnet-1', 'i-0123456789abcdef0': 'Unsupported: x',
    'us-west-2a': 'Not a code: capacity in subnet-1'}, regions: [{region: 'us-west-2'}, {region: 'subnet-1'}]});
  await capacity(h.event, h.io);
  assert.equal(h.claims.get(claimKey(h.event.cases[0])).error, 'No runner capacity before tier deadline. Last zone errors: Error in us-west-2a.',
    'Only codes, zone names and region names reach the public error');
  assert.ok(!h.calls.some(c => c.input?.Key?.includes('/closed/subnet-1')));
});
test('every zone of nine regions fits the public capacity error with zone names and no AWS text', async () => {
  const config = configure(9, 3, ['g7e.8xlarge'], k => 'subnet-' + String(k).padStart(17, '0'));
  const f = await started('weekly', false, ['25B'], config, () => ({scratch_gib: 100, instance: 'g7e.8xlarge'})), event = f.sweeps[0];
  f.zones['ap-south-1/' + config[8].templates['g7e.8xlarge'].subnets[2]] = {code: 'Unsupported'};
  const result = await sweep(event, f.io);
  assert.equal(result.answers.length, 27);
  f.io.now = () => f.deadline;
  await capacity(f.event, f.io);
  const error = f.claims.get(claimKey(f.event.cases[0])).error;
  assert.equal(error, 'No runner capacity before tier deadline. Last zone errors: InsufficientInstanceCapacity in ' +
    REGIONS.flatMap(region => ['a', 'b', 'c'].map(zone => region + zone)).slice(0, 26).join(', ') + '; Unsupported in ap-south-1c.');
  assert.ok(error.length < 1000 && !/subnet-|We currently|fixture/.test(error));
});
test('a long capacity error drops the region reasons first and then the end of the zone list, so the closed region names stay', async () => {
  const names = REGIONS.slice(0, 8), reason = 'Bootstrap failed: ' + 'x'.repeat(130) + ' (exit 1)', tail = 'Closed regions: ' + names.join(', ') + '.';
  for (const zones of [36, 1]) {
    const f = await expired();
    f.claims.set(stateKey('g7e.8xlarge'), {regions: REGIONS.map(region => ({region})), last_errors: Object.fromEntries(REGIONS.flatMap(region =>
      ['a', 'b', 'c', 'd'].map(zone => [region + zone, 'Capacity' + region.replace(/-/g, '') + zone + 'x'.repeat(30) + ': message'])).slice(0, zones))});
    for (const region of names) f.claims.set(closedKey('g7e.8xlarge', region), {region, reason, at: 0});
    await capacity(f.event, f.io);
    const error = f.claims.get(claimKey(f.event.cases[0])).error;
    if (zones === 1)
      assert.equal(error, 'No runner capacity before tier deadline. Last zone errors: Capacityuswest2a' + 'x'.repeat(30) + ' in us-west-2a. ' + tail);
    else assert.ok(error.length === 1000 && error.endsWith('... ' + tail) && !error.includes('Bootstrap failed') &&
      error.startsWith('No runner capacity before tier deadline. Last zone errors: Capacityuswest2a' + 'x'.repeat(30) + ' in us-west-2a; '), error);
    assert.equal((await report({...f.event, cases: [f.event.cases[0]]}, f.io)).include[0].error, error);
  }
});
test('the public capacity error names every closed region but only the known close reasons', async () => {
  const f = await expired(), reasons = {'us-west-2': 'Private text from a marker', 'us-east-2': 'VM ended without a runner'};
  f.claims.set(stateKey('g7e.8xlarge'), {regions: Object.keys(reasons).map(region => ({region}))});
  for (const [region, reason] of Object.entries(reasons)) f.claims.set(closedKey('g7e.8xlarge', region), {region, reason, at: 0});
  await capacity(f.event, f.io);
  assert.equal(f.claims.get(claimKey(f.event.cases[0])).error,
    'No runner capacity before tier deadline. Closed regions: us-west-2, us-east-2 (VM ended without a runner).');
});
test('registration tokens require the live source, exact pool label, owned VM and an open deadline', async () => {
  const f = fixture(), original = {...f.run}, event = poolEvent(f, f.io.now() + 1);
  for (const status of ['queued', 'in_progress']) {
    f.run.status = status; assert.equal(await registration(event, f.io), 'fresh-registration');
  }
  for (const bad of [{repo: 'someone/else'}, {attempt: '3'}, {sha: 'b'.repeat(40)}, {instance: 'invalid'}, {label: 'another-pool'},
    {region: 'us-west'}, {region: undefined}, {region: ['us-west-2']}, {instance_id: 'i-xyz'}, {instance_id: undefined},
    {instance_id: 'i-00000098'}, {region: 'us-east-2'}]) await assert.rejects(registration({...event, ...bad}, f.io));
  assert.equal(await registration({...event, deadline: f.io.now()}, f.io), null);
  for (const bad of [{head_branch: 'feature'}, {event: 'pull_request'}, {status: 'completed'},
    {status: 'completed', conclusion: 'cancelled'}, {path: 'other.yml'}]) {
    Object.assign(f.run, bad); await assert.rejects(registration(event, f.io)); Object.assign(f.run, original);
  }
  assert.equal(f.calls.filter(c => c.path?.endsWith('/registration-token')).length, 2);
  assert.ok(f.calls.some(c => c.operation === 'describe-auto-scaling-groups' && c.region === 'us-east-2'), 'The group is read in the VM region');
});
test('registration gives no token to a drained pool or a closed region and never suspends launches', async () => {
  const f = fixture(), pool = f.selected.pools[0], event = poolEvent(f, f.io.now() + 60);
  const own = f.jobs.filter(j => j.labels.includes(pool.label));
  own.forEach(j => { j.status = 'completed'; });
  own[0].status = 'in_progress';
  const missing = f.jobs.splice(f.jobs.indexOf(own[1]), 1)[0];
  assert.equal(await registration(event, f.io), 'fresh-registration');
  f.jobs.push(missing);
  assert.equal(await registration(event, f.io), null);
  own[0].status = 'queued';
  assert.equal(await registration(event, f.io), 'fresh-registration');
  f.claims.set(closedKey(pool.instance, 'us-west-2'), {region: 'us-west-2', reason: 'fixture', at: 0});
  assert.equal(await registration(event, f.io), null, 'A closed region cannot register another runner');
  assert.equal(await registration(poolEvent(f, f.io.now() + 60, 'us-east-2', 'i-00000098'), f.io), 'fresh-registration');
  assert.ok(!f.calls.some(c => c.service === 'autoscaling' && c.operation !== 'describe-auto-scaling-groups'));
});
test('registration gives no token to a VM that is not pending or in service', async () => {
  const f = fixture(), event = poolEvent(f, f.io.now() + 60), vm = f.owned[0].Instances.find(i => i.InstanceId === event.instance_id);
  for (const state of ['Pending', 'Pending:Wait', 'Pending:Proceed', 'InService']) {
    vm.LifecycleState = state; assert.equal(await registration({...event, kind: 'registration'}, f.io), 'fresh-registration', state);
  }
  for (const state of ['Terminating', 'Terminating:Wait', 'Terminating:Proceed', 'Terminated', 'Standby', 'EnteringStandby', 'Detaching', 'Detached']) {
    vm.LifecycleState = state; assert.equal(await registration({...event, kind: 'registration'}, f.io), null, state);
  }
  assert.equal(f.calls.filter(c => c.path?.endsWith('/registration-token')).length, 4);
});
test('registration asks for no token once the deadline passes during its checks', async () => {
  const f = fixture(), event = poolEvent(f, f.io.now() + 60), api = f.io.api;
  let now = f.io.now();
  f.io.now = () => now;
  f.io.api = async (path, ...rest) => {
    if (path.includes('/attempts/')) now += 60;
    return api(path, ...rest);
  };
  assert.equal(await registration(event, f.io), null);
  assert.ok(f.calls.some(c => c.path?.includes('/attempts/')) && !f.calls.some(c => c.path?.endsWith('/registration-token')));
});
test('a boot report only closes the region of its VM, and a sweep fails the waiting cases once every region is closed and no VM is left', async () => {
  const f = await started('weekly', true, ['10B', '25B']), pool = f.selected.pools[0], event = f.sweeps[0];
  const cases = f.event.cases.filter(r => r.label === pool.label);
  f.io.now = () => f.deadline - 3600;
  await sweep(event, f.io);
  f.claims.set(claimKey(cases[0]), {state: 'admitted'});
  f.jobs.find(j => j.name.endsWith('engine-e2e:' + cases[1].case + ':' + cases[1].scale)).status = 'in_progress';
  f.jobs.splice(f.jobs.findIndex(j => j.name.endsWith('engine-e2e:' + cases[2].case + ':' + cases[2].scale)), 1);
  const west = {...poolEvent(f, f.deadline, 'us-west-2', 'i-00000101'), kind: 'bootstrap', error: 'Bootstrap failed: GPU (exit 1)'};
  const east = {...poolEvent(f, f.deadline, 'us-east-2', 'i-00000102'), kind: 'bootstrap', error: 'Bootstrap failed: local SSD setup (exit 7)'};
  const before = f.calls.length;
  assert.equal(await registration(west, f.io), null);
  const reports = f.calls.slice(before);
  assert.deepEqual(f.claims.get(closedKey(pool.instance, 'us-west-2')), {region: 'us-west-2', reason: west.error, at: f.io.now()});
  assert.equal(await registration({...west, kind: 'registration'}, f.io), null);
  assert.equal(await registration({...east, kind: 'registration'}, f.io), 'fresh-registration');
  const reported = f.calls.length;
  assert.equal(await registration(east, f.io), null);
  reports.push(...f.calls.slice(reported));
  assert.deepEqual(reports.map(c => c.operation + ' ' + c.region), ['describe-auto-scaling-groups us-west-2', 'put-object undefined',
    'describe-auto-scaling-groups us-east-2', 'put-object undefined'], 'A boot report reads its group, writes its marker and calls nothing else');
  assert.deepEqual([f.claims.has(failureKey(pool.instance)), dispatches(f).length], [false, 0], 'A boot report never fails the cases itself');
  assert.deepEqual([(await sweep(event, f.io)).filled, dispatches(f).length], [2, 0], 'The VMs that reported still run');
  f.owned.forEach(g => { g.Instances = []; });
  await sweep(event, f.io);
  assert.equal(f.claims.get(failureKey(pool.instance)).error, west.error, 'The first hard reason in region order decides the cause');
  const [replay] = dispatches(f);
  assert.deepEqual(replay.cases.map(r => r.case + ':' + r.scale), cases.slice(3).map(r => r.case + ':' + r.scale));
  assert.ok((await report(replay, f.io)).include.every(r => r.phase === 'bootstrap' && r.error === west.error));
  await sweep(event, f.io);
  assert.deepEqual([dispatches(f).length, f.claims.get(stateKey(pool.instance)).dispatched], [1, true], 'A saved dispatch is not repeated');
  f.io.now = () => f.deadline;
  await capacity(f.event, f.io);
  const final = (await report(dispatches(f).at(-1), f.io)).include;
  assert.equal(final.filter(r => r.label === pool.label).length, cases.length - 3);
  assert.ok(final.filter(r => r.label === pool.label).every(r => r.phase === 'bootstrap' && r.error === west.error));
  assert.ok(final.filter(r => r.label !== pool.label).every(r => !r.phase), 'Other pools retain capacity failures');
  assert.equal(f.claims.get(claimKey(cases[0])).state, 'admitted');
  assert.equal(f.claims.has(claimKey(cases[1])), false);
  assert.ok(!f.calls.some(c => c.operation === 'suspend-processes' && c.input.ScalingProcesses.includes('Launch') && c.input.ScalingProcesses.length === 1));
  f.io.now = () => f.deadline - 60;
  const again = {...poolEvent(f, f.deadline, 'us-west-2', 'i-00000101'), kind: 'bootstrap', error: west.error};
  f.owned[0].Tags = [];
  await assert.rejects(registration(again, f.io));
});
test('a boot report at the capacity deadline closes nothing', async () => {
  const f = await started('weekly', false, ['25B'], configure(1)), before = f.calls.length;
  const late = {...poolEvent(f, f.deadline, 'us-west-2', 'i-00000101'), kind: 'bootstrap', error: 'Bootstrap failed: GPU (exit 1)'};
  f.io.now = () => f.deadline;
  assert.equal(await registration(late, f.io), null);
  assert.deepEqual([f.calls.length, f.claims.has(closedKey(late.instance, 'us-west-2'))], [before, false]);
});
test('a boot report that closes the last region makes no GitHub call, and the next sweep without a VM fails the cases with its error', async () => {
  const f = await started('weekly', false, ['25B'], configure(1)), pool = f.selected.pools[0], event = f.sweeps[0];
  await sweep(event, f.io);
  const failed = {...poolEvent(f, f.deadline, 'us-west-2', 'i-00000101'), kind: 'bootstrap', error: 'Bootstrap failed: GPU (exit 1)'};
  const before = f.calls.length;
  assert.equal(await registration(failed, f.io), null);
  assert.ok(f.calls.slice(before).every(c => c.service) && !f.claims.has(failureKey(pool.instance)) && !dispatches(f).length);
  f.owned[0].Instances = [];
  await sweep(event, f.io);
  assert.equal(f.claims.get(failureKey(pool.instance)).error, failed.error);
  const rows = (await report(dispatches(f)[0], f.io)).include;
  assert.ok(rows.length === 11 && rows.every(r => r.label === pool.label && r.phase === 'bootstrap' && r.error === failed.error));
  assert.ok(event.cases.every(r => f.claims.get(claimKey(r)).error === failed.error), 'Every claim carries the reported error');
});
test('at the deadline a pool with an open region reports its zone errors and closed regions, not one boot failure', async () => {
  const f = await started(), event = f.sweeps[0];
  await sweep(event, f.io);
  const failed = {...poolEvent(f, f.deadline, 'us-west-2', 'i-00000101'), kind: 'bootstrap', error: 'Bootstrap failed: GPU (exit 1)'};
  assert.equal(await registration(failed, f.io), null);
  f.zones['us-east-2/subnet-4'] = {code: 'Unsupported'};
  f.io.now = () => f.deadline - 60;
  await sweep(event, f.io);
  f.io.now = () => f.deadline;
  await capacity(f.event, f.io);
  const row = f.event.cases.find(r => r.label === event.label);
  assert.deepEqual(f.claims.get(claimKey(row)), {state: 'capacity_failed', instance: event.instance, deadline: f.deadline,
    error: 'No runner capacity before tier deadline. Last zone errors: InsufficientInstanceCapacity in us-west-2a, us-west-2b, us-east-2a; ' +
      'Unsupported in us-east-2b. Closed regions: us-west-2 (Bootstrap failed: GPU (exit 1)).'});
  const rows = (await report(JSON.parse(f.calls.filter(c => c.path?.endsWith('/dispatches')).at(-1).data.inputs.payload), f.io)).include;
  assert.ok(rows.filter(r => r.label === event.label).every(r => !r.phase && r.error === f.claims.get(claimKey(row)).error));
});
test('a sweep counts every region, then launches only the missing VMs one zone at a time', async () => {
  const f = await started(), event = f.sweeps[0], [west, east] = f.owned;
  west.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'Terminating'}, {InstanceId: 'i-00000102', LifecycleState: 'Pending:Wait'});
  east.Instances.push({InstanceId: 'i-00000103', LifecycleState: 'Detached'}, {InstanceId: 'i-00000104', LifecycleState: 'Terminated'});
  west.DesiredCapacity = 1; f.zones['us-east-2/subnet-4'] = 5;
  const result = await sweep(event, f.io);
  assert.deepEqual([result.filled, result.waiting, result.booting, result.missing], [1, 11, 1, 1]);
  assert.deepEqual(result.answers.map(a => [a.region + '/' + a.subnet, a.requested, a.instances.length, a.error?.split(':')[0]]), [
    ['us-west-2/subnet-1', 1, 0, 'InsufficientInstanceCapacity'], ['us-west-2/subnet-2', 1, 0, 'InsufficientInstanceCapacity'],
    ['us-east-2/subnet-3', 1, 0, 'InsufficientInstanceCapacity'], ['us-east-2/subnet-4', 1, 1, undefined]]);
  assert.deepEqual([result.launched, result.position], [['i-00000001'], 0]);
  const home = Object.fromEntries(event.regions.flatMap(r => r.subnets.map(subnet => [subnet, r.region])));
  assert.ok(launches(f).every(c => c.input.AutoScalingGroupName === event.label && c.input.SubnetIds.length === 1 &&
    /^[0-9a-f]{64}$/.test(c.input.ClientToken) && c.region === home[c.input.SubnetIds[0]]));
  assert.equal(new Set(launches(f).map(c => c.input.ClientToken)).size, 4);
  assert.deepEqual(f.calls.filter(c => c.operation === 'describe-auto-scaling-groups').map(c => c.region), ['us-west-2', 'us-east-2']);
  const state = f.claims.get(stateKey(event.instance));
  assert.deepEqual(Object.keys(state.last_errors), ['us-west-2a', 'us-west-2b', 'us-east-2a']);
  assert.deepEqual([state.launched['i-00000001'], state.launched['i-00000102']],
    [{region: 'us-east-2', at: f.io.now()}, {region: 'us-west-2', at: f.io.now(), seen: true}]);
  assert.deepEqual(state.regions, event.regions);
  const record = f.claims.get('root/41/2/pools/g7e.8xlarge/sweeps/20261105T120000Z.json');
  assert.deepEqual([record.at, record.filled, record.waiting, record.booting, record.missing, record.answers, record.terminated, record.closed],
    ['2026-11-05T12:00:00.000Z', 1, 11, 1, 1, result.answers, [], []]);
  const again = await sweep(event, f.io);
  assert.deepEqual([again.filled, again.booting, again.missing, launches(f).length], [2, 2, 0, 4], 'Booting VMs cover the waiting cases');
  f.jobs.forEach(j => { j.status = 'in_progress'; });
  assert.deepEqual([(await sweep(event, f.io)).missing, launches(f).length], [0, 4]);
});
test('a sweep lowers a group above its live VMs before it asks for more', async () => {
  const f = await started(), event = f.sweeps[0], [west] = f.owned;
  west.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'InService'}); west.DesiredCapacity = 2;
  f.zones['us-west-2/subnet-1'] = 1;
  const result = await sweep(event, f.io);
  assert.deepEqual(f.calls.filter(c => c.operation === 'set-desired-capacity').map(c => [c.region, c.input]),
    [['us-west-2', {AutoScalingGroupName: event.label, DesiredCapacity: 1, HonorCooldown: false}]]);
  assert.ok(f.calls.findIndex(c => c.operation === 'set-desired-capacity') < f.calls.findIndex(c => c.operation === 'launch-instances'));
  assert.deepEqual([result.launched, west.DesiredCapacity], [['i-00000001'], 2]);
});
test('a sweep asks a region only for the room below its group maximum', async () => {
  const config = configure(); config[0].templates['g7e.8xlarge'].limit = 1;
  const f = await started('weekly', false, ['25B'], config), event = f.sweeps[0];
  f.zones['us-west-2/subnet-1'] = 5; f.zones['us-west-2/subnet-2'] = 5; f.zones['us-east-2/subnet-3'] = 5;
  const result = await sweep(event, f.io);
  assert.deepEqual(result.answers.map(a => [a.region, a.subnet, a.requested, a.instances.length]), [['us-west-2', 'subnet-1', 1, 1], ['us-east-2', 'subnet-3', 1, 1]]);
  const tight = configure(2, 2, ['g7e.8xlarge']);
  tight.forEach(r => { r.templates['g7e.8xlarge'].limit = 1; });
  const full = await started('weekly', false, ['25B'], tight, () => ({scratch_gib: 100, instance: 'g7e.8xlarge'}));
  full.owned.forEach((g, n) => { g.Instances.push({InstanceId: 'i-0000010' + n, LifecycleState: 'InService'}); g.DesiredCapacity = 1; });
  full.jobs.slice(0, 2).forEach(j => { j.status = 'in_progress'; }); full.zones['us-west-2/subnet-1'] = 5;
  const blocked = await sweep(full.sweeps[0], full.io);
  assert.deepEqual([full.sweeps[0].count, blocked.missing, launches(full).length], [3, 1, 0], 'A full group gets no call');
});
test('a zone answer moves to the next zone, and a rejected call or a region-wide answer skips the rest of the region', async () => {
  for (const [answer, zone, code] of [[0, 'us-west-2a', 'InsufficientInstanceCapacity'], [{code: 'Unsupported'}, 'us-west-2a', 'Unsupported'],
    [{code: 'InvalidSubnetID.NotFound', az: null}, 'us-west-2#1', 'InvalidSubnetID.NotFound'], [{code: 'VcpuLimitExceeded'}, 'us-west-2', 'VcpuLimitExceeded'],
    [{code: 'InstanceLimitExceeded'}, 'us-west-2', 'InstanceLimitExceeded'], [{code: 'UnauthorizedOperation'}, 'us-west-2', 'UnauthorizedOperation'],
    [{code: 'RequestLimitExceeded'}, 'us-west-2', 'RequestLimitExceeded'], ['AccessDenied', 'us-west-2', 'AccessDenied'],
    ['ValidationError', 'us-west-2', 'ValidationError'], ['Throttling', 'us-west-2', 'Throttling'],
    ['IdempotentParameterMismatchError', 'us-west-2', 'IdempotentParameterMismatchError']]) {
    const f = await started(), event = f.sweeps[0], wide = zone === 'us-west-2';
    f.zones['us-west-2/subnet-1'] = answer;
    const result = await sweep(event, f.io);
    assert.deepEqual(launches(f).map(c => c.input.SubnetIds[0]), ['subnet-1', ...(wide ? [] : ['subnet-2']), 'subnet-3', 'subnet-4']);
    assert.deepEqual([result.answers.length, result.answers[0].zone], [wide ? 3 : 4, zone]);
    const state = f.claims.get(stateKey(event.instance));
    assert.match(state.last_errors[zone], new RegExp('^' + code.replace('.', '\\.') + ': '));
    assert.ok(result.answers.every(a => !a.unknown) && !Object.keys(state.launched).length, 'A definite answer leaves no placeholder');
  }
});
test('a launch with an unknown outcome stops the walk and holds its VMs for ten minutes, even in a blind region', async () => {
  for (const answer of ['TimeoutError', 'ECONNRESET', 'InternalFailure', 'ResourceContentionFault', 'IdempotentCallInProgress',
    'IdempotentCallInProgressFault', {token: true}, sdkError('Throttling', 400, 3), sdkError('InternalFailure', 500, 1)]) {
    const f = await started(), event = f.sweeps[0], aws = f.io.aws, key = stateKey(event.instance), start = f.io.now();
    let now = start, blind = false;
    f.io.now = () => now;
    f.io.aws = async (service, operation, input, region) => {
      if (blind && operation === 'describe-auto-scaling-groups' && region === 'us-west-2')
        throw Object.assign(Error('Rate exceeded'), {name: 'Throttling', $metadata: {httpStatusCode: 400}});
      return aws(service, operation, input, region);
    };
    f.zones['us-west-2/subnet-1'] = answer; f.zones['us-west-2/subnet-2'] = 5; f.zones['us-east-2/subnet-3'] = 5;
    const first = await sweep(event, f.io), token = launches(f)[0].input.ClientToken;
    assert.deepEqual(launches(f).map(c => c.input.SubnetIds[0]), ['subnet-1'], 'The walk stops after an unknown outcome');
    assert.deepEqual([first.answers[0].unknown, first.answers[0].requested, first.launched], [true, 2, []]);
    const placeholder = {['unknown:' + token]: {region: 'us-west-2', at: start, count: 2}};
    assert.deepEqual(f.claims.get(key).launched, placeholder);
    const saves = f.calls.filter(c => c.operation === 'put-object' && c.input.Key === key).map(c => JSON.parse(c.input.Body));
    assert.deepEqual([saves.length, saves[1].launched, saves[1].lease_owner], [3, placeholder, 'owner-1'], 'The placeholder is saved at once');
    now = start + 60;
    const second = await sweep(event, f.io);
    assert.deepEqual([second.filled, second.missing, launches(f).length], [2, 0, 1], 'No other zone gets the same VMs');
    now = start + 300;
    assert.deepEqual([(await sweep(event, f.io)).filled, launches(f).length], [2, 1], 'The slots stay held after two minutes');
    now = start + 599; blind = true;
    const third = await sweep(event, f.io);
    assert.deepEqual([third.blind, third.filled, third.missing, launches(f).length], [['us-west-2'], 2, 0, 1]);
    now = start + 600; blind = false;
    const fourth = await sweep(event, f.io);
    assert.deepEqual([fourth.filled, fourth.closed, f.claims.get(key).launched['unknown:' + token]], [0, [], undefined]);
    assert.deepEqual(fourth.answers.map(a => a.subnet + ' ' + a.instances.length), ['subnet-2 2'], 'After ten minutes the walk goes on');
  }
});
test('only a 4xx answer in one SDK attempt is a definite rejection, and an error after SDK retries is an unknown outcome', async () => {
  for (const [error, rejection] of [[sdkError('ValidationError', 400, 1), true], [sdkError('AccessDenied', 403, undefined), true],
    [sdkError('Throttling', 400, 3), false], [sdkError('ValidationError', 400, 2), false], [sdkError('InternalFailure', 500, 1), false]]) {
    const f = await started(), event = f.sweeps[0], label = error.name + ' ' + error.$metadata.attempts;
    f.zones['us-west-2/subnet-1'] = error; f.zones['us-west-2/subnet-2'] = 5; f.zones['us-east-2/subnet-3'] = 5;
    const result = await sweep(event, f.io), {launched} = f.claims.get(stateKey(event.instance));
    assert.deepEqual(launches(f).map(c => c.input.SubnetIds[0]), rejection ? ['subnet-1', 'subnet-3'] : ['subnet-1'], label);
    assert.deepEqual([result.answers[0].unknown, result.answers[0].zone, result.answers[0].error], [rejection ? undefined : true, 'us-west-2',
      error.name + ': ' + error.name + ' from the fixture'], label);
    assert.deepEqual(Object.values(launched).map(e => e.count), rejection ? [undefined, undefined] : [2], label);
  }
});
test('a VM that describe shows late after an unknown outcome is counted once and never relaunched', async () => {
  for (const lag of [0, 150, 300, 590]) {
    const f = await started(), event = f.sweeps[0], aws = f.io.aws, key = stateKey(event.instance), start = f.io.now();
    let now = start, launched;
    f.io.now = () => now;
    f.io.aws = async (service, operation, input, region) => {
      const answer = await aws(service, operation, input, region);
      if (operation === 'launch-instances' && launched === undefined) {
        launched = now;
        throw Object.assign(Error('Connection timed out after 10000 ms'), {name: 'TimeoutError', $metadata: {attempts: 3}});
      }
      if (operation === 'describe-auto-scaling-groups' && now - launched < lag)
        return {AutoScalingGroups: answer.AutoScalingGroups.map(g => ({...g, Instances: []}))};
      return answer;
    };
    f.zones['us-west-2/subnet-1'] = 5; f.zones['us-west-2/subnet-2'] = 5; f.zones['us-east-2/subnet-3'] = 5;
    const first = await sweep(event, f.io);
    assert.deepEqual([first.answers.map(a => a.unknown), f.owned[0].Instances.length, f.owned[0].DesiredCapacity], [[true], 2, 2]);
    for (const offset of [60, 120, 300, 540, 590, 600, 660]) {
      now = start + offset;
      const result = await sweep(event, f.io), vms = f.owned.reduce((n, g) => n + g.Instances.length, 0);
      assert.deepEqual([result.filled, result.missing, result.launched, vms], [2, 0, [], 2], 'lag ' + lag + ' at ' + offset);
      assert.ok(!f.calls.some(c => c.operation === 'set-desired-capacity'), 'The desired capacity never drops below the hidden VMs');
      if (offset > lag) assert.deepEqual(Object.keys(f.claims.get(key).launched), ['i-00000001', 'i-00000002'], 'The placeholder gives way to the VMs');
    }
  }
});
test('a partial answer without an error holds a placeholder only for the VMs that it did not list', async () => {
  const f = await started(), event = f.sweeps[0], aws = f.io.aws, start = f.io.now();
  let now = start;
  f.io.now = () => now;
  f.io.aws = async (service, operation, input, region) => {
    const answer = await aws(service, operation, input, region);
    return operation === 'launch-instances' && input.SubnetIds[0] === 'subnet-1' ? {...answer, Errors: []} : answer;
  };
  f.zones['us-west-2/subnet-1'] = 1; f.zones['us-west-2/subnet-2'] = 5;
  const first = await sweep(event, f.io), token = launches(f)[0].input.ClientToken;
  assert.deepEqual([first.launched, first.answers.map(a => [a.requested, a.instances.length, a.unknown])], [['i-00000001'], [[2, 1, true]]]);
  assert.deepEqual(f.claims.get(stateKey(event.instance)).launched['unknown:' + token], {region: 'us-west-2', at: start, count: 1});
  now = start + 60;
  assert.deepEqual([(await sweep(event, f.io)).filled, launches(f).length], [2, 1], 'The VM it listed and its placeholder fill the pool');
});
test('a placeholder gives one slot back for each new VM that describe shows in its region, oldest placeholder first', async () => {
  const f = await started('weekly', false, ['25B'], configure(2), () => ({scratch_gib: 100, instance: 'g7e.8xlarge'})), event = f.sweeps[0];
  const [west] = f.owned, start = f.io.now(), key = stateKey(event.instance), errors = {'us-west-2': 'TimeoutError: from an earlier sweep',
    'us-west-2b': 'InsufficientInstanceCapacity: from an earlier sweep', 'us-east-2': 'TimeoutError: from an earlier sweep'};
  let now = start;
  f.io.now = () => now;
  seed(f, event.instance, {'unknown:new': {region: 'us-west-2', at: now - 60, count: 1}, 'unknown:old': {region: 'us-west-2', at: now - 120, count: 2},
    'unknown:east': {region: 'us-east-2', at: now - 300, count: 1}}, {last_errors: {...errors}});
  west.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'Pending'}, {InstanceId: 'i-00000102', LifecycleState: 'Pending'});
  west.DesiredCapacity = 3;
  const result = await sweep(event, f.io), {launched, last_errors} = f.claims.get(key);
  assert.deepEqual([result.filled, launched['unknown:old'], launched['unknown:new'].count, launched['unknown:east'].count], [4, undefined, 1, 1]);
  assert.deepEqual(last_errors, errors, 'The region error stays while a placeholder of its region holds a slot');
  west.Instances.push({InstanceId: 'i-00000103', LifecycleState: 'Pending'});
  now = start + 60;
  assert.equal((await sweep(event, f.io)).filled, 4);
  assert.deepEqual([Object.keys(f.claims.get(key).launched), f.claims.get(key).last_errors], [['unknown:east', 'i-00000101', 'i-00000102', 'i-00000103'],
    {'us-west-2b': errors['us-west-2b'], 'us-east-2': errors['us-east-2']}], 'Only the error of the region whose placeholders are used up goes');
  assert.deepEqual(f.calls.filter(c => c.operation === 'set-desired-capacity').map(c => c.input.DesiredCapacity), []);
});
test('a region-wide error wins over an earlier zone error in the same answer', async () => {
  const f = await started(), event = f.sweeps[0], aws = f.io.aws;
  f.io.aws = async (service, operation, input, region) => operation === 'launch-instances' && input.SubnetIds[0] === 'subnet-1' ?
    (f.calls.push({service, operation, input, region}), {Instances: [], Errors: [{ErrorCode: 'InsufficientInstanceCapacity', AvailabilityZone: 'us-west-2a',
      ErrorMessage: 'no capacity'}, {ErrorCode: 'VcpuLimitExceeded', ErrorMessage: 'vCPU limit'}]}) : aws(service, operation, input, region);
  const result = await sweep(event, f.io);
  assert.deepEqual([result.answers[0].error, result.answers[0].zone], ['VcpuLimitExceeded: vCPU limit', 'us-west-2']);
  assert.deepEqual(launches(f).map(c => c.input.SubnetIds[0]), ['subnet-1', 'subnet-3', 'subnet-4']);
});
test('a launch that returns VMs clears the last errors of its zone, its subnet label and its region', async () => {
  const old = code => code + ': from an earlier sweep';
  for (const [first, second, kept] of [[5, 0, {'us-west-2b': old('InsufficientInstanceCapacity'), 'us-east-2a': old('Unsupported')}],
    [1, 5, {'us-east-2a': old('Unsupported'), 'us-west-2a': 'InsufficientInstanceCapacity: We currently do not have sufficient capacity in us-west-2/subnet-1'}]]) {
    const f = await started(), event = f.sweeps[0];
    seed(f, event.instance, {}, {last_errors: {'us-west-2a': old('InsufficientInstanceCapacity'), 'us-west-2#1': old('InvalidSubnetID.NotFound'),
      'us-west-2': old('TimeoutError'), 'us-west-2b': old('InsufficientInstanceCapacity'), 'us-east-2a': old('Unsupported')}});
    f.zones['us-west-2/subnet-1'] = first; f.zones['us-west-2/subnet-2'] = second;
    assert.equal((await sweep(event, f.io)).launched.length, 2);
    assert.deepEqual(f.claims.get(stateKey(event.instance)).last_errors, kept, 'A partial answer keeps its own new zone error: ' + first);
  }
});
test('a sweep without a missing VM clears the saved zone errors but keeps the unread regions', async () => {
  const f = await started(), event = f.sweeps[0], aws = f.io.aws, start = f.io.now(), key = stateKey(event.instance);
  let now = start, blind = false;
  f.io.now = () => now;
  f.io.aws = async (service, operation, input, region) => {
    if (blind && operation === 'describe-auto-scaling-groups' && region === 'us-east-2')
      throw Object.assign(Error('User is not authorized'), {name: 'AccessDenied', $metadata: {httpStatusCode: 403}});
    return aws(service, operation, input, region);
  };
  await sweep(event, f.io);
  f.zones['us-west-2/subnet-1'] = 5; now = start + 60;
  const second = await sweep(event, f.io);
  assert.deepEqual(Object.keys(f.claims.get(key).last_errors), ['us-west-2b', 'us-east-2a', 'us-east-2b']);
  f.jobs.filter(j => j.labels.includes(event.label)).slice(0, 2).forEach((j, n) =>
    Object.assign(j, {status: 'in_progress', runner_name: runnerName(event.label, second.launched[n])}));
  blind = true; now = start + 3600;
  assert.deepEqual([(await sweep(event, f.io)).missing, f.claims.get(key).last_errors, f.claims.get(key).unread], [0, {}, {'us-east-2': 'AccessDenied'}]);
  now = f.deadline;
  await capacity(f.event, f.io);
  const row = event.cases.find(r => f.jobs.find(j => j.labels.includes(event.label) && j.name.endsWith('engine-e2e:' + r.case + ':' + r.scale)).status === 'queued');
  assert.equal(f.claims.get(claimKey(row)).error, 'No runner capacity before tier deadline. Last zone errors: AccessDenied in us-east-2.');
});
test('a sweep with more booting VMs than waiting cases also clears the saved zone errors', async () => {
  const f = await started('weekly', false, ['25B'], configure(1)), event = f.sweeps[0], start = f.io.now(), key = stateKey(event.instance);
  let now = start;
  f.io.now = () => now;
  f.zones['us-west-2/subnet-2'] = 5;
  const first = await sweep(event, f.io), own = f.jobs.filter(j => j.labels.includes(event.label));
  assert.deepEqual([first.launched.length, Object.keys(f.claims.get(key).last_errors)], [2, ['us-west-2a']]);
  own.slice(1).forEach(j => Object.assign(j, {status: 'completed'}));
  now = start + 60;
  const second = await sweep(event, f.io);
  assert.deepEqual([second.booting, second.waiting, second.missing, f.claims.get(key).last_errors], [2, 1, 0, {}]);
  now = f.deadline;
  await capacity(f.event, f.io);
  assert.equal(f.claims.get(claimKey(event.cases[0])).error, undefined);
});
test('the saved zone errors stay while a call with an unknown outcome holds the missing VMs', async () => {
  const f = await started('weekly', false, ['25B'], configure(1)), event = f.sweeps[0], key = stateKey(event.instance);
  let now = f.deadline - 1200;
  f.io.now = () => now;
  await sweep(event, f.io);
  assert.deepEqual(Object.keys(f.claims.get(key).last_errors), ['us-west-2a', 'us-west-2b']);
  f.zones['us-west-2/subnet-1'] = f.zones['us-west-2/subnet-2'] = 'TimeoutError';
  for (now += 60; now < f.deadline; now += 60) {
    const result = await sweep(event, f.io);
    assert.deepEqual([result.missing, Object.keys(f.claims.get(key).last_errors)], [result.answers.length ? 2 : 0,
      ['us-west-2a', 'us-west-2b', 'us-west-2']], (f.deadline - now) + ' s before the deadline');
  }
  assert.deepEqual(launches(f).map(c => c.input.SubnetIds[0]), ['subnet-1', 'subnet-2', 'subnet-1', 'subnet-2']);
  now = f.deadline;
  await capacity(f.event, f.io);
  assert.equal(f.claims.get(claimKey(event.cases[0])).error, 'No runner capacity before tier deadline. ' +
    'Last zone errors: InsufficientInstanceCapacity in us-west-2a, us-west-2b; TimeoutError in us-west-2.');
});
test('a call that shows only part of its VMs keeps the saved error of its region while it holds a slot', async () => {
  const f = await started('weekly', false, ['25B'], configure(1)), event = f.sweeps[0], aws = f.io.aws, key = stateKey(event.instance);
  let now = f.deadline - 300, lost = false;
  f.io.now = () => now;
  f.io.aws = async (service, operation, input, region) => {
    const answer = await aws(service, operation, input, region);
    if (operation === 'launch-instances' && input.SubnetIds[0] === 'subnet-2' && !lost) {
      lost = true;
      throw Object.assign(Error('Connection timed out after 10000 ms'), {name: 'TimeoutError', $metadata: {attempts: 3}});
    }
    return answer;
  };
  f.zones['us-west-2/subnet-2'] = 1;
  await sweep(event, f.io);
  for (now += 60; now < f.deadline; now += 60) {
    const result = await sweep(event, f.io), {launched, last_errors} = f.claims.get(key);
    assert.deepEqual([result.filled, result.missing, Object.values(launched).map(e => e.count), Object.keys(last_errors)],
      [2, 0, [1, undefined], ['us-west-2a', 'us-west-2']], (f.deadline - now) + ' s before the deadline');
  }
  now = f.deadline;
  await capacity(f.event, f.io);
  assert.equal(f.claims.get(claimKey(event.cases[0])).error, 'No runner capacity before tier deadline. ' +
    'Last zone errors: InsufficientInstanceCapacity in us-west-2a; TimeoutError in us-west-2.');
});
test('a placeholder that ends after ten minutes without VMs keeps the saved error of its region', async () => {
  const f = await started('weekly', false, ['25B'], configure(1)), event = f.sweeps[0], aws = f.io.aws, key = stateKey(event.instance);
  let now = f.deadline - 900, lost = false;
  f.io.now = () => now;
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'launch-instances' && input.SubnetIds[0] === 'subnet-2' && !lost) {
      lost = true;
      throw Object.assign(Error('Connection timed out after 10000 ms'), {name: 'TimeoutError', $metadata: {attempts: 3}});
    }
    return aws(service, operation, input, region);
  };
  await sweep(event, f.io);
  for (now += 60; now < f.deadline; now += 60) {
    const result = await sweep(event, f.io), ended = f.deadline - now <= 300;
    assert.deepEqual([result.filled, result.missing, Object.keys(f.claims.get(key).last_errors)],
      ended ? [0, 2, ['us-west-2a', 'us-west-2', 'us-west-2b']] : [2, 0, ['us-west-2a', 'us-west-2']], (f.deadline - now) + ' s before the deadline');
  }
  now = f.deadline;
  await capacity(f.event, f.io);
  assert.equal(f.claims.get(claimKey(event.cases[0])).error, 'No runner capacity before tier deadline. ' +
    'Last zone errors: InsufficientInstanceCapacity in us-west-2a, us-west-2b; TimeoutError in us-west-2.');
});
test('a call with an unknown outcome in a region that the sweep cannot read keeps its slots and the saved zone errors until a sweep reads it', async () => {
  const f = await started(), event = f.sweeps[0], aws = f.io.aws, start = f.io.now(), key = stateKey(event.instance);
  let now = start, blind = false, lost = false;
  f.io.now = () => now;
  f.io.aws = async (service, operation, input, region) => {
    if (blind && operation === 'describe-auto-scaling-groups' && region === 'us-east-2')
      throw Object.assign(Error('User is not authorized'), {name: 'AccessDenied', $metadata: {httpStatusCode: 403}});
    const answer = await aws(service, operation, input, region);
    if (operation === 'launch-instances' && region === 'us-east-2' && !lost) {
      lost = true;
      throw Object.assign(Error('Connection timed out after 10000 ms'), {name: 'TimeoutError', $metadata: {attempts: 3}});
    }
    return answer;
  };
  f.zones['us-west-2/subnet-1'] = 1; f.zones['us-east-2/subnet-3'] = 5;
  await sweep(event, f.io);
  const vms = f.owned.filter(g => g.AutoScalingGroupName === event.label).flatMap(g => g.Instances.map(i => i.InstanceId));
  f.jobs.filter(j => j.labels.includes(event.label)).slice(0, 2).forEach((j, n) =>
    Object.assign(j, {status: 'in_progress', runner_name: runnerName(event.label, vms[n])}));
  blind = true;
  for (const offset of [60, 660, 3600]) {
    now = start + offset;
    const result = await sweep(event, f.io);
    assert.deepEqual([result.blind, result.filled, result.missing, Object.keys(f.claims.get(key).last_errors)],
      [['us-east-2'], 2, 0, ['us-west-2a', 'us-west-2b', 'us-east-2']], offset + ' s after the call');
  }
  blind = false; now = start + 3660;
  const result = await sweep(event, f.io), state = f.claims.get(key);
  assert.deepEqual([vms.length, result.filled, result.missing, Object.keys(state.launched), state.last_errors], [2, 2, 0, vms, {}]);
});
test('launched VMs that describe does not show yet fill the pool, and the sweep clears the saved zone errors', async () => {
  const f = await started('weekly', false, ['25B'], configure(1)), event = f.sweeps[0], aws = f.io.aws, key = stateKey(event.instance);
  let now = f.io.now(), hide = false;
  f.io.now = () => now;
  f.io.aws = async (service, operation, input, region) => {
    const answer = await aws(service, operation, input, region);
    return hide && operation === 'describe-auto-scaling-groups' ? {AutoScalingGroups: answer.AutoScalingGroups.map(g => ({...g, Instances: []}))} : answer;
  };
  f.zones['us-west-2/subnet-2'] = 5;
  await sweep(event, f.io);
  assert.deepEqual(Object.keys(f.claims.get(key).last_errors), ['us-west-2a']);
  hide = true; now += 60;
  const result = await sweep(event, f.io);
  assert.deepEqual([result.filled, result.missing, f.claims.get(key).last_errors], [2, 0, {}]);
});
test('a new VM that takes a slot from a placeholder clears the saved error of its region', async () => {
  const config = configure(2, 2, ['g7e.8xlarge']);
  config[0].templates['g7e.8xlarge'].limit = 2;
  const f = await started('weekly', false, ['25B'], config, () => ({scratch_gib: 100, instance: 'g7e.8xlarge'})), event = f.sweeps[0];
  const aws = f.io.aws, start = f.io.now(), key = stateKey(event.instance);
  let now = start, timedOut = false;
  f.io.now = () => now;
  f.io.aws = async (service, operation, input, region) => {
    const answer = await aws(service, operation, input, region);
    if (operation === 'launch-instances' && !timedOut) {
      timedOut = true;
      throw Object.assign(Error('Connection timed out after 10000 ms'), {name: 'TimeoutError', $metadata: {attempts: 3}});
    }
    return answer;
  };
  f.zones['us-west-2/subnet-1'] = 2;
  const first = await sweep(event, f.io);
  assert.deepEqual([first.answers.map(a => [a.zone, a.unknown]), Object.keys(f.claims.get(key).last_errors)], [[['us-west-2', true]], ['us-west-2']]);
  now = start + 60;
  const second = await sweep(event, f.io);
  assert.deepEqual([second.filled, second.missing, second.answers.map(a => a.subnet)], [2, 1, ['subnet-3', 'subnet-4']]);
  assert.deepEqual([Object.keys(f.claims.get(key).launched), Object.keys(f.claims.get(key).last_errors)],
    [['i-00000001', 'i-00000002'], ['us-east-2a', 'us-east-2b']], 'The VMs of the unknown outcome showed, so its timeout is not the reason');
  const g = await started('weekly', false, ['25B'], config, () => ({scratch_gib: 100, instance: 'g7e.8xlarge'}));
  seed(g, event.instance, {}, {last_errors: {'us-east-2': 'VcpuLimitExceeded: from an earlier sweep'}});
  g.owned[1].Instances.push({InstanceId: 'i-00000101', LifecycleState: 'InService'}); g.owned[1].DesiredCapacity = 1;
  assert.deepEqual([(await sweep(g.sweeps[0], g.io)).filled, Object.keys(g.claims.get(key).last_errors)],
    [1, ['us-east-2', 'us-west-2a', 'us-west-2b', 'us-east-2a', 'us-east-2b']], 'A VM that no placeholder holds keeps the region error');
});
test('a new VM that takes the last slot of a placeholder keeps the saved error of a subnet label in its region', async () => {
  const config = configure(2, 2, ['g7e.8xlarge']);
  config[0].templates['g7e.8xlarge'].limit = 2;
  const f = await started('weekly', false, ['25B'], config, () => ({scratch_gib: 100, instance: 'g7e.8xlarge'})), event = f.sweeps[0];
  const aws = f.io.aws, key = stateKey(event.instance);
  let now = f.deadline - 120, lost = false;
  f.io.now = () => now;
  f.io.aws = async (service, operation, input, region) => {
    const answer = await aws(service, operation, input, region);
    if (operation === 'launch-instances' && input.SubnetIds[0] === 'subnet-2' && !lost) {
      lost = true;
      throw Object.assign(Error('Connection timed out after 10000 ms'), {name: 'TimeoutError', $metadata: {attempts: 3}});
    }
    return answer;
  };
  f.zones['us-west-2/subnet-1'] = {code: 'InsufficientInstanceCapacity', az: null};
  f.zones['us-west-2/subnet-2'] = 2;
  await sweep(event, f.io);
  assert.deepEqual(Object.keys(f.claims.get(key).last_errors), ['us-west-2#1', 'us-west-2']);
  now += 60;
  const result = await sweep(event, f.io);
  assert.deepEqual([result.filled, result.missing, result.answers.map(a => a.subnet), Object.keys(f.claims.get(key).last_errors)],
    [2, 1, ['subnet-3', 'subnet-4'], ['us-west-2#1', 'us-east-2a', 'us-east-2b']]);
  now = f.deadline;
  await capacity(f.event, f.io);
  assert.equal(f.claims.get(claimKey(event.cases[0])).error, 'No runner capacity before tier deadline. ' +
    'Last zone errors: InsufficientInstanceCapacity in us-west-2#1, us-east-2a, us-east-2b.');
});
test('each launch call uses a client token made of the run, the pool, the zone, the minute and the request', async () => {
  const f = await started(), event = f.sweeps[0], minute = Math.floor(f.io.now() / 60);
  f.zones['us-west-2/subnet-1'] = 1;
  await sweep(event, f.io);
  assert.deepEqual(launches(f).slice(0, 2).map(c => c.input.ClientToken), [['us-west-2', 'subnet-1', 2], ['us-west-2', 'subnet-2', 1]]
    .map(([region, subnet, requested]) => sha256(['41', '2', event.instance, region, subnet, minute, requested].join('/'))));
});
test('a sweep asks a group with VMs only for the room above its desired capacity', async () => {
  const config = configure(2, 2, ['g7e.8xlarge']);
  config[0].templates['g7e.8xlarge'].limit = 2;
  const f = await started('weekly', false, ['25B'], config, () => ({scratch_gib: 100, instance: 'g7e.8xlarge'})), event = f.sweeps[0], [west] = f.owned;
  west.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'InService'}); west.DesiredCapacity = 1;
  f.zones['us-west-2/subnet-1'] = 5;
  const result = await sweep(event, f.io);
  assert.deepEqual([event.count, west.MaxSize, result.missing, launches(f)[0].input.RequestedCapacity, result.answers[0].instances.length], [3, 2, 2, 1, 1]);
});
test('a group whose desired capacity is below its believed VMs gets no launch call', async () => {
  const config = configure(2, 2, ['g7e.8xlarge']);
  config[0].templates['g7e.8xlarge'].limit = 2;
  const f = await started('weekly', false, ['25B'], config, () => ({scratch_gib: 100, instance: 'g7e.8xlarge'})), event = f.sweeps[0], [west] = f.owned;
  seed(f, event.instance, {'unknown:held': {region: 'us-west-2', at: f.io.now() - 60, count: 2}});
  f.zones['us-west-2/subnet-1'] = 5; f.zones['us-west-2/subnet-2'] = 5; f.zones['us-east-2/subnet-3'] = 5;
  const result = await sweep(event, f.io);
  assert.deepEqual([west.MaxSize, west.DesiredCapacity, result.filled, result.missing], [2, 0, 2, 1]);
  assert.deepEqual([launches(f).map(c => c.region + '/' + c.input.SubnetIds[0]), result.launched], [['us-east-2/subnet-3'], ['i-00000001']],
    'The held slots fill the group, so only the other region is asked');
});
test('booting VMs count against the waiting cases, so a pool with one booting VM and one waiting case launches nothing', async () => {
  const f = await started('weekly', false, ['25B'], configure(1), () => ({scratch_gib: 100, instance: 'g7e.8xlarge'})), event = f.sweeps[0];
  f.owned[0].Instances.push({InstanceId: 'i-00000101', LifecycleState: 'Pending'}); f.owned[0].DesiredCapacity = 1;
  f.jobs.slice(1).forEach(j => { j.status = 'completed'; j.runner_id = 7; });
  f.zones['us-west-2/subnet-1'] = 5;
  const result = await sweep(event, f.io);
  assert.deepEqual([event.count, result.filled, result.waiting, result.booting, result.missing, launches(f).length], [3, 1, 1, 1, 0, 0]);
});
test('an unsupported zone before a zone with capacity in the same region does not hide that zone', async () => {
  const region = (name, subnets) => ({region: name, templates: Object.fromEntries(['g7e.8xlarge', 'g7e.24xlarge'].map(instance =>
    [instance, {template: 'lt-123', version: '2', subnets}]))});
  const f = await started('weekly', false, ['25B'], [region('us-east-1', ['subnet-1', 'subnet-2']), region('ap-south-1', ['subnet-3'])]);
  f.zones['us-east-1/subnet-1'] = {code: 'Unsupported'}; f.zones['us-east-1/subnet-2'] = 5;
  const result = await sweep(f.sweeps[0], f.io);
  assert.deepEqual(result.answers.map(a => [a.region, a.subnet, a.instances.length, a.zone]),
    [['us-east-1', 'subnet-1', 0, 'us-east-1a'], ['us-east-1', 'subnet-2', 2, undefined]]);
  assert.deepEqual([result.launched.length, result.position], [2, 2]);
});
test('each sweep stops after 45 seconds and the next one continues at the next zone', async () => {
  const f = await started(), event = f.sweeps[0], aws = f.io.aws; let now = f.io.now();
  f.io.now = () => now;
  f.io.aws = async (...args) => { if (args[1] === 'launch-instances') now += 20; return aws(...args); };
  const first = await sweep(event, f.io);
  assert.deepEqual([first.answers.map(a => a.subnet), first.position], [['subnet-1', 'subnet-2', 'subnet-3'], 3]);
  now += 60;
  const second = await sweep(event, f.io);
  assert.deepEqual([second.answers.map(a => a.subnet), second.position], [['subnet-4', 'subnet-1', 'subnet-2'], 2]);
  assert.deepEqual([launches(f)[0].input.SubnetIds, launches(f)[4].input.SubnetIds], [['subnet-1'], ['subnet-1']]);
  assert.notEqual(launches(f)[0].input.ClientToken, launches(f)[4].input.ClientToken, 'A new minute gets a new token');
});
test('a sweep launches nothing after the deadline, for a completed run or for an older attempt', async () => {
  const rerun = f => {
    f.run.run_started_at = new Date((f.io.now() - 60) * 1000).toISOString();
    return {...f.sweeps[0], attempt: '1', label: 'engine-e2e-41-1-' + f.sweeps[0].instance, deadline: f.deadline - 3600};
  };
  for (const change of [f => { f.io.now = () => f.deadline; }, f => { f.run.status = 'completed'; }, f => { f.run.run_attempt = 3; }, rerun]) {
    const f = await started(), before = f.calls.length;
    f.zones['us-west-2/subnet-1'] = 5;
    assert.deepEqual(await sweep(change(f) || f.sweeps[0], f.io), {skipped: 'stopped'});
    assert.ok(f.calls.slice(before).every(c => c.path === 'repos/NVIDIA/kumo-engine/actions/runs/41'), 'Only the source run was read');
  }
});
test('the walk stops at the capacity deadline', async () => {
  const f = await started(), event = f.sweeps[0], aws = f.io.aws, asked = [];
  let now = event.deadline - 30;
  f.io.now = () => now;
  f.io.aws = async (...args) => { if (args[1] === 'launch-instances') { asked.push(now); now += 20; } return aws(...args); };
  const result = await sweep(event, f.io);
  assert.deepEqual(asked, [event.deadline - 30, event.deadline - 10]);
  assert.deepEqual(result.answers.map(a => a.subnet), ['subnet-1', 'subnet-2']);
});
test('a region that closes during a sweep gets no launch in that sweep', async () => {
  const f = await started(), event = f.sweeps[0], aws = f.io.aws;
  f.zones['us-east-2/subnet-3'] = 5;
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'launch-instances' && region === 'us-west-2')
      f.claims.set(closedKey(event.instance, 'us-east-2'), {region: 'us-east-2', reason: 'Bootstrap failed: GPU (exit 1)', at: f.io.now()});
    return aws(service, operation, input, region);
  };
  const result = await sweep(event, f.io);
  assert.deepEqual([launches(f).map(c => c.region), result.launched], [['us-west-2', 'us-west-2'], []]);
  assert.equal(f.calls.filter(c => c.input?.Key === closedKey(event.instance, 'us-west-2')).length, 2, 'One more read before the first launch');
  assert.deepEqual([dispatches(f).length, f.claims.has(failureKey(event.instance))], [0, false], 'An open region can still serve the cases');
});
test('a pool without waiting cases is never failed at once', async () => {
  const f = await started(), event = f.sweeps[0];
  for (const region of ['us-west-2', 'us-east-2']) f.claims.set(closedKey(event.instance, region), {region, reason: 'Bootstrap failed: GPU (exit 1)', at: 0});
  f.jobs.forEach(j => { j.status = 'completed'; j.runner_id = 7; });
  assert.equal((await sweep(event, f.io)).waiting, 0);
  assert.deepEqual([dispatches(f).length, f.claims.has(failureKey(event.instance)), f.claims.get(stateKey(event.instance)).dispatched],
    [0, false, undefined]);
});
test('a slow closed-marker read right before a launch stops the walk at the 45-second budget or the deadline', async () => {
  for (const [offset, delay] of [[24 * 3600, 46], [30, 31]]) {
    const f = await started(), event = f.sweeps[0], aws = f.io.aws, key = closedKey(event.instance, 'us-west-2');
    let now = event.deadline - offset;
    f.io.now = () => now;
    f.io.aws = async (service, operation, input, region) => {
      if (operation === 'get-object' && input.Key === key && f.calls.some(c => c.input?.Key === key)) now += delay;
      return aws(service, operation, input, region);
    };
    f.zones['us-west-2/subnet-1'] = 5;
    const result = await sweep(event, f.io);
    assert.deepEqual([f.calls.filter(c => c.input?.Key === key).length, launches(f).length, result.answers, result.position], [2, 0, [], 0], String(delay));
  }
});
test('sweep events are validated against the trusted run and the owned groups', async () => {
  const f = await started(), event = f.sweeps[0];
  for (const bad of [{kind: 'capacity'}, {results: 's3://other/root'}, {sha: 'b'.repeat(40)}, {count: 0}, {count: 49}, {count: '2'},
    {label: 'engine-e2e-41-2-other'}, {instance: 'invalid'}, {regions: []}, {regions: [null]},
    {regions: [{region: 'us-west-2', subnets: ['subnet-1']}, {region: 'us-west-2', subnets: ['subnet-2']}]},
    {regions: [{region: 'mars', subnets: ['subnet-1']}]}, {regions: [{region: 'us-west-2', subnets: ['sg-1']}]},
    {regions: [{region: 'us-west-2', subnets: []}]}, {cases: []}, {cases: [{case: 'unknown', scale: '25B'}]},
    {cases: [{case: 'adtech_binary', scale: '1B'}]}, {deadline: f.deadline - 7 * 24 * 3600}])
    await assert.rejects(sweep({...event, ...bad}, f.io), /Invalid sweep event|Untrusted sweep event|Invalid Engine attempt/);
  f.zones['us-west-2/subnet-1'] = 5;
  for (const bad of [{deadline: event.deadline + 60}, {count: 48}]) await assert.rejects(sweep({...event, ...bad}, f.io), /Untrusted sweep event/);
  assert.ok(!f.calls.some(c => ['put-object', 'launch-instances'].includes(c.operation)), 'A group with other owner tags is never changed');
  f.run.head_branch = 'other';
  await assert.rejects(sweep(event, f.io), /Untrusted sweep event/);
});
test('a forged sweep event without the key or with a wrong key writes nothing and launches nothing', async () => {
  const f = await started(), [event, other] = f.sweeps, wrong = 'f'.repeat(64);
  f.zones['us-west-2/subnet-1'] = 5; f.zones['us-east-2/subnet-3'] = 5;
  for (const bad of [{key: undefined}, {key: ''}, {key: event.key.toUpperCase()}, {key: event.key + '0'}])
    await assert.rejects(sweep({...event, ...bad}, f.io), /Invalid sweep event/);
  for (const bad of [{key: wrong}, {key: other.key}, {key: wrong, regions: event.regions.slice(1)}, {key: wrong, count: 48},
    {key: wrong, regions: [{region: 'us-east-2', subnets: ['subnet-99']}]}])
    await assert.rejects(sweep({...event, ...bad}, f.io), /Untrusted sweep event/);
  assert.ok(!f.calls.some(c => ['put-object', 'launch-instances', 'set-desired-capacity', 'terminate-instance-in-auto-scaling-group'].includes(c.operation)));
  assert.deepEqual([f.claims.size, f.owned.reduce((n, g) => n + g.Instances.length, 0)], [0, 0]);
  const result = await sweep(event, f.io);
  assert.deepEqual([result.launched.length, f.claims.get(stateKey(event.instance)).regions], [2, event.regions], 'Only the real key writes the region list');
});
test('a region whose group lacks a subnet of the event is not owned and gets no change', async () => {
  const f = await started(), event = f.sweeps[0];
  f.zones['us-east-2/subnet-99'] = 5;
  const result = await sweep({...event, regions: [event.regions[0], {region: 'us-east-2', subnets: ['subnet-3', 'subnet-99']}]}, f.io);
  assert.deepEqual([result.blind, launches(f).map(c => c.region + '/' + c.input.SubnetIds[0])], [['us-east-2'], ['us-west-2/subnet-1', 'us-west-2/subnet-2']]);
  assert.equal(result.answers[0].error, 'AssertionError: Expected an owned runner group');
  await assert.rejects(sweep({...event, regions: [{region: 'us-east-2', subnets: ['subnet-99']}]}, f.io), /Untrusted sweep event/);
  assert.equal(f.calls.filter(c => c.operation === 'put-object' && c.input.Key === stateKey(event.instance)).length, 2,
    'Lease and release of the first sweep only');
});
test('a sweep counts only cases whose job exists and carries the pool label', async () => {
  const f = await started(), event = f.sweeps[0];
  f.zones['us-west-2/subnet-1'] = 5;
  const foreign = await sweep({...event, cases: [{case: 'fraud_binary', scale: '25B'}]}, f.io);
  assert.deepEqual([foreign.waiting, foreign.missing, launches(f).length], [0, 0, 0]);
  f.jobs.splice(0, f.jobs.length);
  const unlisted = await sweep(event, f.io);
  assert.deepEqual([unlisted.waiting, unlisted.missing, launches(f).length], [0, 0, 0]);
});
test('only one sweep of a pool works at a time', async () => {
  const f = await started(), event = f.sweeps[0], key = stateKey(event.instance);
  f.zones['us-west-2/subnet-1'] = 1;
  const results = await Promise.all([sweep(event, f.io), sweep(event, f.io)]);
  assert.deepEqual(results.map(r => r.skipped).sort(), ['leased', undefined]);
  assert.equal(launches(f).length, 4);
  const writes = f.calls.filter(c => c.operation === 'put-object' && c.input.Key === key).map(c => ({...c.input, Body: JSON.parse(c.input.Body)}));
  assert.deepEqual([writes[0].IfNoneMatch, writes[0].Body.lease_until, writes[0].Body.lease_owner], ['*', f.io.now() + 120, 'owner-1']);
  const saved = writes.find(w => w.IfMatch === '"1"');
  assert.deepEqual([Object.keys(saved.Body.launched), saved.Body.lease_owner], [['i-00000001'], 'owner-1'], 'The launch is saved under the lease');
  assert.deepEqual([writes.at(-1).IfMatch, writes.at(-1).Body.lease_until, writes.at(-1).Body.lease_owner], ['"2"', 0, null]);
  const state = f.claims.get(key);
  f.claims.set(key, {...state, lease_until: f.io.now() + 30, lease_owner: 'other'});
  assert.deepEqual(await sweep(event, f.io), {skipped: 'leased'});
  f.claims.set(key, {...state, lease_until: f.io.now() - 1, lease_owner: 'other'}); f.etags.set(key, '"seed"');
  assert.equal((await sweep(event, f.io)).filled, 1);
  assert.equal(f.calls.filter(c => c.operation === 'put-object' && c.input.Key === key).at(-2).input.IfMatch, '"seed"');
  const aws = f.io.aws;
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'put-object' && input.Key === key && input.IfMatch) f.etags.set(key, '"other"');
    return aws(service, operation, input, region);
  };
  assert.deepEqual(await sweep(event, f.io), {skipped: 'leased'}, 'A lost conditional write leaves the work to the other sweep');
});
test('launches are saved at once, so a lost release does not hide them from a blind region in the next sweep', async () => {
  const f = await started(), event = f.sweeps[0], key = stateKey(event.instance), aws = f.io.aws, start = f.io.now();
  let now = start, release = 'fail', blind = null;
  f.io.now = () => now;
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'put-object' && input.Key === key && JSON.parse(input.Body).lease_until === 0 && release === 'fail')
      throw Object.assign(Error('We encountered an internal error. Please try again.'), {name: 'InternalError', $metadata: {httpStatusCode: 500}});
    if (operation === 'describe-auto-scaling-groups' && region === blind)
      throw Object.assign(Error('Rate exceeded'), {name: 'Throttling', $metadata: {httpStatusCode: 400}});
    return aws(service, operation, input, region);
  };
  f.zones['us-west-2/subnet-1'] = 5; f.zones['us-east-2/subnet-3'] = 5;
  await assert.rejects(sweep(event, f.io), {name: 'InternalError'});
  const kept = f.claims.get(key);
  assert.deepEqual([Object.keys(kept.launched), kept.lease_until], [['i-00000001', 'i-00000002'], start + 120]);
  release = 'pass'; now = start + 121; blind = 'us-west-2';
  const next = await sweep(event, f.io);
  assert.deepEqual([next.blind, next.filled, next.missing, next.launched], [['us-west-2'], 2, 0, []]);
  assert.equal(f.owned.reduce((n, g) => n + g.Instances.length, 0), event.count, 'The pool stays at its VM count');
});
test('the final lease release is tried once more after an S3 409 conflict', async () => {
  for (const conflicts of [1, 2]) {
    const f = await started(), event = f.sweeps[0], key = stateKey(event.instance), aws = f.io.aws;
    let releases = 0;
    f.io.aws = async (service, operation, input, region) => {
      if (operation === 'put-object' && input.Key === key && JSON.parse(input.Body).lease_until === 0 && ++releases <= conflicts)
        throw Object.assign(Error('A conflicting conditional operation is currently in progress against this resource.'),
          {name: 'ConditionalRequestConflict', $metadata: {httpStatusCode: 409}});
      return aws(service, operation, input, region);
    };
    await sweep(event, f.io);
    assert.deepEqual([releases, f.claims.get(key).lease_until], [2, conflicts === 1 ? 0 : f.io.now() + 120], conflicts + ' conflicts');
  }
});
test('a lost write after a launch stops the walk', async () => {
  const f = await started(), event = f.sweeps[0], key = stateKey(event.instance), aws = f.io.aws;
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'put-object' && input.Key === key && input.IfMatch && JSON.parse(input.Body).lease_until) f.etags.set(key, '"other"');
    return aws(service, operation, input, region);
  };
  f.zones['us-west-2/subnet-1'] = 1; f.zones['us-west-2/subnet-2'] = 1;
  const result = await sweep(event, f.io);
  assert.deepEqual([launches(f).map(c => c.input.SubnetIds[0]), result.launched], [['subnet-1'], ['i-00000001']]);
});
test('a VM without a runner 15 minutes after its launch is terminated, and closes its region only while cases wait', async () => {
  for (const waiting of [true, false]) {
    const f = await started(), event = f.sweeps[0], [west, east] = f.owned, now = f.io.now();
    west.Instances.push(...['i-00000101', 'i-00000102'].map(InstanceId => ({InstanceId, LifecycleState: 'InService'}))); west.DesiredCapacity = 2;
    east.Instances.push(...['i-00000103', 'i-00000104', 'i-00000105'].map(InstanceId => ({InstanceId, LifecycleState: 'InService'})));
    east.DesiredCapacity = 3;
    f.claims.set(stateKey(event.instance), {position: 0, last_errors: {}, lease_until: 0, launched: {'i-00000101': {region: 'us-west-2', at: now - 900},
      'i-00000102': {region: 'us-west-2', at: now - 3600}, 'i-00000103': {region: 'us-east-2', at: now - 3600},
      'i-00000105': {region: 'us-east-2', at: now - 899}}});
    f.etags.set(stateKey(event.instance), '"seed"');
    Object.assign(f.jobs[0], {status: 'in_progress', runner_name: runnerName(event.label, 'i-00000102')});
    f.runners.push({id: 9, name: runnerName(event.label, 'i-00000103'), labels: [{name: event.label}]},
      {id: 10, name: event.label + '-i-00000101', labels: [{name: event.label}]});
    if (!waiting) f.jobs.forEach(j => { j.status = 'completed'; });
    const result = await sweep(event, f.io);
    assert.deepEqual(f.calls.filter(c => c.operation === 'terminate-instance-in-auto-scaling-group').map(c => [c.region, c.input]),
      [['us-west-2', {InstanceId: 'i-00000101', ShouldDecrementDesiredCapacity: true}]], 'A runner name with the instance ID is not its runner');
    assert.deepEqual([result.terminated, result.closed, result.filled, west.DesiredCapacity], [['i-00000101'], waiting ? ['us-west-2'] : [], 4, 1]);
    assert.deepEqual(f.claims.get(closedKey(event.instance, 'us-west-2')),
      waiting ? {region: 'us-west-2', reason: 'No runner 15 minutes after launch', at: now, instance: 'i-00000101'} : undefined);
    const state = f.claims.get(stateKey(event.instance));
    assert.equal(state.launched['i-00000104'].at, now, 'A VM seen for the first time starts its 15 minutes');
    assert.equal(state.launched['i-00000101'], undefined, 'A terminated VM is no longer believed live');
    assert.equal(f.calls.filter(c => c.path?.startsWith('repos/NVIDIA/kumo-engine/actions/runners?')).length, 2, 'One more read before the terminate');
    if (!waiting) continue;
    f.jobs.slice(1).forEach(j => { j.status = 'completed'; j.runner_id = 0; });
    f.io.now = () => f.deadline;
    await capacity(f.event, f.io);
    const error = 'No runner capacity before tier deadline. Closed regions: us-west-2 (No runner 15 minutes after launch).';
    assert.equal(f.claims.get(claimKey(event.cases[1])).error, error);
    const rows = (await report(JSON.parse(f.calls.find(c => c.path?.endsWith('/dispatches')).data.inputs.payload), f.io)).include;
    assert.equal(rows.find(r => r.case === event.cases[1].case).error, error);
  }
});
test('the 15-minute rule reads the runner list again and spares a VM whose runner registered meanwhile', async () => {
  const f = await started(), event = f.sweeps[0], [west] = f.owned, now = f.io.now(), api = f.io.api;
  west.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'InService'}); west.DesiredCapacity = 1;
  f.claims.set(stateKey(event.instance), {position: 0, last_errors: {}, lease_until: 0, launched: {'i-00000101': {region: 'us-west-2', at: now - 900}}});
  f.etags.set(stateKey(event.instance), '"seed"');
  let reads = 0;
  f.io.api = async (path, ...rest) => {
    const answer = await api(path, ...rest);
    if (!path.startsWith('repos/NVIDIA/kumo-engine/actions/runners?') || ++reads < 2) return answer;
    return {runners: [...answer.runners, {id: 9, name: runnerName(event.label, 'i-00000101'), labels: [{name: event.label}]}]};
  };
  const result = await sweep(event, f.io);
  assert.deepEqual([reads, result.terminated, result.closed, west.Instances.length], [2, [], [], 1]);
  assert.ok(!f.calls.some(c => c.operation === 'terminate-instance-in-auto-scaling-group'));
  assert.equal(f.claims.has(closedKey(event.instance, 'us-west-2')), false);
});
test('a VM without a runner after 15 minutes still closes its region when its terminate fails', async () => {
  const f = await started(), event = f.sweeps[0], [west] = f.owned, aws = f.io.aws, now = f.io.now();
  west.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'InService'}); west.DesiredCapacity = 1;
  seed(f, event.instance, {'i-00000101': {region: 'us-west-2', at: now - 900}});
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'terminate-instance-in-auto-scaling-group') throw sdkError('ScalingActivityInProgress', 400, 1);
    return aws(service, operation, input, region);
  };
  const result = await sweep(event, f.io);
  assert.deepEqual([result.terminated, result.closed, result.filled], [[], ['us-west-2'], 1]);
  assert.deepEqual(result.answers[0], {region: 'us-west-2', instance: 'i-00000101', error: 'ScalingActivityInProgress: ScalingActivityInProgress from the fixture'});
  assert.equal(f.claims.get(closedKey(event.instance, 'us-west-2')).reason, 'No runner 15 minutes after launch');
  assert.deepEqual(f.claims.get(stateKey(event.instance)).launched['i-00000101'], {region: 'us-west-2', at: now - 900, seen: true},
    'A VM that is still live keeps its launch time');
});
test('a terminated VM without a runner stays in the state until its close is saved, so a failed close closes the region in the next sweep', async () => {
  for (const listed of [true, false]) {
    const f = await started(), event = f.sweeps[0], [west] = f.owned, aws = f.io.aws, start = f.io.now(), marker = closedKey(event.instance, 'us-west-2');
    let now = start, fail = true;
    f.io.now = () => now;
    west.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'InService'}); west.DesiredCapacity = 1;
    seed(f, event.instance, {'i-00000101': {region: 'us-west-2', at: start - 900, seen: true}});
    f.io.aws = async (service, operation, input, region) => {
      if (fail && operation === 'put-object' && input.Key === marker)
        throw Object.assign(Error('We encountered an internal error. Please try again.'), {name: 'InternalError', $metadata: {httpStatusCode: 500, attempts: 3}});
      return aws(service, operation, input, region);
    };
    f.zones['us-west-2/subnet-1'] = 5; f.zones['us-east-2/subnet-3'] = 5;
    await assert.rejects(sweep(event, f.io), {name: 'InternalError'});
    assert.deepEqual([f.calls.filter(c => c.operation === 'terminate-instance-in-auto-scaling-group').length, launches(f).length, west.Instances], [1, 0, []]);
    assert.deepEqual(f.claims.get(stateKey(event.instance)).launched, {'i-00000101': {region: 'us-west-2', at: start - 900, seen: true}});
    if (listed) west.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'Terminating'});
    fail = false; now = start + 60;
    const next = await sweep(event, f.io);
    assert.deepEqual([next.closed, f.claims.get(marker), launches(f).map(c => c.region + '/' + c.input.SubnetIds[0])],
      [['us-west-2'], {region: 'us-west-2', reason: 'VM ended without a runner', at: now, instance: 'i-00000101'}, ['us-east-2/subnet-3']], String(listed));
    assert.equal(Object.hasOwn(f.claims.get(stateKey(event.instance)).launched, 'i-00000101'), false);
  }
});
test('a VM that ends before its runner registers closes its region only while cases wait', async () => {
  for (const [variant, closes] of [['no runner', true], ['ran a job', false], ['listed runner', false], ['no waiting case', false]]) {
    const f = await started(), event = f.sweeps[0], now = f.io.now();
    f.claims.set(stateKey(event.instance), {position: 0, last_errors: {}, lease_until: 0,
      launched: {'i-00000101': {region: 'us-west-2', at: now - 300, seen: true}, 'i-00000102': {region: 'us-east-2', at: now - 60}}});
    f.etags.set(stateKey(event.instance), '"seed"');
    f.zones['us-west-2/subnet-1'] = 5; f.zones['us-east-2/subnet-3'] = 5;
    if (variant === 'ran a job') Object.assign(f.jobs[0], {status: 'completed', runner_id: 7, runner_name: runnerName(event.label, 'i-00000101')});
    if (variant === 'listed runner') f.runners.push({id: 9, name: runnerName(event.label, 'i-00000101'), labels: [{name: event.label}]});
    if (variant === 'no waiting case') f.jobs.forEach(j => { j.status = 'completed'; });
    const result = await sweep(event, f.io);
    assert.deepEqual(f.claims.get(closedKey(event.instance, 'us-west-2')),
      closes ? {region: 'us-west-2', reason: 'VM ended without a runner', at: now, instance: 'i-00000101'} : undefined, variant);
    assert.deepEqual(result.closed, closes ? ['us-west-2'] : []);
    const state = f.claims.get(stateKey(event.instance));
    assert.deepEqual([state.launched['i-00000101'], state.launched['i-00000102']], [undefined, {region: 'us-east-2', at: now - 60}]);
    assert.ok(!f.claims.has(closedKey(event.instance, 'us-east-2')), 'A VM that describe has not shown yet is believed to boot for ten minutes');
    if (!closes) continue;
    assert.deepEqual(launches(f).map(c => c.region + '/' + c.input.SubnetIds[0]), ['us-east-2/subnet-3']);
    f.io.now = () => f.deadline;
    await capacity(f.event, f.io);
    assert.equal(f.claims.get(claimKey(event.cases[1])).error,
      'No runner capacity before tier deadline. Closed regions: us-west-2 (VM ended without a runner).');
  }
});
test('an ended VM stays in the state until its close decision is made, so a GitHub error does not lose it', async () => {
  const f = await started(), event = f.sweeps[0], key = stateKey(event.instance), start = f.io.now(), api = f.io.api, aws = f.io.aws;
  let now = start, fail = 'runners';
  f.io.now = () => now;
  seed(f, event.instance, {'i-00000101': {region: 'us-west-2', at: start - 300, seen: true}, 'i-00000102': {region: 'us-east-2', at: start - 60}});
  f.io.api = async (path, ...rest) => {
    if (fail === 'runners' && path.startsWith('repos/NVIDIA/kumo-engine/actions/runners?')) throw Error('GitHub request failed: 502');
    return api(path, ...rest);
  };
  f.io.aws = async (service, operation, input, region) => {
    if (fail === 'marker' && operation === 'put-object' && input.Key === closedKey(event.instance, 'us-west-2'))
      throw Object.assign(Error('We encountered an internal error. Please try again.'), {name: 'InternalError', $metadata: {httpStatusCode: 500}});
    return aws(service, operation, input, region);
  };
  f.zones['us-west-2/subnet-1'] = 5;
  for (const step of ['runners', 'marker']) {
    fail = step;
    await assert.rejects(sweep(event, f.io), step === 'runners' ? /GitHub request failed: 502/ : {name: 'InternalError'});
    assert.deepEqual([Object.keys(f.claims.get(key).launched), f.claims.get(key).lease_until, launches(f).length],
      [['i-00000101', 'i-00000102'], 0, 0], step);
    now += 60;
  }
  fail = null;
  const result = await sweep(event, f.io);
  assert.deepEqual([result.closed, Object.keys(f.claims.get(key).launched)], [['us-west-2'], ['i-00000102']]);
  assert.equal(f.claims.get(closedKey(event.instance, 'us-west-2')).reason, 'VM ended without a runner');
});
test('a known VM that describe lists in an ended state ends at once, whether or not a sweep saw it live', async () => {
  for (const state of ['Terminating', 'Terminating:Wait', 'Terminating:Proceed', 'Terminated', 'Detaching', 'Detached'])
    for (const seen of [true, false]) {
      const f = await started('weekly', false, ['25B'], configure(1)), event = f.sweeps[0], [west] = f.owned, now = f.io.now();
      seed(f, event.instance, {'i-00000101': {region: 'us-west-2', at: now - 60, ...(seen ? {seen} : {})}});
      west.Instances.push({InstanceId: 'i-00000101', LifecycleState: state}); west.DesiredCapacity = 1;
      const result = await sweep(event, f.io);
      assert.deepEqual([result.filled, result.closed, f.claims.get(stateKey(event.instance)).launched], [0, ['us-west-2'], {}], state + ' ' + seen);
      assert.deepEqual([f.claims.get(closedKey(event.instance, 'us-west-2')).reason, dispatches(f).length], ['VM ended without a runner', 0]);
      assert.deepEqual(f.calls.filter(c => c.operation === 'set-desired-capacity').map(c => c.input.DesiredCapacity), [0]);
    }
});
test('a VM that ended before any sweep saw it live keeps that evidence when its close decision fails', async () => {
  const f = await started(), event = f.sweeps[0], [west] = f.owned, api = f.io.api, start = f.io.now(), key = stateKey(event.instance);
  let now = start, fail = true;
  f.io.now = () => now;
  seed(f, event.instance, {'i-00000101': {region: 'us-west-2', at: start - 100}});
  west.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'Terminated'}); west.DesiredCapacity = 1;
  f.io.api = async (path, ...rest) => {
    if (fail && path.startsWith('repos/NVIDIA/kumo-engine/actions/runners?')) throw Error('GitHub request failed: 502');
    return api(path, ...rest);
  };
  f.zones['us-west-2/subnet-1'] = 5; f.zones['us-east-2/subnet-3'] = 5;
  await assert.rejects(sweep(event, f.io), /GitHub request failed: 502/);
  assert.deepEqual([f.claims.get(key).launched, launches(f).length], [{'i-00000101': {region: 'us-west-2', at: start - 100, seen: true}}, 0]);
  fail = false; now = start + 60; west.Instances = [];
  const next = await sweep(event, f.io);
  assert.deepEqual([next.closed, next.filled, f.claims.get(closedKey(event.instance, 'us-west-2')).reason], [['us-west-2'], 0, 'VM ended without a runner']);
  assert.deepEqual([launches(f).map(c => c.region + '/' + c.input.SubnetIds[0]), f.claims.get(key).launched['i-00000101']], [['us-east-2/subnet-3'], undefined]);
});
test('a region that is already closed keeps its reason when a VM ends or reaches 15 minutes there', async () => {
  const f = await started(), event = f.sweeps[0], [west] = f.owned, now = f.io.now(), marker = closedKey(event.instance, 'us-west-2');
  const reason = 'Bootstrap failed: runner registration (exit 31)';
  f.claims.set(marker, {region: 'us-west-2', reason, at: 0});
  west.Instances.push({InstanceId: 'i-00000102', LifecycleState: 'InService'}); west.DesiredCapacity = 1;
  seed(f, event.instance, {'i-00000101': {region: 'us-west-2', at: now - 300, seen: true}, 'i-00000102': {region: 'us-west-2', at: now - 900}});
  const result = await sweep(event, f.io);
  assert.deepEqual([result.terminated, result.closed, f.claims.get(marker).reason], [['i-00000102'], [], reason]);
  assert.ok(!f.calls.some(c => c.operation === 'put-object' && c.input.Key === marker), 'No second close of a closed region');
});
test('a close that meets a marker written by another writer keeps the stored reason in memory', async () => {
  const f = await started(), event = f.sweeps[0], [west, east] = f.owned, aws = f.io.aws, now = f.io.now();
  const local = 'Bootstrap failed: local SSD setup (exit 7)', gpu = 'Bootstrap failed: GPU (exit 1)';
  f.claims.set(closedKey(event.instance, 'us-east-2'), {region: 'us-east-2', reason: local, at: 0});
  seed(f, event.instance, {'i-00000101': {region: 'us-west-2', at: now - 300, seen: true}});
  let written = false;
  f.io.aws = async (service, operation, input, region) => {
    if (!written && operation === 'put-object' && input.Key === closedKey(event.instance, 'us-west-2')) {
      written = true;
      f.claims.set(input.Key, {region: 'us-west-2', reason: gpu, at: now});
    }
    return aws(service, operation, input, region);
  };
  assert.deepEqual([west.Instances, east.Instances], [[], []]);
  const result = await sweep(event, f.io);
  assert.deepEqual([result.closed, f.claims.get(closedKey(event.instance, 'us-west-2')).reason], [['us-west-2'], gpu]);
  assert.equal(f.claims.get(failureKey(event.instance)).error, gpu, 'The at-once rule uses the reason that S3 holds');
  assert.ok((await report(dispatches(f)[0], f.io)).include.every(r => r.error === gpu));
});
test('a close that meets an S3 409 without a stored marker writes the marker once more, and throws if that fails too', async () => {
  for (const conflicts of [1, 2]) {
    const f = await started(), event = f.sweeps[0], aws = f.io.aws, now = f.io.now(), marker = closedKey(event.instance, 'us-west-2');
    seed(f, event.instance, {'i-00000101': {region: 'us-west-2', at: now - 300, seen: true}});
    let puts = 0;
    f.io.aws = async (service, operation, input, region) => {
      if (operation === 'put-object' && input.Key === marker && ++puts <= conflicts)
        throw Object.assign(Error('A conflicting conditional operation is currently in progress against this resource.'),
          {name: 'ConditionalRequestConflict', $metadata: {httpStatusCode: 409}});
      return aws(service, operation, input, region);
    };
    if (conflicts === 2) await assert.rejects(sweep(event, f.io), {name: 'ConditionalRequestConflict'});
    else assert.deepEqual((await sweep(event, f.io)).closed, ['us-west-2']);
    assert.deepEqual([puts, f.claims.get(marker)?.reason], [2, conflicts === 1 ? 'VM ended without a runner' : undefined], String(conflicts));
    assert.equal(Object.hasOwn(f.claims.get(stateKey(event.instance)).launched, 'i-00000101'), conflicts === 2, 'A failed close keeps the VM');
  }
});
test('a reported boot failure that is not runner registration or configuration replaces a soft close reason of its region', async () => {
  const gpu = 'Bootstrap failed: GPU (exit 1)';
  for (const soft of ['Bootstrap failed: runner registration (exit 31)', 'VM ended without a runner', 'No runner 15 minutes after launch']) {
    const f = await started('weekly', false, ['25B'], configure(1)), event = f.sweeps[0], [west] = f.owned, now = f.io.now();
    const marker = closedKey(event.instance, 'us-west-2');
    if (soft.startsWith('Bootstrap')) await registration({...poolEvent(f, f.deadline, 'us-west-2', 'i-00000101'), kind: 'bootstrap', error: soft}, f.io);
    else {
      seed(f, event.instance, {'i-00000101': {region: 'us-west-2', at: now - (soft.startsWith('VM') ? 300 : 900), seen: true}});
      if (soft.startsWith('No')) { west.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'InService'}); west.DesiredCapacity = 1; }
      await sweep(event, f.io);
    }
    assert.equal(f.claims.get(marker).reason, soft);
    assert.equal(await registration({...poolEvent(f, f.deadline, 'us-west-2', 'i-00000102'), kind: 'bootstrap', error: gpu}, f.io), null);
    assert.deepEqual(f.claims.get(marker), {region: 'us-west-2', reason: gpu, at: now}, soft);
    west.Instances = [];
    await sweep(event, f.io);
    await sweep(event, f.io);
    assert.deepEqual([dispatches(f).length, f.claims.get(failureKey(event.instance)).error], [1, gpu], soft);
    const rows = (await report(dispatches(f)[0], f.io)).include;
    assert.ok(rows.length === 11 && rows.every(r => r.phase === 'bootstrap' && r.error === gpu), soft);
  }
});
test('a stored hard close reason is never replaced, and a soft reason never replaces a stored one', async () => {
  const gpu = 'Bootstrap failed: GPU (exit 1)', local = 'Bootstrap failed: local SSD setup (exit 7)';
  const registering = 'Bootstrap failed: runner registration (exit 31)', configuring = 'Bootstrap failed: runner configuration (exit 7)';
  for (const [first, later] of [[gpu, registering], [gpu, local], [registering, configuring]]) {
    const f = await started('weekly', false, ['25B'], configure(1)), marker = closedKey(f.sweeps[0].instance, 'us-west-2');
    for (const [id, error] of [['i-00000101', first], ['i-00000102', later]])
      assert.equal(await registration({...poolEvent(f, f.deadline, 'us-west-2', id), kind: 'bootstrap', error}, f.io), null);
    assert.equal(f.claims.get(marker).reason, first, first + ' then ' + later);
    assert.deepEqual(f.calls.filter(c => c.operation === 'put-object' && c.input.Key === marker).map(c => c.input.IfNoneMatch), ['*', '*']);
  }
});
test('a hard reason whose conditional replace fails throws and changes nothing, and the retried report keeps a hard reason that won', async () => {
  const registering = 'Bootstrap failed: runner registration (exit 31)', gpu = 'Bootstrap failed: GPU (exit 1)', local = 'Bootstrap failed: local SSD setup (exit 7)';
  for (const [race, kept] of [['conflict', gpu], ['other writer', local]]) {
    const f = await started('weekly', false, ['25B'], configure(1)), event = f.sweeps[0], aws = f.io.aws, marker = closedKey(event.instance, 'us-west-2');
    const late = () => registration({...poolEvent(f, f.deadline, 'us-west-2', 'i-00000102'), kind: 'bootstrap', error: gpu}, f.io);
    await registration({...poolEvent(f, f.deadline, 'us-west-2', 'i-00000101'), kind: 'bootstrap', error: registering}, f.io);
    const stored = f.claims.get(marker);
    let raced = false;
    f.io.aws = async (service, operation, input, region) => {
      if (!raced && operation === 'put-object' && input.Key === marker && input.IfMatch) {
        raced = true;
        if (race === 'conflict') throw Object.assign(Error('A conflicting conditional operation is currently in progress against this resource.'),
          {name: 'ConditionalRequestConflict', $metadata: {httpStatusCode: 409}});
        await aws(service, operation, {...input, Body: JSON.stringify({region: 'us-west-2', reason: local, at: 0})}, region);
      }
      return aws(service, operation, input, region);
    };
    await assert.rejects(late(), race === 'conflict' ? {name: 'ConditionalRequestConflict'} : /existing claim/);
    assert.deepEqual(f.claims.get(marker), race === 'conflict' ? stored : {region: 'us-west-2', reason: local, at: 0}, race);
    f.owned[0].Instances = [];
    await sweep(event, f.io);
    assert.equal(dispatches(f).length, race === 'conflict' ? 0 : 1, 'Only a stored hard reason fails the cases at once');
    assert.equal(await late(), null);
    f.owned[0].Instances = [];
    await sweep(event, f.io);
    assert.deepEqual([f.claims.get(marker).reason, dispatches(f).length, f.claims.get(failureKey(event.instance)).error], [kept, 1, kept], race);
  }
});
test('a region whose read fails stays blind: its believed VMs count and it gets no change', async () => {
  const f = await started(), event = f.sweeps[0], [, east] = f.owned, aws = f.io.aws, now = f.io.now(), before = f.calls.length;
  east.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'InService'}); east.DesiredCapacity = 2;
  f.claims.set(stateKey(event.instance), {position: 0, last_errors: {}, lease_until: 0, launched: {'i-00000101': {region: 'us-east-2', at: now - 3600}}});
  f.etags.set(stateKey(event.instance), '"seed"'); f.zones['us-west-2/subnet-1'] = 5;
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'describe-auto-scaling-groups' && region === 'us-east-2') throw Object.assign(Error('Rate exceeded'), {name: 'Throttling'});
    return aws(service, operation, input, region);
  };
  const result = await sweep(event, f.io);
  assert.deepEqual([result.filled, result.missing, result.launched, result.blind], [1, 1, ['i-00000001'], ['us-east-2']]);
  assert.ok(result.answers.some(a => a.region === 'us-east-2' && a.error === 'Throttling: Rate exceeded'));
  assert.ok(!f.calls.slice(before).some(c => c.region === 'us-east-2' && c.operation !== 'describe-auto-scaling-groups'), 'No terminate, reclaim or launch');
  assert.equal(f.owned.reduce((n, g) => n + g.Instances.length, 0), event.count);
  assert.deepEqual(f.claims.get(stateKey(event.instance)).launched['i-00000101'], {region: 'us-east-2', at: now - 3600});
});
test('a region that no sweep could read is named in the public capacity error until a sweep reads it', async () => {
  const f = await started('weekly', false, ['25B'], configure(4)), event = f.sweeps[0], aws = f.io.aws, key = stateKey(event.instance);
  const failing = {'us-east-2': Object.assign(Error('User is not authorized'), {name: 'AccessDenied', $metadata: {httpStatusCode: 403}}),
    'eu-west-2': Object.assign(Error('socket hang up'), {name: 'socket hang up'})};
  f.owned.find(g => g.region === 'us-east-1' && g.AutoScalingGroupName === event.label).Tags = [];
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'describe-auto-scaling-groups' && failing[region]) throw failing[region];
    return aws(service, operation, input, region);
  };
  const result = await sweep(event, f.io);
  assert.deepEqual([result.blind, f.claims.get(key).unread], [['us-east-2', 'us-east-1', 'eu-west-2'],
    {'us-east-2': 'AccessDenied', 'us-east-1': 'NotOwned', 'eu-west-2': 'Error'}]);
  f.io.now = () => f.deadline;
  await capacity(f.event, f.io);
  const error = f.claims.get(claimKey(f.event.cases.find(r => r.label === event.label))).error;
  assert.equal(error, 'No runner capacity before tier deadline. Last zone errors: InsufficientInstanceCapacity in us-west-2a, us-west-2b; ' +
    'AccessDenied in us-east-2; NotOwned in us-east-1; Error in eu-west-2.');
  f.io.now = () => f.deadline - 60;
  delete failing['us-east-2'];
  await sweep(event, f.io);
  assert.deepEqual(f.claims.get(key).unread, {'us-east-1': 'NotOwned', 'eu-west-2': 'Error'}, 'A region read again leaves the list');
  const g = await expired();
  g.claims.set(key, {last_errors: {'us-east-2': 'AccessDenied: x', 'us-west-2a': 'InsufficientInstanceCapacity: x'},
    unread: {'us-east-2': 'AccessDenied', 'mars': 'Error'}});
  await capacity(g.event, g.io);
  assert.equal(g.claims.get(claimKey(g.event.cases[0])).error, 'No runner capacity before tier deadline. Last zone errors: ' +
    'AccessDenied in us-east-2; InsufficientInstanceCapacity in us-west-2a.', 'A region is listed once per code, and only region names are listed');
});
test('a sweep that reads no owned region writes nothing and says why', async () => {
  for (const [errors, message] of [[{'us-west-2': 'Throttling', 'us-east-2': ''}, 'No region could be read: Throttling in us-west-2, Error in us-east-2'],
    [{'us-west-2': 'Throttling', 'us-east-2': null}, 'Untrusted sweep event']]) {
    const f = await started(), event = f.sweeps[0], aws = f.io.aws, before = f.calls.length;
    f.owned.find(g => g.region === 'us-east-2' && g.AutoScalingGroupName === event.label).Tags = [];
    f.io.aws = async (service, operation, input, region) => {
      if (operation === 'describe-auto-scaling-groups' && typeof errors[region] === 'string')
        throw Object.assign(Error('describe failed'), {name: errors[region]});
      return aws(service, operation, input, region);
    };
    await assert.rejects(sweep(event, f.io), {message});
    assert.ok(!f.calls.slice(before).some(c => c.operation === 'put-object'), 'Nothing is written');
  }
});
test('a VM that describe shows 150 seconds late still counts, and its 15 minutes start at its launch', async () => {
  const f = await started(), event = f.sweeps[0], [west] = f.owned, aws = f.io.aws, start = f.io.now();
  let now = start, lag = true;
  f.io.now = () => now;
  f.io.aws = async (service, operation, input, region) => {
    const answer = await aws(service, operation, input, region);
    return operation === 'describe-auto-scaling-groups' && lag ? {AutoScalingGroups: answer.AutoScalingGroups.map(g => ({...g, Instances: []}))} : answer;
  };
  f.zones['us-west-2/subnet-1'] = 5; f.zones['us-east-2/subnet-3'] = 5;
  assert.deepEqual((await sweep(event, f.io)).launched, ['i-00000001', 'i-00000002']);
  now = start + 150;
  const late = await sweep(event, f.io);
  assert.deepEqual([late.filled, late.missing, late.launched, launches(f).length, west.DesiredCapacity], [2, 0, [], 1, 2], 'No second launch');
  assert.ok(!f.calls.some(c => c.operation === 'set-desired-capacity'), 'No reclaim below the VMs');
  now = start + 160; lag = false;
  assert.deepEqual((await sweep(event, f.io)).filled, 2);
  assert.deepEqual(f.claims.get(stateKey(event.instance)).launched['i-00000001'], {region: 'us-west-2', at: start, seen: true});
  now = start + 900;
  assert.deepEqual((await sweep(event, f.io)).terminated, ['i-00000001', 'i-00000002'], 'The clock does not restart when describe shows the VM');
});
test('a seen VM that leaves describe ends at once, and a VM that describe never shows ends after ten minutes', async () => {
  for (const [entry, ends] of [[{at: -60, seen: true}, true], [{at: -599}, false], [{at: -600}, true]]) {
    const f = await started(), event = f.sweeps[0], [west] = f.owned, now = f.io.now();
    f.claims.set(stateKey(event.instance), {position: 0, last_errors: {}, lease_until: 0,
      launched: {'i-00000101': {region: 'us-west-2', ...entry, at: now + entry.at}}});
    f.etags.set(stateKey(event.instance), '"seed"'); west.DesiredCapacity = 1; f.zones['us-east-2/subnet-3'] = 5;
    const result = await sweep(event, f.io), launched = f.claims.get(stateKey(event.instance)).launched;
    assert.deepEqual([result.closed, result.filled, Object.hasOwn(launched, 'i-00000101')], [ends ? ['us-west-2'] : [], ends ? 0 : 1, !ends],
      JSON.stringify(entry));
    assert.deepEqual(f.claims.get(closedKey(event.instance, 'us-west-2')),
      ends ? {region: 'us-west-2', reason: 'VM ended without a runner', at: now, instance: 'i-00000101'} : undefined);
    assert.deepEqual(f.calls.filter(c => c.operation === 'set-desired-capacity').map(c => c.input.DesiredCapacity), ends ? [0] : []);
  }
});
test('a closed region keeps counting its VMs but gets no launch', async () => {
  const f = await started(), event = f.sweeps[0], [west] = f.owned;
  f.claims.set(closedKey(event.instance, 'us-west-2'), {region: 'us-west-2', reason: 'fixture', at: 0});
  west.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'InService'}); west.DesiredCapacity = 1;
  f.zones['us-west-2/subnet-1'] = 5; f.zones['us-east-2/subnet-3'] = 5;
  const result = await sweep(event, f.io);
  assert.deepEqual([result.filled, result.missing, launches(f).map(c => c.region + '/' + c.input.SubnetIds[0])], [1, 1, ['us-east-2/subnet-3']]);
});
test('when every region of a pool is closed with a boot failure and no VM is left, its waiting cases fail at once and only once', async () => {
  const f = await started(), event = f.sweeps[0], local = 'Bootstrap failed: local SSD setup (exit 7)';
  f.jobs.find(j => j.name.endsWith('engine-e2e:' + event.cases[0].case + ':' + event.cases[0].scale)).status = 'in_progress';
  f.claims.set(closedKey(event.instance, 'us-west-2'), {region: 'us-west-2', reason: 'No runner 15 minutes after launch', at: 0});
  f.claims.set(closedKey(event.instance, 'us-east-2'), {region: 'us-east-2', reason: local, at: 0});
  f.zones['us-west-2/subnet-1'] = 5;
  await sweep(event, f.io);
  await sweep(event, f.io);
  assert.equal(launches(f).length, 0);
  assert.equal(dispatches(f).length, 1);
  const [payload] = dispatches(f);
  assert.equal(payload.cases.length, event.cases.length - 1);
  assert.equal(payload.error, undefined, 'The error reaches the report through the saved claims only');
  const rows = (await report(payload, f.io)).include;
  assert.ok(rows.length === event.cases.length - 1 && rows.every(r => r.phase === 'bootstrap' && r.error === local), 'A soft reason is never the cause');
  assert.deepEqual(f.claims.get(failureKey(event.instance)), {sha: event.sha, deadline: event.deadline, instance: event.instance, error: local});
  assert.deepEqual([f.claims.get(stateKey(event.instance)).dispatched, f.claims.get(stateKey(event.instance)).lease_until], [true, 0]);
});
test('a failed at-once dispatch is tried again by the next sweep', async () => {
  const f = await started(), event = f.sweeps[0], api = f.io.api, gpu = 'Bootstrap failed: GPU (exit 1)';
  for (const region of ['us-west-2', 'us-east-2']) f.claims.set(closedKey(event.instance, region), {region, reason: gpu, at: 0});
  let failures = 1;
  f.io.api = async (path, ...rest) => {
    if (path.endsWith('/dispatches') && failures-- > 0) throw Error('GitHub request failed: 502');
    return api(path, ...rest);
  };
  await assert.rejects(sweep(event, f.io), /GitHub request failed: 502/);
  const state = f.claims.get(stateKey(event.instance));
  assert.deepEqual([state.dispatched, state.lease_until, dispatches(f).length, f.claims.get(failureKey(event.instance)).error], [undefined, 0, 0, gpu]);
  await sweep(event, f.io);
  await sweep(event, f.io);
  assert.deepEqual([dispatches(f).map(p => p.cases.length), f.claims.get(stateKey(event.instance)).dispatched], [[11], true]);
  assert.ok((await report(dispatches(f)[0], f.io)).include.every(r => r.phase === 'bootstrap' && r.error === gpu));
});
test('a pool failure write whose answer was lost and that meets its own object still fails the cases once', async () => {
  const f = await started(), event = f.sweeps[0], aws = f.io.aws, gpu = 'Bootstrap failed: GPU (exit 1)';
  for (const region of ['us-west-2', 'us-east-2']) f.claims.set(closedKey(event.instance, region), {region, reason: gpu, at: 0});
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'put-object' && input.Key === failureKey(event.instance)) {
      if (!f.claims.has(input.Key)) await aws(service, operation, input, region);
      throw Object.assign(Error('At least one of the pre-conditions you specified did not hold'), {name: 'PreconditionFailed',
        $metadata: {httpStatusCode: 412, attempts: 2}});
    }
    return aws(service, operation, input, region);
  };
  await sweep(event, f.io);
  await sweep(event, f.io);
  assert.deepEqual([dispatches(f).map(p => p.cases.length), f.claims.get(failureKey(event.instance)).error], [[11], gpu]);
});
test('a retried at-once dispatch keeps the first stored pool failure when the first hard reason changes', async () => {
  const f = await started(), event = f.sweeps[0], api = f.io.api, local = 'Bootstrap failed: local SSD setup (exit 7)', gpu = 'Bootstrap failed: GPU (exit 1)';
  await registration({...poolEvent(f, f.deadline, 'us-west-2', 'i-00000101'), kind: 'bootstrap', error: 'Bootstrap failed: runner registration (exit 31)'}, f.io);
  await registration({...poolEvent(f, f.deadline, 'us-east-2', 'i-00000102'), kind: 'bootstrap', error: local}, f.io);
  f.owned.forEach(g => { g.Instances = []; });
  let failures = 1;
  f.io.api = async (path, ...rest) => {
    if (path === 'repos/NVIDIA/kumo-engine/actions/runs/41' && f.claims.has(failureKey(event.instance)) && failures-- > 0) throw Error('GitHub request failed: 502');
    return api(path, ...rest);
  };
  await assert.rejects(sweep(event, f.io), /GitHub request failed: 502/);
  assert.deepEqual([f.claims.get(failureKey(event.instance)).error, f.claims.has(claimKey(event.cases[0])), dispatches(f).length], [local, false, 0]);
  await registration({...poolEvent(f, f.deadline, 'us-west-2', 'i-00000103'), kind: 'bootstrap', error: gpu}, f.io);
  assert.equal(f.claims.get(closedKey(event.instance, 'us-west-2')).reason, gpu, 'The first hard reason in region order is now another one');
  f.owned.forEach(g => { g.Instances = []; });
  await sweep(event, f.io);
  assert.deepEqual([dispatches(f).length, f.claims.get(failureKey(event.instance)).error], [1, local]);
  assert.ok((await report(dispatches(f)[0], f.io)).include.every(r => r.phase === 'bootstrap' && r.error === local), 'The claims keep the first stored error');
});
test('a live VM of the pool in any region holds off the at-once failure until it ends', async () => {
  const f = await started('nightly', false, ['100M', '1B']), event = f.sweeps[0], [west] = f.owned, now = f.io.now();
  const gpu = 'Bootstrap failed: GPU (exit 1)', own = f.jobs.filter(j => j.labels.includes(event.label));
  for (const region of ['us-west-2', 'us-east-2']) f.claims.set(closedKey(event.instance, region), {region, reason: gpu, at: 0});
  west.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'InService'}); west.DesiredCapacity = 1;
  seed(f, event.instance, {'i-00000101': {region: 'us-west-2', at: now - 3600, seen: true}});
  Object.assign(own[0], {status: 'completed', runner_id: 7, runner_name: runnerName(event.label, 'i-00000101')});
  Object.assign(own[1], {status: 'in_progress', runner_id: 7, runner_name: runnerName(event.label, 'i-00000101')});
  const busy = await sweep(event, f.io);
  assert.deepEqual([busy.filled, busy.waiting, busy.terminated, dispatches(f).length], [1, 20, [], 0], 'A healthy reused runner can take the cases');
  own[1].status = 'completed'; west.Instances = [];
  const ended = await sweep(event, f.io);
  assert.deepEqual([ended.filled, ended.closed, dispatches(f).map(p => p.cases.length)], [0, [], [20]]);
});
test('a known VM in a region that the sweep cannot read holds off the at-once failure', async () => {
  const f = await started('nightly', false, ['100M', '1B']), event = f.sweeps[0], [, east] = f.owned, aws = f.io.aws, now = f.io.now();
  for (const region of ['us-west-2', 'us-east-2']) f.claims.set(closedKey(event.instance, region), {region, reason: 'Bootstrap failed: GPU (exit 1)', at: 0});
  east.Instances.push({InstanceId: 'i-00000101', LifecycleState: 'InService'}); east.DesiredCapacity = 1;
  seed(f, event.instance, {'i-00000101': {region: 'us-east-2', at: now - 3600, seen: true}});
  Object.assign(f.jobs.find(j => j.labels.includes(event.label)), {status: 'completed', runner_id: 7, runner_name: runnerName(event.label, 'i-00000101')});
  f.io.aws = async (service, operation, input, region) => {
    if (operation === 'describe-auto-scaling-groups' && region === 'us-east-2')
      throw Object.assign(Error('Rate exceeded'), {name: 'Throttling', $metadata: {httpStatusCode: 400, attempts: 3}});
    return aws(service, operation, input, region);
  };
  const result = await sweep(event, f.io);
  assert.deepEqual([result.blind, result.filled, result.waiting, dispatches(f).length, f.claims.has(failureKey(event.instance))], [['us-east-2'], 1, 21, 0, false],
    'The healthy runner VM in the blind region can still take the cases');
});
test('regions closed only by runner errors or VMs without runners never fail the waiting cases at once', async () => {
  const f = await started('weekly', false, ['25B'], configure(3)), event = f.sweeps[0];
  const reasons = {'us-west-2': 'No runner 15 minutes after launch', 'us-east-2': 'VM ended without a runner',
    'us-east-1': 'Bootstrap failed: runner configuration (exit 7)'};
  for (const [region, reason] of Object.entries(reasons)) f.claims.set(closedKey(event.instance, region), {region, reason, at: 0});
  for (let k = 0; k < 3; k++) await sweep(event, f.io);
  assert.deepEqual([dispatches(f).length, f.claims.has(failureKey(event.instance)), launches(f).length], [0, false, 0]);
  f.io.now = () => f.deadline;
  await capacity(f.event, f.io);
  const row = f.event.cases.find(r => r.label === event.label);
  assert.deepEqual(f.claims.get(claimKey(row)), {state: 'capacity_failed', instance: event.instance, deadline: f.deadline,
    error: 'No runner capacity before tier deadline. Closed regions: us-west-2 (No runner 15 minutes after launch), ' +
      'us-east-2 (VM ended without a runner), us-east-1 (Bootstrap failed: runner configuration (exit 7)).'});
});
test('runner registration and configuration failures close their region but never fail the waiting cases at once', async () => {
  const single = await started('weekly', false, ['25B'], configure(1)), event = single.sweeps[0];
  await sweep(event, single.io);
  const registering = {...poolEvent(single, single.deadline, 'us-west-2', 'i-00000101'), kind: 'bootstrap',
    error: 'Bootstrap failed: runner registration (exit 31)'};
  assert.equal(await registration(registering, single.io), null);
  assert.equal(single.claims.get(closedKey(event.instance, 'us-west-2')).reason, registering.error);
  single.owned[0].Instances = [];
  assert.equal((await sweep(event, single.io)).filled, 0);
  assert.deepEqual([single.claims.has(failureKey(event.instance)), dispatches(single).length], [false, 0], 'Healthy runners can still take the cases');
  single.io.now = () => single.deadline;
  await capacity(single.event, single.io);
  assert.equal(single.claims.get(claimKey(single.event.cases[0])).error, 'No runner capacity before tier deadline. Last zone errors: ' +
    'InsufficientInstanceCapacity in us-west-2a, us-west-2b. Closed regions: us-west-2 (Bootstrap failed: runner registration (exit 31)).');
  const f = await started(), pool = f.sweeps[0];
  await sweep(pool, f.io);
  const gpu = {...poolEvent(f, f.deadline, 'us-west-2', 'i-00000101'), kind: 'bootstrap', error: 'Bootstrap failed: GPU (exit 1)'};
  const configuring = {...poolEvent(f, f.deadline, 'us-east-2', 'i-00000102'), kind: 'bootstrap', error: 'Bootstrap failed: runner configuration (exit 7)'};
  assert.equal(await registration(configuring, f.io), null);
  assert.equal(await registration(gpu, f.io), null);
  assert.deepEqual([f.claims.has(failureKey(pool.instance)), dispatches(f).length], [false, 0], 'A boot report never fails the cases itself');
  f.owned.forEach(g => { g.Instances = []; });
  await sweep(pool, f.io);
  assert.equal(f.claims.get(failureKey(pool.instance)).error, gpu.error);
  assert.deepEqual(dispatches(f).map(p => p.cases.length), [11], 'The sweep fails the cases because the GPU failure closed a region');
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
test('a failed command reaches the job log only through its masked error text', () => {
  const stderr = '\nAn error occurred (AccessDenied) when calling the CreateAutoScalingGroup operation: User: arn:aws:sts::123456789012:' +
    'assumed-role/engine-e2e-pool/GitHubActions is not authorized to perform: autoscaling:CreateAutoScalingGroup on resource: ' +
    'arn:aws:autoscaling:us-east-2:123456789012:autoScalingGroup:*:autoScalingGroupName/engine-e2e-41-2-g7e.8xlarge with lt-0123456789abcdef0, ' +
    'subnet-0a1b2c3d, sg-0123abcd, i-0123456789abcdef0, ami-0abc1234, vpc-1a2b3c4d, eni-0a1b2c3d4e5f60789 and vol-0123456789abcdef0 for key ' +
    'cd'.repeat(32) + ' at ' + 'a'.repeat(40) + '\n';
  const runs = [['aws', ['autoscaling', 'create-auto-scaling-group']], ['gh', ['api', 'repos/NVIDIA/kumo-engine/actions/runners/1', '--method', 'DELETE']],
    ['gh', ['api', 'repos/NVIDIA/kumo-engine/actions/runs/41', '--method', 'GET']]], results = [], streams = [];
  runInNewContext(readFileSync(join(__dirname, '../../.github/scripts/engine-e2e-pool.cjs'), 'utf8') + `
    for (const [file, args] of runs) try { results.push(command(file, args)); } catch (error) { results.push(error.message); }`, {
    runs, results, module: {exports: {}}, process: {env: {}}, require: name => name === 'node:child_process' ? {execFileSync: (file, args, options) => {
      streams.push(JSON.stringify(options.stdio));
      throw Object.assign(Error('Command failed'), {stderr: file === 'aws' ? stderr : 'gh: Not Found (HTTP 404)'});
    }} : require(name)
  });
  assert.deepEqual(streams, Array(3).fill('["pipe","pipe","pipe"]'), 'No raw stderr goes to the job log');
  assert.deepEqual(results, ['\nAn error occurred (AccessDenied) when calling the CreateAutoScalingGroup operation: User: arn:aws:sts::<account>:' +
    'assumed-role/engine-e2e-pool/GitHubActions is not authorized to perform: autoscaling:CreateAutoScalingGroup on resource: ' +
    'arn:aws:autoscaling:us-east-2:<account>:autoScalingGroup:*:autoScalingGroupName/engine-e2e-41-2-g7e.8xlarge with <id>, <id>, <id>, <id>, <id>, <id>, ' +
    '<id> and <id> for key <key> at ' + 'a'.repeat(40) + '\n', null, 'gh: Not Found (HTTP 404)']);
});
function broker(f, mode, autoscaling = {}, s3 = {}, pool = {registration, capacity, sweep}) {
  const handler = {}, calls = [], clients = [], signals = new Map(), secret = {app_id: 1, installation_id: 7,
    private_key: generateKeyPairSync('rsa', {modulusLength: 2048}).privateKey.export({type: 'pkcs8', format: 'pem'})};
  const responses = {
    '/app/installations/7/access_tokens': {token: 'installation-token'},
    '/repos/NVIDIA/kumo-engine/actions/runs/41': f.run,
    '/repos/NVIDIA/kumo-engine/actions/runners/registration-token': {token: 'fresh-registration'},
    '/repos/NVIDIA/kumo-engine/actions/runs/41/attempts/2/jobs': {jobs: f.jobs},
    '/repos/NVIDIA/kumo-engine/actions/workflows/engine-e2e-lifecycle.yml/dispatches': null
  };
  const command = name => class { constructor(input) { this.name = name; this.input = input; } };
  runInNewContext(readFileSync(join(__dirname, '../../.github/scripts/engine-e2e-capacity.cjs'), 'utf8'), {
    exports: handler, Buffer, process: {env: {...f.io.env, RUNNER_SECRET: 'registration-secret'}},
    AbortSignal: {timeout: ms => { const signal = AbortSignal.timeout(ms); signals.set(signal, ms); return signal; }},
    require: name => name.endsWith('engine-e2e-pool.cjs') ? pool :
      name === '@aws-sdk/client-secrets-manager' ? {
        SecretsManagerClient: class {
          constructor(options) { clients.push(['secrets', options?.region, options?.requestHandler]); }
          async send() { assert.notEqual(mode, true); return {SecretString: JSON.stringify(secret)}; }
        }, GetSecretValueCommand: class {}
      } :name === '@aws-sdk/client-s3/package.json' ? {version: s3.version || '3.934.0'} : name === '@aws-sdk/client-s3' ? {
        S3Client: class {
          constructor(options) { clients.push(['s3', options.region, options.requestHandler]); }
          async send({name, input}) {
            if (name === 'put-object') {
              const status = s3.put?.(input);
              if (status) throw Object.assign(Error('A conflicting conditional operation is currently in progress against this resource.'),
                {name: 'ConditionalRequestConflict', $metadata: {httpStatusCode: status}});
              try { return await f.io.aws('s3', 'put-object', input); } catch (error) { throw Object.assign(error, {$metadata: {httpStatusCode: 412}}); }
            }
            if (name === 'list-objects-v2') {
              f.calls.push({operation: 'list-objects-v2', input});
              return {Contents: mode === 404 || (typeof mode !== 'number' && !f.claims.has(input.Prefix)) ? [] : [{Key: input.Prefix}]};
            }
            f.calls.push({operation: 'GetObject', input});
            if (typeof mode === 'number' || !f.claims.has(input.Key))
              throw Object.assign(Error('S3 read failed'), {name: 'AccessDenied', $metadata: {httpStatusCode: typeof mode === 'number' ? mode : 403}});
            return {Body: {transformToString: async () => JSON.stringify(f.claims.get(input.Key))}, ETag: f.etags.get(input.Key)};
          }
        },
        PutObjectCommand: command('put-object'), GetObjectCommand: command('get-object'), ListObjectsV2Command: command('list-objects-v2')
      } : name === '@aws-sdk/client-auto-scaling' ? {
        AutoScalingClient: class {
          constructor(options) { this.region = options.region; clients.push(['autoscaling', options.region, options.requestHandler]); }
          async send({name, input}) { return f.io.aws('autoscaling', name, input, this.region); }
        },
        DescribeAutoScalingGroupsCommand: command('describe-auto-scaling-groups'), LaunchInstancesCommand: command('launch-instances'),
        SetDesiredCapacityCommand: command('set-desired-capacity'),
        TerminateInstanceInAutoScalingGroupCommand: command('terminate-instance-in-auto-scaling-group'), ...autoscaling
      } : require(name),
    fetch: async (url, options) => {
      calls.push(options);
      const path = new URL(url).pathname; assert.ok(Object.hasOwn(responses, path), 'Unexpected GitHub endpoint');
      return {ok: true, status: responses[path] === null ? 204 : 200, json: async () => responses[path]};
    }
  });
  return {handler: handler.handler, calls, clients, signals};
}
function sweepEvent(f) {
  const pool = f.selected.pools[0], deadline = Math.floor(Date.now() / 1000) + 3600, key = 'ab'.repeat(32);
  f.owned.forEach(g => { g.Tags.find(t => t.Key === 'EngineE2EDeadline').Value = String(deadline); g.Tags.push({Key: 'EngineE2EKey', Value: sha256(key)}); });
  return {kind: 'sweep', repo: 'NVIDIA/kumo-engine', run: '41', attempt: '2', sha: f.run.head_sha, deadline, results: f.io.env.RESULTS_S3,
    label: pool.label, instance: pool.instance, count: pool.count, key, cases: f.selected.matrix.include.filter(r => r.label === pool.label)
      .map(r => ({case: r.case, scale: r.scale})), regions: pool.regions.map(r => ({region: r.region, subnets: r.subnets}))};
}
test('the broker refuses to load without LaunchInstances support', () => {
  assert.throws(() => broker(fixture(), false, {LaunchInstancesCommand: undefined}), /client-auto-scaling 3\.934\.0 or later/);
});
test('the broker refuses to load with a client-s3 older than 3.934.0', () => {
  for (const version of ['3.933.0', '3.637.0', '2.1692.0'])
    assert.throws(() => broker(fixture(), false, {}, {version}),
      /^AssertionError.*: Bundle @aws-sdk\/client-auto-scaling and @aws-sdk\/client-s3 from one SDK release, 3\.934\.0 or later$/);
  for (const version of ['3.934.0', '3.1174.0', '4.0.0']) assert.equal(typeof broker(fixture(), false, {}, {version}).handler, 'function');
});
test('the broker lists before it reads, so S3 403 for a missing key is not a failure, and a 403 for a present key is', async () => {
  const f = fixture(), {handler} = broker(f, 'sweep'), event = sweepEvent(f), key = stateKey(event.instance);
  f.zones['us-east-2/subnet-3'] = 2;
  assert.equal((await handler(event)).launched.length, 2);
  const reads = f.calls.filter(c => ['list-objects-v2', 'GetObject'].includes(c.operation) && [c.input.Key, c.input.Prefix].includes(key));
  assert.deepEqual(reads.map(c => [c.operation, c.input.MaxKeys]), [['list-objects-v2', 1]], 'No GetObject for the missing first state');
  assert.equal((await handler(event)).filled, 2);
  const g = fixture();
  g.claims.set(key, {lease_until: 0}); g.etags.set(key, '"seed"');
  await assert.rejects(broker(g, 403).handler(sweepEvent(g)), {name: 'AccessDenied', message: 'S3 read failed'});
});
test('the broker treats an S3 409 conflict as a lost condition for the lease and the closed marker', async () => {
  const f = fixture(), event = sweepEvent(f), key = stateKey(event.instance);
  const {handler} = broker(f, 'sweep', {}, {put: input => input.Key === key && input.IfNoneMatch === '*' ? 409 : 0});
  assert.deepEqual(await handler(event), {skipped: 'leased'});
  assert.equal(launches(f).length, 0);
  const g = fixture(), boot = {...poolEvent(g, Math.floor(Date.now() / 1000) + 60), kind: 'bootstrap', error: 'Bootstrap failed: GPU (exit 1)'};
  g.owned[0].Tags.find(t => t.Key === 'EngineE2EDeadline').Value = String(boot.deadline);
  let puts = 0;
  const closing = broker(g, false, {}, {put: input => input.Key.includes('/closed/') && !puts++ ? 409 : 0});
  assert.equal(await closing.handler(boot), null);
  assert.deepEqual([puts, g.claims.get(closedKey(boot.instance, boot.region)).reason], [2, boot.error], 'A 409 without a stored marker writes again');
});
test('the broker refuses AWS operations outside its allowlist', async () => {
  const f = fixture(), refused = [];
  const probe = async (event, io) => {
    for (const [service, operation] of [['autoscaling', 'suspend-processes'], ['autoscaling', 'delete-auto-scaling-group'],
      ['s3', 'delete-object'], ['ec2', 'run-instances']])
      await io.aws(service, operation, {AutoScalingGroupName: event.label}, 'us-west-2').catch(error => refused.push(error.message));
    return 'probed';
  };
  const {handler} = broker(f, 'sweep', {}, {}, {registration: probe, capacity: probe, sweep: probe});
  assert.equal(await handler(sweepEvent(f)), 'probed');
  assert.deepEqual(refused, ['Unsupported Auto Scaling operation suspend-processes', 'Unsupported Auto Scaling operation delete-auto-scaling-group',
    'Unsupported AWS operation delete-object', 'Unsupported AWS operation run-instances']);
  assert.ok(!f.calls.some(c => ['suspend-processes', 'delete-auto-scaling-group'].includes(c.operation)));
});
for (const mode of [false, true, 403, 404, 'sweep'])
test('broker preserves authentication, routing, regional clients and S3 failure boundaries: ' + mode, async () => {
  const f = fixture(), {handler, calls, clients, signals} = broker(f, mode);
  const bounded = () => assert.ok(calls.length && calls.every(c => signals.get(c.signal) === 15000), 'Every GitHub call stops after 15 seconds');
  if (typeof mode === 'number') {
    const event = {repo: f.io.env.GITHUB_REPOSITORY, run: '41', attempt: '2', sha: f.run.head_sha,
      results: f.io.env.RESULTS_S3, deadline: Date.parse('2026-11-06T12:00:00Z') / 1000, cases: f.selected.matrix.include};
    if (mode === 403) await assert.rejects(handler(event), /S3 read failed/);
    else await handler(event);
    assert.equal(f.claims.size, mode === 404 ? event.cases.length : 0);
    assert.ok(f.calls.some(c => c.operation === 'list-objects-v2' && c.input.Prefix.endsWith('.admission.json') && c.input.MaxKeys === 1));
    return;
  }
  if (mode === 'sweep') {
    const event = sweepEvent(f), pool = f.selected.pools[0];
    f.zones['us-east-2/subnet-3'] = 2;
    await assert.rejects(handler({...event, results: 's3://untrusted/root'}), /configured result root/);
    assert.equal((await handler(event)).launched.length, 2);
    assert.equal((await handler(event)).filled, 2);
    assert.deepEqual(clients.map(([service, region, handler]) => [service, region, JSON.stringify(handler)]),
      [['secrets', undefined], ['s3', undefined], ['autoscaling', 'us-west-2'], ['autoscaling', 'us-east-2']]
        .map(c => [...c, '{"requestTimeout":10000,"throwOnRequestTimeout":true}']),
      'One client per service and region, and no request waits longer than 10 seconds');
    assert.deepEqual(launches(f).map(c => c.region), ['us-west-2', 'us-west-2', 'us-east-2']);
    const leases = f.calls.filter(c => c.operation === 'put-object' && c.input.Key === stateKey(pool.instance)).map(c => c.input);
    assert.deepEqual(leases.map(c => c.IfNoneMatch || c.IfMatch), ['*', '"1"', '"2"', '"4"', '"5"']);
    assert.deepEqual(leases.map(c => /^[0-9a-f-]{36}$/.test(JSON.parse(c.Body).lease_owner)), [true, true, false, true, false]);
    bounded();
    return;
  }
  if (mode) {
    const event = {...poolEvent(f, Math.floor(Date.now() / 1000) + 60),
      kind: 'bootstrap', error: 'Bootstrap failed: runner registration (exit 31)', results: 's3://untrusted/root'};
    f.owned[0].Tags.find(t => t.Key === 'EngineE2EDeadline').Value = String(event.deadline);
    for (const bad of [{sha: 'b'.repeat(40)}, {deadline: event.deadline + 1}, {instance_id: 'i-00000098'}])
      await assert.rejects(handler({...event, ...bad}));
    assert.equal(await handler(event), null);
    assert.equal(calls.length, 0);
    assert.deepEqual([...f.claims.keys()], [closedKey('g7e.8xlarge', 'us-west-2')]);
    assert.equal([...f.claims.values()][0].reason, event.error);
    assert.ok(!f.calls.some(c => c.operation === 'suspend-processes'));
    assert.ok(f.calls.filter(c => c.operation === 'put-object').every(c => c.input.Bucket === 'results'));
    assert.deepEqual(clients.filter(([service]) => service === 'autoscaling').map(([service, region, handler]) => [service, region,
      JSON.stringify(handler)]), [['autoscaling', 'us-west-2', '{"requestTimeout":10000,"throwOnRequestTimeout":true}']]);
    return;
  }
  assert.equal(await handler({kind: 'registration', ...poolEvent(f, Date.now() / 1000 + 60, 'us-east-2')}), 'fresh-registration');
  assert.deepEqual(JSON.parse(calls[0].body), {repositories: ['kumo-engine'], permissions: {administration: 'write', actions: 'write'}});
  assert.ok(calls.slice(1).every(c => c.headers.Authorization === 'Bearer installation-token'));
  assert.deepEqual(f.calls.filter(c => c.operation === 'describe-auto-scaling-groups').map(c => c.region), ['us-east-2']);
  bounded();
});
