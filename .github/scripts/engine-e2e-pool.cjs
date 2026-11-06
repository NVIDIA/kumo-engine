// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const {readFileSync, appendFileSync} = require('node:fs');
const {join} = require('node:path');
const {setTimeout: sleep} = require('node:timers/promises');
const CASES = ['adtech_binary', 'delivery_multiclass', 'delivery_multilabel', 'delivery_multilabel_ranking',
  'delivery_regression', 'delivery_forecasting', 'commerce_static_link', 'commerce_temporal_link',
  'streaming_temporal_link', 'community_regression', 'payments_binary', 'fraud_binary'];
function identity(env) {
  const repo = env.GITHUB_REPOSITORY, run = env.POOL_RUN_ID || env.GITHUB_RUN_ID;
  const attempt = env.POOL_ATTEMPT || env.GITHUB_RUN_ATTEMPT;
  assert(repo === 'NVIDIA/kumo-engine' && [run, attempt].every(v => /^[1-9][0-9]*$/.test(v)), 'Invalid Engine attempt');
  return {repo, run, attempt, label: 'engine-e2e-' + run + '-' + attempt};
}
function trusted(run, repo) {
  return repo === 'NVIDIA/kumo-engine' && run.head_repository?.full_name === repo && run.head_branch === 'main' &&
    ['schedule', 'workflow_dispatch'].includes(run.event) && ['nightly', 'weekly'].some(tier =>
      run.name === 'Kumo Engine ' + tier && run.path?.split('@')[0] === '.github/workflows/kumo-engine-' + tier + '.yml');
}
function plan(tier, scales, templates, limit, requirement, env = process.env) {
  assert(['nightly', 'weekly'].includes(tier) && Number.isSafeInteger(limit) && limit > 0 && limit <= 48 &&
    Array.isArray(scales) && scales.length && new Set(scales).size === scales.length && scales.every(s =>
      (tier === 'nightly' ? ['100M', '1B'] : ['10B', '25B']).includes(s)), 'Invalid tier, scales or VM limit');
  const {label} = identity(env), pools = [];
  const cases = env.RUN_CONTROLS === 'true' ? [...CASES, ...CASES.map(name => name + '-control')] : CASES;
  const rows = cases.flatMap(name => scales.map(scale => {
    const r = requirement(name, scale), t = templates[r.instance];
    assert(Number.isSafeInteger(r.scratch_gib) && r.scratch_gib > 0 && /^[a-z0-9]+\.[a-z0-9]+$/.test(r.instance), 'Invalid case requirement');
    assert(t && /^lt-[a-z0-9]+$/.test(t.template) && typeof t.version === 'string' && /^[1-9][0-9]*$/.test(t.version) &&
      Array.isArray(t.subnets) && t.subnets.length && t.subnets.every(s => /^subnet-[a-z0-9]+$/.test(s)), 'Set a pinned template and subnets for ' + r.instance);
    let p = pools.find(p => p.instance === r.instance);
    if (!p) { p = {...t, instance: r.instance, label: label + '-' + r.instance, scratch_gib: 0, count: 1}; pools.push(p); }
    p.scratch_gib = Math.max(p.scratch_gib, r.scratch_gib);
    return {case: name, scale, instance: r.instance, label: p.label};
  }));
  assert(limit >= pools.length, 'The VM limit must allow one worker per instance type');
  for (let remaining = limit - pools.length; remaining > 0; remaining--) {
    const p = pools.find(p => p.count < rows.filter(r => r.instance === p.instance).length);
    if (!p) break;
    p.count++;
  }
  return {tier, matrix: {include: rows}, pools};
}
async function list(io, path, key) {
  const all = [];
  for (let page = 1; ; page++) {
    const rows = (await io.api(path + (path.includes('?') ? '&' : '?') + 'per_page=100&page=' + page))[key];
    all.push(...rows);
    if (rows.length < 100) return all;
  }
}
async function cleanup(io) {
  const {repo, label} = identity(io.env);
  const groups = (await io.aws('autoscaling', 'describe-auto-scaling-groups', {})).AutoScalingGroups.filter(g =>
    g.AutoScalingGroupName.startsWith(label + '-') && [['EngineE2EPool', label], ['EngineE2ERepository', repo]]
      .every(([key, value]) => g.Tags?.some(t => t.Key === key && t.Value === value)));
  for (const g of groups) {
    await io.aws('autoscaling', 'delete-auto-scaling-group', {AutoScalingGroupName: g.AutoScalingGroupName, ForceDelete: true});
  }
  const deadline = io.now() + 600;
  while (groups.length && (await io.aws('autoscaling', 'describe-auto-scaling-groups',
    {AutoScalingGroupNames: groups.map(g => g.AutoScalingGroupName)})).AutoScalingGroups.length) {
    assert(io.now() < deadline, 'Group deletion timed out');
    await io.sleep(10000);
  }
  const templates = new Set([...groups.map(g => g.LaunchTemplate.LaunchTemplateId),
    ...Object.values(JSON.parse(io.env.ENGINE_E2E_TEMPLATES_JSON || '{}')).map(t => t.template)]);
  for (const id of templates) {
    const versions = (await io.aws('ec2', 'describe-launch-template-versions', {LaunchTemplateId: id})).LaunchTemplateVersions || [];
    const owned = versions.filter(v => v.VersionDescription?.startsWith(label + '-')).map(v => String(v.VersionNumber));
    if (owned.length) await io.aws('ec2', 'delete-launch-template-versions', {LaunchTemplateId: id, Versions: owned});
  }
  for (const r of await list(io, 'repos/' + repo + '/actions/runners', 'runners'))
    if (r.name.startsWith(label + '-') && r.labels.some(l => l.name.startsWith(label + '-')))
      await io.api('repos/' + repo + '/actions/runners/' + r.id, 'DELETE');
}
async function start(selected, io) {
  const {env} = io, {repo, run, attempt, label} = identity(env);
  const source = await io.api('repos/' + repo + '/actions/runs/' + run);
  assert(trusted(source, repo) && source.name === 'Kumo Engine ' + selected.tier && source.run_attempt === Number(attempt) &&
    source.head_sha === env.GITHUB_SHA && /^[a-f0-9]{40}$/.test(env.GITHUB_SHA) && source.status === 'in_progress', 'Expected the current main-branch Engine run');
  if (selected.tier === 'nightly' && source.event === 'schedule' &&
    (await io.api('repos/' + repo + '/actions/workflows/kumo-engine-nightly.yml/runs?branch=main&event=schedule&per_page=1'))
      .workflow_runs.some(newer => newer.id > Number(run))) return false;
  assert(selected.pools.length && selected.pools.every(p => Number.isSafeInteger(p.count) && p.count > 0 && p.label === label + '-' + p.instance) &&
    selected.pools.reduce((n, p) => n + p.count, 0) <= Number(env.ENGINE_E2E_MAX_PARALLEL || 2), 'Invalid pool size');
  const deadline = Math.floor(io.now()) + (selected.tier === 'nightly' ? 4 : 24) * 3600;
  const expires = deadline + (selected.tier === 'nightly' ? 3 : 6) * 3600 + 1800;
  const event = {run, attempt, sha: env.GITHUB_SHA, deadline, cases: selected.matrix.include,
    repo, results: env.RESULTS_S3};
  // Native AWS capacity retries continue after this short GitHub job ends.
  const schedule = (Name, start, end, ScheduleExpression, Target) => io.aws('scheduler', 'create-schedule', {
    Name, ScheduleExpression,
    StartDate: new Date(start * 1000).toISOString(), EndDate: new Date(end * 1000).toISOString(),
    FlexibleTimeWindow: {Mode: 'OFF'}, ActionAfterCompletion: 'DELETE', Target});
  await schedule(label, deadline, expires + 3600, 'rate(10 minutes)',
    {RoleArn: env.CAPACITY_ROLE, Arn: env.CAPACITY_FUNCTION, Input: JSON.stringify(event)});
  for (const p of selected.pools) {
    await schedule(p.label + '-wait', deadline, deadline + 3600, 'rate(1 minute)',
      {RoleArn: env.EXPIRY_ROLE, Arn: 'arn:aws:scheduler:::aws-sdk:autoscaling:suspendProcesses',
        Input: JSON.stringify({AutoScalingGroupName: p.label, ScalingProcesses: ['Launch']})});
    await schedule(p.label, expires, expires + 3600, 'rate(1 minute)',
      {RoleArn: env.EXPIRY_ROLE, Arn: 'arn:aws:scheduler:::aws-sdk:autoscaling:deleteAutoScalingGroup',
        Input: JSON.stringify({AutoScalingGroupName: p.label, ForceDelete: true})});
    const values = {REPOSITORY: repo, SOURCE_SHA: env.GITHUB_SHA, LABEL: p.label,
      CAPACITY_FUNCTION: env.CAPACITY_FUNCTION, POOL_RUN_ID: run, POOL_ATTEMPT: attempt,
      AWS_REGION: env.AWS_REGION, SCRATCH_GIB: p.scratch_gib,
      INSTANCE_TYPE: p.instance, TIER: selected.tier, CAPACITY_DEADLINE: deadline,
      POOL_CASES: JSON.stringify(selected.matrix.include.filter(r => r.label === p.label))};
    const quote = value => "'" + String(value).replaceAll("'", "'\\''") + "'";
    const script = '#!/bin/bash\n' + Object.entries(values).map(([key, value]) => key + '=' + quote(value)).join('\n') +
      '\n' + readFileSync(join(__dirname, 'engine-e2e-runner.sh'), 'utf8');
    const tags = {EngineE2EPool: label, EngineE2ERepository: repo};
    const version = (await io.aws('ec2', 'create-launch-template-version', {LaunchTemplateId: p.template,
      SourceVersion: p.version, VersionDescription: p.label, ClientToken: p.label,
      LaunchTemplateData: {InstanceType: p.instance, UserData: Buffer.from(script).toString('base64'),
        MetadataOptions: {HttpTokens: 'required'}, InstanceInitiatedShutdownBehavior: 'terminate',
        TagSpecifications: [{ResourceType: 'instance', Tags: Object.entries(tags).map(([Key, Value]) => ({Key, Value}))}]}})).LaunchTemplateVersion.VersionNumber;
    await io.aws('autoscaling', 'create-auto-scaling-group', {AutoScalingGroupName: p.label,
      LaunchTemplate: {LaunchTemplateId: p.template, Version: String(version)}, VPCZoneIdentifier: p.subnets.join(','),
      MinSize: 0, MaxSize: p.count, DesiredCapacity: 0,
      InstanceMaintenancePolicy: {MinHealthyPercentage: 0, MaxHealthyPercentage: 100},
      AvailabilityZoneDistribution: {CapacityDistributionStrategy: 'balanced-best-effort'},
      Tags: Object.entries(tags).map(([Key, Value]) => ({Key, Value, PropagateAtLaunch: true}))});
    await io.aws('autoscaling', 'suspend-processes', {AutoScalingGroupName: p.label, ScalingProcesses: ['AZRebalance']});
    await io.aws('autoscaling', 'set-desired-capacity', {AutoScalingGroupName: p.label, DesiredCapacity: p.count});
  }
}
function admission(event, row) {
  const url = new URL(event.results);
  assert(url.protocol === 's3:' && url.hostname, 'Set an S3 result root');
  return {Bucket: url.hostname, Key: [url.pathname.replace(/^\//, '').replace(/\/$/, ''),
    event.run, event.attempt, row.scale, row.case + '.admission.json'].filter(Boolean).join('/')};
}
async function registration(event, io) {
  const {repo, run, attempt, label} = identity({GITHUB_REPOSITORY: event.repo, POOL_RUN_ID: event.run, POOL_ATTEMPT: event.attempt});
  const source = await io.api('repos/' + repo + '/actions/runs/' + run);
  assert(trusted(source, repo) && source.run_attempt === Number(attempt) && source.head_sha === event.sha &&
    ['queued', 'in_progress'].includes(source.status) && Number.isFinite(event.deadline) && io.now() < event.deadline &&
    /^[a-z0-9]+\.[a-z0-9]+$/.test(event.instance) && event.label === label + '-' + event.instance,
    'Registration requires a live approved pool');
  if (event.kind === 'bootstrap') assert(typeof event.error === 'string' && event.error.length <= 160 &&
    /^Bootstrap failed: [a-zA-Z0-9 ._-]+ \(exit [1-9][0-9]{0,2}\)$/.test(event.error), 'Invalid bootstrap check');
  assert(Array.isArray(event.cases) && event.cases.length > 0 && event.cases.length <= 48 &&
    new Set(event.cases.map(r => r.case + '/' + r.scale)).size === event.cases.length && event.cases.every(r =>
      CASES.includes(r.case.replace(/-control$/, '')) && r.instance === event.instance && r.label === event.label &&
      (source.name.endsWith('nightly') ? ['100M', '1B'] : ['10B', '25B']).includes(r.scale)), 'Invalid pool cases');
  const groups = (await io.aws('autoscaling', 'describe-auto-scaling-groups', {AutoScalingGroupNames: [event.label]})).AutoScalingGroups;
  assert(groups.length === 1 && [['EngineE2EPool', label], ['EngineE2ERepository', repo]].every(([key, value]) =>
    groups[0].Tags?.some(t => t.Key === key && t.Value === value)), 'Expected an owned runner group');
  const jobs = await list(io, 'repos/' + repo + '/actions/runs/' + run + '/attempts/' + attempt + '/jobs', 'jobs');
  const drained = event.cases.every(r => jobs.some(j => j.labels?.includes(event.label) &&
    j.name.endsWith('engine-e2e:' + r.case + ':' + r.scale) && ['in_progress', 'completed'].includes(j.status)));
  if (event.kind === 'bootstrap' || drained) {
    await io.aws('autoscaling', 'suspend-processes', {AutoScalingGroupName: event.label, ScalingProcesses: ['Launch']});
    if (event.kind === 'bootstrap') return capacity({...event, results: io.env.RESULTS_S3}, io);
    return null;
  }
  if (groups[0].SuspendedProcesses?.some(p => p.ProcessName === 'Launch')) return null;
  return (await io.api('repos/' + repo + '/actions/runners/registration-token', 'POST')).token;
}
async function capacity(event, io) {
  identity({GITHUB_REPOSITORY: event.repo, POOL_RUN_ID: event.run, POOL_ATTEMPT: event.attempt});
  const source = await io.api('repos/' + event.repo + '/actions/runs/' + event.run);
  assert(trusted(source, event.repo) && source.head_sha === event.sha, 'Untrusted capacity event');
  if (source.run_attempt !== Number(event.attempt)) return;
  const bootstrap = event.kind === 'bootstrap';
  assert(Number.isFinite(event.deadline) && (bootstrap || io.now() >= event.deadline) && event.deadline >=
    Date.parse(source.run_started_at) / 1000 + (source.name.endsWith('nightly') ? 4 : 24) * 3600,
    'Capacity deadline has not passed');
  if (source.status === 'completed' && Date.parse(source.updated_at) / 1000 < event.deadline) return;
  const jobs = await list(io, 'repos/' + event.repo + '/actions/runs/' + event.run + '/attempts/' + event.attempt + '/jobs', 'jobs');
  const failed = [];
  for (const row of event.cases) {
    const job = jobs.find(j => j.name.endsWith('engine-e2e:' + row.case + ':' + row.scale));
    if ((!job && !bootstrap) || job?.status === 'in_progress' || (job?.status === 'completed' && job.runner_id)) continue;
    try {
      await io.aws('s3', 'put-object', {...admission(event, row), IfNoneMatch: '*',
        Body: JSON.stringify({state: bootstrap ? 'bootstrap_failed' : 'capacity_failed', instance: row.instance,
          deadline: event.deadline, ...(bootstrap ? {error: event.error} : {})})});
      failed.push(row);
    } catch (error) {
      if (!io.preconditionFailed(error)) throw error;
      const saved = await io.aws('s3', 'get-object', admission(event, row));
      if (['capacity_failed', 'bootstrap_failed'].includes(saved.state)) failed.push(row);
    }
  }
  if (failed.length) await io.api('repos/' + event.repo + '/actions/workflows/engine-e2e-lifecycle.yml/dispatches', 'POST',
    {ref: 'main', inputs: {payload: JSON.stringify({...event, cases: failed})}});
}
async function report(event, io) {
  const {run, attempt, sha, deadline, cases} = event;
  assert(event.results === io.env.RESULTS_S3, 'Expected the configured result root');
  identity({GITHUB_REPOSITORY: 'NVIDIA/kumo-engine', POOL_RUN_ID: run, POOL_ATTEMPT: attempt});
  const source = await io.api('repos/NVIDIA/kumo-engine/actions/runs/' + run);
  assert(trusted(source, 'NVIDIA/kumo-engine') && source.run_attempt === Number(attempt) && source.head_sha === sha &&
    Number.isFinite(deadline) && deadline >= Date.parse(source.run_started_at) / 1000 +
      (source.name.endsWith('nightly') ? 4 : 24) * 3600, 'Untrusted or premature capacity report');
  assert(cases.length > 0 && cases.length <= 48 && new Set(cases.map(r => r.case + '/' + r.scale)).size === cases.length, 'Invalid report cases');
  const rows = [];
  for (const row of cases) {
    assert(CASES.includes(row.case.replace(/-control$/, '')) && (source.name.endsWith('nightly') ? ['100M', '1B'] : ['10B', '25B'])
      .includes(row.scale), 'Invalid report case');
    const claim = await io.aws('s3', 'get-object', admission(event, row));
    assert((claim.state === 'bootstrap_failed' || (claim.state === 'capacity_failed' && io.now() >= deadline)) &&
      claim.instance === row.instance && claim.deadline === deadline,
      'A running case cannot be replaced by a capacity report');
    rows.push(claim.state === 'bootstrap_failed' ? {...row, phase: 'bootstrap', error: claim.error} : row);
  }
  return {include: rows};
}
function command(file, args, env = process.env, input) {
  try {
    const value = execFileSync(file, args, {env, input, encoding: 'utf8', timeout: 45000});
    return value.trim() ? JSON.parse(value) : null;
  } catch (error) {
    if (file === 'gh' && args.includes('DELETE') && String(error.stderr).includes('HTTP 404')) return null;
    throw Error(String(error.stderr || 'The controller command failed'));
  }
}
const io = {env: process.env, now: () => Date.now() / 1000, sleep,
  api: (path, method = 'GET', data = {}) => command('gh', ['api', path, '--method', method,
    ...(method === 'POST' ? ['--input', '-'] : [])], {...process.env,
    GH_TOKEN: process.env[path.startsWith('repos/NVIDIA/kumo-engine/actions/runners') ? 'GH_TOKEN' : 'GITHUB_TOKEN']}, JSON.stringify(data)),
  aws: (service, operation, input) => {
    if (service === 's3' && operation === 'get-object')
      return command('aws', ['s3', 'cp', 's3://' + input.Bucket + '/' + input.Key, '-', '--only-show-errors']);
    return command('aws', [service, ...operation.split(' '), '--cli-input-json', JSON.stringify(input), '--output', 'json', '--no-cli-pager']);
  }};
async function main() {
  const {env} = io, mode = process.argv[2];
  assert(process.argv.length === 3, 'Expected start, cleanup or report');
  if (mode === 'cleanup') {
    const event = JSON.parse(readFileSync(env.GITHUB_EVENT_PATH, 'utf8'));
    assert(event.action === 'completed' && trusted(event.workflow_run, env.GITHUB_REPOSITORY) &&
      String(event.workflow_run.id) === env.POOL_RUN_ID && String(event.workflow_run.run_attempt) === env.POOL_ATTEMPT, 'Untrusted cleanup event');
    await cleanup(io);
  } else if (mode === 'report') {
    const event = JSON.parse(readFileSync(env.GITHUB_EVENT_PATH, 'utf8'));
    appendFileSync(env.GITHUB_OUTPUT, 'matrix=' + JSON.stringify(await report(JSON.parse(event.inputs.payload), io)) + '\n');
  } else {
    assert(mode === 'start' && env.GITHUB_REF === 'refs/heads/main' && env.ENGINE_E2E_ENABLED === 'true' &&
      env.ENGINE_E2E_SCHEDULER_ENABLED === 'true', 'VM creation is disabled');
    const selected = plan(env.TIER, JSON.parse(env.SCALES), JSON.parse(env.ENGINE_E2E_TEMPLATES_JSON),
      Number(env.ENGINE_E2E_MAX_PARALLEL || 2), (name, scale) => command('python',
        [join(__dirname, '../../packages/kumo-engine/test/cases', name.replace(/-control$/, '') + '.py'),
          '--scale', scale, '--requirements']), env);
    if (await start(selected, io) === false) return;
    appendFileSync(env.GITHUB_OUTPUT, 'plan=' + JSON.stringify(selected) + '\nattempt=' + env.GITHUB_RUN_ATTEMPT + '\n');
  }
}
module.exports = {CASES, plan, start, cleanup, capacity, report, registration};
if (require.main === module) main().catch(error => { console.error(error.message); process.exitCode = 1; });
