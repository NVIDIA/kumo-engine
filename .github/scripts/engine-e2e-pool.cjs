// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const {createHash, randomBytes} = require('node:crypto');
const {readFileSync, appendFileSync} = require('node:fs');
const {join} = require('node:path');
const {setTimeout: sleep} = require('node:timers/promises');
const CASES = ['adtech_binary', 'delivery_multiclass', 'delivery_multilabel', 'delivery_multilabel_ranking',
  'delivery_regression', 'delivery_forecasting', 'commerce_static_link', 'commerce_temporal_link',
  'streaming_temporal_link', 'community_regression', 'payments_binary', 'fraud_binary'];
const REGION = /^[a-z]{2}(-[a-z]+)+-[0-9]+$/, SUBNET = /^subnet-[a-z0-9]+$/, INSTANCE = /^[a-z0-9]+\.[a-z0-9]+$/;
const ZONE = /^[a-z]{2}(-[a-z0-9]+)+(#[1-9][0-9]{0,2})?$/, CODE = /^[A-Za-z][A-Za-z0-9.]{0,63}$/;
const BOOT_ERROR = /^Bootstrap failed: [a-zA-Z0-9 ._-]+ \(exit [1-9][0-9]{0,2}\)$/, RUNNER_ERROR = /^Bootstrap failed: runner (registration|configuration) /;
const CAPACITY_ERROR = 'No runner capacity before tier deadline', NO_RUNNER = 'No runner 15 minutes after launch';
const ENDED = 'VM ended without a runner';
const REGION_WIDE = new Set(['VcpuLimitExceeded', 'InstanceLimitExceeded', 'UnauthorizedOperation', 'RequestLimitExceeded']);
const LIVE = ['Pending', 'Pending:Wait', 'Pending:Proceed', 'InService'];
const UNSEEN_SECONDS = 600, LEASE_SECONDS = 120;
const matches = (value, pattern) => typeof value === 'string' && pattern.test(value);
const hard = reason => matches(reason, BOOT_ERROR) && !matches(reason, RUNNER_ERROR);
const short = (code, message) => String(code || 'Error') + ': ' + String(message || '').split('\n')[0].slice(0, 120);
const sha256 = value => createHash('sha256').update(value).digest('hex');
const awsCode = error => {
  const code = /An error occurred \(([^)]*)\)/.exec(String(error?.message))?.[1];
  return matches(code, CODE) ? code : 'Error';
};
const rejected = error => error.$metadata?.httpStatusCode >= 400 && error.$metadata.httpStatusCode < 500 &&
  [1, undefined].includes(error.$metadata.attempts);
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
function regions(config) {
  assert(Array.isArray(config) && config.length, 'Set ENGINE_E2E_TEMPLATES_JSON to a list of regions in sweep order');
  for (const [n, entry] of config.entries()) {
    assert(entry && matches(entry.region, REGION) && Object.keys(entry).every(k => ['region', 'templates'].includes(k)) &&
      config.findIndex(e => e.region === entry.region) === n && entry.templates && typeof entry.templates === 'object' &&
      !Array.isArray(entry.templates), 'Set a unique valid region and its templates in region entry ' + n);
    for (const [instance, t] of Object.entries(entry.templates))
      assert(matches(instance, INSTANCE) && t && Object.keys(t).every(k => ['template', 'version', 'subnets', 'limit'].includes(k)) &&
        matches(t.template, /^lt-[a-z0-9]+$/) && matches(t.version, /^[1-9][0-9]*$/) && Array.isArray(t.subnets) && t.subnets.length &&
        new Set(t.subnets).size === t.subnets.length && t.subnets.every(s => matches(s, SUBNET)) &&
        (t.limit === undefined || (Number.isSafeInteger(t.limit) && t.limit > 0)),
      'Set a pinned template, unique subnets and an optional positive limit for ' + instance + ' in ' + entry.region);
  }
  return config;
}
function plan(tier, scales, config, limit, requirement, env = process.env) {
  assert(['nightly', 'weekly'].includes(tier) && Number.isSafeInteger(limit) && limit > 0 && limit <= 48 &&
    Array.isArray(scales) && scales.length && new Set(scales).size === scales.length && scales.every(s =>
      (tier === 'nightly' ? ['100M', '1B'] : ['10B', '25B']).includes(s)), 'Invalid tier, scales or VM limit');
  regions(config);
  const {label} = identity(env), pools = [];
  const cases = env.RUN_CONTROLS === 'true' ? [...CASES, ...CASES.map(name => name + '-control')] : CASES;
  const rows = cases.flatMap(name => scales.map(scale => {
    const r = requirement(name, scale);
    assert(Number.isSafeInteger(r.scratch_gib) && r.scratch_gib > 0 && matches(r.instance, INSTANCE), 'Invalid case requirement');
    let p = pools.find(p => p.instance === r.instance);
    if (!p) {
      const defined = config.filter(c => Object.hasOwn(c.templates, r.instance)).map(c => ({region: c.region, ...c.templates[r.instance]}));
      assert(defined.length, 'Set a pinned template and subnets for ' + r.instance + ' in at least one region');
      p = {instance: r.instance, label: label + '-' + r.instance, scratch_gib: 0, count: 1, regions: defined}; pools.push(p);
    }
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
  const config = regions(JSON.parse(io.env.ENGINE_E2E_TEMPLATES_JSON || '[]'));
  const groups = config.map(() => []), failed = new Map();
  const fail = (region, error) => {
    const code = error.code === 'ERR_ASSERTION' ? error.message : awsCode(error);
    console.error('Cleanup failed in ' + region + ': ' + code);
    if (!failed.has(region)) failed.set(region, code);
  };
  const each = async action => {
    for (const [n, entry] of config.entries())
      try { await action(n, entry); } catch (error) { fail(entry.region, error); }
  };
  await each(async (n, {region}) => {
    for (const g of (await io.aws('autoscaling', 'describe-auto-scaling-groups', {}, region)).AutoScalingGroups.filter(g =>
      g.AutoScalingGroupName.startsWith(label + '-') && [['EngineE2EPool', label], ['EngineE2ERepository', repo]]
        .every(([key, value]) => g.Tags?.some(t => t.Key === key && t.Value === value))))
      try {
        await io.aws('autoscaling', 'delete-auto-scaling-group', {AutoScalingGroupName: g.AutoScalingGroupName, ForceDelete: true}, region);
        groups[n].push(g);
      } catch (error) { fail(region, error); }
  });
  const deadline = io.now() + 600;
  await each(async (n, {region}) => {
    while (groups[n].length && (await io.aws('autoscaling', 'describe-auto-scaling-groups',
      {AutoScalingGroupNames: groups[n].map(g => g.AutoScalingGroupName)}, region)).AutoScalingGroups.length) {
      assert(io.now() < deadline, 'Group deletion timed out');
      await io.sleep(10000);
    }
  });
  await each(async (n, {region, templates}) => {
    for (const id of new Set([...groups[n].map(g => g.LaunchTemplate.LaunchTemplateId), ...Object.values(templates).map(t => t.template)])) {
      const versions = (await io.aws('ec2', 'describe-launch-template-versions', {LaunchTemplateId: id}, region)).LaunchTemplateVersions || [];
      const owned = versions.filter(v => v.VersionDescription?.startsWith(label + '-')).map(v => String(v.VersionNumber));
      if (owned.length) await io.aws('ec2', 'delete-launch-template-versions', {LaunchTemplateId: id, Versions: owned}, region);
    }
  });
  for (const r of await list(io, 'repos/' + repo + '/actions/runners', 'runners'))
    if (r.name.startsWith(label + '-') && r.labels.some(l => l.name.startsWith(label + '-')))
      await io.api('repos/' + repo + '/actions/runners/' + r.id, 'DELETE');
  assert(!failed.size, 'Cleanup failed in ' + [...failed].map(([region, code]) => region + ' (' + code + ')').join(', '));
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
  const schedule = (Name, start, end, ScheduleExpression, Target, region) => io.aws('scheduler', 'create-schedule', {
    Name, ScheduleExpression,
    ...(start === undefined ? {} : {StartDate: new Date(start * 1000).toISOString()}), EndDate: new Date(end * 1000).toISOString(),
    FlexibleTimeWindow: {Mode: 'OFF'}, ActionAfterCompletion: 'DELETE', Target}, region);
  await schedule(label, deadline, expires + 3600, 'rate(10 minutes)',
    {RoleArn: env.CAPACITY_ROLE, Arn: env.CAPACITY_FUNCTION, Input: JSON.stringify(event)}, env.AWS_REGION);
  const ready = selected.pools.map(() => []), keys = selected.pools.map(() => randomBytes(32).toString('hex'));
  for (const [n, p] of selected.pools.entries()) {
    const values = {REPOSITORY: repo, SOURCE_SHA: env.GITHUB_SHA, LABEL: p.label,
      CAPACITY_FUNCTION: env.CAPACITY_FUNCTION, POOL_RUN_ID: run, POOL_ATTEMPT: attempt,
      AWS_REGION: env.AWS_REGION, SCRATCH_GIB: p.scratch_gib,
      INSTANCE_TYPE: p.instance, TIER: selected.tier, CAPACITY_DEADLINE: deadline,
      POOL_CASES: JSON.stringify(selected.matrix.include.filter(r => r.label === p.label))};
    const quote = value => "'" + String(value).replaceAll("'", "'\\''") + "'";
    const script = '#!/bin/bash\n' + Object.entries(values).map(([key, value]) => key + '=' + quote(value)).join('\n') +
      '\n' + readFileSync(join(__dirname, 'engine-e2e-runner.sh'), 'utf8');
    const tags = {EngineE2EPool: label, EngineE2ERepository: repo,
      EngineE2ESha: env.GITHUB_SHA, EngineE2EDeadline: String(deadline)};
    for (const r of p.regions) {
      try {
        await schedule(p.label, expires, expires + 3600, 'rate(1 minute)',
          {RoleArn: env.EXPIRY_ROLE, Arn: 'arn:aws:scheduler:::aws-sdk:autoscaling:deleteAutoScalingGroup',
            Input: JSON.stringify({AutoScalingGroupName: p.label, ForceDelete: true})}, r.region);
        const version = (await io.aws('ec2', 'create-launch-template-version', {LaunchTemplateId: r.template,
          SourceVersion: r.version, VersionDescription: p.label, ClientToken: p.label,
          LaunchTemplateData: {InstanceType: p.instance, UserData: Buffer.from(script).toString('base64'),
            MetadataOptions: {HttpTokens: 'required'}, InstanceInitiatedShutdownBehavior: 'terminate',
            TagSpecifications: [{ResourceType: 'instance', Tags: Object.entries(tags).map(([Key, Value]) => ({Key, Value}))}]}}, r.region))
          .LaunchTemplateVersion.VersionNumber;
        await io.aws('autoscaling', 'create-auto-scaling-group', {AutoScalingGroupName: p.label,
          LaunchTemplate: {LaunchTemplateId: r.template, Version: String(version)}, VPCZoneIdentifier: r.subnets.join(','),
          MinSize: 0, MaxSize: Math.min(p.count, r.limit || p.count), DesiredCapacity: 0, NewInstancesProtectedFromScaleIn: true,
          InstanceMaintenancePolicy: {MinHealthyPercentage: 0, MaxHealthyPercentage: 100},
          AvailabilityZoneDistribution: {CapacityDistributionStrategy: 'balanced-best-effort'},
          Tags: [...Object.entries(tags).map(([Key, Value]) => ({Key, Value, PropagateAtLaunch: true})),
            ...Object.entries({EngineE2ECount: String(p.count), EngineE2EKey: sha256(keys[n])})
              .map(([Key, Value]) => ({Key, Value, PropagateAtLaunch: false}))]}, r.region);
        await io.aws('autoscaling', 'suspend-processes', {AutoScalingGroupName: p.label, ScalingProcesses: ['Launch', 'AZRebalance']}, r.region);
        ready[n].push({region: r.region, subnets: r.subnets});
      } catch (error) {
        const message = 'Skipped ' + r.region + ' for ' + p.label + ': ' + awsCode(error);
        console.error(message);
        if (env.GITHUB_STEP_SUMMARY) appendFileSync(env.GITHUB_STEP_SUMMARY, '- ' + message + '\n');
      }
    }
  }
  const empty = selected.pools.filter((p, n) => !ready[n].length).map(p => p.label);
  assert(!empty.length, 'No region passed setup for ' + empty.join(', '));
  for (const [n, p] of selected.pools.entries())
    await schedule(p.label + '-sweep', undefined, deadline, 'rate(1 minute)', {RoleArn: env.CAPACITY_ROLE,
      Arn: env.CAPACITY_FUNCTION, RetryPolicy: {MaximumRetryAttempts: 0}, Input: JSON.stringify({kind: 'sweep', repo, run, attempt,
        sha: env.GITHUB_SHA, deadline, results: env.RESULTS_S3, label: p.label, instance: p.instance, count: p.count, key: keys[n],
        cases: selected.matrix.include.filter(r => r.label === p.label).map(r => ({case: r.case, scale: r.scale})), regions: ready[n]})},
    env.AWS_REGION);
}
function objectKey(event, ...parts) {
  const url = new URL(event.results);
  assert(url.protocol === 's3:' && url.hostname, 'Set an S3 result root');
  return {Bucket: url.hostname, Key: [url.pathname.replace(/^\//, '').replace(/\/$/, ''),
    event.run, event.attempt, ...parts].filter(Boolean).join('/')};
}
function admission(event, row) {
  return objectKey(event, row.scale, row.case + '.admission.json');
}
function poolFailure(event, instance) {
  return admission(event, {scale: 'pools', case: instance});
}
const sweepState = (event, instance) => objectKey(event, 'pools', instance + '.sweep.json');
const closedRegion = (event, instance, region) => objectKey(event, 'pools', instance, 'closed', region + '.json');
async function close(event, instance, region, reason, io, id) {
  const key = closedRegion(event, instance, region), put = condition => io.aws('s3', 'put-object', {...key, ...condition,
    Body: JSON.stringify({region, reason, at: io.now(), ...(id ? {instance: id} : {})})});
  try {
    await put({IfNoneMatch: '*'});
    return reason;
  } catch (error) { if (!io.preconditionFailed(error)) throw error; }
  const marker = await io.aws('s3', 'read-object', key);
  if (marker && (hard(marker.Body.reason) || !hard(reason))) return marker.Body.reason;
  await put(marker ? {IfMatch: marker.ETag} : {IfNoneMatch: '*'});
  return reason;
}
async function closed(event, instance, names, io) {
  const found = new Map();
  for (const region of names) {
    const marker = await io.aws('s3', 'get-object', closedRegion(event, instance, region));
    if (marker) found.set(region, marker.reason);
  }
  return found;
}
async function shortage(event, instance, io) {
  const state = await io.aws('s3', 'get-object', sweepState(event, instance)), codes = new Map(), shut = [];
  const add = (code, zone) => {
    const name = matches(code, CODE) ? code : 'Error';
    if (!codes.get(name)?.includes(zone)) codes.set(name, [...(codes.get(name) || []), zone]);
  };
  for (const [zone, error] of Object.entries(state?.last_errors || {})) if (matches(zone, ZONE)) add(String(error).split(':')[0], zone);
  for (const [region, code] of Object.entries(state?.unread || {})) if (matches(region, REGION)) add(code, region);
  for (const {region} of state?.regions || []) {
    const marker = matches(region, REGION) && await io.aws('s3', 'get-object', closedRegion(event, instance, region));
    if (marker) shut.push([region, [NO_RUNNER, ENDED].includes(marker.reason) || matches(marker.reason, BOOT_ERROR) ? ' (' + marker.reason + ')' : '']);
  }
  if (!codes.size && !shut.length) return undefined;
  const zoneText = codes.size ? 'Last zone errors: ' + [...codes].map(([code, zones]) => code + ' in ' + zones.join(', ')).join('; ') + '.' : '';
  const shutText = reasons => shut.length ? 'Closed regions: ' + shut.map(([region, reason]) => region + (reasons ? reason : '')).join(', ') + '.' : '';
  const text = (zones, regions) => [CAPACITY_ERROR + '.', zones, regions].filter(Boolean).join(' ');
  for (const reasons of [true, false]) if (text(zoneText, shutText(reasons)).length <= 1000) return text(zoneText, shutText(reasons));
  return text(zoneText.slice(0, 1000 - text('', shutText(false)).length - 4) + '...', shutText(false));
}
async function registration(event, io) {
  const {repo, run, attempt, label} = identity({GITHUB_REPOSITORY: event.repo, POOL_RUN_ID: event.run, POOL_ATTEMPT: event.attempt});
  assert(Number.isFinite(event.deadline) && matches(event.instance, INSTANCE) && event.label === label + '-' + event.instance &&
    matches(event.region, REGION) && matches(event.instance_id, /^i-[0-9a-f]{8,17}$/), 'Registration requires a live approved pool');
  if (io.now() >= event.deadline) return null;
  const groups = (await io.aws('autoscaling', 'describe-auto-scaling-groups', {AutoScalingGroupNames: [event.label]}, event.region)).AutoScalingGroups;
  const instance = groups[0]?.Instances?.find(i => i.InstanceId === event.instance_id);
  assert(groups.length === 1 && [['EngineE2EPool', label], ['EngineE2ERepository', repo]].every(([key, value]) =>
    groups[0].Tags?.some(t => t.Key === key && t.Value === value)) && instance, 'Expected an owned runner group');
  const root = {...event, results: io.env.RESULTS_S3};
  if (event.kind === 'bootstrap') {
    assert(typeof event.error === 'string' && event.error.length <= 160 && BOOT_ERROR.test(event.error) &&
      [['EngineE2ESha', event.sha], ['EngineE2EDeadline', String(event.deadline)]].every(([key, value]) =>
        groups[0].Tags?.some(t => t.Key === key && t.Value === value)), 'Invalid bootstrap failure');
    await close(root, event.instance, event.region, event.error, io);
    return null;
  }
  if (!LIVE.includes(instance.LifecycleState) || await io.aws('s3', 'get-object', closedRegion(root, event.instance, event.region))) return null;
  const source = await io.api('repos/' + repo + '/actions/runs/' + run);
  assert(trusted(source, repo) && source.run_attempt === Number(attempt) && source.head_sha === event.sha &&
    ['queued', 'in_progress'].includes(source.status),
    'Registration requires a live approved pool');
  assert(Array.isArray(event.cases) && event.cases.length > 0 && event.cases.length <= 48 &&
    new Set(event.cases.map(r => r.case + '/' + r.scale)).size === event.cases.length && event.cases.every(r =>
      CASES.includes(r.case.replace(/-control$/, '')) && r.instance === event.instance && r.label === event.label &&
      (source.name.endsWith('nightly') ? ['100M', '1B'] : ['10B', '25B']).includes(r.scale)), 'Invalid pool cases');
  const jobs = await list(io, 'repos/' + repo + '/actions/runs/' + run + '/attempts/' + attempt + '/jobs', 'jobs');
  const drained = event.cases.every(r => jobs.some(j => j.labels?.includes(event.label) &&
    j.name.endsWith('engine-e2e:' + r.case + ':' + r.scale) && ['in_progress', 'completed'].includes(j.status)));
  if (drained || io.now() >= event.deadline) return null;
  return (await io.api('repos/' + repo + '/actions/runners/registration-token', 'POST')).token;
}
async function sweep(event, io) {
  const started = io.now();
  const {repo, run, attempt, label} = identity({GITHUB_REPOSITORY: event.repo, POOL_RUN_ID: event.run, POOL_ATTEMPT: event.attempt});
  assert(event.kind === 'sweep' && event.results === io.env.RESULTS_S3 && matches(event.sha, /^[a-f0-9]{40}$/) && matches(event.key, /^[a-f0-9]{64}$/) &&
    Number.isSafeInteger(event.deadline) && matches(event.instance, INSTANCE) && event.label === label + '-' + event.instance &&
    Number.isSafeInteger(event.count) && event.count > 0 && event.count <= 48 && Array.isArray(event.regions) && event.regions.length &&
    event.regions.every(r => r && matches(r.region, REGION) && Array.isArray(r.subnets) && r.subnets.length &&
      new Set(r.subnets).size === r.subnets.length && r.subnets.every(s => matches(s, SUBNET))) &&
    new Set(event.regions.map(r => r.region)).size === event.regions.length && Array.isArray(event.cases) && event.cases.length &&
    event.cases.length <= 48 && event.cases.every(r => r && matches(r.case, /^[a-z_]+$|^[a-z_]+-control$/) && typeof r.scale === 'string') &&
    new Set(event.cases.map(r => r.case + '/' + r.scale)).size === event.cases.length, 'Invalid sweep event');
  const source = await io.api('repos/' + repo + '/actions/runs/' + run), nightly = source.name?.endsWith('nightly');
  assert(trusted(source, repo) && source.head_sha === event.sha, 'Untrusted sweep event');
  if (source.run_attempt !== Number(attempt) || source.status === 'completed') return {skipped: 'stopped'};
  assert(event.deadline >= Date.parse(source.run_started_at) / 1000 + (nightly ? 4 : 24) * 3600 && event.cases.every(r =>
    CASES.includes(r.case.replace(/-control$/, '')) && (nightly ? ['100M', '1B'] : ['10B', '25B']).includes(r.scale)), 'Untrusted sweep event');
  if (io.now() >= event.deadline) return {skipped: 'stopped'};
  const key = sweepState(event, event.instance), saved = await io.aws('s3', 'read-object', key);
  if (saved?.Body.lease_until > io.now()) return {skipped: 'leased'};
  const now = io.now(), names = event.regions.map(r => r.region), found = new Map(), answers = [], blind = [], unread = {...saved?.Body.unread};
  const owner = [['EngineE2EPool', label], ['EngineE2ERepository', repo], ['EngineE2ESha', event.sha],
    ['EngineE2EDeadline', String(event.deadline)], ['EngineE2ECount', String(event.count)], ['EngineE2EKey', sha256(event.key)]];
  for (const {region, subnets} of event.regions)
    try {
      const g = (await io.aws('autoscaling', 'describe-auto-scaling-groups', {AutoScalingGroupNames: [event.label]}, region)).AutoScalingGroups[0];
      assert(g && owner.every(([k, v]) => g.Tags?.some(t => t.Key === k && t.Value === v)) &&
        subnets.every(s => String(g.VPCZoneIdentifier).split(',').includes(s)), 'Expected an owned runner group');
      found.set(region, g);
      delete unread[region];
    } catch (error) {
      unread[region] = error.code === 'ERR_ASSERTION' ? 'NotOwned' : matches(error.name, CODE) ? error.name : 'Error';
      answers.push({region, error: short(error.name, error.message)});
      blind.push(region);
    }
  assert(found.size || blind.every(r => unread[r] !== 'NotOwned'), 'Untrusted sweep event');
  assert(found.size, 'No region could be read: ' + blind.map(r => unread[r] + ' in ' + r).join(', '));
  const state = {position: 0, launched: {}, last_errors: {}, ...saved?.Body, unread, regions: event.regions,
    lease_until: io.now() + LEASE_SECONDS, lease_owner: io.uuid()};
  let etag;
  const write = async (body, condition) => {
    etag = (await io.aws('s3', 'put-object', {...key, Body: JSON.stringify(body), ...condition})).ETag;
  };
  try {
    await write(state, saved ? {IfMatch: saved.ETag} : {IfNoneMatch: '*'});
  } catch (error) {
    if (io.preconditionFailed(error)) return {skipped: 'leased'};
    throw error;
  }
  try {
    const shut = await closed(event, event.instance, names, io);
    const jobs = await list(io, 'repos/' + repo + '/actions/runs/' + run + '/attempts/' + attempt + '/jobs', 'jobs');
    const job = row => jobs.find(j => j.labels?.includes(event.label) && j.name.endsWith('engine-e2e:' + row.case + ':' + row.scale));
    const waiting = event.cases.filter(r => job(r) && !['in_progress', 'completed'].includes(job(r).status));
    const running = event.cases.filter(r => job(r)?.status === 'in_progress').length;
    const groups = new Map(), terminated = [], closing = [], ended = [];
    const size = id => state.launched[id].count || 1, runner = id => event.label + '-' + sha256(id).slice(0, 12);
    let unseen = 0, runners;
    const named = async id => jobs.some(j => j.runner_name === runner(id)) ||
      (runners ??= await list(io, 'repos/' + repo + '/actions/runners', 'runners')).some(r => r.name === runner(id));
    const shutDown = async (region, reason, id) => {
      shut.set(region, await close(event, event.instance, region, reason, io, id));
      closing.push(region);
    };
    for (const region of names) {
      const known = Object.keys(state.launched).filter(id => state.launched[id].region === region), g = found.get(region);
      if (!g) {
        unseen += known.reduce((n, id) => n + size(id), 0);
        continue;
      }
      const live = (g.Instances || []).filter(i => /^Pending/.test(i.LifecycleState) || ['InService', 'Standby'].includes(i.LifecycleState));
      const shown = new Set(live.map(i => i.InstanceId)), recent = [];
      const gone = new Set((g.Instances || []).filter(i => /^(Terminat|Detach)/.test(i.LifecycleState)).map(i => i.InstanceId));
      for (const id of known) {
        const entry = state.launched[id];
        if (shown.has(id)) entry.seen = true;
        else if (!gone.has(id) && !entry.seen && now - entry.at < UNSEEN_SECONDS) recent.push(id);
        else if (id.startsWith('unknown:')) delete state.launched[id];
        else {
          entry.seen = true;
          ended.push({region, id});
        }
      }
      const placeholders = recent.filter(id => id.startsWith('unknown:')).sort((a, b) => state.launched[a].at - state.launched[b].at);
      for (const i of live)
        if (!state.launched[i.InstanceId]) {
          state.launched[i.InstanceId] = {region, at: now, seen: true};
          if (!placeholders.length) continue;
          if (!--state.launched[placeholders[0]].count) delete state.launched[placeholders.shift()];
          if (!placeholders.length) delete state.last_errors[region];
        }
      groups.set(region, {max: g.MaxSize, desired: g.DesiredCapacity, live,
        recent: recent.filter(id => state.launched[id]).reduce((n, id) => n + size(id), 0)});
    }
    for (const {region, id} of ended) {
      if (waiting.length && !shut.has(region) && !await named(id)) await shutDown(region, ENDED, id);
      delete state.launched[id];
    }
    for (const [region, g] of groups)
      for (const i of [...g.live]) {
        if (now - state.launched[i.InstanceId].at < 900 || await named(i.InstanceId)) continue;
        runners = await list(io, 'repos/' + repo + '/actions/runners', 'runners');
        if (runners.some(r => r.name === runner(i.InstanceId))) continue;
        try {
          await io.aws('autoscaling', 'terminate-instance-in-auto-scaling-group', {InstanceId: i.InstanceId, ShouldDecrementDesiredCapacity: true}, region);
          g.live.splice(g.live.indexOf(i), 1); g.desired--;
          terminated.push({region, instance: i.InstanceId});
        } catch (error) { answers.push({region, instance: i.InstanceId, error: short(error.name, error.message)}); }
        if (waiting.length && !shut.has(region)) await shutDown(region, NO_RUNNER, i.InstanceId);
        if (terminated.some(t => t.instance === i.InstanceId)) delete state.launched[i.InstanceId];
      }
    for (const [region, g] of groups) {
      const believed = g.live.length + g.recent;
      if (g.desired > believed)
        try {
          await io.aws('autoscaling', 'set-desired-capacity', {AutoScalingGroupName: event.label, DesiredCapacity: believed, HonorCooldown: false}, region);
          g.desired = believed;
        } catch (error) { answers.push({region, error: short(error.name, error.message)}); }
      g.desired = Math.max(g.desired, believed);
    }
    const filled = unseen + [...groups.values()].reduce((n, g) => n + g.live.length + g.recent, 0), booting = Math.max(0, filled - running);
    let missing = Math.min(event.count - filled, waiting.length - booting), last;
    const wanted = Math.max(0, missing), zones = event.regions.flatMap(r => r.subnets.map((subnet, n) => ({region: r.region, subnet, n: n + 1})));
    if (!wanted && !Object.keys(state.launched).some(id => id.startsWith('unknown:'))) state.last_errors = {};
    const skip = new Set(shut.keys()), checked = new Set(), minute = Math.floor(started / 60);
    const open = () => io.now() - started < 45 && io.now() < event.deadline;
    for (let k = 0; missing > 0 && k < zones.length && open(); k++) {
      const index = (state.position + k) % zones.length, {region, subnet, n} = zones[index], g = groups.get(region);
      if (!g || skip.has(region)) continue;
      if (g.max - g.desired <= 0) { skip.add(region); continue; }
      if (!checked.has(region)) {
        checked.add(region);
        const marker = await io.aws('s3', 'get-object', closedRegion(event, event.instance, region));
        if (marker) { shut.set(region, marker.reason); skip.add(region); continue; }
      }
      if (!open()) break;
      const answer = {region, subnet, requested: Math.min(missing, g.max - g.desired), instances: []};
      const token = sha256([run, attempt, event.instance, region, subnet, minute, answer.requested].join('/'));
      answers.push(answer); last = index;
      let unknown = false;
      try {
        const result = await io.aws('autoscaling', 'launch-instances', {AutoScalingGroupName: event.label, RequestedCapacity: answer.requested,
          ClientToken: token, SubnetIds: [subnet]}, region);
        answer.instances = (result.Instances || []).flatMap(i => i.InstanceIds || []);
        if (answer.instances.length)
          for (const zone of [region, region + '#' + n, ...result.Instances.map(i => i.AvailabilityZone)]) delete state.last_errors[zone];
        const errors = result.Errors || [], error = errors.find(e => REGION_WIDE.has(e.ErrorCode)) || errors[0];
        if (error) Object.assign(answer, {error: short(error.ErrorCode, error.ErrorMessage), zone: REGION_WIDE.has(error.ErrorCode) ? region :
          matches(error.AvailabilityZone, ZONE) ? error.AvailabilityZone : region + '#' + n});
        if (REGION_WIDE.has(error?.ErrorCode)) skip.add(region);
        unknown = !error && answer.instances.length < answer.requested;
      } catch (error) {
        Object.assign(answer, {error: short(error.name, error.message), zone: region});
        unknown = !rejected(error);
        if (!unknown) skip.add(region);
      }
      missing -= answer.instances.length; g.desired += answer.instances.length;
      answer.instances.forEach(id => { state.launched[id] = {region, at: io.now()}; });
      if (answer.error) state.last_errors[answer.zone] = answer.error;
      if (unknown) {
        answer.unknown = true;
        state.launched['unknown:' + token] = {region, at: io.now(), count: answer.requested - answer.instances.length};
      }
      if (answer.instances.length || unknown)
        try { await write(state, {IfMatch: etag}); } catch (error) {
          if (!io.preconditionFailed(error)) throw error;
          break;
        }
      if (unknown) break;
    }
    if (last !== undefined) state.position = (last + 1) % zones.length;
    const record = {at: new Date(started * 1000).toISOString(), filled, waiting: waiting.length, booting, missing: wanted,
      answers, terminated, closed: closing, blind};
    await io.aws('s3', 'put-object', {...objectKey(event, 'pools', event.instance, 'sweeps',
      record.at.replace(/[-:]|\.[0-9]+/g, '') + '.json'), Body: JSON.stringify(record)});
    const error = names.map(r => shut.get(r)).find(hard);
    if (waiting.length && names.every(r => shut.has(r)) && error && !filled && !state.dispatched) {
      try {
        await io.aws('s3', 'put-object', {...poolFailure(event, event.instance), IfNoneMatch: '*',
          Body: JSON.stringify({sha: event.sha, deadline: event.deadline, instance: event.instance, error})});
      } catch (failure) { if (!io.preconditionFailed(failure)) throw failure; }
      await capacity({kind: 'bootstrap', repo, run, attempt, sha: event.sha, deadline: event.deadline, results: event.results,
        cases: waiting.map(r => ({case: r.case, scale: r.scale, instance: event.instance, label: event.label})), error}, io);
      state.dispatched = true;
    }
    return {filled, waiting: waiting.length, booting, missing: wanted, launched: answers.flatMap(a => a.instances || []),
      terminated: terminated.map(t => t.instance), closed: closing, blind, position: state.position, answers};
  } finally {
    const release = () => write({...state, lease_until: 0, lease_owner: null}, {IfMatch: etag});
    try {
      await release().catch(error => {
        if (error.$metadata?.httpStatusCode === 409) return release();
        throw error;
      });
    } catch (error) { if (!io.preconditionFailed(error)) throw error; }
  }
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
  const failed = [], causes = new Map(), shortages = new Map();
  for (const row of event.cases) {
    const job = jobs.find(j => j.labels?.includes(row.label) && j.name.endsWith('engine-e2e:' + row.case + ':' + row.scale));
    if ((!job && !bootstrap) || job?.status === 'in_progress' || (job?.status === 'completed' && job.runner_id)) continue;
    if (!causes.has(row.instance)) {
      const saved = await io.aws('s3', 'get-object', poolFailure(event, row.instance));
      if (saved) assert(saved.sha === event.sha && saved.deadline === event.deadline && saved.instance === row.instance &&
        typeof saved.error === 'string' && BOOT_ERROR.test(saved.error), 'Invalid saved pool failure');
      causes.set(row.instance, saved?.error);
    }
    const error = causes.get(row.instance) || (bootstrap && event.error);
    if (!error && !shortages.has(row.instance)) shortages.set(row.instance, await shortage(event, row.instance, io));
    const detail = error || shortages.get(row.instance), claim = () => io.aws('s3', 'put-object', {...admission(event, row), IfNoneMatch: '*',
      Body: JSON.stringify({state: error ? 'bootstrap_failed' : 'capacity_failed', instance: row.instance,
        deadline: event.deadline, ...(detail ? {error: detail} : {})})});
    try {
      await claim();
      failed.push(row);
    } catch (error) {
      if (!io.preconditionFailed(error)) throw error;
      const saved = await io.aws('s3', 'get-object', admission(event, row));
      if (!saved) await claim();
      if (!saved || ['capacity_failed', 'bootstrap_failed'].includes(saved.state)) failed.push(row);
    }
  }
  const {repo, run, attempt, sha, deadline, results} = event;
  if (failed.length) await io.api('repos/' + repo + '/actions/workflows/engine-e2e-lifecycle.yml/dispatches', 'POST',
    {ref: 'main', inputs: {payload: JSON.stringify({repo, run, attempt, sha, deadline, results, cases: failed})}});
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
    assert(claim.state === 'bootstrap_failed' || claim.error === undefined || (typeof claim.error === 'string' &&
      claim.error.length <= 1000 && claim.error.startsWith(CAPACITY_ERROR + '. ')), 'Invalid capacity failure detail');
    rows.push(claim.state === 'bootstrap_failed' ? {...row, phase: 'bootstrap', error: claim.error} : claim.error ? {...row, error: claim.error} : row);
  }
  return {include: rows};
}
function command(file, args, env = process.env, input) {
  try {
    const value = execFileSync(file, args, {env, input, encoding: 'utf8', timeout: 45000, stdio: ['pipe', 'pipe', 'pipe']});
    return value.trim() ? JSON.parse(value) : null;
  } catch (error) {
    if (file === 'gh' && args.includes('DELETE') && String(error.stderr).includes('HTTP 404')) return null;
    throw Error(String(error.stderr || 'The controller command failed').replace(/\b[0-9a-f]{64}\b/g, '<key>')
      .replace(/\b(lt|subnet|sg|i|ami|vpc|eni|vol)-[0-9a-f]+\b/g, '<id>').replace(/\b[0-9]{12}\b/g, '<account>'));
  }
}
const io = {env: process.env, now: () => Date.now() / 1000, sleep,
  api: (path, method = 'GET', data = {}) => command('gh', ['api', path, '--method', method,
    ...(method === 'POST' ? ['--input', '-'] : [])], {...process.env,
    GH_TOKEN: process.env[path.startsWith('repos/NVIDIA/kumo-engine/actions/runners') ? 'GH_TOKEN' : 'GITHUB_TOKEN']}, JSON.stringify(data)),
  aws: (service, operation, input, region) => {
    const where = region ? ['--region', region] : [];
    if (service === 's3' && operation === 'get-object')
      return command('aws', ['s3', 'cp', 's3://' + input.Bucket + '/' + input.Key, '-', '--only-show-errors', ...where]);
    return command('aws', [service, ...operation.split(' '), '--cli-input-json', JSON.stringify(input), '--output', 'json', '--no-cli-pager', ...where]);
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
module.exports = {CASES, plan, start, cleanup, capacity, report, registration, sweep};
if (require.main === module) main().catch(error => { console.error(error.message); process.exitCode = 1; });
