// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const {createSign, randomUUID} = require('node:crypto');
const assert = require('node:assert/strict');
const {capacity, registration, sweep} = require('./engine-e2e-pool.cjs');
const {SecretsManagerClient, GetSecretValueCommand} = require('@aws-sdk/client-secrets-manager');
const {S3Client, PutObjectCommand, GetObjectCommand, ListObjectsV2Command} = require('@aws-sdk/client-s3');
const {AutoScalingClient, DescribeAutoScalingGroupsCommand, LaunchInstancesCommand, SetDesiredCapacityCommand,
  TerminateInstanceInAutoScalingGroupCommand} = require('@aws-sdk/client-auto-scaling');
assert(LaunchInstancesCommand, 'Bundle @aws-sdk/client-auto-scaling 3.934.0 or later with the broker for LaunchInstances');
const [s3Major, s3Minor] = require('@aws-sdk/client-s3/package.json').version.split('.').map(Number);
assert(s3Major > 3 || (s3Major === 3 && s3Minor >= 934), 'Bundle @aws-sdk/client-auto-scaling and @aws-sdk/client-s3 from one SDK release, 3.934.0 or later');
const requestHandler = {requestTimeout: 10000, throwOnRequestTimeout: true};
const secrets = new SecretsManagerClient({requestHandler}), clients = new Map();
const operations = {'describe-auto-scaling-groups': DescribeAutoScalingGroupsCommand, 'launch-instances': LaunchInstancesCommand,
  'set-desired-capacity': SetDesiredCapacityCommand, 'terminate-instance-in-auto-scaling-group': TerminateInstanceInAutoScalingGroupCommand};
const send = (service, region, command) => {
  const key = service + '/' + (region || '');
  if (!clients.has(key)) clients.set(key, new (service === 's3' ? S3Client : AutoScalingClient)({...(region ? {region} : {}), requestHandler}));
  return clients.get(key).send(command);
};
// App credentials are needed only for GitHub calls, after AWS failure handling.
exports.handler = async event => {
  const worker = ['registration', 'bootstrap'].includes(event.kind);
  assert(worker || event.results === process.env.RESULTS_S3, 'Expected the configured result root');
  let installation;
  const api = async (path, method = 'GET', data, token) => {
    if (!token && !installation) {
      const secret = JSON.parse((await secrets.send(new GetSecretValueCommand({SecretId: process.env.RUNNER_SECRET}))).SecretString);
      const now = Math.floor(Date.now() / 1000), encode = value => Buffer.from(JSON.stringify(value)).toString('base64url');
      const body = encode({alg: 'RS256', typ: 'JWT'}) + '.' + encode({iat: now - 60, exp: now + 540, iss: String(secret.app_id)});
      const jwt = body + '.' + createSign('RSA-SHA256').update(body).sign(secret.private_key, 'base64url');
      installation = (await api('app/installations/' + secret.installation_id + '/access_tokens', 'POST',
        {repositories: ['kumo-engine'], permissions: {administration: 'write', actions: 'write'}}, jwt)).token;
    }
    const response = await fetch('https://api.github.com/' + path, {method,
      headers: {Authorization: 'Bearer ' + (token || installation), Accept: 'application/vnd.github+json',
        'User-Agent': 'engine-e2e-capacity', 'X-GitHub-Api-Version': '2026-03-10'},
      signal: AbortSignal.timeout(15000), ...(data ? {body: JSON.stringify(data)} : {})});
    if (!response.ok) throw Error('GitHub request failed: ' + response.status);
    return response.status === 204 ? null : response.json();
  };
  return (event.kind === 'sweep' ? sweep : worker ? registration : capacity)(event, {env: process.env,
    now: () => Date.now() / 1000, api, uuid: randomUUID,
    preconditionFailed: error => [409, 412].includes(error.$metadata?.httpStatusCode),
    aws: async (service, operation, input, region) => {
      if (service === 'autoscaling') {
        assert(Object.hasOwn(operations, operation), 'Unsupported Auto Scaling operation ' + operation);
        return send(service, region, new operations[operation](input));
      }
      assert(service === 's3' && ['put-object', 'read-object', 'get-object'].includes(operation), 'Unsupported AWS operation ' + operation);
      if (operation === 'put-object') return send(service, region, new PutObjectCommand(input));
      const objects = await send(service, region, new ListObjectsV2Command({Bucket: input.Bucket, Prefix: input.Key, MaxKeys: 1}));
      if (objects.Contents?.[0]?.Key !== input.Key) return null;
      const object = await send(service, region, new GetObjectCommand(input)), body = JSON.parse(await object.Body.transformToString());
      return operation === 'read-object' ? {Body: body, ETag: object.ETag} : body;
    }});
};
