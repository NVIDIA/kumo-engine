// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const {createSign} = require('node:crypto');
const assert = require('node:assert/strict');
const {capacity, registration} = require('./engine-e2e-pool.cjs');
const {SecretsManagerClient, GetSecretValueCommand} = require('@aws-sdk/client-secrets-manager');
const {S3Client, PutObjectCommand, GetObjectCommand} = require('@aws-sdk/client-s3');
const {AutoScalingClient, DescribeAutoScalingGroupsCommand, SuspendProcessesCommand} = require('@aws-sdk/client-auto-scaling');
const secrets = new SecretsManagerClient(), s3 = new S3Client(), autoscaling = new AutoScalingClient();
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
      ...(data ? {body: JSON.stringify(data)} : {})});
    if (!response.ok) throw Error('GitHub request failed: ' + response.status);
    return response.status === 204 ? null : response.json();
  };
  return (worker ? registration : capacity)(event, {env: process.env, now: () => Date.now() / 1000, api,
    preconditionFailed: error => error.$metadata?.httpStatusCode === 412,
    aws: async (service, operation, input) => {
      if (service === 'autoscaling') return autoscaling.send(new (operation === 'describe-auto-scaling-groups' ?
        DescribeAutoScalingGroupsCommand : SuspendProcessesCommand)(input));
      if (operation === 'put-object') return s3.send(new PutObjectCommand(input));
      try { return JSON.parse(await (await s3.send(new GetObjectCommand(input))).Body.transformToString()); }
      catch (error) { if (error.$metadata?.httpStatusCode === 404) return null; throw error; }
    }});
};
