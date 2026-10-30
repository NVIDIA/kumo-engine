// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { execFileSync } = require('node:child_process');

async function readCopiedPullRequest({ github, context }) {
  const match = /^refs\/heads\/pull-request\/([1-9][0-9]*)$/.exec(context.ref);
  if (context.eventName !== 'push' || !match) {
    throw new Error('Expected a copy-pr-bot branch push');
  }
  const { data: pr } = await github.rest.pulls.get({
    ...context.repo,
    pull_number: Number(match[1]),
  });
  if (pr.base.repo.full_name !== `${context.repo.owner}/${context.repo.repo}` || pr.state !== 'open') {
    throw new Error('Expected an open pull request targeting this repository');
  }
  if (pr.head.sha !== context.sha) {
    throw new Error('Copied PR is stale; approve the current SHA');
  }
  return pr;
}

async function setCopiedPullRequestOutputs({ github, context, core }) {
  const pr = await readCopiedPullRequest({ github, context });
  // Diff the complete PR, excluding changes made only on the base branch.
  const base = execFileSync('git', ['merge-base', pr.base.sha, context.sha], { encoding: 'utf8' }).trim();
  core.setOutput('number', String(pr.number));
  core.setOutput('head', pr.head.sha);
  core.setOutput('base', base);
}

module.exports = { readCopiedPullRequest, setCopiedPullRequestOutputs };
