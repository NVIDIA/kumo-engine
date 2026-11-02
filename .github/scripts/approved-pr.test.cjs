// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { readCopiedPullRequest, setCopiedPullRequestOutputs } = require('./copied-pr.cjs');
const { reportOpenAPI } = require('./openapi-pr.cjs');

function fixture({ changed = true, comments = [], fork = true } = {}) {
  const context = {
    eventName: 'push', ref: 'refs/heads/pull-request/42', sha: 'a'.repeat(40),
    repo: { owner: 'example', repo: 'engine' }, serverUrl: 'https://github.com', runId: 123,
  };
  const pr = {
    number: 42, state: 'open',
    head: { sha: context.sha, repo: { full_name: fork ? 'contributor/engine' : 'example/engine' } },
    base: { sha: 'b'.repeat(40), repo: { full_name: 'example/engine' } },
  };
  const writes = [];
  const github = {
    rest: {
      pulls: { get: async (args) => { assert.equal(args.pull_number, 42); return { data: pr }; } },
      issues: {
        listComments: () => {},
        createComment: async (args) => writes.push({ kind: 'create', ...args }),
        updateComment: async (args) => writes.push({ kind: 'update', ...args }),
      },
    },
    paginate: async () => comments,
  };
  return { github, context, changed, pr, writes };
}

test('only current copied commits of open PRs targeting this repository are accepted', async () => {
  const f = fixture();
  assert.equal(await readCopiedPullRequest(f), f.pr);
  for (const [field, value] of [['eventName', 'pull_request'], ['ref', 'refs/heads/main'], ['ref', 'refs/heads/pull-request/42/extra']]) {
    await assert.rejects(readCopiedPullRequest({ ...f, context: { ...f.context, [field]: value } }), /copy-pr-bot/);
  }
  f.pr.head.sha = 'c'.repeat(40);
  await assert.rejects(readCopiedPullRequest(f), /stale/);
  f.pr.head.sha = f.context.sha;
  f.pr.state = 'closed';
  await assert.rejects(readCopiedPullRequest(f), /open pull request/);
  f.pr.state = 'open';
  f.pr.base.repo.full_name = 'other/engine';
  await assert.rejects(readCopiedPullRequest(f), /this repository/);
});

test('merge-base selection retains earlier PR changes and excludes independent base changes', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'copied-pr-base-'));
  const previous = process.cwd();
  const git = (...args) => execFileSync('git', ['-c', 'user.name=CI Test', '-c', 'user.email=ci@example.test',
    '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null', ...args], { cwd: dir, encoding: 'utf8' }).trim();
  const commit = (file) => {
    fs.writeFileSync(path.join(dir, file), file);
    git('add', file);
    git('commit', '-qm', file);
    return git('rev-parse', 'HEAD');
  };
  try {
    git('init', '-qb', 'main');
    const common = commit('base.txt');
    git('checkout', '-qb', 'feature');
    commit('service.py');
    const head = commit('README.md');
    git('checkout', '-q', 'main');
    const base = commit('unrelated-gcp.py');
    git('checkout', '-q', 'feature');
    const f = fixture();
    f.context.sha = f.pr.head.sha = head;
    f.pr.base.sha = base;
    const outputs = {};
    process.chdir(dir);
    await setCopiedPullRequestOutputs({ ...f, core: { setOutput: (key, value) => { outputs[key] = value; } } });
    assert.deepEqual(outputs, { number: '42', head, base: common });
    assert.deepEqual(git('diff', '--name-only', `${outputs.base}..${head}`).split('\n'), ['README.md', 'service.py']);
  } finally {
    process.chdir(previous);
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('fork warning links the analyzed SHA, schema patch, and local generation command', async () => {
  const f = fixture();
  await reportOpenAPI(f);
  assert.equal(f.writes.length, 1);
  assert.equal(f.writes[0].issue_number, 42);
  assert.match(f.writes[0].body, /<!-- kumo-openapi-check -->/);
  assert.match(f.writes[0].body, /make generate-ui-client-code/);
  assert.match(f.writes[0].body, /push.*fork/);
  assert.match(f.writes[0].body, /actions\/runs\/123/);
  assert.match(f.writes[0].body, /View the schema patch in the workflow run/);
  assert.ok(f.writes[0].body.includes(f.context.sha));
  assert.doesNotMatch(f.writes[0].body, /trigger-ui-codegen/);
});

test('same-repository warning preserves the authorized code-generation command', async () => {
  const f = fixture({ fork: false });
  await reportOpenAPI(f);
  assert.match(f.writes[0].body, /trigger-ui-codegen/);
});

test('an existing bot warning is updated when schemas become current', async () => {
  const f = fixture({ changed: false, comments: [
    { id: 1, user: { login: 'contributor' }, body: '<!-- kumo-openapi-check --> spoof' },
    { id: 2, user: { login: 'github-actions[bot]' }, body: '⚠️ **OpenAPI spec changes detected in this PR**' },
  ] });
  await reportOpenAPI(f);
  assert.equal(f.writes[0].kind, 'update');
  assert.equal(f.writes[0].comment_id, 2);
  assert.match(f.writes[0].body, /up to date/);
});

test('a repeated warning updates the bot comment instead of creating duplicates', async () => {
  const f = fixture({ comments: [{ id: 7, user: { login: 'github-actions[bot]' }, body: '<!-- kumo-openapi-check --> old' }] });
  await reportOpenAPI(f);
  assert.equal(f.writes[0].kind, 'update');
  assert.equal(f.writes[0].comment_id, 7);
});

test('current schemas do not create an unsolicited success comment', async () => {
  const f = fixture({ changed: false });
  await reportOpenAPI(f);
  assert.equal(f.writes.length, 0);
});

test('a PR update during reporting prevents a stale write', async () => {
  const f = fixture();
  f.github.paginate = async () => {
    f.pr.head.sha = 'c'.repeat(40);
    return [];
  };
  await assert.rejects(reportOpenAPI(f), /stale/);
  assert.equal(f.writes.length, 0);
});
