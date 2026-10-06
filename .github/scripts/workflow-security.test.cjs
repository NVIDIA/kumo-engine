// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { main, scan } = require('./workflow-security.cjs');

function fixture(source) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'workflow-security-'));
  const dir = path.join(root, '.github', 'workflows');
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'test.yml'), source);
  return root;
}

function addLocalAction(root, source, prefix = '') {
  const dir = path.join(root, prefix, '.github', 'actions', 'example');
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'action.yml'), source);
}

test('accepts pinned actions in read-only pull request workflows', () => {
  const root = fixture(`"on":\n  pull_request:\npermissions:\n  contents: read\njobs:\n  test:\n    steps:\n      - uses: actions/checkout@${'a'.repeat(40)}\n`);
  assert.deepEqual(scan(root), []);
});

test('finds mutable actions and top-level writes on untrusted triggers', () => {
  const root = fixture(`"on":\n  issue_comment:\npermissions:\n  contents: write\njobs:\n  test:\n    steps:\n      - uses: actions/checkout@v6\n`);
  assert.deepEqual(scan(root).map((item) => item.kind).sort(), [
    'mutable-action-ref',
    'top-level-write-on-untrusted-trigger',
  ]);
});

test('finds event text used directly in shell expressions', () => {
  const root = fixture(`"on":\n  issues:\njobs:\n  test:\n    steps:\n      - run: echo "${'${{ github.event.issue.title }}'}"\n`);
  assert.equal(scan(root)[0].kind, 'event-text-in-expression');
});

test('finds pull_request_target checkout of an untrusted head', () => {
  const root = fixture(`"on":\n  pull_request_target:\njobs:\n  test:\n    steps:\n      - uses: actions/checkout@${'a'.repeat(40)}\n        with:\n          ref: ${'${{ github.event.pull_request.head.sha }}'}\n`);
  assert.equal(scan(root)[0].kind, 'untrusted-checkout-on-pull-request-target');
});

test('finds untrusted checkout with scalar and flow triggers', () => {
  for (const trigger of ['pull_request_target', '[push, pull_request_target]', '{pull_request_target: {}}']) {
    const root = fixture(`"on": ${trigger}\njobs:\n  test:\n    steps:\n      - uses: actions/checkout@${'a'.repeat(40)}\n        with:\n          ref: ${'${{ github.event.pull_request.head.sha }}'}\n`);
    assert.deepEqual(scan(root).map((item) => item.kind), [
      'untrusted-checkout-on-pull-request-target',
    ]);
  }
});

test('rejects untrusted checkout even in the policy workflow', () => {
  const root = fixture(`"on": push\njobs: {}\n`);
  const workflow = path.join(root, '.github', 'workflows', 'workflow-security.yml');
  const source = `"on": pull_request_target\njobs:\n  validate:\n    steps:\n      - name: Check out candidate files\n        uses: actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10 # v6\n        with:\n          ref: ${'${{ github.event.pull_request.head.sha || github.sha }}'}\n          path: candidate\n          persist-credentials: false\n`;
  fs.writeFileSync(workflow, source);
  assert.equal(scan(root)[0].kind, 'untrusted-checkout-on-pull-request-target');
});

test('finds top-level writes in flow permissions', () => {
  const root = fixture(`"on": issue_comment\npermissions: {contents: write, issues: write}\njobs: {}\n`);
  assert.deepEqual(scan(root).map((item) => item.detail), [
    'contents: write',
    'issues: write',
  ]);
});

test('finds double-quoted pull_request_target checkout of an untrusted head', () => {
  const root = fixture(`"on":
  pull_request_target:
jobs:
  test:
    steps:
      - uses: actions/checkout@${'a'.repeat(40)}
        with:
          ref: "${'${{ github.head_ref }}'}"
`);
  assert.equal(scan(root)[0].kind, 'untrusted-checkout-on-pull-request-target');
});

test('finds single-quoted pull_request_target checkout of an untrusted head', () => {
  const root = fixture(`"on":
  pull_request_target:
jobs:
  test:
    steps:
      - uses: actions/checkout@${'a'.repeat(40)}
        with:
          ref: '${'${{ github.event.pull_request.head.sha }}'}'
`);
  assert.equal(scan(root)[0].kind, 'untrusted-checkout-on-pull-request-target');
});

test('finds mutable actions used by local composite actions', () => {
  const root = fixture(`"on":\n  push:\njobs:\n  test:\n    steps:\n      - uses: ./\.github/actions/example\n`);
  addLocalAction(root, `name: Example\nruns:\n  using: composite\n  steps:\n    - uses: actions/setup-node@v6\n`);
  assert.deepEqual(scan(root).map((item) => [item.kind, item.path]), [
    ['mutable-action-ref', '.github/actions/example/action.yml'],
  ]);
});

test('finds mutable actions in nested repository local actions', () => {
  const root = fixture(`"on": push\njobs: {}\n`);
  addLocalAction(root, `name: Example\nruns:\n  using: composite\n  steps:\n    - uses: actions/setup-node@v6\n`, 'packages/kumo-api');
  assert.deepEqual(scan(root).map((item) => [item.kind, item.path]), [
    ['mutable-action-ref', 'packages/kumo-api/.github/actions/example/action.yml'],
  ]);
});

test('uses the trusted baseline instead of a candidate baseline', () => {
  const root = fixture(`"on": push\njobs:\n  test:\n    steps:\n      - uses: actions/checkout@v6\n`);
  const finding = scan(root);
  fs.writeFileSync(path.join(root, '.github', 'workflow-security-baseline.json'), JSON.stringify(finding));
  const trustedBaseline = path.join(os.tmpdir(), `workflow-security-baseline-${process.pid}.json`);
  fs.writeFileSync(trustedBaseline, '[]');
  try {
    assert.equal(main([root]), 0);
    assert.equal(main([root, '--baseline', trustedBaseline]), 1);
  } finally {
    fs.unlinkSync(trustedBaseline);
  }
});
