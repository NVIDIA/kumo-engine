// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { materialize } = require('./workflow-security-materialize.cjs');

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'workflow-metadata-'));
  const source = path.join(root, 'source');
  const gitDir = path.join(root, 'objects.git');
  fs.mkdirSync(source);
  execFileSync('git', ['init', '-q', source]);
  execFileSync('git', ['init', '-q', '--bare', gitDir]);
  return { root, source, gitDir, destination: path.join(root, 'candidate') };
}

function addFile(source, name, contents) {
  const target = path.join(source, name);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, contents);
}

function commitAndFetch({ source, gitDir }) {
  execFileSync('git', ['-C', source, 'add', '.']);
  execFileSync('git', ['-C', source, '-c', 'user.name=Policy Test',
    '-c', 'user.email=policy-test@example.com', 'commit', '-qm', 'fixture']);
  const sha = execFileSync('git', ['-C', source, 'rev-parse', 'HEAD']).toString().trim();
  execFileSync('git', ['-C', gitDir, 'fetch', '-q', source, sha]);
  return sha;
}

test('materializes only regular workflow and nested Action metadata from an exact commit', () => {
  const fixturePaths = fixture();
  addFile(fixturePaths.source, '.github/workflows/test.yml', 'on: push\n');
  addFile(fixturePaths.source, 'packages/api/.github/actions/check/action.yaml', 'name: Check\n');
  addFile(fixturePaths.source, 'src/untrusted.js', 'throw new Error("must not materialize")\n');
  const sha = commitAndFetch(fixturePaths);
  assert.equal(materialize(fixturePaths.gitDir, fixturePaths.destination, sha), 2);
  assert.equal(fs.readFileSync(path.join(fixturePaths.destination, '.github/workflows/test.yml'), 'utf8'), 'on: push\n');
  assert.equal(fs.existsSync(path.join(fixturePaths.destination, 'src/untrusted.js')), false);
  assert.throws(() => materialize(fixturePaths.gitDir, path.join(fixturePaths.root, 'wrong'), '0'.repeat(40)),
    /does not match/);
});

test('rejects symlinks in policy metadata', () => {
  const fixturePaths = fixture();
  addFile(fixturePaths.source, '.github/workflows/test.yml', 'on: push\n');
  const link = path.join(fixturePaths.source, '.github/actions');
  fs.symlinkSync('../workflows', link);
  const sha = commitAndFetch(fixturePaths);
  assert.throws(() => materialize(fixturePaths.gitDir, fixturePaths.destination, sha), /symlink/);
});

test('rejects oversized metadata before reading its blob', () => {
  const fixturePaths = fixture();
  addFile(fixturePaths.source, '.github/workflows/large.yml', 'x'.repeat(1024 * 1024 + 1));
  const sha = commitAndFetch(fixturePaths);
  assert.throws(() => materialize(fixturePaths.gitDir, fixturePaths.destination, sha), /too large/);
});
