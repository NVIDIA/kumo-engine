// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const MAX_TREE_BYTES = 50 * 1024 * 1024;
const MAX_FILE_BYTES = 1024 * 1024;
const MAX_TOTAL_BYTES = 20 * 1024 * 1024;
const MAX_FILES = 2000;

function git(gitDir, args, maxBuffer = MAX_TREE_BYTES) {
  return execFileSync('git', ['--git-dir', gitDir, ...args], { maxBuffer });
}

function relevant(name) {
  return /^\.github\/workflows\/[^/]+\.ya?ml$/.test(name) ||
    /(^|\/)\.github\/actions\/(?:[^/]+\/)*action\.ya?ml$/.test(name);
}

function policyScope(name) {
  return name === '.github' || name.startsWith('.github/workflows') ||
    /(^|\/)\.github(?:$|\/actions(?:$|\/))/.test(name);
}

function materialize(gitDir, destination, expectedSha) {
  if (!/^[0-9a-f]{40}$/.test(expectedSha)) throw new Error('Expected a 40-character commit SHA');
  const actualSha = git(gitDir, ['rev-parse', 'FETCH_HEAD']).toString('utf8').trim();
  if (actualSha !== expectedSha) throw new Error('Fetched commit does not match the event head SHA');
  if (git(gitDir, ['cat-file', '-t', 'FETCH_HEAD']).toString('utf8').trim() !== 'commit') {
    throw new Error('Fetched object is not a commit');
  }

  const root = path.resolve(destination);
  const tree = git(gitDir, ['ls-tree', '-r', '-z', 'FETCH_HEAD']).toString('utf8');
  let count = 0;
  let totalBytes = 0;
  for (const record of tree.split('\0')) {
    if (!record) continue;
    const tab = record.indexOf('\t');
    if (tab < 0) throw new Error('Invalid Git tree record');
    const name = record.slice(tab + 1);
    const [mode, type, objectId] = record.slice(0, tab).split(' ');
    if (mode === '120000' && policyScope(name)) {
      throw new Error(`Policy metadata cannot use a symlink: ${name}`);
    }
    if (!relevant(name)) continue;
    const parts = name.split('/');
    if (path.isAbsolute(name) || parts.some((part) => part === '.' || part === '..' || !part)) {
      throw new Error(`Unsafe metadata path: ${name}`);
    }
    if (!['100644', '100755'].includes(mode) || type !== 'blob' || !/^[0-9a-f]{40}$/.test(objectId)) {
      throw new Error(`Metadata must be a regular file: ${name}`);
    }
    count += 1;
    if (count > MAX_FILES) throw new Error('Too many workflow metadata files');
    const size = Number(git(gitDir, ['cat-file', '-s', objectId]).toString('utf8').trim());
    if (!Number.isSafeInteger(size) || size > MAX_FILE_BYTES) throw new Error(`Metadata file too large: ${name}`);
    totalBytes += size;
    if (totalBytes > MAX_TOTAL_BYTES) throw new Error('Workflow metadata exceeds size limit');
    const data = git(gitDir, ['cat-file', 'blob', objectId], MAX_FILE_BYTES + 1024);
    const target = path.resolve(root, ...parts);
    if (!target.startsWith(`${root}${path.sep}`)) throw new Error(`Unsafe metadata path: ${name}`);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, data, { flag: 'wx' });
  }
  if (count === 0) throw new Error('No workflow metadata found in candidate commit');
  return count;
}

if (require.main === module) {
  const [gitDir, destination, expectedSha] = process.argv.slice(2);
  console.log(`Materialized ${materialize(gitDir, destination, expectedSha)} workflow metadata file(s).`);
}

module.exports = { materialize };
