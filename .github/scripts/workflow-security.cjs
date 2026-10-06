// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const UNTRUSTED_TRIGGERS = [
  'issues',
  'issue_comment',
  'pull_request',
  'pull_request_review',
  'pull_request_review_comment',
  'pull_request_target',
];

function workflowFiles(root) {
  const dir = path.join(root, '.github', 'workflows');
  return fs.readdirSync(dir)
    .filter((name) => /\.ya?ml$/.test(name))
    .sort()
    .map((name) => path.join(dir, name));
}

function localActionFiles(root) {
  const files = [];
  const visit = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const entryPath = path.join(dir, entry.name);
      if (!entry.isDirectory() || ['.git', 'node_modules', '.venv'].includes(entry.name)) continue;
      if (entry.name === '.github') {
        const actionsDir = path.join(entryPath, 'actions');
        if (fs.existsSync(actionsDir)) collectActions(actionsDir);
      } else {
        visit(entryPath);
      }
    }
  };
  const collectActions = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const entryPath = path.join(dir, entry.name);
      if (entry.isDirectory()) collectActions(entryPath);
      if (entry.isFile() && /^action\.ya?ml$/.test(entry.name)) files.push(entryPath);
    }
  };
  visit(root);
  return files.sort();
}

function topLevelBlock(lines, keyPattern) {
  const start = lines.findIndex((line) => keyPattern.test(line));
  if (start < 0) return [];
  const block = [];
  for (let i = start + 1; i < lines.length; i += 1) {
    if (/^[A-Za-z_"'][^:]*:/.test(lines[i])) break;
    block.push(lines[i]);
  }
  return block;
}

function finding(kind, file, line, detail) {
  const relative = file.replaceAll(path.sep, '/');
  const key = `${kind}:${relative}:${detail}`;
  return {
    id: crypto.createHash('sha256').update(key).digest('hex').slice(0, 16),
    kind,
    path: relative,
    line,
    detail,
  };
}

function scanFile(root, file) {
  const relative = path.relative(root, file);
  const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/);
  const onBlock = topLevelBlock(lines, /^(?:"on"|'on'|on):\s*$/);
  const onLine = lines.find((line) => /^(?:"on"|'on'|on):/.test(line)) || '';
  const triggerText = [onLine, ...onBlock].join('\n');
  const untrusted = UNTRUSTED_TRIGGERS.some((event) =>
    new RegExp(`(^|[\\s,{[])${event}($|[\\s,:}\\]])`).test(triggerText));
  const pullRequestTarget = /(^|[\s,{[])pull_request_target($|[\s,:}\]])/.test(triggerText);
  const permissionsBlock = topLevelBlock(lines, /^permissions:\s*$/);
  const inlinePermissions = lines.find((line) => /^permissions:\s*\{/.test(line)) || '';
  const results = [];

  lines.forEach((line, index) => {
    const lineNumber = index + 1;
    const use = line.match(/^\s*(?:-\s*)?uses:\s*([^\s#]+)/);
    if (use && !use[1].startsWith('./') && !use[1].startsWith('docker://') &&
        !/@[0-9a-fA-F]{40}$/.test(use[1])) {
      results.push(finding('mutable-action-ref', relative, lineNumber, use[1]));
    }
    if (/^\s*permissions:\s*write-all\s*$/.test(line)) {
      results.push(finding('write-all-permissions', relative, lineNumber, line.trim()));
    }
    if (/\$\{\{\s*github\.event\.[^}]*\b(body|title|message)\b[^}]*\}\}/.test(line)) {
      results.push(finding('event-text-in-expression', relative, lineNumber, line.trim()));
    }
    if (untrusted && /^\s*secrets:\s*inherit\s*$/.test(line)) {
      results.push(finding('inherited-secrets-on-untrusted-trigger', relative, lineNumber, line.trim()));
    }
  });

  if (untrusted) {
    permissionsBlock.forEach((line) => {
      const match = line.match(/^\s+([\w-]+):\s*write\s*$/);
      if (match) {
        results.push(finding(
          'top-level-write-on-untrusted-trigger', relative,
          lines.indexOf(line) + 1, `${match[1]}: write`));
      }
    });
    for (const match of inlinePermissions.matchAll(/([\w-]+):\s*write\b/g)) {
      results.push(finding(
        'top-level-write-on-untrusted-trigger', relative,
        lines.indexOf(inlinePermissions) + 1, `${match[1]}: write`));
    }
  }

  if (pullRequestTarget) {
    lines.forEach((line, index) => {
      if (/^\s*ref:\s*["']?\$\{\{[^}]*(pull_request\.head|head_ref)/.test(line)) {
        results.push(finding(
          'untrusted-checkout-on-pull-request-target', relative, index + 1, line.trim()));
      }
    });
  }
  return results;
}

function scan(root) {
  return [...workflowFiles(root), ...localActionFiles(root)]
    .flatMap((file) => scanFile(root, file))
    .sort((a, b) => a.path.localeCompare(b.path) || a.line - b.line || a.kind.localeCompare(b.kind));
}

function main(argv = process.argv.slice(2)) {
  const root = path.resolve(argv[0] || '.');
  const baselineOption = argv.indexOf('--baseline');
  const baselinePath = baselineOption < 0
    ? path.join(root, '.github', 'workflow-security-baseline.json')
    : path.resolve(argv[baselineOption + 1]);
  const findings = scan(root);
  if (argv.includes('--update-baseline')) {
    fs.writeFileSync(baselinePath, `${JSON.stringify(findings, null, 2)}\n`);
    console.log(`Recorded ${findings.length} workflow-security finding(s).`);
    return 0;
  }
  const baseline = fs.existsSync(baselinePath) ? JSON.parse(fs.readFileSync(baselinePath, 'utf8')) : [];
  const allowed = new Set(baseline.map((item) => item.id));
  const newFindings = findings.filter((item) => !allowed.has(item.id));
  if (newFindings.length) {
    for (const item of newFindings) {
      console.error(`${item.path}:${item.line}: ${item.kind}: ${item.detail} [${item.id}]`);
    }
    console.error(`${newFindings.length} new workflow-security finding(s). Fix them; do not update the baseline.`);
    return 1;
  }
  console.log(`Workflow security check passed (${findings.length} baseline finding(s), 0 new).`);
  return 0;
}

if (require.main === module) process.exitCode = main();

module.exports = { main, scan, scanFile };
