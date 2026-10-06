// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const fs = require('node:fs');

function visibleLines(body) {
  let fence;
  return body.replace(/<!--[\s\S]*?(?:-->|$)/g, '').split(/\r?\n/).filter(line => {
    const marker = line.match(/^ {0,3}(`{3,}|~{3,})/);
    if (marker) {
      if (!fence) fence = marker[1];
      else if (marker[1][0] === fence[0] && marker[1].length >= fence.length &&
          /^\s*$/.test(line.slice(marker[0].length))) fence = undefined;
      return false;
    }
    return !fence;
  });
}

function cells(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '')
    .split(/(?<!\\)\|/).map(cell => cell.trim());
}

function placeholder(value) {
  const text = value.replace(/[`*_]/g, '').trim();
  return !text || /^(?:todo|tbd|n\/?a|none|\.\.\.|…|<[^>]+>|\[[^\]]+\])$/i.test(text);
}

function validateTable(pr, files, body) {
  const errors = [];
  // A path can appear more than once in the files API — e.g. a symlink converted
  // to a regular file is returned as a removed + added pair — while changed_files
  // counts it once. Compare de-duplicated filenames so a type change is not
  // mistaken for a truncated (incomplete) inventory.
  const filenames = new Set(files.map(file => file.filename));
  if (filenames.size !== pr.changed_files) {
    errors.push(`Incomplete file inventory: received ${filenames.size} of ${pr.changed_files} files.`);
  }
  const required = ['file', 'what changed', 'why'];
  if (pr.additions + pr.deletions <= 1000) required.push('function / scope');
  const lines = visibleLines(body || '');
  const start = lines.findIndex(line => /^##\s+Change map\s*$/i.test(line));
  if (start < 0) return [...errors, 'Add a ## Change map section in the dedicated comment.'];
  const section = lines.slice(start + 1);
  const end = section.findIndex(line => /^#{1,2}\s/.test(line));
  if (end >= 0) section.splice(end);
  const headerIndex = section.findIndex(line => {
    const header = cells(line).map(cell => cell.toLowerCase());
    return required.every(name => header.includes(name));
  });
  if (headerIndex < 0) return [...errors, `Required table columns: ${required.join(', ')}.`];
  const headers = cells(section[headerIndex]).map(cell => cell.toLowerCase());
  const separator = cells(section[headerIndex + 1] || '');
  if (separator.length !== headers.length || !separator.every(cell => /^:?-{3,}:?$/.test(cell))) {
    return [...errors, 'Add a Markdown table separator below the Change map header.'];
  }
  const expected = new Set(files.map(file => file.filename));
  const covered = new Set();
  for (const line of section.slice(headerIndex + 2)) {
    if (!line.includes('|') || !line.trim()) break;
    const row = cells(line);
    if (row.length !== headers.length) {
      errors.push('Change map row has the wrong number of cells; escape literal pipes as \\|.');
      continue;
    }
    const fileCell = row[headers.indexOf('file')];
    // Exact inline-code paths allow several explicitly named files in one row.
    const paths = [...fileCell.matchAll(/`([^`]+)`/g)].map(match => match[1]);
    if (!paths.length) errors.push('Put each full repository-relative file path in backticks.');
    for (const path of paths) {
      if (!expected.has(path)) errors.push(`Not a changed file: ${path}. Use the new path for a rename.`);
      else covered.add(path);
    }
    for (const name of required) {
      if (placeholder(row[headers.indexOf(name)])) errors.push(`Fill in "${name}" for ${fileCell}.`);
    }
  }
  for (const path of expected) {
    if (!covered.has(path)) errors.push(`Missing changed file: ${path}`);
  }
  return errors;
}

const COMMENT_MARKER = '<!-- kumo-pr-change-map -->';

function changeComments(pr, comments) {
  return comments.filter(comment => {
    const user = comment.user;
    const authorized = (pr.user?.id != null && user?.id === pr.user.id) ||
      (user?.type === 'Bot' && user.login === 'github-actions[bot]');
    return authorized && (comment.body || '').trimStart().startsWith(COMMENT_MARKER);
  });
}

function validate(pr, files, comments) {
  const maps = changeComments(pr, comments);
  if (maps.length !== 1) {
    return [`Expected one change-map comment starting with ${COMMENT_MARKER}; found ${maps.length} from the PR author or authorized automation (github-actions[bot]). Create it once and edit it on subsequent commits.`];
  }
  return validateTable(pr, files, maps[0].body);
}

async function run({ github, context, core }) {
  const number = context.payload.pull_request?.number || context.payload.issue.number;
  const args = { ...context.repo, pull_number: number };
  const comments = () => github.paginate(github.rest.issues.listComments, {
    ...context.repo, issue_number: number, per_page: 100,
  });
  const { data: pr } = await github.rest.pulls.get(args);
  const status = (state, description) => github.rest.repos.createCommitStatus({
    ...context.repo, sha: pr.head.sha, context: 'PR change table', state, description,
    target_url: `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`
  });
  await status('pending', 'Checking the change-map comment and changed files');
  try {
    const files = await github.paginate(github.rest.pulls.listFiles, { ...args, per_page: 100 });
    const originalComments = await comments();
    const errors = validate(pr, files, originalComments);
    const { data: latest } = await github.rest.pulls.get(args);
    if (latest.head.sha !== pr.head.sha || latest.base.sha !== pr.base.sha || JSON.stringify(changeComments(latest, await comments()).map(c => [c.id, c.body])) !==
        JSON.stringify(changeComments(pr, originalComments).map(c => [c.id, c.body]))) {
      errors.push('PR changed during validation; the queued run will check the latest version.');
    }
    await core.summary.addHeading('PR change table')
      .addCodeBlock(errors.length ? errors.join('\n') : 'Every changed file is covered.').write();
    await status(errors.length ? 'failure' : 'success', errors.length
      ? 'Update the Change map; see this run for missing files or fields'
      : 'Change map covers every changed file');
    if (errors.length) core.setFailed('PR change table validation failed');
  } catch (error) {
    await status('error', 'Could not validate the PR change table; rerun this workflow');
    throw error;
  }
}

module.exports = { validate, validateTable, run, COMMENT_MARKER };

if (require.main === module) {
  const [prPath, filesPath, commentsPath] = process.argv.slice(2);
  if (!prPath || !filesPath || !commentsPath) {
    console.error('Usage: node .github/scripts/pr-change-table.cjs PR.json FILES.json COMMENTS.json');
    process.exit(2);
  }
  const errors = validate(JSON.parse(fs.readFileSync(prPath)), JSON.parse(fs.readFileSync(filesPath)),
    JSON.parse(fs.readFileSync(commentsPath)));
  console.log(errors.length ? errors.join('\n') : 'Change map covers every changed file.');
  process.exitCode = errors.length ? 1 : 0;
}
