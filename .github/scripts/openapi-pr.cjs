// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { readCopiedPullRequest } = require('./copied-pr.cjs');

const MARKER = '<!-- kumo-openapi-check -->';
const LEGACY_HEADING = '⚠️ **OpenAPI spec changes detected in this PR**';

async function reportOpenAPI({ github, context, changed }) {
  const pr = await readCopiedPullRequest({ github, context });
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...context.repo,
    issue_number: pr.number,
    per_page: 100,
  });
  const existing = comments.find((comment) => comment.user?.login === 'github-actions[bot]' &&
    (comment.body?.startsWith(MARKER) || comment.body?.startsWith(LEGACY_HEADING)));
  if (!changed && !existing) return;

  const run = `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`;
  const instructions = pr.head.repo.full_name === pr.base.repo.full_name
    ? 'Run `make generate-ui-client-code` locally, or comment `/trigger-ui-codegen` if you have repository write access.'
    : 'Run `make generate-ui-client-code` locally and push the regenerated files to your fork.';
  const body = [
    MARKER,
    changed ? '⚠️ **OpenAPI schema regeneration needed**' : '✅ **OpenAPI schemas are up to date**',
    '',
    `Analyzed commit: \`${context.sha}\`.`,
    '',
    changed ? `${instructions} [View the schema patch in the workflow run](${run}).`
      : `[View the workflow run](${run}).`,
  ].join('\n');

  // The PR may have advanced while comments were being fetched.
  await readCopiedPullRequest({ github, context });
  if (existing) {
    await github.rest.issues.updateComment({ ...context.repo, comment_id: existing.id, body });
  } else {
    await github.rest.issues.createComment({ ...context.repo, issue_number: pr.number, body });
  }
}

module.exports = { reportOpenAPI };
