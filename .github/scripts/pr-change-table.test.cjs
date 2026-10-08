// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const test = require('node:test');
const { validate, validateTable, COMMENT_MARKER } = require('./pr-change-table.cjs');

const table = (scope = true, path = 'src/a.py') => `## Change map
| File | ${scope ? 'Function / scope | ' : ''}What changed | Why |
| --- | ${scope ? '--- | ' : ''}--- | --- |
| \`${path}\` | ${scope ? '\`read()\` | ' : ''}Retain local files | Avoid a second download |
`;
const check = (body, overrides = {}, files = [{ filename: 'src/a.py' }]) => validateTable({
  body, changed_files: files.length, additions: 500, deletions: 500, ...overrides,
}, files, body);

test('1000 lines requires scope, 1001 permits file-level descriptions', () => {
  assert.deepEqual(check(table()), []);
  assert.match(check(table(false)).join(), /Function|function/);
  assert.deepEqual(check(table(false), { additions: 501 }), []);
});
test('requires every exact path, including deletions and generated files', () => {
  const files = [{ filename: 'src/a.py' }, { filename: 'test/a.py', status: 'removed' },
    { filename: 'generated/lock.json' }];
  assert.deepEqual(check(table(), {}, files), [
    'Missing changed file: test/a.py', 'Missing changed file: generated/lock.json',
  ]);
});
test('several explicitly named files can share a row', () => {
  assert.deepEqual(check(table().replace('`src/a.py`', '`src/a.py`, `b.py`'), {},
    [{ filename: 'src/a.py' }, { filename: 'b.py' }]), []);
});
test('renames use new path', () => {
  const files = [{ filename: 'src/a.py', previous_filename: 'old.py', status: 'renamed' }];
  assert.deepEqual(check(table(), {}, files), []);
  assert.match(check(table(true, 'old.py'), {}, files).join(), /Not a changed file/);
});
test('hidden examples do not satisfy the check', () => {
  for (const body of [`<!-- ${table()} -->`, '```markdown\n' + table() + '```',
    '~~~~\n' + table() + '~~~~']) {
    assert.match(check(body).join(), /Add a ## Change map/);
  }
});
test('does not consume unrelated tables after the section', () => {
  assert.match(check('## Change map\n\n## Testing\n' + table().split('\n').slice(1).join('\n')).join(), /Required table/);
});
test('requires separator and correctly sized rows', () => {
  assert.match(check(table().replace('| --- | --- | --- | --- |', '')).join(), /separator/);
  assert.match(check(table().replace('Retain local files', 'retain | files')).join(), /wrong number/);
  assert.deepEqual(check(table().replace('Retain local files', 'retain \\| files')), []);
});
test('rejects empty cells and common placeholders', () => {
  for (const value of ['', 'TODO', 'TBD', '<reason>', '...', 'N/A']) {
    assert.match(check(table().replace('Avoid a second download', value)).join(), /Fill in "why"/);
  }
});
test('a type change reported as a removed + added pair is not an incomplete inventory', () => {
  const files = [{ filename: 'src/a.py' },
    { filename: 'pkg/x', status: 'removed' }, { filename: 'pkg/x', status: 'added' }];
  const body = table().replace('`src/a.py`', '`src/a.py`, `pkg/x`');
  assert.deepEqual(check(body, { changed_files: 2 }, files), []);
});
test('fails closed when the API inventory is incomplete', () => {
  assert.match(check(table(), { changed_files: 3001 }).join(), /Incomplete file inventory/);
});
test('missing body gives an actionable error', () => {
  assert.match(check(null).join(), /Add a ## Change map/);
});

const { run } = require('./pr-change-table.cjs');
const author = { id: 10, login: 'author', type: 'User' };
const mapComment = (id = 1, user = author) => ({ id, user, body: COMMENT_MARKER + '\n' + table() });
function harness({ latest = {}, failFiles = false, files = [{ filename: 'src/a.py' }],
  comments = [mapComment()], latestComments = comments } = {}) {
  const pr = { user: author, body: table(), changed_files: 1, additions: 1, deletions: 0,
    head: { sha: 'head' }, base: { sha: 'base' } };
  const statuses = [];
  const failures = [];
  let gets = 0;
  let commentGets = 0;
  const github = {
    rest: {
      pulls: { get: async () => ({ data: gets++ ? { ...pr, ...latest } : pr }), listFiles: Symbol('listFiles') },
      issues: { listComments: Symbol('listComments') },
      repos: { createCommitStatus: async value => statuses.push(value) },
    },
    paginate: async (method, args) => {
      if (method === github.rest.issues.listComments) {
        assert.equal(args.issue_number, 42);
        assert.equal(args.per_page, 100);
        return commentGets++ ? latestComments : comments;
      }
      assert.equal(method, github.rest.pulls.listFiles);
      assert.equal(args.per_page, 100);
      if (failFiles) throw new Error('API unavailable');
      return files;
    },
  };
  const summary = { addHeading() { return this; }, addCodeBlock() { return this; }, async write() {} };
  return { statuses, failures, args: { github,
    core: { summary, setFailed: message => failures.push(message) },
    context: { repo: { owner: 'kumo-ai', repo: 'kumo' }, payload: { pull_request: { number: 42 } },
      serverUrl: 'https://github.com', runId: 123 },
  } };
}
test('posts pending then success on the head SHA using paginated current metadata', async () => {
  const h = harness();
  await run(h.args);
  assert.deepEqual(h.statuses.map(s => [s.state, s.sha, s.context]), [
    ['pending', 'head', 'PR change table'], ['success', 'head', 'PR change table'],
  ]);
  assert.deepEqual(h.failures, []);
});
test('fork pull request validation reports through the job without posting a status', async () => {
  const valid = harness();
  await run({ ...valid.args, postStatus: false });
  assert.deepEqual(valid.statuses, []);
  assert.deepEqual(valid.failures, []);

  const invalid = harness({ files: [] });
  await run({ ...invalid.args, postStatus: false });
  assert.deepEqual(invalid.statuses, []);
  assert.equal(invalid.failures.length, 1);
});
test('missing inventory fails the commit status', async () => {
  const h = harness({ files: [] });
  await run(h.args);
  assert.equal(h.statuses.at(-1).state, 'failure');
  assert.equal(h.failures.length, 1);
});
test('concurrent head or base changes cannot publish a stale success', async () => {
  for (const latest of [{ head: { sha: 'new-head' } }, { base: { sha: 'new-base' } }]) {
    const h = harness({ latest });
    await run(h.args);
    assert.equal(h.statuses.at(-1).state, 'failure');
  }
});
test('API errors replace pending with error and fail the job', async () => {
  const h = harness({ failFiles: true });
  await assert.rejects(run(h.args), /API unavailable/);
  assert.equal(h.statuses.at(-1).state, 'error');
});

test('requires exactly one dedicated comment, never the PR description', () => {
  const pr = { user: author, body: table(), changed_files: 1, additions: 1, deletions: 0 };
  const files = [{ filename: 'src/a.py' }];
  const comment = mapComment();
  for (const comments of [[], [{ body: table() }], [comment, { ...comment, id: 2 }]]) {
    assert.match(validate(pr, files, comments).join(), /Expected one change-map comment/);
  }
  assert.deepEqual(validate(pr, files, [{ body: 'Unrelated discussion' }, comment]), []);
});
test('edits, deletions, replacement and duplicate comments invalidate an in-flight result', async () => {
  for (const latestComments of [[], [{ ...mapComment(), body: COMMENT_MARKER + '\nEdited' }],
    [mapComment(2)],
    [1, 2].map(id => mapComment(id))]) {
    const h = harness({ latestComments });
    await run(h.args);
    assert.equal(h.statuses.at(-1).state, 'failure');
  }
});
test('comment edit events validate the same PR head and do not post new comments', async () => {
  const h = harness();
  h.args.context.payload = { issue: { number: 42, pull_request: {} } };
  await run(h.args);
  assert.equal(h.statuses.at(-1).state, 'success');
});

test('closing fences require whitespace-only trailing text', () => {
  for (const fence of ['```', '~~~~']) {
    assert.match(check(`${fence}markdown\n${fence} trailing text\n${table()}${fence}`).join(), /Add a ## Change map/);
    assert.deepEqual(check(`${fence}markdown\n${fence} \t\n${table()}`), []);
  }
});
test('only the author or authorized automation supplies maps in both validation reads', async () => {
  const unrelated = mapComment(2, { id: 20, login: 'reviewer', type: 'User' });
  const otherBot = mapComment(3, { id: 30, login: 'other[bot]', type: 'Bot' });
  const impersonator = mapComment(4, { id: 40, login: 'github-actions[bot]', type: 'User' });
  const noise = [unrelated, otherBot, impersonator];
  const automation = mapComment(5, { id: 50, login: 'github-actions[bot]', type: 'Bot' });
  for (const comment of [mapComment(), automation]) {
    for (const [comments, latestComments] of [
      [[comment, ...noise], [comment, ...noise]],
      [[comment], [comment, ...noise]],
      [[comment, ...noise], [comment]],
    ]) {
      const h = harness({ comments, latestComments });
      await run(h.args);
      assert.equal(h.statuses.at(-1).state, 'success');
    }
  }
  for (const options of [
    { comments: noise },
    { comments: [{ id: 1, body: mapComment().body }] },
    { comments: [mapComment(), automation] },
    { latestComments: [unrelated] },
  ]) {
    const h = harness(options);
    await run(h.args);
    assert.equal(h.statuses.at(-1).state, 'failure');
  }
});
