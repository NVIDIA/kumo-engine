// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const fs = require('node:fs');
const path = require('node:path');
const { isDeepStrictEqual } = require('node:util');
const { createHash } = require('node:crypto');
const graph = require('../snowflake_notebook/graph_ui.cjs');
const { safeSnapshot, frontendFailure, VerificationError, verificationMessage } = require('./assertions.cjs');

function requireThat(value, message) {
  if (!value) throw new VerificationError(message);
}

function safePage(url, expectedUrl) {
  try {
    const parsed = new URL(url), expected = new URL(expectedUrl);
    return parsed.protocol === 'https:' && parsed.username === '' && parsed.password === '' &&
      parsed.search === '' && expected.protocol === 'https:' &&
      /^[a-z0-9-]+\.studio\.us-west-2\.sagemaker\.aws$/.test(expected.hostname) &&
      expected.pathname === '/jupyterlab/default' && parsed.origin === expected.origin &&
      (parsed.pathname === expected.pathname || parsed.pathname.startsWith(expected.pathname + '/'));
  } catch { return false; }
}

async function jupyter(page) {
  const deadline = Date.now() + 120000;
  do {
    for (const frame of page.frames()) {
      if (await frame.locator('.jp-FileBrowser').isVisible().catch(() => false)) return frame;
    }
    await page.waitForTimeout(250);
  } while (Date.now() < deadline);
  throw new VerificationError('The dedicated Studio JupyterLab file browser did not open');
}

async function handleKernelDialogs(page, frame, mark) {
  const dialog = frame.getByRole('dialog').filter({ has: frame.getByRole('button', { name: 'No Kernel', exact: true }) });
  await page.addLocatorHandler(dialog, async () => {
    mark('native-dismiss-kernel-dialog');
    await dialog.getByRole('button', { name: 'No Kernel', exact: true }).click();
  }, { times: 20, noWaitAfter: true });
}

async function openNotebook(frame, filename, mark) {
  mark('native-file-visible');
  const item = frame.locator('.jp-FileBrowser').getByText(filename, { exact: true });
  await item.waitFor({ state: 'visible' });
  mark('native-file-open');
  await item.dblclick();
  mark('native-kernel-selection');
  const kernelDialog = frame.getByRole('dialog').filter({ has: frame.getByRole('button', { name: 'No Kernel', exact: true }) });
  await kernelDialog.waitFor({ state: 'visible', timeout: 15000 }).catch(() => {});
  mark('native-notebook-panel');
  const panels = frame.getByRole('tabpanel', { name: filename, exact: true });
  await panels.waitFor({ state: 'visible' });
  await panels.click({ position: { x: 2, y: 2 } });
  await panels.getByRole('button', { name: 'No Kernel', exact: true }).waitFor({ state: 'visible' });
  requireThat(await panels.count() === 1, 'Expected one selected native Studio notebook');
  return panels.first();
}

async function trustNotebook(page, frame) {
  await page.keyboard.press('Control+Shift+C');
  const palette = frame.getByRole('region', { name: 'Command Palette Section', exact: true });
  await palette.waitFor({ state: 'visible' });
  await palette.getByRole('textbox', { name: 'SEARCH', exact: true }).fill('Trust Notebook');
  const command = palette.getByRole('menuitem', { name: 'Trust Notebook', exact: true });
  if (await command.isDisabled()) { await page.keyboard.press('Escape'); return; }
  await command.click();
  const dialog = frame.getByRole('dialog').filter({ hasText: /trust.*notebook/i });
  await dialog.waitFor({ state: 'visible' });
  await dialog.getByRole('button', { name: /^(Confirm Trusting this notebook|Trust)$/ }).click();
  await dialog.waitFor({ state: 'hidden' });
}

async function verifyReceipt(panel, expected) {
  const outputs = await panel.locator('.jp-OutputArea-output').allTextContents();
  const marker = 'KUMO_SAGEMAKER_RECEIPT=';
  const receipts = outputs.flatMap(text => text.split('\n').filter(line => line.startsWith(marker)));
  requireThat(receipts.length === 1, 'Native notebook must show exactly one original execution receipt');
  const actual = JSON.parse(receipts[0].slice(marker.length));
  requireThat(isDeepStrictEqual(actual, expected), 'Native notebook execution receipt differs from the producer');
  return { attempt: actual.attempt, execution_id: actual.execution_id,
    source_commit: actual.source_commit, wheel_sha256: actual.wheel_sha256 };
}

async function writeSaveProbe(page, frame, panel, saveProbe, mark) {
  mark('native-save-probe-edit');
  const notebook = panel.locator('.jp-Notebook');
  await notebook.hover({ position: { x: 5, y: 5 } });
  await page.mouse.wheel(0, -1000000);
  const cells = panel.locator('.jp-MarkdownCell');
  const fixture = cells.filter({ hasText: 'Studio save probe.' });
  await fixture.waitFor({ state: 'visible' });
  requireThat(await fixture.count() === 1, 'Expected one reserved Studio save-probe cell');
  const modelIndex = await fixture.getAttribute('data-windowed-list-index');
  const position = await cells.evaluateAll(nodes => nodes.findIndex(node => node.textContent.includes('Studio save probe.')));
  requireThat(position >= 0, 'Missing reserved Studio save-probe cell');
  const cell = modelIndex !== null && /^\d+$/.test(modelIndex)
    ? panel.locator(`.jp-MarkdownCell[data-windowed-list-index="${modelIndex}"]`)
    : cells.nth(position);
  await cell.dblclick();
  const editor = cell.locator('.cm-content[contenteditable="true"]');
  await editor.click();
  await page.keyboard.press('Control+a');
  await page.keyboard.insertText(saveProbe);
  await editor.getByText(saveProbe, { exact: true }).waitFor({ state: 'visible' });
  await frame.locator('.lm-TabBar-tab.lm-mod-current.jp-mod-dirty').waitFor({ state: 'visible' });
  mark('save-native-notebook');
  const toolbar = panel.getByRole('toolbar', { name: 'main area toolbar', exact: true });
  await toolbar.getByRole('button', { name: /^Save and create checkpoint/ }).click();
  await frame.locator('.lm-TabBar-tab.lm-mod-current.jp-mod-dirty').waitFor({ state: 'hidden' });
}

async function verifySaveProbe(panel, saveProbe, mark) {
  mark('native-save-probe-reopened');
  const cells = panel.locator('.jp-MarkdownCell').filter({ hasText: saveProbe });
  await cells.waitFor({ state: 'visible' });
  await cells.locator('.jp-RenderedHTMLCommon:visible, .cm-content:visible').first().getByText(saveProbe, { exact: true }).waitFor({ state: 'visible' });
  requireThat(await cells.count() === 1, 'Reopened notebook does not contain one new saved edit');
}

async function main(input) {
  const { chromium } = require('../snowflake_notebook/browser/node_modules/playwright');
  const output = path.resolve(input.output);
  fs.mkdirSync(output, { recursive: false });
  const report = { status: 'failed', training_submitted: false,
    scope: 'Interactive rendering and saved/reopened outputs in native Studio' };
  function mark(stage) {
    report.stage = stage;
    fs.writeFileSync(path.join(output, 'progress.json'), JSON.stringify({ stage }));
  }
  const browser = await chromium.launch({ headless: true });
  let page, authenticated = false;
  const errors = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1700, height: 1200 } });
    page = await context.newPage();
    page.on('pageerror', e => errors.push(frontendFailure(e, report.stage)));
    mark('studio-authentication');
    await page.goto(input.url, { waitUntil: 'domcontentloaded' });
    const location = new URL(page.url());
    report.authentication = {
      https: location.protocol === 'https:',
      aws_studio_host: location.hostname.endsWith('.sagemaker.aws'),
      space_origin: location.origin === new URL(input.space_url).origin,
      clean_search: location.search === '',
      path_class: location.pathname.includes('jupyterlab') ? 'jupyterlab' : 'other',
    };
    let frame = await jupyter(page);
    report.authentication.jupyter_visible = true;
    requireThat(safePage(page.url(), input.space_url), 'Studio did not reach its clean authenticated test-domain URL');
    authenticated = true;
    await handleKernelDialogs(page, frame, mark);
    mark('upload-retained-notebook');
    const content = fs.readFileSync(input.notebook);
    requireThat(createHash('sha256').update(content).digest('hex') === input.notebook_sha256,
      'Retained notebook changed before native Studio upload');
    const filename = `kumo-test-${input.receipt.attempt}-ui-${input.invocation.slice(0, 12)}.ipynb`;
    const upload = { name: filename, mimeType: 'application/x-ipynb+json', buffer: content };
    const fileInput = frame.locator('.jp-FileBrowser input[type="file"]');
    if (await fileInput.count()) await fileInput.first().setInputFiles(upload);
    else {
      const [chooser] = await Promise.all([page.waitForEvent('filechooser'),
        frame.getByRole('button', { name: /Upload Files/i }).click()]);
      await chooser.setFiles(upload);
    }
    let panel = await openNotebook(frame, filename, mark);
    mark('trust-retained-notebook');
    await trustNotebook(page, frame);
    mark('native-notebook-receipt');
    report.initial_receipt = await verifyReceipt(panel, input.receipt);
    mark('native-graph-controls');
    if (input.graph) report.initial = await graph.check(page, input.graph, output, 'studio-initial', true, true, async () => authenticated && safePage(page.url(), input.space_url), { root: panel, frame });
    const saveProbe = `KUMO_SAVE_PROBE=${input.receipt.attempt}-${input.invocation}`;
    requireThat(!content.toString().includes(saveProbe), 'Save probe already exists in the upload');
    await writeSaveProbe(page, frame, panel, saveProbe, mark);
    mark('close-native-notebook-tab');
    await frame.getByRole('menuitem', { name: /^File$/ }).click();
    await frame.getByRole('menuitem', { name: /^Close Tab(?:\s|$)/ }).click();
    const cookies = await context.cookies();
    const cleanUrl = page.url();
    mark('close-saved-notebook-context');
    await context.close();
    const fresh = await browser.newContext({ viewport: { width: 1700, height: 1200 } });
    await fresh.addCookies(cookies);
    page = await fresh.newPage();
    page.on('pageerror', e => errors.push(frontendFailure(e, report.stage)));
    mark('reopen-native-notebook');
    await page.goto(cleanUrl, { waitUntil: 'domcontentloaded' });
    frame = await jupyter(page);
    await handleKernelDialogs(page, frame, mark);
    requireThat(safePage(page.url(), input.space_url), 'Reopened Studio notebook lost its test-domain identity');
    panel = await openNotebook(frame, filename, mark);
    await verifySaveProbe(panel, saveProbe, mark);
    report.save_probe = 'passed';
    report.reopened_receipt = await verifyReceipt(panel, input.receipt);
    if (input.graph) report.reopened = await graph.checkSaved(page, input.graph, output, async () => authenticated && safePage(page.url(), input.space_url), { root: panel, frame });
    requireThat(errors.length === 0, 'Native Studio browser emitted an uncaught frontend error');
    report.uploaded_notebook = filename;
    report.uploaded_sha256 = input.notebook_sha256;
    report.status = 'passed';
  } catch (error) {
    report.error = verificationMessage(error);
    if (authenticated && page && safePage(page.url(), input.space_url)) {
      if (/^(native-(file|kernel|restored|notebook|dismiss|graph)|trust-)/.test(report.stage)) {
        report.diagnostic_detail = safeSnapshot(String(error.message), [input.url]).slice(0, 1500);
      }
      await graph.captureElement(page, output, 'failed',
        async () => authenticated && safePage(page.url(), input.space_url),
        { ariaSnapshot: async () => safeSnapshot(await page.locator('body').ariaSnapshot(), [input.url]) }
      ).catch(() => {});
    }
  } finally {
    report.browser_errors = errors;
    await browser.close();
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(report, null, 2));
  }
  process.exitCode = report.status === 'passed' ? 0 : 1;
}

if (require.main === module) {
  let text = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', chunk => text += chunk);
  process.stdin.on('end', () => main(JSON.parse(text)).catch(() => {
    process.stderr.write('Native Studio browser could not start.\n'); process.exitCode = 1;
  }));
}
module.exports = { safePage, writeSaveProbe, verifySaveProbe };
