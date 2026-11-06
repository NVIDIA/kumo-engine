// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require('../snowflake_notebook/browser/node_modules/playwright');
const graph = require('../snowflake_notebook/graph_ui.cjs');
const { historyPoint, renderedGeometry, targetPosition, observedCoordinates,
  historyCsv, VerificationError, verificationMessage, captureAllowed } = require('./assertions.cjs');

function requireThat(value, message) {
  if (!value) throw new VerificationError(message);
}

async function metricValues(page, reference, mark) {
  const observed = {};
  for (const [name, expected] of Object.entries(reference.metrics)) {
    mark('metric-row:' + name);
    const link = page.getByRole('link', { name, exact: true });
    const row = page.getByRole('row').filter({ has: link });
    await row.waitFor({ state: 'visible' });
    await row.scrollIntoViewIfNeeded();
    requireThat((await row.count()) === 1, 'Ambiguous metric row');
    const text = (await row.getByRole('cell').last().innerText()).trim();
    const value = Number(text);
    requireThat(text !== '' && Number.isFinite(value) &&
      Math.abs(value - expected) <= 1e-9 * Math.max(1, Math.abs(expected)),
      'Rendered metric differs from the recorded value');
    observed[name] = value;
  }
  return observed;
}

async function epochHistory(page, metric, reference, mark, output, label, capture) {
  const expected = reference.history[metric];
  mark('history-link:' + metric);
  const link = page.getByRole('link', { name: metric, exact: true });
  capture.clear();
  await link.click();
  mark('history-heading:' + metric);
  await page.getByRole('heading', { name: metric, exact: true }).waitFor({ state: 'visible' });
  const identity = page.getByRole('link', { name: reference.run_name, exact: true });
  await identity.waitFor({ state: 'visible' });
  await capture.set(page, identity);
  mark('history-step-axis:' + metric);
  await page.getByRole('radio', { name: 'Step', exact: true }).check();
  mark('history-show-points:' + metric);
  const showPoints = page.getByTestId('show-point-toggle');
  if (await showPoints.getAttribute('aria-checked') !== 'true') await showPoints.click();
  requireThat(await showPoints.getAttribute('aria-checked') === 'true',
    'Metric history points control did not turn on');
  const plot = page.locator('.js-plotly-plot');
  await plot.waitFor({ state: 'visible' });
  await plot.scrollIntoViewIfNeeded();
  const box = await plot.boundingBox();
  requireThat(box && box.width > 100 && box.height > 100, 'Metric history has no visible chart');
  const axes = await plot.evaluate(renderedGeometry);
  mark('history-download:' + metric);
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: /^Download data/ }).click(),
  ]);
  const downloadedPath = await download.path();
  requireThat(fs.statSync(downloadedPath).size <= 1 << 20, 'UI history download exceeds its size budget');
  const csv = fs.readFileSync(downloadedPath, 'utf8');
  fs.writeFileSync(path.join(output, `history-${label}-${metric.replaceAll('/', '_')}.csv`), csv);
  mark('history-download-content:' + metric);
  const downloaded = historyCsv(csv, reference.run, metric, expected);
  const values = [];
  for (const [index, [step, expectedValue]] of expected.entries()) {
    mark(`history-coordinate:${metric}:${index}`);
    const target = targetPosition(axes, step, expectedValue);
    requireThat(Number.isFinite(target.x) && Number.isFinite(target.y) &&
      target.x >= box.x && target.x <= box.x + box.width &&
      target.y >= box.y && target.y <= box.y + box.height,
      'Expected epoch is outside the rendered metric chart');
    await page.mouse.move(0, 0);
    const tooltip = plot.locator('.hoverlayer g.hovertext');
    await tooltip.first().waitFor({ state: 'hidden' });
    await page.mouse.move(target.x, target.y);
    mark(`history-tooltip:${metric}:${index}`);
    await tooltip.first().waitFor({ state: 'visible' });
    requireThat(await tooltip.count() === 1 &&
      (await tooltip.locator('text.name').textContent()).trim() === metric,
      'Epoch tooltip identifies another metric');
    const geometry = await tooltip.evaluate(renderedGeometry);
    requireThat(Math.abs(geometry.origin.x - target.x) <= 2 &&
      Math.abs(geometry.origin.y - target.y) <= 2,
      'Epoch tooltip is not anchored at the intended rendered point');
    const labels = [await tooltip.locator('text.nums').textContent()];
    values.push({ ...historyPoint(observedCoordinates(geometry), step, expectedValue, labels),
      hover_target: target, observed_anchor: geometry.origin });
  }
  return { downloaded, points: values, method: 'ui_csv_and_rendered_hover_anchors' };
}

async function main(input) {
  const root = path.resolve(input.directory);
  const output = path.join(root, 'browser');
  fs.mkdirSync(output, { recursive: false });
  const report = { status: 'failed', notebook_export: 'not selected', native_mlflow: 'not selected' };
  function mark(assertion) {
    report.assertion = assertion;
    fs.writeFileSync(path.join(output, 'progress.json'), JSON.stringify({
      stage: report.stage, assertion,
    }));
  }
  const browser = await chromium.launch({ headless: true });
  let safeFailure, trustedBase;
  const capture = {
    clear() { safeFailure = undefined; },
    async set(page, identity) {
      const url = new URL(page.url());
      if (url.origin === trustedBase.origin && url.pathname === trustedBase.pathname &&
          url.search === '' && await identity.isVisible()) {
        safeFailure = { page, identity, expectedUrl: url.href };
      }
    },
  };
  const pageErrors = [];
  async function pageIn(context) {
    const page = await context.newPage();
    page.on('pageerror', () => { pageErrors.push({ stage: report.stage, type: 'pageerror' }); });
    return page;
  }
  function noErrors() {
    requireThat(pageErrors.length === 0, 'Browser emitted an uncaught error');
  }
  let server;
  try {
    if (input.graph) {
      report.stage = 'notebook-export';
      const receipt = JSON.parse(fs.readFileSync(path.join(root, 'outputs', 'graph-result.json')));
      server = http.createServer((request, response) => {
        if (request.url !== '/notebook.html') {
          response.writeHead(404).end();
          return;
        }
        response.setHeader('Content-Type', 'text/html; charset=utf-8');
        fs.createReadStream(path.join(root, 'notebook.html')).pipe(response);
      });
      await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
      const url = `http://127.0.0.1:${server.address().port}/notebook.html`;
      const context = await browser.newContext({ viewport: { width: 1600, height: 1100 } });
      const page = await pageIn(context);
      await page.goto(url, { waitUntil: 'domcontentloaded' });
      const first = await graph.checkSaved(page, receipt, output);
      noErrors();
      await context.close();
      report.stage = 'notebook-export-reopened';
      const freshContext = await browser.newContext({ viewport: { width: 1600, height: 1100 } });
      const reopened = await pageIn(freshContext);
      await reopened.goto(url, { waitUntil: 'domcontentloaded' });
      const second = await graph.checkSaved(reopened, receipt, output);
      noErrors();
      await freshContext.close();
      report.notebook_export = { status: 'passed', scope: 'saved-export', initial: first, reopened: second };
    }
    if (input.mlflow_url) {
      const expected = input.expected_mlflow;
      requireThat(expected.runs.length === 6 && expected.versions.length === 2,
        'Missing complete lifecycle UI expectations');
      report.stage = 'mlflow-authentication';
      mark('authentication');
      const context = await browser.newContext({ viewport: { width: 1600, height: 1100 } });
      const page = await pageIn(context);
      page.setDefaultTimeout(60000);
      capture.clear();
      await page.goto(input.mlflow_url, { waitUntil: 'domcontentloaded' });
      await page.getByRole('link', { name: /experiments/i }).first().waitFor();
      const base = new URL(page.url());
      base.search = '';
      base.hash = '';
      trustedBase = base;
      const observations = [];
      report.native_mlflow = { status: 'failed', observations, registry: 'managed MLflow' };
      for (const [index, run] of expected.runs.entries()) {
        const reference = run.reference;
        report.stage = `mlflow-${run.roles.join('-')}-${index}`;
        mark('run-navigation');
        const url = new URL(base);
        url.hash = `/experiments/${encodeURIComponent(reference.experiment_id)}/runs/${encodeURIComponent(reference.run)}`;
        capture.clear();
        await page.goto(url.href, { waitUntil: 'domcontentloaded' });
        mark('run-name');
        const identity = page.getByText(reference.run_name, { exact: true }).first();
        await identity.waitFor({ state: 'visible' });
        mark('run-finished-status');
        await page.getByText('Finished', { exact: true }).first().waitFor({ state: 'visible' });
        await capture.set(page, identity);
        const metrics = await metricValues(page, reference, mark);
        await page.screenshot({ path: path.join(output, `mlflow-run-${index}.png`), fullPage: true });
        const history = {};
        observations.push({ run: reference.run, roles: run.roles, status: 'FINISHED', metrics, history });
        if (run.roles.includes('fit')) {
          for (const metric of ['train/loss', 'val/auroc']) {
            const historyPage = await pageIn(context);
            capture.clear();
            await historyPage.goto(url.href, { waitUntil: 'domcontentloaded' });
            history[metric] = await epochHistory(historyPage, metric, reference, mark, output, index, capture);
            await historyPage.screenshot({ path: path.join(output, `mlflow-history-${index}-${Object.keys(history).length}.png`), fullPage: true });
            await historyPage.close();
            await capture.set(page, identity);
          }
        }
        noErrors();
      }
      const versions = [];
      for (const version of expected.versions) {
        report.stage = `mlflow-version-${version.version}`;
        const url = new URL(base);
        url.hash = `/models/${encodeURIComponent(expected.model)}/versions/${encodeURIComponent(version.version)}`;
        capture.clear();
        await page.goto(url.href, { waitUntil: 'domcontentloaded' });
        await page.getByText(expected.model, { exact: true }).first().waitFor({ state: 'visible' });
        await page.getByText(new RegExp(`^Version[ ]*${version.version}$`)).first().waitFor({ state: 'visible' });
        await page.getByText(version.bundle, { exact: true }).first().waitFor({ state: 'visible' });
        await capture.set(page, page.getByText(version.bundle, { exact: true }).first());
        await page.screenshot({ path: path.join(output, `mlflow-version-${version.version}.png`), fullPage: true });
        noErrors();
        versions.push({ version: version.version, bundle: version.bundle, model: expected.model });
      }
      report.native_mlflow = { status: 'passed', observations, versions, registry: 'managed MLflow' };
      await context.close();
    }
    report.status = 'passed';
  } catch (error) {
    // Browser errors can contain a presigned URL. Retain no raw error or trace.
    report.error = verificationMessage(error);
    if (safeFailure && await safeFailure.identity.isVisible().catch(() => false) &&
        captureAllowed(safeFailure.page.url(), safeFailure.expectedUrl, true)) {
      await safeFailure.page.screenshot({ path: path.join(output, 'failed.png'), fullPage: true }).catch(() => {});
    }
  } finally {
    report.browser_errors = pageErrors;
    if (pageErrors.length) report.status = 'failed';
    await browser.close();
    if (server) await new Promise(resolve => server.close(resolve));
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(report, null, 2));
  }
  if (report.status !== 'passed') process.exitCode = 1;
}

let data = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { data += chunk; });
process.stdin.on('end', () => {
  main(JSON.parse(data)).catch(() => {
    process.stderr.write('Notebook browser verification could not start.\n');
    process.exitCode = 1;
  });
});
