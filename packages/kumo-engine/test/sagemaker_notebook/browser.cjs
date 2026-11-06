// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require('../snowflake_notebook/browser/node_modules/playwright');
const graph = require('../snowflake_notebook/graph_ui.cjs');
const { historyPoint, renderedCoordinates } = require('./assertions.cjs');

function requireThat(value, message) {
  if (!value) throw new Error(message);
}

async function metricValues(page, reference) {
  const observed = {};
  for (const [name, expected] of Object.entries(reference.metrics)) {
    const link = page.getByRole('link', { name, exact: true });
    const row = page.getByRole('row').filter({ has: link });
    await row.waitFor({ state: 'visible' });
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

async function epochHistory(page, metric, expected) {
  const link = page.getByRole('link', { name: metric, exact: true });
  await link.click();
  await page.getByRole('heading', { name: metric, exact: true }).waitFor({ state: 'visible' });
  await page.getByRole('radio', { name: 'Step', exact: true }).check();
  const points = page.locator('.js-plotly-plot .scatterlayer .points .point');
  await points.first().waitFor({ state: 'visible' });
  requireThat((await points.count()) === expected.length, 'Rendered history point count differs');
  const values = [];
  for (const [index, [step, expectedValue]] of expected.entries()) {
    const rendered = await points.nth(index).evaluate(renderedCoordinates);
    await points.nth(index).hover();
    const tooltip = page.locator('.js-plotly-plot .hoverlayer .hovertext');
    await tooltip.first().waitFor({ state: 'visible' });
    const labels = await tooltip.locator('text, tspan').allTextContents();
    values.push(historyPoint(rendered, step, expectedValue, labels));
  }
  return values;
}

async function main(input) {
  const root = path.resolve(input.directory);
  const output = path.join(root, 'browser');
  fs.mkdirSync(output, { recursive: false });
  const report = { status: 'failed', notebook_export: 'not selected', native_mlflow: 'not selected' };
  const browser = await chromium.launch({ headless: true });
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
      const context = await browser.newContext({ viewport: { width: 1600, height: 1100 } });
      const page = await pageIn(context);
      page.setDefaultTimeout(60000);
      await page.goto(input.mlflow_url, { waitUntil: 'domcontentloaded' });
      await page.getByRole('link', { name: /experiments/i }).first().waitFor();
      const base = new URL(page.url());
      base.search = '';
      base.hash = '';
      const observations = [];
      for (const [index, run] of expected.runs.entries()) {
        const reference = run.reference;
        report.stage = `mlflow-${run.roles.join('-')}-${index}`;
        const url = new URL(base);
        url.hash = `/experiments/${encodeURIComponent(reference.experiment_id)}/runs/${encodeURIComponent(reference.run)}`;
        await page.goto(url.href, { waitUntil: 'domcontentloaded' });
        await page.getByText(reference.run_name, { exact: true }).first().waitFor({ state: 'visible' });
        await page.getByText('Finished', { exact: true }).first().waitFor({ state: 'visible' });
        const metrics = await metricValues(page, reference);
        await page.screenshot({ path: path.join(output, `mlflow-run-${index}.png`), fullPage: true });
        const history = {};
        if (run.roles.includes('fit')) {
          for (const metric of ['train/loss', 'val/auroc']) {
            await page.goto(url.href, { waitUntil: 'domcontentloaded' });
            history[metric] = await epochHistory(page, metric, reference.history[metric]);
            await page.screenshot({ path: path.join(output, `mlflow-history-${index}-${Object.keys(history).length}.png`), fullPage: true });
          }
        }
        noErrors();
        observations.push({ run: reference.run, roles: run.roles, status: 'FINISHED', metrics, history });
      }
      const versions = [];
      for (const version of expected.versions) {
        report.stage = `mlflow-version-${version.version}`;
        const url = new URL(base);
        url.hash = `/models/${encodeURIComponent(expected.model)}/versions/${encodeURIComponent(version.version)}`;
        await page.goto(url.href, { waitUntil: 'domcontentloaded' });
        await page.getByText(expected.model, { exact: true }).first().waitFor({ state: 'visible' });
        await page.getByText(new RegExp(`^Version[ ]*${version.version}$`)).first().waitFor({ state: 'visible' });
        await page.getByText(version.bundle, { exact: true }).first().waitFor({ state: 'visible' });
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
    report.error = 'Browser verification failed; inspect the completed check records.';
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
