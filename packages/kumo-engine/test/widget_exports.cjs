// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Browser check of the explainability widget's PDF and HTML exports in
// Chromium, Firefox and WebKit. It renders the widget of each retained run with
// this checkout, then fails on a browser error, invalid plot geometry, a
// setting the HTML export or the print copy loses, or a missing printed view.
//
// Needs retained classification, forecasting and regression fits under
// KUMO_EXPLAINABILITY_RUNS/<task>, a Python environment that
// imports this checkout (PYTHON, default python3), and Playwright with its
// browsers, from NODE_PATH or the locked package next to this file:
//   npm ci --prefix packages/kumo-engine/test/snowflake_notebook/browser
//   node packages/kumo-engine/test/snowflake_notebook/browser/node_modules/playwright/cli.js install chromium firefox webkit
//   KUMO_EXPLAINABILITY_RUNS=<runs> node packages/kumo-engine/test/widget_exports.cjs
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs/promises'), os = require('node:os'), path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium, firefox, webkit} = (() => {
  try { return require('playwright'); } catch { return require('./snowflake_notebook/browser/node_modules/playwright'); }
})();
const runs = process.env.KUMO_EXPLAINABILITY_RUNS;
assert.ok(runs, 'Set KUMO_EXPLAINABILITY_RUNS to the retained runs (<task>/scratch/trainers/*/runs/*-fit-*).');
function widget(task, dir) {
  const file = path.join(dir, task + '.html');
  execFileSync(process.env.PYTHON || 'python3', ['-c', `import glob, sys
from dataclasses import replace
from kumo.engine import Trainer
from kumo.engine.explain import TrainingExplanations
run, = glob.glob(sys.argv[1] + '/scratch/trainers/*/runs/*-fit-*')
result = Trainer.load(run + '/published-model', accelerator='cpu').result
view = replace(result, explanations=TrainingExplanations.load(run)).visualize()
open(sys.argv[2], 'w', encoding='utf-8').write('<!doctype html><meta charset="utf-8">' + view.html())`,
  path.join(runs, task), file], {cwd: path.join(__dirname, '..'), stdio: 'inherit'});
  return pathToFileURL(file).href;
}
async function plots(page, selector) {
  const geometry = await page.locator(selector).evaluateAll(svgs => svgs.map(svg => {
    const shapes = [...svg.querySelectorAll('path[d], rect, circle')];
    return {count: shapes.length, valid: shapes.every(shape => {
      const box = shape.getBBox();
      return [box.x, box.y, box.width, box.height].every(Number.isFinite) &&
        !/NaN|Infinity/.test([...shape.attributes].map(a => a.value).join(' '));
    })};
  }));
  assert.ok(geometry.length, 'Expected plotted SVGs, not only headings.');
  assert.ok(geometry.every(g => g.count > 0 && g.valid), JSON.stringify(geometry));
  return geometry.length;
}
async function print(page) {
  const waiting = page.waitForEvent('popup');
  await page.getByRole('button', {name: 'Export PDF', exact: true}).evaluate(button => button.click());
  const popup = await waiting; await popup.waitForLoadState();
  assert.equal(await popup.locator('.kx-panel').count(), 1);
  return popup;
}
async function save(page, filename) {
  const waiting = page.waitForEvent('download');
  await page.getByRole('button', {name: 'Export HTML', exact: true}).click();
  await (await waiting).saveAs(filename);
}
async function alter(filename, change) {
  const html = await fs.readFile(filename, 'utf8');
  await fs.writeFile(filename, html.replace(/(<script[^>]+id="kumo-export-data"[^>]*>)([\s\S]*?)(<\/script>)/,
    (_, start, json, end) => { const payload = JSON.parse(json); change(payload);
      return start + JSON.stringify(payload).replaceAll('<', '\\u003c') + end; }));
}
(async () => {
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'kumo-exports-')), results = [];
  try {
    const [training, forecast, regression] = ['classification', 'forecasting', 'regression'].map(task => widget(task, tmp));
    for (const engine of [chromium, firefox, webkit]) {
      const browser = await engine.launch(), context = await browser.newContext({acceptDownloads: true});
      const errors = []; context.on('page', page => page.on('pageerror', error => errors.push(error.message)));
      try {
        const page = await context.newPage(); await page.goto(training);
        await page.getByRole('tab', {name: 'Explanation', exact: true}).click();
        const importance = page.locator('[data-card="importance"]');
        await importance.locator('[aria-label^="Show all"]').click();
        await importance.locator('g[role="button"]').nth(11).click();
        const expanded = await importance.locator('svg[role="img"]').textContent();
        const subgraph = page.locator('[data-card="subgraphs"]');
        await subgraph.locator('.kx-sg-entity select').selectOption({index: 1});
        await subgraph.locator('g.kx-sg-node[data-root="false"][aria-disabled="false"]').first().dispatchEvent('click');
        const pinned = await subgraph.locator('g.kx-sg-node[aria-pressed="true"]').getAttribute('data-id');
        const saved = path.join(tmp, engine.name() + '.html'); await save(page, saved);
        const reopened = await context.newPage(); await reopened.goto(pathToFileURL(saved).href);
        assert.equal(await reopened.locator('[data-card="importance"] svg[role="img"]').textContent(), expanded);
        assert.equal(await reopened.locator('[data-card="subgraphs"] g.kx-sg-node[aria-pressed="true"]').getAttribute('data-id'), pinned);
        await page.getByRole('tab', {name: 'Evaluation', exact: true}).click();
        const cutoff = page.locator('[data-card="cutoff"] input[type="range"]');
        await cutoff.evaluate(el => { el.value = '30'; el.dispatchEvent(new Event('input', {bubbles: true})); });
        let metric, points;
        for (const selected of ['Epoch history', 'All metrics']) {
          await page.getByRole('button', {name: selected, exact: true}).click();
          if (selected === 'Epoch history') {
            const picker = page.locator('.kx-history-controls select');
            await picker.selectOption({index: (await picker.locator('option').count()) - 1});
            metric = await picker.inputValue();
            await page.locator('.kx-history-legend input').nth(1).uncheck();
            assert.equal(await plots(page, '.kx-history-plots svg'), 2);
            points = await page.locator('.kx-history-plots circle title').allTextContents();
            assert.ok(points.length > 0);
          }
          await page.evaluate(() => scrollTo(0, 100));
          const position = await page.evaluate(() => [scrollX, scrollY]);
          const popup = await print(page);
          assert.equal(await plots(popup, '.kx-history-plots svg'), 2);
          await plots(popup, '.kx-chart svg');
          assert.equal(await popup.locator('[data-card="cutoff"] input[type="range"]').first().getAttribute('value'), '30');
          assert.equal(await popup.locator('.kx-history-legend input').nth(1).isChecked(), false);
          assert.equal(await popup.locator('.kx-history-controls select').inputValue(), metric);
          assert.deepEqual(await popup.locator('.kx-history-plots circle title').allTextContents(), points);
          for (const label of ['All metrics', 'Epoch history', 'Against baselines', 'Confusion matrix', 'ROC', 'Precision-recall', 'Cumulative gain'])
            assert.ok((await popup.locator('body').textContent()).includes(label), label);
          await page.waitForTimeout(200);
          assert.deepEqual(await page.evaluate(() => [scrollX, scrollY]), position);
          await popup.close();
        }
        await save(page, saved); await reopened.goto(pathToFileURL(saved).href);
        assert.equal(await reopened.locator('[data-card="cutoff"] input[type="range"]').inputValue(), '30');
        await reopened.getByRole('button', {name: 'Epoch history', exact: true}).click();
        assert.equal(await reopened.locator('.kx-history-controls select').inputValue(), metric);
        assert.equal(await reopened.locator('.kx-history-legend input').nth(1).isChecked(), false);
        assert.deepEqual(await reopened.locator('.kx-history-plots circle title').allTextContents(), points);
        await alter(saved, payload => { payload.data.tabs = payload.data.tabs.filter(tab => tab.id === 'evaluation'); payload.saved.tab = 0; });
        await reopened.goto(pathToFileURL(saved).href); const single = await print(reopened);
        assert.equal(await single.title(), 'Kumo · Evaluation'); await single.close();
        await page.goto(forecast); await save(page, saved);
        for (const values of [[-10, -5], [-10, 5]]) {
          await alter(saved, payload => { const totals = payload.data.tabs.flatMap(t => t.cards).find(c => c.id === 'forecast').views.find(v => v.id === 'totals');
            totals.steps.forEach(step => { step.values = values; }); payload.saved = null; });
          await reopened.goto(pathToFileURL(saved).href);
          await reopened.locator('[data-card="forecast"] button[data-view="totals"]').click();
          await plots(reopened, '[data-card="forecast"] .kx-chart svg');
          assert.ok(await reopened.locator('[data-card="forecast"] .kx-chart svg').evaluate(svg => {
            const height = svg.viewBox.baseVal.height, points = [...svg.querySelectorAll('circle')];
            return points.length > 0 && points.every(p => +p.getAttribute('cy') >= 0 && +p.getAttribute('cy') <= height);
          }), 'Signed forecast markers must remain inside the plotted SVG.');
        }
        await page.goto(regression);
        await page.locator('[data-card="predicted"] [data-view="heatmap"]').click();
        await page.getByLabel('Hide outliers', {exact: true}).check();
        await plots(page, '.kx-heatmap svg');
        const heatmap = await page.locator('.kx-heatmap svg').textContent();
        await page.locator('[data-card="predicted"] [data-view="tenths"]').click();
        await save(page, saved); await reopened.goto(pathToFileURL(saved).href);
        await reopened.locator('[data-card="predicted"] [data-view="heatmap"]').click();
        assert.equal(await reopened.getByLabel('Hide outliers', {exact: true}).isChecked(), true);
        assert.equal(await reopened.locator('.kx-heatmap svg').textContent(), heatmap);
        const popup = await print(page);
        assert.equal(await popup.getByLabel('Hide outliers', {exact: true}).isChecked(), true);
        assert.equal(await popup.locator('.kx-heatmap svg').textContent(), heatmap);
        await plots(popup, '.kx-heatmap svg'); await popup.close();
        await alter(saved, payload => {
          const view = payload.data.tabs.flatMap(t => t.cards).find(c => c.id === 'predicted').views.find(v => v.id === 'distributions');
          Object.assign(view, {start: 10000, width: 0.02, folded: false, indices: [0, 500050], bin_count: 500051});
          view.series.forEach(series => { series.counts = [12, 24]; series.stats.values.max = 20001; });
        });
        await reopened.goto(pathToFileURL(saved).href);
        await reopened.locator('[data-card="predicted"] [data-view="distributions"]').click();
        await plots(reopened, '[data-card="predicted"] .kx-chart svg');
        assert.ok(await reopened.locator('[data-card="predicted"] .kx-chart svg').evaluate(svg => {
          const bars = [...svg.querySelectorAll('rect')];
          return bars.length === 4 && bars.every(bar => +bar.getAttribute('width') >= 0) &&
            +bars[1].getAttribute('x') - +bars[0].getAttribute('x') > 100 && svg.querySelectorAll('path').length < 100;
        }), 'Sparse bins must keep their numeric positions without unbounded ticks.');
        const sparse = await print(reopened); await plots(sparse, '.kx-chart svg'); await sparse.close();
        assert.deepEqual(errors, []);
        results.push(engine.name());
      } finally { await browser.close(); }
    }
    console.log('Exports passed in ' + results.join(', '));
  } finally { await fs.rm(tmp, {recursive: true, force: true}); }
})().catch(error => { console.error(error); process.exitCode = 1; });
