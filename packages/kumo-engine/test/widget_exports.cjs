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
// KUMO_EXPLAINABILITY_RUNS/<task>, optionally a retained batch prediction run
// with cohorts and subgraphs at KUMO_BATCH_EXPLANATIONS, a Python environment that
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
function widget(task, dir, source = path.join(runs, task)) {
  const file = path.join(dir, task + '.html');
  execFileSync(process.env.PYTHON || 'python3', ['-c', `import glob, sys
from dataclasses import replace
from kumo.engine import PredictionExplanations, Trainer
from kumo.engine.explain import TrainingExplanations
if '${task}' == 'batch':
    view = PredictionExplanations.load(sys.argv[1]).visualize()
else:
    run, = glob.glob(sys.argv[1] + '/scratch/trainers/*/runs/*-fit-*')
    result = Trainer.load(run + '/published-model', accelerator='cpu').result
    view = replace(result, explanations=TrainingExplanations.load(run)).visualize()
open(sys.argv[2], 'w', encoding='utf-8').write('<!doctype html><meta charset="utf-8">' + view.html())`,
  source, file], {cwd: path.join(__dirname, '..'), stdio: 'inherit'});
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
  const download = await waiting; await download.saveAs(filename);
  return download.suggestedFilename();
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
    const batch = process.env.KUMO_BATCH_EXPLANATIONS && widget('batch', tmp, process.env.KUMO_BATCH_EXPLANATIONS);
    for (const engine of [chromium, firefox, webkit]) {
      const browser = await engine.launch(), context = await browser.newContext({acceptDownloads: true});
      const errors = []; context.on('page', page => page.on('pageerror', error => errors.push(error.message)));
      await context.addInitScript(() => {
        const Original = ResizeObserver;
        window.observed = new Set();
        window.ResizeObserver = class extends Original {
          constructor(callback) { super(callback); this.targets = new Set(); }
          observe(target, ...args) { this.targets.add(target); window.observed.add(target); super.observe(target, ...args); }
          unobserve(target) { this.targets.delete(target); window.observed.delete(target); super.unobserve(target); }
          disconnect() { this.targets.forEach(t => window.observed.delete(t)); this.targets.clear(); super.disconnect(); }
        };
      });
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
        // Class outcomes keep independent pages through HTML and print exports.
        await alter(saved, payload => {
          const graph = payload.data.tabs.flatMap(t => t.cards).find(c => c.id === 'subgraphs');
          graph.task_type = 'multilabel_classification';
          for (const sample of graph.samples) {
            sample.prediction = Object.fromEntries(Array.from({length: 23}, (_, i) => ['class-' + i, i / 100]));
            sample.target = Object.fromEntries(Array.from({length: 23}, (_, i) => ['class-' + i, Number(i === 22)]));
          }
        });
        await reopened.goto(pathToFileURL(saved).href);
        await reopened.getByRole('button', {name: 'Next page of predictions', exact: true}).click();
        await reopened.getByRole('button', {name: 'Next page of ground truth', exact: true}).click();
        await reopened.getByRole('button', {name: 'Next page of ground truth', exact: true}).click();
        const outcomes = await reopened.locator('.kx-sg-class-outcome tbody').allTextContents();
        const outcomePrint = await print(reopened);
        assert.deepEqual(await outcomePrint.locator('.kx-sg-class-outcome tbody').allTextContents(), outcomes);
        await outcomePrint.close();
        await save(reopened, saved); await reopened.goto(pathToFileURL(saved).href);
        assert.deepEqual(await reopened.locator('.kx-sg-class-outcome tbody').allTextContents(), outcomes);
        await reopened.locator('.kx-sg-entity select').selectOption({index: 0});
        assert.deepEqual(await reopened.locator('.kx-sg-class-outcome .kx-sg-pager span').allTextContents(), ['1 of 3', '1 of 3']);
        await page.getByRole('tab', {name: 'Evaluation', exact: true}).click();
        assert.equal(await page.locator('[data-card="cutoff"] .kx-cutoff-curves > *').count(), 3);
        const cutoff = page.locator('[data-card="cutoff"] input[type="range"]');
        await cutoff.evaluate(el => { el.value = '30'; el.dispatchEvent(new Event('input', {bubbles: true})); });
        let metric, points, flipped;
        for (const tab of ['Training', 'Evaluation']) {
          await page.getByRole('tab', {name: tab, exact: true}).click();
          if (tab === 'Training') {
            const picker = page.locator('.kx-history-controls select');
            await picker.selectOption({index: (await picker.locator('option').count()) - 1});
            metric = await picker.inputValue();
            // Keep another experiment visible when toggling the selected best.
            const box = page.locator('.kx-history-legend input').nth(1);
            if (await box.count()) {
              if (await box.isChecked()) await page.locator('.kx-history-legend input').nth(2).check();
              flipped = !(await box.isChecked()); await box.setChecked(flipped);
            }
            /* Every split row of the Training Table is complete. */
            assert.ok(await page.locator('[data-card="training_table"] table').evaluate(table =>
              table.tBodies[0].rows.length > 0 && [...table.tBodies[0].rows].every(
                row => row.cells.length === table.tHead.rows[0].cells.length)));
            assert.equal(await plots(page, '.kx-history-plots svg'), 2);
            points = await page.locator('.kx-history-plots circle title').allTextContents();
            assert.ok(points.length > 0 || flipped === false);
          }
          await page.evaluate(() => scrollTo(0, 100));
          const position = await page.evaluate(() => [scrollX, scrollY]);
          const popup = await print(page);
          if (tab === 'Training') {
            assert.equal(await plots(popup, '.kx-history-plots svg'), 2);
            if (flipped !== undefined) assert.equal(await popup.locator('.kx-history-legend input').nth(1).isChecked(), flipped);
            assert.equal(await popup.locator('.kx-history-controls select').inputValue(), metric);
            assert.deepEqual(await popup.locator('.kx-history-plots circle title').allTextContents(), points);
            assert.ok((await popup.locator('body').textContent()).includes('Training Table'));
          } else {
            await plots(popup, '.kx-chart svg');
            assert.equal(await popup.locator('[data-card="cutoff"] input[type="range"]').first().getAttribute('value'), '30');
            assert.equal(await popup.locator('[data-card="metrics"] .kx-switch').count(), 0);
            for (const label of ['Holdout', 'Validation', 'Training', 'Confusion matrix', 'ROC', 'Precision-recall', 'Cumulative gain'])
              assert.ok((await popup.locator('body').textContent()).includes(label), label);
            for (const label of ['Against baselines', 'Published model', 'Training run'])
              assert.ok(!(await popup.locator('[data-card="metrics"]').textContent()).includes(label), label);
          }
          await page.waitForTimeout(200);
          assert.deepEqual(await page.evaluate(() => [scrollX, scrollY]), position);
          await popup.close();
        }
        await save(page, saved); await reopened.goto(pathToFileURL(saved).href);
        assert.equal(await reopened.locator('[data-card="cutoff"] input[type="range"]').inputValue(), '30');
        await reopened.getByRole('tab', {name: 'Training', exact: true}).click();
        assert.equal(await reopened.locator('.kx-history-controls select').inputValue(), metric);
        if (flipped !== undefined) assert.equal(await reopened.locator('.kx-history-legend input').nth(1).isChecked(), flipped);
        assert.deepEqual(await reopened.locator('.kx-history-plots circle title').allTextContents(), points);
        await alter(saved, payload => { payload.data.tabs = payload.data.tabs.filter(tab => tab.id === 'evaluation'); payload.saved.tab = 0; });
        await reopened.goto(pathToFileURL(saved).href); const single = await print(reopened);
        assert.equal(await single.title(), 'Kumo · Evaluation'); await single.close();
        if (batch) {
          // One panel without tabs; the selected column and pinned record survive both exports.
          await page.goto(batch);
          assert.equal(await page.getByRole('tab').count(), 0);
          await page.locator('[data-card="groups"] select').first().selectOption({index: 1});
          await page.locator('g.kx-sg-node[data-root="false"]').first().dispatchEvent('click');
          const column = await page.locator('[data-card="groups"] select').first().inputValue();
          // Axis ticks follow the page width, which the saved page changes.
          const bars = page => page.locator('[data-card="groups"] .kx-chart svg text:not(.kx-t-axis)').allTextContents();
          const cohort = await bars(page);
          const details = await page.locator('.kx-sg-detail').textContent();
          assert.equal(await save(page, saved), 'kumo-prediction-explanations.html');
          await reopened.goto(pathToFileURL(saved).href);
          assert.equal(await reopened.locator('[data-card="groups"] select').first().inputValue(), column);
          assert.deepEqual(await bars(reopened), cohort);
          assert.equal(await reopened.locator('.kx-sg-detail').textContent(), details);
          const popup = await print(page);
          assert.equal(await popup.title(), 'Kumo · Explanation');
          await plots(popup, '[data-card="groups"] .kx-chart svg'); await plots(popup, '.kx-sg-canvas > svg');
          assert.equal(await popup.locator('.kx-sg-detail').textContent(), details); await popup.close();
        }
        await page.goto(forecast); await save(page, saved);
        await alter(saved, payload => {
          const views = payload.data.tabs.flatMap(t => t.cards).find(c => c.id === 'forecast').views;
          const horizon = views.find(v => v.id === 'horizon');
          horizon.steps = Array.from({length: 90}, (_, i) => ({...horizon.steps[0], label: String(i + 1), values: [1, 2]}));
          views.find(v => v.id === 'totals').total_callout = null;
        });
        await reopened.setViewportSize({width: 360, height: 900});
        await reopened.goto(pathToFileURL(saved).href);
        assert.ok(await reopened.locator('.kx-bar').evaluateAll(bars => bars.length === 180 && bars.every(b => +b.getAttribute('width') > 0)));
        for (const values of [[-10, -5], [-10, 5]]) {
          await alter(saved, payload => { const totals = payload.data.tabs.flatMap(t => t.cards).find(c => c.id === 'forecast').views.find(v => v.id === 'totals');
            totals.steps.forEach(step => { step.values = values; }); payload.saved = null; });
          await reopened.goto(pathToFileURL(saved).href);
          await reopened.locator('[data-card="forecast"] button[data-view="totals"]').click();
          await plots(reopened, '[data-card="forecast"] .kx-chart svg');
          assert.ok(!(await reopened.locator('svg text').allTextContents()).includes('null'));
          assert.ok(await reopened.locator('[data-card="forecast"] .kx-chart svg').evaluate(svg => {
            const height = svg.viewBox.baseVal.height, points = [...svg.querySelectorAll('circle')];
            return points.length > 0 && points.every(p => +p.getAttribute('cy') >= 0 && +p.getAttribute('cy') <= height);
          }), 'Signed forecast markers must remain inside the plotted SVG.');
        }
        await reopened.setViewportSize({width: 1280, height: 900});
        await page.goto(regression);
        for (let i = 0; i < 5; i++) for (const view of ['heatmap', 'tenths'])
          await page.locator('[data-card="predicted"] [data-view="' + view + '"]').click();
        assert.equal(await page.evaluate(() => [...window.observed].filter(n => n.matches('.kx-chart') && !n.isConnected).length), 0);
        await page.locator('[data-card="predicted"] [data-view="heatmap"]').click();
        await page.getByLabel('Hide outliers', {exact: true}).check();
        await plots(page, '.kx-heatmap svg');
        /* Tick density and visible count labels follow the chart width, so
           copies are compared by their cells' bins and counts. */
        const cells = target => target.locator('.kx-heatmap svg rect > title').allTextContents();
        const heatmap = await cells(page);
        assert.ok(heatmap.length > 0);
        await page.locator('[data-card="predicted"] [data-view="tenths"]').click();
        await save(page, saved); await reopened.goto(pathToFileURL(saved).href);
        await reopened.locator('[data-card="predicted"] [data-view="heatmap"]').click();
        assert.equal(await reopened.getByLabel('Hide outliers', {exact: true}).isChecked(), true);
        assert.deepEqual(await cells(reopened), heatmap);
        await plots(reopened, '.kx-heatmap svg');
        const popup = await print(page);
        assert.equal(await popup.getByLabel('Hide outliers', {exact: true}).isChecked(), true);
        assert.deepEqual(await cells(popup), heatmap);
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
