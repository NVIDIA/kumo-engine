/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// NODE_PATH must provide Playwright and its browsers. Inputs are retained widget HTML.
// Usage: node widget_exports.cjs training.html [forecasting.html] [regression.html]
const {chromium, firefox, webkit} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises'), os = require('node:os'), path = require('node:path');
const {pathToFileURL} = require('node:url');
const [training, forecast, regression] = process.argv.slice(2);
assert.ok(training, 'Provide a binary training widget with epoch history and importance.');
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
  try { for (const engine of [chromium, firefox, webkit]) {
    const browser = await engine.launch(), context = await browser.newContext({acceptDownloads: true});
    const errors = []; context.on('page', page => page.on('pageerror', error => errors.push(error.message)));
    try {
      const page = await context.newPage(); await page.goto(pathToFileURL(path.resolve(training)).href);
      await page.getByRole('tab', {name: 'Explanation', exact: true}).click();
      const importance = page.locator('[data-card="importance"]');
      await importance.locator('[aria-label^="Show all"]').click();
      await importance.locator('g[role="button"]').nth(11).click();
      const expanded = await importance.locator('svg[role="img"]').textContent();
      const saved = path.join(tmp, engine.name() + '.html'); await save(page, saved);
      const reopened = await context.newPage(); await reopened.goto(pathToFileURL(saved).href);
      assert.equal(await reopened.locator('[data-card="importance"] svg[role="img"]').textContent(), expanded);
      await page.getByRole('tab', {name: 'Evaluation', exact: true}).click();
      let metric, points;
      for (const selected of ['Epoch history', 'All metrics']) {
        await page.getByRole('button', {name: selected, exact: true}).click();
        if (selected === 'Epoch history') {
          const picker = page.locator('.kx-history-controls select');
          await picker.selectOption({index: (await picker.locator('option').count()) - 1}, {force: true});
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
        if (metric) {
          assert.equal(await popup.locator('.kx-history-legend input').nth(1).isChecked(), false);
          assert.equal(await popup.locator('.kx-history-controls select').inputValue(), metric);
          assert.deepEqual(await popup.locator('.kx-history-plots circle title').allTextContents(), points);
        }
        for (const label of ['All metrics', 'Epoch history', 'Against baselines', 'Confusion matrix', 'ROC', 'Precision-recall', 'Cumulative gain'])
          assert.ok((await popup.locator('body').textContent()).includes(label), label);
        await page.waitForTimeout(200);
        assert.deepEqual(await page.evaluate(() => [scrollX, scrollY]), position);
        await popup.close();
      }
      await save(page, saved); await reopened.goto(pathToFileURL(saved).href);
      await reopened.getByRole('button', {name: 'Epoch history', exact: true}).click();
      assert.equal(await reopened.locator('.kx-history-controls select').inputValue(), metric);
      assert.equal(await reopened.locator('.kx-history-legend input').nth(1).isChecked(), false);
      assert.deepEqual(await reopened.locator('.kx-history-plots circle title').allTextContents(), points);
      await alter(saved, payload => { payload.data.tabs = payload.data.tabs.filter(tab => tab.id === 'evaluation'); payload.saved.tab = 0; });
      await reopened.goto(pathToFileURL(saved).href); const single = await print(reopened);
      assert.equal(await single.title(), 'Kumo · Evaluation'); await single.close();
      if (forecast) {
        await page.goto(pathToFileURL(path.resolve(forecast)).href); await save(page, saved);
        for (const values of [[-10, -5], [-10, 5]]) {
          await alter(saved, payload => { const totals = payload.data.tabs.flatMap(t => t.cards).find(c => c.id === 'forecast').views.find(v => v.id === 'totals');
            totals.steps.forEach(step => { step.actual = values[0]; step.forecast = values[1]; }); payload.saved = null; });
          await reopened.goto(pathToFileURL(saved).href);
          await reopened.locator('[data-card="forecast"] button[data-view="totals"]').click();
          await plots(reopened, '[data-card="forecast"] .kx-chart svg');
          assert.ok(await reopened.locator('[data-card="forecast"] .kx-chart svg').evaluate(svg => {
            const height = svg.viewBox.baseVal.height, points = [...svg.querySelectorAll('circle')];
            return points.length > 0 && points.every(p => +p.getAttribute('cy') >= 0 && +p.getAttribute('cy') <= height);
          }), 'Signed forecast markers must remain inside the plotted SVG.');
        }
      }
      if (regression) {
        await page.goto(pathToFileURL(path.resolve(regression)).href);
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
          Object.assign(view, {start: 10000, width: 0.02, bins: ['10000', '20001'], indices: [0, 500050], bin_count: 500051});
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
      }
      assert.deepEqual(errors, []); results.push({browser: engine.name(), expandedImportance: true, selectedAndUnselectedHistoryPlots: true, singleTabTitle: true, signedForecast: !!forecast, hiddenHeatmapSetting: !!regression, sparseDistributions: !!regression});
    } finally { await browser.close(); }
  } console.log(JSON.stringify(results, null, 2)); } finally { await fs.rm(tmp, {recursive: true, force: true}); }
})().catch(error => { console.error(error); process.exitCode = 1; });
