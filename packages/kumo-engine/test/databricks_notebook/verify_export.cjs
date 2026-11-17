// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('../snowflake_notebook/browser/node_modules/playwright');
const shared = require('../snowflake_notebook/graph_ui.cjs');

async function verify(root) {
  const receipt = JSON.parse(fs.readFileSync(path.join(root, 'outputs/graph-result.json')));
  const report = {
    scope: 'Actual Databricks Jobs HTML export; reopened in a fresh browser context',
    interactive: { status: 'pending' },
    saved_export: { status: 'pending' },
    browser_errors: [],
  };
  console.log('Opening retained Databricks notebook export');
  const browser = await chromium.launch({ headless: true });
  try {
    for (const mode of ['interactive', 'saved_export']) {
      console.log('Checking ' + mode);
      const context = await browser.newContext({ viewport: { width: 1800, height: 1400 } });
      let page;
      try {
        page = await context.newPage();
        page.on('pageerror', error => report.browser_errors.push(error.message));
        await page.goto(require('node:url').pathToFileURL(path.join(root, 'notebook.html')).href,
          { waitUntil: 'load', timeout: 120000 });
        const checks = mode === 'interactive'
          ? await shared.check(page, receipt, root, 'export', true, true)
          : await shared.checkSaved(page, receipt, root);
        report[mode] = { status: 'passed', checks };
      } catch (error) {
        report[mode] = { status: 'failed', error: error.message };
        if (page) await page.screenshot({ path: path.join(root, mode + '-failure.png'), fullPage: true, timeout: 5000 }).catch(() => {});
      } finally {
        await context.close();
      }
    }
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(root, 'ui-result.json'), JSON.stringify(report, null, 2));
  }
  if (report.interactive.status !== 'passed' || report.saved_export.status !== 'passed' || report.browser_errors.length)
    throw new Error('Databricks notebook export UI verification failed');
}
verify(path.resolve(process.argv[2])).catch(error => { console.error(error.message); process.exitCode = 1; });
