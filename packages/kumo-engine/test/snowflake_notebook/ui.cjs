// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

function totp(seed, timestamp = Date.now(), digits = 6) {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
  let bits = "";
  for (const character of seed.toUpperCase().replace(/=+$/, "")) {
    const value = alphabet.indexOf(character);
    if (value < 0) throw new Error("Invalid TOTP seed");
    bits += value.toString(2).padStart(5, "0");
  }
  const bytes = [];
  for (let index = 0; index + 8 <= bits.length; index += 8)
    bytes.push(parseInt(bits.slice(index, index + 8), 2));
  const counter = Buffer.alloc(8);
  counter.writeBigUInt64BE(BigInt(Math.floor(timestamp / 30000)));
  const digest = crypto
    .createHmac("sha1", Buffer.from(bytes))
    .update(counter)
    .digest();
  const offset = digest[digest.length - 1] & 15;
  return ((digest.readUInt32BE(offset) & 0x7fffffff) % 10 ** digits)
    .toString()
    .padStart(digits, "0");
}

function requireThat(condition, message) {
  if (!condition) throw new Error(message);
}
function metricValue(displayed, expected) {
  const text = displayed.trim().replace(/,/g, "");
  requireThat(
    /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(text),
    "Metric is not numeric",
  );
  const value = Number(text);
  requireThat(
    Number.isFinite(value) &&
      Number.isFinite(expected) &&
      Math.abs(value - expected) <= 0.000005000001,
    "Metric differs at the displayed five-decimal precision",
  );
  return value;
}

async function gridValue(page, group, name) {
  const cell = group.getByRole("gridcell", { name, exact: true });
  await cell.waitFor({ state: "visible" });
  const row = group
    .getByRole("row")
    .filter({ has: page.getByRole("gridcell", { name, exact: true }) });
  requireThat(
    (await row.count()) === 1,
    "Metric/parameter name must identify one row",
  );
  const cells = row.getByRole("gridcell");
  requireThat((await cells.count()) === 2, "Unexpected metric/parameter row");
  return (await cells.nth(1).innerText()).trim();
}

async function until(page, predicate, timeout) {
  const deadline = Date.now() + timeout;
  do {
    if (await predicate()) return;
    await page.waitForTimeout(200);
  } while (Date.now() < deadline);
  throw new Error(
    "Rendered UI did not reach the expected state before its deadline",
  );
}

async function history(page, run, metric, timeout) {
  const search = page.getByRole("combobox", {
    name: "Search charts",
    exact: true,
  });
  await search.fill(metric);
  const suggestion = page.getByRole("option", { name: metric, exact: true });
  try {
    await suggestion.waitFor({ state: "visible", timeout: 1500 });
  } catch {}
  if (await suggestion.isVisible()) await suggestion.click();
  else await search.press("Enter");
  await search.press("Escape");
  await page
    .getByRole("heading", { name: metric, exact: true, level: 2 })
    .waitFor();
  const points = [];
  for (const [step, expected] of run.history[metric]) {
    const prefix = `Step: ${step}; ${metric}:`;
    const selector = `svg.marks:visible [aria-roledescription="point"][aria-label^=${JSON.stringify(prefix)}]`;
    const point = page.locator(selector).first();
    await point.waitFor({ state: "attached" });
    await point.scrollIntoViewIfNeeded();
    const box = await point.boundingBox();
    requireThat(
      box && box.width > 0 && box.height > 0,
      "History point is not rendered",
    );
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    const heading = page.getByText(`Step ${step}`, { exact: true });
    await heading.waitFor({ state: "visible", timeout });
    const text = await heading.evaluate((element, name) => {
      for (
        let node = element.parentElement, depth = 0;
        node && depth < 6;
        node = node.parentElement, depth++
      ) {
        const content = node.innerText;
        if (
          content &&
          content.trim().startsWith(element.innerText) &&
          content.includes(name)
        )
          return content;
      }
      return null;
    }, run.name);
    requireThat(text, "Tooltip is not bound to the selected run and step");
    const values = text
      .split("\n")
      .map((value) => value.trim())
      .filter((value) =>
        /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value),
      );
    requireThat(
      values.length === 1,
      "History tooltip must contain one metric value",
    );
    metricValue(values[0], expected);
    points.push({ step, expected, displayed: values[0] });
  }
  requireThat(
    new Set(points.map((point) => point.step)).size >= 2,
    "Multiple rendered epochs are required",
  );
  return points;
}

async function run(config) {
  const started = Date.now();
  const result = {
    status: "blocked",
    assertions: [],
    runs: [],
    versions: [],
    training_submitted: false,
  };
  let browser;
  let page;
  let authenticated = false;
  let phase = "browser_start";
  const events = [];
  function write(name, value) {
    fs.writeFileSync(
      path.join(config.output, name),
      typeof value === "string" ? value : JSON.stringify(value, null, 2),
      { mode: 0o600 },
    );
  }
  function event(action, details = {}) {
    events.push({ at: new Date().toISOString(), action, ...details });
    write("ui-events.json", events);
  }
  async function evidence(name) {
    await page.screenshot({
      path: path.join(config.output, name + ".png"),
      fullPage: true,
    });
    write(name + ".aria.txt", await page.locator("body").ariaSnapshot());
    event("evidence", { name, url: page.url().split("?")[0] });
  }
  try {
    const { chromium } = require(config.playwright_module);
    browser = await chromium.launch({ headless: true });
    result.browser = {
      chromium: browser.version(),
      playwright: require(config.playwright_module + "/package.json").version,
      node: process.version,
    };
    const context = await browser.newContext({
      viewport: { width: 1600, height: 1200 },
      reducedMotion: "reduce",
    });
    page = await context.newPage();
    page.setDefaultTimeout(config.timeout);
    page.setDefaultNavigationTimeout(config.timeout);
    phase = "password_login";
    await page.goto(config.auth.account_url, { waitUntil: "domcontentloaded" });
    await page.waitForURL(
      (url) =>
        url.hostname === "lfwgwbp-kumo_us_west_test_5.snowflakecomputing.com",
      { waitUntil: "domcontentloaded" },
    );
    await page.locator('input[type="text"]').fill(config.auth.user);
    await page.locator('input[type="password"]').fill(config.auth.password);
    await page.getByRole("button", { name: "Sign in", exact: true }).click();
    phase = "totp_login";
    const code = page.getByRole("textbox", {
      name: "Enter the six-digit code from the authenticator",
      exact: true,
    });
    await code.waitFor();
    await page.waitForTimeout(30000 - (Date.now() % 30000) + 500);
    await code.fill(totp(config.auth.totp_secret));
    await page.getByRole("button", { name: "Continue", exact: true }).click();
    const outcome = await Promise.race([
      page
        .waitForURL((url) => url.hostname === "app.snowflake.com", {
          waitUntil: "domcontentloaded",
        })
        .then(() => "redirect")
        .catch(() => "timeout"),
      page
        .getByText(
          /TOTP Invalid|invalid (?:verification )?code|attempts remaining/i,
        )
        .first()
        .waitFor({ state: "visible" })
        .then(() => "rejected")
        .catch(() => "timeout"),
    ]);
    if (outcome !== "redirect") {
      result.reason =
        outcome === "rejected"
          ? "Snowflake rejected the TOTP code. No automatic retries were made."
          : "Snowsight login did not complete.";
      return result;
    }
    await page
      .getByRole("button", {
        name: "Account Menu: " + config.auth.user,
        exact: true,
      })
      .waitFor();
    authenticated = true;
    result.authentication = "passed";
    event("authenticated");
    if (config.notebook) {
      phase = "notebook_visualization";
      const notebook = require("./notebook_ui.cjs");
      const checked = await notebook.verify(page, config, evidence, event);
      Object.assign(result, checked);
      return result;
    }
    const experimentURL = config.expected.runs[0].url.split("/runs/")[0];
    phase = "run_status";
    await page.goto(experimentURL, { waitUntil: "domcontentloaded" });
    await page
      .getByRole("heading", { name: config.expected.model, exact: true })
      .waitFor();
    const statusGrid = page.getByRole("grid");
    for (const run of config.expected.runs) {
      await page
        .getByRole("searchbox", { name: "Search", exact: true })
        .fill(run.name);
      await statusGrid
        .getByRole("gridcell", { name: run.name, exact: true })
        .waitFor();
      await until(
        page,
        async () => {
          const names = await statusGrid
            .getByRole("checkbox", { name: /^Select run / })
            .count();
          return (
            names > 0 &&
            (await statusGrid
              .getByRole("gridcell", { name: "Finished", exact: true })
              .count()) === names
          );
        },
        config.timeout,
      );
      result.assertions.push({
        kind: "run_status",
        run: run.name,
        displayed: "Finished",
        status: "passed",
      });
    }
    await evidence("experiment-status");
    for (const [index, run] of config.expected.runs.entries()) {
      phase = "run_details:" + run.name;
      event("checking_run", { run: run.name });
      await page.goto(run.url, { waitUntil: "domcontentloaded" });
      await page
        .getByRole("heading", { name: run.name, exact: true })
        .waitFor();
      await page.getByRole("tab", { name: "Details", exact: true }).click();
      const metrics = page.getByRole("group", { name: "Metrics", exact: true });
      const observed = {};
      for (const [name, expected] of Object.entries(run.metrics)) {
        const displayed = await gridValue(page, metrics, name);
        metricValue(displayed, expected);
        observed[name] = { expected, displayed };
      }
      const parameters = page.getByRole("group", {
        name: "Parameters",
        exact: true,
      });
      const identities = {};
      for (const name of [
        "kumo.bundle_id",
        "kumo.model_version",
        "kumo.kind",
      ]) {
        requireThat(
          typeof run.params[name] === "string",
          "Receipt lacks required tracking identity",
        );
        await parameters
          .getByRole("searchbox", { name: "Search parameters", exact: true })
          .fill(name);
        const displayed = await gridValue(page, parameters, name);
        requireThat(
          displayed === run.params[name],
          "Rendered tracking identity differs from receipt",
        );
        identities[name] = displayed;
      }
      await parameters
        .getByRole("searchbox", { name: "Search parameters", exact: true })
        .clear();
      await evidence(`run-${index + 1}-details`);
      const epochs = {};
      if (run.kind === "fit") {
        await page.getByRole("tab", { name: "Charts", exact: true }).click();
        for (const metric of ["train/loss", "val/auroc"]) {
          phase = "epoch_history:" + run.name + "/" + metric;
          epochs[metric] = await history(page, run, metric, config.timeout);
          await evidence(`run-${index + 1}-${metric.replace("/", "-")}`);
        }
      }
      result.runs.push({
        name: run.name,
        status: "passed",
        metrics: observed,
        params: identities,
        epoch_history: epochs,
      });
      event("run_verified", {
        run: run.name,
        metrics: Object.keys(observed).length,
      });
    }
    phase = "model_versions";
    await page.getByRole("link", { name: "AI & ML", exact: true }).hover();
    await page.getByRole("option", { name: /^Models\b/ }).click();
    await page.mouse.move(1100, 80);
    await page.keyboard.press("Escape");
    await page.getByRole("heading", { name: "Models", exact: true }).waitFor();
    await page
      .getByRole("searchbox", { name: "Search", exact: true })
      .fill(config.expected.model);
    const modelRow = page.getByRole("row", {
      name: "User model " + config.expected.model,
      exact: true,
    });
    await modelRow
      .getByRole("gridcell", { name: config.expected.database, exact: true })
      .waitFor();
    await modelRow
      .getByRole("gridcell", { name: config.expected.schema, exact: true })
      .waitFor();
    await modelRow.dblclick();
    await page
      .getByRole("heading", {
        name: "Models / " + config.expected.model,
        exact: true,
      })
      .waitFor();
    for (const version of config.expected.versions) {
      const row = page
        .getByRole("row")
        .filter({ has: page.getByText(version.name, { exact: true }) });
      await row.waitFor({ state: "visible" });
      requireThat(
        (await row.count()) === 1,
        "Model version is missing or duplicated",
      );
      result.versions.push({
        name: version.name,
        model: config.expected.model,
        status: "passed",
      });
    }
    await evidence("model-versions");
    requireThat(
      result.runs.length === 10 && result.versions.length === 2,
      "Incomplete UI coverage",
    );
    result.status = "passed";
    result.scope =
      "Exact run statuses, metrics and tracking identities, fit epoch tooltips, and registered model versions";
    event("ui_verified");
  } catch (error) {
    result.status = authenticated ? "failed" : "blocked";
    result.reason = `${phase}: ${error.name}. No credentials or authentication-page artifacts were retained.`;
    if (authenticated) {
      result.detail = String(error.message);
      for (const value of Object.values(config.auth))
        result.detail = result.detail.split(value).join("[redacted]");
      result.reason = `${phase}: ${error.name}: ${result.detail.split("\n")[0]}`;
    }
    event("error", { phase, type: error.name });
    if (
      authenticated &&
      page &&
      new URL(page.url()).hostname === "app.snowflake.com"
    ) {
      try {
        await evidence("failure");
      } catch {
        /* Preserve the original failure. */
      }
      for (const [index, frame] of page.frames().entries()) {
        try {
          write(
            "failure-frame-" + index + ".aria.txt",
            await frame.locator("body").ariaSnapshot(),
          );
        } catch {}
      }
    }
  } finally {
    result.elapsed_seconds = (Date.now() - started) / 1000;
    if (browser) {
      try {
        await browser.close();
      } catch {
        result.status = "failed";
        result.reason = "Browser cleanup failed";
      }
    }
    write("browser-result.json", result);
  }
  return result;
}

module.exports = { totp, metricValue, run };
if (require.main === module) {
  let input = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (data) => {
    input += data;
  });
  process.stdin.on("end", async () => {
    try {
      const result = await run(JSON.parse(input));
      process.exitCode = result.status === "passed" ? 0 : 1;
    } catch {
      console.error("Browser worker failed before a result could be saved");
      process.exitCode = 1;
    }
  });
}
