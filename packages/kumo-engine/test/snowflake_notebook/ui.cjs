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

const { VerificationError, requireThat, until } = require("./verification.cjs");

function captureRoute(value, account, notebook = null) {
  try {
    const url = new URL(value);
    const expected = new URL(account);
    const parts = url.hash.split("?");
    const route = decodeURIComponent(url.pathname + parts[0]);
    const parameters = new URLSearchParams(parts[1] || "");
    const cell = parameters.get("cell") || "";
    const preparedCell = Boolean(notebook &&
      /^[a-f0-9]{12}$/.test(notebook.attempt) &&
      typeof notebook.path === "string" &&
      path.basename(notebook.path) === "kumo-test-graph-" + notebook.attempt + ".ipynb" &&
      [notebook.attempt + "-0", notebook.attempt + "-1"].includes(cell));
    const cellSelection = parts.length === 2 &&
      /^#\/workspaces(?:\/|$)/.test(parts[0]) && parameters.size === 1 &&
      parameters.has("cell") &&
      (/^(?:[0-9]+|[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})$/i.test(cell) || preparedCell);
    return (
      url.protocol === "https:" &&
      url.origin === expected.origin &&
      /^\/[^/]+\/[^/]+\/$/.test(expected.pathname) &&
      (url.pathname.startsWith(expected.pathname) ||
        url.pathname === "/_deeplink/") &&
      url.search === "" &&
      !route.includes("?") && (parts.length === 1 || cellSelection) &&
      !/(?:^|[/#])(login|signin|auth|oauth|authorize)(?:[/#]|$)/i.test(route)
    );
  } catch {
    return false;
  }
}

async function captureAllowed(page, config, authenticated, requireContent = true) {
  try {
    if (
      !authenticated ||
      !page ||
      !captureRoute(page.url(), config.auth.account_url, config.notebook)
    )
      return false;
    if (
      !(await page
        .getByRole("button", {
          name: "Account Menu: " + config.auth.user,
          exact: true,
        })
        .isVisible())
    )
      return false;
    for (const frame of page.frames()) {
      if (
        await frame
          .locator(
            'input[type="password"]:visible, input[autocomplete="one-time-code"]:visible',
          )
          .count()
      )
        return false;
      if (
        await frame
          .getByRole("textbox", {
            name: "Enter the six-digit code from the authenticator",
            exact: true,
          })
          .isVisible()
      )
        return false;
    }
    if (!requireContent) return true;
    if (config.notebook) {
      return await page
        .frameLocator("iframe")
        .getByRole("list", { name: "Document breadcrumbs", exact: true })
        .getByRole("button", {
          name: path.basename(config.notebook.path),
          exact: true,
        })
        .isVisible();
    }
    const names = [
      config.expected.model,
      "Models / " + config.expected.model,
      ...config.expected.runs.map((run) => run.name),
    ];
    for (const name of names) {
      if (
        await page
          .getByRole("heading", { name, exact: true })
          .first()
          .isVisible()
      )
        return true;
    }
  } catch {}
  return false;
}

const DIAGNOSTIC_PHASES = new Set([
  "browser_start", "password_login", "totp_login", "notebook_visualization",
  "run_status", "run_details", "epoch_history", "model_versions",
  "notebook_cleanup", "browser_cleanup",
]);
const DIAGNOSTIC_STEPS = new Set([
  "authenticated", "workspace_opened", "dedicated_notebook_service_selected",
  "waiting_for_notebook_compute", "waiting_for_dedicated_pool_after_service_timeout",
  "notebook_connection_retried_once", "test_resources_verified",
  "notebook_install_run_requested", "notebook_install_started",
  "kernel_reset_completed", "notebook_graph_run_requested", "notebook_graph_started",
  "live_graph_verified", "test_notebook_kernel_stopped", "saved_output_verified",
  "saved_output_rechecked", "attempt_notebook_deleted", "checking_run",
  "run_verified", "ui_verified",
]);

function failureDiagnostics(error, phase, step) {
  const code = error instanceof VerificationError ? "assertion_failed" :
    error.name === "TimeoutError" ? "timeout" :
    error.name === "NotebookBlocked" ? "capacity_blocked" : "browser_error";
  const diagnostic = { failure_code: code };
  const selected = typeof phase === "string" ? phase.split(":")[0] : null;
  if (DIAGNOSTIC_PHASES.has(selected)) diagnostic.failure_phase = selected;
  if (DIAGNOSTIC_STEPS.has(step)) diagnostic.failure_step = step;
  return diagnostic;
}

function retainFailure(result, reason, error, phase, step) {
  if (!result.reason) result.reason = reason;
  if (!result.failure_code)
    Object.assign(result, failureDiagnostics(error, phase, step));
}

function safeError(error) {
  if (error instanceof VerificationError)
    return { type: "VerificationError", detail: error.message };
  const type = ["Error", "TimeoutError", "NotebookBlocked"].includes(error.name)
    ? error.name
    : "BrowserError";
  return {
    type,
    detail:
      "Browser verification did not complete; request and authentication details omitted.",
  };
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

function runMetrics(metrics) {
  requireThat(
    metrics && Object.keys(metrics).length > 0 &&
      Object.values(metrics).every(Number.isFinite),
    "Receipt lacks finite run metrics",
  );
  return Object.entries(metrics);
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
  let lastStep;
  const events = [];
  function write(name, value) {
    fs.writeFileSync(
      path.join(config.output, name),
      typeof value === "string" ? value : JSON.stringify(value, null, 2),
      { mode: 0o600 },
    );
  }
  function event(action, details = {}) {
    if (DIAGNOSTIC_STEPS.has(action)) lastStep = action;
    events.push({ at: new Date().toISOString(), action, ...details });
    write("ui-events.json", events);
  }
  async function evidence(name, frame = page) {
    const { captureElement } = require("./graph_ui.cjs");
    await captureElement(
      page, config.output, name,
      () => captureAllowed(page, config, authenticated),
      frame.locator("body"), { fullPage: true },
    );
    event("evidence", { name });
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
        url.hostname ===
        new URL(config.auth.account_url).pathname
          .split("/")
          .filter(Boolean)
          .join("-") +
          ".snowflakecomputing.com",
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
      if (config.notebook.action === "cleanup") {
        config.notebook_creation_started = true;
        result.notebook_cleanup = await notebook.cleanupCreatedNotebook(page, config, event);
        config.notebook_creation_started = false;
        result.status = result.notebook_cleanup.status;
        return result;
      }
      const checked = await notebook.verify(page, config, evidence, event, () =>
        captureAllowed(page, config, authenticated),
      );
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
      for (const [name, expected] of runMetrics(run.metrics)) {
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
    result.status =
      authenticated && error.name !== "NotebookBlocked" ? "failed" : "blocked";
    const failure = safeError(error);
    result.detail = failure.detail;
    retainFailure(result, `${phase}: ${failure.type}: ${failure.detail}`,
      error, phase, lastStep);
    event("error", { phase, type: failure.type });
    if (await captureAllowed(page, config, authenticated)) {
      try {
        await evidence("failure");
      } catch {
        /* Preserve the original failure. */
      }

    }
  } finally {
    result.verification_status = result.status;
    result.notebook_creation_started = Boolean(config.notebook_creation_started);
    if (config.notebook_creation_started) {
      try {
        if (!(await captureAllowed(page, config, authenticated, false)))
          throw new Error("Cleanup requires the authenticated test account");
        result.notebook_cleanup = await require("./notebook_ui.cjs")
          .cleanupCreatedNotebook(page, config, event);
      } catch (error) {
        result.notebook_cleanup = { status: "failed", type: safeError(error).type };
        result.status = "failed";
        retainFailure(result,
          "Attempt notebook cleanup failed; inspect private browser evidence",
          error, "notebook_cleanup", lastStep);
      }
    }
    result.elapsed_seconds = (Date.now() - started) / 1000;
    if (browser) {
      try {
        await browser.close();
        result.browser_cleanup = { status: "passed" };
      } catch {
        result.browser_cleanup = { status: "failed" };
        result.status = "failed";
        retainFailure(result, "Browser cleanup failed", new Error(),
          "browser_cleanup", lastStep);
      }
    }
    write("browser-result.json", result);
  }
  return result;
}

module.exports = {
  runMetrics,
  totp,
  metricValue,
  captureRoute,
  captureAllowed,
  safeError,
  failureDiagnostics,
  retainFailure,
  run,
};
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
