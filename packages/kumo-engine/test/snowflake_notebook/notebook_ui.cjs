// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const fs = require("node:fs");
const path = require("node:path");
const { promisify } = require("node:util");
const execFile = promisify(require("node:child_process").execFile);
const { check } = require("./graph_ui.cjs");

function requireThat(value, message) {
  if (!value) throw new Error(message);
}

async function until(
  page,
  predicate,
  timeout,
  message = "Notebook did not reach the required state before its deadline",
) {
  const deadline = Date.now() + timeout;
  do {
    if (await predicate()) return;
    await page.waitForTimeout(250);
  } while (Date.now() < deadline);
  throw new Error(message);
}

async function workspace(page, event) {
  await page.getByRole("link", { name: "Projects", exact: true }).click();
  const frame = page.frameLocator("iframe");
  await frame
    .getByRole("tab", { name: "Workspaces", exact: true })
    .waitFor({ timeout: 90000 });
  await page.addLocatorHandler(
    frame
      .getByRole("region", {
        name: /^(Welcome to Workspaces|Connect to get started!|Turn off the clutter with focus mode|Introducing Python files|Semantic view is available in workspace)$/,
      })
      .first(),
    async (overlay) => {
      await overlay
        .getByRole("button", { name: "Close", exact: true })
        .first()
        .click();
    },
  );
  if (
    await page
      .getByRole("heading", { name: "CoCo is here", exact: true })
      .isVisible()
  ) {
    await page.getByRole("button", { name: "Ok", exact: true }).click();
  }
  event("workspace_opened");
  return frame;
}

async function readReceipt(frame, spec) {
  const output = frame.getByText(/KUMO_GRAPH_RESULT=\{/);
  await output.waitFor({ state: "visible", timeout: 180000 });
  const text = await output.textContent();
  const match = text.match(/KUMO_GRAPH_RESULT=(\{.*\})/s);
  requireThat(match, "No complete visible notebook result");
  const receipt = JSON.parse(match[1]);
  requireThat(receipt.status === "passed", "Notebook assertions did not pass");
  requireThat(
    receipt.identity.account === "SVB14499" &&
      receipt.identity.user === "KUMO_SNOWFLAKE_UI_TEST" &&
      receipt.identity.role === "KUMO_SNOWFLAKE_TEST_ROLE",
    "Notebook executed under another identity",
  );
  requireThat(
    receipt.identity.warehouse === "KUMO_TEST_WH" &&
      receipt.identity.database === "KUMO_TEST_DB" &&
      receipt.identity.schema === "KUMO_TEST",
    "Notebook executed outside dedicated test resources",
  );
  requireThat(
    receipt.fixture_sha256 === spec.fixture_sha256,
    "Notebook did not use the shared An fixture",
  );
  for (const [name, digest] of Object.entries(spec.test_helper_sha256))
    requireThat(
      receipt.test_helper_sha256[name] === digest,
      "Notebook test helper differs: " + name,
    );
  requireThat(
    receipt.runtime.source.commit === spec.source.commit,
    "Notebook source revision differs",
  );
  for (const [name, digest] of Object.entries(spec.source.modules)) {
    requireThat(
      receipt.runtime.source.modules[name] === digest,
      "Notebook module hash differs: " + name,
    );
  }
  for (const [name, value] of Object.entries(spec.candidate)) {
    requireThat(
      receipt.runtime.wheel[name] === value,
      "Notebook wheel identity differs",
    );
  }
  requireThat(
    /^[0-9a-f]{12}$/.test(receipt.execution_id) &&
      receipt.output_stage ===
        "@KUMO_TEST_DB.KUMO_TEST.TEST_STAGE/mlops-validation/graph-notebook/" +
          spec.attempt +
          "/results/" +
          receipt.execution_id,
    "Notebook evidence stage differs",
  );
  const files = [
    "graph.svg",
    "graph.pdf",
    "graph.md",
    "default.html",
    "sampled.html",
    "graph-result.json",
  ];
  requireThat(
    Object.keys(receipt.uploaded).length === files.length &&
      files.every((name) => /^[a-f0-9]{64}$/.test(receipt.uploaded[name])),
    "Incomplete retained evidence inventory",
  );
  for (const suffix of ["svg", "pdf", "md"])
    requireThat(
      receipt.uploaded["graph." + suffix] === receipt.exports[suffix].sha256,
      "Uploaded graph differs from verified export",
    );
  return receipt;
}

async function profile(frame, event) {
  const warehouse = frame.getByRole("button", { name: / warehouse size$/ });
  await warehouse.waitFor();
  requireThat(
    await frame
      .getByRole("button", {
        name: "KUMO_SNOWFLAKE_TEST_ROLE KUMO_TEST_WH warehouse size",
        exact: true,
      })
      .isVisible(),
    "Notebook query warehouse/role differs from dedicated test resources",
  );
  await frame
    .getByRole("button", { name: "Service List dropdown button", exact: true })
    .click();
  const dialog = frame.getByRole("dialog", {
    name: "Notebook service management",
    exact: true,
  });
  await dialog.getByText(/Compute pool:/).waitFor();
  const text = await dialog.innerText();
  requireThat(
    /Compute pool:\s*KUMO_TEST_POOL \(GEN_X64_G2_8\)(?:\s|$)/.test(text),
    "Notebook service uses a different compute pool",
  );
  const service = text.match(/Connected service\s+(EGU_GRAPH_UI_[A-Z0-9_]+)/);
  requireThat(
    service && service[1] === "EGU_GRAPH_UI_SERVICE",
    "Use the dedicated EGU_GRAPH_UI_SERVICE notebook service",
  );
  await dialog.press("Escape");
  const actual = {
    service: service[1],
    pool: "KUMO_TEST_POOL",
    warehouse: "KUMO_TEST_WH",
    role: "KUMO_SNOWFLAKE_TEST_ROLE",
  };
  event("test_resources_verified", actual);
  return actual;
}

async function showOutputs(frame) {
  const tools = frame.getByRole("button", { name: "Tools panel", exact: true });
  if ((await tools.getAttribute("aria-pressed")) === "true")
    await tools.click();
  for (const index of [1, 2]) {
    const cell = frame.getByRole("listitem").filter({
      has: frame.getByRole("checkbox", {
        name: "Select cell " + index,
        exact: true,
      }),
    });
    await cell
      .getByRole("button", { name: "More cell options", exact: true })
      .click();
    const option = frame.getByRole("menuitemcheckbox", {
      name: "Results only",
      exact: true,
    });
    if ((await option.getAttribute("aria-checked")) === "true")
      await option.press("Escape");
    else await option.click();
  }
}

async function runCell(frame, index) {
  const cell = frame.getByRole("listitem").filter({
    has: frame.getByRole("checkbox", {
      name: "Select cell " + index,
      exact: true,
    }),
  });
  const editor = cell.getByRole("textbox", {
    name: "Code Editor",
    exact: true,
  });
  const run = cell.getByRole("button", { name: "Run cell", exact: true });
  await run.waitFor({ state: "visible" });
  await until(
    editor.page(),
    () => run.isEnabled(),
    60000,
    "Notebook cell " + index + " is not ready for execution",
  );
  await editor.focus();
  requireThat(
    await editor.evaluate(
      (element) =>
        element === element.ownerDocument.activeElement ||
        element.contains(element.ownerDocument.activeElement),
    ),
    "Notebook cell " + index + " editor did not receive keyboard focus",
  );
  await editor.press("Control+Enter");
}

async function selectTestService(frame, event) {
  await frame
    .getByRole("button", { name: "Service List dropdown button", exact: true })
    .click();
  const management = frame.getByRole("dialog", {
    name: "Notebook service management",
    exact: true,
  });
  await management
    .getByRole("button", { name: "Select service", exact: true })
    .click();
  const selection = frame.getByRole("dialog", {
    name: "Select service",
    exact: true,
  });
  const service = selection.getByRole("listitem").filter({
    has: frame.getByRole("button", { name: /^EGU_GRAPH_UI_SERVICE - / }),
  });
  await service.waitFor({ timeout: 60000 });
  requireThat(
    /Compute pool:\s*KUMO_TEST_POOL \(GEN_X64_G2_8\)(?:\s|$)/.test(
      await service.innerText(),
    ),
    "Dedicated notebook service uses a different compute pool",
  );
  await service
    .getByRole("button", { name: /^EGU_GRAPH_UI_SERVICE - / })
    .click();
  event("dedicated_notebook_service_selected", {
    service: "EGU_GRAPH_UI_SERVICE",
    pool: "KUMO_TEST_POOL",
  });
}

async function execute(page, frame, spec, event, output, config) {
  const status = frame.getByRole("button", {
    name: "Connection status",
    exact: true,
  });
  await until(
    page,
    async () => (await status.innerText()).trim().length > 0,
    60000,
  );
  if ((await status.innerText()).trim() === "Connect")
    await selectTestService(frame, event);
  fs.writeFileSync(
    path.join(output, "notebook-connection.aria.txt"),
    await frame.locator("body").ariaSnapshot(),
    { mode: 0o600 },
  );
  event("waiting_for_notebook_compute");
  let connected = await Promise.race([
    status
      .filter({ hasText: "Connected" })
      .waitFor({ timeout: 1200000 })
      .then(() => true),
    frame
      .getByRole("alertdialog", { name: /Error on creating service:/ })
      .waitFor({ timeout: 1200000 })
      .then(() => false),
  ]);
  if (!connected) {
    const error = await frame
      .getByRole("alertdialog", { name: /Error on creating service:/ })
      .innerText();
    if (
      /Waiting for service to be RUNNING exceeded maximum wait time/i.test(
        error,
      ) &&
      /PENDING/.test(error)
    ) {
      event("waiting_for_dedicated_pool_after_service_timeout");
      const args = [
        path.join(__dirname, "launch.py"),
        "--wait-compute",
        "--timeout",
        "900",
      ];
      if (config.key_file) args.push("--key-file", config.key_file);
      await execFile(config.python, args, { timeout: 930000 });
      await page.reload({ waitUntil: "domcontentloaded" });
      await frame
        .getByRole("list", { name: "Document breadcrumbs", exact: true })
        .getByRole("button", { name: path.basename(spec.path), exact: true })
        .waitFor({ timeout: 90000 });
      await status.filter({ hasText: /^Connect$/ }).waitFor();
      await selectTestService(frame, event);
      await status
        .filter({ hasText: "Connected" })
        .waitFor({ timeout: 180000 });
      connected = true;
      event("notebook_connection_retried_once");
    }
  }
  requireThat(
    connected,
    "Snowflake could not start the dedicated notebook service; no cell was executed",
  );
  await profile(frame, event);
  const started = frame.getByText(/KUMO_GRAPH_INSTALL_STARTED=[0-9a-f]{12}/);
  const previous = (await started.count())
    ? (await started.innerText()).match(
        /KUMO_GRAPH_INSTALL_STARTED=([0-9a-f]{12})/,
      )[1]
    : null;
  await runCell(frame, 1);
  event("notebook_install_run_requested");
  let setup;
  await until(
    page,
    async () => {
      if (await started.count())
        setup = (await started.innerText()).match(
          /KUMO_GRAPH_INSTALL_STARTED=([0-9a-f]{12})/,
        )[1];
      return setup && setup !== previous;
    },
    120000,
    "Notebook install cell 1 did not acknowledge the run request; inspect its saved cell output",
  );
  event("notebook_install_started", { execution_id: setup });
  await Promise.race([
    frame
      .getByText(new RegExp("KUMO_GRAPH_INSTALL_COMPLETE=" + setup))
      .waitFor({ timeout: 1200000 }),
    frame
      .getByText(/Traceback.*Cell In/s)
      .waitFor({ timeout: 1200000 })
      .then(() => {
        throw new Error("Notebook setup failed; inspect its cell output");
      }),
  ]);
  await frame
    .getByRole("button", { name: "Service List dropdown button", exact: true })
    .click();
  await frame
    .getByRole("button", {
      name: "Restart kernel Clear variables but keep packages",
      exact: true,
    })
    .click();
  const dialog = frame.getByRole("dialog", {
    name: "Restart kernel",
    exact: true,
  });
  await dialog.waitFor();
  await dialog.getByRole("button", { name: "Restart", exact: true }).click();
  await dialog.waitFor({ state: "hidden" });
  event("kernel_restarted");
  await status.filter({ hasText: "Connected" }).waitFor({ timeout: 180000 });
  await profile(frame, event);
  const priorOutput = frame.getByText(/KUMO_GRAPH_RESULT=\{/);
  const graphStarted = frame.getByText(
    /KUMO_GRAPH_EXECUTION_STARTED=[0-9a-f]{12}/,
  );
  const prior = (await graphStarted.count())
    ? (await graphStarted.textContent()).match(
        /KUMO_GRAPH_EXECUTION_STARTED=([0-9a-f]{12})/,
      )[1]
    : null;
  await runCell(frame, 2);
  event("notebook_graph_run_requested");
  let execution;
  await until(
    page,
    async () => {
      const cell = frame
        .getByRole("listitem")
        .filter({
          has: frame.getByRole("checkbox", {
            name: "Select cell 2",
            exact: true,
          }),
        });
      if (await cell.getByText(/Traceback.*Cell In/s).isVisible())
        throw new Error(
          "Notebook graph cell failed before its execution acknowledgment; inspect the saved cell output",
        );
      if ((await status.innerText()).trim() !== "Connected")
        throw new Error(
          "Notebook service disconnected before graph execution acknowledgment",
        );
      if (await graphStarted.count())
        execution = (await graphStarted.textContent()).match(
          /KUMO_GRAPH_EXECUTION_STARTED=([0-9a-f]{12})/,
        )[1];
      return execution && execution !== prior;
    },
    120000,
    "Notebook graph cell 2 did not acknowledge the run request; inspect its saved execution state and output",
  );
  event("notebook_graph_started", { execution_id: execution });
  await until(
    page,
    async () => {
      if (await frame.getByText(/Traceback.*Cell In/s).isVisible())
        throw new Error(
          "Notebook graph checks failed; inspect its cell output",
        );
      if (!(await priorOutput.count())) return false;
      return (await readReceipt(frame, spec)).execution_id === execution;
    },
    180000,
  );
}

async function verify(page, config, evidence, event) {
  const spec = config.notebook;
  let frame = await workspace(page, event);
  const name = path.basename(spec.path);
  const file = frame.getByRole("treeitem", { name, exact: true });
  if (spec.action !== "create") {
    await file.dblclick();
    if (spec.action === "run") {
      const cells = JSON.parse(fs.readFileSync(spec.path, "utf8")).cells;
      requireThat(
        cells.length === 2 && cells.every((cell) => cell.cell_type === "code"),
        "Expected the prepared two-cell graph notebook",
      );
      for (const [index, source] of cells.entries()) {
        const cell = frame.getByRole("listitem").filter({
          has: frame.getByRole("checkbox", {
            name: "Select cell " + (index + 1),
            exact: true,
          }),
        });
        await cell
          .getByRole("button", { name: "More cell options", exact: true })
          .click();
        await frame
          .getByRole("menuitemcheckbox", {
            name: "Code and results",
            exact: true,
          })
          .click();
        await cell
          .getByRole("textbox", { name: "Code Editor", exact: true })
          .fill(source.source.join(""));
      }
    }
  } else {
    requireThat(
      (await file.count()) === 0,
      "Test notebook already exists; choose reuse or a new attempt",
    );
    await frame
      .getByRole("button", { name: "Add New Menu", exact: true })
      .click();
    const chooser = page.waitForEvent("filechooser");
    await frame
      .getByRole("menuitem", { name: "Upload files", exact: true })
      .click();
    await (await chooser).setFiles(spec.path);
    await file.waitFor();
  }
  await frame
    .getByRole("list", { name: "Document breadcrumbs", exact: true })
    .getByRole("button", { name, exact: true })
    .waitFor();
  await frame
    .getByRole("checkbox", { name: "Select cell 2", exact: true })
    .waitFor();
  if (spec.action === "verify") {
    const receipt = await readReceipt(frame, spec);
    fs.writeFileSync(
      path.join(config.output, "notebook-receipt.json"),
      JSON.stringify(receipt, null, 2),
      { mode: 0o600 },
    );
    const saved = await require("./graph_ui.cjs").checkSaved(
      page,
      receipt,
      config.output,
    );
    const connected =
      (
        await frame
          .getByRole("button", { name: "Connection status", exact: true })
          .innerText()
      ).trim() === "Connected";
    fs.writeFileSync(
      path.join(config.output, "notebook-checks.json"),
      JSON.stringify(
        { live: { status: "not_run" }, saved, kernel_connected: connected },
        null,
        2,
      ),
      { mode: 0o600 },
    );
    await evidence("notebook-saved");
    event("saved_output_rechecked", {
      execution_submitted: false,
      kernel_connected: connected,
    });
    return {
      status: "passed",
      scope:
        "Read-only existing notebook graph outputs and retained diagram exports",
      notebook: { name, receipt, reopened: saved, kernel_connected: connected },
      training_submitted: false,
    };
  }
  if (spec.action !== "verify")
    await execute(page, frame, spec, event, config.output, config);
  const actualProfile = await profile(frame, event);
  await showOutputs(frame);
  const receipt = await readReceipt(frame, spec);
  fs.writeFileSync(
    path.join(config.output, "notebook-receipt.json"),
    JSON.stringify(receipt, null, 2),
    { mode: 0o600 },
  );
  const initial = await check(page, receipt, config.output, "live");
  fs.writeFileSync(
    path.join(config.output, "notebook-checks.json"),
    JSON.stringify({ live: initial, saved: { status: "pending" } }, null, 2),
    { mode: 0o600 },
  );
  event("live_graph_verified");
  await evidence("notebook-live");
  fs.writeFileSync(
    path.join(config.output, "notebook-live.aria.txt"),
    await frame.locator("body").ariaSnapshot(),
    { mode: 0o600 },
  );

  await frame
    .getByRole("button", { name: "Service List dropdown button", exact: true })
    .click();
  await frame
    .getByRole("dialog", { name: "Notebook service management", exact: true })
    .getByRole("button", { name: "Shut down kernel", exact: true })
    .click();
  const shutdown = frame.getByRole("dialog", {
    name: /^Shut down(?: kernel)?$/,
  });
  try {
    await shutdown.waitFor({ timeout: 3000 });
  } catch {}
  if (await shutdown.isVisible()) {
    await shutdown
      .getByRole("button", { name: /^Shut down(?: kernel)?$/ })
      .click();
    await shutdown.waitFor({ state: "hidden" });
  }
  await frame
    .getByRole("button", { name: "Connection status", exact: true })
    .filter({ hasText: /^(Connect|Disconnected)$/ })
    .waitFor();
  event("test_notebook_kernel_stopped");

  const url = page.url();
  await page.goto("about:blank");
  await page.goto(url, { waitUntil: "domcontentloaded" });
  frame = page.frameLocator("iframe");
  await frame
    .getByRole("tab", { name: "Workspaces", exact: true })
    .waitFor({ timeout: 90000 });
  if (
    !(await frame
      .getByRole("button", { name: "Run all cells", exact: true })
      .isVisible())
  ) {
    await frame.getByRole("treeitem", { name, exact: true }).dblclick();
  }
  const reopened = await readReceipt(frame, spec);
  requireThat(
    JSON.stringify(reopened) === JSON.stringify(receipt),
    "Reopened notebook lost or replaced its saved output",
  );
  const saved = await require("./graph_ui.cjs").checkSaved(
    page,
    receipt,
    config.output,
  );
  fs.writeFileSync(
    path.join(config.output, "notebook-checks.json"),
    JSON.stringify({ live: initial, saved }, null, 2),
    { mode: 0o600 },
  );
  await evidence("notebook-reopened");
  event("saved_output_verified", { execution_submitted: false });
  return {
    status: "passed",
    scope:
      "Live Workspace graph rendering, tooltips, controls, saved outputs after kernel shutdown, and retained graph exports",
    notebook: {
      name,
      profile: actualProfile,
      receipt,
      initial,
      reopened: saved,
    },
    training_submitted: false,
  };
}

module.exports = { verify, readReceipt };
