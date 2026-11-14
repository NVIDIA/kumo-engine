// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const fs = require("node:fs");
const path = require("node:path");

const { VerificationError, requireThat } = require("./verification.cjs");

async function captureElement(
  target,
  output,
  name,
  allowed = async () => true,
  aria = target,
  screenshotOptions = {},
) {
  requireThat(
    await allowed(),
    "Evidence capture requires the current authenticated test page",
  );
  let complete = false;
  try {
    await target.screenshot({
      ...screenshotOptions,
      path: path.join(output, name + ".png"),
    });
    const snapshot = aria ? await aria.ariaSnapshot() : null;
    requireThat(
      await allowed(),
      "Authentication changed during evidence capture",
    );
    if (snapshot !== null)
      fs.writeFileSync(path.join(output, name + ".aria.txt"), snapshot, {
        mode: 0o600,
      });
    complete = true;
  } finally {
    if (!complete) {
      for (const suffix of [".png", ".aria.txt"])
        fs.rmSync(path.join(output, name + suffix), { force: true });
    }
  }
}

async function hoverTooltip(page, graph, svg, table, column) {
  const row = svg.locator(
    `[data-kumo-table="${table}"][data-kumo-column="${column}"]`,
  );
  await row.hover({ timeout: 10000 });
  const tooltip = graph.getByRole("tooltip");
  const deadline = Date.now() + 5000;
  do {
    if (
      (await tooltip.isVisible()) &&
      (await tooltip.locator(".kumo-tip-context").innerText()) === table &&
      (await tooltip.locator(".kumo-tip-name").innerText()) === column
    )
      return tooltip.innerText();
    await page.waitForTimeout(50);
  } while (Date.now() < deadline);
  throw new VerificationError("Visible tooltip is not bound to the expected schema row");
}

function displayedSample(value) {
  return value.startsWith("'") && value.endsWith("'")
    ? value.slice(1, -1)
    : value;
}

async function widgets(page) {
  const found = [];
  for (const frame of page.frames()) {
    const views = frame.locator(".kumo-widget.kumo-graph");
    for (let index = 0; index < (await views.count()); index++)
      found.push({ graph: views.nth(index), frame });
  }
  return found;
}

async function check(
  page,
  receipt,
  output,
  label,
  controls = true,
  interactive = true,
  allowed = async () => true,
) {
  requireThat(
    receipt.status === "passed" && receipt.runtime.workspace === true,
    "Kernel graph checks did not pass in a Workspace",
  );
  requireThat(
    receipt.default_reads.backend_accesses === 0 &&
      receipt.default_reads.sample_calls === 0,
    "Default graph read source data",
  );
  requireThat(
    receipt.opt_in_reads.sample_calls ===
      Object.keys(receipt.schema.tables).length && receipt.canaries.length >= 6,
    "Missing explicit sample coverage",
  );
  let views;
  const deadline = Date.now() + 60000;
  do {
    views = await widgets(page);
    if (views.length === 2) break;
    await page.waitForTimeout(500);
  } while (Date.now() < deadline);
  requireThat(
    views.length === 2,
    "Exactly two native graph widgets must render",
  );
  const results = [];
  for (const [index, { graph, frame }] of views.entries()) {
    const owner = frame.parentFrame() ? await frame.frameElement() : graph;
    await owner.evaluate((element) =>
      element.scrollIntoView({ block: "center", behavior: "instant" }),
    );
    const region = graph.getByRole("region", {
      name: "Pannable graph visualization",
      exact: true,
    });
    await region.waitFor({ state: "visible" });
    const box = await region.boundingBox();
    requireThat(
      box && box.width > 100 && box.height > 100,
      "Graph has no visible rendering area",
    );
    const svg = region.locator("svg");
    for (const name of Object.keys(receipt.schema.tables)) {
      await svg
        .locator(".kumo-card-name")
        .filter({ hasText: new RegExp("^" + name + "$") })
        .waitFor({ state: "visible" });
    }
    let schemaRows = 0;
    for (const [table, definition] of Object.entries(receipt.schema.tables)) {
      for (const column of definition.cols.map((value) => value.name)) {
        const row = svg.locator(
          `[data-kumo-table="${table}"][data-kumo-column="${column}"]`,
        );
        requireThat(
          (await row.count()) === 1 && (await row.isVisible()),
          "Missing visible schema row",
        );
        schemaRows++;
      }
    }
    requireThat(
      (await svg.locator("path.kumo-edge").count()) ===
        receipt.schema.edges.length,
      "Unexpected relationship count",
    );
    for (const edge of receipt.schema.edges) {
      const primary = receipt.schema.tables[edge.dst_table].pkey;
      const line = svg.locator(
        `path.kumo-edge[data-src-table="${edge.src_table}"][data-src-column="${edge.fkey}"][data-dst-table="${edge.dst_table}"][data-dst-column="${primary}"]`,
      );
      requireThat(
        (await line.count()) === 1 && (await line.isVisible()),
        "Relationship is missing from rendered graph",
      );
    }
    if (!interactive) {
      for (const sample of receipt.samples) {
        const tip = await svg
          .locator(
            `[data-kumo-table="${sample.table}"][data-kumo-column="${sample.column}"]`,
          )
          .getAttribute("data-kumo-tip");
        if (index === 0)
          requireThat(
            !tip.includes("Sample values:"),
            "Saved default graph contains sample data",
          );
        else
          requireThat(
            sample.values.every((value) => tip.includes(value)),
            "Saved opted-in preview data was lost",
          );
      }
      await captureElement(
        graph,
        output,
        `${label}-graph-${index + 1}`,
        allowed,
      );
      results.push({
        kind: index === 0 ? "default" : "sample_opt_in",
        schema_rows: schemaRows,
        relationships: receipt.schema.edges.length,
        interactive: false,
        status: "passed",
      });
      continue;
    }
    const tips = [];
    const primary = Object.entries(receipt.schema.tables).find(
      ([, table]) => table.pkey,
    );
    const foreign = receipt.schema.edges[0];
    const keys = [
      [primary[0], primary[1].pkey, [primary[0], primary[1].pkey, "PK"]],
      [
        foreign.src_table,
        foreign.fkey,
        [
          foreign.src_table,
          foreign.fkey,
          "FK",
          foreign.dst_table +
            "." +
            receipt.schema.tables[foreign.dst_table].pkey,
        ],
      ],
    ];
    for (const [table, column, required] of keys) {
      const text = await hoverTooltip(page, graph, svg, table, column);
      requireThat(
        required.every((value) => text.includes(value)),
        "Rendered tooltip lost schema/key identity",
      );
      if (index === 0)
        requireThat(
          !text.includes("Sample values"),
          "Default tooltip contains samples",
        );
      tips.push(text);
    }
    for (const sample of receipt.samples) {
      const text = await hoverTooltip(
        page,
        graph,
        svg,
        sample.table,
        sample.column,
      );
      if (index === 0)
        requireThat(
          !text.includes("Sample values"),
          "Default label tooltip contains samples",
        );
      else
        requireThat(
          sample.values.every((value) => text.includes(displayedSample(value))),
          "Explicit sample tooltip lost real fixture values",
        );
      tips.push(text);
    }
    if (controls) {
      const scale = graph.locator(".kumo-graph-scale");
      const before = await scale.innerText();
      await graph.getByRole("button", { name: "Zoom in", exact: true }).click();
      requireThat(
        (await scale.innerText()) !== before,
        "Zoom did not change the displayed scale",
      );
      await graph
        .getByRole("button", { name: "Fit graph", exact: true })
        .click();
      for (const layout of ["Compact", "Standard"]) {
        const radio = graph.getByRole("radio", { name: layout, exact: true });
        await radio.check();
        requireThat(
          (await radio.isChecked()) &&
            (await graph.getAttribute("data-layout")) === layout.toLowerCase(),
          "Layout control did not update the rendered graph",
        );
      }
      await graph
        .getByRole("button", { name: "Export commands", exact: true })
        .click();
      const commands = graph.locator(".kumo-graph-export-options");
      await commands.waitFor({ state: "visible" });
      const text = await commands.innerText();
      requireThat(
        text.includes("graph.visualize(") &&
          text.includes("graph.pdf") &&
          text.includes("graph.md"),
        "Export commands are incomplete",
      );
      await graph
        .getByRole("button", { name: "Export commands", exact: true })
        .click();
    }
    await captureElement(graph, output, `${label}-graph-${index + 1}`, allowed);
    results.push({
      kind: index === 0 ? "default" : "sample_opt_in",
      schema_rows: schemaRows,
      relationships: receipt.schema.edges.length,
      tooltips: tips,
      controls,
      status: "passed",
    });
  }
  return results;
}

async function checkSaved(page, receipt, output, allowed = async () => true) {
  const deadline = Date.now() + 60000;
  do {
    const views = await widgets(page);
    if (
      views.length === 2 &&
      (
        await Promise.all(
          views.map(({ graph }) =>
            graph
              .getByRole("region", {
                name: "Pannable graph visualization",
                exact: true,
              })
              .isVisible(),
          ),
        )
      ).every(Boolean)
    ) {
      return {
        mode: "cached_readonly_widgets",
        checks: await check(
          page,
          receipt,
          output,
          "reopened",
          false,
          false,
          allowed,
        ),
      };
    }
    const images = [];
    for (const frame of page.frames()) {
      const candidates = frame.getByRole("img", {
        name: "Table relationships",
        exact: true,
      });
      for (let index = 0; index < (await candidates.count()); index++)
        if (await candidates.nth(index).isVisible())
          images.push(candidates.nth(index));
    }
    if (images.length === 2) {
      const checks = [];
      let retry = false;
      for (const [index, image] of images.entries()) {
        const details = await image.evaluate((element) => {
          if (
            !element.complete ||
            element.naturalWidth < 100 ||
            element.naturalHeight < 100 ||
            !element.src.startsWith("data:image/svg+xml;base64,")
          )
            return null;
          const xml = new DOMParser().parseFromString(
            atob(element.src.split(",")[1]),
            "image/svg+xml",
          );
          return {
            tables: [
              ...new Set(
                [...xml.querySelectorAll("[data-kumo-table]")].map((node) =>
                  node.getAttribute("data-kumo-table"),
                ),
              ),
            ].sort(),
            rows: [
              ...xml.querySelectorAll(
                '[data-kumo-column]:not([data-kumo-column=""])',
              ),
            ].map((node) => ({
              table: node.getAttribute("data-kumo-table"),
              column: node.getAttribute("data-kumo-column"),
              tip: node.getAttribute("data-kumo-tip"),
            })),
            edges: [...xml.querySelectorAll("path.kumo-edge")]
              .map((node) =>
                [
                  "data-src-table",
                  "data-src-column",
                  "data-dst-table",
                  "data-dst-column",
                ].map((name) => node.getAttribute(name)),
              )
              .sort(),
          };
        });
        if (!details) {
          retry = true;
          break;
        }
        requireThat(
          JSON.stringify(details.tables) ===
            JSON.stringify(Object.keys(receipt.schema.tables).sort()),
          "Saved static graph lost tables",
        );
        const edges = receipt.schema.edges
          .map((edge) => [
            edge.src_table,
            edge.fkey,
            edge.dst_table,
            receipt.schema.tables[edge.dst_table].pkey,
          ])
          .sort();
        requireThat(
          JSON.stringify(details.edges) === JSON.stringify(edges),
          "Saved static graph lost relationships",
        );
        const rows = Object.entries(receipt.schema.tables)
          .flatMap(([table, definition]) =>
            definition.cols.map((column) => [table, column.name]),
          )
          .sort();
        requireThat(
          JSON.stringify(
            details.rows.map((row) => [row.table, row.column]).sort(),
          ) === JSON.stringify(rows),
          "Saved static graph lost schema columns",
        );
        for (const [table, definition] of Object.entries(
          receipt.schema.tables,
        )) {
          if (definition.pkey)
            requireThat(
              details.rows
                .find(
                  (row) =>
                    row.table === table && row.column === definition.pkey,
                )
                .tip.includes("PK"),
              "Saved static graph lost primary key identity",
            );
        }
        for (const edge of receipt.schema.edges)
          requireThat(
            details.rows
              .find(
                (row) =>
                  row.table === edge.src_table && row.column === edge.fkey,
              )
              .tip.includes("FK"),
            "Saved static graph lost foreign key identity",
          );
        requireThat(
          details.rows.every((row) => !row.tip.includes("Sample values:")),
          "Static fallback diagram contains sample values",
        );
        await captureElement(
          image,
          output,
          `reopened-static-${index + 1}`,
          allowed,
          null,
        );
        checks.push({ status: "passed", ...details });
      }
      if (!retry) return {
        mode: "static_fallback",
        scope:
          "Saved schema diagrams; static exports omit sample previews and live controls",
        sample_previews: "not_available_in_static_fallback",
        checks,
      };
    }
    await page.waitForTimeout(500);
  } while (Date.now() < deadline);
  throw new VerificationError(
    "Saved graph outputs are missing after notebook kernel shutdown",
  );
}

module.exports = {
  check,
  checkSaved,
  widgets,
  displayedSample,
  captureElement,
};
