// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const { createHash } = require("node:crypto");
const { requireThat, until } = require("./verification.cjs");
const { captureElement } = require("./graph_ui.cjs");

const PLOTS = ["gradient", "cohort", "cohort-class-0", "cohort-class-1",
  "cohort-multilabel-0", "cohort-multilabel-1", "cohort-regression",
  "gradient-layout", "gradient-zero", "cohort-layout"];

async function checkPlots(page, receipt, output, label, allowed, scope) {
  const report = receipt.explanations;
  requireThat(
    report?.status === "passed" && report.fresh_process === "passed" &&
      report.refit.samples > 0 && report.bundle && report.bundle === report.refit.bundle,
    "Missing actual refit explanations and fresh-process model identity",
  );
  for (const [task, prefix] of [["multiclass", "class"], ["multilabel", "multilabel"]]) {
    const classification = report[task];
    requireThat(
      classification?.status === "passed" && classification.fresh_process === "passed" &&
        classification.bundle && classification.classes?.length === 3 &&
        classification.invalid_selection === "passed" &&
        classification.selected_classes?.length === 2 &&
        new Set(classification.selected_classes).size === 2 &&
        classification.selected_classes.every((name, index) =>
          classification.classes.includes(name) &&
          report.plots[`cohort-${prefix}-${index}`]?.class_name === name &&
          report.plots[`cohort-${prefix}-${index}`]?.title.endsWith(` — ${name}`)),
      `Missing distinct explicit-class ${task} cohort verification`,
    );
  }
  requireThat(report.regression?.status === "passed" &&
    report.regression.fresh_process === "passed" && report.regression.bundle &&
    report.layout_examples?.status === "passed" &&
    report.layout_examples.columns === 24 && report.layout_examples.cohorts === 12,
  "Missing regression and layout boundary verification");
  const checks = {};
  for (const kind of PLOTS) {
    const expected = report.plots[kind];
    requireThat(expected?.sha256?.png && expected.layout === "passed" &&
      expected.source === (["gradient-layout", "gradient-zero", "cohort-layout"].includes(kind)
        ? "deterministic-plot-input" : "retained-training-run"),
    "Missing explanation plot identity, layout or source");
    let images;
    await until(page, async () => {
      images = [];
      for (const root of (scope.root ? [scope.root] : page.frames())) {
        for (const image of await root.locator("img").all()) {
          const source = await image.getAttribute("src");
          if (!source?.startsWith("data:image/png;base64,")) continue;
          const sha = createHash("sha256")
            .update(Buffer.from(source.split(",")[1], "base64")).digest("hex");
          if (sha === expected.sha256.png && await image.isVisible() &&
            await image.evaluate(node => node.complete &&
              node.naturalWidth > 100 && node.naturalHeight > 100)) images.push(image);
        }
      }
      return images.length === 1;
    }, 60000, `Expected exactly one visible native ${kind} explanation matching the generated plot`);
    const image = images[0];
    await image.scrollIntoViewIfNeeded();
    requireThat(await image.evaluate(node =>
      node.complete && node.naturalWidth > 100 && node.naturalHeight > 100),
    "Explanation image did not decode");
    const geometry = await image.evaluate(node => {
      const box = node.getBoundingClientRect();
      return { width: box.width, height: box.height,
        naturalWidth: node.naturalWidth, naturalHeight: node.naturalHeight,
        aspect: node.naturalWidth / node.naturalHeight };
    });
    requireThat(geometry.width > 100 && geometry.height > 100 &&
      Math.abs(geometry.width / geometry.height / geometry.aspect - 1) < .02,
    `Explanation image is distorted or too small: ${kind} ${JSON.stringify(geometry)}`);
    await captureElement(image, output, `${label}-explanation-${kind}`, allowed, null);
    checks[kind] = { status: "passed", sha256: expected.sha256.png };
  }
  return { status: "passed", ...checks };
}

async function check(page, receipt, output, label, allowed = async () => true, scope = {}) {
  const originalViewport = page.viewportSize();
  requireThat(originalViewport, "Explanation matrix requires an explicit browser viewport");
  const originalScheme = await page.evaluate(() =>
    matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const variants = [];
  try {
    for (const width of [1600, 800]) {
      await page.setViewportSize({ width, height: 1000 });
      for (const colorScheme of ["light", "dark"]) {
        await page.emulateMedia({ colorScheme });
        requireThat(await page.evaluate(scheme =>
          matchMedia(`(prefers-color-scheme: ${scheme})`).matches, colorScheme),
        "Browser did not apply the requested color preference");
        const proof = await checkPlots(page, receipt, output,
          `${label}-${width}-${colorScheme}`, allowed, scope);
        variants.push({ width, preferred_color_scheme: colorScheme, ...proof });
      }
    }
  } finally {
    await page.setViewportSize(originalViewport);
    await page.emulateMedia({ colorScheme: originalScheme });
  }
  return { status: "passed", variants };
}

module.exports = { check, PLOTS };
