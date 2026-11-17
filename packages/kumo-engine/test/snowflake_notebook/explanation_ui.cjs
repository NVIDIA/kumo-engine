// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const { createHash } = require("node:crypto");
const { requireThat, until } = require("./verification.cjs");
const { captureElement } = require("./graph_ui.cjs");

async function check(page, receipt, output, label, allowed) {
  const report = receipt.explanations;
  requireThat(
    report?.status === "passed" && report.fresh_process === "passed" &&
      report.refit.samples > 0 && report.bundle && report.bundle === report.refit.bundle,
    "Missing actual refit explanations and fresh-process model identity",
  );
  const multiclass = report.multiclass;
  requireThat(
    multiclass?.status === "passed" && multiclass.fresh_process === "passed" &&
      multiclass.bundle && multiclass.classes?.length === 3 &&
      multiclass.invalid_selection === "passed" &&
      multiclass.selected_classes?.length === 2 &&
      new Set(multiclass.selected_classes).size === 2 &&
      multiclass.selected_classes.every((name, index) =>
        multiclass.classes.includes(name) &&
        report.plots[`cohort-class-${index}`]?.class_name === name &&
        report.plots[`cohort-class-${index}`]?.title.endsWith(` — ${name}`)),
    "Missing distinct explicit-class cohort verification",
  );
  const checks = {};
  for (const kind of ["gradient", "cohort", "cohort-class-0", "cohort-class-1"]) {
    const expected = report.plots[kind];
    requireThat(expected?.sha256?.png, "Missing explanation plot identity");
    let images;
    await until(page, async () => {
      images = [];
      for (const frame of page.frames()) {
        for (const image of await frame.locator("img").all()) {
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
    await captureElement(image, output, `${label}-explanation-${kind}`, allowed, null);
    checks[kind] = { status: "passed", sha256: expected.sha256.png };
  }
  return { status: "passed", ...checks };
}

module.exports = { check };
