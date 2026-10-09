// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SEEDED_PROJECT_BASE } from "./constants";
import { expect, test } from "./fixtures";

const TABLE_NAME = "ACCOUNT";
const GRAPH_NAME = "graph_d1fa556b";

test.describe("Route tests", () => {
    test.beforeEach(async ({ sidebarHelper, page }) => {
        await sidebarHelper.gotoHome();
        await page.getByTestId("main-layout").waitFor({ state: "visible" });
    });

    test("connectors page loads", async ({ sidebarHelper, page }) => {
        await sidebarHelper.navigateTo("Connectors");

        await expect(page).toHaveURL(`${SEEDED_PROJECT_BASE}/connectors/`);
        await expect(sidebarHelper.getHeading("Connectors")).toBeVisible();
        // The authenticated user's selected project supplies connector context;
        // no separate group selection is required.
        await expect(
            page.getByText("Get started by creating your first connector!")
        ).toBeVisible();
        await expect(sidebarHelper.addConnectorButton).toBeVisible();

        // Local fixture data belongs to the project's hidden upload connector,
        // so the external-connector page intentionally has no seeded row.
    });

    test("tables page loads", async ({ sidebarHelper, page }) => {
        await sidebarHelper.navigateTo("Tables");

        await page.waitForURL(`${SEEDED_PROJECT_BASE}/tables/`);
        await expect(sidebarHelper.getHeading("Tables")).toBeVisible();
        await expect(sidebarHelper.addTableButton).toBeVisible();
        await expect(sidebarHelper.scrollArea).toBeVisible();

        await sidebarHelper.searchAndSelect(TABLE_NAME);
        await page.waitForURL(`${SEEDED_PROJECT_BASE}/tables/${TABLE_NAME}`);
        await expect(sidebarHelper.getHeading(TABLE_NAME)).toBeVisible();
        await expect(
            sidebarHelper.getHeading("Column Statistics")
        ).toBeVisible();
        await expect(page.locator("#column-stats-table")).toBeVisible();
    });

    test("graphs page loads", async ({ sidebarHelper, page }) => {
        await sidebarHelper.navigateTo("Graphs");

        await page.waitForURL(`${SEEDED_PROJECT_BASE}/graphs/`);
        await expect(sidebarHelper.getHeading("Graphs")).toBeVisible();
        await expect(sidebarHelper.createGraphButton).toBeVisible();
        await expect(sidebarHelper.scrollArea).toBeVisible();

        await sidebarHelper.searchAndSelect(GRAPH_NAME);
        await page.waitForURL(`${SEEDED_PROJECT_BASE}/graphs/${GRAPH_NAME}`);
        await expect(sidebarHelper.getHeading(GRAPH_NAME)).toBeVisible();
        await expect(page.locator('[data-testid="rf__wrapper"]')).toBeVisible();
        await expect(page.locator("#graph-overview-dropdown")).toBeVisible();
    });

    test("models page loads", async ({ sidebarHelper, page }) => {
        await sidebarHelper.navigateTo("Models");

        await page.waitForURL(
            `${SEEDED_PROJECT_BASE}/models/?page=1&job_type=TRAINING_JOB`
        );
        await expect(sidebarHelper.getHeading("Models")).toBeVisible();
        await expect(sidebarHelper.newModelButton).toBeVisible();
        await expect(page.locator("#allJobs")).toBeVisible();
    });

    test("predictions page loads", async ({ sidebarHelper, page }) => {
        await sidebarHelper.navigateTo("Predictions");

        await page.waitForURL(
            `${SEEDED_PROJECT_BASE}/predictions/?page=1&job_type=BATCH_PREDICTION_JOB`
        );
        await expect(sidebarHelper.getHeading("Predictions")).toBeVisible();
        // This project has only its hidden upload connector. The prediction
        // page therefore exposes the external-connector prerequisite and CTA.
        await expect(
            page.getByText("Get started by creating your first connector!")
        ).toBeVisible();
        await expect(sidebarHelper.addConnectorButton).toBeVisible();
    });
});
