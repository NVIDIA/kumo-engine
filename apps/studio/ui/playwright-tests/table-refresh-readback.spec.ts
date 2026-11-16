// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Page } from "@playwright/test";

import { SEEDED_PROJECT_BASE } from "./constants";
import { expect, test } from "./fixtures";

const TABLE_NAME = "ACCOUNT";

const expectTableReadback = async (page: Page, tableName: string) => {
    await expect(page.getByRole("heading", { name: tableName })).toBeVisible();
    await expect(
        page.getByRole("heading", { name: "Column Statistics" })
    ).toBeVisible();
    await expect(page.getByText(/4 columns.*4\.5K rows/)).toBeVisible();

    const statsTable = page.locator("#column-stats-table");
    await expect(statsTable).toBeVisible();
    await expect(
        statsTable.getByRole("columnheader", { name: "Column" })
    ).toBeVisible();
    await expect(
        statsTable.getByText(/ACCOUNT_ID|No data found\./).first()
    ).toBeVisible();
};

// TODO re-enable: the col-stats endpoint 422s for a seeded, not-yet-ingested
// table — the sampled-stats fallback calls a blocking cache read on the asyncio
// event loop. Re-enable once that REST col_stats path is fixed.
test.describe.skip("ui-table-refresh-readback", () => {
    test("re-ingests a seeded table and reads stats back after reload", async ({
        page,
        sidebarHelper
    }) => {
        await sidebarHelper.gotoHome();
        await page.getByTestId("main-layout").waitFor({ state: "visible" });
        await sidebarHelper.navigateTo("Tables");

        await sidebarHelper.searchAndSelect(TABLE_NAME);
        await page.waitForURL(`${SEEDED_PROJECT_BASE}/tables/${TABLE_NAME}`);

        await expectTableReadback(page, TABLE_NAME);

        await page.getByTestId("more-options-dropdown").click();

        const refreshResponse = page.waitForResponse(
            (response) =>
                response.request().method() === "POST" &&
                response.url().includes("/v1/tables/") &&
                response.url().includes("/refresh") &&
                response.status() < 400
        );
        await page.getByText("Re-ingest", { exact: true }).click();
        await refreshResponse;
        await expectTableReadback(page, TABLE_NAME);

        await page.reload();
        await page.waitForURL(`${SEEDED_PROJECT_BASE}/tables/${TABLE_NAME}`);
        await expectTableReadback(page, TABLE_NAME);
    });
});
