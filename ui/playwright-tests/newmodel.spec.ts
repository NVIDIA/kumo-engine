// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SEEDED_PROJECT_BASE } from "./constants";
import { expect, test } from "./fixtures";

const GRAPH_NAME = "graph_d1fa556b";
const PQL =
    "PREDICT SUM(LOAN.AMOUNT, 0, 360, days) FOR EACH ACCOUNT.ACCOUNT_ID";

test.describe("New job page", () => {
    test.beforeEach(async ({ newModelPage, page }) => {
        await newModelPage.gotoModelsPage();
        await newModelPage.openNewJobDropdown();
        await newModelPage.selectModelJob();
        await page.waitForURL(
            `${SEEDED_PROJECT_BASE}/newjob/training/graph-select/`
        );
    });

    test("enter predictive query in query editor", async ({
        newModelPage,
        page
    }) => {
        await expect(newModelPage.pageHeading).toBeVisible();
        await newModelPage.searchAndSelect(GRAPH_NAME);
        await expect(newModelPage.graphDropdown).toContainText(GRAPH_NAME);
        await newModelPage.enterPQL(PQL);
        await expect(page.getByText("Checking query")).toBeVisible();
        await expect(page.getByText("Looks good!")).toBeVisible();
    });
});
