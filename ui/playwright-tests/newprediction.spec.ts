// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SEEDED_PROJECT_BASE } from "./constants";
import { expect, test } from "./fixtures";

const TRAINING_JOB_TAG = "playwright-local-training";

const now = new Date();
const CURRENT_MONTH = now.toLocaleString("en-US", { month: "long" });
const CURRENT_MONTH_NUMBER = now.getMonth() + 1;
const CURRENT_YEAR = now.getFullYear().toString();
const CURRENT_YEAR_NUMBER = now.getFullYear();

const SELECTED_DAY = 13;
const SELECTED_MONTH = CURRENT_MONTH;
const SELECTED_MONTH_NUMBER = CURRENT_MONTH_NUMBER;
const SELECTED_YEAR_NUMBER = CURRENT_YEAR_NUMBER - 1;
const SELECTED_YEAR = SELECTED_YEAR_NUMBER.toString();

const SELECTED_HOURS = "08";
const SELECTED_MINUTES = "30";
const SELECTED_SECONDS = "45";

test.describe("New prediction page", () => {
    test.beforeEach(async ({ newPredictionPage, page }) => {
        await newPredictionPage.gotoPredictionsPage();
        await newPredictionPage.openNewJobDropdown();
        await newPredictionPage.selectPredictionJob();
        await page.waitForURL(`${SEEDED_PROJECT_BASE}/newjob/prediction/`);
    });

    test("select training job and set prediction anchor time", async ({
        newPredictionPage
    }) => {
        await expect(newPredictionPage.pageHeading).toBeVisible();
        await newPredictionPage.searchAndSelectTrainingJob(TRAINING_JOB_TAG);
        await expect(newPredictionPage.pageHeading).toBeVisible();
        await expect(newPredictionPage.startPredictingButton).toBeVisible();
        await expect(newPredictionPage.startPredictingButton).toBeDisabled();
        await expect(newPredictionPage.predictionAnchorTime).toBeVisible();

        await newPredictionPage.waitForQueryValidation(30000);
        await newPredictionPage.clickSelectDate();

        await expect(newPredictionPage.selectMonthCombobox).toContainText(
            CURRENT_MONTH
        );
        await expect(newPredictionPage.selectYearCombobox).toContainText(
            CURRENT_YEAR
        );
        await expect(newPredictionPage.currentDayCell).toBeVisible();

        await newPredictionPage.goToPreviousMonth(12);
        await expect(newPredictionPage.selectMonthCombobox).toContainText(
            SELECTED_MONTH
        );
        await expect(newPredictionPage.selectYearCombobox).toContainText(
            SELECTED_YEAR
        );

        await newPredictionPage.selectDay(SELECTED_DAY);
        await expect(
            newPredictionPage.getDateDisplay(
                SELECTED_DAY,
                SELECTED_MONTH_NUMBER,
                SELECTED_YEAR_NUMBER
            )
        ).toBeVisible();

        await newPredictionPage.setHours(SELECTED_HOURS);
        await newPredictionPage.setMinutes(SELECTED_MINUTES);
        await newPredictionPage.setSeconds(SELECTED_SECONDS);
        await newPredictionPage.waitForQueryValidation();
        await expect(
            newPredictionPage.getDateDisplay(
                SELECTED_DAY,
                SELECTED_MONTH_NUMBER,
                SELECTED_YEAR_NUMBER,
                `${SELECTED_HOURS}:${SELECTED_MINUTES}:${SELECTED_SECONDS}`
            )
        ).toBeVisible();
    });
});
