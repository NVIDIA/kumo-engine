// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, Locator, Page } from "@playwright/test";

import { SEEDED_PROJECT_BASE } from "../constants";
import { SharedLocators } from "../helpers/SharedLocators";

export class NewPredictionPage extends SharedLocators {
    predictionJobOption: Locator;
    pageHeading: Locator;
    startPredictingButton: Locator;
    predictionAnchorTime: Locator;
    selectMonthCombobox: Locator;
    selectYearCombobox: Locator;
    currentDayCell: Locator;
    selectDateButton: Locator;
    previousMonthButton: Locator;
    hoursInput: Locator;
    minutesInput: Locator;
    secondsInput: Locator;

    constructor(page: Page) {
        super(page);
        this.predictionJobOption = page
            .getByRole("button")
            .filter({ has: page.getByText("Prediction", { exact: true }) });
        this.pageHeading = page.getByRole("heading", {
            name: "New Prediction",
            exact: true
        });
        this.startPredictingButton = page.getByRole("button", {
            name: "Start predicting"
        });
        this.predictionAnchorTime = page.getByText("Prediction Anchor Time");
        this.selectMonthCombobox = page.getByRole("combobox", {
            name: "Select month"
        });
        this.selectYearCombobox = page.getByRole("combobox", {
            name: "Select year"
        });
        // Target today's cell by the auto-focused day button rather than
        // matching the bare day number. The current-month grid also renders
        // adjacent-month overflow cells, so a day number like "2" matches both
        // this month's day and next month's overflow day (which broke the
        // previous `.last()` selector). react-day-picker focuses today when no
        // date is selected, so today's gridcell is the only one with
        // tabindex="0".
        this.currentDayCell = page
            .getByRole("dialog")
            .getByRole("gridcell", {
                name: String(new Date().getDate()),
                exact: true
            })
            .filter({ hasNot: page.locator(".rdp-day_outside") })
            .first();
        this.selectDateButton = page.getByRole("button", {
            name: "Select Date"
        });
        this.previousMonthButton = page.getByRole("button", {
            name: "Go to previous month"
        });
        this.hoursInput = page.getByRole("spinbutton", { name: "Hours" });
        this.minutesInput = page.getByRole("spinbutton", { name: "Minutes" });
        this.secondsInput = page.getByRole("spinbutton", { name: "Seconds" });
    }

    async gotoPredictionsPage() {
        await this.page.goto(
            `${SEEDED_PROJECT_BASE}/predictions/?page=1&job_type=BATCH_PREDICTION_JOB`
        );
    }

    async selectPredictionJob() {
        await this.predictionJobOption.waitFor({ state: "visible" });
        await this.predictionJobOption.click();
    }

    async searchAndSelectTrainingJob(name: string) {
        await this.searchInput.waitFor({ state: "visible" });
        await this.searchInput.click();
        await this.searchInput.fill(name);

        const jobRow = this.page
            .getByRole("row")
            .filter({ has: this.page.getByText(name, { exact: true }) })
            .last();
        await jobRow.waitFor({ state: "visible" });
        await jobRow.click();
        await this.startPredictingButton.waitFor({ state: "visible" });
    }

    async clickSelectDate() {
        await this.selectDateButton.waitFor({ state: "visible" });
        await this.selectDateButton.click();
    }

    async waitForQueryValidation(timeout = 15000) {
        const checkingQuery = this.page.getByText("Checking query");
        await checkingQuery
            .waitFor({ state: "visible", timeout: 1000 })
            .catch(() => {
                // The validation can already be settled by the time this helper runs.
            });
        await expect(checkingQuery).toBeHidden({ timeout });
    }

    async goToPreviousMonth(count: number) {
        await this.previousMonthButton.waitFor({ state: "visible" });
        for (let i = 0; i < count; i++) {
            const currentMonth = await this.selectMonthCombobox.innerText();
            await this.previousMonthButton.click();
            await expect(this.selectMonthCombobox).not.toHaveText(currentMonth);
        }
    }

    async selectCurrentDay() {
        await this.currentDayCell.waitFor({ state: "visible" });
        await this.currentDayCell.click();
    }

    async selectDay(day: number) {
        const cell = this.page.getByRole("gridcell", {
            name: String(day),
            exact: true
        });
        await cell.waitFor({ state: "visible" });
        await cell.click();
    }

    async setHours(value: string) {
        await this.hoursInput.fill(value);
    }

    async setMinutes(value: string) {
        await this.minutesInput.fill(value);
    }

    async setSeconds(value: string) {
        await this.secondsInput.fill(value);
    }

    getDateDisplay(
        day: number,
        month?: number,
        year?: number,
        time = "00:00:00"
    ): Locator {
        const today = new Date();
        const m = month ?? today.getMonth() + 1;
        const y = year ?? today.getFullYear();
        const formattedMonth = String(m).padStart(2, "0");
        const formattedDay = String(day).padStart(2, "0");
        return this.page.getByText(
            `${y}-${formattedMonth}-${formattedDay} ${time}`,
            { exact: true }
        );
    }
}
