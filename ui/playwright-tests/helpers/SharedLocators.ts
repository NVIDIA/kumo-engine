// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Locator, Page } from "@playwright/test";

export class SharedLocators {
    page: Page;
    newJobDropdown: Locator;
    searchInput: Locator;

    constructor(page: Page) {
        this.page = page;
        this.newJobDropdown = page.getByRole("button", {
            name: "New",
            exact: true
        });
        this.searchInput = page.getByPlaceholder("Search");
    }

    async openNewJobDropdown() {
        await this.newJobDropdown.waitFor({ state: "visible" });
        await this.newJobDropdown.click();
    }

    async searchAndSelect(name: string) {
        await this.searchInput.waitFor({ state: "visible" });
        await this.searchInput.click();
        await this.searchInput.fill(name);
        await this.page.getByText(name, { exact: true }).click();
        await this.page.waitForLoadState("domcontentloaded");
    }
}
