// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Locator, Page } from "@playwright/test";

import { SharedLocators } from "./SharedLocators";

export class SidebarHelper extends SharedLocators {
    sidebarMenu: Locator;
    scrollArea: Locator;
    addConnectorButton: Locator;
    addTableButton: Locator;
    createGraphButton: Locator;
    newModelButton: Locator;

    constructor(page: Page) {
        super(page);
        this.sidebarMenu = page.locator('[data-sidebar="menu-button"]');
        this.scrollArea = page.locator('[data-slot="scroll-area"]');
        this.addConnectorButton = page.getByRole("button", {
            name: "Add connector"
        });
        this.addTableButton = page.getByRole("button", { name: "Add table" });
        this.createGraphButton = page.getByRole("button", {
            name: "Create graph"
        });
        this.newModelButton = page.getByRole("button", { name: "New model" });
    }

    async gotoHome() {
        await this.page.goto("/");
    }

    async navigateTo(
        menuItem: "Connectors" | "Tables" | "Graphs" | "Models" | "Predictions"
    ) {
        const menu = this.sidebarMenu.filter({ hasText: menuItem });
        await menu.waitFor({ state: "visible" });
        await menu.click();
        await this.page.waitForLoadState("networkidle");
    }

    getHeading(name: string): Locator {
        return this.page.getByRole("heading", { name });
    }
}
