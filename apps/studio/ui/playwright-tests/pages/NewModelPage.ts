// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Locator, Page } from "@playwright/test";

import { SEEDED_PROJECT_BASE } from "../constants";
import { SharedLocators } from "../helpers/SharedLocators";

export class NewModelPage extends SharedLocators {
    modelJobOption: Locator;
    pageHeading: Locator;
    graphDropdown: Locator;
    pqlEditor: Locator;

    constructor(page: Page) {
        super(page);
        this.modelJobOption = page
            .getByRole("button")
            .filter({ has: page.getByText("Model", { exact: true }) });
        this.pageHeading = page.getByRole("heading", {
            name: "New Model",
            exact: true
        });
        this.graphDropdown = page.getByTestId("testid-select-graph");
        this.pqlEditor = page.getByRole("textbox");
    }

    async gotoModelsPage() {
        await this.page.goto(
            `${SEEDED_PROJECT_BASE}/models/?page=1&job_type=TRAINING_JOB`
        );
    }

    async selectModelJob() {
        await this.modelJobOption.waitFor({ state: "visible" });
        await this.modelJobOption.click();
    }

    async enterPQL(input: string) {
        const editorContainer = this.page.getByTestId("pql-editor");
        await editorContainer.waitFor({ state: "visible" });
        const textbox = editorContainer.getByRole("textbox");
        await textbox.waitFor({ state: "visible" });
        await textbox.click();
        await textbox.fill(input);
    }
}
