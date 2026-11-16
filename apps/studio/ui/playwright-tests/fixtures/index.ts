// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { test as base } from "@playwright/test";

import { NewModelPage } from "../pages/NewModelPage";
import { NewPredictionPage } from "../pages/NewPredictionPage";

import { SEEDED_GROUP_ID, SEEDED_PROJECT_ID } from "../constants";
import { SidebarHelper } from "../helpers/SidebarHelper";

type Fixtures = {
    skipIntro: void;
    sidebarHelper: SidebarHelper;
    newModelPage: NewModelPage;
    newPredictionPage: NewPredictionPage;
};

export const test = base.extend<Fixtures>({
    skipIntro: [
        async ({ page }, provide) => {
            await page.addInitScript(
                ({ groupId, projectId }) => {
                    window.localStorage.setItem(
                        "new_user_v2",
                        JSON.stringify("1")
                    );
                    window.localStorage.setItem(
                        "selected-group",
                        JSON.stringify(groupId)
                    );
                    window.localStorage.setItem(
                        "selected-project",
                        JSON.stringify(projectId)
                    );
                },
                {
                    groupId: SEEDED_GROUP_ID,
                    projectId: SEEDED_PROJECT_ID
                }
            );
            await provide();
        },
        { auto: true }
    ],
    sidebarHelper: async ({ page }, provide) => {
        await provide(new SidebarHelper(page));
    },
    newModelPage: async ({ page }, provide) => {
        await provide(new NewModelPage(page));
    },
    newPredictionPage: async ({ page }, provide) => {
        await provide(new NewPredictionPage(page));
    }
});

export { expect } from "@playwright/test";
