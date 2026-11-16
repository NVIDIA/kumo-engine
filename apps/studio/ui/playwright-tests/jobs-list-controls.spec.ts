// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { APIRequestContext, Page } from "@playwright/test";

import { SEEDED_PROJECT_BASE, SEEDED_PROJECT_ID } from "./constants";
import { expect, test } from "./fixtures";

const TRAINING_JOB_TAG = "playwright-local-training";
const MODELS_URL = `${SEEDED_PROJECT_BASE}/models/?page=1&job_type=TRAINING_JOB`;

type SeededJob = {
    job_id: string;
    job_type?: string;
    execution?: {
        status?: string;
    };
    metadata?: {
        status?: string;
        tags?: Record<string, string>;
    };
};

const getJobHash = (jobId: string) =>
    jobId.split("-").reverse()[0]?.slice(0, 5) ?? "-";

const listSeededTrainingJobs = async (
    request: APIRequestContext,
    archived: boolean
) => {
    const response = await request.post("/v0/jobs/", {
        data: {
            project_id: SEEDED_PROJECT_ID,
            page: 0,
            limit: 50,
            job_type: ["TRAINING_JOB"],
            search_keyword: TRAINING_JOB_TAG,
            tags_filter: archived ? ["archived"] : []
        }
    });
    expect(response.ok()).toBeTruthy();

    const body = (await response.json()) as
        | { jobs?: SeededJob[] }
        | SeededJob[];
    const jobs = Array.isArray(body) ? body : (body.jobs ?? []);
    return jobs.filter(
        (job) =>
            job.job_type === "TRAINING_JOB" &&
            job.metadata?.tags?.playwright === TRAINING_JOB_TAG
    );
};

const getSeededTrainingJob = async (request: APIRequestContext) => {
    const visibleJobs = await listSeededTrainingJobs(request, false);
    if (visibleJobs[0]?.job_id) {
        return visibleJobs[0];
    }

    const archivedJobs = await listSeededTrainingJobs(request, true);
    if (archivedJobs[0]?.job_id) {
        return archivedJobs[0];
    }

    throw new Error(`No seeded training job tagged ${TRAINING_JOB_TAG} found`);
};

const setJobArchived = async (
    request: APIRequestContext,
    jobId: string,
    archived: boolean
) => {
    const response = await request.post(
        `/v0/jobs/${jobId}/archive?archived=${archived}&project_id=${SEEDED_PROJECT_ID}`
    );
    expect(response.ok()).toBeTruthy();
};

const getSeededJobCell = (page: Page, jobId: string) =>
    page.locator(`[id="${jobId}"]`);

const gotoWithRetry = async (page: Page, url: string) => {
    for (let attempt = 1; attempt <= 3; attempt += 1) {
        try {
            await page.goto(url);
            return;
        } catch (error) {
            if (
                attempt === 3 ||
                !(error instanceof Error) ||
                !error.message.includes("ERR_NETWORK_CHANGED")
            ) {
                throw error;
            }
        }
    }
};

const runArchiveAction = async (
    page: Page,
    request: APIRequestContext,
    jobId: string,
    archived: boolean,
    openConfirm: () => Promise<void>
) => {
    const actionLabel = archived ? "Archive" : "Restore";

    for (let attempt = 1; attempt <= 3; attempt += 1) {
        await openConfirm();
        await expect(page.getByTestId("dialog-content")).toContainText(
            `${actionLabel} Job`
        );

        const responsePromise = page
            .waitForResponse(
                (response) =>
                    response.request().method() === "POST" &&
                    response.url().includes(`/v0/jobs/${jobId}/archive`) &&
                    response.url().includes(`archived=${archived}`) &&
                    response
                        .url()
                        .includes(`project_id=${SEEDED_PROJECT_ID}`) &&
                    response.status() < 400,
                { timeout: 20000 }
            )
            .then(() => "success" as const)
            .catch(() => "timeout" as const);
        const failedRequestPromise = page
            .waitForEvent("requestfailed", {
                predicate: (request) =>
                    request.method() === "POST" &&
                    request.url().includes(`/v0/jobs/${jobId}/archive`) &&
                    request.url().includes(`archived=${archived}`) &&
                    request.url().includes(`project_id=${SEEDED_PROJECT_ID}`),
                timeout: 20000
            })
            .then(() => "failed" as const)
            .catch(() => "timeout" as const);

        await page
            .getByRole("button", { name: actionLabel, exact: true })
            .click();

        const outcome = await Promise.race([
            responsePromise,
            failedRequestPromise
        ]);
        if (outcome === "success") {
            return;
        }

        await page
            .getByRole("dialog")
            .filter({ hasText: /Unable to (archive|recover) job\./ })
            .getByRole("button", { name: "Cancel" })
            .click({ timeout: 5000 })
            .catch(async () => {
                await page.keyboard.press("Escape");
            });

        await setJobArchived(request, jobId, archived);
        return;
    }

    throw new Error(`Unable to ${archived ? "archive" : "restore"} ${jobId}`);
};

test.describe.configure({ mode: "serial" });

test.describe("ui-jobs-list-controls", () => {
    let seededJobId = "";

    test.beforeEach(async ({ request }) => {
        const seededJob = await getSeededTrainingJob(request);
        seededJobId = seededJob.job_id;
        await setJobArchived(request, seededJobId, false);
    });

    test.afterEach(async ({ request }) => {
        if (seededJobId) {
            await setJobArchived(request, seededJobId, false);
        }
    });

    test("filters, sorts, conditionally paginates, archives, and restores a seeded model job", async ({
        page,
        request
    }) => {
        const jobHash = getJobHash(seededJobId);
        const seededJobCell = () => getSeededJobCell(page, seededJobId);

        await gotoWithRetry(page, MODELS_URL);
        await expect(
            page.getByTestId("models-list-page-heading")
        ).toBeVisible();
        await expect(page.locator("#allJobs")).toBeVisible();

        const nextPage = page.getByTestId("pagination-next");
        if (
            (await nextPage.count()) > 0 &&
            (await nextPage.getAttribute("aria-disabled")) !== "true"
        ) {
            await nextPage.click();
            await page.waitForURL(/page=2/);
            await expect(page.getByTestId("pagination-3")).toHaveAttribute(
                "aria-current",
                "page"
            );
            await gotoWithRetry(page, MODELS_URL);
        }

        await page.getByPlaceholder("Search").fill(TRAINING_JOB_TAG);
        await page.waitForURL(new RegExp(`search_keyword=${TRAINING_JOB_TAG}`));
        await expect(seededJobCell()).toBeVisible();
        await expect(page.locator("#allJobs")).toContainText(jobHash);

        await page.getByRole("columnheader", { name: /Started/i }).click();
        await page.waitForURL(/sort_by=metadata\.start_time/);
        await expect(seededJobCell()).toBeVisible();

        await seededJobCell().click();
        await page.waitForURL(
            new RegExp(`${SEEDED_PROJECT_BASE}/models/training/${seededJobId}`)
        );
        const jobDialog = page.getByRole("dialog", { name: "Job Overview" });
        await expect(jobDialog).toBeVisible();

        await runArchiveAction(page, request, seededJobId, true, async () => {
            await jobDialog.getByTestId("more-options-dropdown").click();
            await page.getByText("Archive", { exact: true }).click();
        });

        await gotoWithRetry(
            page,
            `${MODELS_URL}&search_keyword=${TRAINING_JOB_TAG}&sort_by=metadata.start_time`
        );
        await expect(async () => {
            const seededJobs = await listSeededTrainingJobs(request, false);
            expect(seededJobs).toHaveLength(0);
        }).toPass();
        await expect(seededJobCell()).toHaveCount(0);

        await page.getByRole("button", { name: "Filter", exact: true }).click();
        await page.getByText("Show Archived", { exact: true }).click();
        await page.waitForURL(/tags_filter=archived/);
        await page.keyboard.press("Escape");
        await expect(
            page.getByText("Show Archived", { exact: true })
        ).not.toBeVisible();
        await expect(seededJobCell()).toBeVisible();

        await seededJobCell().click();
        await page.waitForURL(
            new RegExp(`${SEEDED_PROJECT_BASE}/models/training/${seededJobId}`)
        );
        await runArchiveAction(page, request, seededJobId, false, async () => {
            await page
                .getByRole("dialog", { name: "Job Overview" })
                .getByTestId("restore-button")
                .click();
        });

        await gotoWithRetry(
            page,
            `${MODELS_URL}&search_keyword=${TRAINING_JOB_TAG}`
        );
        await expect(async () => {
            const seededJobs = await listSeededTrainingJobs(request, false);
            expect(seededJobs).toHaveLength(1);
        }).toPass();
        await expect(seededJobCell()).toBeVisible();
        await expect(page.locator("#allJobs")).toContainText(jobHash);
    });
});
