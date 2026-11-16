// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { focusManager, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TrainingWorkflowResponse } from "gen/v0/types";
import { setDeploymentTypeInStore } from "stores/__tests__/debugInfoStore";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import { testResponseType } from "tests/testResponseType";
import { deploymentType } from "tests/testType";
import { createQueryClient, normalizeString } from "tests/testUtil";

import { setAddTagTestResponseType } from "components/Jobs/Prediction/__tests__/MockFunctions/AddTag.mock";
import { setDeleteTagTestResponseType } from "components/Jobs/Prediction/__tests__/MockFunctions/DeleteTag.mock";

import { PredictiveQueryInfo } from "types/predictiveQueries";
import { CTAInfo } from "types/warnings";

import { getProjectPath, jobStatus } from "utils/Constants";
import { parseTrainingWorkflowResponse } from "utils/Converters/predictiveQueriesConverter";

import TrainingJobOverview, {
    TrainingJobOverviewDetailProps
} from "../TrainingJobOverview";

import { minimalTrainingWorkflow } from "./MockFunctions/TrainingJobWorkflow.mock";
import mockWorkflowResponse from "./MockResponses/TrainingWorkflow.test.json";
import mockWorkflowResponseCancelledJob from "./MockResponses/TrainingWorkflowCancelled.test.json";
import mockWorkflowResponseFailedJob from "./MockResponses/TrainingWorkflowFailed.test.json";
import mockWorkflowResponseQueuedJob from "./MockResponses/TrainingWorkflowQueued.test.json";
import mockWorkflowResponseRunningJob from "./MockResponses/TrainingWorkflowRunning.test.json";

const id = "trainingjob-00059000000000000000000000000000";
const displayId = id.replace("trainingjob-", "").slice(0, 5);

const expectedWorkflow = { ...mockWorkflowResponse };
expectedWorkflow.training_job.id = id;
const queryInfo = parseTrainingWorkflowResponse(
    expectedWorkflow as unknown as TrainingWorkflowResponse
);

// Minimal parsed queryInfo for header/tag/action tests that don't assert on
// evaluation or explanation content. Nulls the eval-driving fields so those
// child components (and their query cascade) don't mount, but preserves the
// tags so tag add/delete flows behave identically. Much faster than the full
// queryInfo above.
const minimalQueryInfo = parseTrainingWorkflowResponse(
    minimalTrainingWorkflow(id, "DONE", {
        test_new_tag: "test_new_tag",
        pquery_id: "pquery-0002e000000000000000000000000000"
    }) as unknown as TrainingWorkflowResponse
);

const defaultProps: TrainingJobOverviewDetailProps = {
    hidePreTitle: false,
    title: id,
    buttons: [],
    validationComponent: <></>,
    handleError: () => {},
    ctaMsgs: null as unknown as CTAInfo,
    loadingStatus: false,
    onClose: () => {},
    showRelatedJobs: false,
    onClickArchive: () => {},
    onClickRecover: () => {},
    queryInfo: queryInfo,
    warnings: []
};

describe("TrainingJobOverview", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });
    const queryClient = createQueryClient();

    const renderTrainingJobOverviewDetail = async (
        props: TrainingJobOverviewDetailProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={["/project/project-test/models/training/"]}>
                    <QueryClientProvider client={queryClient}>
                        <TrainingJobOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const waitForJobIdToRender = async () => {
        const jobId = await screen.findAllByText(displayId);
        expect(jobId[0]).toBeInTheDocument();
    };

    const testSDKCode = async () => {
        const dropdownButton = await screen.findByTestId(
            "more-options-dropdown"
        );
        await userEvent.click(dropdownButton);

        const sdkCodeItem = await screen.findByText(/SDK Code/i);
        expect(sdkCodeItem).toBeInTheDocument();

        await userEvent.click(sdkCodeItem);

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/training_jobs/${id}/sdk`,
                undefined,
                { headers: undefined, params: { project_id: "project-tests" } }
            );
        });

        const modal = await screen.findByTestId("sdk-content");
        expect(modal).toBeInTheDocument();
        expect(modal).toHaveTextContent(/View SDK Code/i);

        const copyButton = await screen.findByRole("button", { name: "Copy" });
        expect(copyButton).toBeInTheDocument();

        const downloadButton = await screen.findByRole("button", {
            name: /Download/i
        });
        expect(downloadButton).toBeInTheDocument();

        const closeIcons = screen.getAllByTestId("close-icon");
        expect(closeIcons.length).toBeGreaterThan(1);

        await userEvent.click(closeIcons[1]);

        await waitFor(() => {
            expect(screen.queryByTestId("sdk-content")).not.toBeInTheDocument();
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        mockNavigateCall();
        setDeploymentTypeInStore(deploymentType.SAAS);
        queryClient.clear();
        // Suppress RUNNING/QUEUED refetchInterval timer accumulation across
        // tests.
        focusManager.setFocused(false);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
    });

    it("Should render TrainingJobOverview page", async () => {
        await renderTrainingJobOverviewDetail(defaultProps);

        await waitFor(async () => {
            await screen.findAllByText("Training Job");
        });

        await waitForJobIdToRender();

        const queryCard = await waitFor(() => {
            const el = document.querySelector(
                '[role="textbox"][data-language="PQL1"]'
            );
            expect(el).toBeInTheDocument();
            return el as HTMLElement;
        });
        const cmLines = queryCard.querySelectorAll(".cm-line");
        const queryText = Array.from(cmLines)
            .map((line) => line.textContent || "")
            .join("\n");
        expect(normalizeString(queryText)).toBe(
            normalizeString(expectedWorkflow.pquery_definition.query_string)
        );

        const status = await screen.findByText("Done");
        expect(status).toBeInTheDocument();

        const texts = [
            "Model Performance",
            "Evaluation",
            "Training",
            "Explanation"
        ];
        texts.forEach((text) => {
            const element = screen.getAllByText(text);
            expect(element.length).toBeGreaterThanOrEqual(0);
        });

        expect(screen.queryByText("GRAPH SNAPSHOT")).not.toBeInTheDocument();
        expect(
            screen.queryByText("TRAINING TABLE GENERATION")
        ).not.toBeInTheDocument();
    });

    it("Should show elapsed-over-window progress while the Training stage runs", async () => {
        // A 2-minute estimate window with 1 minute elapsed: exactly 50%.
        const start = "2026-07-17T01:19:31.237505Z";
        const end = "2026-07-17T01:21:31.237505Z";
        await renderTrainingJobOverviewDetail({
            ...defaultProps,
            buttons: [<button key="cancel">Cancel</button>],
            queryInfo: {
                ...minimalQueryInfo,
                trainingStatus: jobStatus.Running,
                log: {
                    stage_name: "Training",
                    last_updated_at: start,
                    detail: "This job is currently training your model."
                },
                trialProgress: {
                    start_time: start,
                    // Wire format since the pydantic v2 migration: an
                    // ISO-8601 duration string (kumo#29935).
                    elapsed_time: "PT1M0S",
                    estimated_end_time: end
                }
            } as PredictiveQueryInfo
        });

        const bar = await waitFor(() => {
            const el = document.querySelector('[role="progressbar"]');
            expect(el).toBeInTheDocument();
            return el as HTMLElement;
        });
        // The catalog ProgressBar conveys its value through the indicator's
        // translateX(-(100 - progress)%) transform.
        const style =
            bar
                .querySelector('[data-slot="progress-indicator"]')
                ?.getAttribute("style") ?? "";
        const offset = /translateX\(-([\d.]+)%\)/.exec(style);
        expect(100 - Number(offset?.[1])).toBe(50);
    });

    describe("SDK Code Modal", () => {
        // SDK Code is driven by the `/sdk` endpoint and is status-gated only —
        // it does not depend on eval/explanation content. Use a minimal
        // workflow per status so the eval cascade never mounts; the header
        // dropdown + SDK modal still render and behave identically.
        it.each([
            ["Done", "DONE", {}],
            ["Cancelled", "CANCELLED", {}],
            ["Failed", "FAILED", { showRelatedJobs: true }],
            ["Running", "RUNNING", {}],
            ["Queued", "QUEUED", {}]
        ])(
            "Should render 'SDK Code' button and open modal for jobStatus '%s'",
            async (_status, rawStatus, extraProps) => {
                const queryInfo = parseTrainingWorkflowResponse(
                    minimalTrainingWorkflow(
                        id,
                        rawStatus
                    ) as unknown as TrainingWorkflowResponse
                );

                await renderTrainingJobOverviewDetail({
                    ...defaultProps,
                    ...extraProps,
                    queryInfo: queryInfo
                });

                await waitForJobIdToRender();
                await testSDKCode();
            }
        );
    });

    it("Should navigate to `New Prediction` page when the `Generate predictions` button is clicked", async () => {
        await renderTrainingJobOverviewDetail(defaultProps);

        await waitForJobIdToRender();

        const queryCard = await waitFor(() => {
            const el = document.querySelector(
                '[role="textbox"][data-language="PQL1"]'
            );
            expect(el).toBeInTheDocument();
            return el as HTMLElement;
        });
        const cmLines = queryCard.querySelectorAll(".cm-line");
        const queryText = Array.from(cmLines)
            .map((line) => line.textContent || "")
            .join("\n");
        expect(normalizeString(queryText)).toBe(
            normalizeString(expectedWorkflow.pquery_definition.query_string)
        );

        const status = await screen.findByText("Done");
        expect(status).toBeInTheDocument();

        const generatePredButton = await screen.findByRole("button", {
            name: "Generate predictions"
        });
        await userEvent.click(generatePredButton);

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                getProjectPath("project-tests").NewPredictionJob,
                { state: { queryInfo } }
            );
        });
    });

    it("Should render TrainingJobOverview page for `empty` queryInfo", async () => {
        await renderTrainingJobOverviewDetail({
            ...defaultProps,
            queryInfo: {} as unknown as PredictiveQueryInfo
        });

        await waitForJobIdToRender();
    });

    it("Should render `buttons` in TrainingJobOverview page", async () => {
        // Only asserts a passed-in button renders — independent of eval data.
        const runningQueryInfo = parseTrainingWorkflowResponse(
            minimalTrainingWorkflow(
                id,
                "RUNNING"
            ) as unknown as TrainingWorkflowResponse
        );

        await renderTrainingJobOverviewDetail({
            ...defaultProps,
            queryInfo: runningQueryInfo,
            buttons: [
                <button key="test-btn" data-testid="test-button-123">
                    Test button
                </button>
            ]
        });

        await waitForJobIdToRender();

        const button = await screen.findByTestId("test-button-123");
        expect(button).toBeInTheDocument();
    });

    it("Should render TrainingJobOverview page with cta info in Training setting", async () => {
        // The CTA info icon/tooltip is driven by `ctaMsgs`, not by eval data,
        // so the minimal (DONE) queryInfo renders the Training tab + icon
        // without mounting the eval/explanation cascade.
        await renderTrainingJobOverviewDetail({
            ...defaultProps,
            queryInfo: minimalQueryInfo,
            ctaMsgs: {
                info_items: [
                    {
                        id: "split_adjustment_for_empty_validation_set",
                        description: "Test cta info"
                    }
                ]
            } as unknown as CTAInfo
        });

        await waitForJobIdToRender();

        const trainingTab = await screen.findByRole("tab", {
            name: "Training"
        });
        await userEvent.click(trainingTab);

        const svgContainer = screen.getByTestId("training-info-icon");
        await userEvent.hover(svgContainer);

        const tooltip = await screen.findAllByText("Test cta info");
        expect(tooltip[0]).toBeInTheDocument();
    });

    describe("Job Status Display", () => {
        it.each([
            [
                "Failed",
                mockWorkflowResponseFailedJob,
                { showRelatedJobs: true },
                "Training job failed",
                /workflow did not complete due to the following validation issues/i
            ],
            ["Canceled", mockWorkflowResponseCancelledJob, {}, null, null],
            [
                "Running",
                mockWorkflowResponseRunningJob,
                {},
                "Ingesting Data",
                null
            ],
            [
                "Queued",
                mockWorkflowResponseQueuedJob,
                {},
                "This training job is queued",
                null
            ]
        ])(
            "Should render TrainingJobOverview page for '%s' status",
            async (
                expectedStatus,
                mockResponse,
                extraProps,
                alertMessage,
                optionalMessage
            ) => {
                const workflow = { ...mockResponse };
                workflow.training_job.id = id;
                const queryInfo = parseTrainingWorkflowResponse(
                    workflow as unknown as TrainingWorkflowResponse
                );

                await renderTrainingJobOverviewDetail({
                    ...defaultProps,
                    ...extraProps,
                    queryInfo: queryInfo
                });

                await waitForJobIdToRender();

                const status = await screen.findAllByTestId(
                    "header-status-chip-id"
                );
                expect(status[0].textContent).toBe(expectedStatus);

                if (alertMessage) {
                    const alert = await screen.findByText(alertMessage);
                    expect(alert).toBeInTheDocument();
                }

                if (optionalMessage) {
                    const description =
                        await screen.findByText(optionalMessage);
                    expect(description).toBeInTheDocument();
                }

                expect(
                    screen.queryByRole("button", {
                        name: "Generate Predictions"
                    })
                ).not.toBeInTheDocument();
            }
        );
    });

    describe("Duplicate Model", () => {
        it.each([
            ["Done", mockWorkflowResponse],
            ["Failed", mockWorkflowResponseFailedJob],
            ["Cancelled", mockWorkflowResponseCancelledJob]
        ])(
            "Should navigate to 'New Model' page when 'Duplicate' is clicked for job status '%s'",
            async (_status, mockResponse) => {
                const workflow = { ...mockResponse };
                workflow.training_job.id = id;
                const queryInfo = parseTrainingWorkflowResponse(
                    workflow as unknown as TrainingWorkflowResponse
                );
                queryInfo.tags = {
                    bp: "bp",
                    team: "ml",
                    pquery_id: "pquery-original",
                    archived: "archived"
                };

                await renderTrainingJobOverviewDetail({
                    ...defaultProps,
                    queryInfo: queryInfo
                });

                await waitForJobIdToRender();

                const dropdownButton = await screen.findByTestId(
                    "more-options-dropdown"
                );
                await userEvent.click(dropdownButton);

                const duplicateButton = await screen.findByText(/Duplicate/i);
                expect(duplicateButton).toBeInTheDocument();

                await userEvent.click(duplicateButton);
                await waitFor(() => {
                    expect(mockUseNavigate).toHaveBeenCalledWith(
                        getProjectPath("project-tests").NewTrainingJob,
                        {
                            state: expect.objectContaining({
                                copytrainingargs: expect.any(Object),
                                copytrainingtags: {
                                    bp: "bp",
                                    team: "ml"
                                }
                            })
                        }
                    );
                });
            }
        );

        it.each([
            ["Queued", "QUEUED"],
            ["Running", "RUNNING"]
        ])(
            "Should not show 'Duplicate' option for job status '%s'",
            async (_status, rawStatus) => {
                // Only asserts the Duplicate option is hidden (status-gated) —
                // no eval/nav-state assertion, so a minimal workflow is enough.
                const queryInfo = parseTrainingWorkflowResponse(
                    minimalTrainingWorkflow(
                        id,
                        rawStatus
                    ) as unknown as TrainingWorkflowResponse
                );

                await renderTrainingJobOverviewDetail({
                    ...defaultProps,
                    queryInfo: queryInfo
                });

                await waitForJobIdToRender();

                const dropdownButton = await screen.findByTestId(
                    "more-options-dropdown"
                );
                await userEvent.click(dropdownButton);

                const duplicateButton = screen.queryByText(/Duplicate/i);
                expect(duplicateButton).not.toBeInTheDocument();
            }
        );
    });

    describe("Tags", () => {
        // Tag flows don't depend on eval/explanation content — use the minimal
        // queryInfo so the eval query cascade doesn't mount (much faster).
        const tagProps = { ...defaultProps, queryInfo: minimalQueryInfo };

        it("Should add tag", async () => {
            await renderTrainingJobOverviewDetail(tagProps);

            const addTag = screen.getByText("Add tag");
            await userEvent.click(addTag);

            const keyInput = screen.getByPlaceholderText("key");
            const key = "tag_key";
            await userEvent.type(keyInput, key);

            const valueInput = screen.getByPlaceholderText("value");
            const value = "tag_value";
            await userEvent.type(valueInput, value);

            await userEvent.keyboard("{Enter}");

            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    "v0/jobs/add_tag",
                    {
                        job_id: "trainingjob-00059000000000000000000000000000",
                        key: key,
                        project_id: "project-tests",
                        value: value
                    },
                    {
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }
                );
            });

            expect(await screen.findByText(value)).toBeInTheDocument();
            expect(screen.getByText(/TAG_KEY=/i)).toBeInTheDocument();
        });

        it("Should delete tag", async () => {
            await renderTrainingJobOverviewDetail(tagProps);

            const closeButton = screen.getByTestId("tag-remove-icon");
            await userEvent.click(closeButton);

            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    "v0/jobs/delete_tag",
                    {
                        job_id: "trainingjob-00059000000000000000000000000000",
                        key: "test_new_tag",
                        project_id: "project-tests",
                        value: null
                    },
                    {
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }
                );
            });

            expect(screen.queryByText("test_new_tag")).not.toBeInTheDocument();
        });

        it.each([
            {
                operation: "add",
                responseType: testResponseType.ERROR,
                text: "tag_value",
                shouldExist: false
            },
            {
                operation: "delete",
                responseType: testResponseType.ERROR,
                text: "test_new_tag",
                shouldExist: true
            }
        ])(
            "Should not $operation tag when API returns error",
            async ({ operation, responseType, text, shouldExist }) => {
                if (operation === "add") {
                    setAddTagTestResponseType(responseType);
                } else {
                    setDeleteTagTestResponseType(responseType);
                }

                await renderTrainingJobOverviewDetail(tagProps);

                if (operation === "add") {
                    const addTag = screen.getByText("Add tag");
                    await userEvent.click(addTag);

                    const keyInput = screen.getByPlaceholderText("key");
                    const key = "tag_key";
                    await userEvent.type(keyInput, key);

                    const valueInput = screen.getByPlaceholderText("value");
                    const value = "tag_value";
                    await userEvent.type(valueInput, value);

                    await userEvent.keyboard("{Enter}");

                    await waitFor(() => {
                        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                            "v0/jobs/add_tag",
                            expect.any(Object),
                            expect.any(Object)
                        );
                    });
                } else {
                    const closeButton = screen.getByTestId("tag-remove-icon");
                    await userEvent.click(closeButton);

                    await waitFor(() => {
                        expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                            "v0/jobs/delete_tag",
                            expect.any(Object),
                            expect.any(Object)
                        );
                    });
                }

                await waitFor(() => {
                    const element = screen.queryByText(text);
                    if (shouldExist) {
                        expect(element).toBeInTheDocument();
                    } else {
                        expect(element).not.toBeInTheDocument();
                    }
                });
            }
        );
    });
});
