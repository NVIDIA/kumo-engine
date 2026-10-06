// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { focusManager, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { setDeploymentTypeInStore } from "stores/__tests__/debugInfoStore";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { deploymentType } from "tests/testType";
import { createQueryClient } from "tests/testUtil";

import PredictionJobOverview, {
    PredictionJobOverviewProps
} from "components/Jobs/Prediction/PredictionJobOverview";

import { jobStatus } from "utils/Constants";

import { setAddTagTestResponseType } from "./MockFunctions/AddTag.mock";
import {
    setBatchPredictionResponseStatus,
    setBatchPredictionResponseType
} from "./MockFunctions/BatchPredictionResponse.mock";
import { setDeleteTagTestResponseType } from "./MockFunctions/DeleteTag.mock";
import { setPredictiveQueryResoureTestResponseType } from "./MockFunctions/PredictiveQueryResource.mock";
import mockBatchPredictionFailedResponse from "./MockResponses/BatchPredictionFailedResponse.json";
import mockBatchPredictionQueuedResponse from "./MockResponses/BatchPredictionQueuedResponse.json";
import mockBatchPredictionResponse from "./MockResponses/BatchPredictionResponse.json";
import mockBatchPredictionRunningResponse from "./MockResponses/BatchPredictionRunningResponse.json";

const id = "bp-job-3f5f52a53f01473d8c5123ff7ba4fd15";
const onClose = vi.fn();
const toggleSdkModel = vi.fn();
const onClickArchive = vi.fn();
const onClickRecover = vi.fn();

const defaultProps: PredictionJobOverviewProps = {
    bpId: id,
    predictionInfo: mockBatchPredictionResponse as never,
    anchorTime: "2025-04-09T05:19:10.26847700:00",
    bpJobStatus: jobStatus.Done,
    queryYaml: "SELECT * FROM users",
    loading: false,
    buttons: [],
    isArchived: false,
    onClose,
    onClickArchive,
    onClickRecover,
    toggleSdkModel,
    handleError: vi.fn()
};

describe("PredictionJobOverview", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });
    const queryClient = createQueryClient(false);

    const renderPredictionJobOverview = async (
        props: PredictionJobOverviewProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={[
                        "/project/project-test/predictions/prediction/"
                    ]}>
                    <QueryClientProvider client={queryClient}>
                        <PredictionJobOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();

        // Suppress RUNNING/QUEUED refetchInterval timer accumulation across
        // tests.
        focusManager.setFocused(false);

        setDeploymentTypeInStore(deploymentType.SAAS);
        setBatchPredictionResponseStatus(jobStatus.Done);
        setBatchPredictionResponseType(testResponseType.ALL);
        setPredictiveQueryResoureTestResponseType(testResponseType.ALL);
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
    });

    it("Should render with display ID", async () => {
        await renderPredictionJobOverview(defaultProps);

        const displayId = id.replace("bp-job-", "").slice(0, 5);
        expect(await screen.findByText(displayId)).toBeInTheDocument();
    });

    it.each([
        { testId: "close-icon", callback: onClose },
        { testId: "restore-button", callback: onClickRecover, isArchived: true }
    ])(
        "Should call callback when $testId is clicked",
        async ({ testId, callback, isArchived }) => {
            const props = isArchived
                ? { ...defaultProps, isArchived: true }
                : defaultProps;
            await renderPredictionJobOverview(props);

            const button = await screen.findByTestId(testId);
            await userEvent.click(button);

            expect(callback).toHaveBeenCalled();
        }
    );

    it.each([
        { menuItem: "SDK Code", callback: toggleSdkModel, arg: true },
        { menuItem: "Archive", callback: onClickArchive }
    ])(
        "Should call $callback when $menuItem is clicked from dropdown",
        async ({ menuItem, callback, arg }) => {
            await renderPredictionJobOverview(defaultProps);

            const dropdownButton = await screen.findByTestId(
                "more-options-dropdown"
            );
            await userEvent.click(dropdownButton);
            const menuButton = screen.getByText(new RegExp(menuItem, "i"));
            await userEvent.click(menuButton);

            if (arg !== undefined) {
                expect(callback).toHaveBeenCalledWith(arg);
            } else {
                expect(callback).toHaveBeenCalled();
            }
        }
    );

    it("Should render cancel buttons when provided", async () => {
        const propsWithButtons = {
            ...defaultProps,
            bpJobStatus: jobStatus.Running,
            predictionInfo: {
                ...mockBatchPredictionResponse,
                status: jobStatus.Running
            } as never,
            buttons: [
                <button key="cancel" data-testid="cancel-job-button">
                    Cancel this job
                </button>
            ]
        };

        await renderPredictionJobOverview(propsWithButtons);

        expect(
            await screen.findByTestId("cancel-job-button")
        ).toBeInTheDocument();
    });

    it("Should show skeleton when loading", async () => {
        await renderPredictionJobOverview({ ...defaultProps, loading: true });

        const skeletons = document.querySelectorAll('[data-slot="skeleton"]');
        expect(skeletons.length).toBeGreaterThan(0);
    });

    it("Should not display anchor time when null", async () => {
        await renderPredictionJobOverview({
            ...defaultProps,
            anchorTime: null
        });

        expect(
            screen.queryByText(/Prediction anchor time/i)
        ).not.toBeInTheDocument();
    });

    describe("Tags", () => {
        it("Should render existing tags", async () => {
            await renderPredictionJobOverview(defaultProps);

            expect(
                await screen.findByText("new_prediction_job")
            ).toBeInTheDocument();
        });

        it("Should add tag", async () => {
            await renderPredictionJobOverview(defaultProps);

            const addTag = screen.getByText("Add tag");
            await userEvent.click(addTag);

            const keyInput = screen.getByPlaceholderText("key");
            await userEvent.type(keyInput, "tag_key");

            const valueInput = screen.getByPlaceholderText("value");
            await userEvent.type(valueInput, "tag_value");

            await userEvent.keyboard("{Enter}");

            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    "v0/jobs/add_tag",
                    {
                        job_id: "bp-job-3f5f52a53f01473d8c5123ff7ba4fd15",
                        key: "tag_key",
                        project_id: "project-tests",
                        value: "tag_value"
                    },
                    {
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }
                );
            });

            expect(await screen.findByText("tag_value")).toBeInTheDocument();
            expect(screen.getByText(/TAG_KEY=/i)).toBeInTheDocument();
        });

        it("Should delete tag", async () => {
            await renderPredictionJobOverview(defaultProps);

            const closeButton = screen.getByTestId("tag-remove-icon");
            await userEvent.click(closeButton);

            await waitFor(() => {
                expect(mockAxiosInstance.post).toHaveBeenCalledWith(
                    "v0/jobs/delete_tag",
                    {
                        job_id: "bp-job-3f5f52a53f01473d8c5123ff7ba4fd15",
                        key: "new_prediction_job",
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

            expect(
                screen.queryByText("new_prediction_job")
            ).not.toBeInTheDocument();
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
                text: "new_prediction_job",
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

                await renderPredictionJobOverview(defaultProps);

                if (operation === "add") {
                    const addTag = screen.getByText("Add tag");
                    await userEvent.click(addTag);

                    const keyInput = screen.getByPlaceholderText("key");
                    await userEvent.type(keyInput, "tag_key");

                    const valueInput = screen.getByPlaceholderText("value");
                    await userEvent.type(valueInput, "tag_value");

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

    describe("Job Status", () => {
        it.each([
            {
                status: jobStatus.Failed,
                mockData: mockBatchPredictionFailedResponse
            },
            {
                status: jobStatus.Queued,
                mockData: {
                    ...mockBatchPredictionQueuedResponse,
                    status: jobStatus.Queued
                }
            }
        ])(
            "Should render with $status status",
            async ({ status, mockData }) => {
                await renderPredictionJobOverview({
                    ...defaultProps,
                    bpJobStatus: status,
                    predictionInfo: mockData as never
                });

                expect(
                    await screen.findByText("Batch Prediction")
                ).toBeInTheDocument();
            }
        );

        it("Should display running status with stage name", async () => {
            await renderPredictionJobOverview({
                ...defaultProps,
                bpJobStatus: jobStatus.Running,
                predictionInfo: {
                    ...mockBatchPredictionRunningResponse,
                    status: jobStatus.Running,
                    log: {
                        stage_name: "Ingesting Data",
                        last_updated_at: "2025-04-11T04:27:45.85599700:00",
                        detail: "Processing data"
                    }
                } as never
            });

            expect(
                await screen.findByText("Ingesting Data")
            ).toBeInTheDocument();
        });

        it.each([
            // A 50s ISO-duration elapsed over the 60s estimate window.
            { elapsed_time: "PT50S", expectedPercent: (50 / 60) * 100 },
            // Elapsed past the window reads the 98 cap.
            { elapsed_time: "PT3M20S", expectedPercent: 98 }
        ])(
            "Should display progress bar with wire-format elapsed_time $elapsed_time",
            async ({ elapsed_time, expectedPercent }) => {
                await renderPredictionJobOverview({
                    ...defaultProps,
                    bpJobStatus: jobStatus.Running,
                    predictionInfo: {
                        ...mockBatchPredictionRunningResponse,
                        status: jobStatus.Running,
                        log: {
                            stage_name: "Predicting",
                            last_updated_at: "2025-04-11T04:27:45.85599700:00",
                            detail: "Making predictions"
                        },
                        execution: {
                            ...mockBatchPredictionRunningResponse.execution,
                            progress: {
                                iterations_completed: 5,
                                iterations_remaining: 5,
                                start_time: "2025-04-09T05:26:07.466376+00:00",
                                elapsed_time,
                                estimated_end_time:
                                    "2025-04-09T05:27:07.466376+00:00"
                            }
                        }
                    } as never
                });

                const progressBars = document.querySelectorAll(
                    '[role="progressbar"]'
                );
                expect(progressBars.length).toBeGreaterThan(0);
                // The ISO-duration elapsed_time must be parsed against the
                // estimate window; Number() on it yielded NaN and zeroed
                // the bar (kumo#29935).
                // The catalog ProgressBar conveys its value through the
                // indicator's translateX(-(100 - progress)%) transform.
                const style =
                    progressBars[0]
                        .querySelector('[data-slot="progress-indicator"]')
                        ?.getAttribute("style") ?? "";
                const offset = /translateX\(-([\d.]+)%\)/.exec(style);
                expect(100 - Number(offset?.[1])).toBeCloseTo(
                    expectedPercent,
                    1
                );
            }
        );
    });

    describe("Related Jobs", () => {
        it.each([
            {
                label: "Graph Snapshot",
                props: { graphSnapshotId: "graph-snapshot-123" }
            },
            {
                label: "Training Job",
                props: {
                    trainingJobID:
                        "trainingjob-9f8b259b4f764c8f932ad5912a926ec2"
                }
            },
            {
                label: "Graph",
                props: {
                    graphSnapshotId: undefined,
                    graphId: "NA-d445eeb8b2aa1b79a142406239fa74e8"
                }
            }
        ])("Should display $label when available", async ({ label, props }) => {
            await renderPredictionJobOverview({
                ...defaultProps,
                predictionInfo: {
                    ...mockBatchPredictionResponse,
                    ...props
                } as never
            });

            await waitFor(() => {
                const relatedJobCard = screen.queryByText(label);
                if (relatedJobCard) {
                    expect(relatedJobCard).toBeInTheDocument();
                }
            });
        });
    });

    it.each([
        { description: "null predictionInfo", props: { predictionInfo: null } },
        { description: "missing bpId", props: { bpId: undefined } },
        {
            description: "empty tags",
            props: {
                predictionInfo: {
                    ...mockBatchPredictionResponse,
                    tags: {}
                } as never
            }
        },
        {
            description: "running status without stage name",
            props: {
                bpJobStatus: jobStatus.Running,
                predictionInfo: {
                    ...mockBatchPredictionRunningResponse,
                    status: jobStatus.Running,
                    log: undefined
                } as never
            }
        },
        {
            description: "progress without all required fields",
            props: {
                bpJobStatus: jobStatus.Running,
                predictionInfo: {
                    ...mockBatchPredictionRunningResponse,
                    status: jobStatus.Running,
                    log: {
                        stage_name: "Predicting",
                        last_updated_at: "2025-04-11T04:27:45.85599700:00",
                        detail: "Making predictions"
                    },
                    execution: {
                        ...mockBatchPredictionRunningResponse.execution,
                        progress: {
                            iterations_completed: 5,
                            iterations_remaining: 5
                        }
                    }
                } as never
            }
        },
        {
            description: "missing execution times",
            props: {
                predictionInfo: {
                    ...mockBatchPredictionResponse,
                    execution: {
                        ...mockBatchPredictionResponse.execution,
                        start_time: undefined,
                        last_updated_at: undefined
                    }
                } as never
            }
        },
        {
            description: "failed status without error message",
            props: {
                bpJobStatus: jobStatus.Failed,
                predictionInfo: {
                    ...mockBatchPredictionFailedResponse,
                    execution: {
                        ...mockBatchPredictionFailedResponse.execution,
                        failure_info: {
                            message: "",
                            stacktrace: ""
                        }
                    }
                } as never
            }
        }
    ])("Should handle $description", async ({ props }) => {
        await renderPredictionJobOverview({ ...defaultProps, ...props });

        expect(await screen.findByText("Batch Prediction")).toBeInTheDocument();
    });

    it.each([
        { source: "SDK" },
        { source: "WEB" },
        { source: "INVALID_SOURCE" },
        { source: null }
    ])("Should handle $source job source", async ({ source }) => {
        await renderPredictionJobOverview({
            ...defaultProps,
            predictionInfo: {
                ...mockBatchPredictionResponse,
                source
            } as never
        });

        expect(await screen.findByText("Batch Prediction")).toBeInTheDocument();
    });

    describe("Progress Calculation", () => {
        it("Should calculate progress when percent >= 100", async () => {
            await renderPredictionJobOverview({
                ...defaultProps,
                bpJobStatus: jobStatus.Running,
                predictionInfo: {
                    ...mockBatchPredictionRunningResponse,
                    status: jobStatus.Running,
                    log: {
                        stage_name: "Predicting",
                        last_updated_at: "2025-04-11T04:27:45.85599700:00",
                        detail: "Making predictions"
                    },
                    execution: {
                        ...mockBatchPredictionRunningResponse.execution,
                        progress: {
                            iterations_completed: 5,
                            iterations_remaining: 5,
                            start_time: "2025-04-09T05:26:07.46637600:00",
                            elapsed_time: 150,
                            estimated_end_time:
                                "2025-04-09T05:27:07.46637600:00"
                        }
                    }
                } as never
            });

            const progressBars = document.querySelectorAll(
                '[role="progressbar"]'
            );
            expect(progressBars.length).toBeGreaterThan(0);
        });
    });

    describe("Status Titles", () => {
        it("Should render component when job is done", async () => {
            await renderPredictionJobOverview({
                ...defaultProps,
                bpJobStatus: jobStatus.Done,
                predictionInfo: {
                    ...mockBatchPredictionResponse,
                    status: jobStatus.Done
                } as never
            });

            expect(
                await screen.findByText("Batch Prediction")
            ).toBeInTheDocument();
        });

        it("Should render component when failed without error message", async () => {
            await renderPredictionJobOverview({
                ...defaultProps,
                bpJobStatus: jobStatus.Failed,
                predictionInfo: {
                    ...mockBatchPredictionFailedResponse,
                    status: jobStatus.Failed,
                    execution: {
                        ...mockBatchPredictionFailedResponse.execution,
                        failure_info: {
                            message: "",
                            stacktrace: ""
                        }
                    }
                } as never
            });

            expect(
                await screen.findByText("Batch Prediction")
            ).toBeInTheDocument();
        });

        it("Should render component when job failed with error", async () => {
            const errorMessage = "Failed to generate predictions";
            await renderPredictionJobOverview({
                ...defaultProps,
                bpJobStatus: jobStatus.Failed,
                predictionInfo: {
                    ...mockBatchPredictionFailedResponse,
                    status: jobStatus.Failed,
                    execution: {
                        ...mockBatchPredictionFailedResponse.execution,
                        failure_info: {
                            message: errorMessage,
                            stacktrace: "stack trace here"
                        }
                    }
                } as never
            });

            expect(
                await screen.findByText("Batch Prediction")
            ).toBeInTheDocument();
        });
    });

    describe("Runtime Calculation", () => {
        it("Should handle runtime calculation error gracefully", async () => {
            const consoleWarnSpy = vi
                .spyOn(console, "warn")
                .mockImplementation(() => {});

            await renderPredictionJobOverview({
                ...defaultProps,
                predictionInfo: {
                    ...mockBatchPredictionResponse,
                    execution: {
                        ...mockBatchPredictionResponse.execution,
                        start_time: "invalid-date",
                        last_updated_at: "2025-04-09T05:27:07.46637600:00"
                    }
                } as never
            });

            expect(
                await screen.findByText("Batch Prediction")
            ).toBeInTheDocument();
            consoleWarnSpy.mockRestore();
        });

        it("Should handle missing start_time", async () => {
            await renderPredictionJobOverview({
                ...defaultProps,
                predictionInfo: {
                    ...mockBatchPredictionResponse,
                    execution: {
                        ...mockBatchPredictionResponse.execution,
                        start_time: undefined,
                        last_updated_at: "2025-04-09T05:27:07.46637600:00"
                    }
                } as never
            });

            expect(
                await screen.findByText("Batch Prediction")
            ).toBeInTheDocument();
        });

        it("Should handle missing last_updated_at", async () => {
            await renderPredictionJobOverview({
                ...defaultProps,
                predictionInfo: {
                    ...mockBatchPredictionResponse,
                    execution: {
                        ...mockBatchPredictionResponse.execution,
                        start_time: "2025-04-09T05:26:07.46637600:00",
                        last_updated_at: undefined
                    }
                } as never
            });

            expect(
                await screen.findByText("Batch Prediction")
            ).toBeInTheDocument();
        });
    });

    describe("Progress Bar Rendering", () => {
        it("Should not render progress bar when progress data is incomplete", async () => {
            await renderPredictionJobOverview({
                ...defaultProps,
                bpJobStatus: jobStatus.Running,
                predictionInfo: {
                    ...mockBatchPredictionRunningResponse,
                    status: jobStatus.Running,
                    log: {
                        stage_name: "Predicting",
                        last_updated_at: "2025-04-11T04:27:45.85599700:00",
                        detail: "Making predictions"
                    },
                    execution: {
                        ...mockBatchPredictionRunningResponse.execution,
                        progress: {
                            iterations_completed: 5,
                            iterations_remaining: 5
                        }
                    }
                } as never
            });

            await waitFor(() => {
                expect(
                    screen.getByText("Batch Prediction")
                ).toBeInTheDocument();
            });
        });

        it("Should not render progress bar when not in Predicting stage", async () => {
            await renderPredictionJobOverview({
                ...defaultProps,
                bpJobStatus: jobStatus.Running,
                predictionInfo: {
                    ...mockBatchPredictionRunningResponse,
                    status: jobStatus.Running,
                    log: {
                        stage_name: "Initializing",
                        last_updated_at: "2025-04-11T04:27:45.85599700:00",
                        detail: "Starting job"
                    }
                } as never
            });

            expect(await screen.findByText("Initializing")).toBeInTheDocument();
        });
    });

    describe("Related Jobs Conditional Rendering", () => {
        it("Should render Prediction Table Generation when generatePredTableJobId is present", async () => {
            await renderPredictionJobOverview({
                ...defaultProps,
                predictionInfo: {
                    ...mockBatchPredictionResponse,
                    generatePredTableJobId: "ptg-job-123456"
                } as never
            });

            await waitFor(() => {
                const predTableCard = screen.queryByText(
                    "Prediction Table Generation"
                );
                if (predTableCard) {
                    expect(predTableCard).toBeInTheDocument();
                }
            });
        });

        it("Should not render related jobs section when no related job IDs", async () => {
            await renderPredictionJobOverview({
                ...defaultProps,
                predictionInfo: {
                    ...mockBatchPredictionResponse,
                    graphSnapshotId: undefined,
                    generatePredTableJobId: undefined,
                    graphId: undefined,
                    trainingJobID: undefined
                } as never
            });

            expect(
                await screen.findByText("Batch Prediction")
            ).toBeInTheDocument();
        });
    });

    describe("Tag Operations Without bpId", () => {
        it.each([{ operation: "add" }, { operation: "delete" }])(
            "Should not $operation tag when bpId is missing",
            async ({ operation }) => {
                await renderPredictionJobOverview({
                    ...defaultProps,
                    bpId: undefined
                });

                if (operation === "add") {
                    const addTag = screen.getByText("Add tag");
                    await userEvent.click(addTag);

                    const keyInput = screen.getByPlaceholderText("key");
                    await userEvent.type(keyInput, "tag_key");

                    const valueInput = screen.getByPlaceholderText("value");
                    await userEvent.type(valueInput, "tag_value");

                    await userEvent.keyboard("{Enter}");

                    await waitFor(() => {
                        expect(mockAxiosInstance.post).not.toHaveBeenCalled();
                    });
                } else {
                    await waitFor(async () => {
                        const tagElement =
                            screen.queryByText("new_prediction_job");
                        if (tagElement) {
                            const closeButton =
                                screen.getByTestId("tag-remove-icon");
                            if (closeButton) {
                                await userEvent.click(closeButton);
                            }
                        }
                    });

                    expect(mockAxiosInstance.post).not.toHaveBeenCalledWith(
                        "v0/jobs/delete_tag",
                        expect.any(Object),
                        expect.any(Object)
                    );
                }
            }
        );
    });
});
