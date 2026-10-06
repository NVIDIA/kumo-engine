// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from "@testing-library/react";
import { GraphSnapshotResponse } from "gen/v0/types";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { describe, expect, it } from "vitest";

import {
    PredictionTableGeneration,
    UnlinkedBatchPredictionJobData
} from "types/batchPredictions";
import { PredictiveQueryInfo, TaskType } from "types/predictiveQueries";
import { TrainTableGeneration } from "types/training_job";

import { jobStatus } from "utils/Constants";

import {
    useGraphSnapshotProperties,
    usePredictionJobSettingProperties,
    usePredictionJobsProperties,
    usePredictionTableGenProperties,
    useTrainingJobProperties,
    useTrainingTableGenProperties
} from "../JobProperties";

const renderWithQueryClient = <T>(hook: () => T) =>
    renderHook(hook, { wrapper: createWrapper(createQueryClient()) });

const getPropertyNames = (properties: { name: string }[]) =>
    properties.map((p) => p.name);

const basePredictionJobInfo = {
    id: "job-abc-12345",
    description: "",
    workflow_id: null,
    predictiveQuery: "",
    predictiveQueryId: "",
    predictiveQuery_link: "",
    status: jobStatus.Running,
    created: "2024-01-01T00:00:00Z",
    dateCreated: "Jan 1, 2024",
    sortCreated: "2024-01-01",
    runtime: "5m",
    num_rows_original: 100,
    taskType: TaskType.BINARY_CLASSIFICATION
} as UnlinkedBatchPredictionJobData;

const baseQueryInfo = {
    id: "pq-abc-12345",
    queryName: "test-query",
    queryYaml: "",
    query_dialect: "PQL",
    trainingConfigYaml: "",
    queryDescription: "",
    advancedOptionsYaml: "",
    modelPlannerYaml: null,
    runMode: null,
    graphName: "test-graph",
    created: "2024-01-01T00:00:00Z",
    updated: "2024-01-01T00:00:00Z",
    dateCreated: "Jan 1, 2024",
    dateUpdated: "Jan 1, 2024",
    trainingJobId: null,
    latestTrainingJobId: null,
    trainingStatus: undefined,
    statusKey: "",
    log: {} as never,
    taskType: TaskType.BINARY_CLASSIFICATION,
    trainingStatusMessage: "",
    callToActions: [],
    timePeriods: false,
    trainedOnData: "",
    evaluatedOnData: "",
    testSplitSize: undefined,
    trainSplitSize: undefined,
    runtime: "",
    status: null,
    isV2: false
} as unknown as PredictiveQueryInfo;

const basePredictionTableGen = {
    jobId: "ptgen-abc-12345",
    jobStatusReport: { status: jobStatus.Running },
    createdAt: "2024-01-01T00:00:00Z",
    created: "2024-01-01T00:00:00Z",
    dateCreated: "Jan 1, 2024",
    config: {},
    graphSnapshotId: null,
    status: jobStatus.Running
} as unknown as PredictionTableGeneration;

const baseTrainTableGen = {
    jobId: "ttgen-abc-12345",
    jobStatusReport: { status: jobStatus.Running },
    createdAt: "2024-01-01T00:00:00Z",
    created: "2024-01-01T00:00:00Z",
    dateCreated: "Jan 1, 2024",
    config: {},
    graphSnapshotId: null,
    status: jobStatus.Running,
    archived: false
} as unknown as TrainTableGeneration;

const baseGraphSnapshot = {
    snapshot_resource: {
        table_ids: {},
        stages: {},
        id: "snap-abc-12345"
    },
    definition: {
        col_groups: []
    }
} as unknown as GraphSnapshotResponse;

describe("usePredictionJobsProperties", () => {
    it("Should return empty array when info is null", () => {
        const { result } = renderWithQueryClient(() =>
            usePredictionJobsProperties(null, jobStatus.Running, "5m", "UI")
        );
        expect(result.current).toEqual([]);
    });

    it("Should include base properties when info is provided", () => {
        const { result } = renderWithQueryClient(() =>
            usePredictionJobsProperties(
                basePredictionJobInfo,
                jobStatus.Running,
                "5m",
                "UI"
            )
        );
        expect(getPropertyNames(result.current)).toEqual(
            expect.arrayContaining(["Status", "Created", "Task type", "Source"])
        );
    });

    it.each([
        ["john", true],
        ["  ", false],
        ["", false],
        [null, false]
    ])("Should include User when user is '%s': %s", (user, shouldInclude) => {
        const { result } = renderWithQueryClient(() =>
            usePredictionJobsProperties(
                { ...basePredictionJobInfo, user: user as string | null },
                jobStatus.Running,
                "5m",
                "UI"
            )
        );
        expect(getPropertyNames(result.current).includes("User")).toBe(
            shouldInclude
        );
    });

    it.each([
        [jobStatus.Done, true],
        [jobStatus.Running, false],
        [jobStatus.Failed, false],
        [jobStatus.Queued, false]
    ])(
        "Should include Total Runtime when status is '%s': %s",
        (status, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionJobsProperties(
                    basePredictionJobInfo,
                    status,
                    "5m",
                    "UI"
                )
            );
            expect(
                getPropertyNames(result.current).includes("Total runtime")
            ).toBe(shouldInclude);
        }
    );
});

describe("usePredictionJobSettingProperties", () => {
    it("Should return empty array when info is null", () => {
        const { result } = renderWithQueryClient(() =>
            usePredictionJobSettingProperties(null)
        );
        expect(result.current).toEqual([]);
    });

    it("Should include base properties when info is provided", () => {
        const { result } = renderWithQueryClient(() =>
            usePredictionJobSettingProperties(basePredictionJobInfo, null)
        );
        expect(getPropertyNames(result.current)).toEqual(
            expect.arrayContaining(["Destination", "Explanations"])
        );
    });

    it.each([
        ["2024-01-01T00:00:00Z", true],
        [null, false],
        [undefined, false],
        ["", false],
        ["-", false]
    ])(
        "Should include Anchor Time when anchorTime is '%s': %s",
        (anchorTime, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionJobSettingProperties(
                    basePredictionJobInfo,
                    anchorTime as string | null
                )
            );
            expect(
                getPropertyNames(result.current).includes("Anchor time")
            ).toBe(shouldInclude);
        }
    );

    it.each([
        [4, true],
        [undefined, false]
    ])(
        "Should include Parallel Workers when numParallelWorkersLimit is %s: %s",
        (numParallelWorkersLimit, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionJobSettingProperties(
                    {
                        ...basePredictionJobInfo,
                        numParallelWorkersLimit
                    },
                    null
                )
            );
            expect(
                getPropertyNames(result.current).includes("Parallel workers")
            ).toBe(shouldInclude);
        }
    );

    it.each([
        [0.5, true],
        [null, false],
        [undefined, false]
    ])(
        "Should include Threshold when threshold is %s: %s",
        (threshold, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionJobSettingProperties(
                    {
                        ...basePredictionJobInfo,
                        threshold: threshold as number
                    },
                    null
                )
            );
            expect(getPropertyNames(result.current).includes("Threshold")).toBe(
                shouldInclude
            );
        }
    );

    it.each([
        ["conn-123", true],
        [undefined, false]
    ])(
        "Should include Connector when output has connector_id '%s': %s",
        (connectorId, shouldInclude) => {
            const output = connectorId
                ? {
                      artifact_type: "PREDICTIONS",
                      storage_type: "S3",
                      connector_id: connectorId
                  }
                : { artifact_type: "PREDICTIONS", storage_type: "S3" };
            const { result } = renderWithQueryClient(() =>
                usePredictionJobSettingProperties(
                    { ...basePredictionJobInfo, outputs: [output as never] },
                    null
                )
            );
            expect(getPropertyNames(result.current).includes("Connector")).toBe(
                shouldInclude
            );
        }
    );
});

describe("useTrainingJobProperties", () => {
    it("Should return empty array when queryInfo is null", () => {
        const { result } = renderWithQueryClient(() =>
            useTrainingJobProperties({
                queryInfo: null,
                jobId: "job-abc-12345",
                status: jobStatus.Running,
                source: "UI"
            })
        );
        expect(result.current).toEqual([]);
    });

    it("Should include base properties when queryInfo is provided", () => {
        const { result } = renderWithQueryClient(() =>
            useTrainingJobProperties({
                queryInfo: baseQueryInfo,
                jobId: null,
                status: jobStatus.Running,
                source: "UI"
            })
        );
        expect(getPropertyNames(result.current)).toEqual(
            expect.arrayContaining(["Status", "Created", "Task type", "Source"])
        );
    });

    it.each([
        ["job-abc-12345", true],
        [null, false],
        ["", false]
    ])(
        "Should include Training job ID when jobId is '%s': %s",
        (jobId, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                useTrainingJobProperties({
                    queryInfo: baseQueryInfo,
                    jobId,
                    status: jobStatus.Running,
                    source: "UI"
                })
            );
            expect(
                getPropertyNames(result.current).includes("Training job ID")
            ).toBe(shouldInclude);
        }
    );

    it("Should set copyText to jobId on Training job ID property", () => {
        const jobId = "job-abc-12345";
        const { result } = renderWithQueryClient(() =>
            useTrainingJobProperties({
                queryInfo: baseQueryInfo,
                jobId,
                status: jobStatus.Running,
                source: "UI"
            })
        );
        const idProp = result.current.find((p) => p.name === "Training job ID");
        expect(idProp?.copyText).toBe(jobId);
    });

    it.each([
        ["john", true],
        ["  ", false],
        ["", false],
        [null, false]
    ])("Should include User when user is '%s': %s", (user, shouldInclude) => {
        const { result } = renderWithQueryClient(() =>
            useTrainingJobProperties({
                queryInfo: {
                    ...baseQueryInfo,
                    user: user as string | null
                },
                jobId: null,
                status: jobStatus.Running,
                source: "UI"
            })
        );
        expect(getPropertyNames(result.current).includes("User")).toBe(
            shouldInclude
        );
    });

    it.each([
        [jobStatus.Done, true],
        [jobStatus.Running, false],
        [jobStatus.Failed, false]
    ])(
        "Should include Total Runtime when trainingStatus is '%s': %s",
        (trainingStatus, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                useTrainingJobProperties({
                    queryInfo: { ...baseQueryInfo, trainingStatus },
                    jobId: null,
                    status: jobStatus.Running,
                    source: "UI"
                })
            );
            expect(
                getPropertyNames(result.current).includes("Total runtime")
            ).toBe(shouldInclude);
        }
    );
});

describe("usePredictionTableGenProperties", () => {
    it("Should return empty array when info is null", () => {
        const { result } = renderWithQueryClient(() =>
            usePredictionTableGenProperties({
                id: null,
                info: null,
                source: "UI",
                status: jobStatus.Running,
                taskType: TaskType.BINARY_CLASSIFICATION,
                user: null,
                numEntities: null
            })
        );
        expect(result.current).toEqual([]);
    });

    it("Should include base properties when info is provided", () => {
        const { result } = renderWithQueryClient(() =>
            usePredictionTableGenProperties({
                id: null,
                info: basePredictionTableGen,
                source: "UI",
                status: jobStatus.Running,
                taskType: TaskType.BINARY_CLASSIFICATION,
                user: null,
                numEntities: null
            })
        );
        expect(getPropertyNames(result.current)).toEqual(
            expect.arrayContaining(["Status", "Created", "Task type", "Source"])
        );
    });

    it.each([
        ["ptgen-abc-12345", true],
        [null, false]
    ])(
        "Should include Prediction Table ID when id is '%s': %s",
        (id, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionTableGenProperties({
                    id: id as string | null,
                    info: basePredictionTableGen,
                    source: "UI",
                    status: jobStatus.Running,
                    taskType: TaskType.BINARY_CLASSIFICATION,
                    user: null,
                    numEntities: null
                })
            );
            expect(
                getPropertyNames(result.current).includes("Prediction table ID")
            ).toBe(shouldInclude);
        }
    );

    it.each([
        ["john", true],
        ["  ", false],
        [null, false]
    ])("Should include User when user is '%s': %s", (user, shouldInclude) => {
        const { result } = renderWithQueryClient(() =>
            usePredictionTableGenProperties({
                id: null,
                info: basePredictionTableGen,
                source: "UI",
                status: jobStatus.Running,
                taskType: TaskType.BINARY_CLASSIFICATION,
                user: user as string | null,
                numEntities: null
            })
        );
        expect(getPropertyNames(result.current).includes("User")).toBe(
            shouldInclude
        );
    });

    it.each([
        [jobStatus.Done, true],
        [jobStatus.Running, false],
        [jobStatus.Failed, false]
    ])(
        "Should include Total Runtime when status is '%s': %s",
        (status, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionTableGenProperties({
                    id: null,
                    info: basePredictionTableGen,
                    source: "UI",
                    status,
                    taskType: TaskType.BINARY_CLASSIFICATION,
                    user: null,
                    numEntities: null
                })
            );
            expect(
                getPropertyNames(result.current).includes("Total runtime")
            ).toBe(shouldInclude);
        }
    );

    it.each([
        ["anchor-time-value", true],
        [null, false]
    ])(
        "Should include Prediction Anchor Time when anchorTime is '%s': %s",
        (anchorTime, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionTableGenProperties({
                    id: null,
                    info: basePredictionTableGen,
                    source: "UI",
                    status: jobStatus.Running,
                    taskType: TaskType.BINARY_CLASSIFICATION,
                    user: null,
                    anchorTime: anchorTime as string | null,
                    numEntities: null
                })
            );
            expect(
                getPropertyNames(result.current).includes(
                    "Prediction anchor time"
                )
            ).toBe(shouldInclude);
        }
    );

    it.each([
        ["1000", true],
        ["0", true],
        [null, false],
        [undefined, false]
    ])(
        "Should include Total Entities when numEntities is '%s': %s",
        (numEntities, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionTableGenProperties({
                    id: null,
                    info: basePredictionTableGen,
                    source: "UI",
                    status: jobStatus.Running,
                    taskType: TaskType.BINARY_CLASSIFICATION,
                    user: null,
                    numEntities: numEntities as string | null
                })
            );
            expect(
                getPropertyNames(result.current).includes("Total entities")
            ).toBe(shouldInclude);
        }
    );
});

describe("useTrainingTableGenProperties", () => {
    it("Should return empty array when info is null", () => {
        const { result } = renderWithQueryClient(() =>
            useTrainingTableGenProperties({
                info: null,
                jobId: null,
                status: jobStatus.Running,
                user: null,
                taskType: TaskType.BINARY_CLASSIFICATION,
                source: "UI"
            })
        );
        expect(result.current).toEqual([]);
    });

    it("Should include base properties when info is provided", () => {
        const { result } = renderWithQueryClient(() =>
            useTrainingTableGenProperties({
                info: baseTrainTableGen,
                jobId: null,
                status: jobStatus.Running,
                user: null,
                taskType: TaskType.BINARY_CLASSIFICATION,
                source: "UI"
            })
        );
        expect(getPropertyNames(result.current)).toEqual(
            expect.arrayContaining(["Status", "Created", "Task type", "Source"])
        );
    });

    it.each([
        ["ttgen-abc-12345", true],
        [null, false]
    ])(
        "Should include Training Table ID when jobId is '%s': %s",
        (jobId, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                useTrainingTableGenProperties({
                    info: baseTrainTableGen,
                    jobId,
                    status: jobStatus.Running,
                    user: null,
                    taskType: TaskType.BINARY_CLASSIFICATION,
                    source: "UI"
                })
            );
            expect(
                getPropertyNames(result.current).includes("Training table ID")
            ).toBe(shouldInclude);
        }
    );

    it.each([
        ["john", true],
        ["  ", false],
        [null, false]
    ])("Should include User when user is '%s': %s", (user, shouldInclude) => {
        const { result } = renderWithQueryClient(() =>
            useTrainingTableGenProperties({
                info: baseTrainTableGen,
                jobId: null,
                status: jobStatus.Running,
                user: user as string | null,
                taskType: TaskType.BINARY_CLASSIFICATION,
                source: "UI"
            })
        );
        expect(getPropertyNames(result.current).includes("User")).toBe(
            shouldInclude
        );
    });

    it.each([
        [jobStatus.Done, true],
        [jobStatus.Running, false],
        [jobStatus.Failed, false]
    ])(
        "Should include Total Runtime when status is '%s': %s",
        (status, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                useTrainingTableGenProperties({
                    info: baseTrainTableGen,
                    jobId: null,
                    status,
                    user: null,
                    taskType: TaskType.BINARY_CLASSIFICATION,
                    source: "UI"
                })
            );
            expect(
                getPropertyNames(result.current).includes("Total runtime")
            ).toBe(shouldInclude);
        }
    );
});

describe("useGraphSnapshotProperties", () => {
    it("Should return empty array when graphSnapshot is null", () => {
        const { result } = renderWithQueryClient(() =>
            useGraphSnapshotProperties({
                graphSnapshot: null as unknown as GraphSnapshotResponse,
                id: "snap-abc-12345",
                status: jobStatus.Running,
                totalRows: null,
                totalSizeBytes: null
            })
        );
        expect(result.current).toEqual([]);
    });

    it("Should include base properties when graphSnapshot is provided", () => {
        const { result } = renderWithQueryClient(() =>
            useGraphSnapshotProperties({
                graphSnapshot: baseGraphSnapshot,
                id: "snap-abc-12345",
                status: jobStatus.Running,
                totalRows: null,
                totalSizeBytes: null
            })
        );
        expect(getPropertyNames(result.current)).toEqual(
            expect.arrayContaining(["Status", "Source"])
        );
    });

    it.each([
        ["snap-abc-12345", true],
        ["", false]
    ])(
        "Should include Snapshot ID when id is '%s': %s",
        (id, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                useGraphSnapshotProperties({
                    graphSnapshot: baseGraphSnapshot,
                    id,
                    status: jobStatus.Running,
                    totalRows: null,
                    totalSizeBytes: null
                })
            );
            expect(
                getPropertyNames(result.current).includes("Snapshot ID")
            ).toBe(shouldInclude);
        }
    );

    it.each([
        ["2024-01-01T00:00:00Z", true],
        [null, false]
    ])(
        "Should include Snapshotted at when startTime is '%s': %s",
        (startTime, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                useGraphSnapshotProperties({
                    graphSnapshot: baseGraphSnapshot,
                    id: "snap-abc-12345",
                    status: jobStatus.Running,
                    startTime: startTime as string,
                    totalRows: null,
                    totalSizeBytes: null
                })
            );
            expect(
                getPropertyNames(result.current).includes("Snapshotted")
            ).toBe(shouldInclude);
        }
    );

    it.each([
        [false, true],
        [true, false]
    ])(
        "Should include Configs and Graph Size when isLoading is %s: %s",
        (isLoading, shouldInclude) => {
            const { result } = renderWithQueryClient(() =>
                useGraphSnapshotProperties({
                    graphSnapshot: baseGraphSnapshot,
                    id: "snap-abc-12345",
                    status: jobStatus.Running,
                    isLoading,
                    totalRows: 1000,
                    totalSizeBytes: 2048
                })
            );
            const names = getPropertyNames(result.current);
            expect(names.includes("Configs")).toBe(shouldInclude);
            expect(names.includes("Graph size")).toBe(shouldInclude);
        }
    );
});
