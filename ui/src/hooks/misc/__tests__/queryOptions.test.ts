// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from "@testing-library/react";
import { AxiosResponse } from "axios";
import {
    GenerateTrainTableResponse,
    GraphSnapshotResponse,
    ListJobsResponse,
    TrainTableGenerationProgress
} from "gen/v0/types";
import { BatchPredictionJobResponse } from "gen/v1/types";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { JobSource } from "types/jobs";

import { jobStatus } from "utils/Constants";
import { ReactQueryKeys } from "utils/reactQueryKeys";

import { useQueryOptions } from "../queryOptions";

beforeEach(() => {
    vi.clearAllMocks();
});

describe("useQueryOptions", () => {
    describe("getQueryOptions", () => {
        it("Should return empty object when queryKey is empty", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: []
            });

            expect(options).toEqual({});
        });

        it("Should use default values by default", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: ["test-key"]
            });

            expect(options.query.staleTime).toBe(300000);
            expect(options.query.enabled).toBe(true);
        });

        it("Should use custom staleTime when provided", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: ["test-key"],
                staleTime: 60000
            });

            expect(options.query.staleTime).toBe(60000);
        });

        it("Should use custom enabled value when provided", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: ["test-key"],
                enabled: false
            });

            expect(options.query.enabled).toBe(false);
        });

        it("Should configure ColStatsByTableId with refetchOnWindowFocus false and retry false", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: [ReactQueryKeys.ColStatsByTableId, "table-123"]
            });

            expect(options.query.refetchOnWindowFocus).toBe(false);
            expect(options.query.retry).toBe(false);
        });

        it("Should configure SnapshotsColStatsById with long staleTime", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: [ReactQueryKeys.SnapshotsColStatsById, "snapshot-123"]
            });

            expect(options.query.staleTime).toBe(1800000);
            expect(options.query.retry).toBe(false);
        });

        it("Should configure download URLs with staleTime 0", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: [
                    ReactQueryKeys.TrainingJobsDownloadHoldoutById,
                    "job-123"
                ]
            });

            expect(options.query.staleTime).toBe(0);
            expect(options.query.refetchInterval).toBe(false);
            expect(options.query.retry).toBe(false);
        });

        it("Should configure JobList with short staleTime and transform null sources", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: [ReactQueryKeys.JobList]
            });

            expect(options.query.staleTime).toBe(120000);

            const mockResponse: AxiosResponse<ListJobsResponse> = {
                data: {
                    jobs: [
                        { id: "1", source: null } as unknown,
                        { id: "2", source: "UI" } as unknown
                    ]
                }
            } as AxiosResponse<ListJobsResponse>;

            const transformed = options.query.select(mockResponse);

            expect(transformed.data.jobs[0].source).toBe(JobSource.NONE);
            expect(transformed.data.jobs[1].source).toBe("UI");
        });

        it("Should provide custom queryFn when specified", () => {
            const { result } = renderHook(() => useQueryOptions());
            const customQueryFn = vi.fn();

            const options = result.current.getQueryOptions({
                queryKey: ["test-key"],
                queryFn: customQueryFn
            });

            expect(options.query.queryFn).toBe(customQueryFn);
        });

        it("Should apply custom refetchOnWindowFocus when provided", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: ["test-key"],
                refetchOnWindowFocus: true
            });

            expect(options.query.refetchOnWindowFocus).toBe(true);
        });

        it("Should apply custom retry when provided", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: ["test-key"],
                retry: 3
            });

            expect(options.query.retry).toBe(3);
        });

        it("Should apply custom refetchInterval when provided", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: ["test-key"],
                refetchInterval: 10000
            });

            expect(options.query.refetchInterval).toBe(10000);
        });
    });

    describe("getQueryOptionsById", () => {
        it("Should create query options with id in queryKey", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptionsById(
                "TestQuery",
                "test-123"
            );

            expect(options.query.queryKey).toEqual(["TestQuery", "test-123"]);
            expect(options.query.enabled).toBe(true);
        });

        it("Should disable query when id is null", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptionsById(
                "TestQuery",
                null
            );

            expect(options.query.enabled).toBe(false);
        });

        it("Should disable query when id is undefined", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptionsById(
                "TestQuery",
                undefined
            );

            expect(options.query.enabled).toBe(false);
        });

        it("Should use custom enabled value when provided", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptionsById(
                "TestQuery",
                "test-123",
                false
            );

            expect(options.query.enabled).toBe(false);
        });

        it("Should apply custom refetchInterval", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptionsById(
                "TestQuery",
                "test-123",
                true,
                5000
            );

            expect(options.query.refetchInterval).toBe(5000);
        });
    });

    describe("getQueryOptionsByEntityId", () => {
        it("Should create query options with id and entityId in queryKey", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptionsByEntityId(
                "TestQuery",
                "job-123",
                "entity-456"
            );

            expect(options.query.queryKey).toEqual([
                "TestQuery",
                "job-123",
                "entity-456"
            ]);
            expect(options.query.enabled).toBe(true);
        });

        it("Should use empty string for null id", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptionsByEntityId(
                "TestQuery",
                null,
                "entity-456"
            );

            expect(options.query.queryKey).toEqual([
                "TestQuery",
                "",
                "entity-456"
            ]);
        });

        it("Should disable query when id is null and enabled not explicitly set", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptionsByEntityId(
                "TestQuery",
                null,
                "entity-456"
            );

            expect(options.query.enabled).toBe(false);
        });
    });

    describe("getQueryOptionsByIdAndJobStatus", () => {
        it("Should use short cache time for running jobs", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptionsByIdAndJobStatus(
                "TestQuery",
                "job-123",
                true,
                jobStatus.Running
            );

            expect(options.query.staleTime).toBe(120000);
        });

        it("Should use long cache time for done jobs", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptionsByIdAndJobStatus(
                "TestQuery",
                "job-123",
                true,
                jobStatus.Done
            );

            expect(options.query.staleTime).toBe(1800000);
        });
    });

    describe("getModelPlansQueryOptionsById", () => {
        it("Should include bestExperimentIndex in queryKey", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getModelPlansQueryOptionsById(
                "TestQuery",
                "job-123",
                jobStatus.Done,
                5
            );

            expect(options.query.queryKey).toEqual([
                "TestQuery",
                "job-123",
                jobStatus.Done,
                "5"
            ]);
        });

        it("Should use appropriate cache time based on status", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getModelPlansQueryOptionsById(
                "TestQuery",
                "job-123",
                jobStatus.Running,
                0
            );

            expect(options.query.staleTime).toBe(120000);
        });
    });

    describe("queryOption.connectorTableCols", () => {
        it("Should create query options for connector table columns", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.queryOption.connectorTableCols(
                "test_table",
                "connector-123",
                true
            );

            expect(options.query.queryKey).toEqual([
                ReactQueryKeys.ConnectorTableCols,
                "test_table",
                "connector-123"
            ]);
            expect(options.query.staleTime).toBe(300000);
            expect(options.query.retry).toBe(false);
            expect(options.query.enabled).toBe(true);
        });

        it("Should not enable query when enabled is false", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.queryOption.connectorTableCols(
                "test_table",
                "connector-123",
                false
            );

            expect(options.query.enabled).toBe(false);
        });
    });

    describe("SnapshotsGraph refetchInterval", () => {
        it("Should poll when status is Running", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: [ReactQueryKeys.SnapshotsGraph, "graph-123"]
            });

            const mockQuery = {
                state: {
                    data: {
                        data: {
                            snapshot_resource: {
                                stages: {
                                    INGEST: { status: jobStatus.Running }
                                }
                            }
                        }
                    } as unknown as GraphSnapshotResponse
                } as AxiosResponse<GraphSnapshotResponse>
            };

            const interval = options.query.refetchInterval(mockQuery);
            expect(interval).toBe(120000);
        });

        it("Should not poll when status is Done", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.getQueryOptions({
                queryKey: [ReactQueryKeys.SnapshotsGraph, "graph-123"]
            });

            const mockQuery = {
                state: {
                    data: {
                        data: {
                            snapshot_resource: {
                                stages: {
                                    INGEST: { status: jobStatus.Done }
                                }
                            }
                        }
                    } as unknown as GraphSnapshotResponse
                } as AxiosResponse<GraphSnapshotResponse>
            };

            const interval = options.query.refetchInterval(mockQuery);
            expect(interval).toBe(false);
        });
    });

    describe("BatchPredictionJobsById refetchInterval", () => {
        it("Should poll when status is Running", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.queryOption.batchPredictionJobsById(
                "job-123",
                undefined
            );

            const mockQuery = {
                state: {
                    data: {
                        data: {
                            execution: { status: jobStatus.Running }
                        }
                    } as unknown as BatchPredictionJobResponse
                } as AxiosResponse<BatchPredictionJobResponse>
            };

            const interval = options.query.refetchInterval(mockQuery);
            expect(interval).toBe(10000);
        });

        it("Should not poll when status is Done", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.queryOption.batchPredictionJobsById(
                "job-123",
                undefined
            );

            const mockQuery = {
                state: {
                    data: {
                        data: {
                            execution: { status: jobStatus.Done }
                        }
                    } as unknown as BatchPredictionJobResponse
                } as AxiosResponse<BatchPredictionJobResponse>
            };

            const interval = options.query.refetchInterval(mockQuery);
            expect(interval).toBe(false);
        });

        it("Should refetch immediately when refetchTrigger changes", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options = result.current.queryOption.batchPredictionJobsById(
                "job-123",
                1
            );

            const mockQuery = {
                state: {
                    data: {
                        data: {
                            execution: { status: jobStatus.Done }
                        }
                    } as unknown as BatchPredictionJobResponse
                } as AxiosResponse<BatchPredictionJobResponse>
            };

            const interval = options.query.refetchInterval(mockQuery);
            expect(interval).toBe(0);
        });
    });

    describe("TraintableJobsById refetchInterval", () => {
        it("Should poll with 15s when status is Running", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options =
                result.current.queryOption.traintableJobsById("job-123");

            const mockQuery = {
                state: {
                    data: {
                        data: {
                            resource: {
                                job_status_report: { status: jobStatus.Running }
                            }
                        }
                    } as unknown as GenerateTrainTableResponse
                } as AxiosResponse<GenerateTrainTableResponse>
            };

            const interval = options.query.refetchInterval(mockQuery);
            expect(interval).toBe(15000);
        });

        it("Should not poll when status is Done", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options =
                result.current.queryOption.traintableJobsById("job-123");

            const mockQuery = {
                state: {
                    data: {
                        data: {
                            resource: {
                                job_status_report: { status: jobStatus.Done }
                            }
                        }
                    } as unknown as GenerateTrainTableResponse
                } as AxiosResponse<GenerateTrainTableResponse>
            };

            const interval = options.query.refetchInterval(mockQuery);
            expect(interval).toBe(false);
        });

        it("Should poll faster when status is Queued for first 10 polls", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options =
                result.current.queryOption.traintableJobsById("job-123");

            const mockQuery = {
                state: {
                    data: {
                        data: {
                            resource: {
                                job_status_report: { status: jobStatus.Queued }
                            }
                        }
                    } as unknown as GenerateTrainTableResponse
                } as AxiosResponse<GenerateTrainTableResponse>
            };

            const interval = options.query.refetchInterval(mockQuery);
            expect(interval).toBe(5000);
        });
    });

    describe("TraintableJobsGenerationStatsById refetchInterval", () => {
        it("Should poll when iterations are not finished", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options =
                result.current.queryOption.traintableJobsGenerationStatsById(
                    "job-123",
                    jobStatus.Running
                );

            const mockQuery = {
                state: {
                    data: {
                        num_expected_iterations: 10,
                        num_finished_iterations: 5
                    } as TrainTableGenerationProgress
                }
            };

            const interval = options.query.refetchInterval(mockQuery);
            expect(interval).toBe(10000);
        });

        it("Should not poll when iterations are finished", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options =
                result.current.queryOption.traintableJobsGenerationStatsById(
                    "job-123",
                    jobStatus.Running
                );

            const mockQuery = {
                state: {
                    data: {
                        num_expected_iterations: 10,
                        num_finished_iterations: 10
                    } as TrainTableGenerationProgress
                }
            };

            const interval = options.query.refetchInterval(mockQuery);
            expect(interval).toBe(false);
        });

        it("Should not poll when status is not Running", () => {
            const { result } = renderHook(() => useQueryOptions());

            const options =
                result.current.queryOption.traintableJobsGenerationStatsById(
                    "job-123",
                    jobStatus.Done
                );

            const interval = options.query.refetchInterval({});
            expect(interval).toBe(false);
        });
    });
});
